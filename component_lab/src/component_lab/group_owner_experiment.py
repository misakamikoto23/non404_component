"""Prediction-only, opt-in reverse text-group evidence and shadow text guard."""
from collections import Counter
from copy import deepcopy
import math

from .text_matching import (_role, _type, _height, _distance, _center,
                            PRIMARY_VALUE_KIND, PREFIX_TYPES)


class ReverseGroupOwner:
    """Reward a unique parameter-nearest owner only on originally ambiguous rows.

    Uses existing groups, explicit independent fields, normalized 1.0h margin,
    -0.20 reward, unchanged reference search. This is not a sandwich generator.
    """
    def __init__(self):
        self.counts = Counter()
        self.events = []

    def __call__(self, reference, components, texts, group, real, ambiguity_gap):
        self.counts['reference_rows_seen'] += 1
        if len(real) < 2 or real[1][0] - real[0][0] >= ambiguity_gap:
            return {}
        self.counts['originally_ambiguous_rows'] += 1
        if not group or not group.get('edges'):
            return {}
        self.counts['ambiguous_rows_with_group'] += 1
        by_id = {t['id']: t for t in texts}
        ref_role = _role(reference, 'reference')
        prefix = ref_role.get('prefix') if ref_role else None
        expected_kind = {'R': 'resistance', 'C': 'capacitance', 'L': 'inductance'}.get(prefix)
        expected_types = PREFIX_TYPES.get(prefix, set())
        for edge in sorted(group['edges'], key=lambda e: e['normalized_geometry_cost']):
            parameter = by_id[edge['text_id']]
            role = _role(parameter, edge['role'], allow_context=False)
            eligible = bool(role and ((edge['role'] == 'value' and expected_kind and role.get('value_kind') == expected_kind)
                                      or (edge['role'] == 'part_number' and prefix in {'U', 'IC'})))
            if not eligible:
                continue
            self.counts['independent_group_parameters_seen'] += 1
            h = max(_height(reference), _height(parameter))
            candidates = []
            for _, j in real:
                kind = _type(components[j])
                if kind not in expected_types and kind != 'component':
                    continue
                distance = _distance(parameter['bbox'], components[j]['bbox']) / h
                if distance <= 8:
                    candidates.append((distance, j))
            candidates.sort()
            if len(candidates) < 2 or candidates[1][0] - candidates[0][0] < 1.0:
                continue
            j = candidates[0][1]
            result = {'delta': -.20, 'parameter_text_id': parameter['id'],
                      'parameter_role': edge['role'], 'nearest_margin_heights': candidates[1][0]-candidates[0][0],
                      'reason': 'independent_parameter_nearest_on_ambiguous_reference', 'uses_ground_truth': False}
            self.counts['adjusted_reference_rows'] += 1
            self.events.append({'reference_text_id': reference['id'], 'component_id': components[j]['id'], **result})
            return {j: result}
        return {}


def sandwich(box, reference, parameter):
    """Strict geometric observation only, not proof of text ownership."""
    a, b, c = _center(reference['bbox']), _center(parameter['bbox']), _center(box)
    vx, vy = b[0]-a[0], b[1]-a[1]
    length2 = vx*vx+vy*vy
    if length2 == 0:
        return False
    projection = ((c[0]-a[0])*vx+(c[1]-a[1])*vy)/length2
    cross = abs((c[0]-a[0])*vy-(c[1]-a[1])*vx)/math.sqrt(length2)
    h = max(_height(reference), _height(parameter))
    opposite_x = (a[0] < box[0] and b[0] > box[2]) or (b[0] < box[0] and a[0] > box[2])
    opposite_y = (a[1] < box[1] and b[1] > box[3]) or (b[1] < box[1] and a[1] > box[3])
    return .15 <= projection <= .85 and cross <= .5*h and (opposite_x or opposite_y)


def clear_by_three_locks(association, prediction, traces):
    """Explicitly destructive shadow ablation. Never enabled in production.

    Existing fields failing the proposed geometry guard are cleared in a COPY.
    This does not infer that the image or GT has no text. No global reassignment.
    """
    out = deepcopy(prediction)
    texts = {t['id']: t for t in association['texts']}
    components = {c['id']: c for c in association['components']}
    groups = {g['reference_text_id']: g for g in association['groups']}
    ref_tokens = {a['text_id'] for a in association['assignments'] if a['role'] == 'reference'}
    decisions = []
    for trace in traces:
        record = out['components'][trace['key']]
        field = 'value' if record['type'] in {'r', 'c'} else 'Name' if record['type'] == 'box' else None
        if field is None or record[field] is None:
            continue
        component = components[trace['source_id']]
        evidence = component.get('association_evidence', {})
        ref = texts.get(evidence.get('reference', {}).get('text_id'))
        source = ('value' if field == 'value' or trace['name_source'] == 'value_fallback' else trace['name_source'])
        parameter = texts.get(evidence.get(source, {}).get('text_id'))
        reasons = []
        if ref is None or parameter is None:
            reasons.append('missing_reference_or_parameter_evidence')
        else:
            h = max(_height(ref), _height(parameter))
            group = groups.get(ref['id'])
            paired = bool(group and parameter['id'] in group['text_ids'] and
                          _distance(ref['bbox'], parameter['bbox']) / h <= 1.5)
            if not paired and not sandwich(component['bbox'], ref, parameter):
                reasons.append('no_compact_pair_or_sandwich')
            family = _type(component)
            compatible = [c for c in association['components'] if _type(c) in {family, 'component'}]
            distances = sorted((_distance(parameter['bbox'], c['bbox']), c['id']) for c in compatible)
            if (not distances or distances[0][1] != component['id'] or
                    len(distances) > 1 and distances[1][0]-distances[0][0] < 8):
                reasons.append('no_unique_parameter_nearest_margin_8px')
            if parameter['id'] in ref_tokens:
                reasons.append('parameter_is_reference_token')
            # Only flag a numeric candidate as a dense pin sequence when at
            # least three nearby number boxes align at this body's boundary.
            if parameter['raw'].isdigit():
                pc = _center(parameter['bbox']); body = component['bbox']
                near = [t for t in texts.values() if t['raw'].isdigit() and
                        _distance(t['bbox'], body) <= 2*h and math.dist(_center(t['bbox']), pc) <= 6*h]
                aligned = [t for t in near if min(abs(_center(t['bbox'])[0]-pc[0]), abs(_center(t['bbox'])[1]-pc[1])) <= .5*h]
                if len(aligned) >= 3:
                    reasons.append('dense_numeric_pin_sequence')
        before = record[field]
        if reasons:
            record[field] = None
        decisions.append({'key': trace['key'], 'source_id': trace['source_id'], 'field': field,
                          'before': before, 'after': record[field], 'reasons': reasons,
                          'cleared': bool(reasons), 'uses_ground_truth': False})
    return out, decisions
