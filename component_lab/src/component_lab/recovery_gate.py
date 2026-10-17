"""Prediction-only admission and deterministic sampling for identity recovery."""
from collections import Counter
import hashlib

from .identity_recovery import unresolved_key
from .text_matching import _distance, _height, _role, _type
from .text_roles import REFERENCE, NUMERIC_UNIT_REFERENCE, PREFIX_TYPES

PHYSICAL_TYPES = {'resistor', 'capacitor', 'inductor', 'connector', 'diode', 'led',
                  'switch', 'ic', 'bjt', 'mosfet', 'buzzer', 'opamp'}


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def recovery_candidates(case, baseline, association, traces, width, height):
    """Return candidates and rejection counts without image labels or GT boxes."""
    trace_by_source = {row['source_id']: row for row in traces}
    owners = {a['text_id']: a['component_id'] for a in association['assignments']
              if a['role'] == 'reference'}
    candidates, rejected = [], Counter()
    for component in association['components']:
        source = component['id']
        trace = trace_by_source[source]
        if not unresolved_key(trace['key']):
            continue
        x1, y1, x2, y2 = component['bbox']
        kind = _type(component)
        reason = None
        if kind not in PHYSICAL_TYPES:
            reason = 'not_in_physical_family_pool'
        elif component.get('detection', {}).get('score', 0) < .5:
            reason = 'low_detection_score'
        elif component.get('classification', {}).get('score', 0) < .8:
            reason = 'classifier_abstention'
        elif min(x2-x1, y2-y1) < 4:
            reason = 'too_small'
        elif (x2-x1)/width > .6 and (y2-y1)/height > .6:
            reason = 'page_sized_frame'
        if reason:
            rejected[reason] += 1
            continue
        refs, fields = [], []
        for text in association['texts']:
            gap = _distance(text['bbox'], component['bbox'])/_height(text)
            raw = text['raw']
            match = REFERENCE.fullmatch(raw) or NUMERIC_UNIT_REFERENCE.fullmatch(raw)
            ref = _role(text, 'reference')
            if (gap <= 6 and text.get('score', 0) >= .8 and match and ref
                    and ref.get('reason') != 'unverified_ocr_alternative'
                    and kind in PREFIX_TYPES.get(match.group(1).upper(), set())
                    and raw not in baseline['components'] and owners.get(text['id']) in {None, source}):
                refs.append({'text_id': text['id'], 'raw': raw, 'distance_heights': gap})
            if gap <= 4:
                value = _role(text, 'value')
                model = _role(text, 'part_number')
                expected = {'resistor': 'resistance', 'capacitor': 'capacitance', 'inductor': 'inductance'}.get(kind)
                if model or value and expected and value.get('value_kind') == expected:
                    fields.append({'text_id': text['id'], 'raw': raw, 'distance_heights': gap})
        if len({r['raw'] for r in refs}) > 8:
            rejected['dense_pin_number_row'] += 1
            continue
        if not refs and not fields:
            rejected['no_text_group_evidence'] += 1
            continue
        candidates.append({'case': case, 'source_id': source, 'baseline_key': trace['key'],
                           'bbox': list(component['bbox']), 'type_hint': baseline['components'][trace['key']]['type'],
                           'family_hint': kind, 'detection_score': component['detection']['score'],
                           'classification_score': component['classification']['score'],
                           'tier': 'reference_candidate' if refs else 'field_context_only',
                           'reference_candidates': sorted(refs, key=lambda r:(r['distance_heights'],r['text_id'])),
                           'field_candidates': sorted(fields, key=lambda r:(r['distance_heights'],r['text_id'])),
                           'selection_hash': _digest(f'{case}/{source}/unresolved_v1')})
    return candidates, dict(rejected)


def sample_candidates(candidates, *, per_tier=12, per_case=3, exclude_sources=()):
    excluded = set(exclude_sources)
    selected, counts = [], Counter()
    for tier in ['reference_candidate', 'field_context_only']:
        buckets = {}
        for row in candidates:
            if row['tier'] == tier and (row['case'], row['source_id']) not in excluded:
                buckets.setdefault(row['case'], []).append(row)
        for rows in buckets.values():
            rows.sort(key=lambda r:r['selection_hash'])
        cases = sorted(buckets, key=lambda case:_digest(case+'/unresolved_v1'))
        picked = 0
        while picked < per_tier:
            progress = False
            for case in cases:
                if buckets[case] and counts[case] < per_case:
                    selected.append(buckets[case].pop(0))
                    counts[case] += 1; picked += 1; progress = True
                    if picked == per_tier:
                        break
            if not progress:
                break
    return selected
