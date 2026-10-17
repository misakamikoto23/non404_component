"""Recover only unresolved detector identities; existing named records are immutable.

The caller supplies predictions and machine readings, never evaluation labels.
All evidence is advisory. No strings, units, case, geometry or existing keys are
normalized. This module does not call external services.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re

from .contest_components import PREFIX_PRIOR, TYPE_MAP
from .text_roles import REFERENCE, NUMERIC_UNIT_REFERENCE

UNRESOLVED = re.compile(r'^__unresolved_detection_\d+_*$')
FAMILIES = {
    'resistor', 'capacitor', 'inductor', 'ic_or_module', 'diode', 'led',
    'mosfet', 'bjt', 'connector', 'switch', 'crystal', 'fuse', 'battery',
    'transformer', 'antenna', 'optocoupler', 'motor', 'testpoint', 'other',
    'not_component',
}
FIELDS = {'reference', 'Name', 'value'}


def unresolved_key(key):
    return isinstance(key, str) and UNRESOLVED.fullmatch(key) is not None


def named_fingerprint(predictions):
    named = {case: {key: value for key, value in pred['components'].items()
                    if not unresolved_key(key)} for case, pred in predictions.items()}
    return hashlib.sha256(json.dumps(named, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def record_fingerprint(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def parse_recovery(content, targets):
    data = json.loads(content)
    expected = {target['id'] for target in targets}
    if len(expected) != len(targets):
        raise ValueError('Duplicate input target')
    if not isinstance(data, dict) or set(data) != {'items'} or not isinstance(data['items'], list):
        raise ValueError('Expected items array')
    seen = set()
    for row in data['items']:
        if not isinstance(row, dict) or set(row) != FIELDS | {'id', 'family', 'target_complete', 'uncertain_fields'}:
            raise ValueError('Invalid reading fields')
        if not isinstance(row['id'], str) or row['id'] not in expected or row['id'] in seen:
            raise ValueError('Unknown or repeated target')
        for field in FIELDS:
            value = row[field]
            if value is not None and (not isinstance(value, str) or len(value) > 120):
                raise ValueError('Invalid literal')
        if not isinstance(row['family'], str) or row['family'] not in FAMILIES or not isinstance(row['target_complete'], bool):
            raise ValueError('Invalid visual family or completeness')
        uncertain = row['uncertain_fields']
        if not isinstance(uncertain, list) or any(not isinstance(v, str) or v not in FIELDS for v in uncertain):
            raise ValueError('Invalid uncertainty')
        if len(uncertain) != len(set(uncertain)):
            raise ValueError('Repeated uncertainty field')
        seen.add(row['id'])
    if seen != expected:
        raise ValueError('Missing target')
    return data['items']


def _prefix(reference):
    match = REFERENCE.fullmatch(reference or '') or NUMERIC_UNIT_REFERENCE.fullmatch(reference or '')
    return match.group(1).upper() if match else None


def apply_recovery(baseline, targets, readings, *, include_fields=True):
    """Rename eligible unresolved records, preserving count and named records.

    Conflicting new references are rejected as a group, independent of order.
    Legacy cached readings may omit family/target_complete; this is recorded by
    the caller. New API responses must pass parse_recovery's stricter schema.
    """
    by_id = {target['id']: target for target in targets}
    if len(by_id) != len(targets):
        raise ValueError('Duplicate input target')
    seen, sources = set(), set()
    for target in targets:
        source = (target['case'], target['baseline_key'])
        if source in sources:
            raise ValueError('Duplicate source detection')
        sources.add(source)
        current = baseline.get(source[0], {}).get('components', {}).get(source[1])
        if target.get('expected_record_sha256') and (current is None or
                record_fingerprint(current) != target['expected_record_sha256']):
            raise ValueError('Stale baseline record')
    for row in readings:
        if row['id'] not in by_id or row['id'] in seen:
            raise ValueError('Unknown or repeated reading')
        seen.add(row['id'])
    out = deepcopy(baseline)
    before = named_fingerprint(baseline)
    proposed = Counter((by_id[row['id']]['case'], row.get('reference')) for row in readings
                       if row.get('reference') and 'reference' not in row.get('uncertain_fields', [])
                       and row.get('target_complete', True) and row.get('family') != 'not_component')
    changed, blocked = [], []
    for row in readings:
        target = by_id[row['id']]
        case, old = target['case'], target['baseline_key']
        reference = row.get('reference')
        reason = None
        if case not in out or old not in out[case]['components']:
            reason = 'missing_source'
        elif not unresolved_key(old):
            reason = 'named_source_locked'
        elif not row.get('target_complete', True) or row.get('family') == 'not_component':
            reason = 'not_complete_component'
        elif not reference or 'reference' in row.get('uncertain_fields', []):
            reason = 'reference_unresolved'
        elif not _prefix(reference):
            reason = 'unsupported_reference_syntax'
        elif reference in baseline[case]['components'] or proposed[(case, reference)] != 1:
            reason = 'existing_or_competing_key'
        if reason:
            blocked.append({'id': row['id'], 'reason': reason})
            continue
        component = out[case]['components'].pop(old)
        if component['type'] == 'component':
            family = PREFIX_PRIOR.get(_prefix(reference))
            if family is None and row.get('family') not in {None, 'other', 'not_component'}:
                family = row['family']
            if family:
                component['type'] = TYPE_MAP.get(family, family)
        if include_fields:
            for field in ['Name', 'value']:
                if field not in row.get('uncertain_fields', []):
                    component[field] = row.get(field)
        out[case]['components'][reference] = component
        changed.append({'id': row['id'], 'case': case, 'source_id': target.get('source_id'),
                        'before_key': old, 'after_key': reference,
                        'before': deepcopy(baseline[case]['components'][old]), 'after': deepcopy(component),
                        'source': 'machine_visual_reading_not_verified_label'})
    for case, original in baseline.items():
        for key, component in original['components'].items():
            if not unresolved_key(key) and out[case]['components'].get(key) != component:
                raise AssertionError('Named component changed')
        if len(out[case]['components']) != len(original['components']):
            raise AssertionError('Detection count changed')
        for section in set(original) - {'components'}:
            if out[case][section] != original[section]:
                raise AssertionError('Non-component section changed')
    # A recovery adds named records, so hash only the ORIGINAL named subset.
    original_named_view = {case: {'components': {key: out[case]['components'][key]
                            for key in pred['components'] if not unresolved_key(key)}}
                           for case, pred in baseline.items()}
    assert named_fingerprint(original_named_view) == before
    return out, {'changes': changed, 'blocked': blocked, 'named_before_sha256': before,
                 'original_named_after_sha256': named_fingerprint(original_named_view),
                 'original_named_unchanged': True, 'detection_count_unchanged': True,
                 'include_fields': include_fields}
