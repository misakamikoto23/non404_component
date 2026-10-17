"""Offline, hash-bound writeback of machine readings into prediction bundles."""
import json
from pathlib import Path
import re

from .candidate_verifier import atomic_json
from .identity_recovery import apply_recovery, parse_recovery, record_fingerprint


def recover_files(baseline_path, manifest_path, readings_path, output, *, with_fields=False):
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    baseline, manifest = read(baseline_path), read(manifest_path)
    if not manifest.get('baseline_sha256') or record_fingerprint(baseline) != manifest['baseline_sha256']:
        raise ValueError('Missing or stale baseline fingerprint')
    targets = manifest['targets']
    if any(not t.get('expected_record_sha256') for t in targets):
        raise ValueError('Target lacks frozen record fingerprint')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', str(case)) for case in baseline):
        raise ValueError('Unsafe case identifier')
    content = Path(readings_path).read_text(encoding='utf-8')
    data = json.loads(content)
    if not isinstance(data, dict) or not isinstance(data.get('items'), list):
        raise ValueError('Expected items array')
    ids = {row.get('id') for row in data['items'] if isinstance(row, dict) and isinstance(row.get('id'), str)}
    selected = [target for target in targets if target['id'] in ids]
    readings = parse_recovery(content, selected)
    recovered, audit = apply_recovery(baseline, targets, readings, include_fields=with_fields)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True)
    for case, prediction in recovered.items():
        atomic_json(output/f'{case}.json', prediction)
    atomic_json(output/'writeback_audit.json', audit)
    return {'output': str(output.resolve()), 'new_named_records': len(audit['changes']),
            'blocked': len(audit['blocked']), 'original_named_unchanged': True,
            'detection_count_unchanged': True, 'with_fields': with_fields,
            'api_calls': 0, 'label_source': 'machine_readings_not_verified_gold'}
