"""Bounded local PP-OCRv6 fallback with immutable primary text ownership."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .locked_ocr_recovery import lock_tokens, merge_locked, remove_anchor_duplicates
from .ocr_regions import select_regions

LAB = Path(__file__).resolve().parents[2]
VERSION = 'local-v6-locked-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def validate_options(*, mode, max_rois, area_fraction, timeout):
    if mode not in {'empty-fields', 'unresolved-only'}:
        raise ValueError('Unknown local-v6 mode')
    if not isinstance(max_rois, int) or not 0 <= max_rois <= 8:
        raise ValueError('local-v6 max ROIs must be an integer in 0..8')
    if not math.isfinite(area_fraction) or not 0 < area_fraction <= 1:
        raise ValueError('local-v6 area fraction must be in (0,1]')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('local-v6 timeout must be positive and finite')


def run_worker(image, regions, directory, *, python, model_root, timeout):
    """Cache a single bounded CPU subprocess. Incomplete attempts never auto-retry."""
    runtime = Path(python)
    models = [Path(model_root)/n for n in ('PP-OCRv6_det_medium.onnx', 'PP-OCRv6_rec_medium.onnx')]
    missing = [str(p) for p in [runtime, *models] if not p.is_file()]
    if missing:
        return [], {'status': 'missing_local_runtime_or_models', 'missing': missing, 'api_calls': 0}
    worker = Path(__file__).with_name('local_v6_worker.py')
    request = {'version': VERSION, 'image_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
               'size': list(image.size), 'regions': deepcopy(regions), 'model_root': str(Path(model_root).resolve()),
               'model_hashes': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in models},
               'worker_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
               'python': str(runtime.resolve()), 'settings': {'scale': 2, 'orientation': False, 'threads': 2, 'backend': 'ONNX CPU'}}
    site = runtime.resolve().parents[1]/'Lib/site-packages'
    request['runtime_metadata'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for name in ('rapidocr', 'onnxruntime')
                                   for p in site.glob(name+'-*.dist-info/METADATA')}
    # Bind cached observations to target IDs, exact crop pixels and OCR configuration.
    for roi in request['regions']:
        box = roi['box']
        roi['scaled_size'] = [(box[2]-box[0])*2, (box[3]-box[1])*2]
    key = fingerprint(request)
    cache = directory/'cache'/key
    response_path = cache/'response.json'
    cached = cache.exists()
    if not cached:
        try:
            cache.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            cached = True
    state = 'cache_hit' if cached else 'finished'
    seconds = 0.
    if not cached:
        for i, roi in enumerate(request['regions']):
            path = cache/f'roi_{i:02d}.png'
            image.crop(roi['box']).save(path)
            roi['crop_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        request['request_hash'] = key
        save(cache/'request.json', request)
        save(cache/'attempt.json', {'status': 'attempted', 'automatic_retries': 0})
        tick = time.perf_counter()
        try:
            with (cache/'worker.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen([str(runtime.resolve()), str(worker), str(cache.resolve())],
                                           stdout=log, stderr=subprocess.STDOUT, cwd=LAB)
                try:
                    code = process.wait(timeout=timeout)
                    if code:
                        state = 'worker_failed'
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
                    else:
                        process.kill()
                    process.wait()
                    state, code = 'timeout', process.returncode
        except OSError as exc:
            state, code = 'worker_start_failed', None
            save(cache/'start_error.json', {'error_type': type(exc).__name__})
        seconds = time.perf_counter()-tick
        save(cache/'run.json', {'status': state, 'exit_code': code, 'wall_seconds': seconds})
    response = json.loads(response_path.read_text(encoding='utf-8')) if response_path.exists() else None
    if response is None or response.get('request_hash') != key:
        return [], {'status': state if not cached else 'previous_attempt_incomplete', 'cache': str(cache),
                    'wall_seconds': seconds, 'api_calls': 0, 'texts': 0}
    tokens = [t for row in response['regions'] if row['status'] == 'ok' for t in row['texts']]
    # Namespace by request: it cannot collide with caller's primary IDs.
    for token in tokens:
        token['id'] = key[:12]+'_'+token['id']
    return tokens, {'status': state, 'cache': str(cache), 'complete': response['complete'],
                    'successful_rois': sum(r['status'] == 'ok' for r in response['regions']),
                    'requested_rois': len(regions), 'texts': len(tokens), 'wall_seconds': seconds,
                    'cached_ocr_seconds': response['seconds'], 'api_calls': 0,
                    'original_run_status': json.loads((cache/'run.json').read_text(encoding='utf-8'))['status'],
                    'runtime_metadata_bound': bool(request['runtime_metadata']),
                    'ocr_runtime': response.get('runtime')}


def build_proposals(identity, baseline, trace, targets, regions, tokens, height, display_names):
    from .contest_components import export_contest
    from .display_names import assign_display_names
    from .ocr_spans import split_ocr_spans
    from .text_matching import associate_structured
    from .text_roles import REFERENCE
    bytext = {t['id']: t for t in identity['texts']}
    selected = {r['target_id']: r for r in regions}
    bindings, anchors = [], []
    for row in trace:
        ident = row['source_id']
        for field, evidence in row['evidence'].items():
            if evidence.get('text_id') in bytext:
                bindings.append({'owner': ident, 'field': field, 'bbox': bytext[evidence['text_id']]['bbox']})
        evidence = row['evidence'].get('reference', {})
        if ident in selected and evidence.get('text_id') in bytext:
            token = deepcopy(bytext[evidence['text_id']])
            token.update(id='v6_anchor_'+ident, source_tier='primary_anchor', owner_ids=[ident])
            anchors.append(token)
    tokens, spans = split_ocr_spans(tokens)
    tokens, blocked = lock_tokens(tokens, bindings)
    tokens, duplicate_blocks = remove_anchor_duplicates(tokens, anchors)
    allowed = {t['id']: t['owner_ids'] for t in anchors}
    # All boxes still compete for fallback tokens; the merge checks original ROI
    # ownership again. An ROI does not force every visible word onto its target.
    adapted = deepcopy(identity['components'])
    semantic = {'r': 'resistor', 'c': 'capacitor', 'l': 'inductor', 'box': 'ic_or_module',
                'd': 'diode', 'gnd': 'ground', 'v': 'power_port'}
    for row in adapted:
        row['type'] = semantic.get(row['type'], row['type'])
    proposal_identity = associate_structured(adapted, tokens+anchors, clamp_costs=False,
                                             allowed_text_owners=allowed)
    if display_names:
        proposal_identity = assign_display_names(proposal_identity)
    proposed, proposed_trace = export_contest(proposal_identity, height, ic_value_as_name=True, display_names=display_names)
    local = {t['id']: t for t in tokens}
    byid = {t['id']: t for t in targets}
    proposals = []
    primary_refs = {t['source_id']: t['reference_raw'] for t in trace}
    for row in proposed_trace:
        ident = row['source_id']
        if ident not in selected:
            continue
        request = selected[ident]
        fields = {}
        for field in ('reference', 'Name', 'value'):
            if field not in request['slots'] and not (request['unresolved'] and field != 'reference'):
                continue
            source = 'reference' if field == 'reference' else 'value' if field == 'value' else row['name_source']
            if source == 'value_fallback':
                source = 'value'
            evidence = row['evidence'].get(source, {}) if source else {}
            oid = evidence.get('text_id')
            text = row['reference_raw'] if field == 'reference' else proposed['components'][row['key']][field]
            if oid not in local or text is None:
                continue
            if field == 'reference':
                previous_ref = primary_refs.get(ident)
                if previous_ref and sum(raw == previous_ref for raw in primary_refs.values()) > 1:
                    blocked.append({'id': ident, 'field': field, 'reason': 'duplicate_primary_reference_requires_review'})
                    continue
                match = REFERENCE.fullmatch(text.strip())
                kind = byid[ident]['baseline_record']['type']
                strong = {'R': {'r'}, 'C': {'c'}, 'L': {'l'}, 'U': {'box', 'amp', 'amp_3pin', 'amp_5pin', 'opt'}, 'IC': {'box'}}
                if not match or (match.group(1).upper() in strong and kind not in strong[match.group(1).upper()]):
                    blocked.append({'id': ident, 'field': field, 'reason': 'reference_prefix_visual_type_conflict'})
                    continue
                if any(raw == text and owner != ident for owner, raw in primary_refs.items()):
                    blocked.append({'id': ident, 'field': field, 'reason': 'existing_primary_reference_collision'})
                    continue
            fields[field] = {'text': text, 'ocr_id': oid}
        if fields:
            proposals.append({'id': ident, 'fields': fields})
    return proposals, tokens, {'blocked': blocked+duplicate_blocks, 'span_audit': spans,
                               'primary_bindings': bindings, 'fallback_trace': proposed_trace}


def synchronize(identity, baseline, merged, audit, proposals, tokens, height, display_names):
    """Commit accepted fields to the single evidence source used by both outputs."""
    from .contest_components import export_contest
    out = deepcopy(identity)
    byid = {r['id']: r for r in out['components']}
    byproposal = {r['id']: r['fields'] for r in proposals}
    bytoken = {t['id']: t for t in tokens}
    accepted = set()
    for change in audit['merge']['objects']:
        row = byid[change['id']]
        record = merged['components'][change['key']]
        for field in change['changed_fields']:
            proposal = byproposal[row['id']][field]
            token = bytoken[proposal['ocr_id']]
            slot = {'reference': 'reference', 'Name': 'part_number', 'value': 'value'}[field]
            raw = change['key'] if field == 'reference' else record[field]
            evidence = {'text_id': token['id'], 'raw': raw, 'role': slot, 'source_tier': 'fallback_v6',
                        'owner_ids': token['owner_ids'], 'source_window': token.get('source_window'),
                        'reason': 'local_v6_fill_empty_locked', 'uses_ground_truth': False}
            row['association_evidence'][slot] = evidence
            row[slot] = raw
            if field == 'Name':
                row['Name'] = raw
            if field == 'reference':
                row['physical_reference'] = raw
                row['review'] = [r for r in row.get('review', []) if r != 'reference_unresolved']
            row.setdefault('local_v6_changes', []).append({'field': field, 'text_id': token['id']})
            if token['id'] not in {t['id'] for t in row['texts']}:
                row['texts'].append(deepcopy(token))
            out['assignments'].append({'component_id': row['id'], **evidence})
            accepted.add(token['id'])
    out['texts'] += deepcopy(tokens)
    out['unassigned_texts'] += [{**deepcopy(t), 'reason': 'fallback_not_accepted'} for t in tokens if t['id'] not in accepted]
    exported, _ = export_contest(out, height, ic_value_as_name=True, display_names=display_names)
    if exported != merged:
        raise ValueError('Local-v6 internal evidence and contest export disagree')
    return out


def recover_local(identity, image, directory, *, mode='empty-fields', max_rois=2, area_fraction=.20,
                  timeout=120., python=None, model_root=None, display_names=False):
    from .contest_components import export_contest
    validate_options(mode=mode, max_rois=max_rois, area_fraction=area_fraction, timeout=timeout)
    started = time.perf_counter()
    directory = Path(directory)
    baseline, trace = export_contest(identity, image.height, ic_value_as_name=True, display_names=display_names)
    targets = [{'id': row['id'], 'bbox': row['bbox'], 'score': row.get('score'),
                'baseline_key': t['key'], 'baseline_record': baseline['components'][t['key']]}
               for row, t in zip(identity['components'], trace)]
    regions, selection = select_regions(image, targets, identity['texts'], max_rois=max_rois,
                                         area_fraction=area_fraction, mode=mode)
    save(directory/'baseline_contest.json', baseline)
    summary = {'mode': mode, 'selection': selection, 'regions': regions, 'api_calls': 0,
               'existing_named_references_locked': True, 'null_fill_can_change_correct_null': mode == 'empty-fields'}
    if not regions:
        summary.update(status='no_eligible_regions', changed_objects=0, wall_seconds=time.perf_counter()-started)
        save(directory/'summary.json', summary)
        return identity, summary
    tokens, worker = run_worker(image, regions, directory, python=python or LAB/'.venv-ocr/Scripts/python.exe',
                                model_root=model_root or LAB/'weights/ocr_rapid', timeout=timeout)
    summary['worker'] = worker
    if not tokens:
        status = worker['status']
        if status in {'finished', 'cache_hit'}:
            status = 'baseline_preserved_no_texts' if worker.get('successful_rois') == len(regions) else 'ocr_failed_or_incomplete'
        summary.update(status=status, changed_objects=0, wall_seconds=time.perf_counter()-started)
        save(directory/'summary.json', summary)
        return identity, summary
    proposals, tokens, evidence = build_proposals(identity, baseline, trace, targets, regions, tokens, image.height, display_names)
    merged, audit = merge_locked(baseline, targets, proposals, tokens,
                                 'empty_fields_shadow' if mode == 'empty-fields' else 'safe_unresolved')
    recovered = synchronize(identity, baseline, merged, audit, proposals, tokens, image.height, display_names)
    save(directory/'evidence.json', {'tokens': tokens, 'proposals': proposals, **evidence})
    save(directory/'audit.json', audit)
    all_success = worker.get('complete') and worker.get('successful_rois', len(regions)) == len(regions)
    summary.update(status='complete' if all_success else 'partial',
                   changed_objects=audit['merge']['changed_objects'], wall_seconds=time.perf_counter()-started)
    save(directory/'summary.json', summary)
    return recovered, summary
