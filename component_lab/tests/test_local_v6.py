import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
import pytest

from component_lab.contest_components import associate_variants, export_contest
from component_lab.local_v6 import recover_local, run_worker, save, validate_options
from component_lab.pipeline import infer
from component_lab.ocr_regions import context_box, select_regions


def identity(value=None, reference='R1'):
    texts = []
    if reference:
        texts.append({'id': 'primary_ref', 'raw': reference, 'bbox': [245, 205, 265, 215]})
    if value:
        texts.append({'id': 'primary_value', 'raw': value, 'bbox': [245, 220, 285, 230]})
    return associate_variants([{'id': 'a', 'type': 'resistor', 'bbox': [220, 205, 240, 235]}], texts)[0]['identity_v1']


def fallback(raw='10kΩ', owner='a', ident='fallback', box=None):
    return {'id': ident, 'raw': raw, 'text': raw, 'bbox': box or [245, 220, 285, 230],
            'score': .9, 'source_tier': 'fallback_v6', 'owner_ids': [owner]}


def test_pipeline_synchronizes_primary_and_contest_outputs(tmp_path):
    image = tmp_path/'image.png'
    Image.new('RGB', (800, 600), 'white').save(image)
    detections, ocr = tmp_path/'d.json', tmp_path/'o.json'
    save(detections, {'components': [{'type': 'resistor', 'bbox': [220, 205, 240, 235]}]})
    save(ocr, {'texts': [{'raw': 'R1', 'text': 'R1', 'bbox': [245, 205, 265, 215]}]})
    args = SimpleNamespace(image=str(image), detections=str(detections), ocr_json=str(ocr),
                           output=str(tmp_path/'out'), resnet=None, qwen=False, qwen_max_calls=0,
                           context_margin=20, text_margin=60, classifier_threshold=.8,
                           association='identity-v1', contest_export=True, local_v6=True)
    with patch('component_lab.local_v6.run_worker', return_value=([fallback(owner='component_0001')], {'status': 'finished', 'complete': True})):
        result = infer(args)
    rich = json.loads((tmp_path/'out/components.json').read_text('utf-8'))
    contest = json.loads((tmp_path/'out/contest_components.json').read_text('utf-8'))
    assert result['local_v6_changed_objects'] == 1
    assert rich['components'][0]['value'] == contest['components']['R1']['value'] == '10kΩ'
    assert rich['components'][0]['association_evidence']['value']['source_tier'] == 'fallback_v6'
    assert contest['components']['R1']['bbox'] == [220, 365, 240, 395]
    assert len(contest['components']) == 1 and contest['pins'] == contest['nets'] == {}


def test_runtime_failure_keeps_primary_export(tmp_path):
    image = tmp_path/'image.png'
    Image.new('RGB', (800, 600), 'white').save(image)
    save(tmp_path/'d.json', {'components': [{'type': 'resistor', 'bbox': [220, 205, 240, 235]}]})
    save(tmp_path/'o.json', {'texts': [{'raw': 'R1', 'bbox': [245, 205, 265, 215]}]})
    args = SimpleNamespace(image=str(image), detections=str(tmp_path/'d.json'), ocr_json=str(tmp_path/'o.json'),
                           output=str(tmp_path/'out'), resnet=None, qwen=False, qwen_max_calls=0,
                           context_margin=20, text_margin=60, classifier_threshold=.8,
                           association='identity-v1', contest_export=True, local_v6=True)
    with patch('component_lab.local_v6.run_worker', side_effect=RuntimeError('worker unavailable')):
        infer(args)
    rich = json.loads((tmp_path/'out/components.json').read_text('utf-8'))
    exported = json.loads((tmp_path/'out/contest_components.json').read_text('utf-8'))
    assert rich['local_v6']['status'] == 'failed_baseline_preserved'
    assert exported['components']['R1']['value'] is None
    assert len(exported['components']) == 1


def test_region_budget_and_geometry_bounds():
    targets = [{'id': str(i), 'baseline_key': '__unresolved_'+str(i), 'bbox': [i*100+5, 30, i*100+20, 60],
                'baseline_record': {'type': 'r', 'Name': None, 'value': None}} for i in range(6)]
    with patch('component_lab.ocr_regions.extract_features', return_value=[]):
        regions, summary = select_regions(Image.new('RGB', (800, 600)), targets, [])
    assert len(regions) <= 2 and summary['crop_area_fraction'] <= .2
    for box in [r['box'] for r in regions]+[context_box([.1, .2, 999.3, 999.4], (1000, 1000))]:
        assert max(box[2]-box[0], box[3]-box[1]) <= 320
        assert all(isinstance(v, int) for v in box)


def test_safe_mode_does_not_query_named_record(tmp_path):
    original = identity()
    with patch('component_lab.local_v6.run_worker') as worker:
        after, summary = recover_local(original, Image.new('RGB', (800, 600)), tmp_path, mode='unresolved-only')
    worker.assert_not_called()
    assert after == original and summary['changed_objects'] == 0


def test_nonempty_literal_is_never_replaced(tmp_path):
    original = identity('0.01 μF')
    with patch('component_lab.local_v6.run_worker') as worker:
        after, _ = recover_local(original, Image.new('RGB', (800, 600)), tmp_path)
    worker.assert_not_called()
    assert after == original


def test_unresolved_key_and_fields_can_be_recovered_without_type_change(tmp_path):
    original = identity(reference=None)
    tokens = [fallback('R2', ident='ref', box=[245, 205, 265, 215]), fallback()]
    with patch('component_lab.local_v6.run_worker', return_value=(tokens, {'status': 'finished', 'complete': True})):
        after, summary = recover_local(original, Image.new('RGB', (800, 600)), tmp_path)
    exported, _ = export_contest(after, 600, ic_value_as_name=True)
    assert exported['components']['R2'] == {'type': 'r', 'bbox': [220, 365, 240, 395], 'Name': None, 'value': '10kΩ'}
    assert original['components'][0]['reference'] is None
    assert summary['changed_objects'] == 1


def test_foreign_roi_token_rejected(tmp_path):
    original = identity()
    with patch('component_lab.local_v6.run_worker', return_value=([fallback(owner='neighbor')], {'status': 'finished', 'complete': True})):
        after, summary = recover_local(original, Image.new('RGB', (800, 600)), tmp_path)
    assert export_contest(after, 600) == export_contest(original, 600)
    assert summary['changed_objects'] == 0


def test_duplicate_primary_refs_do_not_cascade_rename_other_records(tmp_path):
    original = associate_variants([
        {'id': 'a', 'type': 'resistor', 'bbox': [220, 205, 240, 235]},
        {'id': 'b', 'type': 'resistor', 'bbox': [520, 205, 540, 235]}], [
        {'id': 'pa', 'raw': 'R1', 'bbox': [245, 205, 265, 215]},
        {'id': 'pb', 'raw': 'R1', 'bbox': [545, 205, 565, 215]}])[0]['identity_v1']
    tokens = [fallback('R2', ident='ref', box=[245, 205, 265, 215]), fallback()]
    with patch('component_lab.local_v6.run_worker', return_value=(tokens, {'status': 'finished', 'complete': True})):
        after, _ = recover_local(original, Image.new('RGB', (800, 600)), tmp_path)
    before_export, _ = export_contest(original, 600)
    after_export, _ = export_contest(after, 600)
    assert set(before_export['components']) == set(after_export['components'])
    assert all(key.startswith('__unresolved') for key in after_export['components'])
    assert after_export['components']['__unresolved_detection_0001']['value'] == '10kΩ'
    evidence = json.loads((tmp_path/'evidence.json').read_text('utf-8'))
    assert any(row['reason'] == 'duplicate_primary_reference_requires_review' for row in evidence['blocked'])


def test_partial_worker_is_not_reported_as_complete(tmp_path):
    with patch('component_lab.local_v6.run_worker', return_value=([fallback()], {
            'status': 'finished', 'complete': True, 'successful_rois': 0})):
        _, summary = recover_local(identity(), Image.new('RGB', (800, 600)), tmp_path)
    assert summary['status'] == 'partial'


def test_ic_name_updates_the_evidence_read_by_exporter(tmp_path):
    original = associate_variants([{'id': 'a', 'type': 'ic_or_module', 'bbox': [200, 230, 300, 330]}],
                                  [{'id': 'ref', 'raw': 'U1', 'bbox': [220, 195, 240, 205]}])[0]['identity_v1']
    token = fallback('STM32F103', box=[220, 210, 280, 220])
    with patch('component_lab.local_v6.run_worker', return_value=([token], {'status': 'finished', 'complete': True})):
        after, _ = recover_local(original, Image.new('RGB', (800, 600)), tmp_path)
    exported, _ = export_contest(after, 600, ic_value_as_name=True)
    assert after['components'][0]['Name'] == exported['components']['U1']['Name'] == 'STM32F103'


def test_no_model_files_falls_back_without_launch(tmp_path):
    with patch('component_lab.local_v6.subprocess.Popen') as spawn:
        tokens, status = run_worker(Image.new('RGB', (20, 20)), [], tmp_path,
                                    python=tmp_path/'absent.exe', model_root=tmp_path/'models', timeout=1)
    spawn.assert_not_called()
    assert not tokens and status['status'] == 'missing_local_runtime_or_models'


def resources(root):
    runtime = root/'python.exe'
    runtime.write_bytes(b'fake runtime')
    for name in ('PP-OCRv6_det_medium.onnx', 'PP-OCRv6_rec_medium.onnx'):
        (root/name).write_bytes(b'fake model')
    return runtime


def test_exact_cache_prevents_repeated_subprocess(tmp_path):
    runtime = resources(tmp_path)
    def spawn(command, **kwargs):
        cache = Path(command[-1])
        request = json.loads((cache/'request.json').read_text('utf-8'))
        save(cache/'response.json', {'request_hash': request['request_hash'], 'complete': True, 'seconds': 1,
                                    'regions': [{'status': 'ok', 'texts': [fallback()]}]})
        return Mock(wait=Mock(return_value=0))
    regions = [{'target_id': 'a', 'slots': ['value'], 'box': [0, 0, 20, 20]}]
    kwargs = dict(python=runtime, model_root=tmp_path, timeout=1)
    with patch('component_lab.local_v6.subprocess.Popen', side_effect=spawn) as process:
        first, _ = run_worker(Image.new('RGB', (20, 20)), regions, tmp_path, **kwargs)
        second, status = run_worker(Image.new('RGB', (20, 20)), regions, tmp_path, **kwargs)
    assert process.call_count == 1
    assert first == second and status['status'] == 'cache_hit'


def test_timeout_keeps_baseline_and_does_not_retry(tmp_path):
    runtime = resources(tmp_path)
    process = Mock(pid=12345, returncode=-1)
    process.wait.side_effect = [subprocess.TimeoutExpired('worker', 1), -1]
    regions = [{'target_id': 'a', 'slots': ['value'], 'box': [0, 0, 20, 20]}]
    kwargs = dict(python=runtime, model_root=tmp_path, timeout=1)
    with patch('component_lab.local_v6.subprocess.Popen', return_value=process) as spawn, patch('component_lab.local_v6.subprocess.run'):
        tokens, status = run_worker(Image.new('RGB', (20, 20)), regions, tmp_path, **kwargs)
        _, again = run_worker(Image.new('RGB', (20, 20)), regions, tmp_path, **kwargs)
    assert not tokens and status['status'] == 'timeout'
    assert spawn.call_count == 1 and again['status'] == 'previous_attempt_incomplete'


@pytest.mark.parametrize('options', [{'timeout': float('nan')}, {'max_rois': -1}, {'area_fraction': 0}])
def test_invalid_settings_rejected(options):
    params = dict(mode='empty-fields', max_rois=2, area_fraction=.2, timeout=120)
    with pytest.raises(ValueError):
        validate_options(**{**params, **options})


@pytest.mark.parametrize('extra,enabled', [([], True), (['--no-local-v6'], False), (['--local-v6'], True)])
def test_cli_local_v6_default_and_disable(extra, enabled):
    from component_lab.cli import main
    with patch('sys.argv', ['component-lab', 'infer', '--image', 'image.png', '--detections', 'd.json',
                            '--output', 'out', '--association', 'identity-v1', *extra]), \
            patch('component_lab.pipeline.infer', return_value={}) as run:
        main()
    assert run.call_args[0][0].local_v6 is enabled
