"""Fixed dev20 inference/assignment experiment. No GT is read before freeze."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from PIL import Image

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB/'src'))
from component_lab.local_v6 import save

OUT = LAB/'outputs/dev20_v6_qwen_assignment_20261018'
PRIOR = LAB/'outputs/multiclass_vs_two_stage_20261017'
SEMANTIC = {'r': 'resistor', 'c': 'capacitor', 'l': 'inductor', 'box': 'ic_or_module',
            'd': 'diode', 'gnd': 'ground', 'v': 'power_port'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def prepare_local():
    if (OUT/'manifest.json').exists():
        return
    split = read(LAB/'configs/contest6_group_split_v1.json')
    cases = sorted(r['case'] for r in split['partitions']['dev'])
    frozen = read(PRIOR/'predictions_frozen.json')['files']
    pixels = {r['case']: r['pixel_sha256'] for r in read(PRIOR/'ocr_source_checks.json')}
    sources = []
    for case in cases:
        detector = PRIOR/'predictions/whole/multiclass'/f'{case}.json'
        assert sha(detector) == frozen[str(detector.relative_to(PRIOR))]
        cache = read(detector)
        image = LAB/'outputs/component_identity_round_001_20261017/inputs'/case/'source.png'
        ocr = LAB/'outputs/unattended_20261017_lunch/span_inputs'/case/'ocr_spans.json'
        assert sha(ocr) == cache['frozen_ocr_sha256']
        with Image.open(image) as im:
            rgb = im.convert('RGB')
            assert hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode()+rgb.tobytes()).hexdigest() == pixels[case]
        rows = deepcopy(cache['rows'])
        for row in rows:
            row['raw_detector_type'] = row['type']
            row['type'] = SEMANTIC.get(row['type'], row['type'])
        detection = OUT/'inputs'/case/'detections.json'
        save(detection, {'components': rows})
        sources.append({'case': case, 'image': str(image), 'image_sha256': sha(image),
                        'ocr': str(ocr), 'ocr_sha256': sha(ocr), 'detection': str(detection),
                        'detection_sha256': sha(detection), 'original_detector': str(detector),
                        'original_detector_sha256': sha(detector), 'count': len(rows)})
    save(OUT/'manifest.json', {'cases': sources, 'split_sha256': sha(LAB/'configs/contest6_group_split_v1.json'),
                              'original_weights': {str(p): sha(p) for p in (LAB/'weights').glob('*.pt')},
                              'settings': {'max_rois': 2, 'area_fraction': .2, 'timeout': 120,
                                           'mode': 'empty-fields', 'adapter': SEMANTIC},
                              'GT_read': False})
    print('Prepared', len(sources), 'cases /', sum(r['count'] for r in sources), 'frozen detections', flush=True)


def run_local():
    from component_lab.pipeline import infer
    m = read(OUT/'manifest.json')
    if (OUT/'local_predictions_frozen.json').exists():
        print('Local predictions already frozen; no repeated inference', flush=True)
        return
    records, frozen = [], {}
    previous = LAB/'outputs/integrated_v6_five_case_test_20261018'
    for source in m['cases']:
        case = source['case']
        for field in ('image', 'ocr', 'detection'):
            assert sha(source[field]) == source[field+'_sha256']
        status_path = OUT/'local'/case/'complete.json'
        if status_path.exists():
            row = read(status_path)
            records.append(row)
            frozen.update(row['frozen'])
            continue
        attempt = status_path.with_name('attempt.json')
        if attempt.exists():
            raise FileExistsError(f'Case {case} already attempted; inspect before resuming')
        save(attempt, {'status': 'attempted', 'automatic_retries': 0})
        row = {'case': case, 'seconds': {}, 'frozen': {}}
        for mode in ('primary', 'local_v6'):
            output = OUT/'local'/case/mode
            if mode == 'local_v6':
                cache = previous/case/'enabled/local_v6/cache'
                if cache.exists():
                    shutil.copytree(cache, output/'local_v6/cache')
            args = SimpleNamespace(image=source['image'], detections=source['detection'], ocr_json=source['ocr'],
                                   output=str(output), resnet=None, qwen=False, qwen_max_calls=0, context_margin=48,
                                   text_margin=60, classifier_threshold=.8, association='identity-v1',
                                   ocr='none', ocr_spans=True, display_names=True, contest_export=True,
                                   local_v6=(mode == 'local_v6'))
            tick = time.perf_counter()
            infer(args)
            row['seconds'][mode] = time.perf_counter()-tick
            target = output/'contest_components.json'
            row['frozen'][str(target.relative_to(OUT))] = sha(target)
        a = read(OUT/'local'/case/'primary/contest_components.json')['components']
        b = read(OUT/'local'/case/'local_v6/contest_components.json')['components']
        assert len(a) == len(b) == source['count']
        for key, record in a.items():
            if key.startswith('__unresolved'):
                continue
            assert key in b and record['type'] == b[key]['type'] and record['bbox'] == b[key]['bbox']
            for field in ('Name', 'value'):
                if record[field] is not None:
                    assert record[field] == b[key][field]
        row['local_v6'] = read(OUT/'local'/case/'local_v6/components.json')['local_v6']
        save(status_path, row)
        records.append(row)
        frozen.update(row['frozen'])
        save(OUT/'LOCAL_STATUS.json', {'state': 'running', 'completed': len(records), 'cases': records})
        print(case, 'complete', row['local_v6']['status'], 'changed', row['local_v6'].get('changed_objects', 0),
              's', round(row['seconds']['local_v6'], 2), flush=True)
    assert all(sha(p) == expected for p, expected in m['original_weights'].items())
    save(OUT/'LOCAL_STATUS.json', {'state': 'complete', 'completed': len(records), 'cases': records})
    save(OUT/'local_predictions_frozen.json', {'files': frozen, 'GT_read': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare-local', 'run-local'])
    args = parser.parse_args()
    {'prepare-local': prepare_local, 'run-local': run_local}[args.stage]()
