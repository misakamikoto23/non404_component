"""One real-image CLI check and cache replay; no labels, training or network API."""
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB/'src'))
from component_lab.local_v6 import save

OUT = LAB/'outputs/local_v6_integration_20261018'
if len(sys.argv) > 1:
    OUT = OUT/sys.argv[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (OUT/'smoke.json').exists():
        print('Already verified; see', OUT/'smoke.json')
        return
    pilot = read(LAB/'outputs/targeted_v6_recovery_20261018/manifest.json')
    row = next(r for r in pilot['cases'] if r['case'] == '0131')
    assert digest(row['image']) == row['image_sha256']
    assert digest(row['detector_path']) == row['detector_sha256']
    assert digest(row['ocr_v5_cache']) == row['ocr_sha256']
    protected = {str(p): digest(p) for p in (LAB/'weights').glob('*.pt')}
    semantic = {'r': 'resistor', 'c': 'capacitor', 'l': 'inductor', 'box': 'ic_or_module',
                'd': 'diode', 'gnd': 'ground', 'v': 'power_port'}
    detections = read(row['detector_path'])['rows']
    for det in detections:
        det['type'] = semantic.get(det['type'], det['type'])
    detection_path = OUT/'input_detections.json'
    save(detection_path, {'components': detections})
    args = [sys.executable, '-m', 'component_lab.cli', 'infer', '--image', row['image'],
            '--detections', str(detection_path), '--ocr-json', row['ocr_v5_cache'],
            '--association', 'identity-v1', '--ocr-spans', '--display-names', '--contest-export']
    commands, timing = [], {}
    env = dict(os.environ, PYTHONUTF8='1', YOLO_CONFIG_DIR=str(OUT/'ultralytics'))
    def run(name, output, extra=()):
        command = [*args, '--output', str(output), *extra]
        commands.append(command)
        tick = time.perf_counter()
        with (OUT/(name+'.log')).open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=LAB, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        timing[name] = time.perf_counter()-tick
    run('baseline', OUT/'baseline', ['--no-local-v6'])
    run('enabled', OUT/'enabled')
    baseline = read(OUT/'baseline/contest_components.json')
    enabled = read(OUT/'enabled/contest_components.json')
    rich = read(OUT/'enabled/components.json')
    first_summary = rich['local_v6']
    observations = list((OUT/'enabled/local_v6/cache').glob('*/response.json'))
    frozen_ocr = {str(p): digest(p) for p in observations}
    run('cache_replay', OUT/'enabled')
    replay = read(OUT/'enabled/contest_components.json')
    replay_rich = read(OUT/'enabled/components.json')
    assert replay == enabled
    assert frozen_ocr and all(digest(p) == h for p, h in frozen_ocr.items())
    assert replay_rich['local_v6']['worker']['status'] == 'cache_hit'
    assert len(enabled['components']) == len(baseline['components'])
    changes = []
    for key, record in baseline['components'].items():
        if key.startswith('__unresolved'):
            continue
        new = enabled['components'][key]
        assert record['bbox'] == new['bbox'] and record['type'] == new['type']
        for field in ('Name', 'value'):
            if record[field] is not None:
                assert new[field] == record[field]
            if new[field] != record[field]:
                changes.append({'key': key, 'field': field, 'before': record[field], 'after': new[field]})
    assert all(digest(path) == h for path, h in protected.items())
    result = {'case': '0131', 'input_source': row, 'timing_seconds': timing, 'first_run': first_summary,
              'replay': replay_rich['local_v6'], 'named_field_changes': changes,
              'checks': {'replay_identical_contest_json': True, 'ocr_response_hashes_unchanged': True,
                         'existing_nonempty_fields_unchanged': True, 'original_weights_unchanged': True,
                         'detector_count_unchanged': True},
              'api_calls': 0, 'ground_truth_read': False, 'commands': commands,
              'protected_weights': protected, 'ocr_response_hashes': frozen_ocr}
    save(OUT/'smoke.json', result)
    cards = []
    for req in first_summary['regions']:
        cache = Path(first_summary['worker']['cache'])
        i = first_summary['regions'].index(req)
        cards.append(f'<p>{html.escape(req["target_id"])} / {html.escape(str(req["slots"]))}</p><img src="{(cache/f"roi_{i:02d}.png").as_uri()}" style="max-width:100%">')
    text = f'已接入 identity-v1：主OCR关联 → 最多2个局部v6 → 空字段锁定合并 → 统一导出。真实CLI检查共{len(enabled["components"])}个框；不读GT，不把该检查称作准确率评测。'
    (OUT/'gallery.html').write_text('<meta charset="utf-8"><style>body{max-width:1000px;margin:30px auto;font:16px Arial}pre{white-space:pre-wrap}</style><h1>局部v6工作流接入验证</h1><p>'+text+'</p><pre>'+html.escape(json.dumps({'changes': changes, 'timing_seconds': timing, 'checks': result['checks']}, ensure_ascii=False, indent=2))+'</pre>'+''.join(cards), encoding='utf-8')
    print(json.dumps({'changes': changes, 'timing_seconds': timing, 'checks': result['checks'],
                      'status': first_summary['status'], 'worker': first_summary.get('worker')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
