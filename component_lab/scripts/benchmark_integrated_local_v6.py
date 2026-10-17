"""Fixed five-dev-case CLI ablation; freeze predictions before reading GT."""
from collections import Counter
from copy import deepcopy
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB/'src'))
from component_lab.local_v6 import save
from component_lab.ocr_regions import select_regions

OUT = LAB/'outputs/integrated_v6_five_case_test_20261018'
CASES = ['0024', '0034', '0058', '0107', '0131']
SEMANTIC = {'r': 'resistor', 'c': 'capacitor', 'l': 'inductor', 'box': 'ic_or_module',
            'd': 'diode', 'gnd': 'ground', 'v': 'power_port'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    if (OUT/'predictions_frozen.json').exists():
        print('Predictions already frozen; inference not repeated', flush=True)
        return
    if (OUT/'run_started.json').exists():
        raise FileExistsError('Prior run exists; inspect status instead of starting a duplicate')
    original = read(LAB/'outputs/targeted_v6_recovery_20261018/manifest.json')
    sources = [next(r for r in original['cases'] if r['case'] == case) for case in CASES]
    for row in sources:
        for path, expected in [(row['image'], row['image_sha256']), (row['detector_path'], row['detector_sha256']),
                               (row['ocr_v5_cache'], row['ocr_sha256'])]:
            assert sha(path) == expected
    protected = {str(p): sha(p) for p in (LAB/'weights').glob('*.pt')}
    source_files = ['src/component_lab/local_v6.py', 'src/component_lab/local_v6_worker.py',
                    'src/component_lab/ocr_regions.py', 'src/component_lab/pipeline.py',
                    'src/component_lab/cli.py', 'configs/defaults.json']
    manifest = {'sources': sources, 'cases': CASES, 'protected': protected,
                'source_hashes': {p: sha(LAB/p) for p in source_files},
                'settings': {'mode': 'empty-fields', 'max_rois': 2, 'area_fraction': .20, 'timeout': 120,
                             'visual_input': 'frozen 37-class short-trained YOLO26n at 1024; no new detection/classification',
                             'primary_ocr': 'existing v5 cache', 'api_calls': 0,
                             'type_adapter': SEMANTIC, 'selection_uses_GT': False}}
    save(OUT/'manifest.json', manifest)
    save(OUT/'run_started.json', {'manifest_sha256': sha(OUT/'manifest.json')})
    env = dict(os.environ, PYTHONUTF8='1', YOLO_CONFIG_DIR=str(OUT/'ultralytics'))
    completed, frozen, roi_plans = [], {}, {}
    for row in sources:
        case = row['case']
        root = OUT/case
        detections = deepcopy(read(row['detector_path'])['rows'])
        for det in detections:
            det['raw_detector_type'] = det['type']
            det['type'] = SEMANTIC.get(det['type'], det['type'])
        save(root/'detections.json', {'components': detections})
        command = [sys.executable, '-m', 'component_lab.cli', 'infer', '--image', row['image'],
                   '--detections', str(root/'detections.json'), '--ocr-json', row['ocr_v5_cache'],
                   '--association', 'identity-v1', '--ocr-spans', '--display-names', '--contest-export']
        record = {'case': case, 'commands': {}, 'seconds': {}}
        for mode in ['baseline', 'enabled']:
            if mode == 'enabled':
                rich = read(root/'baseline/components.json')
                baseline = read(root/'baseline/contest_components.json')
                trace = rich['contest_export']['trace']
                targets = [{'id': c['id'], 'bbox': c['bbox'], 'score': c.get('score'),
                            'baseline_key': t['key'], 'baseline_record': baseline['components'][t['key']]}
                           for c, t in zip(rich['components'], trace)]
                from PIL import Image
                with Image.open(row['image']) as image:
                    regions, info = select_regions(image.convert('RGB'), targets, rich['ocr_texts'])
                roi_plans[case] = {'regions': regions, 'selection': info}
                save(OUT/'roi_plans_prediction_only.json', roi_plans)
                # Only copy an exact production cache; the worker independently
                # verifies its request key, model/worker/runtime fingerprints.
                if case == '0131':
                    old = LAB/'outputs/local_v6_integration_20261018/verified/enabled/local_v6/cache'
                    if old.exists():
                        shutil.copytree(old, root/'enabled/local_v6/cache')
            args = [*command, '--output', str(root/mode)]
            if mode == 'baseline':
                args += ['--no-local-v6']
            record['commands'][mode] = args
            tick = time.perf_counter()
            with (root/(mode+'.log')).open('w', encoding='utf-8') as log:
                try:
                    result = subprocess.run(args, cwd=LAB, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=160)
                    record.setdefault('exit_codes', {})[mode] = result.returncode
                except subprocess.TimeoutExpired:
                    record.setdefault('exit_codes', {})[mode] = 'outer_timeout'
            record['seconds'][mode] = time.perf_counter()-tick
            path = root/mode/'contest_components.json'
            # A failed case stays in the denominator. Never silently skip it.
            if not path.exists():
                save(path, {'components': {}, 'pins': {}, 'nets': {}})
                record.setdefault('failures', []).append(mode)
            frozen[str(path.relative_to(OUT))] = sha(path)
        rich = read(root/'enabled/components.json') if (root/'enabled/components.json').exists() else {}
        record['local_v6'] = rich.get('local_v6', {'status': 'cli_failed'})
        assert record['local_v6'].get('regions', []) == roi_plans[case]['regions']
        completed.append(record)
        save(OUT/'inference_status.json', {'state': 'running', 'cases': completed})
        print(case, record['local_v6']['status'], 'changes', record['local_v6'].get('changed_objects', 0),
              'seconds', {k: round(v, 2) for k, v in record['seconds'].items()}, flush=True)
    for path, expected in protected.items():
        assert sha(path) == expected
    for path, expected in manifest['source_hashes'].items():
        assert sha(LAB/path) == expected
    save(OUT/'predictions_frozen.json', {'files': frozen, 'GT_read_before_freeze': False})
    save(OUT/'inference_status.json', {'state': 'complete', 'cases': completed})


def evaluate():
    from component_lab.eval.metric import ScoreConfig, score_case, normalize_case, bbox_ok, text_ok
    frozen = read(OUT/'predictions_frozen.json')
    for path, expected in frozen['files'].items():
        assert sha(OUT/path) == expected
    config = ScoreConfig()
    per_case, aggregate = [], {m: Counter() for m in ['baseline', 'enabled']}
    identities = {}
    changes = []
    invariants = []
    for case in CASES:
        gt = read(next((LAB/'data/official_contest6/train200').glob(case+'-*.json')))
        normalized_gt = normalize_case(gt)['components']
        preds = {}
        row = {'case': case}
        for mode in ['baseline', 'enabled']:
            pred = read(OUT/case/mode/'contest_components.json')
            preds[mode] = pred
            score = score_case(pred, gt, config)
            row[mode] = score
            aggregate[mode].update(score['counts']['component_f1'])
            norm = normalize_case(pred)['components']
            identities[case, mode] = {key for key, target in normalized_gt.items()
                                      if key in norm and norm[key][0] == target[0]
                                      and bbox_ok(norm[key][3], target[3], config) and text_ok(norm[key], target)}
        before, after = preds['baseline']['components'], preds['enabled']['components']
        row['added'] = sorted(identities[case, 'enabled']-identities[case, 'baseline'])
        row['lost'] = sorted(identities[case, 'baseline']-identities[case, 'enabled'])
        for key, old in before.items():
            if key.startswith('__unresolved'):
                continue
            assert key in after and after[key]['bbox'] == old['bbox'] and after[key]['type'] == old['type']
            for field in ['Name', 'value']:
                if old[field] is not None:
                    assert after[key][field] == old[field]
        assert len(before) == len(after)
        audit_path = OUT/case/'enabled/local_v6/audit.json'
        if audit_path.exists():
            baseline_rich = read(OUT/case/'baseline/components.json')
            references = {t['source_id']: t['reference_raw'] for t in baseline_rich['contest_export']['trace']}
            for changed in read(audit_path)['merge']['objects']:
                for field in changed['changed_fields']:
                    previous = references[changed['id']] if field == 'reference' else before[changed['baseline_key']][field]
                    current = changed['key'] if field == 'reference' else after[changed['key']][field]
                    changes.append({'case': case, 'key': changed['key'], 'baseline_key': changed['baseline_key'],
                                    'field': field, 'before': previous, 'after': current,
                                    'GT_contains_key': changed['key'] in normalized_gt,
                                    'added_strict_TP': changed['key'] in row['added'],
                                    'lost_strict_TP': changed['baseline_key'] in row['lost']})
        invariants.append({'case': case, 'named_keys_nonempty_fields_geometry_type_preserved': True,
                           'detection_count_preserved': True})
        per_case.append(row)
    summaries = {}
    for mode in aggregate:
        counts = aggregate[mode]
        summaries[mode] = {**dict(counts), 'macro_component_f1': sum(r[mode]['component_f1'] for r in per_case)/len(CASES),
                           'micro_component_f1': 2*counts['tp']/max(1, 2*counts['tp']+counts['fp']+counts['fn']),
                           'added_TP': sum(len(r['added']) for r in per_case) if mode == 'enabled' else 0,
                           'lost_TP': sum(len(r['lost']) for r in per_case) if mode == 'enabled' else 0}
    status = read(OUT/'inference_status.json')
    timing = {'baseline_cli_seconds': sum(c['seconds']['baseline'] for c in status['cases']),
              'enabled_cli_seconds': sum(c['seconds']['enabled'] for c in status['cases']),
              'new_worker_seconds': sum(c['local_v6'].get('worker', {}).get('wall_seconds', 0) for c in status['cases']),
              'new_worker_images': sum(c['local_v6'].get('worker', {}).get('status') == 'finished' for c in status['cases']),
              'cache_hit_images': sum(c['local_v6'].get('worker', {}).get('status') == 'cache_hit' for c in status['cases']),
              'rois': sum(len(c['local_v6'].get('regions', [])) for c in status['cases'])}
    report = {'summaries': summaries, 'per_case': per_case, 'changes': changes, 'timing': timing,
               'invariants': invariants, 'metric': 'local strict, raw types, strict anonymous IDs, max per-coordinate error <=20, exact required text; macro over all five dev cases',
               'inference': status, 'api_calls': 0, 'test40_read': False, 'tuned_after_scoring': False}
    save(OUT/'report.json', report)
    lines = ['# 已集成局部v6：固定五图真实CLI测试', '',
             '两组都复用相同检测框/type与v5主OCR缓存；唯一处理开关为局部v6。使用刚接入的实际CLI，先冻结两组10份预测再读取GT评分。不是重新运行YOLO/ResNet，也不是完整端到端耗时。', '',
             '|方法|严格macro ComponentF1|完整正确TP|新增TP|损失旧TP|', '|---|---:|---:|---:|---:|']
    for mode, s in summaries.items():
        lines.append(f'|{mode}|{s["macro_component_f1"]:.2%}|{s["tp"]}|{s["added_TP"]}|{s["lost_TP"]}|')
    lines += ['', '|图|关闭补读F1|开启补读F1|新增/损失TP|ROI数|状态|', '|---|---:|---:|---|---:|---|']
    for row, execution in zip(per_case, status['cases']):
        local = execution['local_v6']
        lines.append(f'|{row["case"]}|{row["baseline"]["component_f1"]:.2%}|{row["enabled"]["component_f1"]:.2%}|{row["added"]} / {row["lost"]}|{len(local.get("regions", []))}|{local["status"]}|')
    lines += ['', '## 实际字段改动', '', '```json', json.dumps(changes, ensure_ascii=False, indent=2), '```', '',
              '## 耗时与缓存', '', '```json', json.dumps(timing, ensure_ascii=False, indent=2), '```', '',
              '总计时含CLI启动、缓存加载、关联、选区、裁图及写文件；主OCR与检测/分类是缓存，不能当作完整系统耗时。0131复用上一轮真实生产worker的精确缓存，其过去OCR耗时不计本轮新增worker耗时。其余选区只执行一次，没有付费请求或自动重试。', '',
              '## 限制', '',
              '固定5张dev图片，不是独立测试集；旧37类短训模型、1024下仅76框。评分为本地解释的严格ComponentF1，不是mAP，也不是官方完整总分。无ROI图和失败图仍保留在五图分母中。当前CLI含多轮关联，不能假定与旧独立实验逐条相同；本次关闭补读组是唯一直接基线。', '',
              'raw detector类型仅通过固定输入映射适配CLI（v→power_port等）；保留CLI真实导出，不使用GT修类型，也不把它称为历史原rawtype流水线的复现。', '',
              '本轮4个元件发生改动，共6个字段：0024恢复K7/K6及各自Name，但公开GT没有这些key，得分不变；0107为错误位号R4330补33Ω，仍不计TP；0131的R3补10kΩ，新增1TP。当前真实CLI基线14.10%不同于旧独立实验13.29%，因此不能把两次绝对分数差当作v6收益。', '',
              '旧命名key、非空原文、bbox/type、检测数量通过检查。允许补null依然可能破坏正确null，本次无损失不能保证其他图片安全。0024存在已记录的图面K*与公开GT U*编号冲突，无法单凭图像猜出内部编号。', '',
              '未读取test40、未修改权重或人工/原始标签、未训练。运行入口：', '',
              '```powershell', '.\\.venv-yolo\\Scripts\\python.exe scripts/benchmark_integrated_local_v6.py', '```', '',
              '已冻结预测时脚本不再启动推理；所有实际CLI参数在inference_status.json中。']
    text = '\n'.join(lines)
    (OUT/'REPORT.md').write_text(text, encoding='utf-8')
    cards = []
    for row, execution in zip(per_case, status['cases']):
        case = row['case']
        local = execution['local_v6']
        links = ' · '.join(f'<a href="{case}/{m}/contest_components.json">{m} JSON</a>' for m in ['baseline', 'enabled'])
        pictures = ''
        if local.get('worker', {}).get('cache'):
            cache = Path(local['worker']['cache'])
            pictures = ''.join(f'<figure><img src="{(cache/f"roi_{i:02d}.png").as_uri()}"><figcaption>{html.escape(str(region["slots"]))} / {html.escape(region["target_id"])}</figcaption></figure>'
                               for i, region in enumerate(local['regions']))
        cards.append(f'<section><h2>{case}: {row["baseline"]["component_f1"]:.2%} → {row["enabled"]["component_f1"]:.2%}</h2><p>{links}</p>{pictures or "没有合格ROI，保留基线并计分。"}</section>')
    (OUT/'gallery.html').write_text('<meta charset="utf-8"><style>body{max-width:1100px;margin:30px auto;font:16px Arial;line-height:1.5}pre{white-space:pre-wrap}img{max-width:100%}figure{display:inline-block;margin:12px;border:1px solid #ccc;padding:10px}section{border-top:1px solid #ccc;padding-top:20px}</style><h1>已集成局部v6：五图对照</h1><pre>'+html.escape(text)+'</pre>'+''.join(cards), encoding='utf-8')
    save(OUT/'STATUS.json', {'state': 'complete', 'summaries': summaries, 'api_calls': 0, 'new_worker_images': timing['new_worker_images']})
    print(json.dumps({'summaries': summaries, 'timing': timing, 'changes': changes}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    run()
    evaluate()
