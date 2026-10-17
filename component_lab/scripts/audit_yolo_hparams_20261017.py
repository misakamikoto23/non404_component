"""Bounded one-factor YOLO26n pilot; historical review; frozen dev diagnostics."""
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / 'outputs/yolo_hparam_review_20261017'
PRIOR = LAB / 'outputs/binary_yolo_3050_20261017'
DATA = LAB.parent / 'Data/yolo_experiments/clean_binary_20261017/rgb'
os.environ['YOLO_CONFIG_DIR'] = str(LAB / 'outputs/ultralytics_settings')
os.environ['WANDB_DISABLED'] = 'true'
sys.path.insert(0, str(LAB / 'src'))
from component_lab.review_workflow import atomic_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    return {str(p.relative_to(LAB)): sha(p) for p in sorted((LAB / 'weights').glob('*.pt'))}


def protect():
    assert snapshot() == read(OUT / 'protected_weights.json'), 'Business weights changed'
    for relative, digest in read(OUT / 'data_hashes.json').items():
        assert sha(DATA / relative) == digest, 'Frozen train/dev data changed'


def history():
    import yaml
    rows = []
    for directory in sorted((LAB / 'runs').glob('pcbnet_component_yolo26n*')):
        if not (directory / 'results.csv').exists():
            continue
        args = yaml.safe_load((directory / 'args.yaml').read_text(encoding='utf-8'))
        with (directory / 'results.csv').open() as stream:
            epochs = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(stream)]
        if not epochs:
            continue
        best = max(epochs, key=lambda r: float(r['metrics/mAP50-95(B)']))
        rows.append(dict(run=directory.name, config={k: args.get(k) for k in
            ['model', 'data', 'epochs', 'patience', 'imgsz', 'batch', 'nbs', 'optimizer', 'lr0',
             'warmup_epochs', 'warmup_bias_lr', 'box', 'cls', 'dfl', 'max_det',
             'translate', 'scale', 'mosaic', 'hsv_v', 'amp', 'cos_lr']},
            actual_epochs=len(epochs), best_epoch=int(best['epoch']),
            best_map=float(best['metrics/mAP50-95(B)']),
            last_map=float(epochs[-1]['metrics/mAP50-95(B)']),
            first_train_box=float(epochs[0]['train/box_loss']),
            last_train_box=float(epochs[-1]['train/box_loss'])))
    atomic_json(OUT / 'history.json', rows)


def prepare():
    if OUT.exists():
        raise FileExistsError('Do not overwrite or repeat a paid/compute experiment')
    OUT.mkdir(parents=True)
    atomic_json(OUT / 'protected_weights.json', snapshot())
    atomic_json(OUT / 'data_hashes.json', {p.relative_to(DATA).as_posix(): sha(p)
        for p in sorted(DATA.rglob('*')) if p.is_file() and p.suffix in ('.png', '.txt', '.yaml')})
    assert sha(LAB / 'weights/yolo26n.pt') == read(PRIOR / 'CONFIG.json')['initial_sha256']
    baseline = read(PRIOR / 'rgb_result.json')
    assert baseline['state'] == 'complete' and len(baseline['epochs']) == 8
    history()
    plan = dict(initial=str(LAB / 'weights/yolo26n.pt'), epochs=8, seed=42,
        baseline='reuse completed RGB8 pilot; same data, initialization, seed, augmentation and schedule',
        arms={'lower_lr': {'lr0': .0003}, 'box15': {'box': 15.0}},
        recipe={'lr0': .001, 'box': 7.5, 'warmup_bias_lr': .001, 'imgsz': 1280,
                'batch': 1, 'nbs': 16, 'amp': False, 'optimizer': 'AdamW'},
        device='RTX 3050 Ti only', test_used=False, api_calls=0,
        scope='Short-run optimization sensitivity, not converged hyperparameter ranking. No default replacement.',
        frozen_thresholds=[.05, .1, .25, .4, .6])
    atomic_json(OUT / 'PLAN.json', plan)
    atomic_json(OUT / 'STATUS.json', {'state': 'prepared'})


def device():
    import torch
    found = [i for i in range(torch.cuda.device_count()) if '3050 Ti' in torch.cuda.get_device_name(i)]
    assert len(found) == 1, 'Refusing another GPU'
    torch.cuda.set_device(found[0])
    torch.set_num_threads(4)
    return torch, found[0]


def train_arm(name, changes):
    from ultralytics import YOLO
    import yaml
    protect()
    torch, gpu = device()
    path = OUT / 'runs' / name
    assert not path.exists(), 'Never silently resume a completed arm'
    model = YOLO(str(LAB / 'weights/yolo26n.pt'))
    progress, epoch_start = [], [time.perf_counter()]
    started = time.perf_counter()
    def begin(trainer):
        epoch_start[0] = time.perf_counter()
    def end(trainer):
        if trainer.epoch >= trainer.epochs:
            return
        row = dict(epoch=trainer.epoch + 1, seconds=time.perf_counter() - epoch_start[0],
                   peak_mib=torch.cuda.max_memory_allocated(gpu) / 1024**2,
                   metrics={k: float(v) for k, v in trainer.metrics.items()})
        progress.append(row)
        atomic_json(OUT / 'STATUS.json', dict(state='training', arm=name, epochs=progress))
        print('PROGRESS', name, row['epoch'], round(row['seconds'], 1), flush=True)
        if row['seconds'] > 300 or time.perf_counter() - started > 900:
            trainer.stop = True
    model.add_callback('on_train_epoch_start', begin)
    model.add_callback('on_fit_epoch_end', end)
    kwargs = dict(data=str(DATA / 'dataset.yaml'), epochs=8, fraction=1., imgsz=1280,
        batch=1, device=str(gpu), workers=0, amp=False, cache=False,
        project=str(OUT / 'runs'), name=name, exist_ok=False,
        optimizer='AdamW', lr0=.001, lrf=.1, weight_decay=.0005,
        warmup_epochs=1, warmup_bias_lr=.001, seed=42, deterministic=True,
        nbs=16, patience=8, plots=False, save=True, save_period=-1,
        hsv_h=0, hsv_s=0, hsv_v=0, translate=.05, scale=.2, degrees=0, shear=0,
        perspective=0, flipud=0, fliplr=0, mosaic=0, close_mosaic=0,
        mixup=0, copy_paste=0, max_det=1000)
    kwargs.update(changes)
    model.train(**kwargs)
    assert len(progress) == 8, 'Incomplete run; do not compare partial epochs'
    # Check the resolved settings, not only the caller's intended parameters.
    base_args = yaml.safe_load((PRIOR / 'runs/rgb/args.yaml').read_text(encoding='utf-8'))
    arm_args = yaml.safe_load((path / 'args.yaml').read_text(encoding='utf-8'))
    ignored = {'project', 'name', 'save_dir'}
    differences = {k: [base_args.get(k), arm_args.get(k)] for k in base_args.keys() | arm_args.keys()
                   if k not in ignored and base_args.get(k) != arm_args.get(k)}
    assert set(differences) == set(changes), differences
    atomic_json(OUT / (name + '_result.json'), dict(state='complete', seconds=time.perf_counter() - started,
        epochs=progress, differences_from_baseline=differences, best=str(path / 'weights/best.pt'),
        last=str(path / 'weights/last.pt')))
    protect()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def geometric_counts(pred, scores, gt, threshold, rule):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    p = pred[scores >= threshold]
    if not len(p) or not len(gt):
        return {'tp': 0, 'fp': len(p), 'fn': len(gt)}
    if rule == 'maxcoord20':
        error = np.max(np.abs(p[:, None, :] - gt[None, :, :]), axis=2)
        good = error <= 20
        tie = np.minimum(error / 20, 1)
    else:
        lo = np.maximum(p[:, None, :2], gt[None, :, :2])
        hi = np.minimum(p[:, None, 2:], gt[None, :, 2:])
        inter = np.prod(np.maximum(hi - lo, 0), axis=2)
        pa = np.prod(np.maximum(p[:, 2:] - p[:, :2], 0), axis=1)
        ga = np.prod(np.maximum(gt[:, 2:] - gt[:, :2], 0), axis=1)
        iou = inter / np.maximum(pa[:, None] + ga[None, :] - inter, 1e-9)
        good, tie = iou >= .5, 1 - iou
    # Maximize match cardinality first. The bounded tie term cannot trade away a valid match.
    cardinality_scale = min(len(p), len(gt)) + 1
    rows, cols = linear_sum_assignment((~good).astype(float) * cardinality_scale + tie)
    tp = int(good[rows, cols].sum())
    return dict(tp=tp, fp=len(p)-tp, fn=len(gt)-tp)


def evaluate():
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    torch, gpu = device()
    arms = {'baseline_rgb8': Path(read(PRIOR / 'rgb_result.json')['best']),
            **{n: Path(read(OUT / (n + '_result.json'))['best']) for n in ('lower_lr', 'box15')}}
    results = []
    for name, weights in arms.items():
        target = OUT / 'evaluations' / name / 'metrics.json'
        if target.exists():
            results.append(read(target))
            continue
        model = YOLO(str(weights))
        metrics = model.val(data=str(DATA / 'dataset.yaml'), split='val', imgsz=1280,
            batch=1, device=str(gpu), workers=0, plots=False, max_det=1000,
            project=str(OUT / 'evaluations'), name=name, exist_ok=False, verbose=False)
        cases = []
        for image in sorted((DATA / 'images/val').glob('*.png')):
            with Image.open(image) as im:
                width, height = im.size
            gt = []
            for line in (DATA / 'labels/val' / image.with_suffix('.txt').name).read_text().splitlines():
                _, cx, cy, w, h = map(float, line.split())
                gt.append([(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height])
            output = model.predict(str(image), imgsz=1280, conf=.01, max_det=1000,
                                   device=str(gpu), verbose=False)[0]
            pred = output.boxes.xyxy.cpu().numpy()
            scores = output.boxes.conf.cpu().numpy()
            counts = {rule: {str(t): geometric_counts(pred, scores, np.asarray(gt).reshape(-1, 4), t, rule)
                             for t in [.05, .1, .25, .4, .6]}
                      for rule in ['iou50', 'maxcoord20']}
            cases.append(dict(case=image.stem, predictions=pred.tolist(), scores=scores.tolist(), counts=counts))
        atomic_json(OUT / 'evaluations' / name / 'predictions.json', cases)
        totals = {}
        for rule in ['iou50', 'maxcoord20']:
            totals[rule] = {}
            for t in ['0.05', '0.1', '0.25', '0.4', '0.6']:
                row = {k: sum(c['counts'][rule][t][k] for c in cases) for k in ['tp', 'fp', 'fn']}
                row.update(precision=row['tp']/max(row['tp']+row['fp'], 1),
                           recall=row['tp']/max(row['tp']+row['fn'], 1),
                           f1=2*row['tp']/max(2*row['tp']+row['fp']+row['fn'], 1))
                totals[rule][t] = row
        result = dict(name=name, checkpoint=str(weights), checkpoint_sha256=sha(weights),
                      metrics={k: float(v) for k, v in metrics.results_dict.items()},
                      detection_only_sweep=totals, test_used=False)
        atomic_json(target, result)
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(OUT / 'comparison.json', results)
    protect()


def report():
    import html
    rows = read(OUT / 'comparison.json')
    os.environ['MPLCONFIGDIR'] = str(OUT / 'matplotlib_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, directory in [('baseline_rgb8', PRIOR / 'runs/rgb'),
                            ('lower_lr', OUT / 'runs/lower_lr'), ('box15', OUT / 'runs/box15')]:
        with (directory / 'results.csv').open() as stream:
            curve = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream)]
        axes[0].plot([int(r['epoch']) for r in curve],
                     [100*float(r['metrics/mAP50(B)']) for r in curve], marker='o', label=name)
        row = next(r for r in rows if r['name'] == name)
        points = list(row['detection_only_sweep']['iou50'].values())
        axes[1].plot([100*p['recall'] for p in points], [100*p['precision'] for p in points],
                     marker='o', label=name)
    axes[0].set(xlabel='Epoch', ylabel='dev20 mAP50 (%)', title='Same budget, one-factor training pilot')
    axes[1].set(xlabel='Recall (%)', ylabel='Precision (%)', title='Fixed confidence thresholds; IoU >= 0.5')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.png', dpi=160)
    plt.close(fig)
    lines = ['# YOLO26n超参数回顾与单变量短训 — 2026-10-17', '',
        '保留全部原模型、人工标签和默认流程；只用RTX 3050 Ti，新增API 0，test40未使用。', '',
        '官方train140/dev20：6417/931框。复用已有RGB 8轮基线；另从同一通用预训练权重独立训练两组8轮。seed42、1280、batch1、nbs16、AdamW、AMP关闭；全部增强、调度与基线相同。', '',
        '|单变量|P（曲线工作点）|R（曲线工作点）|mAP50|mAP50–95|', '|---|---:|---:|---:|---:|']
    labels = {'baseline_rgb8': '基线 lr=.001 / box=7.5', 'lower_lr': '仅 lr0=.0003', 'box15': '仅 box=15'}
    for row in rows:
        m = row['metrics']
        lines.append(f"|{labels[row['name']]}|{m['metrics/precision(B)']:.2%}|{m['metrics/recall(B)']:.2%}|{m['metrics/mAP50(B)']:.2%}|{m['metrics/mAP50-95(B)']:.2%}|")
    lines += ['', '结论：统一重评下，降低lr0并未改善8轮预算的结果；box加倍的mAP50/mAP50–95与基线基本相当，没有替换当前配方的证据。不能据此判定所有学习率或长训练已无价值。',
              '', 'box15在conf=.25的IoU50计数中召回37.81%（基线34.26%），但精确率64.71%（基线70.89%）；这包含置信度工作点取舍，不代表总体AP提高。',
              '', 'max-coordinate20几何F1在conf=.25时box15为59.80%、基线55.32%；但预声明阈值网格中基线conf=.1已达60.35%，box15网格最高59.80%。因此也不能只选某个阈值证明框权重优胜。这仍只是几何诊断，完全不检查元件身份/必要文字。']
    lines += ['', '## 固定阈值整图检测计数', '',
              '以下为IoU≥0.5、最大匹配数的一对一几何匹配，非Ultralytics工作点，非ComponentF1。另有max-coordinate≤20px几何诊断保存在comparison.json；它不检查key/type/text，不能视为比赛得分。', '',
              '|训练组|conf|TP|FP|FN|P|R|F1|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        for threshold, c in row['detection_only_sweep']['iou50'].items():
            lines.append(f"|{row['name']}|{threshold}|{c['tp']}|{c['fp']}|{c['fn']}|{c['precision']:.2%}|{c['recall']:.2%}|{c['f1']:.2%}|")
    lines += ['', '## 历史回顾与解释限制', '',
        '- 初始PCBnet模型：32轮早停，best17；第4轮：12轮早停，best7。第4轮训练box loss继续下降（1.125→1.082），末轮验证mAP50–95低于best（.5261 vs .5325），不支持简单追加epoch必然有效。',
        '- 第1—4轮均lr0=.0001、box=7.5、batch2/nbs16，多为继承配方，并未系统调参；1920实验已修正不适合微调的默认bias预热lr=.1。',
        '- max_det历史为300。本次1000；不能自动认为密集图一定被截断，需独立检查具体图及检测头限制。',
        '- 每组只8轮、单seed，比较的是短训练预算下的敏感性，不是充分收敛后的最优超参数。降低学习率可能仅延缓收敛。原PCBnet30图86.20%与此dev20不能直接比较。',
        '- box是框回归损失权重，不是“框缩小比例”；其影响须用结果验证。未比较所有优化器、增强、batch或长训练。',
        '- nbs16控制梯度累积与损失/衰减缩放，不等同于真实batch16的批归一化统计。',
        '- 必要文字错误主要属于OCR/归属/字段协议；检测超参数可以间接影响裁剪和归属，但本轮未重跑OCR，不能声明修复了严格ComponentF1。',
        '- 当前未替换业务模型，也不根据封存测试选参数。复现源代码scripts/audit_yolo_hparams_20261017.py，完整resolved args在runs各目录。', '',
        '训练参数说明参考：https://docs.ultralytics.com/modes/train/ 。本机实际行为以锁定环境和落盘args.yaml为准。']
    lines += ['', '## 必要文字失败的含义', '',
        '按本地严格口径：先匹配元件key；IC比较Name，电阻/电容比较value。原文大小写、空格和单位不归一化，null双方一致才算匹配。该null解释是待主办方确认的本地假设。', '',
        '固定17.42%基线中，57个元件只卡在必要文字：18预测空/GT非空，12预测非空/GT空，17字面差异小，10差异较大。14项比较Name，43项比较value。112项key命中但失败的元件中，合计72项含text失败（另外15项还同时错框/类型）。', '',
        '|case/key|字段|预测|GT|可确认的现象|', '|---|---|---|---|---|',
        '|0130/R12|value|10k|10K|大小写不同，严格判错|',
        '|0107/R13|value|10k0|10kΩ|原文字符不同，可能为OCR误读，不能仅凭GT自动改写|',
        '|0011/R8|value|null|100k|漏填；需区分读不到与读到未归属|',
        '|0144/U1|Name|CTRL1|ADUM1411BRWZ|字段差异很大，可能错误归属网络/引脚文字，需核图|',
        '|0001/C18|value|0.1uF|null|非空与null协议差异；不能仅凭此认定OCR错|', '',
        '18个漏填项有12个在全页OCR缓存中已有同字符串，6个没有；同字符串存在不证明owner正确。170个完整TP中有11个正确null，盲目填空可毁掉原正确项。详见../component_text_only_audit_20261017/REPORT.md。', '',
        '本轮没有重读文字或按GT修正预测。检测训练超参数不能直接承担Ω识别、文本归属与null协议的修复。']
    text = '\n'.join(lines) + '\n'
    (OUT / 'REPORT.md').write_text(text, encoding='utf-8')
    (OUT / 'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>YOLO超参数对照</title><style>body{max-width:1050px;margin:32px auto;font:16px/1.65 system-ui;padding:16px}pre{white-space:pre-wrap}a{color:#1870bc}img{max-width:100%}</style><h1>YOLO超参数回顾</h1><p><a href="REPORT.md">完整报告</a> · <a href="comparison.json">指标JSON</a></p><img src="comparison.png" alt="Training and fixed-threshold detection curves"><pre>'+html.escape(text)+'</pre>', encoding='utf-8')
    atomic_json(OUT / 'STATUS.json', dict(state='complete', protected_weights_unchanged=True,
                                         test_used=False, api_calls=0))


def main():
    prepare()
    try:
        for name, changes in read(OUT / 'PLAN.json')['arms'].items():
            train_arm(name, changes)
        evaluate()
        report()
    except Exception as error:
        atomic_json(OUT / 'STATUS.json', dict(state='failed', error_type=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
