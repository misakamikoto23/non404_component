"""Bounded round005 fine-tuning from frozen human geometry; preserve defaults."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / 'outputs/active_learning/round_005'
os.environ['YOLO_CONFIG_DIR'] = str(LAB / 'outputs/ultralytics_settings')
os.environ['WANDB_DISABLED'] = 'true'
sys.path.insert(0, str(LAB / 'src'))
from component_lab.review_workflow import atomic_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    from ultralytics import YOLO
    initial = LAB / 'weights/pcbnet_component_yolo26n_review_round_004.pt'
    data = LAB / 'data/reviewed_versions/manual_round_005/yolo/dataset.yaml'
    run = LAB / 'runs/pcbnet_component_yolo26n_review_round_005_3050'
    if run.exists() or (OUT / 'comparison.json').exists():
        raise FileExistsError('No overwrite or automatic repeat')
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    matches = [i for i,n in enumerate(names) if '3050 Ti' in n]
    assert len(matches) == 1, 'Require RTX 3050 Ti'
    gpu = matches[0]
    torch.cuda.set_device(gpu)
    torch.set_num_threads(4)
    protected = {str(p.relative_to(LAB)): sha(p) for p in (LAB / 'weights').glob('*.pt')}
    atomic_json(OUT / 'protected_weights.json', protected)
    audit = read(OUT / 'review_audit.json')
    assert audit['completed_images'] == 20 and audit['holdouts_unchanged']
    recipe = dict(data=str(data), epochs=15, patience=5, imgsz=1280, batch=1, nbs=16,
        device=str(gpu), workers=0, amp=False, cache=False,
        project=str(LAB / 'runs'), name=run.name, exist_ok=False,
        optimizer='AdamW', lr0=.0001, lrf=.1, warmup_epochs=2, warmup_bias_lr=.0001,
        weight_decay=.0005, box=7.5, seed=42, deterministic=True,
        hsv_h=0, hsv_s=0, hsv_v=.1, degrees=0, translate=.05, scale=.2,
        shear=0, perspective=0, flipud=0, fliplr=0, mosaic=0, mixup=0, copy_paste=0,
        max_det=300, plots=False, save=True, save_period=-1)
    atomic_json(OUT / 'training_recipe.json', dict(initial=str(initial), initial_sha256=sha(initial),
        gpu=names[gpu], recipe=recipe, api_calls=0, test_used=False,
        caveat='Incremental legacy checkpoint with historical exposure;25 older train images removed by group rules. Not single-variable annotation ablation or clean retrain.'))
    baseline = YOLO(str(initial)).val(data=str(data), split='val', imgsz=1280, batch=1,
        device=str(gpu), workers=0, plots=False, max_det=300,
        project=str(OUT / 'evaluation'), name='round004_same_settings', exist_ok=False, verbose=False)
    old_metrics = {k:float(v) for k,v in baseline.results_dict.items()}
    atomic_json(OUT / 'round004_revalidation.json', old_metrics)
    model = YOLO(str(initial))
    epochs, epoch_start = [], [time.perf_counter()]
    def begin(trainer):
        epoch_start[0] = time.perf_counter()
    def end(trainer):
        if trainer.epoch >= trainer.epochs or (epochs and epochs[-1]['epoch'] == trainer.epoch+1):
            return
        row = dict(epoch=trainer.epoch+1, seconds=time.perf_counter()-epoch_start[0],
            peak_mib=torch.cuda.max_memory_allocated(gpu)/1024**2,
            metrics={k:float(v) for k,v in trainer.metrics.items()})
        epochs.append(row)
        atomic_json(OUT / 'STATUS.json', dict(state='training', gpu=names[gpu], epochs=epochs))
        print('ROUND005_EPOCH', json.dumps(row), flush=True)
        if row['seconds'] > 300:
            trainer.stop = True
    model.add_callback('on_train_epoch_start', begin)
    model.add_callback('on_fit_epoch_end', end)
    started = time.perf_counter()
    model.train(**recipe)
    current = YOLO(str(run / 'weights/best.pt')).val(data=str(data), split='val', imgsz=1280,
        batch=1, device=str(gpu), workers=0, plots=False, max_det=300,
        project=str(OUT / 'evaluation'), name='round005_same_settings', exist_ok=False, verbose=False)
    current_metrics = {k:float(v) for k,v in current.results_dict.items()}
    for name, digest in protected.items():
        assert sha(LAB / name) == digest, 'Existing model changed'
    target = LAB / 'weights/pcbnet_component_yolo26n_review_round_005.pt'
    assert not target.exists(), 'Never replace a published checkpoint'
    shutil.copy2(run / 'weights/best.pt', target)
    assert sha(target) == sha(run / 'weights/best.pt')
    with (run / 'results.csv').open() as stream:
        curve = [{k.strip():v for k,v in row.items()} for row in csv.DictReader(stream)]
    best_epoch = int(max(curve, key=lambda r:float(r['metrics/mAP50-95(B)']))['epoch'])
    result = dict(state='complete', gpu=names[gpu], initial=str(initial), weights=str(target),
        weights_sha256=sha(target), previous=old_metrics, current=current_metrics,
        epochs_completed=len(curve), best_epoch=best_epoch, seconds=time.perf_counter()-started,
        review=audit, recipe=recipe, protected_weights_unchanged=True, default_replaced=False,
        test_used=False, api_calls=0, evaluation='legacy PCBnet30 val; identical batch1/imgsz1280/whole-image; no independent HF val or strict ComponentF1')
    atomic_json(OUT / 'comparison.json', result)
    atomic_json(OUT / 'STATUS.json', result)
    lines = ['# 第5轮人工审核微调 — 2026-10-17', '',
        f'本批20张已冻结：最终{audit["totals"]["accepted"]}框，新增{audit["totals"]["added"]}、删除{audit["totals"]["rejected"]}、调整{audit["totals"]["moved_accepted"]}。累计历史100张；本次训练保留92张审核图。类别确认数为0，仅几何监督。', '',
        f'训练{audit["split_images"]["train"]}图/{audit["train_counts"]["component"]}框。25张旧训练图因设计组与保留集合冲突被移出（其中8张审核图），原文件及人工快照保留；此前72张有效审核标签保留。原val/test120文件哈希保持不变。', '',
        f'RTX3050 Ti cuda:{gpu}，1280/batch1/nbs16/AMP关闭/AdamW lr=.0001/box7.5；最多15轮patience5，实际{len(curve)}轮，best第{best_epoch}轮。', '',
        '|指标|Round004同设置重评|Round005|变化百分点|', '|---|---:|---:|---:|']
    for label,key in [('P','metrics/precision(B)'),('R','metrics/recall(B)'),('mAP50','metrics/mAP50(B)'),('mAP50–95','metrics/mAP50-95(B)')]:
        lines.append(f'|{label}|{old_metrics[key]:.2%}|{current_metrics[key]:.2%}|{100*(current_metrics[key]-old_metrics[key]):+.2f}|')
    lines += ['', 'P/R为Ultralytics验证曲线工作点，不是固定conf的计数。新增标注与移除旧图同时发生；物理batch也受3050显存约束改变，不能将全部差异归因于人工20图。旧模型有历史数据暴露，本轮不是clean retrain。无test40、HF独立验证或严格ComponentF1结论。', '',
        '所有原权重哈希未变，新权重另存round005；主infer默认未替换。API0，不恢复Small训练。', '',
        '复现脚本：scripts/train_review_round005_3050.py，全部实际参数、每轮指标和冻结标签审计在本目录。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('ROUND005_COMPLETE', json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(OUT / 'STATUS.json', dict(state='failed', error_type=type(error).__name__, message=str(error)))
        raise
