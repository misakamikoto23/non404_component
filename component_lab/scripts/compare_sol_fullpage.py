"""Evaluate blinded native-full-image Sol annotations against frozen human boxes.

This tool does no inference, API calls, coordinate repair or training.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import random
from PIL import Image, ImageDraw
import benchmark_qwen_fullpage as base
from analyze_yolo_errors import box_iou

MODELS = [('gpt-5.6-sol','medium'),('gpt-6-sol','low')]

def label(model, effort):
    return model+'_'+effort

def counts(rows):
    tp, fp, fn = [sum(r[k] for r in rows) for k in ('tp','fp','fn')]
    result = base.count_summary(tp,fp,fn)
    result['f1'] = 2*tp/(2*tp+fp+fn) if tp+fp+fn else 0
    return result

def load_annotations(out, reference, require_completion=True):
    neutral = base.read(out/'annotation_task.json')
    if set(neutral) != {'prompt','coordinate_system','images'}:
        raise ValueError('Neutral task has unexpected fields')
    if any(set(s) != {'case','path','width','height','sha256'} for s in neutral['images']):
        raise ValueError('Neutral image task has unexpected fields')
    neutral_images = {s['case']: s for s in neutral['images']}
    if set(neutral_images) != {s['case'] for s in reference['samples']}:
        raise ValueError('Image selection differs')
    if neutral['prompt'] != reference['prompt']:
        raise ValueError('Detection prompts differ')
    records, file_hashes, completions = [], {}, {}
    for model, effort in MODELS:
        ident = label(model,effort)
        directory = out/'annotations'/ident
        completion_path = directory/'completion.json'
        if require_completion and not completion_path.exists():
            raise ValueError(f'Agent has not completed: {ident}')
        if completion_path.exists():
            completions[ident] = base.read(completion_path)
        for sample in reference['samples']:
            case = sample['case']
            image = Path(neutral_images[case]['path'])
            if base.sha(image) != sample['api_sha256']:
                raise ValueError('Benchmark input bytes changed')
            with Image.open(image) as original:
                if original.size != (sample['width'],sample['height']):
                    raise ValueError('Native full-image dimensions differ')
            annotation = directory/(case+'.json')
            if not annotation.exists():
                raise ValueError(f'Missing annotation: {ident}/{case}')
            content = annotation.read_text(encoding='utf-8-sig')
            predictions, invalid = base.parse_boxes(content,sample['width'],sample['height'],allow_extra_keys=True)
            file_hashes[str(annotation.relative_to(out))] = base.sha(annotation)
            row = {'case':case,'model':model,'reasoning_effort':effort,'label':ident,
                   'stratum':sample['stratum'],'predictions':predictions,'invalid_box_indices':invalid,
                   'extra_keys_ignored':set(json.loads(content)) != {'boxes'},'metrics':{},'matching':{}}
            for threshold in (.3,.5,.75):
                stats, matching = base.evaluate(sample['ground_truth'],predictions,threshold,len(invalid))
                row['metrics'][str(threshold)] = stats
                row['matching'][str(threshold)] = matching
            records.append(row)
    return records, file_hashes, completions

def bootstrap(records, cases):
    labels = [label(*m) for m in MODELS]
    pairs = [{name:next(r for r in records if r['case']==case and r['label']==name)['metrics']['0.5'] for name in labels} for case in cases]
    wins = {name:sum(p[name]['f1']>p[labels[1-labels.index(name)]]['f1'] for p in pairs) for name in labels}
    deltas = {key:[] for key in ('precision','recall','f1')}
    rng = random.Random(20261016)
    for _ in range(10000):
        draw = rng.choices(pairs,k=len(pairs))
        scores = {name:counts([p[name] for p in draw]) for name in labels}
        for key in deltas:
            deltas[key].append((scores[labels[0]][key] or 0)-(scores[labels[1]][key] or 0))
    return {'first_minus_second':labels,'f1_wins':wins,'ties':sum(p[labels[0]]['f1']==p[labels[1]]['f1'] for p in pairs),
            'bootstrap_95_intervals':{key:[sorted(values)[249],sorted(values)[9749]] for key,values in deltas.items()},
            'note':'Exploratory paired image bootstrap, 10000 resamples, seed 20261016; only ten images.'}

def validation_checks(reference, records):
    """Independent matching and coordinate roundtrip, never modifying labels."""
    samples={s['case']:s for s in reference['samples']}
    maximum={label(*m):0 for m in MODELS}
    for sample in reference['samples']:
        w,h=sample['width'],sample['height']
        boxes=[[b['bbox'][0]/w*1000,b['bbox'][1]/h*1000,b['bbox'][2]/w*1000,b['bbox'][3]/h*1000] for b in sample['ground_truth']]
        preds,invalid=base.parse_boxes(json.dumps({'boxes':boxes}),w,h)
        stats,_=base.evaluate(sample['ground_truth'],preds,.999999)
        if invalid or stats['tp']!=len(sample['ground_truth']):
            raise AssertionError('Coordinate roundtrip failed')
    for row in records:
        sample=samples[row['case']]
        graph=[[j for j,g in enumerate(sample['ground_truth']) if box_iou(p['bbox'],g['bbox'])>=.5] for p in row['predictions']]
        owner={}
        def augment(i,seen):
            for j in graph[i]:
                if j not in seen:
                    seen.add(j)
                    if j not in owner or augment(owner[j],seen):
                        owner[j]=i
                        return True
            return False
        maximum[row['label']]+=sum(augment(i,set()) for i in range(len(graph)))
        for stats in row['metrics'].values():
            assert stats['tp']+stats['fn']==len(sample['ground_truth'])
            assert stats['tp']+stats['fp']==len(row['predictions'])+len(row['invalid_box_indices'])
    return {'coordinate_roundtrip_all_gt_at_iou_0_999999':True,'count_denominators_verified':True,
            'maximum_cardinality_tp_at_0_5':maximum,
            'primary_greedy_tp_at_0_5':{label(*m):sum(r['metrics']['0.5']['tp'] for r in records if r['label']==label(*m)) for m in MODELS}}

def report(out):
    reference = base.read(out/'reference.json')
    experiment = base.read(out/'experiment.json')
    records, hashes, completions = load_annotations(out,reference)
    cases = [s['case'] for s in reference['samples']]
    summary = {'created_at':datetime.now(timezone.utc).isoformat(),'experiment':experiment,
               'image_count':len(cases),'ground_truth':sum(len(s['ground_truth']) for s in reference['samples']),
               'models':{},'paired_comparison':bootstrap(records,cases),
               'usage':'Per-agent tokens and isolated monetary cost are unavailable; not estimated.',
               'matching':'Original-pixel IoU; one-to-one greedy score=1; invalid boxes count FP; no coordinate repair, clipping or NMS.',
               'annotation_sha256':hashes,'completions':completions}
    summary['validation']=validation_checks(reference,records)
    base.atomic_json(out/'validation_checks.json',summary['validation'])
    for model,effort in MODELS:
        ident = label(model,effort)
        subset = [r for r in records if r['label']==ident]
        metrics = {str(t):counts([r['metrics'][str(t)] for r in subset]) for t in (.3,.5,.75)}
        invalid = sum(len(r['invalid_box_indices']) for r in subset)
        primary = metrics['0.5']
        filtered = counts([dict(primary,fp=primary['fp']-invalid)])
        summary['models'][ident] = {'model':model,'reasoning_effort':effort,'images':len(subset),'metrics':metrics,
            'invalid_boxes':invalid,'valid_boxes_only_at_0.5':filtered,
            'by_count_stratum':{name:counts([r['metrics']['0.5'] for r in subset if r['stratum']==name]) for name in ('sparse','medium','dense')}}
    # Original Qwen statistics are reference-only: different execution protocol.
    qwen = base.read(Path(experiment['qwen_reference_output'])/'summary.json')
    summary['qwen_reference_only'] = {'models':qwen['models'],
        'limitation':'Qwen used fresh single-call-per-image API requests with temperature=0 and max_tokens=16000. Sol uses tool-viewing agents, one agent per model with ten images in context. Reasoning budgets and preprocessing are not identical.'}
    base.atomic_json(out/'evaluated_annotations.json',records)
    base.atomic_json(out/'summary.json',summary)
    lines = ['# Sol整图标注实验：5.6-Sol中等推理与6-Sol轻度推理','',
             '沿用上一轮同10张原图和825个人工框；相同检测提示词；包含地/电源/网络端口，不拆IC/连接器引脚。两个独立智能体不继承父对话，只读取中性任务与完整原图，人工标签和其他模型输出不提供给标注者。','',
             '**执行方式：每款Sol模型一个智能体，逐图以原始分辨率看完整图片，不使用裁剪、切片、OCR或检测器。智能体会保留同组之前图片的上下文。这与Qwen每图一次独立API请求不同，temperature、输出预算和内部视觉处理也无法完全对齐；四模型表仅作参考。**','',
             '所有框按全图0..1000归一化xyxy还原；IoU一对一匹配；非法框计误检；不做坐标修复、裁边、NMS或基于GT的调框。不提供分类准确率或mAP。','',
             '| 模型与推理级别 | IoU | P | R | F1 | TP | FP | FN |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for ident,result in summary['models'].items():
        for threshold,stats in result['metrics'].items():
            lines.append(f"| {ident} | {threshold} | {(stats['precision'] or 0):.2%} | {(stats['recall'] or 0):.2%} | {stats['f1']:.2%} | {stats['tp']} | {stats['fp']} | {stats['fn']} |")
    lines += ['','| 模型 | 非法框 | 自动过滤非法框后P@0.5 |','|---|---:|---:|']
    for ident,result in summary['models'].items():
        lines.append(f"| {ident} | {result['invalid_boxes']} | {(result['valid_boxes_only_at_0.5']['precision'] or 0):.2%} |")
    lines += ['','## 四模型参考（IoU≥0.5）','','| 模型 | P | R | F1 |','|---|---:|---:|---:|']
    for ident,result in {**summary['models'],**qwen['models']}.items():
        stats=result['metrics']['0.5']
        lines.append(f"| {ident} | {(stats['precision'] or 0):.2%} | {(stats['recall'] or 0):.2%} | {stats['f1']:.2%} |")
    lines += ['','## 逐图结果（IoU≥0.5）','','| 图片 | 模型 | 人工框 | TP | FP | FN | P | R |','|---|---|---:|---:|---:|---:|---:|---:|']
    pages=[]
    for sample in reference['samples']:
        case=sample['case']
        source=out/'inputs'/(case+'.png')
        with Image.open(source) as image:
            original=image.convert('RGB')
        panels=[base.draw_panel(original,sample['ground_truth'])]
        captions=[f'Human: {len(sample["ground_truth"])} boxes']
        detail_links=[]
        for model,effort in MODELS:
            ident=label(model,effort)
            row=next(r for r in records if r['case']==case and r['label']==ident)
            stats=row['metrics']['0.5']
            panel=base.draw_panel(original,sample['ground_truth'],row['predictions'],row['matching']['0.5'])
            panel.save(out/f'{case}_{ident}.png')
            panels.append(panel)
            captions.append(f"{ident}: P={(stats['precision'] or 0):.1%} R={(stats['recall'] or 0):.1%}; TP {stats['tp']} FP {stats['fp']} FN {stats['fn']}")
            detail_links.append(f'<a href="{case}_{ident}.png">{ident}单图</a>')
            lines.append(f"| {case} | {ident} | {len(sample['ground_truth'])} | {stats['tp']} | {stats['fp']} | {stats['fn']} | {(stats['precision'] or 0):.1%} | {(stats['recall'] or 0):.1%} |")
        w,h=max(p.width for p in panels),max(p.height for p in panels)
        sheet=Image.new('RGB',(3*w,h+45),'white')
        draw=ImageDraw.Draw(sheet)
        for i,(panel,caption) in enumerate(zip(panels,captions)):
            sheet.paste(panel,(i*w,45))
            draw.text((i*w+6,9),caption,fill='black')
        sheet.save(out/f'{case}_comparison.png')
        pages.append(f'<h2>{case} · {sample["width"]}×{sample["height"]}</h2><p>'+ ' · '.join(detail_links)+f'</p><a href="{case}_comparison.png"><img loading="lazy" src="{case}_comparison.png"></a>')
    lines += ['','## 小样本不确定性','',json.dumps(summary['paired_comparison'],ensure_ascii=False,indent=2),
              '','标注执行配置由主智能体spawn记录确定；没有可归属到每个智能体的精确tokens或费用，不猜测用量。10张图不能代表整个数据集，人工参考可能有错误或边界口径差异。结果不会自动用于训练。','',
              '原始标注保存在annotations/，文件哈希写入summary.json；人工参考与中性任务分别保存在reference.json和annotation_task.json。蓝=人工参考，绿=正确匹配预测，橙=未匹配预测，紫=漏检人工框。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    table='<table><tr><th>模型</th><th>P@0.5</th><th>R@0.5</th><th>F1</th></tr>'
    for ident,result in {**summary['models'],**qwen['models']}.items():
        stats=result['metrics']['0.5']
        table+=f"<tr><td>{html.escape(ident)}</td><td>{(stats['precision'] or 0):.2%}</td><td>{(stats['recall'] or 0):.2%}</td><td>{stats['f1']:.2%}</td></tr>"
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Sol fullpage benchmark</title><style>body{font:16px system-ui;margin:24px}img{width:100%}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:9px}h2{margin-top:40px}</style><h1>5.6-Sol中等 vs 6-Sol轻度：10张整图标注</h1><p>每张图：左=人工参考，中=5.6-Sol中等，右=6-Sol轻度；蓝=人工框，绿=匹配预测，橙=未匹配预测，紫=漏检人工框。点击放大。</p><p>相同10张图、825个人工框。无YOLO/人工标签提示，仅完整原图。Qwen为此前API结果，执行方式不同，四模型表仅供参考。</p>'+table+'</table><p><a href="REPORT.md">完整报告</a> · <a href="summary.json">统计JSON</a></p>'+''.join(pages),encoding='utf-8')
    print(json.dumps({'models':summary['models'],'paired_comparison':summary['paired_comparison']},ensure_ascii=False),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=base.LAB/'outputs/sol_fullpage_10_20261016')
    args=parser.parse_args()
    report(args.output.resolve())

if __name__=='__main__':
    main()
