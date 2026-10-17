"""Evaluate a shadow pilot locally; never turn machine decisions into training truth."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB/'scripts'))
from analyze_yolo_errors import load_split, match_detections


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def counts(tp, fp, fn):
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    pilot = LAB/'outputs/automation_round_001/qwen_pilot'
    p.add_argument('--verifier-output', type=Path, default=pilot/'verifier')
    p.add_argument('--reference', type=Path, default=pilot/'evaluation_reference.json')
    p.add_argument('--source-cache', type=Path, default=LAB/'outputs/active_learning/round_003/candidate_val/predictions.json')
    p.add_argument('--output', type=Path, default=pilot/'report')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = read(args.reference)
    if reference.get('reference_labels_sent_to_model') is not False:
        raise ValueError('Expected a blind pilot reference')
    prepared = read(args.verifier_output/'prepared_manifest.json')
    summary = read(args.verifier_output/'summary.json')
    by_id = {r['id']:r for r in summary.get('results',[])}
    if len(by_id) != len(summary.get('results',[])):
        raise ValueError('Duplicate API result ID')
    items, cohort_counts, cards = [], {}, []
    for item in prepared:
        ident = item['candidate']['id']
        ref = reference['items'][ident]
        result = by_id.get(ident,{}).get('result',{})
        decision = result.get('decision','not_processed')
        keep = decision != 'reject'  # Uncertain, failure and unprocessed retain the baseline proposal.
        row = {'id':ident,'cohort':ref['cohort'],'reference_kind':ref['reference_kind'],
               'reference_keep':ref['reference_keep'],'qwen_decision':decision,
               'effective_keep_in_shadow_simulation':keep,'reason':result.get('reason'),
               'type':result.get('type'),'shadow_only':True}
        items.append(row)
        stats = cohort_counts.setdefault(ref['cohort'], Counter())
        stats['selected'] += 1
        stats['api_processed'] += decision != 'not_processed'
        stats['uncertain'] += decision == 'uncertain'
        stats['unprocessed_retained'] += decision == 'not_processed'
        stats['reference_positive'] += ref['reference_keep']
        stats['reference_negative'] += not ref['reference_keep']
        stats['positive_retained'] += ref['reference_keep'] and keep
        stats['positive_rejected'] += ref['reference_keep'] and not keep
        stats['negative_rejected'] += not ref['reference_keep'] and not keep
        stats['negative_retained'] += not ref['reference_keep'] and keep
        escape = html.escape
        cards.append('<article><h3>'+escape(ident)+'</h3><p>Qwen: '+escape(decision)+
                     ' / '+escape(str(result.get('reason','')))+
                     '；本地参考：'+('保留' if ref['reference_keep'] else '删除／未匹配')+
                     '</p><div class="views"><img src="'+Path(item['tight_path']).as_uri()+
                     '"><img src="'+Path(item['context_path']).as_uri()+'"></div></article>')
    source = read(args.source_cache)
    _, records = load_split(Path(source['data_root']), 'val')
    val_rejects = Counter()
    for row in items:
        if row['cohort']=='val_stratified_diagnostic' and not row['effective_keep_in_shadow_simulation']:
            _, image_id, index = row['id'].split('_')
            val_rejects[(image_id,int(index))] += 1
    before_totals, after_totals, predictions = Counter(), Counter(), {}
    for record in records:
        image_id = Path(record['image']).stem
        original = source['predictions'][record['image']]
        kept = [pred for index,pred in enumerate(original) if pred['score']>=.03 and not val_rejects[(image_id,index)]]
        predictions[record['image']] = kept
        for total, rows in ((before_totals,original),(after_totals,kept)):
            matched = match_detections(record['ground_truth'],rows,.5,.03)
            total['tp'] += len(matched['matches'])
            total['fp'] += len(matched['false_positive_indices'])
            total['fn'] += len(matched['false_negative_indices'])
    before = counts(**before_totals)
    after = counts(**after_totals)
    ledger_path = args.verifier_output/'usage_ledger.json'
    calls = list(read(ledger_path)['requests'].values()) if ledger_path.exists() else []
    usage = {key:sum(c.get('usage',{}).get(key,0) for c in calls)
             for key in ('prompt_tokens','completion_tokens','total_tokens')}
    report = {'shadow_only':True,'not_an_independent_dual95_claim':True,
              'selected_candidates':len(items),'api_processed':len(by_id),
              'cohort_counts':{k:dict(v) for k,v in cohort_counts.items()},
              'uncertain_policy':'retain_baseline_and_flag', 'provider_usage':usage,
              'accounted_tokens':summary.get('accounted_tokens',0),
              'stop_reason':summary.get('stop_reason'),
              'partial_val_simulation':{'images':30,'gt':1044,'candidate_confidence':.03,'match_iou':.5,
                  'note':'Only the selected 16 val candidates were reviewed. Others remain unchanged. Original GT is a geometry proxy.',
                  'before':before,'after':after,'selected_val_rejections':sum(val_rejects.values())},
              'machine_training_labels_created':False,'items':items}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    prediction_cache = {**source,'mode':'partial_qwen_shadow_simulation_not_deployed','confidence':.03,
                        'predictions':predictions,'shadow_only':True}
    (output/'val_shadow_partial_predictions.json').write_text(json.dumps(prediction_cache,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Qwen候选复核旁路试验</title>'
        '<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:24px auto;background:#f6f7f9}'
        'article{background:white;border:1px solid #ddd;padding:16px;margin:16px 0}.views{display:flex;gap:20px;align-items:flex-start}'
        'img{max-width:65%;max-height:500px;object-fit:contain;image-rendering:auto}img:first-child{max-width:28%}</style>'
        '<h1>Qwen候选有效性旁路试验</h1><p>固定选择的诊断样例，不是全量准确率。原图和人工记录未修改。'
        '不确定或未处理的候选在模拟中保留；部分验证参考仍有几何口径争议。</p>'+''.join(cards),encoding='utf-8')
    recovered_count = sum(r.get('source_experiment')=='offline_recovered' for r in by_id.values())
    lines = ['# Qwen候选有效性旁路试验','',
        f'选择{len(items)}个诊断候选，获得{len(by_id)}条独立校验结果（含{recovered_count}条离线恢复）；累计{len(calls)}次接口请求。已返回usage合计{usage["total_tokens"]} token；未处理和不确定项保留原候选。',
        '', '这些是分层／选择性样例，不能外推全量双95%。人工标签没有发送给模型；API结果未写回训练数据。', '',
        '| 样例来源 | 参考保留 | 参考删除／未匹配 | 拒绝保留参考 | 拒绝删除参考 | 不确定 |',
        '|---|---:|---:|---:|---:|---:|']
    for cohort,s in cohort_counts.items():
        lines.append(f'| {cohort} | {s["reference_positive"]} | {s["reference_negative"]} | {s["positive_rejected"]} | {s["negative_rejected"]} | {s["uncertain"]} |')
    lines.extend(['','在原30张val完整候选缓存上，仅对选中的16个val候选应用拒绝建议、其他候选不变，重新一对一匹配得到：','',
        '| 模拟 | TP | FP | FN | P | R |','|---|---:|---:|---:|---:|---:|'])
    for name,values in [('原始conf0.03',before),('部分Qwen影子模拟',after)]:
        lines.append(f'| {name} | {values["tp"]} | {values["fp"]} | {values["fn"]} | {values["precision"]:.2%} | {values["recall"]:.2%} |')
    lines.extend(['','这不是“全量Qwen复核”的成绩；不能用它估算所有候选都复核后的收益。参考框争议仍需独立审计。',
                  '', '[查看32例对比图集]('+str(output/'gallery.html').replace('\\','/')+')'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('selected_candidates','api_processed','cohort_counts','provider_usage','partial_val_simulation')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
