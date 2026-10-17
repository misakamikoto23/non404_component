"""Offline, post-hoc oracle bounds and visible-value/GT-null evidence.

These GT-assisted diagnostics are NOT model predictions and never feed back
into requests, field selection, production output or training labels.
"""
from pathlib import Path
import sys
import os
import json
import html
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from pilot_fullpage_component_identity import LAB,OUT,read,sha
from component_lab.candidate_verifier import atomic_json
from component_lab.eval.metric import bbox_ok,ScoreConfig
from evaluate_component_architectures import pair_counts


def main():
    frozen=read(OUT/'predictions_frozen.json')
    for path,digest in frozen['files'].items():assert sha(OUT/path)==digest
    manifest=read(OUT/'manifest.json');rows=[]
    for item in manifest['cases']:
        case=item['case'];p=read(OUT/'baseline/predictions'/(case+'.json'))
        gt=read(next((LAB/'data/official_contest6/train200').glob(case+'-*.json')))['components']
        pred=list(p['components'].values());named=[v for k,v in gt.items() if not k.startswith('∅')]
        rows.append({'case':case,'pred':len(pred),'gt':len(gt),'named_gt':len(named),
            'fixed_key_text_oracle_tp':sum(k in p['components'] and p['components'][k]['type']==v['type'] and
                bbox_ok(p['components'][k]['bbox'],v['bbox'],ScoreConfig()) for k,v in gt.items()),
            'named_identity_text_oracle_tp':pair_counts(pred,named,'maxcoord20',True)['tp'],
            'all_id_including_hidden_oracle_tp':pair_counts(pred,list(gt.values()),'maxcoord20',True)['tp']})
    summaries={}
    for field in ['fixed_key_text_oracle_tp','named_identity_text_oracle_tp','all_id_including_hidden_oracle_tp']:
        tp=sum(r[field] for r in rows)
        summaries[field]={'tp':tp,'macro_f1':float(np.mean([2*r[field]/(r['pred']+r['gt']) for r in rows])),
                          'micro_f1':2*tp/sum(r['pred']+r['gt'] for r in rows)}
    atomic_json(OUT/'oracle_diagnostics.json',{'kind':'posthoc_eval_only_component_upper_bounds_NOT_model_results',
        'warning':'GT IDs/fields assumed known. Named keys can still be visually inaccessible or inconsistent; these are structural bounds, not achievable accuracy claims. All632pred+931GT retained in denominators; never used for tuning.',
        'summaries':summaries,'per_case':rows})
    case='0038';targets=read(OUT/'inputs'/case/'targets.json')
    baseline=read(OUT/'baseline/predictions'/(case+'.json'))['components']
    prediction=read(OUT/'full_replace/predictions'/(case+'.json'))['components']
    gt_file=next((LAB/'data/official_contest6/train200').glob(case+'-*.json'));gt=read(gt_file)['components']
    ledger=read(OUT/'usage_ledger.json');response=next(r for r in ledger['requests'] if r['case']==case)
    choices={r['id']:r for r in response['choices']};source=next(r for r in manifest['cases'] if r['case']==case)
    examples=[]
    for target in targets:
        key=target['baseline_key']
        if key not in ('C21','C22'):continue
        value=choices[target['id']]['fields']['value']
        examples.append({'key':key,'gt_value':gt[key]['value'],'baseline_value':baseline[key]['value'],
            'qwen_value':prediction[key]['value'],'field_evidence':value,'bbox_unchanged':prediction[key]['bbox']==baseline[key]['bbox']})
    assert len(examples)==2 and all(e['gt_value'] is None and e['qwen_value']=='12pF/50V' for e in examples)
    evidence={'case':case,'source_image':source['image'],'source_image_sha256':source['image_sha256'],
        'gt_file_sha256':sha(gt_file),'response_file_sha256':sha(response['response_file']),
        'human_visual_audit':'Source PNG inspected: 12pF/50V is visible next to C21 and C22. This confirms visible text, not an organizer ruling on value semantics.',
        'examples':examples,'gt_modified':False,'evaluation_policy_modified':False}
    atomic_json(OUT/'value_conflict_evidence.json',evidence)
    followup='''# Component评分协议补充问题（2026-10-18）

公开用例0038的C21/C22旁存在清楚的文本 `12pF/50V`；冻结OCR分别为o0023/o0035，Qwen整图归属也选择相同原文。但对应GT组件的value均为null。预测bbox/type未改，原value=null时通过本地严格检查，填入图面可见数值后反而丢失两个TP。

请澄清：正式测试是否确保图面主参数与GT value一致？对于容量/额定电压组合串 `12pF/50V`，应输出完整原文、仅容量、还是null？当图面文字和源文件value属性不同，评分以哪一个为准？

我们未修改GT、评分器或根据此案例写例外。证据与哈希保存在value_conflict_evidence.json，原图和对照见value_conflict.html。本文件仅为提问草稿，未向任何人发送。
'''
    (OUT/'ORGANIZER_FOLLOWUP.md').write_text(followup,encoding='utf-8')
    src=Path(os.path.relpath(source['image'],OUT)).as_posix()
    table='<table><tr><th>元件</th><th>GT value</th><th>原预测</th><th>Qwen归属文字</th><th>OCR ID</th></tr>'
    for e in examples:
        table+=f'<tr><td>{e["key"]}</td><td>null</td><td>null</td><td>{html.escape(e["qwen_value"])}</td><td>{e["field_evidence"]["ocr_id"]}</td></tr>'
    table+='</table>'
    page='<!doctype html><meta charset="utf-8"><title>Visible value vs GT null</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}img{width:100%}td,th{padding:12px;border:1px solid #ccc}table{border-collapse:collapse}</style><a href="gallery.html">返回实验</a><h1>0038：可见数值与GT null</h1><p>图面确认C21/C22旁有12pF/50V。对“value字段应如何填写”尚无主办方澄清；不把此证据自动改成新真值。</p><img src="'+html.escape(src)+'">'+table+'<p>严格逐图F1：27.27% → 9.09%；固定框/type/数量，损失2TP。该诊断不修改任何正式评分。</p><a href="ORGANIZER_FOLLOWUP.md">主办方问题草稿</a><a href="value_conflict_evidence.json">完整证据及哈希</a>'
    (OUT/'value_conflict.html').write_text(page,encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
