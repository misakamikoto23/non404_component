"""Frozen zero-API DEV ablations; all predictions precede GT evaluation."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from component_lab.contest_components import associate_variants, export_contest
from component_lab.display_names import assign_display_names
from component_lab.group_owner_experiment import ReverseGroupOwner, clear_by_three_locks
from component_lab.eval.metric import ScoreConfig, score_case, bbox_ok, text_ok, canonical_type


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def correct_keys(pred, gt):
    return {key for key,g in gt['components'].items() if key in pred['components'] and
        pred['components'][key]['type']==g['type'] and bbox_ok(pred['components'][key]['bbox'],g['bbox'],ScoreConfig()) and
        text_ok((pred['components'][key]['type'],pred['components'][key]['Name'],pred['components'][key]['value'],pred['components'][key]['bbox']),
                (g['type'],g.get('Name'),g.get('value'),g['bbox']))}


ARMS = {
    'baseline': {},
    'numeric_units': {'units':True},
    'reverse_anchor': {'reverse':True},
    'units_reverse': {'units':True,'reverse':True},
    'guard_only': {'guard':True},
    'units_reverse_guard': {'units':True,'reverse':True,'guard':True},
    'full_margin015': {'units':True,'reverse':True,'guard':True,'margin':.15},
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/group_owner_coupling_20261017')
    args=parser.parse_args();output=args.output
    if (output/'frozen_manifest.json').exists():raise FileExistsError('Choose fresh output; frozen runs are immutable')
    jobs=read(ROOT/'outputs/unattended_20261017_lunch/span_inputs/manifest.json')['jobs']
    expected={c['case'] for c in read(ROOT/'configs/contest6_group_split_v1.json')['partitions']['dev']}
    assert {j['id'] for j in jobs}==expected and len(jobs)==20
    save(output/'CONFIG.json',{'arms':ARMS,'api_calls':0,'training_runs':0,'parameters_fitted_on_dev':False,
                              'reverse_reward':-.2,'reverse_margin_text_heights':1.0,
                              'requires_independent_parameter':True,'guard_margin_pixels':8,
                              'guard_compact_gap_heights':1.5,'guard_is_destructive_shadow_only':True})
    all_predictions,all_traces,stats,metadata={},{},{},[]
    probe_pools={'null':[],'nonnull':[]}
    for job in sorted(jobs,key=lambda j:j['id']):
        case=job['id'];components=read(Path(job['detections']))['components'];texts=read(Path(job['ocr']))['texts']
        with Image.open(job['image']) as image:height=image.height
        all_predictions[case]={};all_traces[case]={};stats[case]={}
        for name,config in ARMS.items():
            adjuster=ReverseGroupOwner() if config.get('reverse') else None
            variants,edits=associate_variants(components,texts,numeric_unit_refs=config.get('units',False),
                reference_cost_adjuster=adjuster,ambiguity_gap=config.get('margin',.10))
            association=assign_display_names(variants['identity_v1'])
            pred,traces=export_contest(association,height,ic_value_as_name=True,display_names=True)
            decisions=[]
            if config.get('guard'):pred,decisions=clear_by_three_locks(association,pred,traces)
            if name=='baseline':
                reference=read(ROOT/'outputs/advance_until_bottleneck_20261017/lexical_experiment/identity_v1/predictions'/f'{case}.json')
                assert pred==reference,f'Baseline parity failed: {case}'
                by_source={c['id']:c for c in association['components']}
                for trace in traces:
                    c=by_source[trace['source_id']];record=pred['components'][trace['key']]
                    if trace['key'].startswith('__unresolved') or record['type'] not in {'r','c','box'}:continue
                    refcost=c.get('association_evidence',{}).get('reference',{}).get('cost',999)
                    if c.get('detection',{}).get('score',0)<.5 or refcost>.5:continue
                    field='Name' if record['type']=='box' else 'value';kind='null' if record[field] is None else 'nonnull'
                    identifier=f"{case}/{c['id']}/probe_v1"
                    probe_pools[kind].append({'case':case,'source_id':c['id'],'key':trace['key'],'field':field,
                        'pred_text':record[field],'bbox':c['bbox'],'probe_hash':hashlib.sha256(identifier.encode()).hexdigest()})
            all_predictions[case][name]=pred;all_traces[case][name]=traces
            stats[case][name]={'reverse_counts':dict(adjuster.counts) if adjuster else {},
                               'reverse_events':adjuster.events if adjuster else [],'guard':decisions}
            save(output/name/'predictions'/f'{case}.json',pred)
            save(output/name/'traces'/f'{case}.json',{'components':traces,'edits':edits,**stats[case][name]})
        metadata.append({'case':case,'image':job['image'],'image_sha256':hashlib.sha256(Path(job['image']).read_bytes()).hexdigest(),
                         'detections_sha256':hashlib.sha256(Path(job['detections']).read_bytes()).hexdigest(),
                         'ocr_sha256':hashlib.sha256(Path(job['ocr']).read_bytes()).hexdigest()})
    probes=[]
    for kind,rows in probe_pools.items():
        probes += [dict(r,prediction_stratum=kind) for r in sorted(rows,key=lambda r:r['probe_hash'])[:15]]
    save(output/'safety_probes.json',{'selection_uses_gt':False,'pools':{k:len(v) for k,v in probe_pools.items()},'probes':probes})
    save(output/'frozen_manifest.json',{'all_predictions_frozen_before_gt':True,'arms':list(ARMS),'metadata':metadata,
                                       'baseline_parity_cases':20,'api_calls':0})
    # Stage 2: targets are only opened after every variant and safety list is frozen.
    scores,correct,targets,original_null=[],{},{},[]
    source_for_case={j['id']:j['image'] for j in jobs}
    for case in sorted(expected):
        gt=read(next((ROOT/'data/official_contest6/train200').glob(f'{case}-*.json')));targets[case]=gt
        correct[case]={}
        for name,pred in all_predictions[case].items():
            result=score_case(pred,gt,ScoreConfig());correct[case][name]=correct_keys(pred,gt)
            scores.append({'case':case,'variant':name,'f1':result['component_f1'],'counts':result['counts']['component_f1']})
        for key in correct[case]['baseline']:
            g=gt['components'][key];kind=canonical_type(g['type'])
            if kind not in {'resistor','capacitor','ic'}:continue
            field='Name' if kind=='ic' else 'value'
            if g.get(field) is None:original_null.append(f'{case}/{key}')
    coverage=Counter()
    for probe in probes:
        case,key,field=probe['case'],probe['key'],probe['field']
        expected_text=targets[case]['components'].get(key,{}).get(field)
        full_correct=key in correct[case]['baseline']
        probe.update(evaluation_only={'baseline_complete_correct':full_correct,'gt_text':expected_text,
                                     'key_in_gt':key in targets[case]['components']})
        coverage[f"{probe['prediction_stratum']}_{'correct' if full_correct else 'incorrect'}"]+=1
    save(output/'safety_probe_evaluation.json',{'coverage':dict(coverage),'probes':probes,
                                             'warning':'GT only added after frozen selection; deployment must use safety_probes.json'})
    summaries={}
    for name in ARMS:
        rows=[r for r in scores if r['variant']==name];counts=Counter()
        for row in rows:counts.update(row['counts'])
        gained=[f'{case}/{key}' for case in correct for key in sorted(correct[case][name]-correct[case]['baseline'])]
        lost=[f'{case}/{key}' for case in correct for key in sorted(correct[case]['baseline']-correct[case][name])]
        null_lost=[key for key in lost if key in original_null]
        reverse=Counter();unique=set();clears=0
        for case in stats:
            reverse.update(stats[case][name]['reverse_counts'])
            unique.update((case,e['reference_text_id']) for e in stats[case][name]['reverse_events'])
            clears+=sum(d['cleared'] for d in stats[case][name]['guard'])
        summaries[name]={'macro_component_f1':sum(r['f1'] for r in rows)/20,'counts':dict(counts),
                        'gained_tp':gained,'lost_tp':lost,'net_tp':len(gained)-len(lost),'lost_null_tp':null_lost,
                        'fp_net_delta':counts['fp']-sum(r['counts']['fp'] for r in scores if r['variant']=='baseline'),
                        'reverse_function_counts_across_two_passes':dict(reverse),'unique_adjusted_references':len(unique),
                        'cleared_required_fields':clears}
    # GT-selected null visual diagnostic is never an inference or API manifest.
    audit=read(ROOT/'outputs/component_text_only_audit_20261017/audit.json')
    visual=[dict(r,category='correct_null') for r in audit['correct_null_controls']]
    visual += [dict(r,category='pred_nonnull_gt_null') for r in audit['rows'] if r['category']=='B_pred_nonnull_gt_null']
    visual=sorted(visual,key=lambda r:(r['category'],r['case'],r['key']))
    folder=output/'null_visual_eval_only';folder.mkdir(exist_ok=True)
    sheets=[]
    for start in range(0,len(visual),12):
        sheet=Image.new('RGB',(1600,960),'white');draw=ImageDraw.Draw(sheet)
        for local,row in enumerate(visual[start:start+12]):
            case,key=row['case'],row['key'];pred=all_predictions[case]['baseline']['components'][key]
            with Image.open(source_for_case[case]) as im:
                x1,y1,x2,y2=pred['bbox'];box=(max(0,int(x1)-65),max(0,int(im.height-y2)-65),
                    min(im.width,int(x2)+65),min(im.height,int(im.height-y1)+65))
                crop=im.convert('RGB').crop(box);crop.save(folder/f'{case}_{key}.png')
                view=ImageOps.contain(crop,(390,280),Image.Resampling.LANCZOS)
            x,y=local%4*400,local//4*320
            draw.text((x+8,y+6),f"{case}/{key} / {row['category']}",fill='black')
            sheet.paste(view,(x+(400-view.width)//2,y+32+(280-view.height)//2))
        filename=f'sheet_{start//12+1}.png';sheet.save(folder/filename);sheets.append(filename)
    report={'kind':'zero_API_group_owner_controlled_DEV_ablation','summaries':summaries,'per_case':scores,
            'safety_probe_coverage':dict(coverage),'original_correct_null_count':len(original_null),
            'all_predictions_frozen_before_gt':True,'api_calls':0,'training_runs':0,'sealed_test_used':False,
            'production_enabled':False,'labels_modified':False,'parameters_fitted_on_dev':False,
            'null_visual_diagnostic_cases':len(visual),'source_sha256':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                for path in ['src/component_lab/text_roles.py','src/component_lab/text_matching.py',
                             'src/component_lab/contest_components.py','src/component_lab/group_owner_experiment.py']}}
    save(output/'report.json',report)
    lines=['# Component零API关联消融 — 2026-10-17','','|版本|严格dev20宏F1|TP|新增TP|损失旧TP|损失正确null|FP净变化|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():lines.append(f"|{name}|{s['macro_component_f1']:.4%}|{s['counts']['tp']}|{len(s['gained_tp'])}|{len(s['lost_tp'])}|{len(s['lost_null_tp'])}|{s['fp_net_delta']:+d}|")
    lines+=['','20图baseline与旧冻结预测逐字段一致。新选项默认关闭；全部预测与探针先冻结再读GT，零API/训练/test。', '',
            '反向锚是既有紧凑group的保守增量：只在原歧义行使用显式独立参数、归一化最近差≥1h、-0.20。没有新增夹心候选生成，不能据此否定所有双向结构方法。具体配置见PLAN/CONFIG。', '',
            'guard是强制清空输出副本的独立消融，不等于真实null检测。若损失正确非null，保持实验状态，不接生产。净增≤0的代价方案不继续扫参数。','',
            '## 实际覆盖与新增/损失','']
    for name,s in summaries.items():
        lines += [f"- {name}: adjusted_ref={s['unique_adjusted_references']}, cleared={s['cleared_required_fields']}; gained={s['gained_tp']}; lost={s['lost_tp']}."]
    lines += ['',f'## 固定哈希探针\n\n{json.dumps(dict(coverage),ensure_ascii=False)}。这只是冻结清单事后实测覆盖，不是哈希采样天然保证；没有换seed。预测池与无GT清单见safety_probes.json。', '',
              '## 正确null与过度填充局部图\n\nnull_visual_eval_only含11个正确null与12个预测非null/GT null。仅用于事后视觉诊断，不得送进推理或API，也不修改真值。','',
              '## 限制\n\n本地评分实现、旧权重有历史暴露，属于开发证据；不宣称独立泛化。bbox检查通过不证明真实owner。当前主要目标仍是Component，不延伸Pin/Net。', '',
              '## 复现\n\n`python scripts/run_group_owner_experiment.py --output outputs/group_owner_fresh`，输出目录应无frozen_manifest；旧实验不可覆盖。']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    images=''.join(f'<img src="null_visual_eval_only/{s}" style="width:100%">' for s in sheets)
    (output/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Component group coupling</title><style>body{font-family:system-ui;max-width:1250px;margin:30px auto}pre{white-space:pre-wrap}</style><h1>Component零API归属对照</h1><p><a href="REPORT.md">报告</a> · <a href="report.json">逐图指标</a></p><pre>'+html.escape('\n'.join(lines[:13]))+'</pre><h2>GT评价后选取的null诊断图，非推理输入</h2>'+images,encoding='utf-8')
    print(json.dumps({'summaries':summaries,'probe_coverage':dict(coverage)},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
