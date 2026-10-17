"""Train-derived public bbox-template diagnostic, never a visual extent claim."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from component_lab.eval.metric import score_case,ScoreConfig


def learn_templates(train_cases, gt_dir):
    counts=defaultdict(Counter)
    for case in train_cases:
        targets=list(gt_dir.glob(case['case']+'-*.json'))
        if len(targets)!=1:raise ValueError('Need unique training target')
        gt=json.loads(targets[0].read_text('utf-8'))
        for item in gt['components'].values():
            if item['type'] not in {'r','c'}:continue
            x1,y1,x2,y2=item['bbox'];pair=tuple(sorted([round(x2-x1,6),round(y2-y1,6)]))
            counts[item['type']][pair]+=1
    learned={}
    for kind, histogram in counts.items():
        pair,count=histogram.most_common(1)[0]
        if sum(histogram.values())>=20 and count==sum(histogram.values()):
            learned[kind]={'short':pair[0],'long':pair[1],'support':count,'consistency':1.0}
    return learned


def adapt_prediction(prediction,templates):
    out=deepcopy(prediction);changes=[]
    for key,row in out['components'].items():
        template=templates.get(row['type'])
        if not template:continue
        x1,y1,x2,y2=row['bbox'];cx,cy=(x1+x2)/2,(y1+y2)/2
        w,h=(template['long'],template['short']) if x2-x1>y2-y1 else (template['short'],template['long'])
        before=list(row['bbox']);row['bbox']=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
        changes.append({'key':key,'body_prediction_bbox':before,'export_protocol_bbox':row['bbox'],'orientation_source':'predicted_bbox_aspect'})
    return out,changes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh immutable output')
    split_path=ROOT/'configs/contest6_group_split_v1.json';split=json.loads(split_path.read_text('utf-8'))
    gt_dir=ROOT/'data/official_contest6/train200'
    templates=learn_templates(split['partitions']['train'],gt_dir)
    args.output.mkdir(parents=True)
    (args.output/'TRAIN_TEMPLATES.json').write_text(json.dumps(templates,indent=2),'utf-8')
    # No DEV targets enter conversion; freeze all predictions first.
    frozen=[]
    for case in split['partitions']['dev']:
        ident=case['case'];source=args.predictions/(ident+'.json');baseline=json.loads(source.read_text('utf-8'))
        prediction,changes=adapt_prediction(baseline,templates)
        target=args.output/'predictions'/(ident+'.json');target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(prediction,ensure_ascii=False,indent=2),'utf-8')
        (args.output/(ident+'_trace.json')).write_text(json.dumps(changes,ensure_ascii=False,indent=2),'utf-8')
        frozen.append((ident,baseline,prediction))
    results=[]
    for ident,baseline,prediction in frozen:
        gt=json.loads(next(gt_dir.glob(ident+'-*.json')).read_text('utf-8'))
        for label,payload in [('unadapted',baseline),('public_template',prediction)]:
            results.append({'case':ident,'variant':label,**score_case(payload,gt,ScoreConfig())})
    summaries=[]
    for label in ['unadapted','public_template']:
        rows=[r for r in results if r['variant']==label];counts=Counter();diag=Counter()
        for r in rows:counts.update(r['counts']['component_f1']);diag.update(r['diagnostics']['components'])
        summaries.append({'variant':label,'macro_component_f1':sum(r['component_f1'] for r in rows)/len(rows),'counts':dict(counts),'diagnostics':dict(diag)})
    report={'kind':'public_annotation_protocol_diagnostic_NOT_symbol_extent_estimation','templates':templates,
            'template_source':'train140 GT only; require >=20 support and100% identical short/long pair',
            'input_predictions':str(args.predictions.resolve()),'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
            'used_dev_targets_for_conversion':False,'paid_api_calls':0,'deep_model_training':0,
            'summaries':summaries,'per_case':results,
            'limitations':['Fits public annotation dimensions, NOT full visible symbol bounds or pin search limits.',
                           'Predicted bbox aspect may not equal physical orientation; no GT direction used.',
                           'Existing checkpoints historically exposed; dev comparison only, no test inference.',
                           'Public bbox-template convention may not apply to hidden test or other rendered DPI.',
                           'Original model boxes, labels and weights unchanged. Not enabled in production export.']}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    lines=['# Public bbox protocol diagnostic','','Train-derived r/c output templates; not true symbol geometry.','', '|Variant|Macro ComponentF1|TP|FP|FN|bbox errors|','|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        c=s['counts'];lines.append(f"|{s['variant']}|{s['macro_component_f1']:.2%}|{c['tp']}|{c['fp']}|{c['fn']}|{s['diagnostics']['bbox_wrong']}|")
    lines+=['','## Limitations','']+['- '+x for x in report['limitations']]
    (args.output/'REPORT.md').write_text('\n'.join(lines),'utf-8')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
