"""One frozen five-DEV color transition pin-location oracle experiment."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from component_lab.pin_transitions import infer_color_transitions,TransitionConfig
from scripts.probe_pin_geometry_oracle import evaluate,prf,overlay


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/advance_until_bottleneck_20261017/pin_color_v1')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh immutable output')
    split=ROOT/'configs/contest6_group_split_v1.json'
    cases=json.loads(split.read_text('utf-8'))['partitions']['dev'][:5]
    cfg=TransitionConfig();frozen=[]
    for case in cases:
        ident=case['case'];source=ROOT/'outputs/component_identity_round_001_20261017/inputs'/ident/'source.png'
        image=cv2.imdecode(np.frombuffer(source.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('image unavailable')
        target=ROOT/'data/official_contest6/train200'/(case['stem']+'.json')
        components=json.loads(target.read_text('utf-8'))['components'];h=image.shape[0]
        boxes=[];owners={}
        for i,(key,row) in enumerate(components.items()):
            x1,y1,x2,y2=row['bbox'];owner=f'owner_{i:04d}'
            boxes.append({'id':owner,'bbox':[x1,h-y2,x2,h-y1]});owners[owner]=key
        started=time.perf_counter();result=infer_color_transitions(image,boxes,cfg);elapsed=time.perf_counter()-started
        save(args.output/ident/'candidates.json',{**result,'oracle_inputs':['component_bboxes'],
             'pin_number_or_name_inference':False,'seconds':elapsed,'image_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
        frozen.append((ident,image,target,components,boxes,owners,result,elapsed))
        print(f'{ident}: {len(result["candidates"])} color candidates frozen',flush=True)
    # Only after every inference output is frozen are pin labels evaluated.
    rows=[];reasons=Counter()
    for ident,image,target,components,boxes,owners,result,elapsed in frozen:
        pins=json.loads(target.read_text('utf-8'))['pins']
        evaluation=evaluate(result['candidates'],components,pins,owners,5.)
        save(args.output/ident/'evaluation.json',evaluation)
        rendered=overlay(image,boxes,result['candidates'],evaluation,pins,owners)
        ok,encoded=cv2.imencode('.png',rendered)
        if not ok:raise ValueError('overlay unavailable')
        (args.output/ident/'overlay.png').write_bytes(encoded.tobytes())
        count=evaluation['counts'];reasons.update(result['reason_counts'])
        resolved=len({x['owner_id'] for x in result['candidates']})
        rows.append({'case':ident,**count,'candidates':len(result['candidates']),'seconds':elapsed,
                     'resolved_components':resolved,'components':len(boxes),'unresolved_reasons':result['reason_counts']})
    totals={k:sum(x[k] for x in rows) for k in ['tp','fp','fn']}
    report={'kind':'color_transition_GT_bbox_oracle_pin_location_NOT_official_PinF1','micro':prf(**totals),
            'macro':{k:float(np.mean([x[k] for x in rows])) for k in ['precision','recall','f1']},'per_case':rows,
            'reason_counts':dict(reasons),'paid_api_calls':0,'training_runs':0,'gt_pins_used_for_inference':False,
            'config':vars(cfg),'split_sha256':hashlib.sha256(split.read_bytes()).hexdigest(),
            'implementation_sha256':hashlib.sha256((ROOT/'src/component_lab/pin_transitions.py').read_bytes()).hexdigest(),
            'limitations':['Five fixed DEV cases, not independent generalization.','GT boxes may be template sizes, not true body boundaries.',
                           'No observed color boundary means unresolved; monochrome cases remain in denominator.',
                           'Color candidates are not pin identities or topology. No official PinF1 claimed.',
                           'No threshold search, no pin GT used for seed/trace/color inference.']}
    save(args.output/'report.json',report)
    lines=['# Color transition pin-tip shadow experiment','','Same first5 DEV GT-box oracle as fixed3px baseline. Not official PinF1.','',
           '|Case|Candidates|TP|Precision|Recall|F1|Resolved components|Seconds|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['case']}|{r['candidates']}|{r['tp']}|{r['precision']:.2%}|{r['recall']:.2%}|{r['f1']:.2%}|{r['resolved_components']}/{r['components']}|{r['seconds']:.3f}|")
    lines+=['',f"Micro: {json.dumps(report['micro'])}",f"Macro: {json.dumps(report['macro'])}",'','## Unresolved',json.dumps(dict(reasons),indent=2),'','## Limitations']+['- '+x for x in report['limitations']]
    (args.output/'REPORT.md').write_text('\n'.join(lines),'utf-8')
    links=''.join(f'<h2>{r["case"]}</h2><img src="{r["case"]}/overlay.png" style="width:100%">' for r in rows)
    (args.output/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Pin color transition oracle</title><h1>Pin location shadow experiment</h1><p>Green matched; red unmatched candidates; blue missedGT. GT only for display/evaluation. No officialPinF1.</p>'+links,'utf-8')
    print(json.dumps({'micro':report['micro'],'macro':report['macro'],'reasons':dict(reasons)},indent=2))


if __name__=='__main__':main()
