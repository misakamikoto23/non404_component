"""One cache-only shadow: retain round004 named records; recover unresolved identities."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB/'src'))
from component_lab.review_workflow import atomic_json
from component_lab.eval.metric import ScoreConfig, score_case
from native_reviewed_alignment_20261017 import match_boxes, association_gate

SOURCE=LAB/'outputs/active_learning/round_005/component_comparison'
OUT=LAB/'outputs/active_learning/round_005/safe_fusion_shadow'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def frozen(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def fuse(base,new):
    pred=copy.deepcopy(base)
    named={k:v for k,v in base['components'].items() if not k.startswith('__unresolved')}
    unresolved=[k for k in base['components'] if k.startswith('__unresolved')]
    fresh=[k for k,v in new['components'].items() if k not in base['components']
           and not k.startswith(('__unresolved','∅')) and v['type']!='component'
           and re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*\d+[A-Za-z0-9_.-]*',k)]
    matrix,pairs=match_boxes([base['components'][k]['bbox'] for k in unresolved],
                            [new['components'][k]['bbox'] for k in fresh])
    changes=[]
    for i,j in pairs:
        if not association_gate(matrix,i,j):
            continue
        old_key,new_key=unresolved[i],fresh[j]
        before=base['components'][old_key]
        proposed=new['components'][new_key]
        if before['type'] not in ('component',proposed['type']):
            continue
        replacement=copy.deepcopy(before)
        replacement['type']=proposed['type']
        del pred['components'][old_key]
        pred['components'][new_key]=replacement
        changes.append(dict(from_key=old_key,to_key=new_key,iou=float(matrix[i,j]),
                            before=before,after=replacement))
    assert len(pred['components'])==len(base['components'])
    assert all(pred['components'][k]==v for k,v in named.items())
    assert pred['pins']==base['pins'] and pred['nets']==base['nets']
    return pred,dict(changes=changes,old_named_count=len(named),old_named_hash=frozen(named),
                     output_named_hash=frozen({k:pred['components'][k] for k in named}),
                     old_named_unchanged=True,existing_bbox_and_fields_preserved=True)


def main():
    assert not OUT.exists(),'No repeated experiment'
    OUT.mkdir(parents=True)
    atomic_json(OUT/'PLAN.json',dict(
        rule='Round004 named and all fields frozen. New round005 identities may replace only old unresolved records; key/type only, original bbox/Name/value retained. Same count. No reserved-key overwrite.',
        gates=dict(iou=.75,row_margin=.15,column_margin=.15,type='same or old generic'),
        caveat='One adaptive development shadow after the paired detector diagnostic; no independent generalization claim.',
        api_calls=0,training_runs=0,test_used=False))
    cases=sorted(r['case'] for r in read(LAB/'configs/contest6_group_split_v1.json')['partitions']['dev'])
    audit={}
    # Freeze all outputs before reading GT labels.
    for case in cases:
        before=read(SOURCE/'round004/scored/identity_v1/predictions'/f'{case}.json')
        new=read(SOURCE/'round005/scored/identity_v1/predictions'/f'{case}.json')
        pred,log=fuse(before,new)
        atomic_json(OUT/'predictions'/f'{case}.json',pred)
        audit[case]=log
    atomic_json(OUT/'writeback_audit.json',audit)
    scored=[]
    for case in cases:
        gt_paths=list((LAB/'data/official_contest6/train200').glob(case+'-*.json'))
        assert len(gt_paths)==1
        gt=read(gt_paths[0])
        before=read(SOURCE/'round004/scored/identity_v1/predictions'/f'{case}.json')
        after=read(OUT/'predictions'/f'{case}.json')
        scored.append(dict(case=case,before=score_case(before,gt,ScoreConfig()),after=score_case(after,gt,ScoreConfig())))
    result=dict(before_macro_f1=float(np.mean([r['before']['component_f1'] for r in scored])),
                after_macro_f1=float(np.mean([r['after']['component_f1'] for r in scored])),
                before_tp=sum(r['before']['counts']['component_f1']['tp'] for r in scored),
                after_tp=sum(r['after']['counts']['component_f1']['tp'] for r in scored),
                identity_replacements=sum(len(v['changes']) for v in audit.values()),
                old_named_unchanged=all(v['old_named_hash']==v['output_named_hash'] for v in audit.values()),
                prediction_count_unchanged=True,api_calls=0,training_runs=0,test_used=False,default_replaced=False,
                limitation='Cache-only two-detector development shadow, not default CLI result or clean independent test.',
                per_case=scored)
    atomic_json(OUT/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_case'}),flush=True)


if __name__=='__main__':
    main()
