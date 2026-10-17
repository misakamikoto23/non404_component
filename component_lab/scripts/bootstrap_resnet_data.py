"""Prepare crop supervision and bounded teacher jobs; never edit reviewed labels."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import hashlib
import math
from pathlib import Path
import random
import shutil
import sys
from PIL import Image,ImageDraw,ImageOps
import numpy as np

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB/'src'))
from component_lab.review_workflow import image_path,atomic_json,read_json
from prepare_reviewed_component_crops import integer_crops,iou_matrix,assign_one_to_one

ROOT=LAB/'data/component_crops/resnet_bootstrap_20261016'
OUT=LAB/'outputs/resnet_bootstrap_20261016'
ONTOLOGY=LAB/'outputs/qwen_crop_classification_round_001/ontology.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def write_rows(path,items):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in items),encoding='utf-8')

def prepare():
    if (ROOT/'manifest.jsonl').exists():raise ValueError('Crop manifest already frozen')
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    source=LAB/'data/reviewed_versions/manual_round_003/yolo'
    manifest=read_json(source/'manifest.json')
    original=read_json(LAB/'data/yolo/pcbnet_component_v1/manifest.json')
    raw_by_id={x['case']:x for x in original['samples']}
    ontology=read_json(ONTOLOGY);mapping=ontology['raw_to_family_for_unambiguous']
    cached_rows=rows(LAB/'data/component_crops/review_round_001/manifest.jsonl')
    cached_preds={p['id']:p for part in ('pilot_context','main_context') for p in rows(LAB/'outputs/qwen_crop_classification_round_001'/part/'predictions.jsonl')}
    cache={(r['source_image_sha256'],r['source_box_id'],tuple(r['context_pixel_box_xyxy'])):(r,cached_preds.get(r['crop_id'])) for r in cached_rows}
    records=[];n=0;reused=0
    for sample in manifest['samples']:
        if sample['split'] not in ('train','val'):continue # Test is never read.
        path=image_path(source,sample);source_sha=sha(path)
        raw=raw_by_id[sample['case']]['labels'];labels=sample['labels']
        reviewed=sample.get('annotation_status')=='human_reviewed'
        if reviewed:
            matrix=iou_matrix(labels,raw);pairs=assign_one_to_one(matrix,.65)
        else:
            pairs={i:i for i in range(len(labels))};matrix=None
        with Image.open(path) as im:
            im=im.convert('RGB')
            for i,box in enumerate(labels):
                n+=1;ident=f'c{n:06d}'
                tight,context=integer_crops(box['bbox'],*im.size,.08,2)
                original_label=raw[pairs[i]] if i in pairs else None
                raw_category=original_label['category'] if original_label else None
                family=mapping.get(raw_category)
                record={'crop_id':ident,'image_id':sample['case'],'source_split':sample['split'],
                    'source_image_sha256':source_sha,'source_box_id':box['component_id'],
                    'geometry_human_reviewed':reviewed,'bbox':box['bbox'],'context_pixel_box_xyxy':context,
                    'tight_image_path':f'tight/{ident}.png','context_image_path':f'context/{ident}.png',
                    'raw_category':raw_category,'original_family':family,'original_match_iou':float(matrix[i,pairs[i]]) if reviewed and i in pairs else (1.0 if not reviewed else None),
                    'category':None,'label_origin':None,'quarantine_reason':None}
                for kind,pixels in [('tight',tight),('context',context)]:
                    target=ROOT/record[kind+'_image_path'];target.parent.mkdir(exist_ok=True)
                    im.crop(tuple(pixels)).save(target)
                record['crop_sha256']=sha(ROOT/record['context_image_path'])
                if min(tight[2]-tight[0],tight[3]-tight[1])<4:
                    record['quarantine_reason']='symbol_short_side_under_4px'
                old=cache.get((source_sha,box['component_id'],tuple(context)))
                if old and old[1]:
                    previous,prediction=old
                    if sha(LAB/'data/component_crops/review_round_001'/previous['context_image_path'])!=record['crop_sha256']:
                        raise ValueError('Cached crop bytes differ')
                    record['cached_teacher']=prediction;reused+=1
                    if prediction['category']!='unknown' and not prediction['ambiguous'] and prediction['confidence']>=.9:
                        if family and prediction['category']!=family:
                            record['quarantine_reason']='cached_teacher_disagrees_original'
                        else:
                            record.update(category=prediction['category'],label_origin='cached_qwen_agrees_original' if family else 'cached_qwen_only_silver')
                    else:record['quarantine_reason']='cached_teacher_unknown_ambiguous_or_low_confidence'
                elif family:
                    record.update(category=family,label_origin='original_category_mapping_weak')
                records.append(record)
        print(sample['case'],len(labels),'reviewed' if reviewed else 'original',flush=True)
    # Original val labels remain unchanged even if teacher disagrees later.
    # Prioritize unresolved reviewed geometry and ambiguous original semantics.
    unresolved=[r for r in records if not r['quarantine_reason'] and not r['category']]
    train=[r for r in unresolved if r['source_split']=='train'];val=[r for r in unresolved if r['source_split']=='val']
    rng=random.Random(20261016);rng.shuffle(train);rng.shuffle(val)
    train.sort(key=lambda r:(not r['geometry_human_reviewed'],r['raw_category'] not in ('Box','PIN','Opt','M','CIRCLE_HEADER','Other')))
    selected=[];per_image=Counter();per_raw=Counter()
    for row in train:
        # One page/ambiguous raw bucket cannot consume the teacher budget.
        if len(selected)>=850:break
        if per_image[row['image_id']]>=12 or per_raw[row['raw_category'] or '_unmatched']>=400:continue
        selected.append(row);per_image[row['image_id']]+=1;per_raw[row['raw_category'] or '_unmatched']+=1
    selected.extend(val[:150])
    # Independent audit of known weak labels; no hints in the teacher prompt.
    audit_pool=[r for r in records if r['source_split']=='train' and r['label_origin']=='original_category_mapping_weak' and not r['quarantine_reason']]
    rng.shuffle(audit_pool)
    audit=[];audit_counts=Counter()
    for row in sorted(audit_pool,key=lambda r:sum(x['original_family']==r['original_family'] for x in audit_pool)):
        if len(audit)>=120:break
        if audit_counts[row['original_family']]>=12:continue
        audit.append(row);audit_counts[row['original_family']]+=1
    selected_ids={r['crop_id'] for r in selected}
    jobs=selected+[r for r in audit if r['crop_id'] not in selected_ids]
    # API jobs intentionally contain no raw class/family/GT or source filename.
    neutral=[{k:r[k] for k in ('crop_id','tight_image_path','context_image_path','crop_sha256')} for r in jobs]
    write_rows(ROOT/'manifest.jsonl',records);write_rows(OUT/'teacher_jobs.jsonl',neutral)
    summary={'source':str(source),'source_manifest_sha256':sha(source/'manifest.json'),'crops':len(records),
        'source_images':len({r['image_id'] for r in records}),'reviewed_source_images':len({r['image_id'] for r in records if r['geometry_human_reviewed']}),
        'cached_qwen_predictions_reused':reused,'new_teacher_crops':len(jobs),'teacher_batch_size':12,
        'max_new_paid_requests':110,'max_new_accounted_tokens':400000,'test_images_read':False,
        'label_origin_counts':dict(Counter(r['label_origin'] for r in records)),
        'split_policy':'Preserve original PCBnet train/val by complete source image; original test never used. Human review confirms geometry, not class. All category labels are weak/silver.'}
    atomic_json(OUT/'preparation.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

def assemble():
    records=rows(ROOT/'manifest.jsonl');teacher={r['id']:r for r in rows(OUT/'teacher/predictions.jsonl')}
    confirmations={r['id']:r for r in rows(OUT/'teacher_confirm/predictions.jsonl')}
    agent={r['id']:r for r in read_json(OUT/'visual_audit/agent_predictions.json')['items']}
    overrides={'ANT','Battary','Motor'}
    for r in records:
        if r['raw_category'] in overrides:
            r['deprecated_proxy_family']=r['original_family'];r['original_family']=None
            if r['label_origin']=='original_category_mapping_weak':r.update(category=None,label_origin=None)
            elif r['label_origin']=='cached_qwen_agrees_original':r['label_origin']='cached_qwen_only_silver'
            if r['quarantine_reason']=='cached_teacher_disagrees_original':
                r['quarantine_reason']=None
                p=r.get('cached_teacher')
                if p and p['category']!='unknown' and not p['ambiguous'] and p['confidence']>=.9:
                    r.update(category=p['category'],label_origin='cached_qwen_only_silver')
    for r in records:
        p=teacher.get(r['crop_id'])
        if not p:continue
        r['teacher']=p
        if p['category']=='unknown' or p['ambiguous'] or p['confidence']<.9:
            if r['source_split']=='train':r['quarantine_reason']='new_teacher_unknown_ambiguous_or_low_confidence'
            continue
        if r['original_family'] and r['original_family']!=p['category']:
            if r['source_split']=='train':r['quarantine_reason']='new_teacher_disagrees_original'
            # Keep independent original validation target; do not cherry-pick agreement.
            continue
        if not r['quarantine_reason']:
            r.update(category=p['category'],label_origin='new_qwen_agrees_original' if r['original_family'] else 'new_qwen_only_silver')
    for r in records:
        a=agent.get(r['crop_id']);p=confirmations.get(r['crop_id'])
        if a and (a['ambiguous'] or a['category']!=r['category']):
            r['quarantine_reason']='independent_model_disagreement_or_uncertainty'
        if r['category'] and r['label_origin'] and 'only_silver' in r['label_origin']:
            independent=a and not a['ambiguous'] and a['category']==r['category']
            dual=p and not p['ambiguous'] and p['confidence']>=.9 and p['category']==r['category']
            if not independent and not dual:r['quarantine_reason']='teacher_only_without_independent_or_dualview_agreement'
            else:r['teacher_confirmation']='independent_agent' if independent else 'tight_context_same_teacher_agreement'
    labelled=[r for r in records if r['category'] and not r['quarantine_reason']]
    train=[r for r in labelled if r['source_split']=='train'];val=[r for r in labelled if r['source_split']=='val']
    # Detect conflicting labels on the exact same decoded crop pixels globally.
    hashes=defaultdict(set)
    for r in labelled:hashes[r['crop_sha256']].add(r['category'])
    conflicts={key for key,values in hashes.items() if len(values)>1}
    train=[r for r in train if r['crop_sha256'] not in conflicts];val=[r for r in val if r['crop_sha256'] not in conflicts]
    val_hashes={r['crop_sha256'] for r in val}
    overlap_removed=sum(r['crop_sha256'] in val_hashes for r in train)
    train=[r for r in train if r['crop_sha256'] not in val_hashes]
    rng=random.Random(20261016);rng.shuffle(train)
    # Prefer stronger evidence, spread class samples across source diagrams.
    train.sort(key=lambda r:(r['label_origin']=='original_category_mapping_weak',not r['geometry_human_reviewed']))
    selected=[];per_class=Counter();per_class_image=Counter();seen=set()
    for r in train:
        key=(r['category'],r['image_id'])
        if per_class[r['category']]>=550 or per_class_image[key]>=25 or r['crop_sha256'] in seen:continue
        selected.append(r);seen.add(r['crop_sha256']);per_class[r['category']]+=1;per_class_image[key]+=1
    classes=sorted(per_class)
    unsupported_val=[r for r in val if r['category'] not in classes]
    val=[r for r in val if r['category'] in classes]
    assert not {r['image_id'] for r in selected}&{r['image_id'] for r in val}
    assert not {r['crop_sha256'] for r in selected}&{r['crop_sha256'] for r in val}
    target=LAB/'data/resnet/bootstrap_20261016'
    if target.exists():raise ValueError('Training dataset already frozen')
    for split in ('train','val'):
        for family in classes:(target/split/family).mkdir(parents=True,exist_ok=True)
    exported=[]
    for split,group in [('train',selected),('val',val)]:
        for r in group:
            dest=target/split/r['category']/(r['crop_id']+'.png')
            shutil.copy2(ROOT/r['context_image_path'],dest)
            exported.append({**r,'split':split,'file':dest.relative_to(target).as_posix()})
    write_rows(target/'manifest.jsonl',exported);write_rows(target/'quarantine.jsonl',[r for r in records if not r['category'] or r['quarantine_reason']])
    atomic_json(target/'classes.json',classes)
    summary={'classes':classes,'num_classes':len(classes),'train_crops':len(selected),'val_crops':len(val),
        'train_source_images':len({r['image_id'] for r in selected}),'val_source_images':len({r['image_id'] for r in val}),
        'class_counts':{split:dict(Counter(r['category'] for r in exported if r['split']==split)) for split in ('train','val')},
        'label_origins':{split:dict(Counter(r['label_origin'] for r in exported if r['split']==split)) for split in ('train','val')},
        'train_crops_removed_for_exact_val_overlap':overlap_removed,'conflicting_crop_hashes':len(conflicts),
        'unsupported_val_crops_excluded':len(unsupported_val),'source_image_disjoint':True,'exact_crop_bytes_disjoint':True,
        'teacher_only_validation_labels_are_not_human_gold':True,'all_labels_weak_or_silver':True,
        'missing_train_ontology_classes':sorted(set(read_json(ONTOLOGY)['allowed_labels'])-set(classes)-{'unknown'}),
        'test_used':False,'api_predictions':len(teacher),'supervision_policy':'Original conservative map + cached/new teacher; unknown/ambiguous/conflicting train labels quarantined. Original known val targets retained on teacher disagreement.'}
    summary['raw_classes_removed_from_automatic_mapping']=sorted(overrides)
    summary['teacher_only_requires_confirmation']=True
    atomic_json(target/'summary.json',summary);atomic_json(OUT/'dataset_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','assemble']);a=p.parse_args()
    (prepare if a.action=='prepare' else assemble)()
