"""Prepare a fixed shadow-verification pilot; evaluation labels stay in a separate file."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'scripts'))
from analyze_yolo_errors import load_split, match_detections, box_iou


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def neighbors(target, rows):
    a = target['bbox']
    width, height = a[2] - a[0], a[3] - a[1]
    region = [a[0] - max(48, width), a[1] - max(48, height),
              a[2] + max(48, width), a[3] + max(48, height)]
    choices = [r for r in rows if r['id'] != target['id'] and overlap(r['bbox'], region) > 0]
    def rank(row):
        b = row['bbox']
        smaller = min(width * height, (b[2] - b[0]) * (b[3] - b[1]))
        distance = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
        return (-overlap(a, b) / max(smaller, 1e-9), distance, -row['score'])
    return [{k: r[k] for k in ('id', 'bbox', 'score')} for r in sorted(choices, key=rank)[:8]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=LAB/'outputs/active_learning/round_003/candidate_val/predictions.json')
    p.add_argument('--snapshot', type=Path, default=LAB/'data/review_snapshots/batch_003_round_003')
    p.add_argument('--output', type=Path, default=LAB/'outputs/automation_round_001/qwen_pilot')
    p.add_argument('--seed', type=int, default=20261016)
    args = p.parse_args()
    output = args.output.resolve()
    if (output/'manifest.jsonl').exists():
        raise ValueError('Pilot already exists; refusing to change an API experiment')
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    cache = read(args.cache)
    if cache.get('split') != 'val' or cache.get('confidence', 1) > .03:
        raise ValueError('Expected val cache covering confidence .03')
    root = Path(cache['data_root'])
    _, records = load_split(root, 'val')
    all_rows, evaluations, groups = {}, {}, [[] for _ in range(4)]
    edges = [(.03,.1),(.1,.4),(.4,.8),(.8,1.000001)]
    for record in records:
        image_id = Path(record['image']).stem
        predictions = cache['predictions'][record['image']]
        matched = match_detections(record['ground_truth'], predictions, .5, .03)
        true_indices = {m['prediction_index'] for m in matched['matches']}
        local_rows = []
        for index, pred in enumerate(predictions):
            if pred['score'] < .03:
                continue
            row = {'id':f'val_{image_id}_{index:04d}', 'image_path':str(record['image_path']),
                   'bbox':pred['bbox'], 'score':pred['score'], 'source':pred.get('source','yolo'),
                   'image_sha256':sha(record['image_path']), 'ocr':[]}
            local_rows.append(row)
            all_rows[row['id']] = row
            evaluations[row['id']] = {'cohort':'val_stratified_diagnostic', 'image_id':image_id,
                'reference_kind':'original_PCBnet_geometry_proxy_not_new_human_adjudication',
                'reference_keep':index in true_indices,
                'best_original_gt_iou':max((box_iou(pred['bbox'],g['bbox']) for g in record['ground_truth']),default=0)}
            for bucket, (lo, hi) in enumerate(edges):
                if lo <= pred['score'] < hi:
                    groups[bucket].append(row['id'])
        for row in local_rows:
            row['neighbors'] = neighbors(row, local_rows)
    selected, counts = [], Counter()
    for group in groups:
        rng.shuffle(group)
        chosen = []
        for ident in group:
            image_id = evaluations[ident]['image_id']
            if counts[image_id] < 2:
                chosen.append(ident)
                counts[image_id] += 1
            if len(chosen) == 4:
                break
        if len(chosen) != 4:
            raise ValueError('Not enough distinct validation candidates for a score stratum')
        selected.extend(chosen)

    initial = read(args.snapshot/'batch.json')
    final_items = {item['id']: read(args.snapshot/'items'/f"{item['id']}.json")
                   for item in initial['items']}
    all_review, accepted = {}, []
    for item in initial['items']:
        final = {b['id']:b for b in final_items[item['id']]['boxes']}
        local_rows = []
        for box in item['boxes']:
            ident = f"review_{item['id']}_{box['id']}"
            row = {'id':ident,'image_path':str((args.snapshot/item['image_file']).resolve()),
                   'bbox':box['bbox'],'score':box['score'],'source':box.get('source','yolo'),
                   'image_sha256':sha(args.snapshot/item['image_file']),'ocr':[]}
            local_rows.append(row)
            all_rows[ident] = row
            all_review[(item['id'],box['id'])] = ident
            edited = final[box['id']]['bbox'] != box['bbox']
            evaluations[ident] = {'cohort':'reviewed_train_selected_diagnostic', 'image_id':item['id'],
                'reference_kind':'human_proposal_decision_not_semantic_noncomponent_label',
                'reference_keep':final[box['id']]['review_status']=='accepted',
                'bbox_edited_after_proposal':edited,'not_independent_detection_validation':True}
            if final[box['id']]['review_status']=='accepted' and not edited:
                accepted.append(ident)
        for row in local_rows:
            row['neighbors'] = neighbors(row, local_rows)
    # Fixed difficult cases span duplicate, parent-fragment, text and oversized boxes.
    cases = [('0087','yolo_0020'),('0087','yolo_0075'),('0150','yolo_0039'),
             ('0114','yolo_0067'),('0288','yolo_0073'),('0288','yolo_0075'),
             ('0288','yolo_0087'),('0087','yolo_0031')]
    rejected = [all_review[key] for key in cases]
    if any(evaluations[i]['reference_keep'] for i in rejected):
        raise ValueError('A fixed rejection case is not actually rejected')
    rng.shuffle(accepted)
    positive, positive_counts = [], Counter()
    for ident in accepted:
        case = evaluations[ident]['image_id']
        if positive_counts[case] < 1:
            positive.append(ident)
            positive_counts[case] += 1
        if len(positive) == 8:
            break
    if len(positive) != 8:
        raise ValueError('Insufficient unchanged accepted diagnostic proposals')
    # Alternate positives/negatives so a partial run is not one-sided.
    for yes, no in zip(positive, rejected):
        selected.extend([yes,no])
    assert len(selected)==32 and len(set(selected))==32
    request_rows = [all_rows[ident] for ident in selected]
    # Reference fields never enter this input manifest or the API prompt.
    forbidden = ('reference_keep','review_status','category_hint','ground_truth','accepted','rejected')
    assert all(not any(key in row for key in forbidden) for row in request_rows)
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in request_rows),encoding='utf-8')
    (output/'evaluation_reference.json').write_text(json.dumps({
        'seed':args.seed,'selection':'16 val score-stratified + 8 unchanged accepted + 8 selected rejected train proposals',
        'not_an_independent_95_percent_claim':True,'reference_labels_sent_to_model':False,
        'items':{ident:evaluations[ident] for ident in selected}},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'candidates':32,'cohorts':dict(Counter(evaluations[i]['cohort'] for i in selected)),
                      'reference_labels_sent_to_model':False,'manifest':str(output/'manifest.jsonl')},ensure_ascii=False))


if __name__=='__main__':
    main()
