"""Freeze a completed review batch and audit cumulative training data before training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'src'))
from component_lab.review_workflow import (atomic_json, export_reviewed_dataset,
                                          image_path, load_batch, read_json,
                                          validate_review_update)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    if args.round < 1:
        raise ValueError('Round must be positive')
    batch = args.batch.resolve()
    state = load_batch(batch)
    if not state['items']:
        raise ValueError('Empty batch')
    ids = [item['id'] for item in state['items']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate review image IDs')
    for item in state['items']:
        checked = validate_review_update(item, {
            'revision': item.get('revision', 0), 'boxes': item['boxes'], 'review': item['review']
        }, state['categories'])
        if not checked['review']['completed']:
            raise ValueError(f"Review incomplete: {item['id']}")
    previous = Path(state['data_root'])
    old_manifest = read_json(previous / 'manifest.json')
    old_by_id = {s['case']: s for s in old_manifest['samples']}
    previous_reviewed = {s['case'] for s in old_manifest['samples']
                         if s.get('annotation_status') == 'human_reviewed'}
    overlap = previous_reviewed.intersection(ids)
    if overlap:
        raise ValueError(f'Batch overlaps earlier rounds: {sorted(overlap)}')
    for item in state['items']:
        if item['split'] != 'train' or old_by_id[item['id']]['split'] != 'train':
            raise ValueError('Non-training review image')
        if sha(batch / item['image_file']) != sha(image_path(previous, old_by_id[item['id']])):
            raise ValueError(f"Review/source image mismatch: {item['id']}")
    snapshot = LAB / 'data/review_snapshots' / f'{batch.name}_round_{args.round:03d}'
    output = LAB / 'data/reviewed_versions' / f'manual_round_{args.round:03d}'
    report_dir = LAB / 'outputs/active_learning' / f'round_{args.round:03d}'
    if snapshot.exists() or output.exists():
        raise ValueError('Snapshot/output already exists; refusing to overwrite')
    before = tree_hashes(batch)
    shutil.copytree(batch, snapshot)
    frozen = tree_hashes(snapshot)
    if before != frozen or before != tree_hashes(batch):
        raise ValueError('Review changed during snapshot; do not train')
    atomic_json(snapshot / 'snapshot_integrity.json', {
        'source': str(batch), 'sha256': frozen, 'round': args.round,
        'all_reviews_complete': True
    })
    summary = export_reviewed_dataset(snapshot, previous, output)
    if summary['reviewed_train_images'] != len(ids) or summary['excluded_pending_images']:
        raise ValueError('Incomplete export')
    yolo = output / 'yolo'
    retained = []
    for ident in sorted(previous_reviewed):
        name = f'{ident}.txt'
        if sha(previous / 'labels/train' / name) != sha(yolo / 'labels/train' / name):
            raise ValueError(f'Earlier correction lost: {ident}')
        retained.append(ident)
    original = LAB / 'data/yolo/pcbnet_component_v1'
    holdout_count = 0
    for subdir in ('images/val', 'images/test', 'labels/val', 'labels/test'):
        expected = {p.name: sha(p) for p in (original / subdir).iterdir()
                    if p.is_file() and p.suffix != '.cache'}
        actual = {p.name: sha(p) for p in (yolo / subdir).iterdir()
                  if p.is_file() and p.suffix != '.cache'}
        if expected != actual:
            raise ValueError(f'Original holdout changed: {subdir}')
        holdout_count += len(actual)
    manifest = read_json(yolo / 'manifest.json')
    split_counts = Counter(s['split'] for s in manifest['samples'])
    expected_split_counts = Counter(s['split'] for s in old_manifest['samples'])
    if split_counts != expected_split_counts:
        raise ValueError(f'Unexpected split sizes: {split_counts}')
    cumulative = sum(s.get('annotation_status') == 'human_reviewed' for s in manifest['samples'])
    if cumulative != len(previous_reviewed) + len(ids):
        raise ValueError('Cumulative reviewed count mismatch')
    initial = {i['id']: i for i in read_json(batch / 'batch.json')['items']}
    rows = []
    for item in state['items']:
        old_boxes = {b['id']: b for b in initial[item['id']]['boxes']}
        accepted = [b for b in item['boxes'] if b['review_status'] == 'accepted']
        rows.append({
            'id': item['id'], 'initial_proposals': len(old_boxes), 'accepted': len(accepted),
            'rejected': sum(b['review_status'] == 'rejected' for b in item['boxes']),
            'added': sum(b['id'] not in old_boxes for b in accepted),
            'moved_accepted': sum(b['id'] in old_boxes and b['bbox'] != old_boxes[b['id']]['bbox']
                                  for b in accepted)
        })
    audit = {
        'round': args.round, 'batch_id': batch.name, 'snapshot': str(snapshot),
        'data_output': str(yolo), 'completed_images': len(ids),
        'cumulative_reviewed_images': cumulative, 'earlier_labels_retained': retained,
        'holdout_files_checked_against_original': holdout_count, 'holdouts_unchanged': True,
        'totals': {k: sum(r[k] for r in rows) for k in
                   ('initial_proposals', 'accepted', 'rejected', 'added', 'moved_accepted')},
        'train_counts': manifest['counts']['train'], 'split_images': dict(split_counts),
        'unreviewed_train_images': split_counts['train'] - cumulative,
        'per_image': rows, 'test_set_evaluated': False
    }
    atomic_json(report_dir / 'review_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
