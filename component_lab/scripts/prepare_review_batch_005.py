"""Prepare batch 005 using cached design groups, without opening sealed images."""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'src'))
from component_lab.review_workflow import atomic_json, read_json, image_path, prepare_batch
from component_lab.category_audit import select_review_batch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = LAB / 'data/reviewed_versions/manual_round_004/yolo'
    dest = LAB / 'data/review_sources/batch_005/yolo'
    batch = LAB / 'data/review_batches/batch_005'
    audit_root = LAB / 'outputs/eval_split_audit_20261017'
    old = read_json(source / 'manifest.json')
    groups = read_json(audit_root / 'groups.json')
    group_for = {i: g for g, row in groups.items() for i in row['members']}
    fps = read_json(audit_root / 'fingerprints.json')['images']
    prefix = 'lab:reviewed_versions/manual_round_004/yolo/images/'
    case_group = {Path(f['path']).stem: group_for[f['id']]
                  for f in fps if f['id'].startswith(prefix)}
    banned = set(read_json(audit_root / 'forbidden_for_clean_retrain.json')['held_out_groups'])
    banned.update(group_for[f['id']] for f in fps
                  if f['id'].startswith(prefix) and f['role'] in ('val', 'test'))
    previous = sorted(p.parent for p in (LAB / 'data/review_batches').glob('batch_*/batch.json')
                      if p.parent != batch)
    seen, seen_groups, projects = set(), set(), set()
    for directory in previous:
        for item in read_json(directory / 'batch.json')['items']:
            seen.add(item['id'])
            if item['id'] in case_group:
                seen_groups.add(case_group[item['id']])
            project = item.get('source_metadata', {}).get('source_project')
            if project:
                projects.add(project.casefold())
    weights = LAB / 'weights/pcbnet_component_yolo26n_review_round_004.pt'
    before = sha(weights)
    seed = 20261025
    if dest.exists():
        audit = read_json(dest / 'mixed_source_audit.json')
        assert audit['parent_manifest_sha256'] == sha(source / 'manifest.json')
        selected = audit['selection']
    else:
        allowed = [s for s in old['samples'] if s['split'] != 'train' or
                   (case_group.get(s['case']) and case_group[s['case']] not in banned)]
        pool, pool_groups = [], set()
        for sample in allowed:
            g = case_group.get(sample['case'])
            if (sample['split'] == 'train' and not sample['case'].startswith('hf_')
                    and sample['case'] not in seen and g not in seen_groups | pool_groups
                    and 15 <= len(sample['labels']) <= 80
                    and max(sample['width'], sample['height']) <= 2500):
                pool.append(sample)
                pool_groups.add(g)
        pcb = select_review_batch(dict(old, samples=pool), count=10, seed=seed)
        assert len(pcb) == 10
        hf_root = LAB.parent / 'Data/open_schematics'
        raw = [json.loads(line) for line in (hf_root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        occupied = banned | {case_group[s['case']] for s in old['samples'] if s['case'] in case_group}
        candidates = [r for r in raw if r.get('structured_json_valid')
                      and 20 <= r.get('placed_symbol_count', 0) <= 70
                      and 1000 <= max(r['width'], r['height']) <= 2250
                      and min(r['width'], r['height']) >= 700
                      and group_for.get('hf_os:' + r['id']) is not None
                      and group_for['hf_os:' + r['id']] not in occupied
                      and r['source_repository'].casefold() not in projects]
        random.Random(seed).shuffle(candidates)
        picked, covered = [], set()
        while candidates and len(picked) < 10:
            row = max(candidates, key=lambda r: len(set(r.get('component_library_names', [])) - covered))
            candidates.remove(row)
            project, g = row['source_repository'].casefold(), group_for['hf_os:' + row['id']]
            if project in projects or g in occupied:
                continue
            picked.append(row)
            projects.add(project)
            occupied.add(g)
            covered.update(row.get('component_library_names', []))
        assert len(picked) == 10, 'Need ten distinct safe HF projects'
        dest.mkdir(parents=True)
        for sample in allowed:
            original = image_path(source, sample)
            target = dest / original.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            label = source / 'labels' / sample['split'] / (sample['case'] + '.txt')
            if label.exists():
                target_label = dest / label.relative_to(source)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label, target_label)
                assert sha(label) == sha(target_label)
        hf_selected = []
        for row in picked:
            case = 'hf_os_' + row['source_sha256'][:12]
            target = dest / 'images/pending' / (case + '.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hf_root / row['image_file'], target)
            sample = dict(case=case, split='train', width=row['width'], height=row['height'], labels=[],
                          sha256=sha(target), image=str((hf_root / row['image_file']).resolve()),
                          annotation_status='pending_review', pending_image_file=target.relative_to(dest).as_posix(),
                          source_dataset=row['dataset'], dataset_revision=row['revision'], source_record_id=row['id'],
                          source_project=row['source_repository'], source_project_url=row['source_project_url'],
                          source_file=str((hf_root / row['source_file']).resolve()), dataset_license=row['dataset_license'],
                          source_sha256=row['source_sha256'], native_placed_symbol_count_hint=row['placed_symbol_count'])
            allowed.append(sample)
            hf_selected.append(dict(sample, selection_group='hf_project_diversity', priority=0,
                                    selection_reasons=['新项目、适中密度；源文件符号数仅作抽样参考。', '完整检查漏检后再通过；未审核不进入训练。']))
        selected = []
        for a, b in zip(pcb, hf_selected):
            a['source_dataset'] = 'PCBnet/Dataset300'
            selected.extend([a, b])
        for n, row in enumerate(selected, 1):
            row['priority'] = n
        manifest = copy.deepcopy(old)
        manifest['samples'] = allowed
        for sample in allowed:
            if not sample['case'].startswith('hf_'):
                sample.setdefault('source_dataset', 'PCBnet/Dataset300')
        manifest['splits'] = {split: [s['case'] for s in allowed if s['split'] == split] for split in ('train', 'val', 'test')}
        manifest['pending_review_policy'] = 'No raw HF training labels; export only after whole-page human approval.'
        atomic_json(dest / 'manifest.json', manifest)
        (dest / 'dataset.yaml').write_text('path: ' + json.dumps(str(dest), ensure_ascii=False) + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [component]\n', encoding='utf-8')
        atomic_json(dest / 'mixed_source_audit.json', dict(seed=seed, parent_data=str(source),
                    parent_manifest_sha256=sha(source / 'manifest.json'), selection=selected,
                    ratio={'pcbnet': 10, 'huggingface_open_schematics': 10},
                    excluded_training_cases=[s['case'] for s in old['samples'] if s['split'] == 'train' and s not in allowed],
                    split_policy='Cached design groups; official dev/test and legacy val/test groups excluded from training. Previous batches unchanged. No sealed image decoded.',
                    limitation='Legacy round004 proposal model is not a clean-retrained model; cached deduplication is heuristic.',
                    weights_sha256=before))
    state = prepare_batch(dest, batch, weights, count=20, seed=seed, device='0', imgsz=1280,
                          confidence=.1, tile_size=1024, exclude_batches=previous, selected_samples=selected)
    state['selection']['source_ratio'] = {'pcbnet': 10, 'huggingface_open_schematics': 10}
    state['selection']['source_audit'] = str(dest / 'mixed_source_audit.json')
    state['inference']['weights_sha256'] = before
    assert sha(weights) == before, 'Business weights changed'
    atomic_json(batch / 'batch.json', state)
    print(json.dumps(dict(batch_id=state['batch_id'], images=len(state['items']),
                         proposals=sum(len(i['boxes']) for i in state['items'])), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
