"""Create a cumulative, source-balanced review batch without training raw HF images."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import numpy as np
from PIL import Image

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB/'src'))
from component_lab.review_workflow import atomic_json, read_json, image_path, prepare_batch
from component_lab.category_audit import select_review_batch

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint(path):
    with Image.open(path) as original:
        image=original.convert('RGB')
        w,h=image.size
        digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        # Ink masks retain page layout; exclude page-border margins for the
        # near-duplicate comparison so common frames cannot dominate overlap.
        thumb=np.asarray(image.resize((128,128),Image.Resampling.LANCZOS))
        ink=thumb.min(axis=2)<220
        ink[:5,:]=False;ink[-5:,:]=False;ink[:,:5]=False;ink[:,-5:]=False
        gray=np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS))
        dhash=sum(int(x)<<i for i,x in enumerate((gray[:,1:]>gray[:,:-1]).flatten()))
        return {'sha256':sha(path),'rgb_sha256':digest,'aspect':w/h,'dhash':dhash,'ink':ink,'path':str(path)}

def duplicate(candidate, existing):
    for item in existing:
        if candidate['sha256']==item['sha256'] or candidate['rgb_sha256']==item['rgb_sha256']:
            return {'path':item['path'],'reason':'exact_file_or_decoded_rgb'}
        if abs(candidate['aspect']/item['aspect']-1)>.03 or (candidate['dhash']^item['dhash']).bit_count()>4:
            continue
        union=np.logical_or(candidate['ink'],item['ink']).sum()
        overlap=np.logical_and(candidate['ink'],item['ink']).sum()/union if union else 0
        if union>=30 and overlap>=.88:
            return {'path':item['path'],'reason':'near_duplicate_layout','ink_iou':float(overlap)}
    return None

def stage(source, hf_root, destination, previous_batches, seed=20261020):
    if destination.exists():
        raise ValueError('Staging destination already exists; do not overwrite')
    old=read_json(source/'manifest.json')
    excluded=set();previous_projects=set();old_batch_images=[]
    for directory in previous_batches:
        batch=read_json(directory/'batch.json')
        for item in batch['items']:
            excluded.add(item['id'])
            if item.get('source_metadata',{}).get('source_project'):
                previous_projects.add(item['source_metadata']['source_project'].casefold())
            old_batch_images.append(directory/item['image_file'])
    eligible=copy.deepcopy(old)
    eligible['samples']=[s for s in old['samples'] if s['split']=='train' and s['case'] not in excluded and s.get('annotation_status')!='pending_review']
    pcb=select_review_batch(eligible,count=10,seed=seed)
    if len(pcb)!=10:
        raise ValueError('Fewer than ten unseen PCBnet training images')
    print('Screening duplicate candidates against all train/val/test and previous review images',flush=True)
    existing=[];seen_paths=set()
    for path in [image_path(source,s) for s in old['samples']]+old_batch_images:
        key=str(path.resolve())
        if key not in seen_paths:
            existing.append(fingerprint(path));seen_paths.add(key)
    raw=[json.loads(line) for line in (hf_root/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    hf_candidates=[r for r in raw if r.get('structured_json_valid') and 20<=r.get('placed_symbol_count',0)<=110
                   and 1000<=max(r['width'],r['height'])<=2250 and min(r['width'],r['height'])>=700
                   and r.get('source_repository','').casefold() not in previous_projects
                   and 'hf_os_'+r['source_sha256'][:12] not in excluded]
    rng=random.Random(seed);rng.shuffle(hf_candidates)
    picked=[];screen=[];projects=set(previous_projects)
    # Source-native symbol names are sampling metadata, not detection labels.
    # Mix moderate and denser pages, then prefer new libraries/project styles.
    covered=set()
    while hf_candidates and len(picked)<10:
        if len(picked)<4:
            row=hf_candidates.pop(0)
        else:
            target_dense=len(picked)>=7
            row=max(hf_candidates,key=lambda r:(int((r['placed_symbol_count']>=60)==target_dense),
                    len(set(r.get('component_library_names',[]))-covered),-r['placed_symbol_count']))
            hf_candidates.remove(row)
        project=row['source_repository'].casefold()
        if project in projects:continue
        path=hf_root/row['image_file']
        if not path.is_file():continue
        candidate=fingerprint(path)
        match=duplicate(candidate,existing)
        if match:
            screen.append({'hf_id':row['id'],**match});continue
        picked.append(row);projects.add(project);existing.append(candidate)
        covered.update(row.get('component_library_names',[]))
    if len(picked)!=10:
        raise ValueError('Fewer than ten eligible unique HF projects; ratio not changed')
    # All historic files are copied, not changed; raw HF images stay outside
    # images/train. There are deliberately no HF training label placeholders.
    destination.mkdir(parents=True)
    for directory in ('images','labels'):
        shutil.copytree(source/directory,destination/directory,ignore=shutil.ignore_patterns('*.cache'))
    manifest=copy.deepcopy(old)
    selected=[];hf_samples=[]
    for row in picked:
        case='hf_os_'+row['source_sha256'][:12]
        target=destination/'images/pending'/(case+'.png')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(hf_root/row['image_file'],target)
        with Image.open(target) as image:w,h=image.size
        sample={'case':case,'split':'train','width':w,'height':h,'labels':[],
                'sha256':sha(target),'image':str((hf_root/row['image_file']).resolve()),
                'annotation_status':'pending_review','pending_image_file':target.relative_to(destination).as_posix(),
                'source_dataset':row['dataset'],'dataset_revision':row['revision'],'source_record_id':row['id'],
                'source_project':row['source_repository'],'source_project_url':row['source_project_url'],
                'source_file':str((hf_root/row['source_file']).resolve()),'dataset_license':row['dataset_license'],
                'source_sha256':row['source_sha256'],'native_placed_symbol_count_hint':row['placed_symbol_count'],
                'native_symbol_names_for_sampling_only':row.get('component_library_names',[])}
        hf_samples.append(sample)
        selected.append(dict(sample,selection_group='hf_project_diversity',
            selection_reasons=['HF新项目整图；源文件符号数量仅作抽样参考，不是逐元件真值。','未审核前不进入训练；只审核模型框还需检查整图遗漏。'],priority=0))
    for row in pcb:
        sample=next(s for s in manifest['samples'] if s['case']==row['case'])
        sample.setdefault('source_dataset','PCBnet/Dataset300')
        row['source_dataset']='PCBnet/Dataset300'
    manifest['samples'].extend(hf_samples)
    manifest['splits']={split:[s['case'] for s in manifest['samples'] if s['split']==split] for split in ('train','val','test')}
    manifest['pending_review_policy']='images/pending have no labels and are excluded from training until whole-image human review; exporter omits unfinished new images.'
    atomic_json(destination/'manifest.json',manifest)
    (destination/'dataset.yaml').write_text('path: '+json.dumps(str(destination),ensure_ascii=False)+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [component]\n',encoding='utf-8')
    interleaved=[]
    for a,b in zip(pcb,selected):
        interleaved.extend([a,b])
    for i,row in enumerate(interleaved,1):row['priority']=i
    holdouts={}
    for subdir in ('images/val','images/test','labels/val','labels/test'):
        for original in (source/subdir).iterdir():
            if not original.is_file() or original.suffix=='.cache':continue
            relative=original.relative_to(source)
            if sha(original)!=sha(destination/relative):raise ValueError('Holdout copy changed')
            holdouts[relative.as_posix()]=sha(original)
    old_reviewed=[s['case'] for s in old['samples'] if s.get('annotation_status')=='human_reviewed']
    for case in old_reviewed:
        relative=Path('labels/train')/(case+'.txt')
        if sha(source/relative)!=sha(destination/relative):raise ValueError('Earlier correction changed')
    audit={'seed':seed,'parent_data':str(source),'parent_manifest_sha256':sha(source/'manifest.json'),
           'selected_pcbnet':[r['case'] for r in pcb],'selected_hf':[r['case'] for r in selected],
           'ratio':{'pcbnet':10,'huggingface_open_schematics':10},'excluded_previous_ids':sorted(excluded),
           'unique_hf_projects':[r['source_repository'] for r in picked],'hf_duplicate_rejections':screen,
           'duplicate_screen':'Exact file/decoded RGB against all splits and prior batches; aspect within 3%, dHash<=4 plus border-excluded 128px ink IoU>=0.88 for near duplicates. Not proof of absence of all transformed duplicates.',
           'earlier_reviewed_labels_preserved':old_reviewed,'holdout_sha256':holdouts,
           'pending_hf_images_outside_training_directory':True,'hf_training_labels_created':0,
           'selection':[{'case':r['case'],'selection_group':r['selection_group'],'selection_reasons':r['selection_reasons'],'priority':r['priority']} for r in interleaved]}
    atomic_json(destination/'mixed_source_audit.json',audit)
    print(json.dumps({'staging':str(destination),'pcbnet':10,'hf':10,'earlier_reviewed_labels_retained':len(old_reviewed),'holdout_files':len(holdouts)},ensure_ascii=False),flush=True)
    return interleaved

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=LAB/'data/reviewed_versions/manual_round_003/yolo')
    parser.add_argument('--hf-root',type=Path,default=LAB.parent/'Data/open_schematics')
    parser.add_argument('--staging',type=Path,default=LAB/'data/review_sources/batch_004/yolo')
    parser.add_argument('--batch',type=Path,default=LAB/'data/review_batches/batch_004')
    parser.add_argument('--weights',type=Path,default=LAB/'weights/pcbnet_component_yolo26n_review_round_003.pt')
    parser.add_argument('--seed',type=int,default=20261020)
    parser.add_argument('--device',default='0')
    args=parser.parse_args()
    previous=sorted(p for p in (LAB/'data/review_batches').glob('batch_*') if (p/'batch.json').exists() and p.resolve()!=args.batch.resolve())
    if args.staging.exists():
        audit=read_json(args.staging/'mixed_source_audit.json')
        if audit['parent_manifest_sha256']!=sha(args.source/'manifest.json'):raise ValueError('Staging parent changed')
        selected=audit['selection']
    else:
        selected=stage(args.source.resolve(),args.hf_root.resolve(),args.staging.resolve(),previous,args.seed)
    state=prepare_batch(args.staging,args.batch,args.weights,count=20,seed=args.seed,device=args.device,
                        imgsz=1280,confidence=.1,tile_size=1024,exclude_batches=previous,selected_samples=selected)
    state['selection']['source_ratio']={'pcbnet':10,'huggingface_open_schematics':10}
    state['selection']['source_audit']=str((args.staging/'mixed_source_audit.json').resolve())
    state['inference']['weights_sha256']=sha(args.weights)
    atomic_json(args.batch/'batch.json',state)
    print(json.dumps({'batch_id':state['batch_id'],'images':len(state['items']),'proposals':sum(len(i['boxes']) for i in state['items'])},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
