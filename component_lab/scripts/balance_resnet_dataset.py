"""Equal class quotas with explicit train-only augmentation and honest source counts."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
from PIL import Image,ImageOps,ImageEnhance
from bootstrap_resnet_data import LAB,rows,write_rows,sha
from component_lab.review_workflow import atomic_json

def augment(path,rng):
    with Image.open(path) as original:im=original.convert('RGB')
    angle=rng.choice([0,0,0,90,180,270])+rng.uniform(-3,3)
    im=im.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True,fillcolor='white')
    side=rng.randint(176,200)
    im=ImageOps.contain(im,(side,side),Image.Resampling.LANCZOS)
    im=ImageEnhance.Contrast(im).enhance(rng.uniform(.85,1.15))
    im=ImageEnhance.Color(im).enhance(rng.uniform(.8,1.08))
    canvas=Image.new('RGB',(224,224),'white')
    canvas.paste(im,((224-im.width)//2+rng.randint(-7,7),(224-im.height)//2+rng.randint(-7,7)))
    return canvas

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=LAB/'data/resnet/bootstrap_20261016')
    p.add_argument('--output',type=Path,default=LAB/'data/resnet/balanced_20261016')
    p.add_argument('--per-class',type=int,default=256)
    a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('Balanced dataset already exists; do not overwrite')
    if a.per_class<1:raise ValueError('Quota must be positive')
    records=rows(source/'manifest.jsonl');classes=json.loads((source/'classes.json').read_text(encoding='utf-8'))
    val=[r for r in records if r['split']=='val'];train=[r for r in records if r['split']=='train']
    rng=random.Random(20261016);exported=[];stats={}
    val_hashes={r['crop_sha256'] for r in val};train_hashes=set()
    for split in ('train','val'):
        for family in classes:(out/split/family).mkdir(parents=True,exist_ok=True)
    for family in classes:
        available=[r for r in train if r['category']==family]
        if not available:raise ValueError('Cannot fabricate a class without a real seed: '+family)
        rng.shuffle(available)
        # Spread actual examples across diagrams before adding repeated views.
        grouped={}
        for r in available:grouped.setdefault(r['image_id'],[]).append(r)
        ordered=[]
        while any(grouped.values()):
            for group in grouped.values():
                if group:ordered.append(group.pop())
        real=ordered[:a.per_class]
        stats[family]={'available_real_crops':len(available),'selected_real_crops':len(real),
            'real_source_images':len({r['image_id'] for r in real}),
            'augmented_crops':a.per_class-len(real),'total_training_files':a.per_class,
            'validation_crops':sum(r['category']==family for r in val)}
        for i in range(a.per_class):
            base=real[i%len(real)];synthetic=i>=len(real)
            name=(base['crop_id']+f'_aug{i:04d}' if synthetic else base['crop_id'])+'.png'
            dest=out/'train'/family/name
            if synthetic:
                for attempt in range(20):
                    augment(source/base['file'],rng).save(dest)
                    digest=sha(dest)
                    if digest not in val_hashes and digest not in train_hashes:break
                else:raise ValueError('Could not create a distinct train view')
            else:
                shutil.copy2(source/base['file'],dest);digest=sha(dest)
            assert digest not in val_hashes
            train_hashes.add(digest)
            row={**base,'source_crop_sha256':base['crop_sha256'],'crop_sha256':digest,
                'file':dest.relative_to(out).as_posix(),'is_augmented':synthetic,
                'augmentation_parent_crop_id':base['crop_id'] if synthetic else None}
            if synthetic:row['crop_id']=dest.stem
            exported.append(row)
    for r in val:
        dest=out/r['file'];shutil.copy2(source/r['file'],dest)
        assert sha(dest)==r['crop_sha256']
        exported.append({**r,'is_augmented':False})
    counts=Counter(r['category'] for r in exported if r['split']=='train')
    assert set(counts.values())=={a.per_class}
    assert not {r['image_id'] for r in exported if r['split']=='train'}&{r['image_id'] for r in val}
    write_rows(out/'manifest.jsonl',exported);atomic_json(out/'classes.json',classes)
    summary={'source':str(source),'source_manifest_sha256':sha(source/'manifest.jsonl'),'seed':20261016,
        'classes':classes,'num_classes':len(classes),'per_class_training_files':a.per_class,
        'train_crops':sum(counts.values()),'val_crops':len(val),'per_class':stats,
        'all_class_file_counts_exactly_equal':True,'validation_unmodified':True,
        'source_images_disjoint':True,'exact_crop_bytes_disjoint':True,
        'warning':'Augmented views are not independent new real symbols. Rare-class source shortages remain visible; all categories remain weak/silver.',
        'augmentation':'Train only: whole-crop rotations, slight angle/scale/translation and contrast/color; preserve entire symbol inside white canvas. No val/test augmentation.'}
    atomic_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
