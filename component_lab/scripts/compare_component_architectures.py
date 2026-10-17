"""Bounded, isolated single-class/two-stage versus multiclass experiment."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / 'outputs/multiclass_vs_two_stage_20261017'
DATA = LAB.parent / 'Data/yolo_experiments/architecture_20261017'
RGB = LAB.parent / 'Data/yolo_experiments/clean_binary_20261017/rgb'
GOLD = LAB / 'data/resnet/official_gold_v1/manifest.json'
os.environ['YOLO_CONFIG_DIR'] = str(LAB / 'outputs/ultralytics_settings')
os.environ['WANDB_DISABLED'] = 'true'
sys.path.insert(0, str(LAB / 'src'))


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protect():
    for p, digest in read(OUT / 'protected_weights.json').items():
        assert sha(LAB / p) == digest, 'Protected weight changed: ' + p
    if (OUT/'data_hashes.json').exists():
        for p,digest in read(OUT/'data_hashes.json').items():
            assert sha(Path(p))==digest,'Experiment data changed: '+p
def gpu():
    import torch
    torch.set_num_threads(4)
    ids = [i for i in range(torch.cuda.device_count()) if '3050 Ti' in torch.cuda.get_device_name(i)]
    if len(ids) != 1: raise RuntimeError('Require exactly one RTX 3050 Ti')
    torch.cuda.set_device(ids[0])
    return torch, ids[0]
def context_box(box, size):
    x1,y1,x2,y2 = box; w,h = size
    mx,my = max(12., .5*(x2-x1)), max(12., .5*(y2-y1))
    return [max(0,math.floor(x1-mx)),max(0,math.floor(y1-my)),
            min(w,math.ceil(x2+mx)),min(h,math.ceil(y2+my))]


def seal_data():
    from PIL import Image
    sys.path.insert(0,str(LAB/'scripts'))
    from prepare_official_gold_crops import validate_split,pixel_sha256
    split=read(LAB/'configs/contest6_group_split_v1.json')
    validation=validate_split(split,read(LAB/split['groups_file']),read(LAB/split['forbidden_manifest']))
    assert sha(GOLD)==read(OUT/'CONFIG.json')['gold_manifest_sha256']
    data=read(OUT/'dataset.json')
    assert Counter(s['partition'] for s in data['sources'])=={'train':140,'val':20}
    assert Counter(r['partition'] for r in data['crops'])=={'train':6417,'val':931}
    train_types={r['type'] for r in data['crops'] if r['partition']=='train'}
    val_types={r['type'] for r in data['crops'] if r['partition']=='val'}
    assert len(train_types)==37 and val_types-train_types=={'crystal'}
    for source in data['sources']:
        for arm in ['single','multi']:
            p=DATA/arm/'images'/source['partition']/(source['case']+'.png')
            with Image.open(p) as image:assert pixel_sha256(image)==source['pixel_sha256']
    hashes={str(p):sha(p) for p in DATA.rglob('*') if p.is_file()}
    hashes[str(GOLD)]=sha(GOLD)
    write(OUT/'data_hashes.json',hashes)
    write(OUT/'split_validation.json',validation)
    protect();print('DATA_SEALED',len(hashes),flush=True)


def prepare():
    from PIL import Image, ImageOps
    if OUT.exists() or DATA.exists(): raise FileExistsError('Do not overwrite an experiment')
    m=read(GOLD); classes=sorted({r['type'] for r in m['crops'] if r['partition']=='train'})
    assert len(classes)==37
    OUT.mkdir(parents=True); DATA.mkdir(parents=True)
    write(OUT/'protected_weights.json',{str(p.relative_to(LAB)):sha(p) for p in (LAB/'weights').glob('*.pt')})
    config={'seed':42,'yolo_epochs':16,'resnet_epochs':12,'imgsz':1280,'batch':1,'nbs':16,
            'classes':classes,'vocabulary_source':'train140 raw types only; dev-only classes remain evaluation FN',
            'yolo_initial':str(LAB/'weights/yolo26n.pt'),'resnet_initial':str(LAB/'weights/imagenet/resnet18-f37072fd.pth'),
            'primary_checkpoint':'fixed final epoch (not different best-mAP criteria)',
            'confidence':.25,'hybrid_yolo_gate':.5,'hybrid_resnet_gate':.8,
            'crop':'per axis margin=max(12px,50% bbox dimension), clip, letterbox224',
            'classifier_balance':'exactly64 draws per train-supported class per epoch; repeat rare originals',
            'strict_scoring':'frozen OCR; common semantic adapter; no prefix type override; restore raw subtype',
            'inference_views':['whole1280','whole1280_plus_1024tiles_overlap25'],
            'amp':False,'device':'RTX3050 Ti only','api_calls':0,'test40_used':False,
            'gold_manifest_sha256':sha(GOLD)}
    for key in ('yolo_initial','resnet_initial'):
        if not Path(config[key]).exists(): raise FileNotFoundError(config[key])
        config[key+'_sha256']=sha(config[key])
    write(OUT/'CONFIG.json',config)
    (OUT/'PLAN.md').write_text('''# 三种 Component 架构受控短训\n\n仅3050 Ti，API0，test40不打开，业务模型不替换。官方 train140/6417 框、dev20/931 框，按已有设计组划分；只由 train 定义37个 raw type IDs，dev-only crystal计入漏检。两种YOLO从同一通用权重各16轮，同配方1280/batch1/nbs16/AdamW .001/seed42/AMP关；主对照取固定最后一轮，不按不同mAP挑best。ResNet从ImageNet独立初始化，12轮，每类每轮64次曝光，稀有类重复不等于新增独立样本。训练标签仅来自官方结构标签，未把定位审核框当人工分类真值。\n\n分类统一使用每轴 max(12px,50%bbox) 外扩上下文。三臂推理conf=.25；多类低分<.5且ResNet>=.8时才复核替换，其余保留多类输出。比较整图与整图+1024切片两种固定视图。全部预测冻结后读dev GT评分。OCR、关联与导出规则一致，关闭位号前缀覆盖类型。报告几何、类型、完整严格ComponentF1以及耗时/显存；主实验是短预算单seed，不代表收敛极限，也不是相同总训练算力（两阶段额外训练ResNet）。\n\n单轮>300秒、OOM或任何任务超30分钟停止并保留失败记录，不自动换卡或只改单臂条件。\n''',encoding='utf-8')
    by_case=defaultdict(list)
    for r in m['crops']: by_case[r['case']].append(r)
    crops=[]; sources=[]
    for source in m['sources']:
        part='train' if source['partition']=='train' else 'val'; case=source['case']
        original=RGB/'images'/part/(case+'.png')
        with Image.open(original) as im: image=im.convert('RGB')
        w,h=image.size; single=[]; multi=[]
        for row in by_case[case]:
            x1,y1,x2,y2=row['bbox_top_left']; x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            geometry=f'{(x1+x2)/2/w:.9f} {(y1+y2)/2/h:.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}'
            single.append('0 '+geometry)
            # YOLO val requires IDs inside the model schema; unknown class remains in our independent eval.
            if row['type'] in classes: multi.append(str(classes.index(row['type']))+' '+geometry)
            crop=ImageOps.pad(image.crop(context_box([x1,y1,x2,y2],image.size)),(224,224),color='white')
            rel=f'crops/{part}/{case}_{len(crops):05d}.png'; dest=DATA/rel;dest.parent.mkdir(parents=True,exist_ok=True);crop.save(dest)
            crops.append({'path':rel,'case':case,'partition':part,'type':row['type'],'key':row['key'],
                          'class_id':classes.index(row['type']) if row['type'] in classes else -1})
        for arm,lines in [('single',single),('multi',multi)]:
            directory=DATA/arm
            for kind in ('images','labels'): (directory/kind/part).mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,directory/'images'/part/original.name)
            (directory/'labels'/part/(case+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        sources.append(dict(case=case,partition=part,pixel_sha256=source['pixel_sha256'],image_sha256=sha(original)))
    for arm,names in [('single',['component']),('multi',classes)]:
        directory=DATA/arm
        (directory/'dataset.yaml').write_text('path: '+directory.as_posix()+'\ntrain: images/train\nval: images/val\nnames: '+json.dumps(names,ensure_ascii=False)+'\n',encoding='utf-8')
    write(OUT/'dataset.json',{'classes':classes,'sources':sources,'crops':crops,
        'counts':dict(Counter(r['partition'] for r in crops)),
        'yolo_internal_validation_excludes_unknown':{'crystal':1},
        'primary_external_evaluation_includes_all_931':True})
    write(OUT/'STATUS.json',{'state':'prepared'})
    protect();print('PREPARED',len(crops),len(classes),flush=True)


def train_yolo(arm):
    from ultralytics import YOLO
    protect();config=read(OUT/'CONFIG.json');torch,device=gpu();directory=OUT/'runs'/arm
    if directory.exists(): raise FileExistsError(directory)
    torch.cuda.reset_peak_memory_stats(device)
    model=YOLO(config['yolo_initial']);progress=[];started=time.perf_counter();tick=[started];seen=set()
    def begin(trainer): tick[0]=time.perf_counter()
    def end(trainer):
        epoch=trainer.epoch+1
        if epoch in seen or epoch>config['yolo_epochs']:return
        seen.add(epoch)
        row={'epoch':epoch,'seconds':time.perf_counter()-tick[0],
             'peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20,
             'metrics':{k:float(v) for k,v in trainer.metrics.items()}}
        progress.append(row);write(OUT/(arm+'_progress.json'),progress)
        write(OUT/'STATUS.json',{'state':'training_yolo','arm':arm,'epoch':epoch,'epochs':config['yolo_epochs']})
        print('EPOCH',arm,json.dumps(row),flush=True)
        if row['seconds']>300 or time.perf_counter()-started>1800: trainer.stop=True
    model.add_callback('on_train_epoch_start',begin);model.add_callback('on_fit_epoch_end',end)
    model.train(data=str(DATA/arm/'dataset.yaml'),epochs=config['yolo_epochs'],imgsz=1280,batch=1,
        device=str(device),workers=0,amp=False,cache=False,project=str(OUT/'runs'),name=arm,exist_ok=False,
        optimizer='AdamW',lr0=.001,lrf=.1,weight_decay=.0005,warmup_epochs=1,warmup_bias_lr=.001,
        seed=42,deterministic=True,nbs=16,patience=100,plots=False,save=True,save_period=-1,
        hsv_h=0,hsv_s=0,hsv_v=0,translate=.05,scale=.2,degrees=0,shear=0,perspective=0,
        flipud=0,fliplr=0,mosaic=0,close_mosaic=0,mixup=0,copy_paste=0,max_det=1000)
    with (directory/'results.csv').open() as f: actual=len(list(csv.DictReader(f)))
    write(OUT/(arm+'_training.json'),{'epochs':actual,'seconds':time.perf_counter()-started,
        'gpu':torch.cuda.get_device_name(device),'checkpoint':str(directory/'weights/last.pt'),
        'checkpoint_sha256':sha(directory/'weights/last.pt'),'progress':progress})
    assert actual==config['yolo_epochs'],'Incomplete run; do not claim matched training budget'
    protect()
    del model
    gc.collect();torch.cuda.empty_cache()


def balanced_indices(targets, epoch, per_class=64):
    by=defaultdict(list)
    for i,c in enumerate(targets):
        if c>=0:by[c].append(i)
    rng=random.Random(42+epoch);result=[]
    for c in sorted(by):
        choices=by[c]
        result.extend(rng.sample(choices,per_class) if len(choices)>=per_class else rng.choices(choices,k=per_class))
    rng.shuffle(result);return result


def train_resnet():
    from PIL import Image
    import numpy as np
    from torch import nn
    from torch.utils.data import DataLoader,Dataset,Subset
    from torchvision.models import resnet18
    from component_lab.models import transform
    protect();torch,device=gpu();torch.cuda.reset_peak_memory_stats(device)
    dev=f'cuda:{device}';config=read(OUT/'CONFIG.json');dataset=read(OUT/'dataset.json')
    directory=OUT/'runs/resnet';directory.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.backends.cudnn.benchmark=False
    preprocess=transform();classes=dataset['classes'];n=len(classes)
    class Crops(Dataset):
        def __init__(self,part):self.rows=[r for r in dataset['crops'] if r['partition']==part]
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            r=self.rows[i]
            with Image.open(DATA/r['path']) as image:x=preprocess(image)
            return x,r['class_id']
    train,val=Crops('train'),Crops('val');targets=[r['class_id'] for r in train.rows]
    model=resnet18(weights=None);model.load_state_dict(torch.load(config['resnet_initial'],map_location='cpu',weights_only=True))
    model.fc=nn.Linear(model.fc.in_features,n);model.to(dev)
    optimizer=torch.optim.AdamW([{'params':[p for name,p in model.named_parameters() if not name.startswith('fc.')],'lr':1e-4},
                                {'params':model.fc.parameters(),'lr':5e-4}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['resnet_epochs'])
    criterion=nn.CrossEntropyLoss(label_smoothing=.05);history=[];started=time.perf_counter()
    for epoch in range(config['resnet_epochs']):
        model.train()
        for name,p in model.named_parameters():p.requires_grad_(epoch>=2 or name.startswith('fc.'))
        if epoch<2:
            for module in model.modules():
                if isinstance(module,nn.BatchNorm2d):module.eval()
        indices=balanced_indices(targets,epoch);counts=Counter(targets[i] for i in indices)
        assert len(counts)==n and set(counts.values())=={64}
        loader=DataLoader(Subset(train,indices),batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
        loss_sum=0.;correct=0;tick=time.perf_counter()
        for x,y in loader:
            x,y=x.to(dev),y.to(dev);optimizer.zero_grad(set_to_none=True);logits=model(x);loss=criterion(logits,y)
            loss.backward();optimizer.step();loss_sum+=float(loss)*len(y);correct+=int((logits.argmax(1)==y).sum())
        scheduler.step();model.eval();matrix=torch.zeros((n+1,n+1),dtype=torch.long)
        with torch.inference_mode():
            for x,y in DataLoader(val,batch_size=32,num_workers=0):
                pred=model(x.to(dev)).argmax(1).cpu();y=torch.where(y<0,n,y)
                matrix+=torch.bincount(y*(n+1)+pred,minlength=(n+1)**2).reshape(n+1,n+1)
        support=matrix.sum(1);tp=matrix.diag();pr=tp/matrix.sum(0).clamp(min=1);rec=tp/support.clamp(min=1)
        f1=2*pr*rec/(pr+rec).clamp(min=1e-12)
        row={'epoch':epoch+1,'seconds':time.perf_counter()-tick,'loss':loss_sum/len(indices),
             'train_balanced_accuracy':correct/len(indices),'val_accuracy_all931':float(tp.sum()/matrix.sum()),
             'val_macro_f1_present29':float(f1[support>0].mean()),'per_class_exposures':64,
             'peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20}
        history.append(row);write(OUT/'resnet_progress.json',history)
        write(OUT/'STATUS.json',{'state':'training_resnet','epoch':epoch+1,'epochs':config['resnet_epochs']})
        print('RESNET',json.dumps(row),flush=True)
        if row['seconds']>300 or time.perf_counter()-started>1800:raise RuntimeError('Classifier budget exceeded')
    torch.save({'classes':classes,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'epoch':12},directory/'last.pt')
    write(OUT/'resnet_training.json',{'epochs':12,'seconds':time.perf_counter()-started,'history':history,
        'checkpoint':str(directory/'last.pt'),'checkpoint_sha256':sha(directory/'last.pt'),
        'matrix':matrix.tolist(),'evaluation_classes':classes+['__unseen_crystal'],
        'support':[int(v) for v in support],'per_class_f1':[float(v) for v in f1]})
    protect()


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','seal','single','multi','resnet','all_train'])
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='seal':seal_data()
    else:
        try:
            if a.stage=='all_train':
                for arm in ['single','multi']:train_yolo(arm);gc.collect()
                train_resnet()
                write(OUT/'STATUS.json',{'state':'training_complete_evaluation_pending'})
            elif a.stage=='resnet':train_resnet()
            else:train_yolo(a.stage)
        except Exception as error:
            write(OUT/'STATUS.json',{'state':'failed','stage':a.stage,'error':type(error).__name__+': '+str(error)})
            raise

if __name__=='__main__':main()
