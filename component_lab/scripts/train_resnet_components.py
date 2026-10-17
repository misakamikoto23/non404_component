"""Train a reproducible ResNet18 baseline on disjoint weak/silver crop sources."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
from PIL import Image,ImageOps,ImageDraw

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB/'src'))
from bootstrap_resnet_data import rows,sha
from component_lab.review_workflow import atomic_json
from component_lab.training import build_training_sample_weights

def metrics(matrix,confidence=None,correct_flags=None):
    total=matrix.sum(1);predicted=matrix.sum(0);tp=matrix.diag()
    precision=tp/(predicted.clamp(min=1));recall=tp/(total.clamp(min=1))
    f1=2*precision*recall/(precision+recall).clamp(min=1e-12)
    support=total>0
    result={'accuracy':float(tp.sum()/total.sum().clamp(min=1)),
        'macro_recall_present_classes':float(recall[support].mean()) if support.any() else None,
        'macro_f1_present_classes':float(f1[support].mean()) if support.any() else None,
        'samples':int(total.sum()),'support':[int(x) for x in total],
        'per_class_precision':[float(x) for x in precision],'per_class_recall':[float(x) for x in recall],
        'per_class_f1':[float(x) for x in f1]}
    if confidence is not None:
        selected=confidence>=.8
        result['threshold_0_8']={'selected':int(selected.sum()),'coverage':float(selected.float().mean()),
            'accuracy':float(correct_flags[selected].float().mean()) if selected.any() else None,
            'warning':'Softmax is uncalibrated; targets are weak/silver, not human class gold.'}
    return result

def main():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader,WeightedRandomSampler,Dataset
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torchvision.models import resnet18
    from component_lab.models import transform
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=LAB/'data/resnet/bootstrap_20261016')
    p.add_argument('--output',type=Path,default=LAB/'runs/resnet18_bootstrap_20261016')
    p.add_argument('--initial',type=Path,default=LAB/'weights/imagenet/resnet18-f37072fd.pth')
    p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--batch',type=int,default=96);p.add_argument('--patience',type=int,default=6)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'history.json').exists():raise ValueError('Refusing to overwrite existing training run')
    torch.manual_seed(42);random.seed(42);np.random.seed(42);torch.set_num_threads(4)
    if not torch.cuda.is_available() and args.device.startswith('cuda'):raise RuntimeError('CUDA unavailable')
    torch.backends.cudnn.benchmark=True
    classes=json.loads((args.data/'classes.json').read_text(encoding='utf-8'));n=len(classes)
    manifest=rows(args.data/'manifest.jsonl');by_file={r['file']:r for r in manifest}
    train_sources={r['image_id'] for r in manifest if r['split']=='train'}
    val_sources={r['image_id'] for r in manifest if r['split']=='val'}
    assert not train_sources&val_sources
    assert not {r['crop_sha256'] for r in manifest if r['split']=='train'}&{r['crop_sha256'] for r in manifest if r['split']=='val'}
    augmentation=transforms.Compose([
        transforms.Lambda(lambda im:ImageOps.pad(im.convert('RGB'),(224,224),color='white')),
        transforms.RandomApply([transforms.RandomRotation((90,90),fill=(255,255,255))],p=.25),
        transforms.RandomAffine(degrees=4,translate=(.035,.035),scale=(.92,1.06),fill=(255,255,255)),
        transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    folders={'train':ImageFolder(args.data/'train',transform=augmentation,allow_empty=True),
             'val':ImageFolder(args.data/'val',transform=transform(),allow_empty=True)}
    assert all(d.classes==classes for d in folders.values())
    class Indexed(Dataset):
        def __init__(self,folder):self.folder=folder
        def __len__(self):return len(self.folder)
        def __getitem__(self,index):
            image,target=self.folder[index]
            return image,target,index
    plan=build_training_sample_weights(folders['train'].targets,n,5)
    exactly_balanced=len(set(plan['class_counts']))==1
    sampler=None if exactly_balanced else WeightedRandomSampler(plan['sample_weights'],len(folders['train']),replacement=True,generator=torch.Generator().manual_seed(42))
    loaders={'train':DataLoader(Indexed(folders['train']),batch_size=args.batch,sampler=sampler,shuffle=sampler is None,num_workers=0,pin_memory=True),
             'val':DataLoader(Indexed(folders['val']),batch_size=args.batch,shuffle=False,num_workers=0,pin_memory=True)}
    train_quality=[];val_original=[]
    for path,_ in folders['train'].samples:
        r=by_file[Path(path).relative_to(args.data).as_posix()]
        train_quality.append(.5 if 'only_silver' in r['label_origin'] else (.75 if r['label_origin']=='original_category_mapping_weak' else 1.0))
    for path,_ in folders['val'].samples:
        r=by_file[Path(path).relative_to(args.data).as_posix()];val_original.append(r['original_family'] is not None)
    train_quality=torch.tensor(train_quality,dtype=torch.float32)
    val_original=torch.tensor(val_original,dtype=torch.bool)
    model=resnet18(weights=None);model.load_state_dict(torch.load(args.initial,map_location='cpu',weights_only=True))
    model.fc=nn.Linear(model.fc.in_features,n);model.to(args.device)
    head=list(model.fc.parameters());head_ids={id(x) for x in head};backbone=[x for x in model.parameters() if id(x) not in head_ids]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':1e-4},{'params':head,'lr':5e-4}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    criterion=nn.CrossEntropyLoss(label_smoothing=.05,reduction='none')
    amp=args.device.startswith('cuda');scaler=torch.amp.GradScaler('cuda',enabled=amp)
    config={'architecture':'resnet18','classes':classes,'seed':42,'epochs':args.epochs,'batch':args.batch,
        'device':args.device,'device_name':torch.cuda.get_device_name(args.device) if amp else 'cpu','initial_sha256':sha(args.initial),'dataset_manifest_sha256':sha(args.data/'manifest.jsonl'),
        'train_crops':len(folders['train']),'val_crops':len(folders['val']),'train_source_images':len(train_sources),'val_source_images':len(val_sources),
        'balance':'exact equal class file counts; shuffle without replacement each epoch' if exactly_balanced else 'sqrt sampler capped5, no class-weighted loss','class_counts':dict(zip(classes,plan['class_counts'])),
        'class_sampling_weights':dict(zip(classes,plan['class_weights'])),'label_smoothing':.05,'amp':amp,
        'quality_weights':{'teacher_only':.5,'original_weak':.75,'teacher_agrees_original':1.0},
        'warmup':'head-only first2 epochs','preprocess':'letterbox224_imagenet','all_targets_weak_or_silver':True,
        'best_selection':'macro F1 over present val classes; weak/silver targets, not human accuracy','test_used':False}
    atomic_json(out/'config.json',config)
    best=-1.;best_epoch=0;history=[];begin=time.perf_counter()
    def evaluate():
        model.eval();matrix=torch.zeros((n,n),dtype=torch.long);original_matrix=matrix.clone()
        all_conf=[];all_correct=[];predictions=[]
        with torch.inference_mode():
            for images,labels,indices in loaders['val']:
                with torch.autocast(device_type='cuda',enabled=amp):prob=model(images.to(args.device,non_blocking=True)).float().softmax(1)
                conf,pred=prob.max(1);pred=pred.cpu();conf=conf.cpu()
                matrix+=torch.bincount(labels*n+pred,minlength=n*n).reshape(n,n)
                mask=val_original[indices]
                original_matrix+=torch.bincount(labels[mask]*n+pred[mask],minlength=n*n).reshape(n,n)
                all_conf.append(conf);all_correct.append(pred==labels)
                for idx,target,guess,score in zip(indices.tolist(),labels.tolist(),pred.tolist(),conf.tolist()):
                    path=folders['val'].samples[idx][0]
                    predictions.append({'file':str(path),'expected':classes[target],'predicted':classes[guess],'score':score,'original_proxy_label':bool(val_original[idx])})
        return metrics(matrix,torch.cat(all_conf),torch.cat(all_correct)),metrics(original_matrix),matrix,predictions
    def checkpoint(epoch):
        return {'architecture':'resnet18','classes':classes,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'preprocess':'letterbox224_imagenet','epoch':epoch,'label_quality':'weak_and_silver_not_human_verified','training_config':config}
    for epoch in range(1,args.epochs+1):
        start=time.perf_counter();frozen=epoch<=2
        for param in backbone:param.requires_grad=not frozen
        model.train()
        if frozen:
            for layer in model.modules():
                if isinstance(layer,nn.BatchNorm2d):layer.eval()
        total_loss=0.;seen=0
        for images,labels,indices in loaders['train']:
            images=images.to(args.device,non_blocking=True);labels=labels.to(args.device,non_blocking=True)
            quality=train_quality[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=amp):
                losses=criterion(model(images),labels);loss=(losses*quality).sum()/quality.sum()
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            total_loss+=float(loss.detach())*len(images);seen+=len(images)
        val,proxy,_,_=evaluate();scheduler.step()
        result={'epoch':epoch,'train_loss':total_loss/max(seen,1),'val':val,'original_proxy_val':proxy,
                'seconds':round(time.perf_counter()-start,2),'peak_gpu_mb':torch.cuda.max_memory_allocated(args.device)/1024**2 if amp else 0}
        history.append(result);atomic_json(out/'history.json',history)
        print(json.dumps({k:v for k,v in result.items() if k not in ('val','original_proxy_val')}|{'val_accuracy':val['accuracy'],'val_macro_f1':val['macro_f1_present_classes'],'proxy_accuracy':proxy['accuracy']},ensure_ascii=False),flush=True)
        if val['macro_f1_present_classes']>best:
            best=val['macro_f1_present_classes'];best_epoch=epoch;torch.save(checkpoint(epoch),out/'best.pt')
        if epoch>=8 and epoch-best_epoch>=args.patience:break
    torch.save(checkpoint(epoch),out/'last.pt')
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    val,proxy,matrix,predictions=evaluate()
    atomic_json(out/'evaluation.json',{'best_epoch':best_epoch,'epochs_completed':epoch,'elapsed_seconds':time.perf_counter()-begin,
        'val':val,'original_proxy_val':proxy,'classes':classes,'confusion_matrix':matrix.tolist(),
        'warning':'These are independent-source weak/silver label agreement metrics, not human verified classification accuracy.'})
    atomic_json(out/'val_predictions.json',predictions)
    with (out/'per_class.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['class','val_support','precision','recall','f1'])
        for i,name in enumerate(classes):writer.writerow([name,val['support'][i],val['per_class_precision'][i],val['per_class_recall'][i],val['per_class_f1'][i]])
    errors=sorted((r for r in predictions if r['expected']!=r['predicted']),key=lambda r:-r['score'])[:36]
    sheet=Image.new('RGB',(6*210,max(1,(len(errors)+5)//6)*190),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(errors):
        x=i%6*210;y=i//6*190
        with Image.open(r['file']) as im:thumb=ImageOps.contain(im.convert('RGB'),(190,145))
        sheet.paste(thumb,(x+(210-thumb.width)//2,y+40));draw.text((x+4,y+4),'ref:'+r['expected'],fill='black');draw.text((x+4,y+20),'pred:'+r['predicted'],fill='red')
    sheet.save(out/'errors_top36.png')
    print(json.dumps({'best_epoch':best_epoch,'val_accuracy':val['accuracy'],'val_macro_f1':val['macro_f1_present_classes'],'weights':str(out/'best.pt')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
