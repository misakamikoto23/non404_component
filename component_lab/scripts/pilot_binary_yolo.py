"""Isolated RGB/binary YOLO26n experiment, restricted to the RTX 3050 Ti."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/binary_yolo_3050_20261017'
DATA=ROOT.parent/'Data/yolo_experiments/clean_binary_20261017'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'outputs/ultralytics_settings')
os.environ['WANDB_DISABLED']='true'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from component_lab.candidate_verifier import atomic_json
from component_lab.eval.coords import flip_bbox
from prepare_official_gold_crops import validate_split,pixel_sha256,archive_members


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_snapshot():return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'weights').glob('*.pt'))}


def assert_originals():
    previous=read(OUT/'protected_weights.json')
    for name,digest in previous.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Protected model changed: '+name)


def binary_view(image):
    import cv2
    import numpy as np
    gray=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2GRAY)
    threshold,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    from PIL import Image
    return Image.fromarray(binary).convert('RGB'),float(threshold)


def prepare():
    from PIL import Image,ImageDraw,ImageOps
    if OUT.exists() or DATA.exists():raise FileExistsError('Use fresh experiment paths; do not overwrite')
    split=read(ROOT/'configs/contest6_group_split_v1.json')
    check=validate_split(split,read(ROOT/split['groups_file']),read(ROOT/split['forbidden_manifest']))
    initial=ROOT/'weights/yolo26n.pt'
    if not initial.exists():raise FileNotFoundError('Generic YOLO26n weights required; do not substitute a task checkpoint')
    OUT.mkdir(parents=True);DATA.mkdir(parents=True)
    atomic_json(OUT/'protected_weights.json',weights_snapshot())
    config={'initial_model':str(initial),'initial_sha256':sha(initial),'imgsz':1280,'batch':1,'epochs':8,'nbs':16,
            'optimizer':'AdamW','lr0':.001,'lrf':.1,'seed':42,'amp':False,'workers':0,
            'mosaic':0,'hsv_h':0,'hsv_s':0,'hsv_v':0,'device_restriction':'RTX 3050 Ti only',
            'preprocess':'native RGB -> cv2 RGB2GRAY -> global Otsu threshold -> RGB3 replication; no resizing/geometric label changes',
            'dataset':'official train140/dev20 clean design-group split, all official component boxes single class',
            'test_used':False,'existing_task_weights_used_for_initialization':False}
    atomic_json(OUT/'CONFIG.json',config)
    sources=[];skipped=[];counts=Counter();previews=[]
    rows=[(partition,row) for partition in ['train','dev'] for row in split['partitions'][partition]]
    with zipfile.ZipFile(ROOT.parent/'赛题六公开数据集.zip') as archive:
        members=archive_members(archive,[r for _,r in rows])
        for partition,row in rows:
            image_member,label_member=members[row['stem']]
            with Image.open(io.BytesIO(archive.read(image_member))) as loaded:image=loaded.convert('RGB')
            if pixel_sha256(image)!=row['pixel_sha256']:raise ValueError('Frozen source mismatch')
            gt=json.loads(archive.read(label_member).decode('utf-8-sig'));width,height=image.size
            lines=[]
            for key,c in gt['components'].items():
                box=c.get('bbox')
                if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box):
                    skipped.append({'case':row['case'],'key':key,'reason':'invalid_bbox'});continue
                x1,y1,x2,y2=flip_bbox(box,height)
                x1,y1,x2,y2=max(0,x1),max(0,y1),min(width,x2),min(height,y2)
                if x2<=x1 or y2<=y1:
                    skipped.append({'case':row['case'],'key':key,'reason':'nonpositive_or_outside'});continue
                lines.append(f'0 {(x1+x2)/2/width:.9f} {(y1+y2)/2/height:.9f} {(x2-x1)/width:.9f} {(y2-y1)/height:.9f}')
            binary,threshold=binary_view(image);export='train' if partition=='train' else 'val'
            for mode,view in [('rgb',image),('binary',binary)]:
                directory=DATA/mode
                (directory/'images'/export).mkdir(parents=True,exist_ok=True);(directory/'labels'/export).mkdir(parents=True,exist_ok=True)
                view.save(directory/'images'/export/f"{row['case']}.png")
                (directory/'labels'/export/f"{row['case']}.txt").write_text('\n'.join(lines)+'\n',encoding='utf-8')
            counts[export+'_images']+=1;counts[export+'_boxes']+=len(lines)
            sources.append({'case':row['case'],'partition':partition,'group':row['group'],'pixel_sha256':row['pixel_sha256'],
                            'width':width,'height':height,'boxes':len(lines),'otsu_threshold':threshold,'image_member':image_member,
                            'labels_identical':sha(DATA/'rgb/labels'/export/f"{row['case']}.txt")==sha(DATA/'binary/labels'/export/f"{row['case']}.txt")})
            # Previews selected from TRAIN only; no selection by validation scores.
            if partition=='train' and len(previews)<6:
                canvas=Image.new('RGB',(1200,500),'#eee');draw=ImageDraw.Draw(canvas)
                for index,view in enumerate([image,binary]):
                    thumb=ImageOps.contain(view,(590,460));canvas.paste(thumb,(index*600+(600-thumb.width)//2,35))
                    draw.text((index*600+8,8),row['case']+(' RGB' if index==0 else ' grayscale+Otsu'),fill='black')
                path=OUT/f"train_preview_{row['case']}.png";canvas.save(path);previews.append(path.name)
    for mode in ['rgb','binary']:
        directory=DATA/mode
        (directory/'dataset.yaml').write_text('path: '+directory.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: component\n',encoding='utf-8')
    atomic_json(OUT/'dataset_manifest.json',{'counts':dict(counts),'sources':sources,'skipped':skipped,'split_validation':check,
                                          'test_images_or_labels_opened':0,'previews':previews})
    atomic_json(OUT/'STATUS.json',{'state':'prepared','time':datetime.now(timezone.utc).isoformat(),'config':config})
    (OUT/'PLAN.md').write_text('''# RGB与二值化YOLO26n受控小样\n\n用户要求：仅3050 Ti，现有业务权重不受影响，另起模型。两组都从通用yolo26n.pt独立初始化，输出只写本实验runs；全部既有weights/*.pt逐一SHA256保护，不替换默认模型。\n\n使用官方train140/dev20的既有设计组划分，排除dev/test设计组进入训练；不打开test40图片或标签。原GT保持只读，转换为单类YOLO框，两组标签逐字节一致。该框协议存在公开数据局限，结果不能直接与旧PCBnet30图86.20%mAP50比较。\n\n先1280/batch1/AMP关闭的7张训练图显存探针，随后同条件RGB与灰度+Otsu二值源各8epoch。nbs16、AdamW lr0=.001、seed42、无HSV/mosaic/flip。几何变换相同；letterbox/缩放可能产生灰边，称二值源训练而不是网络每层输入严格0/255。标准三通道复制，二值化不会天然降低显存。\n\n若OOM停止，不切换4060 Ti，不自动降低条件后只训练一组。记录显存、时间、loss与mAP；单epoch超过5分钟或GPU总空闲不足时停止新实验。8epoch是可行性小样，未收敛不宣称最终性能。\n''',encoding='utf-8')
    print(json.dumps({'prepared':dict(counts),'skipped':len(skipped),'protected_weights':len(weights_snapshot())}),flush=True)


def gpu_device():
    import torch
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    matches=[i for i,name in enumerate(names) if '3050 Ti' in name]
    if len(matches)!=1:raise RuntimeError('Require exactly one RTX 3050 Ti; refusing another GPU')
    device=matches[0];torch.cuda.set_device(device)
    return torch,device,names[device]


def train(mode,probe=False):
    assert_originals();config=read(OUT/'CONFIG.json');torch,device,name=gpu_device()
    from ultralytics import YOLO
    torch.set_num_threads(4)
    label='memory_probe' if probe else mode
    directory=OUT/'runs'/label
    if directory.exists():raise FileExistsError('Run already exists; do not silently resume/repeat')
    model=YOLO(config['initial_model']);epochs=[]
    start=time.perf_counter();epoch_start=[start]
    atomic_json(OUT/'STATUS.json',{'state':'memory_probe_running' if probe else 'training','mode':label,
                                 'gpu':name,'epochs_requested':1 if probe else config['epochs']})
    def begin(trainer):epoch_start[0]=time.perf_counter()
    def progress(trainer):
        # Ultralytics also invokes this hook after final best-checkpoint validation.
        if trainer.epoch >= trainer.epochs:return
        row={'epoch':trainer.epoch+1,'seconds':round(time.perf_counter()-epoch_start[0],3),
             'allocated_peak_mib':round(torch.cuda.max_memory_allocated(device)/1024**2,1),
             'reserved_peak_mib':round(torch.cuda.max_memory_reserved(device)/1024**2,1),
             'metrics':{k:float(v) for k,v in trainer.metrics.items()}}
        epochs.append(row);atomic_json(OUT/(label+'_progress.json'),{'state':'training','gpu':name,'epochs':epochs})
        print('EPOCH_PROGRESS '+json.dumps(row),flush=True)
        if row['seconds']>300:trainer.stop=True
    model.add_callback('on_train_epoch_start',begin);model.add_callback('on_fit_epoch_end',progress)
    try:
        metrics=model.train(data=str(DATA/('rgb' if probe else mode)/'dataset.yaml'),epochs=1 if probe else config['epochs'],
            fraction=.05 if probe else 1.,imgsz=config['imgsz'],batch=1,device=str(device),workers=0,
            amp=False,cache=False,project=str(OUT/'runs'),name=label,exist_ok=False,
            optimizer='AdamW',lr0=.001,lrf=.1,weight_decay=.0005,warmup_epochs=1,warmup_bias_lr=.001,
            seed=42,deterministic=True,nbs=16,patience=8,plots=False,save=True,save_period=-1,
            hsv_h=0,hsv_s=0,hsv_v=0,translate=.05,scale=.2,degrees=0,shear=0,perspective=0,
            flipud=0,fliplr=0,mosaic=0,close_mosaic=0,mixup=0,copy_paste=0,max_det=1000)
        result={'state':'complete','mode':label,'gpu':name,'logical_device':device,'seconds':round(time.perf_counter()-start,3),
                'epochs':epochs,'validation':{k:float(v) for k,v in metrics.results_dict.items()},
                'best':str(directory/'weights/best.pt'),'last':str(directory/'weights/last.pt'),
                'initial_model_sha256':config['initial_sha256'],'protected_weights_unchanged':True,'test_used':False}
        assert_originals();atomic_json(OUT/(label+'_result.json'),result)
        atomic_json(OUT/'STATUS.json',{'state':'memory_probe_complete' if probe else 'arm_complete','mode':label,
                                     'gpu':name,'epochs_completed':len(epochs),'validation':result['validation']})
        print('TRAIN_RESULT '+json.dumps(result),flush=True)
    except Exception as error:
        assert_originals();atomic_json(OUT/(label+'_result.json'),{'state':'failed','error_type':type(error).__name__,
                    'message':str(error),'gpu':name,'epochs':epochs,'protected_weights_unchanged':True})
        raise


def evaluate_views():
    """Separate removal of color, thresholding, and learned domain adaptation."""
    from PIL import Image,ImageOps
    from ultralytics import YOLO
    assert_originals();_,device,name=gpu_device()
    gray=DATA/'gray_val';(gray/'images/val').mkdir(parents=True,exist_ok=True);(gray/'labels/val').mkdir(parents=True,exist_ok=True)
    for source in sorted((DATA/'rgb/images/val').glob('*.png')):
        with Image.open(source) as image:ImageOps.grayscale(image).convert('RGB').save(gray/'images/val'/source.name)
        (gray/'labels/val'/source.with_suffix('.txt').name).write_bytes((DATA/'rgb/labels/val'/source.with_suffix('.txt').name).read_bytes())
    (gray/'dataset.yaml').write_text('path: '+gray.as_posix()+'\ntrain: '+(DATA/'rgb/images/train').as_posix()+'\nval: images/val\nnames:\n  0: component\n',encoding='utf-8')
    results=[]
    for trained,view in [('rgb','rgb'),('rgb','gray_val'),('rgb','binary'),('binary','binary'),('binary','rgb')]:
        label=f'{trained}_on_{view}';target=OUT/'evaluations'/label/'metrics.json'
        if target.exists():results.append(read(target));continue
        summary=read(OUT/(trained+'_result.json'))
        if summary['state']!='complete' or len(summary['epochs'])!=read(OUT/'CONFIG.json')['epochs']:
            raise ValueError('Need both complete matched runs before comparison')
        model=YOLO(summary['best']);start=time.perf_counter()
        metrics=model.val(data=str(DATA/view/'dataset.yaml'),split='val',device=str(device),imgsz=1280,batch=1,
                          workers=0,plots=False,max_det=1000,project=str(OUT/'evaluations'),name=label,exist_ok=False,
                          save_json=False,save_txt=False,verbose=False)
        row={'trained':trained,'input_view':view,'gpu':name,'seconds':round(time.perf_counter()-start,3),
             'metrics':{k:float(v) for k,v in metrics.results_dict.items()},'checkpoint':summary['best']}
        atomic_json(target,row);results.append(row)
    assert_originals();atomic_json(OUT/'view_comparison.json',results)


def report():
    import html
    assert_originals();config=read(OUT/'CONFIG.json');dataset=read(OUT/'dataset_manifest.json')
    rows={mode:read(OUT/(mode+'_result.json')) for mode in ['rgb','binary'] if (OUT/(mode+'_result.json')).exists()}
    comparison=read(OUT/'view_comparison.json') if (OUT/'view_comparison.json').exists() else []
    lines=['# 3050 Ti独立RGB/二值化YOLO26n小样','',
           '全部既有业务权重SHA256保持一致，默认模型未替换。只用3050 Ti，RGB和二值源两组从同一个通用预训练文件各自初始化。',
           '',f"数据：train {dataset['counts']['train_images']}图/{dataset['counts']['train_boxes']}框；dev {dataset['counts']['val_images']}图/{dataset['counts']['val_boxes']}框。分组去重检查通过，test40未打开。",'',
           '两组均1280、batch1、nbs16、AdamW lr0=.001、8epoch、seed42、AMP关闭；二值源为灰度+全局Otsu，并复制到三通道。几何坐标与标签一致，原彩色图保留用于后续OCR/ResNet。', '',
           '|训练组|实际轮数|P|R|mAP50|mAP50–95|峰值分配MiB|总秒数|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,row in rows.items():
        if row['state']!='complete':lines.append(f'|{mode}|失败|||||||');continue
        matched=next((r for r in comparison if r['trained']==mode and r['input_view']==mode),None)
        m=matched['metrics'] if matched else row['validation'];peak=max(e['allocated_peak_mib'] for e in row['epochs'])
        lines.append(f"|{mode}|{len(row['epochs'])}|{m['metrics/precision(B)']:.2%}|{m['metrics/recall(B)']:.2%}|{m['metrics/mAP50(B)']:.2%}|{m['metrics/mAP50-95(B)']:.2%}|{peak:.1f}|{row['seconds']:.1f}|")
    if comparison:
        lines += ['', '## 同权重跨输入视图对照','', '|训练组|验证输入|mAP50|mAP50–95|','|---|---|---:|---:|']
        for row in comparison:
            m=row['metrics'];lines.append(f"|{row['trained']}|{row['input_view']}|{m['metrics/mAP50(B)']:.2%}|{m['metrics/mAP50-95(B)']:.2%}|")
    lines += ['', '## 口径与限制', '',
              '- 完成跨视图对照后，上表优先采用统一batch1重评结果。P/R是Ultralytics验证曲线选择的工作点，不是固定conf=0.25的计数。',
              '- 仅8epoch的小样；单seed、不同于旧PCBnet30图和审核框协议。不能直接比较旧86.20%mAP50，也不是严格ComponentF1。',
              '- 两组头部均为单类component；不涉及ResNet/OCR/API、标签修订或已有模型续训。',
              '- 使用相同dev规则挑选best checkpoint，训练曲线与last checkpoint保留；未评估test。',
              '- 二值化可能删除浅色细节和文字，不能仅凭视觉变干净判断有效。三通道二值源不天然降低显存；缩放可产生灰边。',
              '- 显存探针实际训练1轮；最初探针记录还含最终best验证的额外回调，主实验已排除此条，不把它当第2轮。', '',
              '复现脚本：scripts/pilot_binary_yolo.py；配置、源图/标签哈希与保护权重哈希均在本目录。日志rgb_training.log、binary_training.log；全部新权重在runs/rgb及runs/binary内，未复制到原weights目录。']
    result={'kind':'clean_grouped_RGB_vs_Otsu_pilot_single_seed','models':rows,'cross_views':comparison,
            'protected_weights_unchanged':True,'original_default_replaced':False,'test_used':False,'config':config}
    primary={r['trained']:r for r in comparison if r['trained']==r['input_view']}
    result['primary_matched_evaluation']=primary
    if set(primary)=={'rgb','binary'}:
        delta=100*(primary['binary']['metrics']['metrics/mAP50(B)']-primary['rgb']['metrics']['metrics/mAP50(B)'])
        lines.insert(2,f'结论：本次8epoch单seed小样，Otsu二值源mAP50比RGB低{abs(delta):.2f}个百分点。尚无替换原模型的依据，不追加训练或修改主模型。' if delta<0 else f'本次二值源mAP50差值为{delta:+.2f}个百分点；仍属小样，不自动替换主模型。')
        result['binary_minus_rgb_mAP50_pp']=delta
    if len(rows)==2 and all(r['state']=='complete' for r in rows.values()):
        os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_config')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for mode,row in rows.items():
            for ax,key,title in zip(axes,['metrics/mAP50(B)','metrics/mAP50-95(B)'],['mAP50','mAP50-95']):
                ax.plot([e['epoch'] for e in row['epochs']],[100*e['metrics'][key] for e in row['epochs']],marker='o',label=mode)
                ax.set(xlabel='Epoch',ylabel='Percent',title=title);ax.grid(alpha=.25);ax.legend()
        fig.suptitle('YOLO26n / 3050 Ti / clean train140-dev20 / single seed')
        fig.savefig(OUT/'training_curves.png',dpi=150);plt.close(fig)
    atomic_json(OUT/'report.json',result);(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    previews=('<h2>Training curves</h2><img src="training_curves.png" style="width:100%">' if (OUT/'training_curves.png').exists() else '')
    previews+=''.join('<h2>'+name+'</h2><img src="'+name+'" style="width:100%">' for name in dataset['previews'])
    (OUT/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>3050 Ti binary YOLO pilot</title><style>body{font-family:system-ui;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap}</style><h1>独立二值化YOLO26n试验</h1><p><a href="REPORT.md">报告</a> · <a href="report.json">指标</a> · <a href="STATUS.json">状态</a></p><pre>'+html.escape('\n'.join(lines))+'</pre>'+previews,encoding='utf-8')
    print(json.dumps({'models':{k:v.get('validation') for k,v in rows.items()},'views':len(comparison),'protected_weights_unchanged':True}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','probe','train-rgb','train-binary','evaluate','report']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='evaluate':evaluate_views()
    elif a.action=='report':report()
    else:train('binary' if a.action=='train-binary' else 'rgb',a.action=='probe')


if __name__=='__main__':main()
