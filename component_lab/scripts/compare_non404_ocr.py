"""Small, prediction-only OCR comparison and read-only non404 rule replay."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

LAB=Path(__file__).resolve().parents[1]
OUT=LAB/'outputs/non404_component_ocr_20261018'
PRIOR=LAB/'outputs/multiclass_vs_two_stage_20261017'
sys.dont_write_bytecode=True
sys.path.insert(0,str(LAB/'src'))
sys.path.insert(0,str(LAB/'scripts'))
sys.path.insert(0,str(LAB.parent/'non404/show_up'))
os.environ['YOLO_CONFIG_DIR']=str(LAB/'outputs/ultralytics_settings')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def load_ocr(row,profile):
    from component_lab.ocr_spans import split_ocr_spans
    payload=read(row['ocr_v5_cache']) if profile=='v5_existing' else read(OUT/'ocr'/profile/f"{row['case']}.json")
    if 'span_view' not in payload:
        texts,audit=split_ocr_spans(payload['texts'])
        payload={**payload,'texts':texts,'span_view':audit}
    return payload


def active_profiles():
    amendment=OUT/'protocol_amendment.json'
    return read(amendment)['active_profiles'] if amendment.exists() else read(OUT/'manifest.json')['profiles']


def prepare():
    source=read(LAB/'outputs/fullpage_component_identity_20261017/manifest.json')
    selected=sorted(random.Random(20261018).sample(source['cases'],5),key=lambda r:r['case'])
    inputs=[]
    for row in selected:
        case=row['case'];ocr=LAB/'outputs/unattended_20261017_lunch/span_inputs'/case/'ocr_spans.json'
        detected=PRIOR/'predictions/whole/multiclass'/f'{case}.json'
        inputs.append({'case':case,'image':row['image'],'image_sha256':sha(row['image']),
                       'ocr_v5_cache':str(ocr),'ocr_sha256':sha(ocr),
                       'detector_cache':str(detected),'detector_sha256':sha(detected)})
    config={'selection':'random.Random(20261018).sample(sorted existing dev20), n=5; no GT selection',
            'inputs':inputs,'profiles':['v5_existing','v5_whole','v6_whole','v6_matched'],
            'profiles_note':'v5 mobile versus v6 medium; generation AND size change. ONNX Runtime, not historical Paddle runtime.',
            'whole':'scale1,whole only,det side2048,orientation disabled',
            'matched':'same scale2/1200tiles/160overlap/max12/text_score.35/v4orientation as existing v5 cache',
            'non404_replay':'original GT-free rule-only build_case_prediction; replacement YOLO26n37, not missing historic YOLO11n41+Qwen27B',
            'api_calls':0,'test40_used':False,'default_changed':False,
            'protected':{str(p):sha(p) for p in (LAB/'weights').glob('*.pt')}}
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==config
    save(OUT/'manifest.json',config)
    print('Prepared cases:',','.join(r['case'] for r in inputs),flush=True)


def windows(width,height,matched):
    result=[[0,0,width,height]]
    if matched and max(width,height)>1800:
        def starts(n):return [0] if n<=1200 else sorted(set(list(range(0,n-1200+1,1040))+[n-1200]))
        tiles=[[x,y,min(width,x+1200),min(height,y+1200)] for y in starts(height) for x in starts(width)]
        if len(tiles)<=12:result+=tiles
    return result


def detect():
    from PIL import Image
    import torch
    from ultralytics import YOLO
    training=read(PRIOR/'multi_training.json')
    assert sha(training['checkpoint'])==training['checkpoint_sha256']
    assert torch.cuda.is_available() and '3050' in torch.cuda.get_device_name(0)
    model=YOLO(training['checkpoint']);provenance={}
    for row in read(OUT/'manifest.json')['inputs']:
        target=OUT/'detector'/f"{row['case']}.json"
        if target.exists():
            cached=read(target)
            assert cached['checkpoint_sha256']==training['checkpoint_sha256'] and cached['image_sha256']==row['image_sha256']
            provenance[str(target.relative_to(OUT))]=sha(target)
            continue
        assert sha(row['image'])==row['image_sha256']
        with Image.open(row['image']) as image:
            image=image.convert('RGB');tick=time.perf_counter()
            out=model.predict(image,imgsz=1024,conf=.25,iou=.7,device=0,max_det=1000,verbose=False)[0]
            rows=[{'bbox':list(map(float,b)),'type':out.names[int(c)],'score':float(s)} for b,c,s in zip(out.boxes.xyxy.cpu().tolist(),out.boxes.cls.cpu().tolist(),out.boxes.conf.cpu().tolist())]
            save(target,{'rows':rows,'image_size':list(image.size),'seconds':time.perf_counter()-tick,
                         'checkpoint':training['checkpoint'],'checkpoint_sha256':training['checkpoint_sha256'],
                         'settings':{'imgsz':1024,'conf':.25,'iou':.7,'max_det':1000},'image_sha256':row['image_sha256']})
        provenance[str(target.relative_to(OUT))]=sha(target)
        print(f"YOLO37 substitute {row['case']}: {len(rows)} boxes",flush=True)
    save(OUT/'detector_provenance.json',{'training':training,'files':provenance,'historical_checkpoint':False})


def ocr(profile):
    from PIL import Image
    import numpy as np
    from rapidocr import RapidOCR,OCRVersion,ModelType
    from run_local_ocr import overlap
    cfg=read(OUT/'manifest.json');matched=profile=='v6_matched';v6=profile.startswith('v6')
    params={'Global.model_root_dir':str(LAB/'weights/ocr_rapid'),
            'Global.max_side_len':4096,'Global.text_score':.35,'Global.use_cls':matched,
            'Det.ocr_version':OCRVersion.PPOCRV6 if v6 else OCRVersion.PPOCRV5,
            'Rec.ocr_version':OCRVersion.PPOCRV6 if v6 else OCRVersion.PPOCRV5,
            'Det.model_type':ModelType.MEDIUM if v6 else ModelType.MOBILE,
            'Rec.model_type':ModelType.MEDIUM if v6 else ModelType.MOBILE,
            'Det.limit_side_len':2048,'Det.limit_type':'max',
            'EngineConfig.onnxruntime.intra_op_num_threads':2,
            'EngineConfig.onnxruntime.inter_op_num_threads':1,
            'EngineConfig.onnxruntime.use_cuda':False}
    start=time.perf_counter();engine=RapidOCR(params=params);init=time.perf_counter()-start
    for row in cfg['inputs']:
        case=row['case'];target=OUT/'ocr'/profile/f'{case}.json'
        assert sha(row['image'])==row['image_sha256']
        if target.exists():continue
        tick=time.perf_counter();texts=[]
        with Image.open(row['image']) as im:
            im=im.convert('RGB');width,height=im.size;views=windows(width,height,matched)
            for vid,box in enumerate(views):
                crop=im.crop(box);scale=2 if matched else 1
                scaled=crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.LANCZOS)
                print(f'{profile} {case}: view {vid+1}/{len(views)} {scaled.size}',flush=True)
                output=engine(np.asarray(scaled))
                if output.boxes is None:continue
                for poly,text,score in zip(output.boxes,output.txts,output.scores):
                    polygon=[[max(0,min(width,float(x)/scale+box[0])),max(0,min(height,float(y)/scale+box[1]))] for x,y in poly]
                    xs,ys=zip(*polygon);bbox=[min(xs),min(ys),max(xs),max(ys)]
                    if bbox[2]<=bbox[0] or bbox[3]<=bbox[1]:continue
                    texts.append({'text':str(text),'raw':str(text),'bbox':bbox,'polygon':polygon,'score':float(score),
                                  'source_view':vid,'source_window':box,
                                  'orientation':math.degrees(math.atan2(polygon[1][1]-polygon[0][1],polygon[1][0]-polygon[0][0]))})
        kept=[]
        for text in sorted(texts,key=lambda t:-t['score']):
            if not any(text['text'].strip()==other['text'].strip() and overlap(text['bbox'],other['bbox'])>.5 for other in kept):kept.append(text)
        for i,text in enumerate(kept):text['id']=f'text_{i+1:04d}'
        save(target,{'texts':kept,'seconds':time.perf_counter()-tick,'initialization_seconds':init,'views':views,
                     'profile':profile,'rapidocr_version':importlib.metadata.version('rapidocr'),'image_sha256':row['image_sha256']})
        print(f'{profile} {case}: {len(kept)} texts, {time.perf_counter()-tick:.2f}s',flush=True)
    prefixes=('PP-OCRv6_',) if v6 else ('ch_PP-OCRv5_',)
    save(OUT/f'models_{profile}.json',{str(p):sha(p) for p in (LAB/'weights/ocr_rapid').glob('*') if p.is_file() and (p.name.startswith(prefixes) or p.name.startswith('ch_ppocr_mobile'))})


def assemble(profile=None):
    from evaluate_component_architectures import export_shared
    from contest6.evaluate_q1_end_to_end import build_case_prediction
    cfg=read(OUT/'manifest.json')
    aliases={'box':'ic','r':'resistor','c':'capacitor','l':'inductor','gnd':'gnd','v':'vcc','pin':'vbus_in'}
    frozen={}
    for row in cfg['inputs']:
        case=row['case'];assert sha(row['detector_cache'])==row['detector_sha256']
        cached=read(OUT/'detector'/f'{case}.json');w,h=cached['image_size'];rows=cached['rows']
        for variant in ([profile] if profile else active_profiles()):
            oc=load_ocr(row,variant)
            texts=oc['texts']
            # Both paths consume exactly the same frozen detector predictions and OCR rows.
            current,trace=export_shared(rows,texts,h)
            classes=sorted({aliases.get(r['type'],r['type']) for r in rows})
            coco=[]
            for r in rows:
                x1,y1,x2,y2=r['bbox'];kind=aliases.get(r['type'],r['type'])
                coco.append({'category_id':classes.index(kind)+1,'bbox':[x1,y1,x2-x1,y2-y1],'score':r['score']})
            ocrrows=[{'text':r['raw'],'box':r['bbox'],'score':r.get('score')} for r in texts]
            tick=time.perf_counter()
            legacy,legacy_rows=build_case_prediction(case,h,coco,ocrrows,classes)
            # Only rename the legacy schema field; do not pretend this is the historical official Q123 path.
            adapted={'components':{k:{'type':v['type'],'Name':v['name'],'value':v['value'],'bbox':v['bbox']} for k,v in legacy['components'].items()},'pins':{},'nets':{}}
            target=OUT/'predictions'/variant/f'{case}.json'
            save(target,{'current':current,'non404_rules':adapted,'non404_original':legacy,
                         'trace':trace,'non404_rule_seconds':time.perf_counter()-tick})
            frozen[str(target.relative_to(OUT))]=sha(target)
    save(OUT/('predictions_frozen.json' if profile is None else f'predictions_partial_{profile}.json'),frozen)
    print('Frozen',len(frozen),'paired predictions',flush=True)


def report():
    from component_lab.eval.metric import score_case,ScoreConfig
    cfg=read(OUT/'manifest.json')
    for file,digest in read(OUT/'predictions_frozen.json').items():assert sha(OUT/file)==digest
    for file,digest in cfg['protected'].items():assert sha(file)==digest
    per=[]
    for row in cfg['inputs']:
        case=row['case'];gts=list((LAB/'data/official_contest6/train200').glob(case+'-*.json'));assert len(gts)==1
        gt=read(gts[0]);keys=[k for k in gt['components'] if not k.startswith('∅')]
        for profile in active_profiles():
            pred=read(OUT/'predictions'/profile/f'{case}.json')
            oc=load_ocr(row,profile)
            literals={t['raw'] for t in oc['texts']}
            a=score_case(pred['current'],gt,ScoreConfig())
            am=score_case(pred['current'],gt,ScoreConfig(type_policy='mapped'))
            # mapped policy only for old rule pipeline, which emits canonical aliases.
            b=score_case(pred['non404_rules'],gt,ScoreConfig(type_policy='mapped'))
            per.append({'case':case,'profile':profile,'ocr_rows':len(oc['texts']),'seconds':oc['seconds'],
                        'named_gt':len(keys),'key_literal_hits':sum(k in literals for k in keys),
                        'current':a,'current_mapped_diagnostic':am,'non404_rules_mapped_diagnostic':b})
    summaries={}
    for profile in active_profiles():
        rows=[r for r in per if r['profile']==profile];summary={}
        for method in ['current','current_mapped_diagnostic','non404_rules_mapped_diagnostic']:
            counts=Counter()
            for row in rows:counts.update(row[method]['counts']['component_f1'])
            summary[method]={'macro_f1':sum(r[method]['component_f1'] for r in rows)/len(rows),**dict(counts)}
        summary.update({'key_literal_hits':sum(r['key_literal_hits'] for r in rows),'named_gt':sum(r['named_gt'] for r in rows),
                        'ocr_rows':sum(r['ocr_rows'] for r in rows),'seconds':sum(r['seconds'] for r in rows)})
        summaries[profile]=summary
    save(OUT/'report.json',{'summaries':summaries,'per_case':per,'limitations':cfg})
    text=['# non404 component适配运行与PP-OCRv5/v6五图对照','',
          '原始41类YOLO11n权重不在checkout，不能复现历史44.31%。本次实际运行non404原版GT-free规则component函数，使用现有冻结37类YOLO26n替代。没有运行历史Q123的Qwen27B语义阶段。',
          '', '五图由固定种子从dev20抽取，不读test40、不根据GT选图/参数。OCR为RapidOCR ONNX，v6 medium与旧v5 mobile同时改变版本及容量；不能归因于版本本身。全图模式无方向分类。可选v6 matched多视图扩展因明显耗时瓶颈在读GT前停止，仅完成2/5图，不纳入主指标；原计划和运行时调整见protocol_amendment.json。主对照三种配置均保留全部五图。', '',
          '| OCR配置 | 位号原文命中/GT非匿名 | 当前关联严格macroF1 | TP | 当前mapped诊断F1 | non404旧规则mapped诊断F1 | OCR秒 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for p,s in summaries.items():
        text.append(f"|{p}|{s['key_literal_hits']}/{s['named_gt']}|{s['current']['macro_f1']:.2%}|{s['current']['tp']}|{s['current_mapped_diagnostic']['macro_f1']:.2%}|{s['non404_rules_mapped_diagnostic']['macro_f1']:.2%}|{s['seconds']:.1f}|")
    text+=['','原文命中仅为全页字符串存在性诊断，不证明对应元件附近文字正确，更不是OCR accuracy。未有逐字真值框，不能报告通用OCR准确率。non404映射诊断和当前strict列评分口径不同，不可横向宣称架构胜负。',
           '', '所有OCR配置均使用同一个已有conservative raw substring splitter，再进入关联；不修正字符、不读取GT。预测在读GT前冻结。机器结果不是人工标签。现有模型、标签与主infer未改变，无API调用、无新训练。模型初始化/下载不计入OCR秒；旧缓存计时来自此前运行。v6两个任务曾并行，时延受负载影响，不是独占性能benchmark。',
           '', '复现：使用.venv-yolo运行本脚本prepare、detect；使用.venv-ocr运行ocr --profile v5_whole/v6_whole/v6_matched；再用.venv-yolo运行assemble和report。每阶段指定一次，见manifest.json与日志。',
           '', '本次detect真正运行替代37类YOLO26n，imgsz1024/conf.25/iou.7/max_det1000；不是读取GT框，也不是原41类私训checkpoint。源码non404只读导入，PYTHONDONTWRITEBYTECODE启用。原版规则模块11项测试通过。']
    if (OUT/'paired_changes.json').exists():
        paired=read(OUT/'paired_changes.json')
        text+=['',f"v6相对现有v5缓存：新增{paired['added']}个完整正确项、损失{paired['lost']}个、保留{paired['retained']}个，净增{paired['added']-paired['lost']}个。详情paired_changes.json。不是零回退方案，五图小样不证明普遍优势。"]
    text+=['','结论：v6 medium值得作为候选，五图严格宏F1由13.29%到15.65%，但当前ONNX CPU部署很慢，未直接替换默认。应优先试较轻的v6 small或限制补读区域，再验证完整component净收益与耗时。',
           '', '后端限制：RapidOCR默认检测mean/std为0.5，与PaddleOCR的ImageNet归一化不同；这是RapidAI文档明确说明的默认策略，不能直接认定为配置bug。原non404使用Paddle，故本轮ONNX结果不能当原Paddle v6实测。',
           '', '参考：https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html ；https://github.com/RapidAI/RapidOCRDocs/blob/main/docs/blog/posts/about_model/rapidocr-paddleocr-param-align-plan.md']
    (OUT/'REPORT.md').write_text('\n'.join(text),encoding='utf8')
    # Standalone linked OCR evidence; raw strings remain unchanged.
    links=[]
    for r in cfg['inputs']:
        case=r['case'];tabs=[]
        for p in active_profiles():
            oc=load_ocr(r,p)
            rect=[]
            for t in oc['texts']:
                x1,y1,x2,y2=t['bbox'];rect.append(f'<g><title>{html.escape(t["raw"])} ({t["score"]:.3f})</title><rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#0077cc" stroke-width="1.5"/></g>')
            w,h=read(OUT/'detector'/f'{case}.json')['image_size']
            table=''.join(f'<tr><td>{html.escape(t["raw"])}</td><td>{t["score"]:.3f}</td></tr>' for t in oc['texts'])
            pred=read(OUT/'predictions'/p/f'{case}.json');components=[]
            for method in ['current','non404_rules']:
                boxes=[]
                for key,c in pred[method]['components'].items():
                    x1,y1,x2,y2=c['bbox'];y=h-y2
                    title=html.escape(json.dumps({'key':key,**c},ensure_ascii=False))
                    boxes.append(f'<g><title>{title}</title><rect x="{x1}" y="{y}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#c05010" stroke-width="2"/></g>')
                components.append(f'<details><summary>{method} component框，悬停看key/type/value</summary><svg viewBox="0 0 {w} {h}" width="100%"><image href="{Path(r["image"]).as_uri()}" width="{w}" height="{h}"/>{"".join(boxes)}</svg></details>')
            tabs.append(f'<details open><summary>{p}: {len(oc["texts"])} OCR行</summary><a href="predictions/{p}/{case}.json">完整component JSON</a>{"".join(components)}<svg viewBox="0 0 {w} {h}" width="100%"><image href="{Path(r["image"]).as_uri()}" width="{w}" height="{h}"/>{"".join(rect)}</svg><table>{table}</table></details>')
        (OUT/f'{case}.html').write_text('<meta charset="utf-8"><style>body{font-family:Arial}details{display:inline-block;vertical-align:top;width:48%;margin:.5%}table{font-size:13px}</style>'+''.join(tabs),encoding='utf8')
        links.append(f'<li><a href="{case}.html">{case}: 三种OCR框与原文</a></li>')
    (OUT/'gallery.html').write_text('<meta charset="utf-8"><h1>non404 component与OCR对照</h1><p>原41类权重缺失；本次是替代权重的旧规则适配运行，不是44.31%复现。默认配置未改。</p><ul>'+''.join(links)+'</ul><pre>'+html.escape('\n'.join(text))+'</pre>',encoding='utf8')
    save(OUT/'STATUS.json',{'state':'complete','cases':len(cfg['inputs']),'api_calls':0,'default_changed':False})
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','detect','ocr','assemble','report']);p.add_argument('--profile',choices=['v5_whole','v6_whole','v6_matched']);args=p.parse_args()
    if args.stage=='ocr':
        if not args.profile:p.error('ocr requires --profile')
        ocr(args.profile)
    elif args.stage=='assemble':assemble(args.profile)
    else:globals()[args.stage]()
