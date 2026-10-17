"""Prediction-only local spectrum proposals; cache coverage is a diagnostic."""
from __future__ import annotations
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB/'src'))
sys.path.insert(0,str(LAB/'scripts'))
from compare_non404_ocr import load_ocr
from component_lab.text_roles import REFERENCE

OLD=LAB/'outputs/non404_component_ocr_20261018'
OUT=LAB/'outputs/frequency_ocr_regions_20261018'
CELL=128
PAD=24
BUDGET=.20
MAX_ROIS=8


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def inter(a,b):return area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
def rank(values):
    vals=np.asarray(values);return np.searchsorted(np.sort(vals),vals,side='right')/max(1,len(vals))


def extract(image,need_boxes):
    gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
    h,w=gray.shape;ink=(gray<200).astype(np.uint8)
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,48),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((48,1),np.uint8))
    lines=np.maximum(horizontal,vertical);residual=ink*(1-lines)
    _,labels,stats,centers=cv2.connectedComponentsWithStats(residual,8)
    good=(stats[:,cv2.CC_STAT_HEIGHT]>=3)&(stats[:,cv2.CC_STAT_HEIGHT]<=40)&(stats[:,cv2.CC_STAT_WIDTH]>=1)&(stats[:,cv2.CC_STAT_WIDTH]<=60)&(stats[:,cv2.CC_STAT_AREA]>=3)&(stats[:,cv2.CC_STAT_AREA]<=1800)
    good[0]=False;centers=centers[good]
    fy,fx=np.meshgrid(np.fft.fftfreq(CELL),np.fft.fftfreq(CELL),indexing='ij')
    radius=np.sqrt(fx*fx+fy*fy);band=(radius>=.10)&(radius<=.45)
    offaxis=band&(np.abs(fx)>2/CELL)&(np.abs(fy)>2/CELL)
    hann=np.outer(np.hanning(CELL),np.hanning(CELL));features=[]
    for y in range(0,h,CELL):
        for x in range(0,w,CELL):
            x2,y2=min(w,x+CELL),min(h,y+CELL);patch=gray[y:y2,x:x2].astype(np.float32)/255
            patch=cv2.copyMakeBorder(patch,0,CELL-patch.shape[0],0,CELL-patch.shape[1],cv2.BORDER_REFLECT_101)
            spectrum=np.abs(np.fft.fft2((patch-patch.mean())*hann))**2
            energy=float(spectrum[band].sum()/CELL**4)
            diagonal=float(spectrum[offaxis].sum()/CELL**4)
            n=int(((centers[:,0]>=x)&(centers[:,0]<x2)&(centers[:,1]>=y)&(centers[:,1]<y2)).sum())
            cx,cy=(x+x2)/2,(y+y2)/2
            distance=min((math.hypot(max(b[0]-cx,cx-b[2],0),max(b[1]-cy,cy-b[3],0)) for b in need_boxes),default=1e6)
            features.append({'box':[x,y,x2,y2],'energy':energy,'offaxis_energy':diagonal,'character_cc_count':n,
                             'line_fraction':float(lines[y:y2,x:x2].mean()),'ink_fraction':float(ink[y:y2,x:x2].mean()),
                             'near_unresolved':max(0,1-distance/128)})
    hf=rank([r['energy'] for r in features]);diag=rank([r['offaxis_energy'] for r in features]);cc=rank([r['character_cc_count'] for r in features])
    for i,r in enumerate(features):
        r['frequency_score']=float(hf[i])
        r['hybrid_score']=float(.30*diag[i]+.45*cc[i]+.15*r['near_unresolved']-.10*min(1,r['line_fraction']/.1))
    return features


def proposals(features,texts,size,kind):
    w,h=size;result=[];cost=0
    for f in sorted(features,key=lambda r:-r[kind+'_score']):
        if f['ink_fraction']<.003 or (kind=='hybrid' and f['character_cc_count']==0):continue
        x1,y1,x2,y2=f['box'];box=[max(0,x1-PAD),max(0,y1-PAD),min(w,x2+PAD),min(h,y2+PAD)]
        # Expand to include known crossing text. This cannot protect unseen words.
        for _ in range(3):
            before=box[:]
            for t in texts:
                b=t['bbox']
                if inter(box,b)/max(1,area(b))>.2:
                    extended=[max(0,min(box[0],math.floor(b[0])-4)),max(0,min(box[1],math.floor(b[1])-4)),
                              min(w,max(box[2],math.ceil(b[2])+4)),min(h,max(box[3],math.ceil(b[3])+4))]
                    if max(extended[2]-extended[0],extended[3]-extended[1])<=384:box=extended
            if box==before:break
        if any(inter(box,r['box'])/max(1,min(area(box),area(r['box'])))>.5 for r in result):continue
        if cost+area(box)>BUDGET*w*h:continue
        result.append({'box':box,'source_cell':f['box'],'score':f[kind+'_score']});cost+=area(box)
        if len(result)==MAX_ROIS:break
    return result


def generate():
    manifest=read(OLD/'manifest.json');files={};cases=[]
    for row in manifest['inputs']:
        case=row['case'];assert sha(row['image'])==row['image_sha256']
        old=read(OLD/'predictions/v5_existing'/f'{case}.json')
        detector=read(OLD/'detector'/f'{case}.json');ocr=load_ocr(row,'v5_existing')
        needs=[]
        for trace,component in zip(old['trace'],detector['rows']):
            rec=old['current']['components'][trace['key']]
            if trace['reference_raw'] is None or (rec['type'] in {'r','c'} and rec['value'] is None) or (rec['type']=='box' and rec['Name'] is None):needs.append(component['bbox'])
        with Image.open(row['image']) as image:
            image=image.convert('RGB');tick=time.perf_counter();features=extract(image,needs)
            result={'case':case,'image':row['image'],'image_sha256':row['image_sha256'],'size':list(image.size),
                    'settings':{'cell':CELL,'padding':PAD,'sum_crop_area_budget':BUDGET,'max_rois':MAX_ROIS},
                    'features':features,'regions':{kind:proposals(features,ocr['texts'],image.size,kind) for kind in ['frequency','hybrid']},
                    'selector_seconds':time.perf_counter()-tick,'uses_GT':False,'uses_v6':False}
        target=OUT/'proposals'/f'{case}.json';save(target,result);files[str(target.relative_to(OUT))]=sha(target);cases.append(case)
        print(case,'selected',[(k,len(v)) for k,v in result['regions'].items()],f"{result['selector_seconds']:.2f}s",flush=True)
    save(OUT/'proposals_frozen.json',{'files':files,'cases':cases,'no_GT_or_v6_in_selection':True})


def det_pixels(w,h,scale):
    ratio=min(scale,2048/max(w,h));return max(32,round(w*ratio/32)*32)*max(32,round(h*ratio/32)*32)


def evaluate():
    from component_lab.eval.metric import normalize_case,bbox_ok,text_ok,ScoreConfig
    frozen=read(OUT/'proposals_frozen.json')
    for file,digest in frozen['files'].items():assert sha(OUT/file)==digest
    report=[];audit=Counter();combos=Counter();errors=[]
    for case in frozen['cases']:
        data=read(OUT/'proposals'/f'{case}.json');w,h=data['size']
        inputs=next(r for r in read(OLD/'manifest.json')['inputs'] if r['case']==case)
        old=load_ocr(inputs,'v5_existing')['texts'];new=load_ocr(inputs,'v6_whole')['texts']
        newref=[t for t in new if REFERENCE.fullmatch(t['raw'].strip()) and not any(t['raw']==a['raw'] and inter(t['bbox'],a['bbox'])/max(1,min(area(t['bbox']),area(a['bbox'])))>.3 for a in old)]
        for kind,regions in data['regions'].items():
            boxes=[r['box'] for r in regions]
            # Full-instance coverage, not center coverage; text cut by boundary is not counted.
            covered=lambda t:any(inter(t['bbox'],b)/max(1e-9,area(t['bbox']))>=.9 for b in boxes)
            report.append({'case':case,'kind':kind,'rois':len(boxes),'crop_area_fraction':sum(map(area,boxes))/(w*h),
                           'v6_cached_texts_covered':sum(map(covered,new)),'v6_cached_texts':len(new),
                           'v6_new_reference_candidates_covered':sum(map(covered,newref)),'v6_new_reference_candidates':len(newref),
                           'detector_pixels_ratio_1x':sum(det_pixels(b[2]-b[0],b[3]-b[1],1) for b in boxes)/det_pixels(w,h,1),
                           'detector_pixels_ratio_2x':sum(det_pixels(b[2]-b[0],b[3]-b[1],2) for b in boxes)/det_pixels(w,h,1),
                           'selector_seconds':data['selector_seconds']})
        # Evaluation-only attribution of the previous frozen whole-image result.
        gtfile=next((LAB/'data/official_contest6/train200').glob(case+'-*.json'))
        gt=normalize_case(read(gtfile))['components'];pred=normalize_case(read(OLD/'predictions/v6_whole'/f'{case}.json')['current'])['components']
        literal={t['raw'] for t in new};dets=read(OLD/'detector'/f'{case}.json')['rows']
        for key,target in gt.items():
            if key.startswith('∅'):audit['anonymous_gt']+=1;continue
            audit['named_gt']+=1
            if key not in pred:
                bucket='no_exact_key_in_cache' if key not in literal else 'literal_present_but_key_not_exported'
                audit[bucket]+=1
                feasible=any(d['type']==target[0] and bbox_ok((d['bbox'][0],h-d['bbox'][3],d['bbox'][2],h-d['bbox'][1]),target[3],ScoreConfig()) for d in dets)
                audit[bucket+('_has_typed_box' if feasible else '_no_typed_box')]+=1
            else:
                p=pred[key];flags={'type':p[0]==target[0],'bbox':bbox_ok(p[3],target[3],ScoreConfig()),'text':text_ok(p,target)}
                if all(flags.values()):audit['complete_tp']+=1
                else:
                    audit['key_hit_other_field_failed']+=1;bad='+'.join(k for k,v in flags.items() if not v);combos[bad]+=1
                    errors.append({'case':case,'key':key,'failures':bad,'pred':p,'gt':target})
    totals={}
    for kind in ['frequency','hybrid']:
        rows=[r for r in report if r['kind']==kind]
        totals[kind]={'mean_area_fraction':float(np.mean([r['crop_area_fraction'] for r in rows])),
                      'mean_detector_pixels_ratio_1x':float(np.mean([r['detector_pixels_ratio_1x'] for r in rows])),
                      'mean_detector_pixels_ratio_2x':float(np.mean([r['detector_pixels_ratio_2x'] for r in rows])),
                      'machine_new_reference_covered':sum(r['v6_new_reference_candidates_covered'] for r in rows),
                      'machine_new_reference_total':sum(r['v6_new_reference_candidates'] for r in rows),
                      'regions':sum(r['rois'] for r in rows)}
    save(OUT/'report.json',{'totals':totals,'per_case':report,'previous_frozen_error_audit':dict(audit),
                          'key_hit_failure_combinations':dict(combos),'errors':errors,
                          'actual_v6_ROI_inference_run':False,'accuracy_or_speedup_claim':False})
    print(json.dumps({'totals':totals,'audit':dict(audit),'combos':dict(combos)},ensure_ascii=False,indent=2))


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    result=read(OUT/'report.json');links=[]
    for case in read(OUT/'proposals_frozen.json')['cases']:
        data=read(OUT/'proposals'/f'{case}.json');w,h=data['size']
        with Image.open(data['image']) as im:image=np.asarray(im.convert('RGB'))
        grid=np.zeros((math.ceil(h/CELL),math.ceil(w/CELL)),dtype=float)
        for f in data['features']:grid[f['box'][1]//CELL,f['box'][0]//CELL]=math.log10(f['energy']+1e-10)
        fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained')
        axes[0].imshow(grid,cmap='inferno',extent=[0,w,h,0],interpolation='nearest');axes[0].set_title('Local FFT band energy (log)')
        for ax,kind in zip(axes[1:],['frequency','hybrid']):
            ax.imshow(image);cost=0
            for r in data['regions'][kind]:
                x1,y1,x2,y2=r['box'];cost+=area(r['box']);ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#ff3030',linewidth=1.7))
            ax.set_title(f'{kind}: {len(data["regions"][kind])} ROIs, {cost/(w*h):.1%} crop area')
        for ax in axes:ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
        fig.savefig(OUT/f'{case}.png',dpi=130);plt.close(fig)
        links.append(f'<h2>{case}</h2><a href="{case}.png"><img width="100%" src="{case}.png"></a>')
    text=['# 局部频域筛选OCR候选区：五图探针','',
          '仅生成候选、离线检查上一轮v6机器OCR框覆盖；本轮没有执行ROI v6识别，不能报告OCR精度提升或实际加速。固定两种方案ROI均在读取v6缓存与GT前生成并hash冻结，参数未按本轮结果再调。',
          '', '|选择器|平均原图裁剪面积|v6新增位号样式候选覆盖（机器代理）|ROI总数|检测输入像素比1x/2x|', '|---|---:|---:|---:|---:|']
    for k,s in result['totals'].items():text.append(f"|{k}|{s['mean_area_fraction']:.1%}|{s['machine_new_reference_covered']}/{s['machine_new_reference_total']}|{s['regions']}|{s['mean_detector_pixels_ratio_1x']:.2f}/{s['mean_detector_pixels_ratio_2x']:.2f}|")
    text+=['','以上新增候选可能包含v6误读，并非真实新增位号真值。面积/输入像素只是工作量代理，检测器会缩图且识别器成本取决于文字数量与长度；不能把面积比例当作耗时比例。',
           '', '上一轮v6严格component未成功项归因（仅原五图）：',json.dumps(result['previous_frozen_error_audit'],ensure_ascii=False,indent=2),
           '', '命中key后失败组合：',json.dumps(result['key_hit_failure_combinations'],ensure_ascii=False,indent=2),
           '', '全页存在某个key字面不证明对应元件旁读对了，也不证明仅关联错误；表中进一步区分是否存在type与20px框均可匹配的检测候选。匿名ID不可从静态图直接推断。',
           '', '已核查两项回退：0107/C8由C8读成=C8，未生成合法reference候选，22uF实际仍被读到；R6读成Rô，原3.3kΩ仍在，但框被错配D22。这些不是模型容量能独自解决的。',
           '', '建议：局部频域只作proposal特征；结合长线抑制、类字符连通域、已有OCR不确定性和元件邻域。保留低对比/模糊/竖排的独立补读入口，因为模糊会削弱高频。不要把滤波残差直接送OCR；裁剪原图并保留上下文。先仅补未解决项，默认不覆盖已绑定字段。',
           '', '后续若运行ROI OCR，验收必须包含：实际耗时、检测输入像素、文字/位号实例召回、旧正确项保留、严格ComponentF1。先不用全量重读或训练；本轮默认模型/标签/评分器未改，API0、v6新推理0、test40未读。',
           '', '复现：.venv-yolo/Scripts/python.exe scripts/probe_frequency_ocr_regions.py（生成、冻结、评价、绘图依次执行）。',
           '', '方法参考：https://docs.opencv.org/4.12.0/dd/dd7/tutorial_morph_lines_detection.html 。局部FFT与组合打分为本项目启发式原型，不是现成论文方法的复现。']
    (OUT/'REPORT.md').write_text('\n'.join(text),encoding='utf8')
    (OUT/'gallery.html').write_text('<meta charset="utf-8"><style>body{font:16px Arial;max-width:1500px;margin:30px auto}pre{white-space:pre-wrap}</style><h1>局部频域与OCR选区</h1><p>红框是候选，不是已验证文字框；未执行ROI v6推理。</p>'+''.join(links)+'<pre>'+html.escape('\n'.join(text))+'</pre>',encoding='utf8')
    save(OUT/'STATUS.json',{'state':'complete_proposal_probe','actual_v6_ROI_inference_run':False,'api_calls':0,'default_changed':False})


if __name__=='__main__':
    generate();evaluate();render()
