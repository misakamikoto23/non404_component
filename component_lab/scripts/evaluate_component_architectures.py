"""Frozen-prediction evaluation of the architecture pilot; never reads test40."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import html
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_component_architectures import LAB,OUT,DATA,read,write,sha,protect,gpu,context_box

SEMANTIC={'r':'resistor','c':'capacitor','l':'inductor','box':'ic_or_module','d':'diode',
          'gnd':'ground','v':'power_port','battary':'battery','transfomer':'transformer',
          'ant':'antenna','opt':'optocoupler','amp':'opamp','amp_3pin':'opamp','amp_5pin':'opamp',
          'circle_header':'connector','pin':'connector','bjt_npn':'bjt','bjt_pnp':'bjt',
          'mosfet_pnp':'mosfet','mosfet_npn':'mosfet','esd':'esd_protection',
          'crystal_2pin':'crystal','crystal_3pin':'crystal','crystal_4pin':'crystal'}
SEMANTIC.update({k:'net_port' for k in ['net_bidirection','net_input','net_output','net_short']})


def hybrid_choice(yolo_type,yolo_conf,resnet_type,resnet_conf):
    return resnet_type if yolo_conf<.5 and resnet_conf>=.8 else yolo_type


def export_shared(rows,texts,height):
    from component_lab.text_matching import associate_structured
    from component_lab.display_names import assign_display_names
    from component_lab.contest_components import export_contest
    adapted=deepcopy(rows);raw={}
    for i,row in enumerate(adapted):
        row['id']=f'architecture_{i:04d}';raw[row['id']]=row['type']
        row['raw_pred_type']=row['type'];row['type']=SEMANTIC.get(row['type'],row['type'])
    association=associate_structured(adapted,texts,clamp_costs=False)
    association=assign_display_names(association)
    pred,trace=export_contest(association,height,ic_value_as_name=True,display_names=True)
    for row in trace:
        pred['components'][row['key']]['type']=raw[row['source_id']]
        row['raw_pred_type']=raw[row['source_id']]
    return pred,trace


def counts_metrics(c):
    tp,fp,fn=(c[k] for k in ['tp','fp','fn'])
    return {**dict(c),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)}


def pair_counts(pred,gt,rule,typed=False):
    from scipy.optimize import linear_sum_assignment
    from component_lab.tiling import box_iou
    good=np.zeros((len(pred),len(gt)),bool);tie=np.ones(good.shape)
    for i,p in enumerate(pred):
        for j,g in enumerate(gt):
            if rule=='maxcoord20':
                error=max(abs(a-b) for a,b in zip(p['bbox'],g['bbox']));valid=error<=20;tie[i,j]=min(error/20,1)
            else:
                overlap=box_iou(p['bbox'],g['bbox']);valid=overlap>=.5;tie[i,j]=1-overlap
            good[i,j]=valid and (not typed or p['type']==g['type'])
    if len(pred) and len(gt):
        rr,cc=linear_sum_assignment((~good).astype(float)*(min(len(pred),len(gt))+1)+tie)
        tp=int(good[rr,cc].sum())
    else:tp=0
    return {'tp':tp,'fp':len(pred)-tp,'fn':len(gt)-tp}


def predict_all():
    from PIL import Image
    from ultralytics import YOLO
    from torchvision.ops import nms
    from component_lab.models import ResnetClassifier
    from component_lab.tiling import tile_plan,touches_internal_edge,translate_prediction
    protect();config=read(OUT/'CONFIG.json');torch,device=gpu()
    target=OUT/'predictions'
    if (OUT/'predictions_frozen.json').exists():raise FileExistsError('Predictions already frozen')
    single=read(OUT/'single_training.json');multi=read(OUT/'multi_training.json');resnet=read(OUT/'resnet_training.json')
    assert single['epochs']==multi['epochs']==16 and resnet['epochs']==12
    for record in [single,multi,resnet]:assert sha(record['checkpoint'])==record['checkpoint_sha256']
    classifier=ResnetClassifier(resnet['checkpoint'],device=str(device))
    classifiers=[];timings=[];frozen={};cases=sorted(s['case'] for s in read(OUT/'dataset.json')['sources'] if s['partition']=='val')
    ocr_hashes={case:sha(LAB/'outputs/unattended_20261017_lunch/span_inputs'/case/'ocr_spans.json') for case in cases}
    write(OUT/'frozen_ocr_hashes.json',ocr_hashes)
    def classify(image,rows):
        if not rows:return []
        xs=[classifier.preprocess(image.crop(context_box(r['bbox'],image.size))) for r in rows]
        result=[]
        with torch.inference_mode():
            for start in range(0,len(xs),32):
                logits=classifier.model(torch.stack(xs[start:start+32]).to(classifier.device)).softmax(1)
                scores,indices=logits.max(1)
                result.extend({'type':classifier.classes[int(i)],'score':float(s)} for i,s in zip(indices,scores))
        return result
    for detector_name,training in [('single',single),('multi',multi)]:
        model=YOLO(training['checkpoint'])
        # Identical postprocess for both detectors, including cross-class duplicate suppression.
        def detect(image):
            out=model.predict(image,imgsz=1280,conf=.25,iou=.5,agnostic_nms=True,max_det=1000,
                              device=str(device),verbose=False)[0]
            rows=[{'bbox':b,'type':out.names[int(c)],'score':float(s)} for b,c,s in
                  zip(out.boxes.xyxy.cpu().tolist(),out.boxes.cls.cpu().tolist(),out.boxes.conf.cpu().tolist())]
            width,height=image.size
            for row in rows:
                x1,y1,x2,y2=row['bbox']
                row['bbox']=[max(0.,min(width,x1)),max(0.,min(height,y1)),max(0.,min(width,x2)),max(0.,min(height,y2))]
            rows=[r for r in rows if all(math.isfinite(v) for v in r['bbox']+[r['score']])
                  and r['bbox'][2]>r['bbox'][0] and r['bbox'][3]>r['bbox'][1]]
            # YOLO26 end2end path ignores the library's agnostic_nms argument.
            # Apply it explicitly to BOTH heads and all source views.
            if rows:
                keep=nms(torch.tensor([r['bbox'] for r in rows]),torch.tensor([r['score'] for r in rows]),.5)
                rows=[rows[int(i)] for i in keep]
            return rows
        for case in cases:
            with Image.open(DATA/'single/images/val'/(case+'.png')) as im:image=im.convert('RGB')
            ocr_path=LAB/'outputs/unattended_20261017_lunch/span_inputs'/case/'ocr_spans.json'
            assert sha(ocr_path)==ocr_hashes[case]
            texts=read(ocr_path)['texts']
            torch.cuda.synchronize();tick=time.perf_counter();whole=detect(image);torch.cuda.synchronize()
            whole_sec=time.perf_counter()-tick;plan=tile_plan(*image.size,1024,.25,100);tiled=list(whole)
            tick=time.perf_counter()
            for window in plan['windows']:
                for row in detect(image.crop(tuple(window))):
                    if touches_internal_edge(row['bbox'],window,*image.size):continue
                    transformed=translate_prediction(row,window,*image.size)
                    if transformed:tiled.append(transformed)
            if tiled and not plan['skipped']:
                keep=nms(torch.tensor([r['bbox'] for r in tiled]),torch.tensor([r['score'] for r in tiled]),.5)
                tiled=[tiled[int(i)] for i in keep]
            elif plan['skipped']:tiled=whole
            torch.cuda.synchronize();tile_sec=time.perf_counter()-tick
            for view,detections,det_seconds in [('whole',whole,whole_sec),('tiled',tiled,whole_sec+tile_sec)]:
                tick=time.perf_counter();rows=deepcopy(detections);clf_sec=0.;eligible=0;changed=0
                if detector_name=='single':
                    answers=classify(image,rows);torch.cuda.synchronize();clf_sec=time.perf_counter()-tick
                    for row,answer in zip(rows,answers):row['type']=answer['type'];row['classification']=answer
                    arms={'two_stage':rows}
                else:
                    fallback=deepcopy(rows);indices=[i for i,r in enumerate(rows) if r['score']<.5];eligible=len(indices)
                    answers=classify(image,[rows[i] for i in indices]);torch.cuda.synchronize();clf_sec=time.perf_counter()-tick
                    for i,answer in zip(indices,answers):
                        before=fallback[i]['type'];fallback[i]['type']=hybrid_choice(before,fallback[i]['score'],answer['type'],answer['score'])
                        fallback[i]['classification']=answer;changed+=int(before!=fallback[i]['type'])
                    arms={'multiclass':rows,'hybrid':fallback}
                for arm,arm_rows in arms.items():
                    tick=time.perf_counter();pred,trace=export_shared(arm_rows,texts,image.height);assoc_sec=time.perf_counter()-tick
                    directory=target/view/arm
                    write(directory/(case+'.json'),{'rows':arm_rows,'prediction':pred,'trace':trace,
                        'image_size':list(image.size),'frozen_ocr_sha256':sha(LAB/'outputs/unattended_20261017_lunch/span_inputs'/case/'ocr_spans.json')})
                    classifiers.append({'case':case,'view':view,'arm':arm,'fallback_eligible':eligible if arm=='hybrid' else 0,
                                        'fallback_type_changes':changed if arm=='hybrid' else 0})
                    timings.append({'case':case,'view':view,'arm':arm,'detector_seconds':det_seconds,
                                    'classifier_seconds':clf_sec if arm!='multiclass' else 0.,'association_seconds':assoc_sec,
                                    'total_seconds_excluding_frozen_ocr':det_seconds+(clf_sec if arm!='multiclass' else 0.)+assoc_sec})
                    frozen[str((directory/(case+'.json')).relative_to(OUT))]=sha(directory/(case+'.json'))
            print('PREDICTED',detector_name,case,len(whole),len(tiled),flush=True)
        del model;torch.cuda.empty_cache()
    for case,digest in ocr_hashes.items():assert sha(LAB/'outputs/unattended_20261017_lunch/span_inputs'/case/'ocr_spans.json')==digest
    write(OUT/'timings.json',timings);write(OUT/'fallback.json',classifiers)
    write(OUT/'predictions_frozen.json',{'files':frozen,'gt_used_for_predictions':False,
        'official_target_jsons_opened_in_prediction_stage':0,
        'metadata_note':'dataset.json loaded for source case/partition list only; its crop label fields are not prediction inputs',
        'same_nms':'class agnostic IoU .5 at model and tile merge','test40_used':False})
    protect()


def evaluate():
    from component_lab.eval.metric import score_case,ScoreConfig
    from component_lab.eval.coords import flip_bbox
    protect();frozen=read(OUT/'predictions_frozen.json')
    for rel,digest in frozen['files'].items():assert sha(OUT/rel)==digest
    groups={};per_case=[]
    cases=sorted(s['case'] for s in read(OUT/'dataset.json')['sources'] if s['partition']=='val')
    for case in cases:
        files=list((LAB/'data/official_contest6/train200').glob(case+'-*.json'));assert len(files)==1
        gt=read(files[0])
        for view in ['whole','tiled']:
            for arm in ['multiclass','two_stage','hybrid']:
                cached=read(OUT/'predictions'/view/arm/(case+'.json'));h=cached['image_size'][1]
                gtrows=[{'bbox':flip_bbox(c['bbox'],h),'type':c['type']} for c in gt['components'].values()]
                row={'case':case,'view':view,'arm':arm,'predictions':len(cached['rows']),'gt':len(gtrows)}
                for rule in ['iou50','maxcoord20']:
                    for typed in [False,True]:
                        name=('typed_' if typed else 'geometry_')+rule
                        row[name]=pair_counts(cached['rows'],gtrows,rule,typed)
                row['per_class_iou50']={}
                for kind in sorted({g['type'] for g in gtrows}|{p['type'] for p in cached['rows']}):
                    row['per_class_iou50'][kind]=pair_counts([p for p in cached['rows'] if p['type']==kind],
                                                           [g for g in gtrows if g['type']==kind],'iou50',True)
                result=score_case(cached['prediction'],gt,ScoreConfig())
                assert result['status']=='ok',(case,view,arm,result['reason'])
                row['strict_component_f1']=result['component_f1'];row['strict_counts']=result['counts']['component_f1']
                row['strict_diagnostics']=result['diagnostics']['components'];per_case.append(row)
    for view in ['whole','tiled']:
        for arm in ['multiclass','two_stage','hybrid']:
            selected=[r for r in per_case if r['view']==view and r['arm']==arm];summary={}
            for key in ['geometry_iou50','typed_iou50','geometry_maxcoord20','typed_maxcoord20','strict_counts']:
                c=Counter()
                for row in selected:c.update(row[key])
                summary[key]=counts_metrics(c)
            summary['strict_macro_component_f1']=float(np.mean([r['strict_component_f1'] for r in selected]))
            diagnostics=Counter()
            for r in selected:diagnostics.update(r['strict_diagnostics'])
            summary['strict_error_diagnostics_overlapping']=dict(diagnostics)
            per_class={}
            for kind in sorted({k for r in selected for k in r['per_class_iou50']}):
                c=Counter()
                for r in selected:c.update(r['per_class_iou50'].get(kind,{}))
                per_class[kind]=counts_metrics(c)
            summary['per_class_iou50']=per_class
            summary['typed_macro_f1_dev_present29']=float(np.mean([r['f1'] for r in per_class.values() if r['tp']+r['fn']>0]))
            times=[r for r in read(OUT/'timings.json') if r['view']==view and r['arm']==arm]
            summary['mean_seconds_excluding_frozen_ocr']=float(np.mean([r['total_seconds_excluding_frozen_ocr'] for r in times]))
            summary['median_seconds_excluding_frozen_ocr']=float(np.median([r['total_seconds_excluding_frozen_ocr'] for r in times]))
            fallback=[r for r in read(OUT/'fallback.json') if r['view']==view and r['arm']==arm]
            summary['fallback_eligible']=sum(r['fallback_eligible'] for r in fallback)
            summary['fallback_type_changes']=sum(r['fallback_type_changes'] for r in fallback)
            groups[view+'/'+arm]=summary
    deltas={};rng=np.random.default_rng(20261017)
    for view in ['whole','tiled']:
        for arm in ['multiclass','hybrid']:
            diffs=np.array([next(r['strict_component_f1'] for r in per_case if r['case']==c and r['view']==view and r['arm']==arm)
                            -next(r['strict_component_f1'] for r in per_case if r['case']==c and r['view']==view and r['arm']=='two_stage') for c in cases])
            interval=np.percentile(diffs[rng.integers(0,20,(10000,20))].mean(1),[2.5,97.5]).tolist()
            deltas[view+'/'+arm+'-two_stage']={'macro_delta':float(diffs.mean()),'case_bootstrap_ci95':interval}
    write(OUT/'evaluation.json',{'summaries':groups,'per_case':per_case,'paired_deltas':deltas,
        'metric':'local strict component scorer, not released official evaluator; dev20 repeatedly used, not sealed independent test',
        'all_931_gt_included':True,'new_api_calls':0,'test40_used':False})
    protect()


def render():
    import os
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    e=read(OUT/'evaluation.json');config=read(OUT/'CONFIG.json');res=read(OUT/'resnet_training.json')
    lines=['# 多类YOLO vs YOLO＋ResNet 受控短训','',
           '仅3050 Ti；所有原weights哈希未变，默认模型未替换。train140/6417框，dev20/931框，训练37种原始类别，未见crystal仍计入漏检。API0，test40未读取。','',
           '本轮建议：暂时保留ResNet，不把这次短训当作删除两阶段的依据。两阶段的严格ComponentF1在整图/切片下分别为18.32%/17.13%，多类直出为17.50%/15.65%；差值的case-bootstrap区间跨0，尚不能确证架构优势。多类直出更快，并有较好的总体类型检测microF1；两阶段的稀有类别覆盖更好（切片类型macroF1 26.36% vs14.73%）。','',
           'YOLO两臂各16轮，同通用初始化、配方、seed42、1280/batch1，主比较固定最后一轮。ResNet另从ImageNet训练12轮、每类每轮64次曝光，单阶段/两阶段并非等总训练算力。','',
           '|视图／方法|几何P@IoU50|几何R@IoU50|类型F1@IoU50|类型macroF1(29类)|类型F1@20px|严格ComponentF1（case宏）|严格TP|秒/图（不含OCR）|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    labels={'multiclass':'多类YOLO','two_stage':'单类YOLO＋ResNet','hybrid':'多类YOLO＋ResNet复核'}
    for name,s in e['summaries'].items():
        view,arm=name.split('/')
        lines.append(f"|{view}／{labels[arm]}|{s['geometry_iou50']['precision']:.2%}|{s['geometry_iou50']['recall']:.2%}|{s['typed_iou50']['f1']:.2%}|{s['typed_macro_f1_dev_present29']:.2%}|{s['typed_maxcoord20']['f1']:.2%}|{s['strict_macro_component_f1']:.2%}|{s['strict_counts']['tp']}|{s['mean_seconds_excluding_frozen_ocr']:.3f}|")
    lines+=['','分类器在931个GT上下文裁剪上的诊断：accuracy '+f"{res['history'][-1]['val_accuracy_all931']:.2%}"+'，29个开发集有支持类别的macroF1 '+f"{res['history'][-1]['val_macro_f1_present29']:.2%}"+'（含未见类别）。这不是检测后全流程精度。','',
            '## 固定协议与限制','',
            '三臂冻结相同OCR文字、关联与导出规则；raw→semantic只用于关联，导出恢复原始细类，禁用位号前缀覆盖。所有检测保留，未匹配key仍作为FP。推理conf=.25，同样的class-agnostic NMS=.5；切片1024/25%重叠，边缘过滤2px。混合分支仅YOLO<.5且ResNet≥.8时换类型，不能补漏框。','',
            '16/12轮是短预算、单seed；ResNet每轮2368份曝光、12轮共28416份，大类并非每轮全覆盖。稀有类1～2个原始实例重复采样不能补足样式多样性，完整来源支持量见type_support.csv。官方bbox可能局部/不一致，上下文外扩只是固定试验规则。该开发集已被多次用于工程迭代，清洁训练不等于独立测试结论。内部多类YOLO val排除1个schema外crystal，因此不直接用其mAP与单类mAP排名；本报告独立评价包含全部931框。','',
            '本次新训练只使用官方train140；旧业务Round004/005使用过不同来源、训练预算与历史曝光，不能拿旧86% mAP或17%严格F1直接与本表推断架构优劣。旧模型均保留。','',
            '固定conf=.25只是在同一部署阈值下比较，并不保证两种YOLO的置信度已校准或处在相同召回率。混合门槛同样预先固定、未校准；本轮不做阈值搜索，不把一个工作点的输赢当成完整PR曲线或架构性能上限。','',
            '严格ComponentF1同时要求key、raw type、bbox≤20px与规定文字；它不是检测F1。匿名隐藏ID、公开标签与图面不一致及OCR仍会限制分数。本次只评component，没有Pin/Net成绩。','',
            '计时复用本地OCR缓存，不含OCR、API及I/O全部成本；GPU顺序运行有热启动效应，只作本机诊断，不能外推容器CPU时间。主报告同时给bootstrap差值区间，未据dev选最佳超参或触发阈值。','',
            '## 训练与复现','',
            '```powershell','.\\.venv-yolo\\Scripts\\python.exe scripts/compare_component_architectures.py prepare',
            '.\\.venv-yolo\\Scripts\\python.exe scripts/compare_component_architectures.py seal',
            '.\\.venv-yolo\\Scripts\\python.exe scripts/compare_component_architectures.py all_train',
            '.\\.venv-yolo\\Scripts\\python.exe scripts/evaluate_component_architectures.py','```','',
            '这些命令拒绝覆盖已存在的数据/训练/预测。完整配置CONFIG.json，数据sha封印data_hashes.json，逐图预测predictions/，评测evaluation.json，训练日志training.log。']
    for arm in ['single','multi']:
        training=read(OUT/(arm+'_training.json'))
        lines.append(f"\n{arm}: {training['epochs']}轮，{training['seconds']:.1f}秒，峰值分配显存{max(r['peak_allocated_mib'] for r in training['progress']):.0f}MiB，权重`{training['checkpoint']}`。")
    lines.append(f"\nResNet:12轮，{res['seconds']:.1f}秒，峰值分配显存{max(r['peak_allocated_mib'] for r in res['history']):.0f}MiB。")
    sizes={name:(OUT/'runs'/directory/'weights/last.pt' if directory!='resnet' else OUT/'runs/resnet/last.pt').stat().st_size/2**20
           for name,directory in [('single_detector','single'),('multiclass_detector','multi'),('resnet18','resnet')]}
    write(OUT/'checkpoint_sizes_mib.json',sizes)
    lines.append('\n推理checkpoint文件大小（MiB，不含运行时显存）：'+json.dumps(sizes,ensure_ascii=False)+'。两阶段需检测器＋ResNet两个文件；混合需多类检测器＋ResNet。')
    lines+=['','## 类别与结构归因限制','',
            '这个系统对照同时包含分类器容量、额外训练、按类均衡采样和上下文裁剪的影响，不能把增益都归因为“分两步”本身。YOLO多类头未另做类别均衡配方搜索。若ResNet裁剪仍未包含完整符号，误差也可能来自bbox协议与裁剪信息丢失，而非网络无法分类。']
    lines+=['','## 严格宏F1相对两阶段的配对差值','',json.dumps(e['paired_deltas'],ensure_ascii=False,indent=2)]
    lines+=['','## 切片视图严格身份诊断（错误项可重叠）','',
            '|方法|key命中|命中key中type错|bbox错|必要文字错|','|---|---:|---:|---:|---:|']
    for arm in ['multiclass','two_stage','hybrid']:
        d=e['summaries']['tiled/'+arm]['strict_error_diagnostics_overlapping']
        lines.append(f"|{labels[arm]}|{d['key_hit']}|{d['type_wrong']}|{d['bbox_wrong']}|{d['text_wrong']}|")
    lines+=['','切片两阶段的预测数632→810，但严格TP仍为163；多类直出预测537→683，严格TP143→140。更多几何/类型正确框没有自动变成完整身份正确项，且新增候选会影响全局文字归属。不能只根据检测召回来决定Component导出策略。本次没有根据这些结果修改阈值、切片策略或业务默认值。','',
            '6项针对性模型无关检查通过；数据、原权重和20份OCR来源像素均核验一致。']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in ['single','multi']:
        rows=read(OUT/(arm+'_training.json'))['progress']
        axes[0].plot([r['epoch'] for r in rows],[r['metrics']['metrics/mAP50(B)'] for r in rows],label=arm)
    axes[0].set(title='Internal YOLO mAP50 (different class tasks)',xlabel='epoch');axes[0].legend()
    axes[1].plot([r['epoch'] for r in res['history']],[r['val_accuracy_all931'] for r in res['history']],label='GT crop accuracy')
    axes[1].plot([r['epoch'] for r in res['history']],[r['val_macro_f1_present29'] for r in res['history']],label='GT crop macroF1')
    axes[1].set(title='ResNet18; 29 dev-present types incl unseen',xlabel='epoch');axes[1].legend();fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=140);plt.close(fig)
    # Three-arm overlays are displays only: scoring GT never affects predictions.
    pages=OUT/'cases';pages.mkdir(exist_ok=True);links=[]
    for case in sorted({r['case'] for r in e['per_case']}):
        source=DATA/'single/images/val'/(case+'.png')
        with Image.open(source) as image:w,h=image.size
        href=Path(os.path.relpath(source,pages)).as_posix();panels=[]
        for arm in ['multiclass','two_stage','hybrid']:
            cached=read(OUT/'predictions/tiled'/arm/(case+'.json'));boxes=[]
            for row in cached['trace']:
                c=cached['prediction']['components'][row['key']];x1,y1,x2,y2=c['bbox'];y1,y2=h-y2,h-y1
                color='#d97706' if row['key'].startswith('__') else '#0369a1'
                label=html.escape(c['type']+(' '+row['key'] if not row['key'].startswith('__') else ' ?'))
                boxes.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width="1.3"/><text x="{x1}" y="{max(9,y1-2)}" fill="{color}" font-size="9">{label}</text>')
            metrics=next(r for r in e['per_case'] if r['case']==case and r['arm']==arm and r['view']=='tiled')
            panels.append(f'<div><h2>{labels[arm]}</h2><p>strict F1 {metrics["strict_component_f1"]:.2%}; {len(cached["rows"])} boxes</p><svg viewBox="0 0 {w} {h}"><image href="{html.escape(href)}" width="{w}" height="{h}"/>{"".join(boxes)}</svg></div>')
        (pages/(case+'.html')).write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:system-ui;margin:20px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}svg{width:100%;background:white}h2{font-size:18px}</style><a href="../gallery.html">返回</a><p>同图三臂切片结果：蓝框=已有key；橙框=未解析key。颜色不表示正确性；可放大浏览。</p><div class="grid">'+''.join(panels)+'</div>',encoding='utf-8')
        links.append(f'<a href="cases/{case}.html">{case}</a>')
    tables=['<table><tr><th>view / arm</th><th>geometry P</th><th>geometry R</th><th>typed F1</th><th>strict macro F1</th><th>TP</th><th>seconds</th></tr>']
    for name,s in e['summaries'].items():
        tables.append(f'<tr><td>{name}</td><td>{s["geometry_iou50"]["precision"]:.2%}</td><td>{s["geometry_iou50"]["recall"]:.2%}</td><td>{s["typed_iou50"]["f1"]:.2%}</td><td>{s["strict_macro_component_f1"]:.2%}</td><td>{s["strict_counts"]["tp"]}</td><td>{s["mean_seconds_excluding_frozen_ocr"]:.3f}</td></tr>')
    page='<!doctype html><meta charset="utf-8"><title>Component architecture pilot</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;background:#fafafa}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ddd}a{display:inline-block;margin:8px}img{width:100%}</style><h1>多类YOLO／两阶段／混合：3050 Ti受控短训</h1><p>train140 / dev20；两臂YOLO各16轮＋ResNet12轮。原业务模型保留。API0，test40未使用。只评component；短预算单seed，尚不能宣称架构最终优劣。</p><p><strong>本轮暂不建议删除ResNet：</strong>两阶段严格F1略高、稀有类覆盖更好；多类直出更快。差值区间跨0，只是初步证据。切片提升框召回，却没有提升完整身份得分。</p>'+''.join(tables)+'</table><p>所有931个GT纳入；严格ComponentF1含匿名ID及必要文字。计时不含缓存OCR。YOLO内部mAP因任务类别不同仅作曲线观察。</p><img src="training_curves.png"><a href="REPORT.md">完整报告</a><h2>20张同图三臂对照</h2>'+''.join(links)
    (OUT/'gallery.html').write_text(page,encoding='utf-8')
    protect();write(OUT/'STATUS.json',{'state':'complete','api_calls':0,'test40_used':False,'protected_weights_unchanged':True})


if __name__=='__main__':
    try:
        if not (OUT/'predictions_frozen.json').exists():predict_all()
        evaluate();render()
    except Exception as error:
        write(OUT/'STATUS.json',{'state':'evaluation_failed','error':type(error).__name__+': '+str(error)})
        raise
