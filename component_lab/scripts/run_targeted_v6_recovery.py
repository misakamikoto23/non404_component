"""Bounded, slot-locked local v6 recovery; inference never reads labels."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[1]
OUT=LAB/'outputs/targeted_v6_recovery_20261018'
OLD=LAB/'outputs/non404_component_ocr_20261018'
FEATURES=LAB/'outputs/frequency_ocr_regions_20261018'
sys.path.insert(0,str(LAB/'src'));sys.path.insert(0,str(LAB/'scripts'))
from component_lab.locked_ocr_recovery import lock_tokens,merge_locked,area,overlap,remove_anchor_duplicates


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def fingerprint(data):return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def distance(a,b):return math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))


def prepare():
    from PIL import Image
    from compare_non404_ocr import load_ocr
    from refine_local_reference_ocr import context_window
    started=time.perf_counter();inputs=[];requests=[]
    sources=read(OLD/'manifest.json')['inputs'];feature_locks=read(FEATURES/'proposals_frozen.json')['files']
    for row in sources:
        case=row['case'];assert sha(row['image'])==row['image_sha256']
        fpath=FEATURES/'proposals'/f'{case}.json';assert sha(fpath)==feature_locks[str(fpath.relative_to(FEATURES))]
        features=read(fpath)['features'];basepath=OLD/'predictions/v5_existing'/f'{case}.json';cache=read(basepath)
        dpath=OLD/'detector'/f'{case}.json';detector=read(dpath);w,h=detector['image_size'];texts=load_ocr(row,'v5_existing')['texts']
        candidates=[];targets=[]
        for i,(trace,det) in enumerate(zip(cache['trace'],detector['rows'])):
            ident=f'architecture_{i:04d}';assert trace['source_id']==ident
            key=trace['key'];record=cache['current']['components'][key]
            target={'id':ident,'baseline_key':key,'baseline_record':record,'bbox':det['bbox'],'type':det['type'],'detector_score':det['score']}
            targets.append(target);slots=[]
            if det['type'] in {'gnd','v','m3螺丝','other','block'} or det['type'].startswith('net_'):continue
            if key.startswith('__unresolved'):slots.append('reference')
            if record['type'] in {'r','c'} and record['value'] is None:slots.append('value')
            if record['type']=='box' and record['Name'] is None:slots.append('Name')
            if not slots:continue
            opts=[]
            for f in features:
                gap=distance(f['box'],det['bbox'])
                if gap>96 or f['character_cc_count']==0:continue
                cx,cy=(f['box'][0]+f['box'][2])/2,(f['box'][1]+f['box'][3])/2
                dx,dy=(det['bbox'][0]+det['bbox'][2])/2,(det['bbox'][1]+det['bbox'][3])/2
                score=f['hybrid_score']-.30*gap/96-.10*min(2,math.hypot(cx-dx,cy-dy)/128)
                opts.append((score,f['box']))
            if opts:
                score,cell=max(opts,key=lambda v:v[0]);box=[max(0,cell[0]-24),max(0,cell[1]-24),min(w,cell[2]+24),min(h,cell[3]+24)];method='target_gated_frequency_character_features'
            else:
                score=0.;box=context_window(det['bbox'],w,h,minimum_margin=48,fraction=.5,max_side=320);method='low_frequency_geometry_fallback'
            for _ in range(3):
                before=box[:]
                for text in texts:
                    b=text['bbox']
                    if overlap(box,b)>.2:
                        new=[max(0,min(box[0],math.floor(b[0])-4)),max(0,min(box[1],math.floor(b[1])-4)),min(w,max(box[2],math.ceil(b[2])+4)),min(h,max(box[3],math.ceil(b[3])+4))]
                        if max(new[2]-new[0],new[3]-new[1])<=320:box=new
                if before==box:break
            candidates.append({'target_id':ident,'slots':slots,'box':box,'score':float(score)+.1*det['score'],'method':method,'unresolved':key.startswith('__unresolved')})
        # A reference recovery and a named empty-text recovery get one chance each.
        ordered=[]
        for group in [[c for c in candidates if c['unresolved']],[c for c in candidates if not c['unresolved']]]:
            if group:ordered.append(max(group,key=lambda c:c['score']))
        ordered+=sorted([c for c in candidates if c not in ordered],key=lambda c:(not c['unresolved'],-c['score']))
        selected=[];cost=0
        for c in ordered:
            if len(selected)==2:break
            if cost+area(c['box'])>.2*w*h:continue
            if any(overlap(c['box'],p['box'])>.7 for p in selected):continue
            cost+=area(c['box']);selected.append(c)
        with Image.open(row['image']) as image:
            for i,c in enumerate(selected):
                ident=f'{case}_{i:02d}';path=OUT/'rois'/f'{ident}.png';path.parent.mkdir(parents=True,exist_ok=True)
                image.convert('RGB').crop(c['box']).save(path)
                requests.append({'roi_id':ident,'case':case,**c,'crop':str(path),'crop_sha256':sha(path),'scaled_size':[(c['box'][2]-c['box'][0])*2,(c['box'][3]-c['box'][1])*2]})
        inputs.append({**row,'baseline_path':str(basepath),'baseline_sha256':sha(basepath),'detector_path':str(dpath),'detector_sha256':sha(dpath),'targets':targets,'size':[w,h]})
    manifest={'cases':inputs,'requests':requests,'settings':{'model':'PP-OCRv6_medium_det_rec','backend':'RapidOCR3.10 ONNX CPU','threads':2,'scale':2,'max_rois_per_image':2,'max_crop_side':320,'max_run_seconds':600,'automatic_retries':0,'selection_uses_GT_or_v6':False},
              'model_hashes':{str(LAB/'weights/ocr_rapid'/n):sha(LAB/'weights/ocr_rapid'/n) for n in ['PP-OCRv6_det_medium.onnx','PP-OCRv6_rec_medium.onnx']},
              'protected_pt':read(OLD/'manifest.json')['protected'],'selector_seconds':time.perf_counter()-started}
    if (OUT/'manifest.json').exists():raise FileExistsError('ROI manifest already frozen; do not reselect')
    save(OUT/'manifest.json',manifest);save(OUT/'manifest_sha256.json',{'sha256':sha(OUT/'manifest.json')})
    print('Frozen',len(requests),'ROIs:',[(r['case'],r['target_id'],r['slots'],r['box']) for r in requests],flush=True)


def validate():
    assert sha(OUT/'manifest.json')==read(OUT/'manifest_sha256.json')['sha256']
    m=read(OUT/'manifest.json')
    for row in m['cases']:
        for path,key in [(row['image'],'image_sha256'),(row['baseline_path'],'baseline_sha256'),(row['detector_path'],'detector_sha256'),(row['ocr_v5_cache'],'ocr_sha256')]:assert sha(path)==row[key]
    for path,digest in m['model_hashes'].items():assert sha(path)==digest
    for request in m['requests']:assert sha(request['crop'])==request['crop_sha256']
    return m


def worker():
    from PIL import Image
    import numpy as np
    from rapidocr import RapidOCR,OCRVersion,ModelType
    from refine_local_reference_ocr import transform_polygon
    m=validate();tick=time.perf_counter()
    engine=RapidOCR(params={'Global.model_root_dir':str(LAB/'weights/ocr_rapid'),'Global.max_side_len':4096,'Global.text_score':.35,'Global.use_cls':False,
                           'Det.ocr_version':OCRVersion.PPOCRV6,'Rec.ocr_version':OCRVersion.PPOCRV6,'Det.model_type':ModelType.MEDIUM,'Rec.model_type':ModelType.MEDIUM,
                           'Det.limit_side_len':2048,'Det.limit_type':'max','EngineConfig.onnxruntime.intra_op_num_threads':2,'EngineConfig.onnxruntime.inter_op_num_threads':1,'EngineConfig.onnxruntime.use_cuda':False})
    save(OUT/'initialization.json',{'seconds':time.perf_counter()-tick})
    for req in m['requests']:
        target=OUT/'ocr'/f"{req['roi_id']}.json";attempt=OUT/'attempts'/f"{req['roi_id']}.json"
        if target.exists() or attempt.exists():continue
        save(attempt,{'status':'attempted','request_hash':fingerprint(req)});started=time.perf_counter()
        try:
            with Image.open(req['crop']) as image:
                scaled=image.convert('RGB').resize(req['scaled_size'],Image.Resampling.LANCZOS);output=engine(np.asarray(scaled))
            texts=[]
            if output.boxes is not None:
                for i,(polygon,text,score) in enumerate(zip(output.boxes,output.txts,output.scores)):
                    poly=transform_polygon(polygon.tolist(),req['box'],req['scaled_size']);xs,ys=zip(*poly);bbox=[min(xs),min(ys),max(xs),max(ys)]
                    if area(bbox)<=0 or not all(math.isfinite(v) for v in bbox):continue
                    texts.append({'id':f"local_{req['roi_id']}_{i:04d}",'raw':str(text),'text':str(text),'score':float(score),'bbox':bbox,'polygon':poly,
                                  'owner_ids':[req['target_id']],'source_tier':'fallback_v6','source_window':req['box'],'roi_id':req['roi_id']})
            save(target,{'status':'ok','request_hash':fingerprint(req),'texts':texts,'seconds':time.perf_counter()-started})
            print(req['roi_id'],len(texts),'texts',round(time.perf_counter()-started,2),'s',flush=True)
        except Exception as exc:
            save(target,{'status':'failed','error_type':type(exc).__name__,'texts':[],'seconds':time.perf_counter()-started,'request_hash':fingerprint(req)})


def run():
    m=validate()
    if (OUT/'run_started.json').exists():raise FileExistsError('Run already attempted; no automatic repeated OCR')
    save(OUT/'run_started.json',{'manifest_sha256':sha(OUT/'manifest.json')});started=time.perf_counter()
    with (OUT/'ocr.log').open('w',encoding='utf8') as log:
        process=subprocess.Popen([str(LAB/'.venv-ocr/Scripts/python.exe'),str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,cwd=LAB)
        try:code=process.wait(timeout=m['settings']['max_run_seconds']);state='finished'
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT,check=False);process.wait();code=process.returncode;state='timeout'
    for req in m['requests']:
        target=OUT/'ocr'/f"{req['roi_id']}.json"
        if not target.exists():save(target,{'status':'incomplete_fallback_baseline','texts':[],'seconds':None,'request_hash':fingerprint(req)})
    save(OUT/'run_status.json',{'state':state,'exit_code':code,'worker_wall_seconds':time.perf_counter()-started})
    print(read(OUT/'run_status.json'),flush=True)


def assemble():
    from compare_non404_ocr import load_ocr
    from evaluate_component_architectures import SEMANTIC
    from component_lab.text_matching import associate_structured
    from component_lab.display_names import assign_display_names
    from component_lab.contest_components import export_contest
    from component_lab.ocr_spans import split_ocr_spans
    from component_lab.text_roles import REFERENCE
    m=validate();frozen={}
    for row in m['cases']:
        case=row['case'];base=read(row['baseline_path']);texts=load_ocr(row,'v5_existing')['texts'];bytext={t['id']:t for t in texts};bindings=[]
        for trace in base['trace']:
            for field,evidence in trace['evidence'].items():
                if evidence.get('text_id') in bytext:bindings.append({'owner':trace['source_id'],'field':field,'bbox':bytext[evidence['text_id']]['bbox']})
        tokens=[];selected={r['target_id']:r for r in m['requests'] if r['case']==case}
        for req in m['requests']:
            if req['case']!=case:continue
            response=read(OUT/'ocr'/f"{req['roi_id']}.json");assert response['request_hash']==fingerprint(req);tokens+=response['texts']
        tokens,span_audit=split_ocr_spans(tokens);tokens,blocked=lock_tokens(tokens,bindings)
        # Primary references serve as read-only anchors for contextual numeric values.
        anchors=[]
        for trace in base['trace']:
            if trace['source_id'] in selected and trace['evidence'].get('reference',{}).get('text_id') in bytext:
                t=deepcopy(bytext[trace['evidence']['reference']['text_id']]);t['id']='anchor_'+trace['source_id'];t['source_tier']='primary_anchor';t['owner_ids']=[trace['source_id']];anchors.append(t)
        tokens,duplicate_blocks=remove_anchor_duplicates(tokens,anchors);blocked+=duplicate_blocks
        detector=read(row['detector_path']);adapted=deepcopy(detector['rows']);raw={}
        for i,r in enumerate(adapted):
            r['id']=f'architecture_{i:04d}';raw[r['id']]=r['type'];r['type']=SEMANTIC.get(r['type'],r['type'])
        # Anchors cannot migrate; fallback still faces all boxes and is owner-checked again at merge.
        allowed={t['id']:t['owner_ids'] for t in anchors}
        association=associate_structured(adapted,tokens+anchors,clamp_costs=False,allowed_text_owners=allowed)
        fallback,trace=export_contest(assign_display_names(association),row['size'][1],ic_value_as_name=True,display_names=True)
        for t in trace:fallback['components'][t['key']]['type']=raw[t['source_id']]
        byid={t['source_id']:t for t in trace};local={t['id']:t for t in tokens};proposals=[];rejected=[]
        for ident,req in selected.items():
            t=byid[ident];record=fallback['components'][t['key']];fields={}
            for field in ['reference','Name','value']:
                if field not in req['slots'] and not (req['unresolved'] and field in ['Name','value']):continue
                source='reference' if field=='reference' else 'value' if field=='value' else t.get('name_source')
                if source=='value_fallback':source='value'
                evidence=t['evidence'].get(source,{}) if source else {};oid=evidence.get('text_id')
                text=t['reference_raw'] if field=='reference' else record[field]
                if oid not in local or text is None:continue
                if field=='reference':
                    match=REFERENCE.fullmatch(text.strip());kind=next(x['type'] for x in row['targets'] if x['id']==ident)
                    strong={'R':{'r'},'C':{'c'},'L':{'l'},'U':{'box','amp','amp_3pin','amp_5pin','opt'},'IC':{'box'}}
                    if not match or (match.group(1).upper() in strong and kind not in strong[match.group(1).upper()]):
                        rejected.append({'id':ident,'field':field,'reason':'reference_prefix_visual_type_conflict'});continue
                fields[field]={'text':text,'ocr_id':oid}
            if fields:proposals.append({'id':ident,'fields':fields})
        save(OUT/'evidence'/f'{case}.json',{'tokens':tokens,'proposals':proposals,'blocked':blocked+rejected,'span_audit':span_audit,'primary_bindings':bindings,'fallback_trace':trace})
        variants={'baseline':(base['current'],{})}
        for mode in ['safe_unresolved','empty_fields_shadow']:
            variants[mode]=merge_locked(base['current'],row['targets'],proposals,tokens,mode)
        for mode,(prediction,audit) in variants.items():
            target=OUT/'predictions'/mode/f'{case}.json';save(target,prediction);save(OUT/'audits'/mode/f'{case}.json',audit);frozen[str(target.relative_to(OUT))]=sha(target)
        print(case,'tokens',len(tokens),'proposals',len(proposals),'primary blocked',len(blocked),flush=True)
    save(OUT/'predictions_frozen.json',{'files':frozen})


def report():
    from component_lab.eval.metric import score_case,ScoreConfig,normalize_case,bbox_ok,text_ok
    m=validate()
    for rel,digest in read(OUT/'predictions_frozen.json')['files'].items():assert sha(OUT/rel)==digest
    for path,digest in m['protected_pt'].items():assert sha(path)==digest
    results=[];summaries={};hits={}
    for row in m['cases']:
        case=row['case'];gt=read(next((LAB/'data/official_contest6/train200').glob(case+'-*.json')));norm=normalize_case(gt)['components']
        for mode in ['baseline','safe_unresolved','empty_fields_shadow']:
            pred=read(OUT/'predictions'/mode/f'{case}.json');score=score_case(pred,gt,ScoreConfig());assert score['status']=='ok'
            results.append({'case':case,'mode':mode,'score':score});pn=normalize_case(pred)['components']
            hits[case,mode]={k for k,v in norm.items() if k in pn and pn[k][0]==v[0] and bbox_ok(pn[k][3],v[3],ScoreConfig()) and text_ok(pn[k],v)}
    for mode in ['baseline','safe_unresolved','empty_fields_shadow']:
        rows=[r for r in results if r['mode']==mode];counts=Counter()
        for r in rows:counts.update(r['score']['counts']['component_f1'])
        summaries[mode]={'macro_f1':sum(r['score']['component_f1'] for r in rows)/len(rows),**dict(counts),
                         'added':sum(len(hits[r['case'],mode]-hits[r['case'],'baseline']) for r in rows),
                         'lost':sum(len(hits[r['case'],'baseline']-hits[r['case'],mode]) for r in rows)}
    ocr=[read(OUT/'ocr'/f"{r['roi_id']}.json") for r in m['requests']]
    timing={'run':read(OUT/'run_status.json'),'initialization':read(OUT/'initialization.json') if (OUT/'initialization.json').exists() else None,
            'roi_inference_seconds':sum(r['seconds'] or 0 for r in ocr),'roi_count':len(ocr),'roi_ok':sum(r['status']=='ok' for r in ocr),
            'selection_seconds':m['selector_seconds'],'old_feature_generation_seconds':sum(read(FEATURES/'proposals'/f"{r['case']}.json")['selector_seconds'] for r in m['cases'])}
    save(OUT/'report.json',{'summaries':summaries,'per_case':results,'timing':timing,'default_changed':False,'api_calls':0})
    text=['# 固定五图：局部v6安全补读','', '基线为前次37类短训YOLO26n、1024、固定76框/类型的五图子集；不是全部dev20或历史non40444.31%复现。ROI选择未读GT或整图v6结果，推理后预测先冻结再评分。', '',
          '|策略|严格macro ComponentF1|TP|新增TP|损失旧TP|','|---|---:|---:|---:|---:|']
    for mode,s in summaries.items():text.append(f"|{mode}|{s['macro_f1']:.2%}|{s['tp']}|{s['added']}|{s['lost']}|")
    text+=['', '耗时与请求：',json.dumps(timing,ensure_ascii=False,indent=2),
           '', 'safe_unresolved锁定全部旧named完整记录，仅允许未命名对象恢复key并补原本空字段；empty_fields_shadow额外允许旧named对象的必要空字段填值，可能破坏GT正确null。二者不改已有非空字符串、bbox/type/pins/nets。',
           '', '局部OCR映射回原图；新增token只能用于其ROI原指定owner，primary已有空间绑定先锁，不能换新OCR ID绕开。fallback关联使用全部检测框竞争，而非强行把ROI内全部文字归给目标。冲突与来源见evidence/和audits/。',
           '', '每图最多2ROI；先给未命名元件及旧named缺必要文字各一个机会，没有该组时用另一组补足。优先目标附近的局部频域/类字符特征，无特征保留几何回退；裁剪原图RGB，未把去线残差送OCR。无旋转补读，本轮并未验证竖排改善。',
           '', '与前次819秒整图v6相比，运行负载、输入和OCR任务量不同，计时是小样可行性测量，不是隔离硬件benchmark。特征读取旧预测生成缓存，原始特征生成耗时单列；v5基线缓存的38秒也不包含在worker时间中。',
           '', '无API、无训练、不改原模型/标签/默认、不读test40。成功不等于有净增益；五图不证明泛化。程序遇到超时/失败会保留baseline，失败ROI不自动重试。',
           '', '复现：.venv-yolo运行prepare；.venv-ocr运行run（内含单进程worker、10分钟超时）；.venv-yolo运行assemble和report。全部命令入口scripts/run_targeted_v6_recovery.py；已有manifest/run拒绝重复启动。']
    text+=['', '结论：只恢复未命名元件未净增TP；允许填写必要空字段的shadow为R3补回原文10kΩ，新增1TP且本次旧TP未损失。五图严格macro13.29%→14.58%，收益有限；不启用主infer默认，不追加本轮ROI或OCR重试。',
           '', '修复记录：初次合并将同位置新R3与旧primary anchor R3同时送入分配，并未锁anchor的owner，导致邻居抢到anchor及数值。初次零增益输出完整保存在initial_assembly。用不依赖GT的反例测试修复：同owner/同raw/同空间reference去重，primary anchor增加硬owner约束。复用原5个OCR响应离线重合并，无新识别。默认匹配API在未提供owner allowlist时行为不变。',
           '', '实际只有3张图触发了5个ROI，0034和0058没有符合条件的目标，保留baseline且仍进入全部五图评分。初始化文件记录的是RapidOCR构造器耗时；实际模型懒加载包含在首个ROI时间里，worker总73秒包括整个子进程。',
           '', '评测数据限制：0024局部读到了可见K6/K7，但它们不是公开GT的key；该图GT使用U1..U16，而图面按键是K0..K15。不能为了得分把图面位号强改成不可观察的内部/转换编号；需要主办方澄清配对或编号协议。见public_label_conflict.json。']
    (OUT/'REPORT.md').write_text('\n'.join(text),encoding='utf8')
    cards=[]
    for row in m['cases']:
        w,h=row['size'];boxes=[]
        for req in m['requests']:
            if req['case']!=row['case']:continue
            x1,y1,x2,y2=req['box'];boxes.append(f'<g><title>{html.escape(req["target_id"]+": "+str(req["slots"]))}</title><rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="red" stroke-width="3"/></g>')
        links=' '.join(f'<a href="predictions/{mode}/{row["case"]}.json">{mode}</a>' for mode in summaries)
        crops=''.join(f'<a href="rois/{r["roi_id"]}.png"><img height="180" src="rois/{r["roi_id"]}.png"></a>' for r in m['requests'] if r['case']==row['case'])
        cards.append(f'<h2>{row["case"]}</h2>{links}<svg width="100%" viewBox="0 0 {w} {h}"><image href="{Path(row["image"]).as_uri()}" width="{w}" height="{h}"/>{"".join(boxes)}</svg>{crops}')
    (OUT/'gallery.html').write_text('<meta charset="utf-8"><style>body{max-width:1200px;margin:25px auto;font:16px Arial}pre{white-space:pre-wrap}svg{max-height:600px}</style><h1>局部v6补读：ROI与结果</h1><pre>'+html.escape('\n'.join(text))+'</pre>'+''.join(cards),encoding='utf8')
    save(OUT/'STATUS.json',{'state':'complete','summaries':summaries,'api_calls':0,'default_changed':False})
    print(json.dumps({'summaries':summaries,'timing':timing},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','worker','assemble','report']);args=p.parse_args();globals()[args.stage]()
