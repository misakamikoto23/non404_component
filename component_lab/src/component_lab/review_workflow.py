"""Local, versioned human review of component proposals; no cloud calls or training."""
from __future__ import annotations

import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import random
import shutil
import threading
import time
from urllib.parse import unquote, urlsplit
import uuid

from PIL import Image
from .tiling import box_iou, class_aware_nms, tile_plan, touches_internal_edge, translate_prediction


LAB = Path(__file__).resolve().parents[2]
DEFAULT_DATA = LAB / 'data/yolo/pcbnet_component_v1'
DEFAULT_BATCH = LAB / 'data/review_batches/batch_001'


class RevisionConflict(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def checked_box(raw, width, height):
    if not isinstance(raw, list) or len(raw) != 4 or any(isinstance(v, bool) or not isinstance(v,(int,float)) for v in raw):
        raise ValueError('框必须为四个数值 [x1,y1,x2,y2]')
    box = list(map(float, raw))
    if not all(math.isfinite(v) for v in box) or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError('框坐标非有限数、倒置或超出原图')
    return box


def image_path(data_root, sample):
    if sample.get('annotation_status')=='pending_review' and sample.get('pending_image_file'):
        root=Path(data_root).resolve()
        pending=(root/sample['pending_image_file']).resolve()
        if not pending.is_relative_to(root/'images'/'pending') or not pending.is_file():
            raise ValueError('待审核图片必须位于数据源images/pending目录')
        return pending
    directory = Path(data_root)/'images'/sample['split']
    matches = [p for p in directory.glob(str(sample['case'])+'.*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'}]
    if len(matches) != 1:
        raise ValueError(f"图片配对失败：{sample['case']}")
    return matches[0]


def validate_review_update(item, payload, categories):
    """Validate edits and preserve provenance; no caller-owned objects are mutated."""
    if payload.get('revision') != item.get('revision',0):
        raise RevisionConflict('图片已被其他窗口修改，请刷新后再编辑')
    boxes = payload.get('boxes')
    review = payload.get('review')
    if not isinstance(boxes,list) or len(boxes)>5000 or not isinstance(review,dict):
        raise ValueError('缺少boxes/review，或框数量过多')
    if not isinstance(review.get('completed'),bool) or not isinstance(review.get('whole_image_checked'),bool):
        raise ValueError('需明确确认整图检查与完成状态')
    seconds = review.get('review_seconds',0)
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('检查时长无效')
    notes = review.get('notes','')
    if not isinstance(notes,str) or len(notes)>4000:
        raise ValueError('备注过长或格式无效')
    previous = {b['id']:b for b in item['boxes']}
    seen, cleaned = set(), []
    for candidate in boxes:
        if not isinstance(candidate,dict):
            raise ValueError('框格式无效')
        ident = candidate.get('id')
        if not isinstance(ident,str) or not ident or len(ident)>100 or ident in seen:
            raise ValueError('框ID无效或重复')
        if ident not in previous and not ident.startswith('human_'):
            raise ValueError('新增框必须使用human_标识')
        seen.add(ident)
        category = candidate.get('category','component')
        status = candidate.get('review_status')
        confirmed = candidate.get('category_confirmed',False)
        reference = candidate.get('reference')
        if category not in set(categories)|{'component'} or status not in {'pending','accepted','rejected'}:
            raise ValueError('未知类别或审核状态')
        if not isinstance(confirmed,bool) or (confirmed and (category=='component' or status!='accepted')):
            raise ValueError('分类确认仅适用于已接受且有具体类别的框')
        if reference is not None and (not isinstance(reference,str) or len(reference)>256):
            raise ValueError('位号格式无效')
        row=copy.deepcopy(previous.get(ident,{'id':ident,'source':'human','score':None,
                                             'original_bbox':None,'review_reasons':['人工补框']}))
        row.update(bbox=checked_box(candidate.get('bbox'),item['width'],item['height']),category=category,
                   category_confirmed=confirmed,reference=reference,review_status=status)
        cleaned.append(row)
    if set(previous)-seen:
        raise ValueError('删除框请标记为rejected，保留历史记录')
    if review['completed'] and (not review['whole_image_checked'] or any(b['review_status']=='pending' for b in cleaned)):
        raise ValueError('完成前必须整图检查，且逐框处理所有待检查项')
    result=copy.deepcopy(item)
    result.update(boxes=cleaned,revision=item.get('revision',0)+1,updated_at=time.time(),
                  review={'completed':review['completed'],'whole_image_checked':review['whole_image_checked'],
                          'review_seconds':seconds,'notes':notes})
    return result


def load_batch(batch_dir):
    batch_dir=Path(batch_dir)
    state=read_json(batch_dir/'batch.json')
    for index,item in enumerate(state['items']):
        changed=batch_dir/'items'/f"{item['id']}.json"
        if changed.exists():
            state['items'][index]=read_json(changed)
    return state


def write_category_reports(data_root, output):
    from .category_audit import build_category_report, classification_balance_plan
    import csv
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=build_category_report(read_json(Path(data_root)/'manifest.json'))
    atomic_json(output/'category_distribution.json',report)
    atomic_json(output/'resnet_balance_plan.json',classification_balance_plan(report))
    with (output/'category_distribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(report['categories'][0]));writer.writeheader();writer.writerows(report['categories'])
    rows=''.join('<tr>'+''.join(f'<td>{html_escape(row.get(k,""))}</td>' for k in ('category','train_boxes','train_images','val_boxes','test_boxes','rarity_tier'))+'</tr>' for row in report['categories'])
    (output/'category_distribution.html').write_text('<!doctype html><meta charset="utf-8"><title>元件类别分布</title><style>body{font-family:Arial;margin:30px}td,th{padding:8px;border-bottom:1px solid #ddd}tr:nth-child(even){background:#f5f7fa}</style><h1>元件类别与来源图片分布</h1><p>单类YOLO不要求各原始类别完全等量；ResNet权重方案仅参考train。稀缺定义按train来源图片数，验证/测试不参与抽样。</p><table><tr><th>类别</th><th>训练框</th><th>训练图</th><th>验证框</th><th>测试框</th><th>稀缺程度</th></tr>'+rows+'</table>',encoding='utf-8')
    return report


def html_escape(value):
    import html
    return html.escape(str(value))


def prepare_batch(data_root, batch_dir, weights, count=20, seed=20261015, device='0', imgsz=1280, confidence=.1, tile_size=1024, exclude_batches=None, selected_samples=None):
    from .category_audit import select_review_batch
    from .models import configure_yolo
    data_root,batch_dir=Path(data_root).resolve(),Path(batch_dir).resolve()
    if count<=0 or imgsz<=0 or not math.isfinite(confidence) or not 0<confidence<1 or tile_size<=0:
        raise ValueError('批次数、输入尺寸、切图尺寸须为正；候选置信度须在0和1之间')
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise ValueError('批次目录非空；请使用新批次名，避免覆盖检查记录')
    manifest=read_json(data_root/'manifest.json')
    excluded_ids,excluded_sources=set(),[]
    for previous_dir in dict.fromkeys(Path(p).resolve() for p in (exclude_batches or [])):
        previous_path=previous_dir/'batch.json'
        previous=read_json(previous_path)
        previous_items=previous.get('items')
        if not isinstance(previous_items,list) or any(not isinstance(item,dict) or not isinstance(item.get('id'),str) or not item['id'] for item in previous_items):
            raise ValueError(f'此前批次缺少有效图片ID：{previous_dir}')
        excluded_ids.update(item['id'] for item in previous_items)
        excluded_sources.append({'batch_dir':str(previous_dir),'batch_id':previous.get('batch_id',previous_dir.name),
            'item_count':len(previous_items),'batch_manifest_sha256':hashlib.sha256(previous_path.read_bytes()).hexdigest()})
    selection_manifest=dict(manifest)
    selection_manifest['samples']=[sample for sample in manifest['samples']
        if sample.get('split')!='train' or str(sample['case']) not in excluded_ids]
    if selected_samples is None:
        selected=select_review_batch(selection_manifest,count=count,seed=seed)
    else:
        source_by_id={s['case']:s for s in manifest['samples']}
        selected=[]
        seen=set()
        for row in selected_samples:
            ident=row['case']
            if ident not in source_by_id or ident in seen or ident in excluded_ids:
                raise ValueError('显式选图包含未知、重复或先前批次图片')
            sample=copy.deepcopy(source_by_id[ident])
            for field in ('selection_group','selection_reasons','priority'):
                sample[field]=row[field]
            selected.append(sample);seen.add(ident)
    if len(selected)!=count:
        raise ValueError(f'排除先前批次后仅剩{len(selected)}张可选训练图片，无法生成{count}张；请降低count或核对排除批次')
    if any(row['split']!='train' for row in selected):
        raise ValueError('复核批次只允许训练图片')
    batch_dir.mkdir(parents=True,exist_ok=True)
    configure_yolo()
    from ultralytics import YOLO
    model=YOLO(str(weights))
    if list(model.names.values())!=['component']:
        raise ValueError('需要单类component模型')
    def predict(im,source):
        result=model.predict(im,imgsz=imgsz,conf=confidence,device=device,max_det=1000,verbose=False)[0]
        return [{'bbox':b,'class_id':0,'score':s,'source_pass':source} for b,s in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist())]
    categories=list(manifest.get('original_category_names',[]))
    state={'schema_version':1,'batch_id':batch_dir.name,'categories':categories,
        'created_at':time.time(),'data_root':str(data_root),
        'source_manifest_sha256':hashlib.sha256((data_root/'manifest.json').read_bytes()).hexdigest(),
        'selection':{'requested_count':count,'seed':seed,'excluded_ids':sorted(excluded_ids),'excluded_batches':excluded_sources},
        'policy':{'coordinate_system':'原图像素xyxy','bbox_scope':'参照本批次框示例，区分符号本体/自身短引脚；不含参数编号和外部导线',
                  'holdouts_frozen':True,'cloud_calls_enabled':False,'all_proposals_require_review':True},
        'inference':{'weights':str(Path(weights).resolve()),'imgsz':imgsz,'confidence':confidence,'tile_size':tile_size,'overlap':.25,'merge_iou':.5},'items':[]}
    for number,sample in enumerate(selected,1):
        path=image_path(data_root,sample)
        with Image.open(path) as im:
            im=im.convert('RGB');w,h=im.size
            whole=predict(im,'whole');tile=[]
            plan=tile_plan(w,h,tile_size,.25,100)
            if max(w,h)>tile_size:
                for window in plan['windows']:
                    for row in predict(im.crop(tuple(window)),'tile'):
                        if touches_internal_edge(row['bbox'],window,w,h):continue
                        row=translate_prediction(row,window,w,h)
                        if row:tile.append(row)
            else:
                plan['skip_reason']='image_fits_one_tile_whole_pass_used'
            candidates=class_aware_nms(whole+tile,.5)
            boxes=[]
            original=[{'id':x['component_id'],'bbox':x['bbox'],'category':x['category']} for x in sample['labels']]
            for index,p in enumerate(candidates,1):
                b=checked_box(p['bbox'],w,h);reasons=[]
                if p['score']<.25:reasons.append('低置信度候选')
                if min(b[2]-b[0],b[3]-b[1])<12:reasons.append('小符号，检查边界')
                whole_seen=any(box_iou(b,x['bbox'])>=.5 for x in whole)
                tile_seen=any(box_iou(b,x['bbox'])>=.5 for x in tile)
                if tile and whole_seen!=tile_seen:reasons.append('整图与切图结果不一致')
                best=max(((box_iou(b,g['bbox']),g) for g in original),key=lambda v:v[0],default=(0,None))
                hint=best[1]['category'] if best[0]>=.7 else None
                boxes.append({'id':f"yolo_{index:04d}",'bbox':b,'original_bbox':b.copy(),'category':'component',
                    'category_hint':hint,'category_hint_source':'original_pcbnet_geometry_match' if hint else None,
                    'category_confirmed':False,'reference':None,'score':p['score'],'source':'yolo',
                    'review_status':'pending','review_reasons':reasons,'seen_whole':whole_seen,'seen_tile':tile_seen})
            rough_matches=sum(any(box_iou(g['bbox'],p['bbox'])>=.5 for p in candidates) for g in original)
            suffix=path.suffix.lower();destination=batch_dir/'images'/f"{sample['case']}{suffix}"
            destination.parent.mkdir(exist_ok=True);shutil.copy2(path,destination)
            item={'id':sample['case'],'split':'train','width':w,'height':h,'revision':0,
                'image_file':str(destination.relative_to(batch_dir)).replace('\\','/'),'image_url':f"/api/image/{sample['case']}",
                'selection_group':sample['selection_group'],'selection_reasons':sample['selection_reasons'],'priority':sample['priority'],
                'boxes':boxes,'original_boxes':original,
                'diagnostics':{'proposal_count':len(boxes),'original_count':len(original),'original_without_overlap':len(original)-rough_matches,
                    'not_official_error_count':True,'whole_count':len(whole),'tile_count':len(tile),'tile_plan':plan},
                'review':{'completed':False,'whole_image_checked':False,'review_seconds':0,'notes':''}}
            item['source_metadata']={key:sample[key] for key in ('source_dataset','source_project','source_project_url','source_record_id','dataset_revision','annotation_status') if key in sample}
            state['items'].append(item)
            atomic_json(batch_dir/'batch.json',state)
            print(f"[{number}/{len(selected)}] {sample['case']}: {len(boxes)} proposals, all pending",flush=True)
    write_category_reports(data_root,batch_dir/'reports')
    return state


def export_reviewed_dataset(batch_dir, data_root, output):
    """Build a new training version, replace completed train images only."""
    batch_dir,data_root,output=map(lambda p:Path(p).resolve(),(batch_dir,data_root,output))
    if output.is_relative_to(data_root) or output.is_relative_to(batch_dir) or data_root.is_relative_to(output) or batch_dir.is_relative_to(output):
        raise ValueError('导出目录必须独立于原数据和审核批次')
    if output.exists() and any(output.iterdir()):
        raise ValueError('导出目录必须为空，不能覆盖已有版本')
    state=load_batch(batch_dir);manifest=read_json(data_root/'manifest.json')
    if state.get('source_manifest_sha256') and state['source_manifest_sha256']!=hashlib.sha256((data_root/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('原始manifest已变化；请先核对数据版本')
    categories=state.get('categories',manifest.get('original_category_names',[]))
    original_by_id={str(s['case']):s for s in manifest['samples']}
    reviewed={}
    for item in state['items']:
        if item['id'] not in original_by_id or item['split']!='train' or original_by_id[item['id']]['split']!='train':
            raise ValueError('审核批次包含非训练样本或未知样本')
        if (item['width'],item['height'])!=(original_by_id[item['id']]['width'],original_by_id[item['id']]['height']):
            raise ValueError('审核图片尺寸与原始数据不一致')
        checked=validate_review_update(item,{'revision':item.get('revision',0),'boxes':item['boxes'],'review':item['review']},categories)
        if checked['review']['completed']:reviewed[item['id']]=checked
    if not reviewed:
        raise ValueError('尚无完成整图与逐框检查的图片；拒绝导出“已审核”数据')
    output.mkdir(parents=True,exist_ok=True)
    yolo=output/'yolo';changes=[];crops=[];holdout_hashes={}
    revised_manifest=copy.deepcopy(manifest)
    revised_by_id={s['case']:s for s in revised_manifest['samples']}
    excluded_unlabelled=[]
    for sample in manifest['samples']:
        ident,split=sample['case'],sample['split']
        if sample.get('annotation_status')=='pending_review' and ident not in reviewed:
            if split!='train':raise ValueError('待审核新来源不能位于验证或测试集')
            excluded_unlabelled.append(ident)
            continue
        image=image_path(data_root,sample);destination=yolo/'images'/split/image.name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(image,destination)
        source_label=data_root/'labels'/split/(ident+'.txt');target=yolo/'labels'/split/source_label.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if ident not in reviewed:
            shutil.copy2(source_label,target)
        else:
            item=reviewed[ident];accepted=[b for b in item['boxes'] if b['review_status']=='accepted']
            lines=[]
            with Image.open(image) as im:
                for index,box in enumerate(accepted,1):
                    x1,y1,x2,y2=box['bbox'];w,h=item['width'],item['height']
                    lines.append(f'0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
                    if box['category_confirmed'] and box['category']!='component':
                        # Explicit category whitelist also prevents filesystem traversal.
                        category=box['category']
                        if Path(category).name!=category or '/' in category or '\\' in category or category in {'.','..'}:
                            raise ValueError('类别不能作为安全目录名')
                        crop=output/'resnet_reviewed/train'/category/f'{ident}_{index:04d}.png'
                        crop.parent.mkdir(parents=True,exist_ok=True)
                        im.convert('RGB').crop((math.floor(x1),math.floor(y1),math.ceil(x2),math.ceil(y2))).save(crop)
                        crops.append({'image_id':ident,'category':category,'bbox':box['bbox'],'file':crop.relative_to(output).as_posix(),'source_box_id':box['id']})
            target.write_text('\n'.join(lines)+ ('\n' if lines else ''),encoding='utf-8')
            revised_by_id[ident]['labels']=[{'category':b['category'] if b['category_confirmed'] else 'component',
                'bbox':b['bbox'],'component_id':b['id'],'category_confirmed':b['category_confirmed'],
                'reference':b['reference'],'source':'human_reviewed','original_category_hint':b.get('category_hint')}
                for b in accepted]
            revised_by_id[ident]['annotation_status']='human_reviewed'
            revised_by_id[ident].pop('pending_image_file',None)
            changes.append({'image_id':ident,'original_boxes':len(sample['labels']),'accepted_boxes':len(accepted),
                'rejected_boxes':sum(b['review_status']=='rejected' for b in item['boxes']),'review_seconds':item['review']['review_seconds']})
        if split in {'val','test'}:
            for source,dest in ((image,destination),(source_label,target)):
                before=hashlib.sha256(source.read_bytes()).hexdigest();after=hashlib.sha256(dest.read_bytes()).hexdigest()
                if before!=after:raise ValueError('验证/测试复制不一致')
                holdout_hashes[dest.relative_to(yolo).as_posix()]=before
    (yolo/'dataset.yaml').write_text('path: '+json.dumps(str(yolo),ensure_ascii=False)+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [component]\n',encoding='utf-8')
    if excluded_unlabelled:
        revised_manifest['samples']=[s for s in revised_manifest['samples'] if s['case'] not in excluded_unlabelled]
    revised_manifest['splits']={split:[s['case'] for s in revised_manifest['samples'] if s['split']==split] for split in ('train','val','test')}
    revised_manifest.update(task='single_class_component_localization_human_reviewed',
        original_category_names=sorted(set(categories)|{'component'}),
        category_policy='Reviewed geometry may have unconfirmed component category; use category_confirmed for ResNet.',
        review_batch=state.get('batch_id',batch_dir.name))
    revised_manifest.pop('original_category_counts',None)
    revised_manifest['counts']={split:{'component':sum(len(s['labels']) for s in revised_manifest['samples'] if s['split']==split)} for split in ('train','val','test')}
    atomic_json(yolo/'manifest.json',revised_manifest)
    atomic_json(output/'resnet_reviewed/crop_manifest.json',{'samples':crops,'note':'Only human-confirmed train crops. No validation/test classification set has been created.'})
    summary={'batch_id':state.get('batch_id',batch_dir.name),'reviewed_train_images':len(reviewed),'changes':changes,
        'classification_crops':len(crops),'holdouts_unchanged':True,'holdout_sha256':holdout_hashes,
        'original_source':str(data_root),'training_started':False,'excluded_pending_images':len(state['items'])-len(reviewed),
        'excluded_unlabelled_new_images':excluded_unlabelled}
    atomic_json(output/'summary.json',summary)
    return summary


def serve_batch(batch_dir, port=8765):
    batch_dir=Path(batch_dir).resolve();lock=threading.Lock();assets=Path(__file__).with_name('review_web')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def json(self,value,status=200):
            body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def file(self,path):
            body=path.read_bytes();self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(str(path))[0] or 'application/octet-stream');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            route=unquote(urlsplit(self.path).path)
            try:
                if route=='/api/state':return self.json(load_batch(batch_dir))
                if route=='/api/stats':return self.json(read_json(batch_dir/'reports/category_distribution.json'))
                if route.startswith('/api/image/'):
                    ident=route.rsplit('/',1)[-1];state=load_batch(batch_dir)
                    item=next((x for x in state['items'] if x['id']==ident),None)
                    if not item:return self.json({'error':'未知图片'},404)
                    path=(batch_dir/item['image_file']).resolve()
                    if not path.is_relative_to(batch_dir/'images'):raise ValueError('无效图片路径')
                    return self.file(path)
                relative='index.html' if route=='/' else route.lstrip('/')
                path=(assets/relative).resolve()
                if not path.is_relative_to(assets.resolve()) or not path.is_file():return self.json({'error':'页面不存在'},404)
                return self.file(path)
            except (ValueError,OSError) as exc:self.json({'error':str(exc)},400)
        def do_POST(self):
            route=unquote(urlsplit(self.path).path)
            origin=self.headers.get('Origin')
            if (origin and origin!=f'http://127.0.0.1:{port}') or self.headers.get('Sec-Fetch-Site')=='cross-site':return self.json({'error':'只允许本地检查页面保存'},403)
            if not self.headers.get('Content-Type','').startswith('application/json'):return self.json({'error':'需要JSON请求'},400)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=4_000_000:raise ValueError('请求大小无效')
                payload=json.loads(self.rfile.read(size))
                if route=='/api/export':
                    with lock:
                        state=load_batch(batch_dir)
                        target=LAB/'data/reviewed_versions'/(batch_dir.name+'_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:6])
                        summary=export_reviewed_dataset(batch_dir,Path(state['data_root']),target)
                    return self.json({'output':str(target),'summary':summary})
                if not route.startswith('/api/items/'):return self.json({'error':'未知入口'},404)
                ident=route.rsplit('/',1)[-1]
                with lock:
                    state=load_batch(batch_dir);item=next((x for x in state['items'] if x['id']==ident),None)
                    if not item:return self.json({'error':'未知图片'},404)
                    changed=validate_review_update(item,payload,state['categories'])
                    event={'time':time.time(),'image_id':ident,'before_revision':item.get('revision',0),'after':changed}
                    with (batch_dir/'review_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
                    atomic_json(batch_dir/'items'/f'{ident}.json',changed)
                return self.json({'item':changed})
            except RevisionConflict as exc:self.json({'error':str(exc)},409)
            except (ValueError,TypeError,KeyError,OSError) as exc:self.json({'error':str(exc)},400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Review UI: http://127.0.0.1:{port} (local only)',flush=True)
    server.serve_forever()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    stat=commands.add_parser('stats');stat.add_argument('--data',type=Path,default=DEFAULT_DATA);stat.add_argument('--output',type=Path,default=LAB/'outputs/review_workflow/category_audit')
    prepare=commands.add_parser('prepare');prepare.add_argument('--data',type=Path,default=DEFAULT_DATA);prepare.add_argument('--batch',type=Path,default=DEFAULT_BATCH);prepare.add_argument('--weights',type=Path,default=LAB/'weights/pcbnet_component_yolo26n_v1.pt');prepare.add_argument('--count',type=int,default=20);prepare.add_argument('--seed',type=int,default=20261015);prepare.add_argument('--device',default='0');prepare.add_argument('--imgsz',type=int,default=1280);prepare.add_argument('--confidence',type=float,default=.1);prepare.add_argument('--tile-size',type=int,default=1024);prepare.add_argument('--exclude-batch',dest='exclude_batches',type=Path,action='append',help='排除此前批次的全部图片，可重复指定多个批次目录')
    serve=commands.add_parser('serve');serve.add_argument('--batch',type=Path,default=DEFAULT_BATCH);serve.add_argument('--port',type=int,default=8765)
    export=commands.add_parser('export');export.add_argument('--batch',type=Path,default=DEFAULT_BATCH);export.add_argument('--data',type=Path,default=DEFAULT_DATA);export.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(argv)
    if a.command=='stats':print(json.dumps(write_category_reports(a.data,a.output)['totals'],ensure_ascii=False))
    elif a.command=='prepare':prepare_batch(a.data,a.batch,a.weights,a.count,a.seed,a.device,a.imgsz,a.confidence,a.tile_size,exclude_batches=a.exclude_batches)
    elif a.command=='serve':serve_batch(a.batch,a.port)
    elif a.command=='export':print(json.dumps(export_reviewed_dataset(a.batch,a.data,a.output),ensure_ascii=False))


if __name__=='__main__':main()
