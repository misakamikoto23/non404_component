"""Bounded cached classification of neutral component crops for bootstrap training."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import threading
import time
import requests
from bootstrap_resnet_data import ROOT,OUT,ONTOLOGY,LAB,rows,write_rows,sha
from classify_reviewed_crops_qwen import prompt_for,build_payload,parse_items
from component_lab.candidate_verifier import credential,atomic_json,accounted_usage

def main(variant='context'):
    output=OUT/('teacher' if variant=='context' else 'teacher_confirm');output.mkdir(parents=True,exist_ok=True)
    jobs=rows(OUT/('teacher_jobs.jsonl' if variant=='context' else 'teacher_confirm_jobs.jsonl'));ontology=json.loads(ONTOLOGY.read_text(encoding='utf-8'))
    prompt=prompt_for(ontology);allowed=set(ontology['allowed_labels'])
    config={'model':'qwen3.8-flash','endpoint':'https://maas.qianwenaiapi.com/compatible-mode/v1',
        'crop_jobs_sha256':sha(OUT/('teacher_jobs.jsonl' if variant=='context' else 'teacher_confirm_jobs.jsonl')),'ontology_sha256':sha(ONTOLOGY),
        'batch_size':12,'max_tokens':2048,'max_requests':110,'token_budget':400000,'prompt':prompt,
        'variants':variant+' padded256','raw_class_or_gt_sent':False,'auto_paid_retry':False}
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text(encoding='utf-8'))!=config:
        raise ValueError('Existing teacher experiment differs')
    atomic_json(output/'config.json',config)
    key=credential(LAB.parent/'qwenapi.txt')
    lock=threading.Lock();stop=threading.Event()
    calls_dir=output/'calls';calls_dir.mkdir(exist_ok=True)
    def own_ledger():return [json.loads(p.read_text(encoding='utf-8')) for p in calls_dir.glob('*.json')]
    def ledger():return [json.loads(p.read_text(encoding='utf-8')) for folder in ('teacher','teacher_confirm') for p in (OUT/folder/'calls').glob('*.json')]
    def call(index,part):
        path=calls_dir/f'{index:04d}.json'
        if path.exists():return # Failed/interrupted attempts are never silently resent.
        payload,hashes=build_payload(ROOT.resolve(),part,variant,prompt,config['model'],config['max_tokens'])
        reservation=2048+2200+len(part)*110
        with lock:
            prior=ledger()
            if stop.is_set() or len(prior)>=110 or sum(r['accounted_tokens'] for r in prior)+reservation>400000:
                stop.set();return
            record={'index':index,'ids':[r['crop_id'] for r in part],'status':'started',
                'accounted_tokens':reservation,'usage_basis':'reservation','input_images':hashes,'proxy_labels_sent':False}
            atomic_json(path,record)
        start=time.perf_counter()
        try:
            with requests.Session() as session:
                session.trust_env=False
                response=session.post(config['endpoint']+'/chat/completions',headers={'Authorization':'Bearer '+key},
                    json=payload,timeout=(15,100),allow_redirects=False)
            record['http_status']=response.status_code
            if response.status_code!=200:
                record['status']='http_error'
                if response.status_code in {401,403,404,429}:stop.set()
            else:
                body=response.json();total,basis,usage=accounted_usage(body,reservation)
                choice=body['choices'][0];content=choice['message']['content']
                record.update(accounted_tokens=total,usage_basis=basis,usage=usage,returned_model=body.get('model'),
                    finish_reason=choice.get('finish_reason'),content=content.replace(key,'[REDACTED]'))
                items=[]
                if choice.get('finish_reason')=='stop':
                    try:
                        items=parse_items(content,record['ids'],allowed)
                        record['status']='ok'
                    except (ValueError,TypeError,KeyError):
                        # Extract individually valid, unique requested IDs from
                        # existing JSON only; no coordinate/semantic repair.
                        data=json.loads(content);raw=data.get('items',[]) if isinstance(data,dict) else []
                        ids=[x.get('id') for x in raw if isinstance(x,dict)]
                        for item in raw:
                            ident=item.get('id') if isinstance(item,dict) else None
                            if ident in record['ids'] and ids.count(ident)==1:
                                try:items.extend(parse_items(json.dumps({'items':[item]}),[ident],allowed))
                                except (ValueError,TypeError,KeyError):pass
                        record['status']='partial' if items else 'invalid_response'
                else:record['status']='incomplete'
                record['items']=items
        except Exception as error:
            record.update(status='request_or_parse_failed',error_type=type(error).__name__)
        record['seconds']=round(time.perf_counter()-start,3)
        with lock:atomic_json(path,record)
        print(json.dumps({k:record.get(k) for k in ('index','status','http_status','seconds','accounted_tokens')},ensure_ascii=False),flush=True)
    todo=[(i,jobs[start:start+12]) for i,start in enumerate(range(0,len(jobs),12))]
    with ThreadPoolExecutor(max_workers=2) as pool:
        iterator=iter(todo);pending=set()
        for _ in range(min(2,len(todo))):
            index,part=next(iterator);pending.add(pool.submit(call,index,part))
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                job=next(iterator,None) if not stop.is_set() else None
                if job:pending.add(pool.submit(call,*job))
    calls=own_ledger();predictions=[dict(item,call_index=r['index'],model_returned=r.get('returned_model')) for r in sorted(calls,key=lambda r:r['index']) for item in r.get('items',[])]
    write_rows(output/'predictions.jsonl',predictions)
    atomic_json(output/'summary.json',{'requested_crops':len(jobs),'classified':len(predictions),'requests':len(calls),
        'statuses':{status:sum(r['status']==status for r in calls) for status in {r['status'] for r in calls}},
        'accounted_tokens':sum(r['accounted_tokens'] for r in calls),
        'reported_usage':{k:sum(r.get('usage',{}).get(k,0) for r in calls) for k in ('prompt_tokens','completion_tokens','total_tokens')},
        'raw_labels_sent':False,'human_class_labels_created':False})
    print(json.dumps(json.loads((output/'summary.json').read_text(encoding='utf-8')),ensure_ascii=False),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['context','tight'],default='context')
    main(parser.parse_args().variant)
