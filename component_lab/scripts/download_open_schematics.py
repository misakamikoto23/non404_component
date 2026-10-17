"""Download a version-pinned schematic-only subset without PCB binaries.

Reads selected Parquet columns using validated HTTP ranges, not HF credentials.
Each accepted row has a KiCad source and a decodable rendered schematic image.
No third-party source files are executed. Outputs remain isolated from training.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import quote

import requests
import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError


class DownloadBudgetExceeded(RuntimeError):
    pass


class TransferBudget:
    def __init__(self,maximum):self.maximum=maximum;self.used=0;self.requests=0
    def account(self,count):
        if self.used+count>self.maximum:raise DownloadBudgetExceeded('Network byte budget reached; resume with a higher --max-download-gb.')
        self.used+=count


class RangeReader(io.RawIOBase):
    """Seekable, strictly validated range reader for public HF Parquet files."""
    def __init__(self,session,url,size,budget):
        self.session,self.url,self.size,self.budget=session,url,size,budget;self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        if whence==0:target=offset
        elif whence==1:target=self.position+offset
        elif whence==2:target=self.size+offset
        else:raise ValueError('Invalid seek mode')
        if target<0:raise ValueError('Negative seek')
        self.position=target;return target
    def read(self,size=-1):
        if size is None or size<0:size=self.size-self.position
        count=min(size,max(0,self.size-self.position))
        if not count:return b''
        start,end=self.position,self.position+count-1
        if self.budget.used+count>self.budget.maximum:raise DownloadBudgetExceeded('Network byte budget reached')
        for attempt in range(3):
            try:
                # A range-specific query avoids mis-keyed intermediary caches.
                response=self.session.get(self.url+f'?download=true&range_start={start}&range_end={end}',
                    headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(20,90))
                self.budget.requests+=1
                with response:
                    response.raise_for_status()
                    expected=f'bytes {start}-{end}/{self.size}'
                    if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError(f'Range response invalid: HTTP {response.status_code}, {response.headers.get("Content-Range")}')
                    chunks=[];received=0
                    for chunk in response.iter_content(1024*1024):
                        self.budget.account(len(chunk));received+=len(chunk)
                        if received>count:raise ValueError('Range response too long')
                        chunks.append(chunk)
                    if received!=count:raise ValueError('Range response truncated')
                self.position+=count
                return b''.join(chunks)
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(1+attempt)
        raise RuntimeError('Unreachable')
    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temporary,path)


def fetch_json(session,url):
    for attempt in range(3):
        try:
            response=session.get(url,timeout=(20,60));response.raise_for_status();return response.json()
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(attempt+1)


def save_row(output,row,shard,row_number,repo,revision,seen):
    source=row.get('schematic')
    if not isinstance(source,str) or not source.lstrip().startswith('(kicad_sch'):
        return None,'not_modern_kicad'
    if '(lib_symbols' not in source or not re.search(r'\(lib_id\s+"',source):
        return None,'missing_library_or_instances'
    raw_image=row.get('schematic_image')
    image_bytes=raw_image.get('bytes') if isinstance(raw_image,dict) else None
    if not image_bytes:return None,'no_embedded_schematic_image'
    source_hash=hashlib.sha256(source.encode('utf-8')).hexdigest()
    if source_hash in seen['source']:return None,'duplicate_source'
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load();image=image.convert('RGB');w,h=image.size
            if min(w,h)<64:return None,'image_too_small'
            pixel_hash=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    except (UnidentifiedImageError,OSError,ValueError):return None,'invalid_image'
    if pixel_hash in seen['image']:return None,'duplicate_image'
    ident=f'{Path(shard).stem}_{row_number:05d}_{source_hash[:10]}'
    folder=output/'samples'/ident;folder.mkdir(parents=True,exist_ok=True)
    (folder/'schematic.kicad_sch').write_bytes(source.encode('utf-8'))
    # Store the dataset's original image bytes, not a recompressed thumbnail.
    with Image.open(io.BytesIO(image_bytes)) as check:ext=(check.format or 'PNG').lower()
    ext={'jpeg':'jpg'}.get(ext,ext)
    (folder/f'schematic.{ext}').write_bytes(image_bytes)
    metadata=row.get('schematic_json')
    json_valid=False
    if isinstance(metadata,str) and metadata:
        try:
            parsed=json.loads(metadata);atomic_json(folder/'schematic_metadata.json',parsed);json_valid=True
        except json.JSONDecodeError:
            (folder/'schematic_metadata_raw.txt').write_text(metadata,encoding='utf-8')
    symbols=row.get('components_used') or []
    info={'id':ident,'dataset':repo,'revision':revision,'parquet':shard,'row_in_parquet':row_number,
        'source_repository':row.get('name'),'description':row.get('description'),
        'extensions_used':row.get('extensions_used'),'component_library_names':symbols,
        'placed_symbol_count':len(re.findall(r'\(lib_id\s+"',source)),
        'width':w,'height':h,'source_sha256':source_hash,'image_sha256':hashlib.sha256(image_bytes).hexdigest(),
        'rgb_sha256':pixel_hash,'structured_json_valid':json_valid,
        'source_file':f'samples/{ident}/schematic.kicad_sch','image_file':f'samples/{ident}/schematic.{ext}',
        'source_project_url':f"https://github.com/{row['name']}" if isinstance(row.get('name'),str) and re.fullmatch(r'[^/\s]+/[^/\s]+',row['name']) else None,
        'dataset_license':'CC-BY-4.0; original project terms must also be retained',
        'label_status':'raw_source_and_render_only_not_yolo_labels','split':'unassigned'}
    atomic_json(folder/'source.json',info)
    seen['source'].add(source_hash);seen['image'].add(pixel_hash)
    return info,None


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.count<1 or args.max_download_gb<=0:raise ValueError('Positive count and byte budget required')
    session=requests.Session();session.trust_env=False
    session.headers.update({'User-Agent':'schematic-component-lab-public-data-downloader/1.0'})
    state_path=output/'download_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if state and state['dataset']!=args.repo:raise ValueError('Output belongs to a different dataset')
    if state and 'next_row_offset' not in state:
        # Old checkpoints pointed past a partially consumed group. Replay at
        # most one group, with existing hashes preventing duplicate imports.
        state['next_row_group']=max(0,state['next_row_group']-1)
        state['next_row_offset']=0
    info=fetch_json(session,f'https://huggingface.co/api/datasets/{args.repo}')
    revision=state['revision'] if state else info['sha']
    if not state:
        state={'dataset':args.repo,'revision':revision,'target_count':args.count,'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'next_shard':0,'next_row_group':0,'next_row_offset':0,'records':[],'skipped':{},'network_bytes_total':0,'completed':False}
        card=session.get(f'https://huggingface.co/datasets/{args.repo}/raw/{revision}/README.md',timeout=40);card.raise_for_status();(output/'DATASET_CARD.md').write_text(card.text,encoding='utf-8')
        atomic_json(state_path,state)
    listing=fetch_json(session,f'https://huggingface.co/api/datasets/{args.repo}/tree/{revision}/data?recursive=false&expand=false&limit=1000')
    files=sorted((r for r in listing if r['type']=='file' and r['path'].endswith('.parquet')),key=lambda r:r['path'])
    (output/'dataset_files.json').write_text(json.dumps({'dataset':args.repo,'revision':revision,'files':files},ensure_ascii=False,indent=2),encoding='utf-8')
    seen={'source':{r['source_sha256'] for r in state['records']},'image':{r['rgb_sha256'] for r in state['records']}}
    budget=TransferBudget(int(args.max_download_gb*1024**3))
    initial_network=state['network_bytes_total'];state['target_count']=args.count
    columns=['schematic','schematic_image','schematic_json','components_used','name','description','extensions_used']
    try:
        for shard_index in range(state['next_shard'],len(files)):
            if len(state['records'])>=args.count:break
            if shutil.disk_usage(output).free<5*1024**3:raise DownloadBudgetExceeded('Less than 5 GiB free; stopped before filling disk')
            shard=files[shard_index]
            url=f'https://huggingface.co/datasets/{args.repo}/resolve/{revision}/{quote(shard["path"],safe="/")}'
            reader=RangeReader(session,url,shard['size'],budget)
            parquet=pq.ParquetFile(reader)
            group_start=state['next_row_group'] if shard_index==state['next_shard'] else 0
            offset_start=state['next_row_offset'] if shard_index==state['next_shard'] else 0
            row_base=sum(parquet.metadata.row_group(i).num_rows for i in range(group_start))
            for group in range(group_start,parquet.num_row_groups):
                table=parquet.read_row_group(group,columns=columns,use_threads=False)
                rows=table.to_pylist()
                processed=offset_start if group==group_start else 0
                for local in range(processed,len(rows)):
                    if len(state['records'])>=args.count:break
                    row=rows[local]
                    accepted,reason=save_row(output,row,shard['path'],row_base+local,args.repo,revision,seen)
                    if accepted:state['records'].append(accepted)
                    else:state['skipped'][reason]=state['skipped'].get(reason,0)+1
                    processed=local+1
                row_base+=parquet.metadata.row_group(group).num_rows
                state.update(next_shard=shard_index,next_row_group=group+1 if processed==len(rows) else group,
                    next_row_offset=0 if processed==len(rows) else processed,network_bytes_total=initial_network+budget.used)
                atomic_json(state_path,state)
                print(json.dumps({'accepted':len(state['records']),'target':args.count,'shard':shard['path'],'row_group':group,
                    'downloaded_mib':round(budget.used/1024**2,2),'skipped':state['skipped']},ensure_ascii=False),flush=True)
                if len(state['records'])>=args.count:break
            if len(state['records'])<args.count:
                state.update(next_shard=shard_index+1,next_row_group=0,next_row_offset=0);atomic_json(state_path,state)
        state['completed']=len(state['records'])>=args.count
        state['status']='complete' if state['completed'] else 'dataset_exhausted'
        state.pop('error_type',None);state.pop('error',None)
    except Exception as exc:
        state.update(status='stopped',error_type=type(exc).__name__,error=str(exc)[:500])
        raise
    finally:
        state['network_bytes_total']=initial_network+budget.used
        state['updated_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        atomic_json(state_path,state)
        with (output/'manifest.jsonl').open('w',encoding='utf-8') as handle:
            for record in state['records']:handle.write(json.dumps(record,ensure_ascii=False)+'\n')
        summary={k:v for k,v in state.items() if k!='records'}
        summary.update(accepted_records=len(state['records']),source_projects=len({r['source_repository'] for r in state['records']}),
            placed_symbols=sum(r['placed_symbol_count'] for r in state['records']),
            valid_structured_json=sum(r['structured_json_valid'] for r in state['records']),
            selected_columns=columns,pcb_data_downloaded=False,yolo_labels_generated=False,training_started=False)
        atomic_json(output/'download_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='bshada/open-schematics')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'Data/open_schematics')
    parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--max-download-gb',type=float,default=4)
    run(parser.parse_args())
