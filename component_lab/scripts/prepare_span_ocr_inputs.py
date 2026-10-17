"""Build an isolated substring-candidate OCR view; original caches stay intact."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from component_lab.ocr_spans import split_ocr_spans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=ROOT/'outputs/component_identity_round_001_20261017/inputs')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new immutable output directory')
    manifest=json.loads((args.inputs/'manifest.json').read_text('utf-8'))
    allowed={x['case'] for x in json.loads((ROOT/'configs/contest6_group_split_v1.json').read_text('utf-8'))['partitions']['dev']}
    ids=[x['id'] for x in manifest['jobs']]
    if set(ids)!=allowed or len(ids)!=len(allowed) or not manifest.get('complete'):
        raise ValueError('Only complete unique frozen DEV20 is allowed')
    jobs=[];statistics=[]
    for job in manifest['jobs']:
        source=Path(job['ocr']);payload=json.loads(source.read_text('utf-8'))
        rows,audit=split_ocr_spans(payload['texts'])
        target=args.output.resolve()/job['id']/'ocr_spans.json';target.parent.mkdir(parents=True)
        result={**payload,'texts':rows,'span_view':{'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),**audit}}
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
        jobs.append({**job,'ocr':str(target)})
        statistics.append({'case':job['id'],'original_count':len(payload['texts']),'candidate_count':len(rows),'split_parents':audit['split_parent_count']})
    (args.output/'manifest.json').write_text(json.dumps({'jobs':jobs,'complete':True,'method':'conservative_raw_substring_replacement','uses_ground_truth':False,'statistics':statistics},ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(statistics,ensure_ascii=True))


if __name__=='__main__':main()
