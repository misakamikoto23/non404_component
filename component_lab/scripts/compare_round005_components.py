"""Paired dev20: change detector only; freeze OCR, classifier and association."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / 'outputs/active_learning/round_005/component_comparison'
sys.path.insert(0,str(LAB/'src'))
from component_lab.models import YoloDetector, ResnetClassifier
from component_lab.review_workflow import atomic_json
import run_component_identity_experiment as experiment


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    assert not OUT.exists(), 'No overwrite or automatic duplicate experiment'
    assert '3050 Ti' in torch.cuda.get_device_name(0), 'Require current 3050 Ti'
    torch.set_num_threads(4)
    OUT.mkdir(parents=True)
    manifest = read(LAB/'outputs/unattended_20261017_lunch/span_inputs/manifest.json')
    split = read(LAB/'configs/contest6_group_split_v1.json')
    experiment.validate_manifest(manifest,{c['case'] for c in split['partitions']['dev']})
    classifier_path = LAB/'weights/resnet18_component_balanced_v1.pt'
    classifier = ResnetClassifier(str(classifier_path),'0')
    atomic_json(OUT/'PLAN.json',dict(
        method='Pair round004 vs round005 on the same3050, OCR spans, balancedResNet, crop margin8%/2px, classifier threshold.8, identity-v1 plus display-names.',
        scope='Fixed development comparison; historical checkpoint exposure, not independent test.',
        api_calls=0,test_used=False,selection_uses_gt=False))
    # Freeze both arms for every case before the evaluator reads any target JSON.
    for arm in ['round004','round005']:
        weights=LAB/'weights'/f'pcbnet_component_yolo26n_review_{arm.replace("round","round_")}.pt'
        detector=YoloDetector(str(weights),'0',1280,.25,tile_size=1024,tile_overlap=.25)
        jobs=[]
        for job in manifest['jobs']:
            started=time.perf_counter()
            image=Image.open(job['image']).convert('RGB')
            rows=detector.predict(image)
            for i,row in enumerate(rows,1):
                x1,y1,x2,y2=row['bbox'];dx=max(2,.08*(x2-x1));dy=max(2,.08*(y2-y1))
                box=[max(0,int(x1-dx)),max(0,int(y1-dy)),min(image.width,int(x2+dx+1)),min(image.height,int(y2+dy+1))]
                prediction=classifier.predict(image.crop(box))
                row.update(id=f'component_{i:04d}',classification=prediction,classification_crop=box,
                           type=prediction['type'] if prediction['score']>=.8 else 'component')
            target=OUT/arm/'inputs'/job['id']/'components.json'
            target.parent.mkdir(parents=True)
            atomic_json(target,dict(components=rows,provenance=dict(
                yolo_sha256=sha(weights),resnet_sha256=sha(classifier_path),
                image_sha256=sha(Path(job['image'])),ocr_sha256=sha(Path(job['ocr'])),
                uses_gt_for_prediction=False),seconds=time.perf_counter()-started))
            jobs.append(dict(job,detections=str(target)))
            print(arm,job['id'],len(rows),'detections',flush=True)
        atomic_json(OUT/arm/'inputs/manifest.json',dict(jobs=jobs,complete=True))
        del detector
        gc.collect();torch.cuda.empty_cache()
    original_argv=sys.argv
    for arm in ['round004','round005']:
        sys.argv=['run_component_identity_experiment.py','--inputs',str(OUT/arm/'inputs'),
                  '--output',str(OUT/arm/'scored'),'--display-names']
        experiment.main()
    sys.argv=original_argv
    summaries={}
    for arm in ['round004','round005']:
        report=read(OUT/arm/'scored/report.json')
        summaries[arm]=next(s for s in report['summaries'] if s['variant']=='identity_v1' and s['bbox_metric']=='max_corner')
    atomic_json(OUT/'comparison.json',dict(summaries=summaries,
        delta_f1_percentage_points=100*(summaries['round005']['macro_component_f1']-summaries['round004']['macro_component_f1']),
        api_calls=0,test_used=False,default_replaced=False))
    print(json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
