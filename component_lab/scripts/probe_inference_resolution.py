"""Compare 1280/1920 single-image Nano inference on val; stop on failure."""
import argparse
import json
import os
import time
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('sizes must be positive integers')
    return value


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=root / 'weights/pcbnet_component_yolo26n_v1.pt')
    parser.add_argument('--output', type=Path, default=root / 'outputs/pcbnet_component_resolution_probe')
    parser.add_argument('--sizes', type=positive_int, nargs='+', default=[1280, 1920])
    parser.add_argument('--overwrite', action='store_true', help='allow replacing an existing summary.json')
    args = parser.parse_args()

    weights = args.weights.resolve()
    out = args.output.resolve()
    summary = out / 'summary.json'
    if summary.exists() and not args.overwrite:
        parser.error(f'{summary} already exists; choose a new --output or pass --overwrite')
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(root / 'outputs/ultralytics_settings'))
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from analyze_yolo_errors import load_validation, main as diagnose

    torch.set_num_threads(4)
    data = root / 'data/yolo/pcbnet_component_v1'
    names, records = load_validation(data)
    results = {'device': torch.cuda.get_device_name(0), 'torch_cuda_total_mib': torch.cuda.get_device_properties(0).total_memory / 1024**2,
               'split': 'val', 'batch': 1, 'confidence': .25, 'weights': str(weights), 'runs': []}
    model = YOLO(str(weights))
    model_names = model.names
    if isinstance(model_names, dict):
        model_names = [model_names[index] for index in sorted(model_names)]
    else:
        model_names = list(model_names)
    if model_names != ['component']:
        raise ValueError(f"expected model names ['component'], got {model_names!r}")
    current = None
    try:
        for size in args.sizes:
            current = size
            directory = out / str(size)
            directory.mkdir(exist_ok=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Exercise the largest square tensor before processing any val case.
            warm = model.predict(Image.new('RGB', (size,size), 'white'), imgsz=size, device='0',
                                 conf=.25, max_det=1000, verbose=False)
            del warm
            torch.cuda.synchronize()
            predictions, times = {}, []
            for index, record in enumerate(records, 1):
                with Image.open(record['image_path']) as source:
                    image = source.convert('RGB')
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = model.predict(image, imgsz=size, device='0', conf=.25, max_det=1000, verbose=False)[0]
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
                predictions[record['image']] = [{'bbox':box, 'class_id':int(cls), 'type':names[int(cls)], 'score':score}
                    for box,cls,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist())]
                print(f'{size}: {index}/{len(records)}', flush=True)
                del result
            cache = directory / 'predictions.json'
            cache.write_text(json.dumps({'schema_version':1,'split':'val','names':names,'confidence':.25,
                             'imgsz':size,'predictions':predictions},ensure_ascii=False),encoding='utf-8')
            diagnose(['--weights',str(weights),'--data-root',str(data),'--predictions-json',str(cache),
                      '--output',str(directory/'diagnostics'),'--imgsz',str(size),'--confidence','.25','--split','val'])
            metrics = json.loads((directory/'diagnostics/metrics.json').read_text(encoding='utf-8'))
            results['runs'].append({'imgsz':size,'images':len(records),'allocated_peak_mib':torch.cuda.max_memory_allocated()/1024**2,
                'reserved_peak_mib':torch.cuda.max_memory_reserved()/1024**2,'seconds_excluding_warmup_and_file_load':sum(times),
                'mean_ms':1000*sum(times)/len(times),'metrics':metrics['overall']})
            summary.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception as exc:
        results.update(status='failed_stopped', failed_imgsz=current,error_type=type(exc).__name__,error=str(exc))
        summary.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(results,ensure_ascii=False),flush=True)
        raise
    results['status']='complete'
    summary.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
