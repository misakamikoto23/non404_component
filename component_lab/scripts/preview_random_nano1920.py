"""Reproducible random test examples, with original-coordinate error overlays."""
from pathlib import Path
import json
import random
import html
import os

from PIL import Image, ImageDraw
from analyze_yolo_errors import load_split, match_detections, count_summary, summarize


def main():
    lab = Path(__file__).resolve().parents[1]
    out = lab / 'outputs/nano1920_random_test_20261015'
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(lab / 'outputs/ultralytics_settings'))
    from ultralytics import YOLO
    weights = lab / 'weights/pcbnet_component_yolo26n_1920_v2.pt'
    names, population = load_split(lab / 'data/yolo/pcbnet_component_v1', 'test')
    records = random.Random(20261015).sample(population, 5)
    model = YOLO(str(weights))
    pages = []
    for record in records:
        ident = Path(record['image']).stem
        result = model.predict(record['image_path'], imgsz=1920, conf=.25, iou=.7,
                               max_det=1000, device=0, verbose=False)[0]
        record['predictions'] = [{'bbox': box, 'score': score, 'class_id': int(cls), 'type': 'component'}
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                      result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]
        record['matching'] = m = match_detections(record['ground_truth'], record['predictions'], .5, .25)
        counts = count_summary(len(m['matches']), len(m['false_positive_indices']), len(m['false_negative_indices']))
        record['counts'] = counts
        with Image.open(record['image_path']) as src:
            original = src.convert('RGB')
        prediction = original.copy()
        pd = ImageDraw.Draw(prediction)
        for p in record['predictions']:
            pd.rectangle(p['bbox'], outline='#125bd8', width=2)
            pd.text((p['bbox'][0], max(0, p['bbox'][1]-12)), f"{p['score']:.2f}", fill='#125bd8')
        prediction.save(out / f'{ident}_prediction.png')
        diagnostic = original.copy()
        dd = ImageDraw.Draw(diagnostic)
        for match in m['matches']:
            dd.rectangle(record['predictions'][match['prediction_index']]['bbox'], outline='#159447', width=2)
        errors = []
        for state, indices, rows, color in (
            ('FN', m['false_negative_indices'], record['ground_truth'], '#dc2424'),
            ('FP', m['false_positive_indices'], record['predictions'], '#e58700')):
            for j, idx in enumerate(indices, 1):
                row = rows[idx]
                label = f'{state}{j}'
                dd.rectangle(row['bbox'], outline=color, width=3)
                dd.text((row['bbox'][0], max(0, row['bbox'][1]-12)), label, fill=color)
                errors.append((label, row, color))
        diagnostic.save(out / f'{ident}_diagnostic.png')
        # Error crops are selected deterministically, misses first, at most 8.
        panels = Image.new('RGB', (1280, 650), 'white')
        ImageDraw.Draw(panels).text((10, 5), f'{ident}: red=missed GT; orange=unmatched prediction. IoU >= 0.5. Crops show context.', fill='black')
        for k, (label, row, color) in enumerate(errors[:8]):
            x1, y1, x2, y2 = row['bbox']
            margin = max(35, .4 * max(x2-x1, y2-y1))
            rect = (max(0, int(x1-margin)), max(0, int(y1-margin)),
                    min(original.width, int(x2+margin)), min(original.height, int(y2+margin)))
            patch = original.crop(rect)
            d = ImageDraw.Draw(patch)
            d.rectangle((x1-rect[0], y1-rect[1], x2-rect[0], y2-rect[1]), outline=color, width=2)
            patch.thumbnail((310, 275))
            px, py = (k % 4)*320, 30+(k//4)*310
            panels.paste(patch, (px+(320-patch.width)//2, py+25))
            ImageDraw.Draw(panels).text((px+8, py+5), label + (f" conf={row['score']:.2f}" if 'score' in row else ''), fill=color)
        if not errors:
            ImageDraw.Draw(panels).text((20, 60), 'No errors at this matching threshold.', fill='black')
        panels.save(out / f'{ident}_details.png')
        title = f"{ident} | {original.width}x{original.height} | GT {counts['ground_truth']} | TP {counts['tp']} FP {counts['fp']} FN {counts['fn']}"
        pages.append(f'<h2>{html.escape(title)}</h2><p>Green: matched prediction; red: missed annotation; orange: unmatched prediction. Click for full resolution.</p><a href="{ident}_diagnostic.png"><img src="{ident}_diagnostic.png"></a><img src="{ident}_details.png"><p><a href="{ident}_prediction.png">All predictions and confidence</a></p>')
        print(json.dumps({'id': ident, 'width': original.width, 'height': original.height, **counts}), flush=True)
    report = {'weights': str(weights), 'seed': 20261015, 'split': 'test', 'population': len(population),
              'imgsz': 1920, 'confidence': .25, 'matching_iou': .5, 'nms_iou_argument': .7,
              'tiling': False, 'metrics': summarize(records, names, 1920), 'records': records,
              'note': 'Random qualitative sample, not full-test mAP or contest ComponentF1. Errors are relative to provided labels.'}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Nano 1920 random test</title><style>body{font-family:Arial;max-width:1500px;margin:24px auto;background:#eee}img{max-width:100%;background:white}h2{margin-top:50px}</style><h1>Nano 1920: 5 random test images</h1><p>Seed 20261015; whole image; confidence .25; matching IoU .5; 1920 fine-tuned Nano weights. No training performed.</p>'+''.join(pages), encoding='utf-8')
    print(json.dumps(report['metrics']['overall']))


if __name__ == '__main__':
    main()
