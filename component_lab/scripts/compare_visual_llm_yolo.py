"""Compare independent visual annotations against the same labels as YOLO."""
from pathlib import Path
import json
import html
import argparse
from PIL import Image, ImageDraw
from analyze_yolo_errors import match_detections, count_summary, valid_box


def main():
    lab = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=lab / 'data/ai_annotations_nano_comparison_20261015/llm_annotations.json')
    parser.add_argument('--output', type=Path, default=lab / 'outputs/nano1920_vs_visual_llm_20261015')
    args = parser.parse_args()
    source = lab / 'outputs/nano1920_random_test_20261015/report.json'
    annotations = args.annotations.resolve()
    baseline = json.loads(source.read_text(encoding='utf-8-sig'))
    payload = json.loads(annotations.read_text(encoding='utf-8-sig'))
    llm = {str(row['id']).zfill(4): row for row in payload['images']}
    model_label = f"{payload['model']} {payload['reasoning_effort']} visual"
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, pages = [], []
    for record in baseline['records']:
        ident = Path(record['image']).stem
        annotation = llm[ident]
        if (annotation['width'], annotation['height']) != (record['width'], record['height']):
            raise ValueError(f'Wrong image dimensions for {ident}')
        preds = []
        for component in annotation['components']:
            bbox = valid_box(component['bbox'])
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > record['width'] or bbox[3] > record['height']:
                raise ValueError(f'Out-of-image annotation {ident}: {bbox}')
            # All manually proposed visual boxes are evaluated; confidence is not calibrated.
            preds.append({**component, 'bbox': bbox, 'class_id': 0, 'score': 1.0})
        match = match_detections(record['ground_truth'], preds, .5, 0)
        stats = count_summary(len(match['matches']), len(match['false_positive_indices']), len(match['false_negative_indices']))
        row = {'id': ident, 'yolo': record['counts'], 'llm': stats,
               'llm_predictions': preds, 'llm_matching': match}
        row['iou_sensitivity'] = {}
        for threshold in (.3, .5, .7):
            row['iou_sensitivity'][str(threshold)] = {}
            for name, candidates, conf in (('yolo', record['predictions'], .25), ('llm', preds, 0)):
                matched = match_detections(record['ground_truth'], candidates, threshold, conf)
                row['iou_sensitivity'][str(threshold)][name] = count_summary(
                    len(matched['matches']), len(matched['false_positive_indices']), len(matched['false_negative_indices']))
        rows.append(row)
        with Image.open(record['image_path']) as src:
            original = src.convert('RGB')
        panels = []
        for label, boxes, matching, counts in (
            ('YOLO26 Nano 1920', record['predictions'], record['matching'], record['counts']),
            (model_label, preds, match, stats)):
            im = original.copy()
            d = ImageDraw.Draw(im)
            for m in matching['matches']:
                d.rectangle(boxes[m['prediction_index']]['bbox'], outline='#159447', width=2)
            for idx in matching['false_positive_indices']:
                d.rectangle(boxes[idx]['bbox'], outline='#e58700', width=3)
            for idx in matching['false_negative_indices']:
                d.rectangle(record['ground_truth'][idx]['bbox'], outline='#dc2424', width=3)
            im.save(out / f'{ident}_{"yolo" if label.startswith("YOLO") else "llm"}.png')
            im.thumbnail((900, 800))
            panel = Image.new('RGB', (920, 850), 'white')
            panel.paste(im, ((920-im.width)//2, 45))
            ImageDraw.Draw(panel).text((12, 12), f"{ident}: {label} | TP {counts['tp']} FP {counts['fp']} FN {counts['fn']}", fill='black')
            panels.append(panel)
        canvas = Image.new('RGB', (1840, 850), 'white')
        canvas.paste(panels[0], (0, 0)); canvas.paste(panels[1], (920, 0))
        canvas.save(out / f'{ident}_comparison.png')
        pages.append(f'<h2>{html.escape(ident)}</h2><a href="{ident}_comparison.png"><img src="{ident}_comparison.png"></a><p><a href="{ident}_yolo.png">YOLO full resolution</a> | <a href="{ident}_llm.png">LLM full resolution</a></p>')
    totals = {}
    for model in ('yolo', 'llm'):
        totals[model] = count_summary(*(sum(row[model][key] for row in rows) for key in ('tp','fp','fn')))
        for row in rows:
            s = row[model]
            s['f1'] = 2*s['tp']/(2*s['tp']+s['fp']+s['fn']) if 2*s['tp']+s['fp']+s['fn'] else 0
        s = totals[model]
        s['f1'] = 2*s['tp']/(2*s['tp']+s['fp']+s['fn'])
    sensitivity = {}
    for threshold in ('0.3', '0.5', '0.7'):
        sensitivity[threshold] = {model: count_summary(*(sum(row['iou_sensitivity'][threshold][model][key] for row in rows)
            for key in ('tp', 'fp', 'fn'))) for model in ('yolo', 'llm')}
    result = {'model': payload['model'], 'reasoning_effort': payload['reasoning_effort'],
              'annotation_source': str(annotations), 'totals': totals, 'images': rows, 'iou': .5, 'iou_sensitivity': sensitivity,
              'note': 'Single-class bbox evaluation on 5 previously selected test examples. LLM used independent visual inspection and may use crops/tools; not a one-call API benchmark. Not contest ComponentF1. LLM confidence not filtered; YOLO conf .25. Ground-truth quality is not assumed perfect.'}
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    table = '<table><tr><th>Image</th><th>YOLO TP/FP/FN</th><th>LLM TP/FP/FN</th></tr>'+''.join(f"<tr><td>{r['id']}</td><td>{r['yolo']['tp']}/{r['yolo']['fp']}/{r['yolo']['fn']}</td><td>{r['llm']['tp']}/{r['llm']['fp']}/{r['llm']['fn']}</td></tr>" for r in rows)+'</table>'
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>YOLO vs visual LLM</title><style>body{font-family:Arial;margin:25px;background:#eee}img{max-width:100%}td,th{padding:10px}h2{margin-top:45px}</style><h1>YOLO26 Nano vs '+html.escape(model_label)+'</h1><p>Green=matched prediction; orange=unmatched prediction; red=missed GT. Matching IoU .5. This is bbox matching, not contest ComponentF1.</p>'+table+''.join(pages),encoding='utf-8')
    print(json.dumps({'totals':totals,'per_image':[{k:v for k,v in r.items() if k in ('id','yolo','llm')} for r in rows]},indent=2))


if __name__ == '__main__':
    main()
