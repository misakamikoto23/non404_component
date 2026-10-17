"""Offline report for the authorized, bounded real-image Qwen pilot."""
from pathlib import Path
import json
import html
from PIL import Image, ImageDraw
from analyze_yolo_errors import count_summary


def main():
    lab=Path(__file__).resolve().parents[1]
    out=lab/'outputs/qwen38flash_real_pilot_20261015'
    data=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    baseline=json.loads((lab/'outputs/nano1920_random_test_20261015/report.json').read_text(encoding='utf-8'))
    original={Path(r['image']).stem:r for r in baseline['records'] if Path(r['image']).stem in data['ids']}
    if len(data['calls'])!=6 or any('metrics' not in r for r in data['calls']):
        raise ValueError('Pilot incomplete; report does not silently exclude failed requests.')
    totals={}
    for mode in ('independent','yolo_review'):
        totals[mode]={}
        records=[r for r in data['calls'] if r['mode']==mode]
        for threshold in ('0.3','0.5','0.7'):
            s=count_summary(*(sum(r['metrics'][threshold][k] for r in records) for k in ('tp','fp','fn')))
            s['f1']=2*s['tp']/(2*s['tp']+s['fp']+s['fn']) if 2*s['tp']+s['fp']+s['fn'] else 0
            totals[mode][threshold]=s
    yolo=count_summary(*(sum(r['counts'][k] for r in original.values()) for k in ('tp','fp','fn')))
    yolo['f1']=2*yolo['tp']/(2*yolo['tp']+yolo['fp']+yolo['fn'])
    usage={k:sum((r.get('usage') or {}).get(k,0) for r in data['calls']) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    result={'image_count':len(original),'model_requested':data['model_requested'],
        'models_returned':sorted({r['model_returned'] for r in data['calls']}),
        'totals':totals,'yolo_at_iou_0_5':yolo,'usage':usage,
        'latency_seconds_total':sum(r['latency_seconds'] for r in data['calls']),
        'latency_seconds_range':[min(r['latency_seconds'] for r in data['calls']),max(r['latency_seconds'] for r in data['calls'])],
        'warnings':['Original PCBnet annotations are imperfect and may follow a different box convention.',
            'YOLO review sees YOLO proposals; it is not independent annotation.',
            'Scores are micro-aggregated single-class bbox counts, not mAP or contest ComponentF1.',
            'No accuracy score for type, reference, or OCR is computed.',
            'Token usage is measured; actual bill and endpoint price are unavailable.',
            'Three test examples are a feasibility pilot; no training or label replacement performed.']}
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    parts=['<!doctype html><meta charset="utf-8"><title>Qwen real schematic pilot</title><style>body{font-family:Arial;margin:25px;background:#eee}img{max-width:100%;background:white}td,th{padding:8px}table.images{width:100%;table-layout:fixed}table.images td{vertical-align:top}h2{margin-top:40px}</style><h1>YOLO vs Qwen3.8-Flash: 3 real schematic images</h1><p>Green=matched prediction; orange=unmatched prediction; red=missed GT. IoU .5. Original labels may contain errors. This is bbox evaluation, not OCR/type accuracy or contest ComponentF1.</p>']
    table=['<table><tr><th>Method</th><th>TP/FP/FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr>']
    for name,s in [('YOLO Nano 1920',yolo),('Qwen independent',totals['independent']['0.5']),('Qwen reviews YOLO proposals',totals['yolo_review']['0.5'])]:
        table.append(f"<tr><td>{name}</td><td>{s['tp']}/{s['fp']}/{s['fn']}</td><td>{s['precision']:.1%}</td><td>{s['recall']:.1%}</td><td>{s['f1']:.1%}</td></tr>")
    table.append('</table>');parts+=table
    for ident,record in original.items():
        # Copy the existing YOLO diagnostic locally for a self-contained gallery.
        src=lab/'outputs/nano1920_random_test_20261015'/f'{ident}_diagnostic.png'
        with Image.open(src) as im: im.save(out/f'{ident}_yolo_diagnostic.png')
        # Original-resolution per-model files are linked; compare complete images here.
        panels=[]
        for label,name in [('YOLO','yolo'),('Qwen independent','independent'),('Qwen review','yolo_review')]:
            file=f'{ident}_{name}_diagnostic.png'
            with Image.open(out/file) as im:
                im=im.convert('RGB');im.thumbnail((900,800))
                panel=Image.new('RGB',(920,840),'white');panel.paste(im,((920-im.width)//2,35))
                ImageDraw.Draw(panel).text((10,10),ident+' '+label,fill='black');panels.append(panel)
        canvas=Image.new('RGB',(2760,840),'white')
        for index,panel in enumerate(panels):canvas.paste(panel,(index*920,0))
        canvas.save(out/f'{ident}_comparison.png')
        parts.append(f'<h2>{ident}</h2><a href="{ident}_comparison.png"><img src="{ident}_comparison.png"></a><p>Full resolution: <a href="{ident}_yolo_diagnostic.png">YOLO</a> | <a href="{ident}_independent_diagnostic.png">Qwen independent</a> | <a href="{ident}_yolo_review_diagnostic.png">Qwen review</a></p>')
    parts.append('<h2>Measured usage and limitations</h2><pre>'+html.escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>')
    (out/'gallery.html').write_text(''.join(parts),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
