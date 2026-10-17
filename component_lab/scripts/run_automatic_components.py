"""One-command local detection, traceable candidate queue and optional bounded Qwen shadow."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

LAB = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def command(stage, args, root):
    print(json.dumps({'stage':stage,'status':'running'},ensure_ascii=False),flush=True)
    with (root/f'{stage}.log').open('a',encoding='utf-8') as log:
        result = subprocess.run([sys.executable,'-u',*map(str,args)],cwd=LAB,
                                stdout=log,stderr=subprocess.STDOUT)
    return result.returncode


def render_gallery(candidates, output, qwen_status):
    raw = read(candidates/'raw_candidates.json')
    baseline = read(candidates/'final_predictions.json')['predictions']
    policy = read(candidates/'policy_pre_qwen_predictions.json')['predictions']
    cards = []
    for key, meta in raw['images'].items():
        layers = []
        for name, predictions in [('candidate',policy[key]),('baseline',baseline[key])]:
            for row in predictions:
                x1,y1,x2,y2 = row['bbox']
                ident = row.get('candidate_id',row.get('id',''))
                title = html.escape(f'{ident} / {row["score"]:.3f}')
                layers.append(f'<rect class="{name}" x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}"><title>{title}</title></rect>')
        uri = html.escape(Path(meta['image_path']).as_uri(),quote=True)
        cards.append('<article><h2>'+html.escape(Path(meta['image_path']).name)+
            f' · YOLO {len(baseline[key])}框 · 候选{len(policy[key])}框</h2>'+
            f'<svg viewBox="0 0 {meta["width"]} {meta["height"]}" xmlns="http://www.w3.org/2000/svg">'+
            f'<image href="{uri}" width="{meta["width"]}" height="{meta["height"]}"/>'+''.join(layers)+'</svg></article>')
    text = '<!doctype html><meta charset="utf-8"><title>自动元件处理结果</title><style>'
    text += 'body{font:16px Arial,sans-serif;background:#f4f6f8;margin:24px}header{position:sticky;top:0;background:white;padding:16px;z-index:2;border:1px solid #ddd}article{background:white;padding:16px;margin:20px 0}svg{width:100%;max-height:85vh}rect{fill:none;stroke-width:1.5;vector-effect:non-scaling-stroke}.baseline{stroke:#1664d7}.candidate{stroke:#e28a16;display:none}.show_candidates .candidate{display:block}.hide_baseline .baseline{display:none}</style>'
    text += '<header><b>自动元件处理结果</b><p>蓝框：YOLO基线输出。橙框：自动候选，包含待复核项；均不是人工真值。悬停框可看ID和分数。</p>'
    text += '<label><input type="checkbox" checked onchange="document.body.classList.toggle(\'hide_baseline\',!this.checked)">显示YOLO基线</label> '
    text += '<label><input type="checkbox" onchange="document.body.classList.toggle(\'show_candidates\',this.checked)">显示自动候选</label>'
    text += '<p>Qwen状态：'+html.escape(qwen_status)+'。本阶段Qwen建议单独保存，正式输出保持现有基线。</p></header>'
    (output/'gallery.html').write_text(text+''.join(cards),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True,help='One image or a directory; labels are not required')
    p.add_argument('--weights',type=Path,default=LAB/'weights/pcbnet_component_yolo26n_review_round_003.pt')
    p.add_argument('--output',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--baseline-confidence',type=float,default=.25)
    p.add_argument('--open',action='store_true',help='Open the local result gallery after processing')
    p.add_argument('--qwen',action='store_true',help='Send a bounded subset of local crops for shadow suggestions')
    p.add_argument('--key-file',type=Path,default=LAB.parent/'qwenapi.txt')
    p.add_argument('--max-calls',type=int,default=8)
    p.add_argument('--token-budget',type=int,default=20000)
    args=p.parse_args()
    images,weights=args.images.resolve(),args.weights.resolve()
    if not images.exists() or not weights.is_file() or args.max_calls<1 or not .03<=args.baseline_confidence<=1:
        raise ValueError('Input/weights missing or invalid request budget')
    output=(args.output or LAB/'outputs/auto_inference'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    # Prevent generated previews from becoming future input images on directory runs.
    if images.is_dir() and output.is_relative_to(images):
        raise ValueError('Output must not be inside the input image directory')
    output.mkdir(parents=True,exist_ok=True)
    settings={'images':str(images),'weights':str(weights),
              'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
              'device':args.device,'confidence':.03,'baseline_confidence':args.baseline_confidence,'tile_size':1024,'overlap':.25}
    config_path=output/'workflow_config.json'
    if config_path.exists() and read(config_path)!=settings:
        raise ValueError('Output belongs to a different image/model configuration')
    save(config_path,settings)
    candidates=output/'candidates'
    raw=candidates/'raw_candidates.json'
    if raw.exists():
        code=command('replay',['scripts/cache_component_candidates.py','replay','--raw-cache',raw,
                              '--output',candidates,'--baseline-confidence',args.baseline_confidence],output)
    else:
        code=command('capture',['scripts/cache_component_candidates.py','capture','--images',images,
            '--weights',weights,'--output',candidates,'--device',args.device,'--baseline-confidence',args.baseline_confidence],output)
    if code:
        save(output/'workflow_status.json',{'status':'capture_failed','log':'capture.log','exit_code':code})
        return code
    queue=candidates/'qwen_candidates.jsonl'
    count=sum(bool(line.strip()) for line in queue.read_text(encoding='utf-8').splitlines())
    qwen_status='未调用API，候选已自动排队'
    if count:
        opts=['scripts/run_candidate_shadow_pilot.py','--manifest',queue,'--output',output/'qwen_shadow',
              '--limit',min(args.max_calls,count),'--batch-size','1','--max-requests',args.max_calls,
              '--token-budget',args.token_budget,'--max-tokens','1200','--preprocess','padded_v2']
        if args.qwen:
            opts.extend(['--key-file',args.key_file,'--execute-api'])
        code=command('qwen_shadow',opts,output)
        qwen_status=('Qwen旁路建议已保存' if not code else 'Qwen未全部完成，原候选保留并标记') if args.qwen else '截图已准备，未调用API'
    render_gallery(candidates,output,qwen_status)
    pipeline=read(candidates/'pipeline_summary.json')
    status={'status':'complete_with_shadow_only','input_requires_labels':False,'qwen_requested':args.qwen,
            'qwen_status':qwen_status,'queued_candidates':count,'counts':pipeline['counts'],
            'evaluation_status':pipeline['evaluation_status'],'machine_training_labels_created':False,
            'output':str(output),'gallery':str(output/'gallery.html'),
            'final_policy':'stable_baseline_held_until_verifier_independently_validated'}
    save(output/'workflow_status.json',status)
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if args.open:
        import webbrowser
        webbrowser.open((output/'gallery.html').as_uri())
    return 0


if __name__=='__main__':
    raise SystemExit(main())
