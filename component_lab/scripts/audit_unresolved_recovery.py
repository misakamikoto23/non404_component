"""Read-only DEV evaluation of frozen newly recovered identities; no API calls."""
from collections import Counter
import html
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from component_lab.candidate_verifier import atomic_json
from component_lab.eval.coords import flip_bbox
from component_lab.eval.metric import text_ok
from pilot_qwen_identity import INPUTS,read


def panel(image,box,title,color):
    x1,y1,x2,y2=box;pad=max(72,min(192,max(x2-x1,y2-y1)))
    window=[max(0,int(x1-pad)),max(0,int(y1-pad)),min(image.width,int(x2+pad+1)),min(image.height,int(y2+pad+1))]
    crop=image.crop(window);scale=min(496/crop.width,470/crop.height,3)
    view=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(512,512),'white');ox,oy=(512-view.width)//2,32+(476-view.height)//2
    canvas.paste(view,(ox,oy));draw=ImageDraw.Draw(canvas);draw.text((8,8),title,fill='black')
    draw.rectangle((ox+(x1-window[0])*scale,oy+(y1-window[1])*scale,ox+(x2-window[0])*scale,oy+(y2-window[1])*scale),outline=color,width=2)
    return canvas


def run(output):
    manifest=read(output/'manifest.json');audits=read(output/'writeback_audits.json')
    targets={t['id']:t for t in manifest['targets']};folder=output/'eval_only_failures';folder.mkdir(exist_ok=True)
    rows=[];counts=Counter();cards=[]
    for change in audits['new_keys']['changes']:
        case,key=change['case'],change['after_key'];t=targets[change['id']]
        pred=read(output/'new_keys/predictions'/f'{case}.json')['components'][key]
        gt=read(next((ROOT/'data/official_contest6/train200').glob(f'{case}-*.json')))['components'].get(key)
        if gt is None:
            reasons=['key_not_in_GT'];error=None
        else:
            error=max(abs(a-b) for a,b in zip(pred['bbox'],gt['bbox']))
            required_ok=text_ok((pred['type'],pred.get('Name'),pred.get('value'),pred['bbox']),
                                (gt['type'],gt.get('Name'),gt.get('value'),gt['bbox']))
            reasons=[]
            if pred['type']!=gt['type']:reasons.append('type')
            if error>20:reasons.append('bbox')
            if not required_ok:reasons.append('required_text')
        label='+'.join(reasons) or 'complete_TP';counts[label]+=1
        row={'id':t['id'],'case':case,'returned_key':key,'reason':label,'max_coordinate_error':error,'prediction':pred,'gt':gt}
        rows.append(row)
        if reasons:
            with Image.open(INPUTS/case/'source.png') as source:
                source=source.convert('RGB');left=panel(source,t['bbox'],f'{case}/{key} - predicted target','#e32828')
                right=panel(source,flip_bbox(gt['bbox'],source.height),'GT for the returned key','#2166ac') if gt else Image.new('RGB',(512,512),'#eee')
            canvas=Image.new('RGB',(1024,512),'white');canvas.paste(left,(0,0));canvas.paste(right,(512,0))
            path=folder/f'{case}_{key}.png';canvas.save(path)
            cards.append('<h2>'+html.escape(f'{case}/{key}: {label}')+'</h2><img style="width:100%" src="'+path.name+'"><pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre>')
    result={'scope':'eval_only_new_keys_13_successful_renames','counts':dict(counts),'rows':rows,
            'no_inference_use':True,'gt_or_predictions_modified':False,'api_calls':0}
    atomic_json(folder/'audit.json',result)
    (folder/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Recovery failures - evaluation only</title><style>body{font-family:system-ui;max-width:1100px;margin:30px auto}pre{white-space:pre-wrap}</style><h1>新命名失败审计：仅评测使用</h1><p>左红框为模型目标，右蓝框为返回key的GT位置。不是训练标签，也不能用于选择本轮API目标。</p><pre>'+html.escape(json.dumps(dict(counts),ensure_ascii=False))+'</pre>'+''.join(cards),encoding='utf-8')
    print(json.dumps(dict(counts),ensure_ascii=True))


if __name__=='__main__':run(ROOT/'outputs/unresolved_identity_recovery_20261017')
