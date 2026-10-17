"""Random dev-only OCR examples: fresh inference and original-text gallery."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import re
import shutil

from PIL import Image, ImageDraw, ImageFont

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / 'outputs/random_ocr_preview_20261017'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare():
    assert not OUT.exists(), 'Keep old experiments unchanged'
    split = json.loads((LAB / 'configs/contest6_group_split_v1.json').read_text(encoding='utf-8'))
    cases = random.Random(20261017).sample(sorted(r['case'] for r in split['partitions']['dev']), 4)
    OUT.mkdir(parents=True)
    jobs = []
    for case in cases:
        folder = OUT / case
        folder.mkdir()
        source = LAB / 'outputs/component_identity_round_001_20261017/inputs' / case / 'source.png'
        shutil.copy2(source, folder / 'source.png')
        jobs.append(dict(id=case, image=str(folder / 'source.png'), ocr=str(folder / 'ocr_rapidocr.json'),
                         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    save(OUT / 'manifest.json', dict(seed=20261017, sampling='uniform four cases from fixed dev20 before OCR; no GT used',
                                   jobs=jobs, test_used=False, api_calls=0))
    print('Selected', cases, flush=True)


STYLE = '''<meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>
body{font:16px/1.6 system-ui;background:#f4f6fa;color:#17233b;margin:24px auto;padding:0 20px;max-width:1450px}a{color:#1265ba}img{max-width:100%}.card{background:white;padding:18px;margin:16px 0;border-radius:12px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d8deea;padding:8px;text-align:left}td img{max-width:360px;max-height:110px;image-rendering:auto}code{white-space:pre-wrap;overflow-wrap:anywhere}.low{background:#fff3d5}.muted{color:#526179}svg{width:100%;height:auto;background:white}rect.token{fill:rgba(0,130,220,.03);stroke:#0089d4;stroke-width:1.3;cursor:pointer}rect.token:hover{fill:#ffbb0055;stroke:#f08400;stroke-width:3}tr:target{background:#fff0b5}text{font-family:Arial;paint-order:stroke;stroke:white;stroke-width:2px;fill:#005ea8}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}</style>'''


def render():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    cards, summaries = [], []
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    for job in manifest['jobs']:
        case, folder = job['id'], Path(job['image']).parent
        image = Image.open(folder / 'source.png').convert('RGB')
        assert hashlib.sha256((folder / 'source.png').read_bytes()).hexdigest() == job['source_sha256']
        result = json.loads((folder / 'ocr_rapidocr.json').read_text(encoding='utf-8'))
        tokens = result['texts']
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        svg, rows = [], []
        (folder / 'crops').mkdir(exist_ok=True)
        for n, token in enumerate(tokens, 1):
            x1, y1, x2, y2 = token['bbox']
            color = '#e07b00' if token['score'] < .8 else '#0089d4'
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            draw.text((x1, max(0, y1-14)), str(n), font=font, fill=color, stroke_width=1, stroke_fill='white')
            crop = image.crop((max(0, int(x1)-10), max(0, int(y1)-10),
                               min(image.width, int(x2)+11), min(image.height, int(y2)+11)))
            # Enlarge small crops for inspection, retaining the original page too.
            factor = min(4, max(1, 60 / max(crop.height, 1)))
            crop.resize((max(1, round(crop.width*factor)), max(1, round(crop.height*factor))), Image.Resampling.LANCZOS).save(folder / 'crops' / f'{n:04}.png')
            text = token['text']
            escaped = html.escape(text)
            svg.append(f'<a href="#t{n}"><rect class="token" style="stroke:{color}" x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}"><title>#{n} {escaped} score={token["score"]:.4f}</title></rect><text x="{x1}" y="{max(12,y1-2)}" font-size="12">{n}</text></a>')
            flag = '低分候选；不等于识别错误' if token['score'] < .8 else ''
            if re.search(r'[A-Za-z]+\d+\s+\S', text):
                flag += ' · 可能含多个字段，请核图'
            rows.append(f'<tr id="t{n}" class="{"low" if token["score"]<.8 else ""}"><td>#{n}</td><td><img loading="lazy" src="crops/{n:04}.png"></td><td><code>{escaped}</code></td><td>{token["score"]:.4f}</td><td>{html.escape(flag)}</td></tr>')
        overlay.save(folder / 'overlay.png')
        low = sum(t['score'] < .8 for t in tokens)
        summary = dict(case=case, width=image.width, height=image.height, tokens=len(tokens),
                       below_08=low, inference_metadata={k:v for k,v in result.items() if k not in ['texts','settings']},
                       settings=result['settings'])
        summaries.append(summary)
        page = STYLE + f'<title>OCR {case}</title><a href="../gallery.html">← 四图概览</a><h1>{case}：{len(tokens)}条OCR文字</h1><p>蓝框为≥0.8，橙框为&lt;0.8。分数不是正确概率；高分也可能错。点击文字框跳转对应裁剪，悬停查看识别原文。</p><p><a href="source.png">原图</a> · <a href="overlay.png">编号叠图</a> · <a href="ocr_rapidocr.json">原始OCR JSON</a></p><div class="card"><svg viewBox="0 0 {image.width} {image.height}"><image href="source.png" width="{image.width}" height="{image.height}"/>' + ''.join(svg) + '</svg></div><h2>全部文字：原图裁剪与逐字输出</h2><p>识别原文未改写大小写、单位、标点或空格。该页面未进行元件归属，也没有人工文字真值；漏检文字不会出现在下面列表中，需对照整图查看。</p><table><tr><th>ID</th><th>原图局部</th><th>OCR原文</th><th>分数</th><th>检查提示（非错误定论）</th></tr>' + ''.join(rows) + '</table>'
        (folder / 'index.html').write_text(page, encoding='utf-8')
        cards.append(f'<div class="card"><h2>{case}</h2><p>{image.width}×{image.height}，{len(tokens)}条文字，低于0.8：{low}条</p><a href="{case}/index.html"><img src="{case}/overlay.png"></a><p><a href="{case}/index.html">查看交互文字框与全部裁剪 →</a></p></div>')
    save(OUT / 'report.json', dict(manifest=manifest, summaries=summaries, limitations='No OCR GT transcription: counts/confidence are not accuracy. No component ownership tested.'))
    intro = '<h1>随机4张原理图：OCR效果检查</h1><p>固定seed=20261017，从dev20均匀随机抽4张，在识别前冻结选择；没有按错误挑图，不使用test40。</p><p>本次重新运行本地RapidOCR：PP-OCRv5 mobile检测/识别、方向模块、2倍输入；大图按现有规则增加1200px切片，文字分数阈值0.35。CPU运行，不调用Qwen、不消耗API token。颜色只区分OCR分数，不代表人工判定。</p><p>建议依次检查：①整图有字却没框；②Ω/μ、大小写、数字误读；③位号与参数粘在同一框；④竖排文字；⑤网络名/引脚名混入。点击图片进入逐条原文对照。</p>'
    intro += '<div class="card"><h2>已目视核对的几个现象（不是全量准确率统计）</h2><ul><li><a href="0083/index.html#t7">0083 #7</a>：图上10k，输出10，score约0.994；同页还有SCLOUT1→SCLOUTI等1/I、O/0混淆。高分不保证正确。</li><li><a href="0131/index.html#t57">0131 #57</a>：R3下方图上10kΩ，输出10kn。小字单位误读；标题栏/页框文字同时进入OCR池，后续需要区分角色。</li><li><a href="0055/index.html#t11">0055 #11</a>：原图连续的5V to 20V被拆成5Vto和20V两条；这是文字分组现象，不自动算OCR错误。LM2941、R1/R2、22μF已读出。</li></ul><p>0083的0.01μF在JSON中保留了μ；不能把终端编码显示异常当作OCR乱码。完整页与全部裁剪保留，尚未完成逐字人工标注。</p></div>'
    (OUT / 'gallery.html').write_text(STYLE+'<title>随机OCR预览</title>'+intro+'<div class="grid">'+''.join(cards)+'</div>', encoding='utf-8')
    (OUT / 'REPORT.md').write_text('# 随机4图OCR预览\n\nseed=20261017，dev20均匀抽样，先选择后识别；无GT参与选择。fresh local OCR; CPU; API 0; test40未使用。\n\n'+ '\n'.join(f'- {s["case"]}: {s["width"]}×{s["height"]}, {s["tokens"]} tokens, below0.8={s["below_08"]}' for s in summaries)+'\n\n没有逐字人工真值，不报告accuracy；低分/高分不等于错/对。仅OCR可视化，不评价文字到元件的归属。\n\n目视样例：0083 #7原图10k，输出10，score .994；0131 #57原图10kΩ，输出10kn；0055的5V to 20V分成5Vto和20V两条。属于展示性样例，不估计全量误差率。0083 JSON中0.01μF保留μ，控制台显示异常不能当OCR乱码。\n\n复现：先用.venv-yolo运行本脚本prepare；再用.venv-ocr/Scripts/python.exe scripts/run_local_ocr.py --manifest outputs/random_ocr_preview_20261017/manifest.json --limit 4；最后用.venv-yolo运行本脚本render。已有目录不覆盖。\n', encoding='utf-8')
    print(json.dumps([{k:s[k] for k in ['case','width','height','tokens','below_08']} for s in summaries]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'render'])
    args = parser.parse_args()
    prepare() if args.phase == 'prepare' else render()
