"""Build original-image previews above the model input long-side threshold."""
import csv
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageOps


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs/oversized_images_1280'
    output.mkdir(parents=True, exist_ok=True)
    with (root / 'outputs/image_size_distribution/images.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] != 'excluded_duplicate' and int(r['long_side']) > 1280]
    rows.sort(key=lambda r: (-int(r['long_side']), -float(r['megapixels']), r['case']))
    cards = []
    cell_w, cell_h, per_page = 400, 230, 28
    for index, row in enumerate(rows):
        with Image.open(row['path']) as image:
            thumb = ImageOps.contain(image.convert('RGB'), (380, 190))
        thumbnail = output / f"{row['case']}_thumbnail.jpg"
        thumb.save(thumbnail, quality=92)
        if index % per_page == 0:
            page = Image.new('RGB', (4*cell_w, math.ceil(min(per_page, len(rows)-index)/4)*cell_h + 45), '#e9edf2')
            draw = ImageDraw.Draw(page)
            draw.text((12, 14), f'PCBnet: long side > 1280 px | page {index//per_page+1} | largest first', fill='black')
        x, y = (index % 4)*cell_w, 45+((index % per_page)//4)*cell_h
        draw.rectangle((x+5,y+3,x+cell_w-5,y+cell_h-4), fill='white')
        page.paste(thumb, (x+(cell_w-thumb.width)//2, y+5+(190-thumb.height)//2))
        draw.text((x+10,y+199), f"{row['case']} | {row['width']} x {row['height']} | {row['split']}", fill='black')
        draw.text((x+10,y+214), f"{float(row['megapixels']):.2f} MP", fill='#555555')
        if index % per_page == per_page-1 or index == len(rows)-1:
            page.save(output / f'overview_{index//per_page+1:02d}.png')
        original = quote(os.path.relpath(row['path'], output).replace('\\','/'), safe='/')
        cards.append(f'<a class="card" href="{original}" target="_blank"><img src="{thumbnail.name}"><b>{row["case"]}: {row["width"]} × {row["height"]}</b><span>{row["split"]} · {float(row["megapixels"]):.2f} MP</span></a>')
    document = '<!doctype html><meta charset="utf-8"><title>PCBnet 大图总览</title><style>body{font:16px sans-serif;background:#eef2f6;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}.card{display:flex;flex-direction:column;background:white;padding:12px;border-radius:8px;color:#17212b;text-decoration:none;gap:8px}.card img{width:100%;height:190px;object-fit:contain}span{color:#666}</style><h1>长边超过 1280 像素的原理图</h1><p>去重后共 '+str(len(rows))+' 张，按长边从大到小排列。1280 是模型输入阈值；点击缩略图打开原图。</p><main>'+''.join(cards)+'</main>'
    (output / 'gallery.html').write_text(document, encoding='utf-8')
    with (output / 'images.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'count':len(rows),'pages':math.ceil(len(rows)/per_page),'output':str(output),'largest':rows[:4]},ensure_ascii=False))


if __name__ == '__main__':
    main()
