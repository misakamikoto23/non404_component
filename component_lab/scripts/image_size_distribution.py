"""Measure original PCBnet images and the deduplicated detection dataset."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(rows):
    result = {'images': len(rows)}
    for field in ('width', 'height', 'long_side', 'short_side', 'megapixels', 'aspect_ratio'):
        values = np.array([r[field] for r in rows])
        result[field] = dict(zip(('min', 'p25', 'median', 'p75', 'p90', 'p95', 'max'),
                                map(float, np.percentile(values, [0, 25, 50, 75, 90, 95, 100]))))
    edges = [0, 640, 1280, 1920, 2560, 3840, float('inf')]
    names = ['<=640', '641-1280', '1281-1920', '1921-2560', '2561-3840', '>3840']
    result['long_side_bins'] = {name: sum(lo < r['long_side'] <= hi for r in rows)
                               for name, lo, hi in zip(names, edges, edges[1:])}
    result['orientation'] = {'landscape': sum(r['width'] > r['height'] for r in rows),
                             'portrait': sum(r['width'] < r['height'] for r in rows),
                             'square': sum(r['width'] == r['height'] for r in rows)}
    result['common_sizes'] = [{'width': w, 'height': h, 'count': n}
                              for (w, h), n in Counter((r['width'], r['height']) for r in rows).most_common(12)]
    result['largest_images'] = sorted(rows, key=lambda r: r['megapixels'], reverse=True)[:5]
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'data/yolo/pcbnet_component_v1/manifest.json').read_text(encoding='utf-8'))
    splits = {r['case']: r['split'] for r in manifest['samples']}
    output = root / 'outputs/image_size_distribution'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(manifest['source']).rglob('*')):
        if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'):
            continue
        with Image.open(path) as im:
            w, h = im.size
        rows.append({'case': path.parent.name, 'path': str(path), 'split': splits.get(path.parent.name, 'excluded_duplicate'),
                     'width': w, 'height': h, 'long_side': max(w, h), 'short_side': min(w, h),
                     'megapixels': w*h/1e6, 'aspect_ratio': w/h})
    selected = [r for r in rows if r['case'] in splits]
    assert len(selected) == len(manifest['samples'])
    for r in manifest['samples']:
        actual = next(x for x in selected if x['case'] == r['case'])
        assert (actual['width'], actual['height']) == (r['width'], r['height'])
    report = {'raw': summarize(rows), 'deduplicated': summarize(selected),
              'splits': {s: summarize([r for r in selected if r['split'] == s]) for s in ('train','val','test')}}
    (output / 'statistics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'images.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for group, data, color in [('Raw', rows, '#8799ae'), ('Deduplicated', selected, '#287eb8')]:
        axes[0].hist([r['long_side'] for r in data], bins=[0,640,1280,1920,2560,3840,5120,7680,12000],
                     histtype='step', linewidth=2, label=f'{group} (n={len(data)})', color=color)
    axes[0].set(xlabel='Long side (pixels)', ylabel='Images', title='Image long-side distribution')
    axes[0].legend()
    for split, color in [('train','#287eb8'),('val','#e7a23d'),('test','#59a976')]:
        data = [r for r in selected if r['split']==split]
        axes[1].scatter([r['width'] for r in data], [r['height'] for r in data], s=18, alpha=.65, label=split, color=color)
    axes[1].set(xlabel='Width (pixels)', ylabel='Height (pixels)', title='Deduplicated image dimensions')
    axes[1].legend()
    axes[2].hist([r['megapixels'] for r in selected], bins=20, color='#287eb8', edgecolor='white')
    axes[2].set(xlabel='Megapixels', ylabel='Images', title='Deduplicated pixel counts')
    fig.tight_layout()
    fig.savefig(output / 'distribution.png', dpi=170)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
