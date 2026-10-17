"""Prediction-only local spectrum/text features. OCR always sees original RGB."""
import math

import numpy as np

from .locked_ocr_recovery import area, overlap


def distance(a, b):
    return math.hypot(max(a[0]-b[2], b[0]-a[2], 0), max(a[1]-b[3], b[1]-a[3], 0))


def extract_features(image, need_boxes):
    import cv2
    cell = 128
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    ink = (gray < 200).astype(np.uint8)
    lines = np.maximum(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 48), np.uint8)),
                       cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((48, 1), np.uint8)))
    _, _, stats, centers = cv2.connectedComponentsWithStats(ink*(1-lines), 8)
    good = ((stats[:, cv2.CC_STAT_HEIGHT] >= 3) & (stats[:, cv2.CC_STAT_HEIGHT] <= 40)
            & (stats[:, cv2.CC_STAT_WIDTH] <= 60) & (stats[:, cv2.CC_STAT_AREA] >= 3)
            & (stats[:, cv2.CC_STAT_AREA] <= 1800))
    good[0] = False
    centers = centers[good]
    fy, fx = np.meshgrid(np.fft.fftfreq(cell), np.fft.fftfreq(cell), indexing='ij')
    radius = np.sqrt(fx*fx+fy*fy)
    band = (radius >= .10) & (radius <= .45) & (np.abs(fx) > 2/cell) & (np.abs(fy) > 2/cell)
    hann = np.outer(np.hanning(cell), np.hanning(cell))
    features = []
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            x2, y2 = min(w, x+cell), min(h, y+cell)
            patch = gray[y:y2, x:x2].astype(np.float32)/255
            patch = cv2.copyMakeBorder(patch, 0, cell-patch.shape[0], 0, cell-patch.shape[1], cv2.BORDER_REFLECT_101)
            spectrum = np.abs(np.fft.fft2((patch-patch.mean())*hann))**2
            count = int(((centers[:, 0] >= x) & (centers[:, 0] < x2)
                         & (centers[:, 1] >= y) & (centers[:, 1] < y2)).sum())
            cx, cy = (x+x2)/2, (y+y2)/2
            gap = min((distance([cx, cy, cx, cy], b) for b in need_boxes), default=1e6)
            features.append({'box': [x, y, x2, y2], 'offaxis_energy': float(spectrum[band].sum()/cell**4),
                             'character_cc_count': count, 'line_fraction': float(lines[y:y2, x:x2].mean()),
                             'near_unresolved': max(0, 1-gap/cell)})
    def rank(key):
        vals = np.asarray([r[key] for r in features])
        return np.searchsorted(np.sort(vals), vals, side='right')/max(1, len(vals))
    diagonal, chars = rank('offaxis_energy'), rank('character_cc_count')
    for i, row in enumerate(features):
        row['hybrid_score'] = float(.30*diagonal[i]+.45*chars[i]+.15*row['near_unresolved']
                                    -.10*min(1, row['line_fraction']/.1))
    return features


def context_box(box, size, max_side=320):
    w, h = size
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    bw, bh = min(w, max_side, math.ceil(box[2]-box[0]+96)), min(h, max_side, math.ceil(box[3]-box[1]+96))
    x, y = max(0, min(w-bw, cx-bw/2)), max(0, min(h-bh, cy-bh/2))
    return [int(x), int(y), int(x)+bw, int(y)+bh]


def select_regions(image, targets, texts, *, max_rois=2, area_fraction=.20, mode='empty-fields'):
    eligible = []
    for target in targets:
        record, key = target['baseline_record'], target['baseline_key']
        kind = record['type']
        if kind in {'gnd', 'v', 'ground', 'power_port', 'net_port', 'no_connect', 'm3螺丝', 'other', 'block'} or kind.startswith('net_'):
            continue
        unresolved = key.startswith('__unresolved')
        if mode == 'unresolved-only' and not unresolved:
            continue
        slots = ['reference'] if unresolved else []
        if kind in {'r', 'c'} and record['value'] is None:
            slots.append('value')
        if kind == 'box' and record['Name'] is None:
            slots.append('Name')
        if slots:
            eligible.append({**target, 'slots': slots, 'unresolved': unresolved})
    if not eligible or max_rois == 0:
        return [], {'eligible_targets': len(eligible), 'feature_status': 'not_needed'}
    feature_status = 'frequency_and_character_features'
    try:
        features = extract_features(image, [t['bbox'] for t in eligible])
    except ImportError:
        features = []
        feature_status = 'opencv_missing_geometry_fallback'
    w, h = image.size
    candidates = []
    for target in eligible:
        opts = []
        box = target['bbox']
        for feature in features:
            gap = distance(feature['box'], box)
            if gap > 96 or feature['character_cc_count'] == 0:
                continue
            f = feature['box']
            center = math.hypot((f[0]+f[2]-box[0]-box[2])/2, (f[1]+f[3]-box[1]-box[3])/2)
            opts.append((feature['hybrid_score']-.30*gap/96-.10*min(2, center/128), f))
        if opts:
            score, cell = max(opts, key=lambda v: v[0])
            crop = [max(0, cell[0]-24), max(0, cell[1]-24), min(w, cell[2]+24), min(h, cell[3]+24)]
            method = 'target_gated_frequency_character_features'
        else:
            score, crop, method = 0., context_box(box, image.size), 'low_frequency_geometry_fallback'
        for _ in range(3):
            before = crop[:]
            for text in texts:
                b = text['bbox']
                if overlap(crop, b) > .2:
                    new = [max(0, min(crop[0], math.floor(b[0])-4)), max(0, min(crop[1], math.floor(b[1])-4)),
                           min(w, max(crop[2], math.ceil(b[2])+4)), min(h, max(crop[3], math.ceil(b[3])+4))]
                    if max(new[2]-new[0], new[3]-new[1]) <= 320:
                        crop = new
            if before == crop:
                break
        candidates.append({'target_id': target['id'], 'slots': target['slots'], 'box': crop,
                           'score': float(score)+.1*float(target.get('score') or 0),
                           'unresolved': target['unresolved'], 'method': method})
    ordered = []
    for unresolved in (True, False):
        group = [c for c in candidates if c['unresolved'] == unresolved]
        if group:
            ordered.append(max(group, key=lambda c: c['score']))
    ordered += sorted([c for c in candidates if c not in ordered], key=lambda c: (not c['unresolved'], -c['score']))
    selected, cost = [], 0
    for row in ordered:
        if len(selected) >= max_rois:
            break
        if cost+area(row['box']) > area_fraction*w*h or any(overlap(row['box'], p['box']) > .7 for p in selected):
            continue
        cost += area(row['box'])
        selected.append(row)
    return selected, {'eligible_targets': len(eligible), 'feature_status': feature_status,
                      'crop_area_fraction': cost/(w*h), 'uses_ground_truth': False}
