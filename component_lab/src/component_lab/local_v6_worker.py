"""Isolated CPU worker; executed by the configured local OCR Python runtime."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def run(root):
    from PIL import Image
    import numpy as np
    from rapidocr import RapidOCR, OCRVersion, ModelType
    request = json.loads((root/'request.json').read_text(encoding='utf-8'))
    for file, digest in request['model_hashes'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != digest:
            raise ValueError('OCR model changed after request was frozen')
    started = time.perf_counter()
    engine = RapidOCR(params={
        'Global.model_root_dir': request['model_root'], 'Global.max_side_len': 4096,
        'Global.text_score': .35, 'Global.use_cls': False,
        'Det.ocr_version': OCRVersion.PPOCRV6, 'Rec.ocr_version': OCRVersion.PPOCRV6,
        'Det.model_type': ModelType.MEDIUM, 'Rec.model_type': ModelType.MEDIUM,
        # Local files are mandatory: no automatic download on failure.
        'Det.model_path': str(Path(request['model_root'])/'PP-OCRv6_det_medium.onnx'),
        'Rec.model_path': str(Path(request['model_root'])/'PP-OCRv6_rec_medium.onnx'),
        'Det.limit_side_len': 2048, 'Det.limit_type': 'max',
        'EngineConfig.onnxruntime.intra_op_num_threads': 2,
        'EngineConfig.onnxruntime.inter_op_num_threads': 1,
        'EngineConfig.onnxruntime.use_cuda': False})
    results = []
    for i, roi in enumerate(request['regions']):
        tick = time.perf_counter()
        try:
            path = root/f'roi_{i:02d}.png'
            if hashlib.sha256(path.read_bytes()).hexdigest() != roi['crop_sha256']:
                raise ValueError('ROI changed after request was frozen')
            with Image.open(path) as crop:
                scaled = crop.convert('RGB').resize(roi['scaled_size'], Image.Resampling.LANCZOS)
                output = engine(np.asarray(scaled))
            rows = []
            x1, y1, x2, y2 = roi['box']
            sx, sy = roi['scaled_size'][0]/(x2-x1), roi['scaled_size'][1]/(y2-y1)
            if output.boxes is not None:
                for j, (poly, raw, score) in enumerate(zip(output.boxes, output.txts, output.scores)):
                    polygon = [[max(x1, min(x2, float(x)/sx+x1)), max(y1, min(y2, float(y)/sy+y1))] for x, y in poly]
                    xs, ys = zip(*polygon)
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    if not all(math.isfinite(v) for v in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    rows.append({'id': f'v6_{i:02d}_{j:04d}', 'raw': str(raw), 'text': str(raw),
                                 'bbox': bbox, 'polygon': polygon, 'score': float(score),
                                 'owner_ids': [roi['target_id']], 'source_tier': 'fallback_v6', 'source_window': roi['box']})
            results.append({'status': 'ok', 'texts': rows, 'seconds': time.perf_counter()-tick})
        except Exception as exc:
            results.append({'status': 'failed', 'error_type': type(exc).__name__, 'texts': [], 'seconds': time.perf_counter()-tick})
        save(root/'response.json', {'request_hash': request['request_hash'], 'regions': results,
                                    'complete': len(results) == len(request['regions']),
                                    'seconds': time.perf_counter()-started,
                                    'runtime': {k: importlib.metadata.version(k) for k in ('rapidocr', 'onnxruntime')}})


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve())
