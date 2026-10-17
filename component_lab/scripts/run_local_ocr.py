"""Bounded local OCR cache with original-image coordinates and resumable hashes."""
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
from pathlib import Path
import subprocess
import time

from PIL import Image
import numpy as np


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--engine", choices=["rapidocr", "tesseract"], default="rapidocr")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--scale", type=float, default=2)
    p.add_argument("--max-tiles", type=int, default=12)
    p.add_argument("--cache-suffix", default="")
    args = p.parse_args()
    if not 1 <= args.max_tiles <= 32 or not re.fullmatch(r"[A-Za-z0-9_-]{0,24}", args.cache_suffix):
        p.error("max-tiles must be 1..32 and cache-suffix a short safe token")
    root = Path(__file__).resolve().parents[1]
    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))["jobs"][:args.limit]
    settings = {"engine": args.engine, "scale": args.scale, "text_score": .35,
                "tile_side": 1200, "overlap": 160, "max_tiles": args.max_tiles, "selection_uses_labels": False}
    cache_tag = args.engine + ("_" + args.cache_suffix if args.cache_suffix else "")
    if args.engine == "rapidocr":
        from rapidocr import RapidOCR, OCRVersion, ModelType
        settings["package_version"] = importlib.metadata.version("rapidocr")
        settings["model"] = "PP-OCRv5_mobile_det_rec_with_PP-OCRv4_orientation"
        engine = RapidOCR(params={"Global.model_root_dir": str(root / "weights/ocr_rapid"),
            "Global.max_side_len": 4096, "Global.text_score": .35,
            "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
            "Det.limit_side_len": 2048, "Det.limit_type": "max",
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False})
    else:
        os.environ["OMP_THREAD_LIMIT"] = "2"
        settings["config"] = "--psm 11 -l eng+chi_sim"
    completed = []
    for job in jobs:
        image_path = Path(job["image"])
        target = Path(job["ocr"]).with_name(f"ocr_{cache_tag}.json")
        config = {**settings, "image_sha256": sha(image_path)}
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing.get("settings") != config:
                raise ValueError(f"Refusing incompatible cached OCR: {target}")
            completed.append({"id": job["id"], "status": "cached", "texts": len(existing["texts"])})
            continue
        started = time.perf_counter()
        with Image.open(image_path) as original:
            original = original.convert("RGB")
            width, height = original.size
            windows = [[0, 0, width, height]]
            if args.engine == "rapidocr" and max(width, height) > 1800:
                def starts(n):
                    if n <= 1200:
                        return [0]
                    return sorted(set(list(range(0, n-1200+1, 1040))+[n-1200]))
                tiles = [[x, y, min(width, x+1200), min(height, y+1200)] for y in starts(height) for x in starts(width)]
                if len(tiles) <= args.max_tiles:
                    windows += tiles
            rows = []
            for view_id, window in enumerate(windows):
                crop = original.crop(window)
                scaled = crop.resize((round(crop.width*args.scale), round(crop.height*args.scale)), Image.Resampling.LANCZOS)
                sx, sy = scaled.width/crop.width, scaled.height/crop.height
                detected = []
                if args.engine == "rapidocr":
                    output = engine(np.asarray(scaled))
                    if output.boxes is not None:
                        detected = [(poly.tolist(), text, float(score)) for poly, text, score in zip(output.boxes, output.txts, output.scores)]
                else:
                    prepared = target.parent / f"tesseract_input_view{view_id}.png"
                    scaled.save(prepared)
                    process = subprocess.run([r"C:\Program Files\Tesseract-OCR\tesseract.exe", str(prepared), "stdout",
                        "-l", "eng+chi_sim", "--psm", "11", "tsv"], capture_output=True, timeout=90, check=True)
                    data = csv.DictReader(io.StringIO(process.stdout.decode("utf-8", errors="replace")), delimiter="\t")
                    for record in data:
                        text = record.get("text", "")
                        score = float(record["conf"])/100
                        if not text.strip() or score < .35:
                            continue
                        x, y, w, h = (int(record[key]) for key in ("left", "top", "width", "height"))
                        detected.append(([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], text, score))
                for poly, text, score in detected:
                    polygon = [[max(0., min(width, x/sx+window[0])), max(0., min(height, y/sy+window[1]))] for x,y in poly]
                    xs, ys = zip(*polygon)
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    rows.append({"text": str(text), "raw": str(text), "score": score, "bbox": bbox,
                                 "polygon": polygon, "orientation": math.degrees(math.atan2(polygon[1][1]-polygon[0][1], polygon[1][0]-polygon[0][0])),
                                 "source_view": view_id, "source_window": window})
        kept = []
        for row in sorted(rows, key=lambda row: -row["score"]):
            if not any(row["text"].strip() == old["text"].strip() and overlap(row["bbox"],old["bbox"]) > .5 for old in kept):
                kept.append(row)
        for i, row in enumerate(kept, 1):
            row["id"] = f"text_{i:04d}"
        result = {"image": str(image_path), "coordinate_system": "top-left-xyxy-original-pixels",
                  "settings": config, "texts": kept, "views": windows, "raw_text_count": len(rows),
                  "seconds": time.perf_counter()-started, "ground_truth_used": False,
                  "warning": "Recognition scores are uncalibrated; this file is machine OCR, not verified text."}
        save(target, result)
        completed.append({"id": job["id"], "status": "complete", "texts": len(kept), "seconds": result["seconds"]})
        save(args.manifest.parent / f"ocr_{cache_tag}_progress.json", {"jobs": completed, "status": "running"})
        print(f"{job['id']}: {len(kept)} texts / {len(windows)} views / {result['seconds']:.1f}s", flush=True)
    save(args.manifest.parent / f"ocr_{cache_tag}_progress.json", {"jobs": completed, "status": "complete", "settings": settings})


if __name__ == "__main__":
    main()
