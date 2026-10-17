"""Held-out comparison of whole-image and overlapping-tile inference.

Tile predictions are translated into original image coordinates. Detections
touching internal tile edges are conservatively discarded before class-aware
NMS combines the remaining tiles with whole-image detections. The latter keep
large objects that do not fit inside one tile. Model loading happens only in
main(), so geometry tests never import torch or execute inference.
Defaults to val. Select test only to evaluate a configuration already selected
on validation data.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def axis_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= 0 or tile_size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Image dimensions and tile size must be positive; overlap must be in [0,1).")
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def tile_plan(width: int, height: int, tile_size: int = 1024, overlap: float = .25,
              max_tiles: int = 100) -> dict:
    """Skip the entire tile pass if it exceeds the cap; never silently truncate."""
    if max_tiles <= 0:
        raise ValueError("max_tiles must be positive.")
    xs, ys = axis_starts(width, tile_size, overlap), axis_starts(height, tile_size, overlap)
    required = len(xs) * len(ys)
    skipped = required > max_tiles
    windows = [] if skipped else [[x, y, min(width, x + tile_size), min(height, y + tile_size)]
                                  for y in ys for x in xs]
    return {"windows": windows, "required_tile_count": required, "tile_count": len(windows),
            "skipped": skipped, "truncated": False,
            "skip_reason": "required_tile_count_exceeds_max_tiles" if skipped else None}


def touches_internal_edge(bbox: list[float], window: list[int], width: int, height: int,
                          margin: float = 2) -> bool:
    """Accept actual outer-image edges; reject boxes within margin of a tile seam."""
    if margin < 0:
        raise ValueError("edge margin must be nonnegative")
    left, top, right, bottom = window
    tile_width, tile_height = right - left, bottom - top
    x1, y1, x2, y2 = bbox
    return ((left > 0 and x1 <= margin) or (top > 0 and y1 <= margin) or
            (right < width and x2 >= tile_width - margin) or
            (bottom < height and y2 >= tile_height - margin))


def translate_prediction(prediction: dict, window: list[int], width: int, height: int) -> dict | None:
    left, top, right, bottom = window
    x1, y1, x2, y2 = prediction["bbox"]
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    # Result boxes are already in crop-original pixels, not model-input pixels.
    bbox = [max(0.0, min(float(width), x1 + left)), max(0.0, min(float(height), y1 + top)),
            max(0.0, min(float(width), x2 + left)), max(0.0, min(float(height), y2 + top))]
    if not all(math.isfinite(v) for v in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return {**prediction, "bbox": bbox, "source": "tile", "tile_window": [left, top, right, bottom]}


def box_iou(a: list[float], b: list[float]) -> float:
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.


def class_aware_nms(predictions: list[dict], iou_threshold: float = .5) -> list[dict]:
    """Stable confidence-ordered NMS; whole-image entries win exact score ties."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("NMS IoU must be in (0,1].")
    ranked = sorted(enumerate(predictions), key=lambda entry: (-entry[1]["score"], entry[0]))
    kept = []
    for _, prediction in ranked:
        if not any(prediction["class_id"] == previous["class_id"] and
                   box_iou(prediction["bbox"], previous["bbox"]) > iou_threshold for previous in kept):
            kept.append(prediction)
    return kept


def result_rows(result: Any, names: list[str], source: str) -> list[dict]:
    return [{"bbox": bbox, "class_id": int(class_id), "type": names[int(class_id)],
             "score": score, "source": source}
            for bbox, class_id, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(),
                                             result.boxes.conf.cpu().tolist())]


def write_cache(path: Path, args: argparse.Namespace, names: list[str], predictions: dict, mode: str) -> None:
    payload = {"schema_version": 1, "split": args.split, "mode": mode, "names": names,
               "coordinate_system": "xyxy_pixels_top_left_in_original_image", "confidence": args.confidence,
               "imgsz": args.imgsz, "tile_imgsz": args.tile_imgsz, "data_root": str(args.data_root),
               "weights": str(args.weights), "predictions": predictions}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compare_reports(whole: dict, combined: dict) -> dict:
    def delta(before: dict, after: dict, fields: tuple[str, ...]) -> dict:
        return {field: after[field] - before[field] if before[field] is not None and after[field] is not None else None
                for field in fields}
    by_class = []
    for before, after in zip(whole["by_class"], combined["by_class"]):
        by_class.append({"class_id": before["class_id"], "name": before["name"], "whole": before,
                         "combined": after, "delta": delta(before, after, ("tp", "fp", "fn", "precision", "recall"))})
    buckets = {}
    for key in ("by_original_area", "by_estimated_resized_short_side"):
        buckets[key] = [{"bucket": before["bucket"], "whole": before, "combined": after,
                         "delta": delta(before, after, ("tp", "fn", "recall"))}
                        for before, after in zip(whole[key], combined[key])]
    return {"metric": "fixed_threshold_counts_not_mAP", "whole": whole["overall"],
            "combined": combined["overall"],
            "delta": delta(whole["overall"], combined["overall"], ("tp", "fp", "fn", "precision", "recall")),
            "by_class": by_class, **buckets}


def run_diagnostics(args: argparse.Namespace) -> dict:
    from analyze_yolo_errors import main as diagnose

    common = ["--weights", str(args.weights), "--data-root", str(args.data_root), "--device", "cpu", "--split", args.split,
              "--imgsz", str(args.imgsz), "--confidence", str(args.confidence), "--iou", str(args.match_iou),
              "--max-det", str(args.max_det), "--visualizations", str(args.visualizations)]
    for directory, cache in (("whole_image", "whole_image_predictions.json"), ("combined", "predictions.json")):
        diagnose(common + ["--predictions-json", str(args.output / cache), "--output", str(args.output / directory)])
    whole = json.loads((args.output / "whole_image/metrics.json").read_text(encoding="utf-8"))
    combined = json.loads((args.output / "combined/metrics.json").read_text(encoding="utf-8"))
    comparison = {"split": args.split, **compare_reports(whole, combined)}
    (args.output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    # Mark both the machine-readable diagnostics and their rendered README:
    # the existing estimator describes the whole-image reference, not tiles.
    note = ("Estimated resized short-side buckets use whole-image imgsz as a common reference for grouping GT. "
            "Tile inference resizes each crop separately and has a different scale; these buckets do not represent "
            "tile input sizes or actual internal feature grids.")
    for directory in ("whole_image", "combined"):
        path = args.output / directory / "metrics.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["tiled_estimated_side_note"] = note
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with (args.output / directory / "README.md").open("a", encoding="utf-8") as handle:
            handle.write("\n" + note + "\n")
    return comparison


def write_summary(args: argparse.Namespace, metadata: dict, comparison: dict) -> None:
    percentage = lambda value: "n/a" if value is None else f"{value:.1%}"
    points = lambda value: "n/a" if value is None else f"{100 * value:+.1f} pp"
    before, after, delta = comparison["whole"], comparison["combined"], comparison["delta"]
    lines = [f"# Whole-image + tiled {args.split} comparison", "", "Fixed-confidence greedy matching counts, not mAP.", "",
             f"Split: {args.split}; images: {metadata['image_count']}; confidence {args.confidence}; GT matching IoU {args.match_iou}.",
             f"Whole-image input {args.imgsz}; original-pixel tiles {args.tile_size}; overlap {args.overlap:.0%}; tile input {args.tile_imgsz}.",
             f"Class-aware merge NMS IoU {args.nms_iou}; internal tile-edge margin {args.edge_margin} px.", "",
             "| Metric | Whole image | Whole + tiles | Change |", "|---|---:|---:|---:|",
             f"| TP | {before['tp']} | {after['tp']} | {delta['tp']:+d} |",
             f"| FP | {before['fp']} | {after['fp']} | {delta['fp']:+d} |",
             f"| FN | {before['fn']} | {after['fn']} | {delta['fn']:+d} |",
             f"| Precision | {percentage(before['precision'])} | {percentage(after['precision'])} | {points(delta['precision'])} |",
             f"| Recall | {percentage(before['recall'])} | {percentage(after['recall'])} | {points(delta['recall'])} |", "",
             f"Processed tiles: {metadata['tile_count']}; images skipping tiles due to max-tiles: {metadata['skipped_image_count']}; truncated images: 0.",
             f"Inference-only time: whole {metadata['whole_inference_seconds']:.2f}s; tiles {metadata['tile_inference_seconds']:.2f}s. See metadata.json for per-image timings.", "",
             "Original-image area buckets and per-class changes are in comparison.json. Error overlays are in whole_image/overlays and combined/overlays.",
             "Estimated resized short-side buckets group GT using the whole-image scale as a common reference. Tile inputs resize each crop separately; these buckets are not tile-input sizes or internal model grids.",
             "If required tiles exceed max-tiles, the entire tile pass for that image is skipped, while its whole-image prediction is preserved. No partial coverage is silently used.",
             "A tile-edge-touching detection is filtered only at internal image boundaries. Merge NMS suppresses same-class overlaps; conflicting classes remain and can increase FP.",
             f"Only {args.split} images and labels are used. This script does not train or select hyperparameters.",
             "A test evaluation should use the configuration already selected on val."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    lab = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=lab / "weights/pcbnet34_yolo26n_v1.pt")
    parser.add_argument("--data-root", type=Path, default=lab / "data/yolo/pcbnet34_v1")
    parser.add_argument("--output", type=Path, help="Defaults to outputs/pcbnet34_tiled_<split>.")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-imgsz", type=int, help="Tile model input size; defaults to --imgsz.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile width/height in original-image pixels.")
    parser.add_argument("--overlap", type=float, default=.25)
    parser.add_argument("--max-tiles", type=int, default=100)
    parser.add_argument("--edge-margin", type=float, default=2)
    parser.add_argument("--nms-iou", type=float, default=.5)
    parser.add_argument("--match-iou", type=float, default=.5)
    parser.add_argument("--confidence", "--conf", type=float, default=.25)
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum model detections per whole image or crop.")
    parser.add_argument("--visualizations", type=int, default=6)
    args = parser.parse_args(argv)
    args.tile_imgsz = args.imgsz if args.tile_imgsz is None else args.tile_imgsz
    if min(args.imgsz, args.tile_imgsz, args.tile_size, args.max_tiles, args.max_det) <= 0:
        parser.error("Input sizes, max-tiles and max-det must be positive.")
    if not 0 <= args.overlap < 1 or args.edge_margin < 0 or args.visualizations < 0:
        parser.error("Overlap must be in [0,1); edge-margin and visualizations must be nonnegative.")
    if not 0 <= args.confidence <= 1 or not 0 < args.nms_iou <= 1 or not 0 < args.match_iou <= 1:
        parser.error("Confidence must be in [0,1]; IoU thresholds in (0,1].")
    args.output = args.output or lab / f"outputs/pcbnet34_tiled_{args.split}"
    args.data_root, args.output, args.weights = args.data_root.resolve(), args.output.resolve(), args.weights.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from analyze_yolo_errors import load_split
    names, records = load_split(args.data_root, args.split)
    import os
    settings = args.output / "ultralytics_settings"
    settings.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
    from PIL import Image
    from ultralytics import YOLO

    started = time.perf_counter()
    model = YOLO(str(args.weights))
    load_seconds = time.perf_counter() - started
    if [model.names[index] for index in range(len(model.names))] != names:
        raise ValueError("Model class names/order do not match dataset.yaml.")
    whole_predictions, combined_predictions, per_image = {}, {}, []
    pipeline_started = time.perf_counter()
    for index, record in enumerate(records, 1):
        image_started = time.perf_counter()
        width, height = record["width"], record["height"]
        plan = tile_plan(width, height, args.tile_size, args.overlap, args.max_tiles)
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
        timer = time.perf_counter()
        whole_result = model.predict(image, imgsz=args.imgsz, device=args.device, conf=args.confidence,
                                     max_det=args.max_det, verbose=False)[0]
        whole = result_rows(whole_result, names, "whole_image")
        whole_seconds = time.perf_counter() - timer
        candidate_rows = list(whole)
        edge_discarded, invalid_discarded, tile_raw_count, tile_seconds = 0, 0, 0, 0.
        for window in plan["windows"]:
            crop = image.crop(tuple(window))
            timer = time.perf_counter()
            result = model.predict(crop, imgsz=args.tile_imgsz, device=args.device, conf=args.confidence,
                                   max_det=args.max_det, verbose=False)[0]
            local_rows = result_rows(result, names, "tile")
            tile_seconds += time.perf_counter() - timer
            tile_raw_count += len(local_rows)
            for row in local_rows:
                if touches_internal_edge(row["bbox"], window, width, height, args.edge_margin):
                    edge_discarded += 1
                    continue
                translated = translate_prediction(row, window, width, height)
                if translated is None:
                    invalid_discarded += 1
                else:
                    candidate_rows.append(translated)
        timer = time.perf_counter()
        # On skipped images keep the exact baseline, so max-tiles fallback
        # cannot alter whole-image predictions with an extra NMS pass.
        combined = list(whole) if plan["skipped"] else class_aware_nms(candidate_rows, args.nms_iou)
        nms_seconds = time.perf_counter() - timer
        whole_predictions[record["image"]], combined_predictions[record["image"]] = whole, combined
        metadata = {"image": record["image"], "split": args.split, "width": width, "height": height,
                    **{key: value for key, value in plan.items() if key != "windows"},
                    "whole_detection_count": len(whole), "tile_raw_detection_count": tile_raw_count,
                    "internal_edge_discarded": edge_discarded, "invalid_box_discarded": invalid_discarded,
                    "pre_nms_detection_count": len(candidate_rows), "combined_detection_count": len(combined),
                    "whole_inference_seconds": whole_seconds, "tile_inference_seconds": tile_seconds,
                    "merge_nms_seconds": nms_seconds, "image_pipeline_seconds": time.perf_counter() - image_started,
                    "tile_windows": plan["windows"]}
        per_image.append(metadata)
        note = f"SKIPPED {plan['required_tile_count']} required tiles" if plan["skipped"] else f"{plan['tile_count']} tiles"
        print(f"[{index}/{len(records)}] {record['image']}: {note}; whole {len(whole)}, combined {len(combined)}", flush=True)
    write_cache(args.output / "whole_image_predictions.json", args, names, whole_predictions, "whole_image")
    write_cache(args.output / "predictions.json", args, names, combined_predictions, "whole_image_plus_tiles")
    metadata = {"split": args.split, "image_count": len(records), "device": args.device,
                "whole_imgsz": args.imgsz, "tile_imgsz": args.tile_imgsz, "tile_size_original_pixels": args.tile_size,
                "overlap": args.overlap, "max_tiles": args.max_tiles, "edge_margin_original_tile_pixels": args.edge_margin,
                "merge_nms_iou": args.nms_iou, "model_max_det_per_call": args.max_det,
                "tile_count": sum(row["tile_count"] for row in per_image),
                "skipped_image_count": sum(row["skipped"] for row in per_image), "truncated_image_count": 0,
                "tile_limit_policy": "skip_entire_tile_pass_and_preserve_whole_image_predictions",
                "model_load_seconds": load_seconds, "inference_pipeline_seconds": time.perf_counter() - pipeline_started,
                "whole_inference_seconds": sum(row["whole_inference_seconds"] for row in per_image),
                "tile_inference_seconds": sum(row["tile_inference_seconds"] for row in per_image), "images": per_image}
    # Save before diagnostics so successful inference caches survive any render failure.
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostic_started = time.perf_counter()
    comparison = run_diagnostics(args)
    metadata["cpu_diagnostics_seconds"] = time.perf_counter() - diagnostic_started
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(args, metadata, comparison)
    print(json.dumps(comparison["delta"], indent=2))
    print(f"{args.split} comparison written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
