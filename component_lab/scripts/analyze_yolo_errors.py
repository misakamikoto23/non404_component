"""Diagnose held-out detection errors at one fixed confidence threshold.

This reports counts and recall, not mAP. Cached predictions let the matching
and rendering steps run without importing torch or using a GPU. Coordinates
in caches and reports are xyxy pixels with a top-left origin in the original
image. The default split is val. Select test only for a final evaluation of a
configuration already chosen on validation data; this tool does not tune it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


AREA_BUCKETS = ("small_lt_32_squared", "medium_32_to_96_squared", "large_ge_96_squared")
SCALED_SIDE_BUCKETS = ("estimated_lt_4px", "estimated_4_to_8px", "estimated_8_to_16px", "estimated_ge_16px")


def box_iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def area_bucket(bbox: list[float]) -> str:
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    return AREA_BUCKETS[0] if area < 32**2 else AREA_BUCKETS[1] if area < 96**2 else AREA_BUCKETS[2]


def scaled_side_bucket(bbox: list[float], width: int, height: int, imgsz: int) -> str:
    """Estimate size after isotropic resize, without assuming actual model grids.

    This deliberately follows min(imgsz/W, imgsz/H); internal preprocessing,
    padding, rounding, rectangular inference or scale-up settings may differ.
    """
    scale = min(imgsz / width, imgsz / height)
    side = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) * scale
    return (SCALED_SIDE_BUCKETS[0] if side < 4 else SCALED_SIDE_BUCKETS[1] if side < 8 else
            SCALED_SIDE_BUCKETS[2] if side < 16 else SCALED_SIDE_BUCKETS[3])


def valid_box(bbox: Any) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Expected four xyxy coordinates, received {bbox!r}")
    values = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in values) or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"Invalid bounding box: {bbox!r}")
    return values


def match_detections(ground_truth: list[dict], predictions: list[dict], iou_threshold: float = 0.5,
                     confidence: float = 0.25) -> dict:
    """Confidence-ordered, class-aware, one-to-one matching within one image.

    Each prediction takes its highest-IoU still-unmatched ground truth of the
    same class if IoU meets the threshold. Indices refer to the input lists.
    """
    if not 0 <= confidence <= 1 or not 0 < iou_threshold <= 1:
        raise ValueError("Confidence must be in [0, 1] and IoU in (0, 1].")
    eligible = sorted((i for i, pred in enumerate(predictions) if pred["score"] >= confidence),
                      key=lambda index: (-predictions[index]["score"], index))
    unmatched = set(range(len(ground_truth)))
    matched, false_positives = [], []
    for pred_index in eligible:
        pred = predictions[pred_index]
        candidates = [(box_iou(pred["bbox"], ground_truth[index]["bbox"]), index)
                      for index in sorted(unmatched) if ground_truth[index]["class_id"] == pred["class_id"]]
        overlap, gt_index = max(candidates, key=lambda pair: pair[0]) if candidates else (0.0, None)
        if gt_index is not None and overlap >= iou_threshold:
            unmatched.remove(gt_index)
            matched.append({"prediction_index": pred_index, "ground_truth_index": gt_index, "iou": overlap})
        else:
            false_positives.append(pred_index)
    return {"matches": matched, "false_positive_indices": false_positives,
            "false_negative_indices": sorted(unmatched), "evaluated_predictions": eligible}


def count_summary(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn, "ground_truth": tp + fn,
            "predictions": tp + fp, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def summarize(records: list[dict], names: list[str], imgsz: int = 1280) -> dict:
    counts = [[0, 0, 0] for _ in names]
    buckets = {name: {"tp": 0, "fn": 0} for name in AREA_BUCKETS}
    resized_buckets = {name: {"tp": 0, "fn": 0} for name in SCALED_SIDE_BUCKETS}
    for record in records:
        gt, preds, matching = record["ground_truth"], record["predictions"], record["matching"]
        for match in matching["matches"]:
            target = gt[match["ground_truth_index"]]
            counts[target["class_id"]][0] += 1
            buckets[area_bucket(target["bbox"])]["tp"] += 1
            resized_buckets[scaled_side_bucket(target["bbox"], record["width"], record["height"], imgsz)]["tp"] += 1
        for index in matching["false_positive_indices"]:
            counts[preds[index]["class_id"]][1] += 1
        for index in matching["false_negative_indices"]:
            target = gt[index]
            counts[target["class_id"]][2] += 1
            buckets[area_bucket(target["bbox"])]["fn"] += 1
            resized_buckets[scaled_side_bucket(target["bbox"], record["width"], record["height"], imgsz)]["fn"] += 1
    by_class = [{"class_id": index, "name": name, **count_summary(*counts[index])}
                for index, name in enumerate(names)]
    by_area = []
    for name, values in buckets.items():
        total = values["tp"] + values["fn"]
        by_area.append({"bucket": name, **values, "ground_truth": total,
                        "recall": values["tp"] / total if total else None})
    totals = [sum(row[index] for row in counts) for index in range(3)]
    by_resized_side = []
    for name, values in resized_buckets.items():
        total = values["tp"] + values["fn"]
        by_resized_side.append({"bucket": name, **values, "ground_truth": total,
                                "recall": values["tp"] / total if total else None})
    return {"overall": count_summary(*totals), "by_class": by_class, "by_original_area": by_area,
            "by_estimated_resized_short_side": by_resized_side}


def load_split(data_root: Path, split: str = "val") -> tuple[list[str], list[dict]]:
    if split not in {"val", "test"}:
        raise ValueError("Only val and test splits are supported.")
    data_root = data_root.resolve()
    import yaml
    from PIL import Image

    config = yaml.safe_load((data_root / "dataset.yaml").read_text(encoding="utf-8-sig"))
    raw_names = config["names"]
    if isinstance(raw_names, dict):
        indexed = {int(index): str(name) for index, name in raw_names.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("dataset.yaml names must use contiguous class indices starting at zero.")
        names = [indexed[index] for index in range(len(indexed))]
    else:
        names = [str(name) for name in raw_names]
    directory = Path(config.get(split, f"images/{split}"))
    image_directory = directory if directory.is_absolute() else data_root / directory
    image_directory = image_directory.resolve()
    if not image_directory.is_relative_to(data_root.resolve()) or not image_directory.is_dir():
        raise ValueError(f"{split} must point to an image directory inside --data-root.")
    image_paths = sorted(path for path in image_directory.rglob("*")
                         if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
    if not image_paths:
        raise ValueError(f"No {split} images found: {image_directory}")
    records = []
    for image_path in image_paths:
        relative = image_path.relative_to(data_root)
        if relative.parts[0] != "images":
            raise ValueError(f"Expected {split} images under data-root/images with matching labels/ paths.")
        label_path = data_root / "labels" / Path(*relative.parts[1:]).with_suffix(".txt")
        if not label_path.is_file():
            raise ValueError(f"Missing label file; refusing to assume an unlabelled image is negative: {label_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        gt = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Expected YOLO detection label with 5 fields: {label_path}:{line_number}")
            class_id = int(fields[0])
            cx, cy, bw, bh = map(float, fields[1:])
            if not 0 <= class_id < len(names) or not all(math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, bw, bh)):
                raise ValueError(f"Invalid YOLO class or normalized coordinates: {label_path}:{line_number}")
            bbox = valid_box([(cx - bw / 2) * width, (cy - bh / 2) * height,
                              (cx + bw / 2) * width, (cy + bh / 2) * height])
            gt.append({"class_id": class_id, "type": names[class_id], "bbox": bbox})
        records.append({"image": relative.as_posix(), "image_path": str(image_path), "split": split,
                        "width": width, "height": height, "ground_truth": gt})
    return names, records


def load_validation(data_root: Path) -> tuple[list[str], list[dict]]:
    """Compatibility entry point for callers intentionally limited to val."""
    return load_split(data_root, "val")


def load_predictions(path: Path, records: list[dict], names: list[str], confidence: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    record_splits = {record.get("split", "val") for record in records}
    if len(record_splits) > 1:
        raise ValueError("Prediction loading requires records from one split.")
    expected_split = next(iter(record_splits), "val")
    if "predictions" in payload:
        if "split" in payload and payload["split"] != expected_split:
            raise ValueError(f"Prediction cache split {payload['split']!r} does not match requested {expected_split!r}.")
        cache_threshold = payload.get("confidence")
        if cache_threshold is not None and cache_threshold > confidence:
            raise ValueError("Cache was generated at a higher confidence threshold; lower-score predictions are unavailable.")
        if "names" in payload and payload["names"] != names:
            raise ValueError("Prediction cache class order does not match dataset.yaml.")
        payload = payload["predictions"]
    if not isinstance(payload, dict):
        raise ValueError("Prediction JSON must contain a map from relative image path to prediction rows.")
    result = {}
    for record in records:
        key = record["image"]
        if key not in payload:
            display_split = "validation" if expected_split == "val" else "test"
            raise ValueError(f"Prediction cache missing {display_split} image {key}; use [] for an image with no detections.")
        rows = []
        for prediction in payload[key]:
            class_id, score = int(prediction["class_id"]), float(prediction["score"])
            if not 0 <= class_id < len(names) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid prediction class or confidence for {key}")
            rows.append({"bbox": valid_box(prediction["bbox"]), "class_id": class_id,
                         "type": names[class_id], "score": score})
        result[key] = rows
    return result


def predict_validation(args: argparse.Namespace, records: list[dict], names: list[str]) -> dict:
    import os
    settings = args.output / "ultralytics_settings"
    settings.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model_names = [model.names[index] for index in range(len(model.names))]
    if model_names != names:
        raise ValueError("Model class names/order differ from dataset.yaml; load the matching custom-trained weights.")
    predictions = {}
    for index, record in enumerate(records, 1):
        result = model.predict(record["image_path"], imgsz=args.imgsz, device=args.device,
                               conf=args.confidence, max_det=args.max_det, verbose=False)[0]
        predictions[record["image"]] = [{"bbox": bbox, "class_id": int(class_id),
                                           "type": names[int(class_id)], "score": score}
                                          for bbox, class_id, score in zip(result.boxes.xyxy.cpu().tolist(),
                                                                          result.boxes.cls.cpu().tolist(),
                                                                          result.boxes.conf.cpu().tolist())]
        print(f"[{index}/{len(records)}] {record['image']}", flush=True)
    return predictions


def render_errors(record: dict, output: Path) -> None:
    from PIL import Image, ImageDraw

    with Image.open(record["image_path"]) as source:
        image = source.convert("RGB")
    image.thumbnail((2400, 2400))
    sx, sy = image.width / record["width"], image.height / record["height"]
    canvas = Image.new("RGB", (image.width, image.height + 40), "white")
    canvas.paste(image, (0, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "Green: matched GT | Red: missed GT | Orange: false positive prediction", fill="black")
    missed = set(record["matching"]["false_negative_indices"])
    entries = [(row, "red" if index in missed else "green", "FN" if index in missed else "TP")
               for index, row in enumerate(record["ground_truth"])]
    entries.extend((record["predictions"][index], "#ff8c00", "FP")
                   for index in record["matching"]["false_positive_indices"])
    for row, color, state in entries:
        x1, y1, x2, y2 = row["bbox"]
        box = (x1 * sx, y1 * sy + 40, x2 * sx, y2 * sy + 40)
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], max(40, box[1] - 12)), f"{state} {row['type']}", fill=color)
    canvas.save(output)


def write_reports(args: argparse.Namespace, records: list[dict], names: list[str]) -> dict:
    report = {"split": args.split, "metric": "fixed_threshold_class_aware_greedy_counts_not_mAP",
              "confidence": args.confidence, "iou_threshold": args.iou, "image_count": len(records),
              "coordinate_system": "xyxy_pixels_top_left_in_original_image",
              "area_definition": "GT width*height in original pixels: <32^2; [32^2,96^2); >=96^2. This is not COCO AP.",
              "estimated_side_definition": "min(GT width,GT height)*min(imgsz/image_width,imgsz/image_height), bins <4, [4,8), [8,16), >=16 px. Estimate only, not actual internal model grids or preprocessing.",
              "imgsz": args.imgsz, "max_det": args.max_det, "weights": str(args.weights),
              "prediction_source": str(args.predictions_json) if args.predictions_json else "model_inference",
              **summarize(records, names, args.imgsz)}
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "per_class.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["by_class"][0]))
        writer.writeheader()
        writer.writerows(report["by_class"])
    ranked = sorted(records, key=lambda row: (-(len(row["matching"]["false_negative_indices"]) +
                                                  len(row["matching"]["false_positive_indices"])), row["image"]))
    errors = []
    for record in ranked:
        matching = record["matching"]
        if not matching["false_negative_indices"] and not matching["false_positive_indices"]:
            continue
        errors.append({"image": record["image"], "width": record["width"], "height": record["height"],
                       "false_negatives": [record["ground_truth"][index] for index in matching["false_negative_indices"]],
                       "false_positives": [record["predictions"][index] for index in matching["false_positive_indices"]],
                       "matched_count": len(matching["matches"])})
    (args.output / "error_cases.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.visualizations:
        directory = args.output / "overlays"
        directory.mkdir(exist_ok=True)
        error_images = {row["image"] for row in errors[:args.visualizations]}
        for record in ranked:
            if record["image"] in error_images:
                filename = "__".join(Path(record["image"]).with_suffix(".png").parts)
                render_errors(record, directory / filename)
    percent = lambda value: "n/a" if value is None else f"{value:.1%}"
    overall = report["overall"]
    lines = [f"# {args.split.capitalize()} detection error diagnosis", "", "Fixed-threshold counts; these are not mAP/AP scores.", "",
             f"Split: {args.split}.", "",
             f"Images: {len(records)}; confidence: {args.confidence}; class-aware IoU: {args.iou}.", "",
             f"TP {overall['tp']}, FP {overall['fp']}, FN {overall['fn']}; precision {percent(overall['precision'])}, recall {percent(overall['recall'])}.",
             "", "Area buckets use ground-truth area in original image pixels; changing image size does not change bucket membership.", "",
             "| Original GT area | GT | TP | FN | Recall |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {row['bucket']} | {row['ground_truth']} | {row['tp']} | {row['fn']} | {percent(row['recall'])} |" for row in report["by_original_area"])
    lines += ["", "Estimated resized short side = min(GT width, GT height) * min(imgsz/image_width, imgsz/image_height).",
              "These are estimates, not actual internal feature grids or an exact account of preprocessing/padding/rounding.", "",
              "| Estimated resized short side | GT | TP | FN | Recall |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {row['bucket']} | {row['ground_truth']} | {row['tp']} | {row['fn']} | {percent(row['recall'])} |"
                 for row in report["by_estimated_resized_short_side"])
    lines += ["", "Per-class counts are in per_class.csv; full metrics in metrics.json; errors ranked by FP+FN in error_cases.json.",
              "Green overlay boxes are matched ground truth; red are missed ground truth; orange are false positive predictions.",
              "A duplicate prediction is FP. A wrong-class prediction can produce both FP and FN. Classes absent from the selected split have undefined (null) recall.",
              f"Only {args.split} images and labels are used. These counts describe annotations as provided and are not a manual audit of annotation quality.",
              "This script does not select hyperparameters. A test evaluation should use the configuration already selected on val."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    lab = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=lab / "weights/pcbnet34_yolo26n_v1.pt")
    parser.add_argument("--data-root", type=Path, default=lab / "data/yolo/pcbnet34_v1")
    parser.add_argument("--output", type=Path, help="Defaults to outputs/pcbnet34_<split>_errors.")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", "--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for GT matching, not NMS.")
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--predictions-json", type=Path,
                        help='Cache map: {"images/val/0001.png": [{"bbox": [x1,y1,x2,y2], "class_id": 0, "score": .9}]}')
    parser.add_argument("--visualizations", type=int, default=12)
    args = parser.parse_args(argv)
    if not 0 <= args.confidence <= 1 or not 0 < args.iou <= 1:
        parser.error("confidence must be in [0,1], IoU in (0,1]")
    if args.visualizations < 0 or args.imgsz <= 0 or args.max_det <= 0:
        parser.error("visualizations must be nonnegative; imgsz and max-det must be positive")
    args.output = args.output or lab / f"outputs/pcbnet34_{args.split}_errors"
    args.data_root, args.output = args.data_root.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    names, records = load_split(args.data_root, args.split)
    predictions = (load_predictions(args.predictions_json, records, names, args.confidence) if args.predictions_json
                   else predict_validation(args, records, names))
    cache = {"schema_version": 1, "coordinate_system": "xyxy_pixels_top_left_in_original_image", "split": args.split,
             "data_root": str(args.data_root), "names": names, "confidence": args.confidence,
             "imgsz": args.imgsz, "weights": str(args.weights), "predictions": predictions}
    (args.output / "predictions.json").write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    for record in records:
        record["predictions"] = predictions[record["image"]]
        record["matching"] = match_detections(record["ground_truth"], record["predictions"], args.iou, args.confidence)
    report = write_reports(args, records, names)
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(f"Reports written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
