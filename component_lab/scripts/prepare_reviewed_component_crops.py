"""Create local, neutral-name component crops from a frozen reviewed batch.

This script never calls a network service. Original PCBnet categories are only
evaluation proxies, kept separate from crop pixels and never rendered on them.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.optimize import linear_sum_assignment

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
from component_lab.review_workflow import checked_box, load_batch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou_matrix(boxes, originals):
    result = np.zeros((len(boxes), len(originals)), dtype=np.float64)
    if not boxes or not originals:
        return result
    a = np.asarray([box["bbox"] for box in boxes], dtype=np.float64)
    b = np.asarray([box["bbox"] for box in originals], dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / np.maximum(aa[:, None] + bb[None, :] - intersection, 1e-12)


def assign_one_to_one(ious, threshold):
    """Maximize eligible pair count, then total IoU, without reading categories."""
    if not ious.size:
        return {}
    # The cardinality bonus exceeds any possible change in summed IoU.
    bonus = min(ious.shape) + 1
    scores = np.where(ious >= threshold, bonus + ious, 0.0)
    rows, cols = linear_sum_assignment(-scores)
    return {int(r): int(c) for r, c in zip(rows, cols)
            if ious[r, c] >= threshold}


def integer_crops(box, width, height, context_fraction, context_min_px):
    x1, y1, x2, y2 = checked_box(box, width, height)
    tight = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
    padx = max(context_min_px, math.ceil((x2 - x1) * context_fraction))
    pady = max(context_min_px, math.ceil((y2 - y1) * context_fraction))
    context = [max(0, tight[0] - padx), max(0, tight[1] - pady),
               min(width, tight[2] + padx), min(height, tight[3] + pady)]
    return tight, context


def dimensions_summary(dimensions):
    values = np.asarray(dimensions, dtype=np.int64)
    if not len(values):
        return {"count": 0}
    result = {"count": len(values)}
    for index, name in enumerate(("width", "height")):
        result[name] = {"min": int(values[:, index].min()),
                        "p10": float(np.percentile(values[:, index], 10)),
                        "median": float(np.median(values[:, index])),
                        "p90": float(np.percentile(values[:, index], 90)),
                        "max": int(values[:, index].max())}
    result["minimum_side_under_12px"] = int((values.min(axis=1) < 12).sum())
    result["minimum_side_under_20px"] = int((values.min(axis=1) < 20).sum())
    return result


def create_contact_sheet(records, output, count=48, seed=20261015):
    selected = random.Random(seed).sample(records, min(count, len(records)))
    columns = 8
    cell_w, cell_h = 176, 160
    sheet = Image.new("RGB", (columns * cell_w,
                             math.ceil(len(selected) / columns) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(selected):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        with Image.open(output / record["context_image_path"]) as crop:
            thumb = ImageOps.contain(crop.convert("RGB"), (cell_w - 12, cell_h - 32))
            sheet.paste(thumb, (x + (cell_w - thumb.width) // 2,
                               y + 23 + (cell_h - 32 - thumb.height) // 2))
        draw.text((x + 5, y + 5), record["crop_id"], fill="black")
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="#c8c8c8")
    sheet.save(output / "contact_sheet_48.png")
    write_json(output / "contact_sheet_48_ids.json", [r["crop_id"] for r in selected])


def prepare(snapshot, output, context_fraction=.08, context_min_px=2):
    snapshot = snapshot.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    batch = load_batch(snapshot)
    if len(batch["items"]) != 20:
        raise ValueError("Expected the frozen first 20-image review batch")
    if any(not item["review"]["completed"] for item in batch["items"]):
        raise ValueError("All snapshot images must have completed review")
    accepted_count = sum(box["review_status"] == "accepted"
                         for item in batch["items"] for box in item["boxes"])
    if accepted_count != 1687:
        raise ValueError(f"Frozen round 001 expected 1687 accepted boxes, got {accepted_count}")
    output.mkdir(parents=True)
    (output / "tight").mkdir()
    (output / "context").mkdir()
    records, image_summaries = [], []
    threshold_counts = Counter()
    proxy_category_counts = Counter()
    tight_dimensions, context_dimensions = [], []
    for item in batch["items"]:
        ident = str(item["id"])
        source_path = (snapshot / item["image_file"]).resolve()
        source_path.relative_to(snapshot / "images")
        source_sha = sha256_file(source_path)
        boxes = [b for b in item["boxes"] if b["review_status"] == "accepted"]
        originals = item.get("original_boxes", [])
        ious = iou_matrix(boxes, originals)
        assignments = {str(t): assign_one_to_one(ious, t) for t in (.5, .7, .9)}
        for threshold, matched in assignments.items():
            threshold_counts[threshold] += len(matched)
        with Image.open(source_path) as source:
            im = source.convert("RGB")
            if im.size != (item["width"], item["height"]):
                raise ValueError(f"Snapshot image dimensions mismatch: {ident}")
            for index, box in enumerate(boxes):
                crop_id = f"{ident}_{index + 1:04d}"
                tight, context = integer_crops(box["bbox"], *im.size,
                                               context_fraction, context_min_px)
                paths = {"tight": f"tight/{crop_id}.png", "context": f"context/{crop_id}.png"}
                for kind, pixels in (("tight", tight), ("context", context)):
                    im.crop(tuple(pixels)).save(output / paths[kind])
                tight_dimensions.append([tight[2] - tight[0], tight[3] - tight[1]])
                context_dimensions.append([context[2] - context[0], context[3] - context[1]])
                ranked = np.argsort(-ious[index], kind="stable").tolist() if originals else []
                top = ranked[:2]
                best_iou = float(ious[index, top[0]]) if top else 0.0
                second_iou = float(ious[index, top[1]]) if len(top) > 1 else 0.0
                assigned = assignments["0.5"].get(index)
                matched = originals[assigned] if assigned is not None else None
                if matched:
                    proxy_category_counts[matched["category"]] += 1
                record = {
                    "crop_id": crop_id, "image_id": ident, "source_box_id": box["id"],
                    "bbox_xyxy_float": box["bbox"], "tight_pixel_box_xyxy": tight,
                    "context_pixel_box_xyxy": context,
                    "tight_image_path": paths["tight"], "context_image_path": paths["context"],
                    "tight_size_wh": tight_dimensions[-1], "context_size_wh": context_dimensions[-1],
                    "source_image_path": str(source_path), "source_image_sha256": source_sha,
                    "review_status": "accepted", "inherited_category": box.get("category", "component"),
                    "inherited_category_confirmed": bool(box.get("category_confirmed", False)),
                    "evaluation_proxy": {
                        "usage": "local_evaluation_only_never_send_to_classifier_api",
                        "is_human_verified_class_label": False,
                        "matched_original_component_id": matched["id"] if matched else None,
                        "matched_original_category": matched["category"] if matched else None,
                        "matched_iou": float(ious[index, assigned]) if matched else None,
                        "one_to_one_matches": {
                            t: {"component_id": originals[pairs[index]]["id"],
                                "category": originals[pairs[index]]["category"],
                                "iou": float(ious[index, pairs[index]])}
                            if index in pairs else None for t, pairs in assignments.items()},
                        "best_overlap": {"component_id": originals[top[0]]["id"],
                                         "category": originals[top[0]]["category"], "iou": best_iou}
                        if top else None,
                        "second_best_overlap": {"component_id": originals[top[1]]["id"],
                                                "category": originals[top[1]]["category"], "iou": second_iou}
                        if len(top) > 1 else None,
                        "ambiguous": bool(second_iou >= .5 and best_iou - second_iou <= .1),
                        "assigned_differs_from_best_overlap": bool(assigned is not None and top and assigned != top[0]),
                        "ambiguity_policy": "second IoU >= .5 and best-minus-second <= .1",
                    },
                }
                records.append(record)
        if sha256_file(source_path) != source_sha:
            raise ValueError(f"Snapshot source changed during crop creation: {ident}")
        image_summaries.append({"image_id": ident, "source_image_sha256": source_sha,
                                "accepted": len(boxes), "original_labels": len(originals),
                                "matched_by_threshold": {t: len(p) for t, p in assignments.items()}})
        print(f"{ident}: {len(boxes)} accepted crops", flush=True)
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    create_contact_sheet(records, output)
    original_count = sum(row["original_labels"] for row in image_summaries)
    summary = {
        "schema_version": 1, "snapshot": str(snapshot), "batch_id": batch["batch_id"],
        "snapshot_loaded_sha256": hashlib.sha256(json.dumps(batch, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "image_count": len(image_summaries), "accepted_crops": len(records),
        "rejected_boxes_excluded": sum(b["review_status"] == "rejected" for i in batch["items"] for b in i["boxes"]),
        "source_split": "train", "val_test_images_used": False,
        "crop_policy": {"tight": "floor x1/y1 and ceil x2/y2, original RGB pixels",
                        "context_fraction_each_side": context_fraction,
                        "context_min_pixels_each_side": context_min_px,
                        "context_clamped_to_original_image": True,
                        "drawn_labels_or_gt_overlays": False},
        "classification_label_policy": "Human acceptance confirms geometry only. Original PCBnet labels remain unverified proxies, not confirmed ResNet labels. Never send proxy fields to classification APIs.",
        "inherited_category_confirmed_count": sum(r["inherited_category_confirmed"] for r in records),
        "matching_policy": "Category-blind Hungarian one-to-one assignment maximizes eligible pair count then summed IoU; assignments independently recomputed at each threshold.",
        "original_labels": original_count,
        "matches_by_iou_threshold": {t: {"matched_crops": n, "crop_coverage": n / len(records),
                                         "original_label_coverage": n / original_count}
                                     for t, n in threshold_counts.items()},
        "proxy_category_counts_at_iou_0_5": dict(proxy_category_counts),
        "ambiguous_crops": sum(r["evaluation_proxy"]["ambiguous"] for r in records),
        "assignment_not_best_overlap": sum(r["evaluation_proxy"]["assigned_differs_from_best_overlap"] for r in records),
        "tight_dimensions": dimensions_summary(tight_dimensions),
        "context_dimensions": dimensions_summary(context_dimensions),
        "images": image_summaries,
        "api_calls": 0,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "accepted_crops": len(records),
                      "matching": summary["matches_by_iou_threshold"]}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=LAB / "data/review_snapshots/batch_001_round_001")
    parser.add_argument("--output", type=Path, default=LAB / "data/component_crops/review_round_001")
    parser.add_argument("--context-fraction", type=float, default=.08)
    parser.add_argument("--context-min-px", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.context_fraction <= .5 or not 0 <= args.context_min_px <= 100:
        parser.error("Context padding must be bounded and nonnegative")
    prepare(args.snapshot, args.output, args.context_fraction, args.context_min_px)
