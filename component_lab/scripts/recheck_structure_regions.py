"""Bounded conflict-driven local detection; no GT chooses regions or boxes.

The active final output remains the fixed .25 whole/tile baseline. Two shadow
experiments add local >=.25 / >=.10 candidates with only near-identical
cross-pass duplicate suppression. Neither experiment merges component boxes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from PIL import Image

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

from component_lab.candidate_postprocess import (COORDINATES, box_geometry, dump_json, make_candidate,
                                               maximum_matching_coverage, sha256_file, validate_raw_cache)
from analyze_yolo_errors import count_summary, load_split, match_detections
from cache_component_candidates import recipe_baseline, resolve_image
from evaluate_tiled_yolo import result_rows


def select_regions(raw: dict, audit: dict, max_regions: int = 24, max_per_image: int = 2) -> list[dict]:
    """Rank structural hypotheses only; no labels/GT/previous metrics accepted."""
    if not 1 <= max_regions <= 24 or not 1 <= max_per_image <= 2:
        raise ValueError("Bounded run allows at most24regions and2perimage")
    proposed = []
    for key, result in audit.items():
        if key not in raw["images"]:
            raise ValueError("Structure audit image not in raw capture")
        rows = {row["candidate_id"]: row for row in result["original_candidates"]}
        raw_rows = {row["candidate_id"]: row for row in raw["predictions"][key]}
        for identifier, row in rows.items():
            if identifier not in raw_rows or any(row[field] != raw_rows[identifier][field] for field in ("bbox", "score", "class_id")):
                raise ValueError("Structural source candidate differs from raw capture")
        annotated = {row["candidate_id"]: row for row in result["annotated_candidates"]}
        width, height = raw["images"][key]["width"], raw["images"][key]["height"]
        for group in result.get("review_groups", []):
            members = [rows[identifier] for identifier in group["candidate_ids"]]
            if len(members) < 2:
                continue
            bbox = [min(row["bbox"][0] for row in members), min(row["bbox"][1] for row in members),
                    max(row["bbox"][2] for row in members), max(row["bbox"][3] for row in members)]
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if not all(math.isfinite(value) for value in bbox) or min(bw, bh) < 8:
                continue
            #20% context on every side, plus a documented64px minimum field.
            pad_x, pad_y = max(.20 * bw, (64 - bw) / 2), max(.20 * bh, (64 - bh) / 2)
            window = [max(0, math.floor(bbox[0] - pad_x)), max(0, math.floor(bbox[1] - pad_y)),
                      min(width, math.ceil(bbox[2] + pad_x)), min(height, math.ceil(bbox[3] + pad_y))]
            fraction = (window[2] - window[0]) * (window[3] - window[1]) / (width * height)
            if not 0 < fraction <= .8:
                continue
            reasons = sorted({reason for member in members for reason in
                              annotated[member["candidate_id"]]["structure"]["review_reasons"]})
            priority = (10 * ("one_large_box_multiple_smaller_candidates_unresolved" in reasons) +
                        5 * ("contained_geometry_split_or_merge_ambiguous" in reasons) +
                        3 * any(member.get("edge", {}).get("touches_internal") for member in members) +
                        2 * (len({member["source_pass"] for member in members}) > 1) +
                        1 - min(member["score"] for member in members))
            group_id = group.get("group_id") or hashlib.sha256("|".join(sorted(group["candidate_ids"])).encode()).hexdigest()[:20]
            proposed.append({"group_id": group_id, "image": key, "candidate_ids": sorted(group["candidate_ids"]),
                             "review_region_bbox": bbox, "window": window, "context_each_side_fraction": .20,
                             "context_minimum_side_pixels": 64, "window_image_area_fraction": fraction,
                             "priority": priority, "selection_reasons": reasons,
                             "selection_uses_ground_truth": False, "is_component_geometry": False})
    proposed.sort(key=lambda region: (-region["priority"], region["image"], region["group_id"]))
    selected, per_image = [], {}
    for region in proposed:
        key = region["image"]
        if per_image.get(key, 0) >= max_per_image:
            continue
        if any(previous["image"] == key and box_geometry(previous["window"], region["window"])["iou"] >= .9
               for previous in selected):
            continue
        region["region_ordinal"] = len(selected)
        selected.append(region)
        per_image[key] = per_image.get(key, 0) + 1
        if len(selected) == max_regions:
            break
    return selected


def fuse_preserving_baseline(baseline: list[dict], local: list[dict], confidence: float) -> tuple[list[dict], list[dict]]:
    """Baseline always retained; only almost-identical new rows are redundant."""
    kept, suppressed = deepcopy(baseline), []
    for row in sorted(local, key=lambda candidate: (-candidate["score"], candidate["candidate_id"])):
        if row["score"] < confidence or not row["valid"]:
            continue
        if row["edge"]["touches_internal"]:
            suppressed.append({"candidate_id": row["candidate_id"], "reason": "local_context_edge_fragment_raw_retained"})
            continue
        representative = None
        for previous in kept:
            if row["class_id"] != previous["class_id"] or previous.get("edge", {}).get("touches_internal"):
                continue
            geometry = box_geometry(row["bbox"], previous["bbox"])
            if geometry["iou"] >= .95 and geometry["smaller_to_larger_area"] >= .95:
                representative = previous
                break
        if representative:
            suppressed.append({"candidate_id": row["candidate_id"], "representative_id": representative["candidate_id"],
                               "reason": "near_identical_new_candidate_only_original_baseline_retained"})
        else:
            kept.append(deepcopy(row))
    return kept, suppressed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-cache", required=True, type=Path)
    parser.add_argument("--structure-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-regions", type=int, default=24)
    parser.add_argument("--max-per-image", type=int, default=2)
    parser.add_argument("--max-det", type=int, default=512)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    raw = json.loads(args.raw_cache.read_text(encoding="utf-8-sig"))
    validate_raw_cache(raw)
    if not raw.get("capture_complete") or raw.get("split") != "val" or raw["generation"]["confidence"] > .25:
        raise ValueError("Complete val raw cache captured at<=.25 required; no final test data")
    audit = json.loads(args.structure_audit.read_text(encoding="utf-8-sig"))
    regions = select_regions(raw, audit, args.max_regions, args.max_per_image)
    data_root = (args.data_root or Path(raw["data_root"])).resolve()
    weights = (args.weights or Path(raw["weights"]["path"])).resolve()
    if sha256_file(weights) != raw["weights"]["sha256"]:
        raise ValueError("Weights must match baseline model exactly")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists() or (args.output / "local_capture.jsonl").exists():
        raise ValueError("Preserve existing results: use a new output directory")
    registration = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "experiment": "bounded_structural_local_recheck_v1",
                    "evaluation_status": "val_development_not_independent_test", "selection_uses_ground_truth": False,
                    "raw_cache_sha256": sha256_file(args.raw_cache), "audit_sha256": sha256_file(args.structure_audit),
                    "weights": {"path": str(weights), "sha256": raw["weights"]["sha256"]},
                    "baseline_confidence": .25, "capture_confidence": .10, "nms_iou_baseline": .5,
                    "policies": {"baseline25": .25, "base25_plus_local25": .25, "base25_plus_local10": .10},
                    "duplicate_iou": .95, "duplicate_minimum_area_ratio": .95, "max_regions": args.max_regions,
                    "max_per_image": args.max_per_image, "selected_regions": len(regions), "imgsz": args.imgsz,
                    "device": args.device, "max_det_per_call": args.max_det, "region_call_limit": 24,
                    "local_edge_fragment_policy": "retain_raw_exclude_shadow_fusion",
                    "active_final_output": "baseline25_unchanged", "no_threshold_search": True,
                    "api_calls": 0, "script_sha256": sha256_file(Path(__file__))}
    dump_json(args.output / "registration.json", registration)
    dump_json(args.output / "region_plan.json", {"registration": registration, "regions": regions})
    if args.plan_only:
        print(json.dumps({"region_plan": str(args.output / "region_plan.json"), "regions": len(regions)}, ensure_ascii=False))
        return 0
    baseline = recipe_baseline(raw, .25, .5)
    local = {key: [] for key in baseline}
    #The plan is finalized before any model inference or label access.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output / "ultralytics_settings"))
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if [model.names[index] for index in range(len(model.names))] != raw["names"]:
        raise ValueError("Model classes differ from raw cache")
    capture = []
    with (args.output / "local_capture.jsonl").open("x", encoding="utf-8") as journal:
        for index, region in enumerate(regions, 1):
            key = region["image"]
            image_path = resolve_image(raw, key, data_root)
            if sha256_file(image_path) != raw["images"][key]["sha256"]:
                raise ValueError("Image changed since baseline capture")
            with Image.open(image_path) as source:
                crop = source.convert("RGB").crop(region["window"])
            start = time.perf_counter()
            prediction = model.predict(crop, imgsz=args.imgsz, device=args.device, conf=.10,
                                       max_det=args.max_det, verbose=False)[0]
            returned = result_rows(prediction, raw["names"], "tile")
            metadata = raw["images"][key]
            rows = [make_candidate(key, "structural_recheck_" + region["group_id"], ordinal, row, "tile",
                                   region["window"], metadata["width"], metadata["height"], args.imgsz)
                    for ordinal, row in enumerate(returned)]
            for row in rows:
                row["recheck_provenance"] = {"region_group_id": region["group_id"], "selection_uses_ground_truth": False,
                                             "raw_cache_sha256": registration["raw_cache_sha256"],
                                             "context_window": region["window"]}
            record = {"region": region, "inference_seconds": time.perf_counter() - start,
                      "max_det_reached": len(returned) >= args.max_det, "candidates": rows}
            local[key].extend(rows)
            capture.append(record)
            journal.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            print(f"[{index}/{len(regions)}] {key}: local{returned.__len__()}", flush=True)
    dump_json(args.output / "local_raw_candidates.json", {"registration": registration, "capture": capture, "predictions": local})
    stages = {"baseline25": baseline}
    audits = {}
    for stage, confidence in (("base25_plus_local25", .25), ("base25_plus_local10", .10)):
        stages[stage], audits[stage] = {}, {}
        for key in baseline:
            stages[stage][key], audits[stage][key] = fuse_preserving_baseline(baseline[key], local[key], confidence)
    common = {"schema_version": 1, "coordinate_system": COORDINATES, "split": "val", "names": raw["names"],
              "data_root": str(data_root), "weights_sha256": raw["weights"]["sha256"],
              "raw_cache_sha256": registration["raw_cache_sha256"], "promotion": "shadow_only"}
    for stage, predictions in stages.items():
        dump_json(args.output / (stage + "_predictions.json"), {**common, "stage": stage,
                                                               "confidence": registration["policies"][stage],
                                                               "predictions": predictions})
    dump_json(args.output / "final_predictions.json", {**common, "stage": "baseline25_unchanged", "confidence": .25,
                                                       "predictions": baseline})
    dump_json(args.output / "fusion_audit.json", audits)
    #Only now load GT, strictly for reporting and never box selection.
    names, records = load_split(data_root, "val")
    if names != raw["names"] or {record["image"] for record in records} != set(baseline):
        raise ValueError("Evaluation data differ from raw cache")
    metrics, matchings, coverage = {}, {}, {}
    for stage, predictions in stages.items():
        totals, matchings[stage], maximal = [0, 0, 0], {}, 0
        for record in records:
            key = record["image"]
            matching = match_detections(record["ground_truth"], predictions[key], .5, registration["policies"][stage])
            matchings[stage][key] = matching
            totals[0] += len(matching["matches"])
            totals[1] += len(matching["false_positive_indices"])
            totals[2] += len(matching["false_negative_indices"])
            maximal += maximum_matching_coverage(record["ground_truth"], predictions[key], .5)["matched_maximum"]
        metrics[stage] = count_summary(*totals)
        coverage[stage] = {"matched_maximum": maximal, "ground_truth": totals[0] + totals[2],
                           "recall_upper_bound": maximal / (totals[0] + totals[2]),
                           "warning": "GT_assisted_diagnostic_not_oracle_selected_predictions"}
    changes = {}
    for stage in ("base25_plus_local25", "base25_plus_local10"):
        gained, lost = [], []
        for record in records:
            key = record["image"]
            before = {match["ground_truth_index"] for match in matchings["baseline25"][key]["matches"]}
            after = {match["ground_truth_index"] for match in matchings[stage][key]["matches"]}
            gained.extend({"image": key, "ground_truth_index": value} for value in sorted(after - before))
            lost.extend({"image": key, "ground_truth_index": value} for value in sorted(before - after))
        changes[stage] = {"gained_matching_targets": gained, "lost_baseline_matching_targets": lost,
                          "gained_count": len(gained), "lost_count": len(lost),
                          "delta_false_positives": metrics[stage]["fp"] - metrics["baseline25"]["fp"]}
    raw_coverage = 0
    for record in records:
        key = record["image"]
        original = [row for row in raw["predictions"][key] if row["valid"]]
        raw_coverage += maximum_matching_coverage(record["ground_truth"], original + local[key], .5)["matched_maximum"]
    gt_count = sum(len(record["ground_truth"]) for record in records)
    summary = {**registration, "metrics": metrics, "maximum_one_to_one_coverage": coverage, "changes": changes,
               "raw_original_plus_local_maximum_coverage": {"matched_maximum": raw_coverage, "ground_truth": gt_count,
                                                           "recall_upper_bound": raw_coverage / gt_count,
                                                           "warning": "GT_diagnostic_all_raw_scores_not_realized_recall"},
               "model_calls": len(capture), "local_candidates": sum(len(rows) for rows in local.values()),
               "label_sha256": {record["image"]: sha256_file(data_root / "labels" / Path(*Path(record["image"]).parts[1:]).with_suffix(".txt"))
                                for record in records},
               "limitations": ["No automatic decisions about split/merge truth.", "Local10 adds lower-score candidates; it is not a same-score comparison.",
                               "All original predictions preserved; added high-score false positives may change greedy matching.",
                               "No independent final test; no default model promotion."]}
    dump_json(args.output / "summary.json", summary)
    lines = ["# 局部重检开发集实验", "", "区域只由结构冲突、拼图边缘和分数选择，未用GT挑区域；最多24区、每图2区。", "",
             "|策略|TP|FP|FN|精确率|召回率|", "|---|---:|---:|---:|---:|---:|"]
    for stage, value in metrics.items():
        lines.append(f"|{stage}|{value['tp']}|{value['fp']}|{value['fn']}|{value['precision']:.2%}|{value['recall']:.2%}|")
    lines += ["", "base25_plus_local10在原基线0.25上增加局部0.10候选，不是相同置信度比较。两种策略只抑制几乎相同的新框，"
              "不自动合并/拆分器件。保留原基线和全部局部原始结果，默认最终输出仍是原基线。", "",
              "使用验证集开发用途；不是mAP或独立最终测试。更高的原始覆盖上界不能视作已实现召回率。", ""]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "model_calls": len(capture), "metrics": metrics}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
