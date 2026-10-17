"""Replay two preregistered CPU structural policies on cached val detections.

This is validation-development evidence, not an independent final test or mAP.
No fitting, threshold sweeps, model inference, or external API calls occur.
Original results and every suppressed frame hypothesis remain available.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))

from component_lab.candidate_postprocess import COORDINATES, dump_json, sha256_file, validate_raw_cache
from component_lab.structure_checks import POLICY_VERSION, SETTINGS, annotate_image
from analyze_yolo_errors import count_summary, load_split, match_detections
from cache_component_candidates import recipe_baseline, resolve_image


def _render(image_path: Path, result: dict, destination: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    original_width, original_height = image.size
    image.thumbnail((1800, 1800))
    sx, sy = image.width / original_width, image.height / original_height
    canvas = Image.new("RGB", (image.width, image.height + 50), "white")
    canvas.paste(image, (0, 50))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Orange: review only | Red: optional page-frame suppression | Green: retained", fill="black")
    draw.text((8, 25), "No box union, split, class changes, or coordinate edits are applied.", fill="black")
    for row in result["annotated_candidates"]:
        structure = row["structure"]
        color = ("red" if structure["decision"].startswith("suppress") else
                 "#ff8c00" if structure["review_reasons"] else "#009944")
        x1, y1, x2, y2 = row["bbox"]
        rectangle = (x1 * sx, y1 * sy + 50, x2 * sx, y2 * sy + 50)
        draw.rectangle(rectangle, outline=color, width=2)
        if structure["review_reasons"] or color == "red":
            draw.text((rectangle[0], max(50, rectangle[1] - 12)), str(row["candidate_id"])[-6:], fill=color)
    canvas.save(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--match-iou", type=float, default=.5)
    parser.add_argument("--nms-iou", type=float, default=.5)
    parser.add_argument("--policy", choices=("annotate_only", "frame_guard"), default="annotate_only",
                        help="Only changes the exported proposed_predictions; both fixed policies are always evaluated")
    parser.add_argument("--texts-json", type=Path,
                        help="Optional {image_key: [{text,bbox,text_id}]} map; OCR affects hints only")
    parser.add_argument("--visualizations", type=int, default=8)
    args = parser.parse_args(argv)
    raw = json.loads(args.raw_cache.read_text(encoding="utf-8-sig"))
    validate_raw_cache(raw)
    if not raw.get("capture_complete") or raw.get("split") != "val":
        raise ValueError("This development benchmark requires a complete val cache; test data must remain untouched")
    if not raw["generation"]["confidence"] <= args.confidence <= 1 or not 0 < args.match_iou <= 1:
        raise ValueError("Invalid confidence/IoU or missing below-capture candidates")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise ValueError("Output already has a summary; use a new directory to preserve previous results")
    data_root = (args.data_root or Path(raw["data_root"])).resolve()
    raw_sha = sha256_file(args.raw_cache)
    baseline = recipe_baseline(raw, args.confidence, args.nms_iou)
    text_map = json.loads(args.texts_json.read_text(encoding="utf-8-sig")) if args.texts_json else {}
    if not isinstance(text_map, dict):
        raise ValueError("OCR hints must be an image-keyed dictionary")
    registration = {"experiment": "fixed_two_policy_structural_diagnostics", "policy_version": POLICY_VERSION,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "settings": SETTINGS,
                    "confidence": args.confidence, "nms_iou": args.nms_iou, "match_iou": args.match_iou,
                    "policies": ["annotate_only", "frame_guard"], "raw_cache_sha256": raw_sha,
                    "evaluation_status": "val_development_not_independent_test",
                    "no_threshold_search": True, "uses_ocr_for_deletion": False,
                    "baseline_selection": "existing_whole_plus_tiles_internal_edge_filter_then_class_aware_nms",
                    "baseline_warning": "Replays cached model outputs; not an independent inference call at another score threshold."}
    dump_json(args.output / "policy_registration.json", registration)
    results, annotated, guarded = {}, {}, {}
    for index, (key, rows) in enumerate(baseline.items(), 1):
        image_path = resolve_image(raw, key, data_root)
        if sha256_file(image_path) != raw["images"][key]["sha256"]:
            raise ValueError(f"Image content changed: {key}")
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        result = annotate_image(image, rows, "frame_guard", text_map.get(key, []))
        results[key] = result
        annotated[key] = [{**row, "structure": {**row["structure"], "decision": "retain"}}
                          for row in result["annotated_candidates"]]
        guarded[key] = result["predictions"]
        print(f"[{index}/{len(baseline)}] {key}: {result['counts']}", flush=True)
    dump_json(args.output / "structure_audit.json", results)
    common = {"schema_version": 1, "coordinate_system": COORDINATES, "split": "val", "names": raw["names"],
              "confidence": args.confidence, "raw_cache_sha256": raw_sha, "weights_sha256": raw["weights"]["sha256"],
              "data_root": str(data_root), "original_geometry_retained": True}
    stages = {"baseline": baseline, "annotate_only": annotated, "frame_guard": guarded}
    for stage, predictions in stages.items():
        dump_json(args.output / f"{stage}_predictions.json", {**common, "stage": stage, "predictions": predictions})
    dump_json(args.output / "proposed_predictions.json", {**common, "stage": args.policy, "predictions": stages[args.policy],
                                                          "promotion": "shadow_only_active_model_unchanged"})
    # Ground truth is loaded after policies and predictions have been finalized.
    names, records = load_split(data_root, "val")
    if names != raw["names"] or {record["image"] for record in records} != set(baseline):
        raise ValueError("Val images/class names differ from cache")
    metrics, stage_matches = {}, {}
    for stage, predictions in stages.items():
        totals = [0, 0, 0]
        stage_matches[stage] = {}
        for record in records:
            key = record["image"]
            matching = match_detections(record["ground_truth"], predictions[key], args.match_iou, args.confidence)
            stage_matches[stage][key] = matching
            totals[0] += len(matching["matches"])
            totals[1] += len(matching["false_positive_indices"])
            totals[2] += len(matching["false_negative_indices"])
        metrics[stage] = count_summary(*totals)
    safety = {"suppressed_candidate_count": 0, "suppressed_baseline_true_matches": 0,
              "suppressed_baseline_false_positives": 0, "lost_baseline_matched_ground_truth": 0,
              "flagged_baseline_true_matches": 0, "flagged_baseline_false_positives": 0,
              "suppressed_candidates": [], "per_image": []}
    by_record = {record["image"]: record for record in records}
    for key, result in results.items():
        matching = stage_matches["baseline"][key]
        true_ids = {baseline[key][match["prediction_index"]]["candidate_id"] for match in matching["matches"]}
        suppressed_ids = {row["candidate_id"] for row in result["suppressed"]}
        flagged_ids = {row["candidate_id"] for row in result["annotated_candidates"] if row["structure"]["review_reasons"]}
        baseline_gt = {match["ground_truth_index"] for match in matching["matches"]}
        guarded_gt = {match["ground_truth_index"] for match in stage_matches["frame_guard"][key]["matches"]}
        image_safety = {"image": key, "suppressed": len(suppressed_ids),
                        "suppressed_baseline_true_matches": len(suppressed_ids & true_ids),
                        "lost_baseline_matched_ground_truth": len(baseline_gt - guarded_gt)}
        safety["per_image"].append(image_safety)
        safety["suppressed_candidate_count"] += len(suppressed_ids)
        safety["suppressed_baseline_true_matches"] += len(suppressed_ids & true_ids)
        safety["suppressed_baseline_false_positives"] += len(suppressed_ids - true_ids)
        safety["lost_baseline_matched_ground_truth"] += len(baseline_gt - guarded_gt)
        safety["flagged_baseline_true_matches"] += len(flagged_ids & true_ids)
        safety["flagged_baseline_false_positives"] += len(flagged_ids - true_ids)
        safety["suppressed_candidates"].extend({"image": key, **row, "baseline_true_match": row["candidate_id"] in true_ids}
                                               for row in result["suppressed"])
    labels_sha = {key: sha256_file(data_root / "labels" / Path(*Path(key).parts[1:]).with_suffix(".txt")) for key in baseline}
    counts = {field: sum(result["counts"][field] for result in results.values()) for field in next(iter(results.values()))["counts"]}
    summary = {**registration, "raw_cache": str(args.raw_cache.resolve()), "data_root": str(data_root),
               "weights": raw["weights"], "images": len(records), "metrics": metrics, "safety": safety,
               "diagnostic_counts": counts, "label_sha256": labels_sha,
               "metric": "fixed_threshold_class_aware_confidence_ordered_one_to_one_not_mAP",
               "api_calls": 0, "gpu_calls": 0, "selected_policy": args.policy,
               "policy_script_sha256": sha256_file(Path(__file__)),
               "policy_module_sha256": sha256_file(LAB / "src/component_lab/structure_checks.py"),
               "limitations": ["Val used for development, not independent final accuracy.",
                               "Flags are review hints and can include correct IC/connector detections.",
                               "No automatic merge/split and no recall recovery is attempted.",
                               "Original boxes and all frame suppression hypotheses are retained.",
                               "No human gold labels identify split/merge or border errors separately."]}
    dump_json(args.output / "summary.json", summary)
    overlay_dir = args.output / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    ranked = sorted(results, key=lambda key: (-results[key]["counts"]["suppressed"],
                                             -results[key]["counts"]["review_candidates"], key))
    examples = []
    for key in ranked[:max(0, args.visualizations)]:
        filename = "__".join(Path(key).with_suffix(".png").parts)
        _render(Path(by_record[key]["image_path"]), results[key], overlay_dir / filename)
        examples.append(f'<h3>{escape(key)}</h3><p>{escape(str(results[key]["counts"]))}</p>'
                        f'<a href="overlays/{escape(filename)}"><img loading="lazy" src="overlays/{escape(filename)}"></a>')
    table_rows = "".join(f"<tr><td>{escape(stage)}</td><td>{value['tp']}</td><td>{value['fp']}</td>"
                         f"<td>{value['fn']}</td><td>{value['precision']:.4%}</td><td>{value['recall']:.4%}</td></tr>"
                         for stage, value in metrics.items())
    (args.output / "gallery.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Structural diagnostics development benchmark</title>'
        '<style>body{font:16px system-ui;margin:24px;max-width:1500px}img{max-width:100%}'
        'td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>'
        '<h1>Structural diagnostics: validation-development</h1>'
        '<p>Default: annotate only. Orange is a review hint, not a wrong detection. '
        'Red is optional frame suppression; all original candidates are preserved.</p>'
        '<table><tr><th>Policy</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th></tr>'
        + table_rows + '</table><p>No API calls or GPU inference. No automatic merge/split. '
        'Same fixed confidence threshold; this is not mAP or an independent final test.</p>'
        + ''.join(examples), encoding="utf-8")
    lines = ["# 结构诊断开发集对照", "", "固定两种策略，不搜索阈值；原验证集属于开发用途，不是独立最终测试。", "",
             f"置信度={args.confidence}，匹配IoU={args.match_iou}，图数={len(records)}。", "",
             "|策略|TP|FP|FN|精确率|召回率|", "|---|---:|---:|---:|---:|---:|"]
    for stage, value in metrics.items():
        lines.append(f"|{stage}|{value['tp']}|{value['fp']}|{value['fn']}|{value['precision']:.2%}|{value['recall']:.2%}|")
    lines += ["", f"可选页框保护策略删除候选{safety['suppressed_candidate_count']}个；其中原基线真匹配"
              f"{safety['suppressed_baseline_true_matches']}个，原已匹配目标丢失{safety['lost_baseline_matched_ground_truth']}个。", "",
              "默认仅加标记，不改变结果。大框、包含关系、一个位号均不能触发删除或合并。可选过滤要求四条长边接近页边，"
              "并排除重复引脚状穿越边界。IC矩形、接口及分区框没有充分证据时保留。", "",
              "框拆分/合并仍是待解决的局部上下文问题；本阶段只提供疑难区域列表，不能宣称已经修复。", "",
              "未改默认模型、未改人工标注、未使用测试集、未调用付费API。", ""]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "metrics": metrics, "safety_suppressed": safety["suppressed_candidate_count"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
