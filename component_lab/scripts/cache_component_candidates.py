"""Capture every model-returned whole/tile detection, then replay policies on CPU.

Capture preserves internal tile-edge rows, model pass provenance and hashes.
Replay never imports torch/ultralytics. Qwen is a separate shadow stage; this
tool does not read credentials, invoke an API, promote pseudo-labels, or change
the active human review batch. Baseline predictions remain the final output
until a separately validated decision stage is explicitly applied.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time


LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from component_lab.candidate_postprocess import (COORDINATES, RAW_SCHEMA, box_geometry, conservative_policy, context_neighbors,
                                               dump_json, make_candidate, maximum_matching_coverage,
                                               sha256_file, validate_raw_cache)
from evaluate_tiled_yolo import class_aware_nms, result_rows, tile_plan


def load_images_without_labels(data_root: Path, split: str) -> tuple[list[str], list[dict]]:
    """Capture cannot use GT: discover image paths/dimensions from dataset.yaml."""
    import yaml
    from PIL import Image

    config = yaml.safe_load((data_root / "dataset.yaml").read_text(encoding="utf-8-sig"))
    indexed = config["names"]
    if isinstance(indexed, dict):
        indexed = {int(key): str(value) for key, value in indexed.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("Dataset class IDs must be contiguous from zero")
        names = [indexed[index] for index in range(len(indexed))]
    else:
        names = [str(value) for value in indexed]
    if not names:
        raise ValueError("No class names in dataset.yaml")
    configured = Path(config.get(split, f"images/{split}"))
    directory = (configured if configured.is_absolute() else data_root / configured).resolve()
    if not directory.is_relative_to(data_root) or not directory.is_dir():
        raise ValueError("Image directory must exist within data-root")
    paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in
                   {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"})
    if not paths:
        raise ValueError(f"No images for split {split}")
    records = []
    for path in paths:
        relative = path.relative_to(data_root).as_posix()
        if not relative.startswith("images/"):
            raise ValueError("Expected dataset images beneath data-root/images")
        with Image.open(path) as image:
            width, height = image.size
        records.append({"image": relative, "image_path": str(path), "width": width, "height": height,
                        "sha256": sha256_file(path)})
    return names, records


def load_plain_images(source: Path) -> tuple[Path, list[dict]]:
    """Use stable synthetic keys without requiring a YOLO dataset or labels."""
    from PIL import Image

    source = source.resolve()
    root = source.parent if source.is_file() else source
    paths = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
    paths = [path for path in paths if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}]
    if not root.is_dir() or not paths:
        raise ValueError("--images must be a supported image file or a directory containing images")
    records = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        with Image.open(path) as image:
            width, height = image.size
        records.append({"image": f"images/input/{relative}", "image_path": str(path),
                        "source_relative_path": relative, "width": width, "height": height,
                        "sha256": sha256_file(path)})
    return root, records


def resolve_image(raw: dict, key: str, data_root: Path) -> Path:
    metadata = raw["images"][key]
    relative = metadata.get("source_relative_path", key)
    target = (data_root / relative).resolve()
    if not target.is_relative_to(data_root):
        raise ValueError("Cached image escapes the selected image root")
    return target


def capture(args: argparse.Namespace) -> Path:
    """Only function that imports/uses a model; never reads label files."""
    from PIL import Image

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "raw_candidates.json"
    journal_path = args.output / "raw_capture_journal.jsonl"
    if raw_path.exists() or journal_path.exists():
        raise ValueError("Capture output already contains raw data; use replay or a new output directory")
    weights = args.weights.resolve()
    weights_sha256 = sha256_file(weights)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output / "ultralytics_settings"))
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model_names = [model.names[index] for index in range(len(model.names))]
    if args.images:
        data_root, records = load_plain_images(args.images)
        names, input_kind = model_names, "ordinary_images_without_labels"
    else:
        data_root = args.data_root.resolve()
        names, records = load_images_without_labels(data_root, args.split)
        input_kind = "yolo_dataset"
    if model_names != names:
        raise ValueError("Model names/order do not match dataset.yaml")
    generation = {"confidence": args.confidence, "imgsz": args.imgsz,
                  "tile_imgsz": args.tile_imgsz, "tile_size": args.tile_size,
                  "overlap": args.overlap, "max_tiles": args.max_tiles, "max_det_per_call": args.max_det,
                  "edge_margin": args.edge_margin, "device": args.device,
                  "raw_definition": "all model.predict returned detections per pass before our cross-pass filtering; not pre-decoder logits",
                  "below_confidence_available": False,
                  "internal_model_postprocess": "Ultralytics model.predict native postprocessing remains in effect",
                  "tile_limit_policy": "skip_entire_tile_pass_if_over_limit_never_partial_coverage"}
    payload = {"schema": RAW_SCHEMA, "schema_version": 1, "stage": "raw", "split": args.split,
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "coordinate_system": COORDINATES,
               "data_root": str(data_root), "input_kind": input_kind,
               "evaluation_available": input_kind == "yolo_dataset" and args.split in {"val", "test"},
               "dataset_yaml_sha256": sha256_file(data_root / "dataset.yaml") if input_kind == "yolo_dataset" else None,
               "names": names, "weights": {"path": str(weights), "sha256": weights_sha256},
               "generation": generation, "images": {}, "predictions": {}, "capture_complete": False,
               "software": {"python": sys.version, "ultralytics": importlib.metadata.version("ultralytics"),
                            "capture_script_sha256": sha256_file(Path(__file__)),
                            "policy_module_sha256": sha256_file(LAB / "src/component_lab/candidate_postprocess.py")}}
    with journal_path.open("x", encoding="utf-8") as journal:
        journal.write(json.dumps({"event": "capture_start", "metadata": payload}, ensure_ascii=False, allow_nan=False) + "\n")
        journal.flush()
        for ordinal, record in enumerate(records, 1):
            key, width, height = record["image"], record["width"], record["height"]
            plan = tile_plan(width, height, args.tile_size, args.overlap, args.max_tiles)
            with Image.open(record["image_path"]) as source:
                image = source.convert("RGB")
            passes = [("whole_image", "whole_image", [0, 0, width, height], args.imgsz)]
            passes += [(f"tile_{index:04d}", "tile", window, args.tile_imgsz)
                       for index, window in enumerate(plan["windows"])]
            rows, pass_metadata = [], []
            for pass_id, source_pass, window, input_size in passes:
                crop = image if source_pass == "whole_image" else image.crop(tuple(window))
                timer = time.perf_counter()
                result = model.predict(crop, imgsz=input_size, device=args.device, conf=args.confidence,
                                       max_det=args.max_det, verbose=False)[0]
                predictions = result_rows(result, names, source_pass)
                pass_metadata.append({"pass_id": pass_id, "source_pass": source_pass, "window": window,
                                      "model_input_size": input_size, "returned_detection_count": len(predictions),
                                      "max_det_reached": len(predictions) >= args.max_det,
                                      "inference_seconds": time.perf_counter() - timer})
                rows.extend(make_candidate(key, pass_id, index, prediction, source_pass, window, width, height,
                                           input_size, args.edge_margin) for index, prediction in enumerate(predictions))
            metadata = {**record, "tile_plan": plan, "passes": pass_metadata,
                        "candidate_count": len(rows), "valid_candidate_count": sum(row["valid"] for row in rows),
                        "internal_edge_candidate_count": sum(bool(row["edge"] and row["edge"]["touches_internal"]) for row in rows),
                        "model_call_saturation_count": sum(entry["max_det_reached"] for entry in pass_metadata)}
            payload["images"][key], payload["predictions"][key] = metadata, rows
            journal.write(json.dumps({"event": "image_complete", "image": key, "metadata": metadata,
                                      "candidates": rows}, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            print(f"[{ordinal}/{len(records)}] {key}: raw {len(rows)}, internal-edge {metadata['internal_edge_candidate_count']}", flush=True)
    payload["capture_complete"] = True
    validate_raw_cache(payload)
    dump_json(raw_path, payload)
    return raw_path


def recipe_baseline(raw: dict, confidence: float, nms_iou: float) -> dict:
    """Reproduce existing edge-filter + confidence-ordered NMS as a control."""
    output = {}
    for key, rows in raw["predictions"].items():
        valid = [row for row in rows if row["valid"] and row["score"] >= confidence]
        whole = [row for row in valid if row["source_pass"] == "whole_image"]
        eligible = whole + [row for row in valid if row["source_pass"] == "tile" and not row["edge"]["touches_internal"]]
        # Existing implementation skips the extra NMS pass on tile-cap fallback.
        output[key] = list(whole) if raw["images"][key]["tile_plan"]["skipped"] else class_aware_nms(eligible, nms_iou)
    return output


def _write_predictions(path: Path, raw: dict, predictions: dict, stage: str,
                       confidence: float, raw_sha256: str, extra: dict | None = None) -> None:
    dump_json(path, {"schema_version": 1, "stage": stage, "coordinate_system": COORDINATES,
                     "split": raw["split"], "names": raw["names"], "confidence": confidence,
                     "data_root": raw["data_root"], "weights": raw["weights"]["path"],
                     "weights_sha256": raw["weights"]["sha256"], "raw_cache_sha256": raw_sha256,
                     "imgsz": raw["generation"]["imgsz"], "predictions": predictions, **(extra or {})})


def evaluate_stages(args: argparse.Namespace, raw: dict, data_root: Path,
                    stages: dict[str, tuple[Path, dict, float]]) -> dict:
    from analyze_yolo_errors import load_split, main as diagnose

    names, records = load_split(data_root, raw["split"])
    if names != raw["names"] or {record["image"] for record in records} != set(raw["images"]):
        raise ValueError("Evaluation split/classes differ from the captured image set")
    coverage, metrics = {}, {}
    for stage, (path, predictions, confidence) in stages.items():
        output = args.output / "evaluation" / stage
        diagnose(["--weights", raw["weights"]["path"], "--data-root", str(data_root),
                  "--output", str(output), "--split", raw["split"], "--device", "cpu",
                  "--imgsz", str(raw["generation"]["imgsz"]), "--confidence", str(confidence),
                  "--iou", str(args.match_iou), "--max-det", str(raw["generation"]["max_det_per_call"]),
                  "--visualizations", str(args.visualizations), "--predictions-json", str(path)])
        metrics[stage] = json.loads((output / "metrics.json").read_text(encoding="utf-8"))["overall"]
        per_image = {record["image"]: maximum_matching_coverage(record["ground_truth"],
                                                               predictions[record["image"]], args.match_iou)
                     for record in records}
        matched = sum(entry["matched_maximum"] for entry in per_image.values())
        gt_count = sum(entry["ground_truth"] for entry in per_image.values())
        coverage[stage] = {"matched_maximum": matched, "ground_truth": gt_count,
                           "candidate_count": sum(entry["candidate_count"] for entry in per_image.values()),
                           "recall_upper_bound": matched / gt_count if gt_count else None,
                           "per_image": per_image}
    labels = {record["image"]: sha256_file(data_root / "labels" / Path(*Path(record["image"]).parts[1:]).with_suffix(".txt"))
              for record in records}
    report = {"iou_threshold": args.match_iou,
              "fixed_metric": "existing_analyze_yolo_errors_confidence_ordered_class_aware_one_to_one_counts_not_mAP",
              "all_undecided_policy_rows_included_in_prediction_denominator": True,
              "maximum_coverage_metric": "ground_truth_assisted_diagnostic_upper_bound_never_used_to_select_predictions",
              "metrics": metrics, "maximum_one_to_one_coverage": coverage, "label_sha256": labels}
    dump_json(args.output / "evaluation_summary.json", report)
    return report


def replay(args: argparse.Namespace, raw_path: Path) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8-sig"))
    validate_raw_cache(raw)
    if not raw.get("capture_complete"):
        raise ValueError("Incomplete capture cannot be evaluated/promoted")
    args.output.mkdir(parents=True, exist_ok=True)
    capture_confidence = raw["generation"]["confidence"]
    if args.baseline_confidence < capture_confidence:
        raise ValueError("Baseline confidence is below capture confidence; missing lower-score boxes cannot be recovered")
    data_root = (args.image_root or args.data_root or Path(raw["data_root"])).resolve()
    for key, metadata in raw["images"].items():
        path = resolve_image(raw, key, data_root)
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Captured image changed/missing; refusing cache reuse: {key}")
    raw_sha256 = sha256_file(raw_path)
    valid_predictions = {key: [row for row in rows if row["valid"]] for key, rows in raw["predictions"].items()}
    policies = {key: conservative_policy(rows, args.near_duplicate_iou, args.conflict_iou, args.containment)
                for key, rows in raw["predictions"].items()}
    policy_predictions = {key: policy["predictions"] for key, policy in policies.items()}
    if args.baseline_cache:
        # The evaluator's loader checks split, names, coordinates, and capture threshold.
        from analyze_yolo_errors import load_predictions

        records = [{"image": key, "split": raw["split"]} for key in raw["images"]]
        provided = json.loads(args.baseline_cache.read_text(encoding="utf-8-sig"))
        if not isinstance(provided, dict):
            raise ValueError("Stable baseline cache must be a JSON object")
        baseline_confidence = float(provided.get("confidence", args.baseline_confidence))
        if not math.isfinite(baseline_confidence) or not 0 <= baseline_confidence <= 1:
            raise ValueError("Invalid stable baseline confidence")
        # Validate the supplied cache, but keep its original complete rows. No
        # experimental threshold or geometry policy can edit the stable control.
        load_predictions(args.baseline_cache, records, raw["names"], baseline_confidence)
        baseline = provided.get("predictions", provided)
        if set(baseline) != set(raw["images"]):
            raise ValueError("Stable baseline must contain exactly the captured image set")
        if any(row["score"] < baseline_confidence for rows in baseline.values() for row in rows):
            raise ValueError("Supplied stable baseline contains rows below its declared confidence; resolve the ambiguous cache first")
        baseline_provenance = {"kind": "provided_stable_baseline_cache",
                               "path": str(args.baseline_cache.resolve()), "sha256": sha256_file(args.baseline_cache),
                               "warning": "Baseline weights/image provenance is exactly as supplied; raw cache hashes are not its proof."}
    else:
        baseline_confidence = args.baseline_confidence
        baseline = recipe_baseline(raw, args.baseline_confidence, args.baseline_nms_iou)
        baseline_provenance = {"kind": "existing_algorithm_replayed_from_raw_same_model_outputs",
                               "confidence": args.baseline_confidence, "nms_iou": args.baseline_nms_iou,
                               "internal_edge_filter": True,
                               "warning": "This is a reproduced algorithm control, not a claim of bit-identical independent detector calls at another threshold."}
    paths = {stage: args.output / filename for stage, filename in
             {"raw": "raw_predictions.json", "baseline": "baseline_predictions.json",
              "policy_pre_qwen": "policy_pre_qwen_predictions.json", "final": "final_predictions.json"}.items()}
    _write_predictions(paths["raw"], raw, valid_predictions, "raw_valid_rows_all_edge_rows_included", capture_confidence, raw_sha256)
    _write_predictions(paths["baseline"], raw, baseline, "stable_baseline", baseline_confidence, raw_sha256,
                       {"baseline_provenance": baseline_provenance})
    _write_predictions(paths["policy_pre_qwen"], raw, policy_predictions, "policy_pre_qwen", capture_confidence, raw_sha256,
                       {"undecided_handling": "retained_and_counted_as_predictions"})
    _write_predictions(paths["final"], raw, baseline, "final_baseline_hold", baseline_confidence, raw_sha256,
                       {"baseline_provenance": baseline_provenance,
                        "final_status": "baseline_hold_no_qwen_applied", "qwen_decisions_applied": False})
    audit = {"schema_version": 1, "raw_cache": str(raw_path.resolve()), "raw_cache_sha256": raw_sha256,
             "baseline_provenance": baseline_provenance,
             "policy": {"near_duplicate_iou": args.near_duplicate_iou, "conflict_iou": args.conflict_iou,
                        "containment": args.containment, "geometry_uses_ground_truth": False,
                        "review_groups_merge_objects": False}, "images": policies}
    dump_json(args.output / "policy_audit.json", audit)
    shadow_rows, support_by_id = {}, {}
    for key, rows in policy_predictions.items():
        for row in rows:
            support = sorted({other["pass_id"] for other in valid_predictions[key]
                              if other["class_id"] == row["class_id"] and
                              box_geometry(row["bbox"], other["bbox"])["iou"] >= .8})
            support_by_id[row["candidate_id"]] = support
        uncertain = [row for row in rows if row["review_reasons"] or row["score"] < args.shadow_confidence or
                     len(support_by_id[row["candidate_id"]]) < 2]
        controls = sorted((row for row in rows if row not in uncertain), key=lambda row: row["candidate_id"])[:args.shadow_controls_per_image]
        shadow_rows[key] = [{**row, "shadow_selection_reason": "uncertain_geometry_low_score_or_single_pass" if row in uncertain else "deterministic_control_without_gt"}
                            for row in uncertain + controls]
    dump_json(args.output / "qwen_shadow_input.json",
              {"schema": "component_candidate_qwen_shadow_input_v1", "stage": "shadow_qwen_request_plan",
               "api_calls_performed": 0, "raw_cache_sha256": raw_sha256, "weights_sha256": raw["weights"]["sha256"],
               "data_root": str(data_root), "coordinate_system": COORDINATES, "names": raw["names"],
               "images": {key: {"sha256": row["sha256"], "width": row["width"], "height": row["height"]}
                          for key, row in raw["images"].items()},
               "predictions": shadow_rows,
               "policy": "shadow only; return keep/reject/uncertain with contextual completeness/duplicate evidence; do not silently promote decisions"})
    with (args.output / "qwen_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for key, rows in shadow_rows.items():
            all_rows = policy_predictions[key]
            for row in rows:
                neighbors, neighbors_truncated_count = context_neighbors(row, all_rows, limit=40)
                reasons = list(row["review_reasons"])
                if row["score"] < args.shadow_confidence:
                    reasons.append("low_score")
                if len(support_by_id[row["candidate_id"]]) < 2:
                    reasons.append("single_pass_support")
                # This is a request queue, not an implicit API budget. A caller
                # must preserve all unprocessed IDs if it caps paid requests.
                entry = {"id": row["candidate_id"], "image_path": str(resolve_image(raw, key, data_root)),
                         "image_sha256": raw["images"][key]["sha256"], "bbox": row["bbox"], "score": row["score"],
                         "source": row["source_pass"], "neighbors": neighbors, "ocr": [],
                         "neighbors_truncated_count": neighbors_truncated_count,
                         "neighbors_selection_policy": "containment_then_overlap_then_center_distance_limit40",
                         "route_reasons": reasons or [row["shadow_selection_reason"]],
                         "support_pass_ids": support_by_id[row["candidate_id"]],
                         "status": "pending_shadow_only", "raw_cache_sha256": raw_sha256}
                handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    stages = {"raw": (paths["raw"], valid_predictions, capture_confidence),
              "baseline": (paths["baseline"], baseline, baseline_confidence),
              "policy_pre_qwen": (paths["policy_pre_qwen"], policy_predictions, capture_confidence),
              "final": (paths["final"], baseline, baseline_confidence)}
    evaluation = None
    if not args.skip_evaluation and raw.get("evaluation_available", raw["split"] in {"val", "test"}):
        evaluation = evaluate_stages(args, raw, data_root, stages)
    totals = {name: sum(policy["counts"][name] for policy in policies.values())
              for name in ("raw", "valid_raw", "invalid", "pre_qwen", "suppressed_duplicates", "pending")}
    summary = {"schema_version": 1, "raw_cache": str(raw_path.resolve()), "raw_cache_sha256": raw_sha256,
               "image_count": len(raw["images"]), "split": raw["split"], "counts": totals,
               "shadow_candidate_count": sum(len(rows) for rows in shadow_rows.values()), "api_calls_performed": 0,
               "stages": {"raw": "all returned rows kept; invalid rows audited, valid rows evaluated",
                          "policy_pre_qwen": "conservative geometry only; undecided rows remain in denominator",
                          "shadow_qwen": "request plan written; no calls or applied decisions",
                          "final": "exact baseline hold; experimental policy does not replace baseline"},
               "baseline_provenance": baseline_provenance,
               "metrics": evaluation["metrics"] if evaluation else None,
               "evaluation_status": "evaluated" if evaluation else "unavailable_or_explicitly_skipped_no_PR_claim",
               "maximum_one_to_one_coverage": {stage: {key: value for key, value in entry.items() if key != "per_image"}
                                                for stage, entry in evaluation["maximum_one_to_one_coverage"].items()} if evaluation else None}
    dump_json(args.output / "pipeline_summary.json", summary)
    print(json.dumps({"counts": totals, "shadow_candidate_count": summary["shadow_candidate_count"],
                      "final_status": "baseline_hold_no_qwen_applied", "output": str(args.output)}, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    for command in ("capture", "replay"):
        mode = modes.add_parser(command)
        mode.add_argument("--output", type=Path, required=True)
        if command == "capture":
            source = mode.add_mutually_exclusive_group(required=True)
            source.add_argument("--data-root", type=Path)
            source.add_argument("--images", type=Path, help="Ordinary image file/directory; no labels or dataset.yaml needed")
        else:
            mode.add_argument("--data-root", type=Path)
        mode.add_argument("--image-root", type=Path, help="Override the original image root on CPU replay")
        mode.add_argument("--baseline-cache", type=Path, help="Optional stable baseline predictions JSON, retained as final")
        mode.add_argument("--baseline-confidence", type=float, default=.1)
        mode.add_argument("--baseline-nms-iou", type=float, default=.5)
        mode.add_argument("--near-duplicate-iou", type=float, default=.95)
        mode.add_argument("--conflict-iou", type=float, default=.15)
        mode.add_argument("--containment", type=float, default=.85)
        mode.add_argument("--shadow-confidence", type=float, default=.6)
        mode.add_argument("--shadow-controls-per-image", type=int, default=2)
        mode.add_argument("--match-iou", type=float, default=.5)
        mode.add_argument("--visualizations", type=int, default=0)
        mode.add_argument("--skip-evaluation", action="store_true")
        if command == "capture":
            mode.add_argument("--weights", type=Path, required=True)
            mode.add_argument("--split", choices=("train", "val", "test"), default="val")
            mode.add_argument("--device", default="cpu")
            mode.add_argument("--confidence", "--conf", type=float, default=.03)
            mode.add_argument("--imgsz", type=int, default=1280)
            mode.add_argument("--tile-imgsz", type=int)
            mode.add_argument("--tile-size", type=int, default=1024)
            mode.add_argument("--overlap", type=float, default=.25)
            mode.add_argument("--max-tiles", type=int, default=100)
            mode.add_argument("--max-det", type=int, default=1000)
            mode.add_argument("--edge-margin", type=float, default=2)
        else:
            mode.add_argument("--raw-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if (not 0 <= args.baseline_confidence <= 1 or not 0 < args.baseline_nms_iou <= 1 or
            not 0 < args.match_iou <= 1 or not 0 <= args.shadow_confidence <= 1 or
            not 0 < args.conflict_iou < args.near_duplicate_iou <= 1 or not 0 < args.containment <= 1 or
            args.visualizations < 0 or args.shadow_controls_per_image < 0):
        parser.error("Invalid confidence/IoU thresholds or nonnegative counts")
    if args.command == "capture":
        args.tile_imgsz = args.tile_imgsz or args.imgsz
        if (not 0 <= args.confidence <= 1 or not 0 <= args.overlap < 1 or args.edge_margin < 0 or
                min(args.imgsz, args.tile_imgsz, args.tile_size, args.max_tiles, args.max_det) <= 0):
            parser.error("Invalid capture confidence, input sizes or tile settings")
        if args.baseline_confidence < args.confidence:
            parser.error("Baseline confidence cannot be lower than capture confidence")
        raw_path = capture(args)
    else:
        raw_path = args.raw_cache.resolve()
    replay(args, raw_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
