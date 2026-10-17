"""Bounded dev-only pin-location diagnostic with GT component boxes, not GT pins.

This does not infer pin numbers/names and does not report official PinF1.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

LAB_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = LAB_ROOT.parent
sys.path.insert(0, str(WORKSPACE / "non404" / "show_up"))
from contest6.pin_tip_crop_scan import scan_boundary_candidates


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def infer_candidates(image_bgr, boxes_top_left, tip_offset=3):
    """Only pixels, artificial owner IDs and boxes cross this inference boundary."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height = image_bgr.shape[0]
    candidates = []
    for component in boxes_top_left:
        points = scan_boundary_candidates(gray, component["bbox"], tip_offset=tip_offset,
                                          ray_depth=18, dark_threshold=210, min_trace_ink=6)
        for x, y in points:
            candidates.append({"candidate_id": f"candidate_{len(candidates):05d}",
                               "owner_id": component["id"],
                               "point": {"x": float(x), "y": float(height - y)}})
    return candidates


def match_locations(predicted, truth, tolerance=5.0):
    """Maximum cardinality one-to-one matching, then minimum distance.

    Dummy penalty exceeds the sum of all possible within-tolerance distances,
    so a nearest-pair greedy failure cannot discard an otherwise matchable pin.
    """
    if not predicted or not truth:
        return []
    distances = np.linalg.norm(np.asarray(predicted)[:, None, :] - np.asarray(truth)[None, :, :], axis=2)
    penalty = (min(len(predicted), len(truth)) + 1) * (tolerance + 1)
    cost = np.full((len(predicted), len(truth) + len(predicted)), penalty)
    cost[:, :len(truth)] = np.where(distances <= tolerance, distances, penalty * 2)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(distances[i, j])) for i, j in zip(rows, cols)
            if j < len(truth) and distances[i, j] <= tolerance]


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def boundary_offset(point, box):
    """Signed GT distance to bbox: positive outside, negative strictly inside."""
    x, y = point
    x1, y1, x2, y2 = box
    dx, dy = max(x1 - x, 0, x - x2), max(y1 - y, 0, y - y2)
    if dx or dy:
        return math.hypot(dx, dy)
    return -min(x - x1, x2 - x, y - y1, y2 - y)


def evaluate(candidates, gt_components, gt_pins, owners, tolerance):
    by_owner = defaultdict(list)
    for row in candidates:
        by_owner[row["owner_id"]].append(row)
    counts = {"tp": 0, "fp": 0, "fn": 0}
    details, offsets = [], []
    all_match_distances = []
    for owner_id, key in owners.items():
        predicted = by_owner[owner_id]
        truth = list(gt_pins.get(key, {}).items())
        pxy = [(r["point"]["x"], r["point"]["y"]) for r in predicted]
        txy = [(r["point"]["x"], r["point"]["y"]) for _, r in truth]
        matches = match_locations(pxy, txy, tolerance)
        matched_p, matched_t = {x[0] for x in matches}, {x[1] for x in matches}
        tp = len(matches)
        row_counts = prf(tp, len(predicted) - tp, len(truth) - tp)
        for name in counts:
            counts[name] += row_counts[name]
        all_match_distances.extend(m[2] for m in matches)
        for i, point in enumerate(txy):
            offsets.append({"component_key": key, "pin_key": truth[i][0],
                            "signed_bbox_offset_px": boundary_offset(point, gt_components[key]["bbox"]),
                            "matched": i in matched_t,
                            "nearest_candidate_px": min((math.dist(point, p) for p in pxy), default=None)})
        details.append({"owner_id": owner_id, "component_key_evaluation_only": key,
                        "component_type_evaluation_only": gt_components[key]["type"],
                        "counts": row_counts,
                        "matches": [{"candidate_id": predicted[i]["candidate_id"],
                                     "gt_pin_key_evaluation_only": truth[j][0], "distance_px": d}
                                    for i, j, d in matches],
                        "unmatched_candidate_ids": [p["candidate_id"] for i, p in enumerate(predicted) if i not in matched_p],
                        "unmatched_gt_pin_keys": [p[0] for i, p in enumerate(truth) if i not in matched_t]})
    return {"counts": prf(**counts), "per_component": details, "gt_bbox_offsets": offsets,
            "matched_distance_median_px": float(np.median(all_match_distances)) if all_match_distances else None}


def overlay(image, boxes, candidates, result, gt_pins, owners):
    out = image.copy()
    height = out.shape[0]
    hits = {m["candidate_id"] for r in result["per_component"] for m in r["matches"]}
    matched_gt = {(r["component_key_evaluation_only"], m["gt_pin_key_evaluation_only"])
                  for r in result["per_component"] for m in r["matches"]}
    for box in boxes:
        x1, y1, x2, y2 = map(round, box["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (170, 170, 170), 1)
    for key in owners.values():
        for pin_key, row in gt_pins.get(key, {}).items():
            p = (round(row["point"]["x"]), round(height - row["point"]["y"]))
            color = (0, 150, 0) if (key, pin_key) in matched_gt else (255, 100, 0)
            cv2.circle(out, p, 5, color, 1)
    for row in candidates:
        p = (round(row["point"]["x"]), round(height - row["point"]["y"]))
        color = (0, 0, 220) if row["candidate_id"] not in hits else (0, 150, 0)
        cv2.drawMarker(out, p, color, cv2.MARKER_CROSS, 6, 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--tip-offset", type=int, default=3)
    p.add_argument("--output", type=Path, default=LAB_ROOT / "outputs/pin_geometry_oracle_20261017")
    args = p.parse_args()
    if not 1 <= args.limit <= 5:
        p.error("This bounded pilot permits 1..5 dev cases only")
    if args.tip_offset != 3:
        p.error("Pilot locks the legacy 3px offset; do not tune using dev GT")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    split_path = LAB_ROOT / "configs/contest6_group_split_v1.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    selected = split["partitions"]["dev"][:args.limit]
    rows = []
    for case in selected:
        case_id = case["case"]
        gt_path = LAB_ROOT / "data/official_contest6/train200" / (case["stem"] + ".json")
        image_path = LAB_ROOT / "outputs/component_identity_round_001_20261017/inputs" / case_id / "source.png"
        image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Missing image {image_path}")
        height = image.shape[0]
        # Only the component subsection is retained before inference. Pin/net
        # sections are read afterwards exclusively for evaluation.
        components = json.loads(gt_path.read_text(encoding="utf-8"))["components"]
        boxes, owners = [], {}
        for i, (key, row) in enumerate(components.items()):
            x1, y1, x2, y2 = row["bbox"]
            owner_id = f"owner_{i:04d}"
            boxes.append({"id": owner_id, "bbox": [x1, height - y2, x2, height - y1]})
            owners[owner_id] = key
        start = time.perf_counter()
        candidates = infer_candidates(image, boxes, args.tip_offset)
        inference_seconds = time.perf_counter() - start
        folder = args.output / case_id
        dump(folder / "candidates.json", {"coordinate_system": "bottom-left-original-pixels",
             "oracle_inputs": ["component_bboxes"], "gt_pins_used": False, "gt_nets_used": False,
             "component_types_or_text_used": False, "paid_api_calls": 0,
             "boxes_top_left": boxes, "candidates": candidates})
        gt_pins = json.loads(gt_path.read_text(encoding="utf-8"))["pins"]
        result = evaluate(candidates, components, gt_pins, owners, 5.0)
        dump(folder / "evaluation.json", result)
        if len(rows) < 3:
            rendered = overlay(image, boxes, candidates, result, gt_pins, owners)
            ok, encoded = cv2.imencode(".png", rendered)
            if not ok:
                raise ValueError("Could not encode overlay")
            (folder / "overlay.png").write_bytes(encoded.tobytes())
        offsets = [r["signed_bbox_offset_px"] for r in result["gt_bbox_offsets"]]
        row = {"case": case_id, "stem": case["stem"], **result["counts"],
               "component_count": len(boxes), "candidate_count": len(candidates),
               "inference_seconds": inference_seconds,
               "gt_offset_median_px": float(np.median(offsets)) if offsets else None,
               "gt_outside_bbox_by_more_than_8px": sum(x > 8 for x in offsets),
               "gt_inside_bbox_by_more_than_2px": sum(x < -2 for x in offsets),
               "image_sha256": digest(image_path), "gt_sha256": digest(gt_path)}
        rows.append(row)
        print(f"{case_id}: {row['tp']} matched / {row['candidate_count']} candidates; recall={row['recall']:.3f}", flush=True)
    totals = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn")}
    report = {"kind": "GT_component_bbox_input_GEOMETRIC_pin_location_diagnostic_NOT_official_PinF1",
              "partition": "dev", "case_selection": "first_up_to_five_in_frozen_manifest_order",
              "case_count": len(rows), "tip_offset_px": 3, "tolerance_px": 5.0,
              "matching": "same_oracle_component_owner_max_cardinality_then_min_distance",
              "number_or_name_inference": False, "paid_api_calls": 0,
              "micro": prf(**totals), "macro": {k: float(np.mean([r[k] for r in rows])) for k in ("precision", "recall", "f1")},
              "gt_outside_bbox_by_more_than_8px": sum(r["gt_outside_bbox_by_more_than_8px"] for r in rows),
              "gt_inside_bbox_by_more_than_2px": sum(r["gt_inside_bbox_by_more_than_2px"] for r in rows),
              "per_case": rows, "split_sha256": digest(split_path), "script_sha256": digest(Path(__file__)),
              "geometry_source_sha256": digest(WORKSPACE / "non404/show_up/contest6/pin_tip_crop_scan.py"),
              "limitations": ["Development cases with historical model-selection exposure; no independent test claim.",
                              "GT component boxes are oracle inputs, not detector predictions.",
                              "No pin number/name recognition; these are candidate geometry metrics, not official PinF1.",
                              "Fixed 3px outward offset and boundary ray assumptions can disagree with real pin length or loose boxes.",
                              "Does not infer connectivity, net labels or crossings; no model training."]}
    dump(args.output / "metrics.json", report)
    lines = ["# GT component bbox -> pin geometry pilot", "", "This is a dev-only location diagnostic, **not official PinF1**.", "",
             "Inference receives only image pixels, artificial owner IDs and GT component boxes; GT pins enter afterwards for evaluation. No types, text, GT pins/nets or paid API are used for inference.", "",
             "```powershell", "python scripts/probe_pin_geometry_oracle.py --limit 5", "```", "",
             "Matching: <=5px Euclidean, one-to-one maximum-cardinality within the same oracle component owner; then minimum distance. Fixed legacy offset=3px, not tuned on dev.", "",
             "| Case | Candidates | GT pins | TP | Precision | Recall | F1 | GT bbox offset median |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['candidate_count']} | {r['tp'] + r['fn']} | {r['tp']} | {r['precision']:.2%} | {r['recall']:.2%} | {r['f1']:.2%} | {r['gt_offset_median_px']} |")
    lines += ["", f"Micro: {json.dumps(report['micro'])}", f"Case-macro: {json.dumps(report['macro'])}", "",
              "Overlay legend: gray bbox; green matched GT circle/candidate cross; blue unmatched GT circle; red unmatched candidate cross. GT appears in overlays for evaluation only.", "",
              "Signed bbox offset is positive outside and negative inside. Counts beyond 8px outside or 2px inside flag incompatibility with the fixed 3px tip convention; they do not establish an optimized alternative. See per-component and per-pin evaluation JSON for nearest-candidate errors.", "",
              f"Observed GT pins outside bbox by >8px: {report['gt_outside_bbox_by_more_than_8px']}; inside bbox by >2px: {report['gt_inside_bbox_by_more_than_2px']}. This is evidence that physical pin-stub length and the annotation endpoint convention must be estimated; a perfect component box does not make a fixed 3px offset accurate.", "",
              "Tests: `python -m unittest tests.test_pin_geometry_oracle` (3 tests: maximum-cardinality vs greedy, inclusive 5px threshold, signed bbox distance).", "", "## Limitations", ""]
    lines.extend("- " + x for x in report["limitations"])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
