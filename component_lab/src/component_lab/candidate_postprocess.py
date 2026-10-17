"""Traceable, conservative postprocessing for cached detector candidates.

This module is deliberately independent of torch, model code and ground truth.
Rows are model-returned detections, not network logits. Every suppression has
an audit record; overlapping but geometrically different objects remain in the
pre-Qwen prediction set. Review groups are hints, never object unions.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any


RAW_SCHEMA = "component_candidates_raw_v1"
COORDINATES = "xyxy_pixels_top_left_in_original_image"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_box(box: Any) -> bool:
    return (isinstance(box, (list, tuple)) and len(box) == 4 and
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box) and
            box[2] > box[0] and box[3] > box[1])


def _json_safe(value: Any) -> Any:
    """Preserve invalid detector output for audit without nonstandard NaN JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    return value


def edge_flags(local_bbox: list[float], window: list[int], width: int, height: int,
               margin: float = 2) -> dict:
    if margin < 0 or not math.isfinite(margin):
        raise ValueError("edge margin must be finite and nonnegative")
    if not _valid_box(local_bbox) or len(window) != 4:
        raise ValueError("Invalid local box or crop window")
    left, top, right, bottom = window
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Crop window must lie inside the original image")
    x1, y1, x2, y2 = local_bbox
    touching = {"left": x1 <= margin, "top": y1 <= margin,
                "right": x2 >= right - left - margin, "bottom": y2 >= bottom - top - margin}
    actual_outer = {"left": left == 0, "top": top == 0, "right": right == width, "bottom": bottom == height}
    internal = [side for side in touching if touching[side] and not actual_outer[side]]
    outer = [side for side in touching if touching[side] and actual_outer[side]]
    return {"internal_sides": internal, "outer_image_sides": outer,
            "touches_internal": bool(internal), "touches_outer_image": bool(outer),
            "completeness": "possibly_truncated" if internal else "not_indicated",
            "margin_original_crop_pixels": margin,
            "note": "An edge flag indicates geometry only; absence does not prove a complete component."}


def make_candidate(image_key: str, pass_id: str, ordinal: int, prediction: dict,
                   source_pass: str, window: list[int], width: int, height: int,
                   input_size: int, margin: float = 2) -> dict:
    """Map crop-native output to the image, retaining invalid outputs as rows."""
    if source_pass not in {"whole_image", "tile"}:
        raise ValueError("Unknown source pass")
    if len(window) != 4 or not (0 <= window[0] < window[2] <= width and 0 <= window[1] < window[3] <= height):
        raise ValueError("Invalid crop window")
    if ordinal < 0 or input_size <= 0:
        raise ValueError("Ordinal must be nonnegative and input size positive")
    key = f"{image_key}|{pass_id}|{ordinal}"
    row = {"candidate_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:24],
           "image": image_key, "source_pass": source_pass, "pass_id": pass_id,
           "ordinal_in_pass": ordinal, "window": list(window), "model_input_size": input_size,
           "class_id": prediction.get("class_id"), "type": prediction.get("type"),
           "score": _json_safe(prediction.get("score")), "local_bbox": _json_safe(prediction.get("bbox")),
           "bbox": None, "valid": False, "invalid_reasons": [], "edge": None,
           "coordinate_mapping": {"from": "crop_original_pixels_not_model_input_pixels",
                                  "to": COORDINATES, "scale": [1.0, 1.0],
                                  "offset": [window[0], window[1]], "clipped_to_image": False}}
    box, score, class_id = prediction.get("bbox"), prediction.get("score"), prediction.get("class_id")
    if not _valid_box(box):
        row["invalid_reasons"].append("invalid_local_bbox")
    if (not isinstance(score, (int, float)) or isinstance(score, bool) or
            not math.isfinite(score) or not 0 <= score <= 1):
        row["invalid_reasons"].append("invalid_confidence")
    if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id < 0:
        row["invalid_reasons"].append("invalid_class_id")
    if row["invalid_reasons"]:
        return row
    translated = [float(box[0] + window[0]), float(box[1] + window[1]),
                  float(box[2] + window[0]), float(box[3] + window[1])]
    clipped = [max(0., min(float(width), translated[0])), max(0., min(float(height), translated[1])),
               max(0., min(float(width), translated[2])), max(0., min(float(height), translated[3]))]
    if not _valid_box(clipped):
        row["invalid_reasons"].append("empty_bbox_after_mapping")
        return row
    row.update({"valid": True, "bbox": clipped, "edge": edge_flags(list(box), window, width, height, margin)})
    row["coordinate_mapping"]["clipped_to_image"] = translated != clipped
    return row


def validate_raw_cache(payload: dict) -> None:
    """Reject ambiguous or malformed caches before policy/evaluation reuse."""
    if payload.get("schema") != RAW_SCHEMA or payload.get("coordinate_system") != COORDINATES:
        raise ValueError("Unsupported raw candidate cache schema or coordinates")
    names, images, predictions = payload.get("names"), payload.get("images"), payload.get("predictions")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError("Raw cache requires a nonempty ordered class-name list")
    if not isinstance(images, dict) or not isinstance(predictions, dict) or set(images) != set(predictions):
        raise ValueError("Raw image metadata and prediction image keys must match exactly")
    if not images:
        raise ValueError("Raw cache contains no images")
    confidence = payload.get("generation", {}).get("confidence")
    if (not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("Raw cache generation confidence must be finite and in [0,1]")
    weights_sha = payload.get("weights", {}).get("sha256", "")
    if len(weights_sha) != 64 or any(char not in "0123456789abcdef" for char in weights_sha):
        raise ValueError("Raw cache requires a weights SHA256 digest")
    all_ids = set()
    for key, metadata in images.items():
        path = Path(key)
        if path.is_absolute() or ".." in path.parts or not key.startswith("images/"):
            raise ValueError(f"Unsafe or non-dataset-relative image key: {key}")
        width, height = metadata.get("width"), metadata.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
            raise ValueError(f"Invalid image dimensions: {key}")
        image_sha = metadata.get("sha256", "")
        if len(image_sha) != 64 or any(char not in "0123456789abcdef" for char in image_sha):
            raise ValueError(f"Image SHA256 missing or invalid: {key}")
        rows = predictions[key]
        if not isinstance(rows, list):
            raise ValueError(f"Candidate rows must be a list: {key}")
        for row in rows:
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id or candidate_id in all_ids:
                raise ValueError("Candidate IDs must be nonempty and globally unique")
            all_ids.add(candidate_id)
            if row.get("image") != key or row.get("source_pass") not in {"whole_image", "tile"}:
                raise ValueError(f"Candidate source image/pass mismatch: {candidate_id}")
            window = row.get("window")
            if not _valid_box(window) or not (0 <= window[0] < window[2] <= width and 0 <= window[1] < window[3] <= height):
                raise ValueError(f"Invalid candidate crop window: {candidate_id}")
            if row.get("valid") is not True:
                if row.get("valid") is not False or not row.get("invalid_reasons"):
                    raise ValueError("Invalid candidates must state invalid_reasons")
                continue
            class_id, score, box = row.get("class_id"), row.get("score"), row.get("bbox")
            if not isinstance(class_id, int) or isinstance(class_id, bool) or not 0 <= class_id < len(names):
                raise ValueError(f"Candidate class out of range: {candidate_id}")
            if (not isinstance(score, (int, float)) or isinstance(score, bool) or
                    not math.isfinite(score) or not confidence <= score <= 1):
                raise ValueError(f"Invalid score or score below capture threshold: {candidate_id}")
            if not _valid_box(box) or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f"Invalid mapped image coordinates: {candidate_id}")
            expected = make_candidate(key, row.get("pass_id"), row.get("ordinal_in_pass"),
                                      {"bbox": row.get("local_bbox"), "class_id": class_id,
                                       "type": row.get("type"), "score": score}, row["source_pass"],
                                      window, width, height, row.get("model_input_size"),
                                      row.get("edge", {}).get("margin_original_crop_pixels", 2))
            if expected["bbox"] != box or expected["edge"] != row.get("edge"):
                raise ValueError(f"Inconsistent coordinate mapping or edge flags: {candidate_id}")
            if expected["coordinate_mapping"] != row.get("coordinate_mapping") or expected["candidate_id"] != candidate_id:
                raise ValueError(f"Inconsistent mapping or candidate identity: {candidate_id}")


def box_geometry(a: list[float], b: list[float]) -> dict:
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return {"iou": intersection / (area_a + area_b - intersection),
            "intersection_over_smaller": intersection / min(area_a, area_b),
            "smaller_to_larger_area": min(area_a, area_b) / max(area_a, area_b)}


def context_neighbors(target: dict, rows: list[dict], limit: int = 40) -> tuple[list[dict], int]:
    """Bound contextual evidence, never the candidate queue itself.

    Contained/containing boxes come first, then other overlaps, then nearby
    boxes. Within each category center distance and ID give a stable order.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("Neighbor limit must be a nonnegative integer")
    x1, y1, x2, y2 = target["bbox"]
    pad = max(16., (x2 - x1) * .4, (y2 - y1) * .4)
    context = [x1 - pad, y1 - pad, x2 + pad, y2 + pad]
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    ranked = []
    for other in rows:
        if (other["candidate_id"] == target["candidate_id"] or
                min(context[2], other["bbox"][2]) <= max(context[0], other["bbox"][0]) or
                min(context[3], other["bbox"][3]) <= max(context[1], other["bbox"][1])):
            continue
        geometry = box_geometry(target["bbox"], other["bbox"])
        category = (0 if geometry["intersection_over_smaller"] >= .85 else
                    1 if geometry["iou"] > 0 else 2)
        box = other["bbox"]
        distance_squared = ((box[0] + box[2]) / 2 - center[0]) ** 2 + ((box[1] + box[3]) / 2 - center[1]) ** 2
        ranked.append(((category, distance_squared, other["candidate_id"]), other))
    ranked.sort(key=lambda pair: pair[0])
    neighbors = [{"id": row["candidate_id"], "bbox": row["bbox"], "score": row["score"]}
                 for _, row in ranked[:limit]]
    return neighbors, max(0, len(ranked) - limit)


def conservative_policy(rows: list[dict], near_duplicate_iou: float = .95,
                        conflict_iou: float = .15, containment: float = .85) -> dict:
    """Suppress only exact / very near duplicate cross-pass boxes, audit all.

    Contained components, edge fragments, same-pass overlaps, and materially
    different whole/tile boxes remain unresolved predictions. A connected
    review cluster does not imply that its members describe one component.
    This function never accepts ground truth and never changes coordinates.
    """
    if not 0 < conflict_iou < near_duplicate_iou <= 1 or not 0 < containment <= 1:
        raise ValueError("Invalid duplicate/conflict thresholds")
    invalid = [row["candidate_id"] for row in rows if not row["valid"]]
    valid = [row for row in rows if row["valid"]]
    # Prefer the whole-image geometry and completeness, not the higher score.
    ranked = sorted(valid, key=lambda row: (bool(row["edge"]["touches_internal"]),
                                            row["source_pass"] != "whole_image", row["candidate_id"]))
    kept, suppressed, exact_keys = [], [], {}
    for row in ranked:
        signature = (row["class_id"], *row["bbox"])
        representative = exact_keys.get(signature)
        reason = "exact_coordinates_same_class" if representative else None
        if representative is None and not row["edge"]["touches_internal"]:
            for previous in kept:
                if (row["class_id"] != previous["class_id"] or row["pass_id"] == previous["pass_id"] or
                        previous["edge"]["touches_internal"]):
                    continue
                geometry = box_geometry(row["bbox"], previous["bbox"])
                if geometry["iou"] >= near_duplicate_iou and geometry["smaller_to_larger_area"] >= .95:
                    representative, reason = previous, "near_identical_cross_pass_geometry"
                    break
        if representative is not None:
            suppressed.append({"candidate_id": row["candidate_id"],
                               "representative_id": representative["candidate_id"], "reason": reason,
                               "iou": box_geometry(row["bbox"], representative["bbox"])["iou"],
                               "coordinates_unchanged": True})
        else:
            kept.append(row)
            exact_keys[signature] = row
    flags = {row["candidate_id"]: (["internal_tile_edge"] if row["edge"]["touches_internal"] else [])
             for row in kept}
    pairs = []
    adjacency = {row["candidate_id"]: set() for row in kept}
    for index, a in enumerate(kept):
        for b in kept[index + 1:]:
            geometry = box_geometry(a["bbox"], b["bbox"])
            if geometry["iou"] < conflict_iou and geometry["intersection_over_smaller"] < containment:
                continue
            reasons = []
            if geometry["intersection_over_smaller"] >= containment:
                reasons.append("containment_or_near_containment")
            if a["source_pass"] != b["source_pass"]:
                reasons.append("whole_tile_geometry_disagreement")
            if a["edge"]["touches_internal"] or b["edge"]["touches_internal"]:
                reasons.append("boundary_overlap")
            if a["class_id"] != b["class_id"]:
                reasons.append("class_disagreement")
            if not reasons:
                reasons.append("overlap_without_safe_duplicate_proof")
            a_id, b_id = a["candidate_id"], b["candidate_id"]
            pairs.append({"candidate_ids": [a_id, b_id], **geometry, "reasons": reasons,
                          "action": "retain_both_pending_contextual_review"})
            adjacency[a_id].add(b_id)
            adjacency[b_id].add(a_id)
            for candidate_id in (a_id, b_id):
                flags[candidate_id].extend(reason for reason in reasons if reason not in flags[candidate_id])
    seen, groups = set(), []
    for start in sorted(adjacency):
        if start in seen or not adjacency[start]:
            continue
        todo, members = [start], []
        while todo:
            current = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            members.append(current)
            todo.extend(sorted(adjacency[current] - seen))
        groups.append({"candidate_ids": sorted(members), "action": "review_only_not_a_merged_component"})
    output = [{**row, "policy_status": "keep_pending_qwen" if flags[row["candidate_id"]] else "keep",
               "review_reasons": flags[row["candidate_id"]]} for row in kept]
    return {"predictions": output, "suppressed": suppressed, "invalid_retained_in_raw": invalid,
            "conflict_pairs": pairs, "review_groups": groups,
            "settings": {"near_duplicate_iou": near_duplicate_iou, "conflict_iou": conflict_iou,
                         "containment": containment},
            "counts": {"raw": len(rows), "valid_raw": len(valid), "invalid": len(invalid),
                       "pre_qwen": len(output), "suppressed_duplicates": len(suppressed),
                       "pending": sum(bool(row["review_reasons"]) for row in output)}}


def maximum_matching_coverage(ground_truth: list[dict], predictions: list[dict],
                              iou_threshold: float = .5) -> dict:
    """Diagnostic maximum bipartite coverage, never an inference selection rule.

    A component cannot be credited twice and one broad candidate cannot cover
    multiple ground-truth objects. No oracle-selected boxes are returned.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU must be in (0,1]")
    neighbors = []
    for prediction in predictions:
        overlaps = [(box_geometry(prediction["bbox"], gt["bbox"])["iou"], index)
                    for index, gt in enumerate(ground_truth) if prediction["class_id"] == gt["class_id"]]
        neighbors.append([index for overlap, index in sorted(overlaps, reverse=True) if overlap >= iou_threshold])
    owners: dict[int, int] = {}
    # Iterative augmenting paths avoid recursion limits on dense schematics.
    for start in range(len(predictions)):
        parents: dict[int, tuple[int, int] | None] = {start: None}
        visited_gt, queue, found = set(), deque([start]), None
        while queue and found is None:
            current = queue.popleft()
            for gt_index in neighbors[current]:
                if gt_index in visited_gt:
                    continue
                visited_gt.add(gt_index)
                if gt_index not in owners:
                    found = (current, gt_index)
                    break
                previous = owners[gt_index]
                if previous not in parents:
                    parents[previous] = (current, gt_index)
                    queue.append(previous)
        if found is not None:
            current, gt_index = found
            owners[gt_index] = current
            while parents[current] is not None:
                previous, previous_gt = parents[current]
                owners[previous_gt] = previous
                current = previous
    return {"matched_maximum": len(owners), "ground_truth": len(ground_truth),
            "candidate_count": len(predictions), "uncovered_ground_truth": len(ground_truth) - len(owners),
            "recall_upper_bound": len(owners) / len(ground_truth) if ground_truth else None,
            "metric": "maximum_class_aware_one_to_one_matching_diagnostic_not_realized_precision_or_recall",
            "iou_threshold": iou_threshold,
            "selection_warning": "Uses ground truth only for aggregate diagnostics; never exported as selected predictions."}


def dump_json(path: Path, payload: Any) -> None:
    """Write strict JSON atomically in the output's own directory."""
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
