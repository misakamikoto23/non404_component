"""CPU structural evidence for schematic detections, without oracle decisions.

Geometry is never repaired here. A relationship is a reason to inspect a
region, not proof that boxes should be merged. The default policy annotates
only; the optional frame guard requires an actual four-sided near-page frame
and checks for repeated boundary pin stubs before suppressing anything.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import re
from typing import Any

import cv2
import numpy as np

from .candidate_postprocess import box_geometry


POLICY_VERSION = "structural_diagnostics_v1"
SETTINGS = {
    "foreground_gray_max": 220, "long_line_image_fraction": .18,
    "minimum_long_line_pixels": 40, "border_tolerance_pixels": 5,
    "frame_minimum_side_coverage": .90, "frame_minimum_width_fraction": .80,
    "frame_minimum_height_fraction": .80, "frame_maximum_page_margin_fraction": .12,
    "containment_over_smaller": .85, "containment_maximum_area_ratio": .70,
    "nearby_gap_fraction": .18, "nearby_minimum_gap_pixels": 6,
}


def _box(box: Any) -> list[float]:
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        raise ValueError("Expected xyxy box")
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values) or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("Invalid xyxy box")
    return values


def line_evidence(image: np.ndarray) -> dict:
    """Extract long horizontal/vertical strokes in original image coordinates.

    The input is RGB or grayscale, not OpenCV BGR. Bright colored schematic
    strokes are retained by their minimum channel as well as grayscale value.
    Masks are transient evidence and never substituted for detector images.
    """
    image = np.asarray(image)
    if image.ndim not in {2, 3} or min(image.shape[:2]) < 2:
        raise ValueError("Expected a nonempty grayscale/RGB image")
    if image.ndim == 3 and image.shape[2] not in {3, 4}:
        raise ValueError("Expected RGB or RGBA image")
    color = image[..., :3] if image.ndim == 3 else image
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else color
    minimum = color.min(axis=2) if image.ndim == 3 else gray
    foreground = ((gray < SETTINGS["foreground_gray_max"]) | (minimum < 180)).astype(np.uint8) * 255
    height, width = foreground.shape
    horizontal_length = min(width, max(SETTINGS["minimum_long_line_pixels"], round(width * .18)))
    vertical_length = min(height, max(SETTINGS["minimum_long_line_pixels"], round(height * .18)))
    horizontal = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((1, horizontal_length), np.uint8))
    vertical = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((vertical_length, 1), np.uint8))
    return {"width": width, "height": height, "foreground": foreground,
            "horizontal": horizontal, "vertical": vertical,
            "kernels": {"horizontal": horizontal_length, "vertical": vertical_length}}


def _coverage(mask: np.ndarray, box: list[float], side: str, tolerance: int = 5) -> float:
    height, width = mask.shape
    x1, y1, x2, y2 = [round(value) for value in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.
    if side in {"top", "bottom"}:
        y = y1 if side == "top" else min(height - 1, y2)
        strip = mask[max(0, y - tolerance):min(height, y + tolerance + 1), x1:x2]
        return float(np.count_nonzero(np.any(strip > 0, axis=0)) / (x2 - x1))
    x = x1 if side == "left" else min(width - 1, x2)
    strip = mask[y1:y2, max(0, x - tolerance):min(width, x + tolerance + 1)]
    return float(np.count_nonzero(np.any(strip > 0, axis=1)) / (y2 - y1))


def _boundary_pin_stubs(foreground: np.ndarray, box: list[float]) -> dict:
    """Count separated strokes extending across a rectangle boundary.

    These are pin-like geometry, not inferred electrical pins. Three or more
    on one side protect a repeated-pin package from optional frame removal.
    """
    height, width = foreground.shape
    x1, y1, x2, y2 = [round(value) for value in box]
    counts = {}
    for side in ("top", "bottom", "left", "right"):
        if side in {"left", "right"}:
            x = x1 if side == "left" else x2
            ylo, yhi = max(0, y1 + 8), min(height, y2 - 8)
            lo, hi = x - 12, x + 13
            if lo < 0 or hi > width or yhi <= ylo:
                counts[side] = 0
                continue
            strip = foreground[ylo:yhi, lo:hi] > 0
            hits = (strip[:, :8].mean(axis=1) >= .65) & (strip[:, -8:].mean(axis=1) >= .65)
        else:
            y = y1 if side == "top" else y2
            xlo, xhi = max(0, x1 + 8), min(width, x2 - 8)
            lo, hi = y - 12, y + 13
            if lo < 0 or hi > height or xhi <= xlo:
                counts[side] = 0
                continue
            strip = foreground[lo:hi, xlo:xhi] > 0
            hits = (strip[:8, :].mean(axis=0) >= .65) & (strip[-8:, :].mean(axis=0) >= .65)
        padded = np.pad(hits.astype(np.int8), (1, 1))
        counts[side] = int(np.count_nonzero(np.diff(padded) == 1))
    return {"counts_by_side": counts, "repeated_boundary_strokes": max(counts.values()) >= 3,
            "meaning": "pin_like_geometry_only_not_electrical_pin_detection"}


def frame_evidence(box: list[float], evidence: dict) -> dict:
    box = _box(box)
    width, height = evidence["width"], evidence["height"]
    x1, y1, x2, y2 = box
    coverage = {side: _coverage(evidence["horizontal"] if side in {"top", "bottom"} else evidence["vertical"],
                                box, side, SETTINGS["border_tolerance_pixels"])
                for side in ("top", "bottom", "left", "right")}
    four_sides = all(value >= SETTINGS["frame_minimum_side_coverage"] for value in coverage.values())
    span = [(x2 - x1) / width, (y2 - y1) / height]
    margins = [x1 / width, y1 / height, (width - x2) / width, (height - y2) / height]
    near_page = (span[0] >= SETTINGS["frame_minimum_width_fraction"] and
                 span[1] >= SETTINGS["frame_minimum_height_fraction"] and
                 all(0 <= margin <= SETTINGS["frame_maximum_page_margin_fraction"] for margin in margins))
    pin_stubs = _boundary_pin_stubs(evidence["foreground"], box)
    corroborated = four_sides and near_page and not pin_stubs["repeated_boundary_strokes"]
    return {"side_long_line_coverage": coverage, "four_sides_supported": four_sides,
            "span_image_fraction": span, "page_margin_fraction": margins,
            "near_page_perimeter": near_page, "boundary_strokes": pin_stubs,
            "corroborated_page_frame_hypothesis": corroborated,
            "rectangular_symbol_protection": not near_page or pin_stubs["repeated_boundary_strokes"],
            "warning": "Four-sided rectangles and large boxes alone are not frame proof; subregion boxes are review-only."}


def _text_refs(text_rows: list[dict], box: list[float]) -> list[dict]:
    """Optional neutral OCR hints; never used to suppress or merge candidates."""
    output = []
    for index, text in enumerate(text_rows):
        value = str(text.get("text", text.get("raw_text", ""))).strip()
        if not re.fullmatch(r"[A-Za-z]{1,4}\d+[A-Za-z]?", value):
            continue
        raw_box = text.get("bbox")
        try:
            bounds = _box(raw_box)
        except (ValueError, TypeError):
            continue
        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        padding = 20
        if box[0] - padding <= cx <= box[2] + padding and box[1] - padding <= cy <= box[3] + padding:
            output.append({"text_id": text.get("text_id", text.get("id", str(index))), "text": value,
                           "bbox": list(raw_box), "role": "reference_candidate_not_assigned"})
    return output


def annotate_image(image: np.ndarray, candidates: list[dict], policy: str = "annotate_only",
                   text_rows: list[dict] | None = None) -> dict:
    """Preserve all originals and return diagnostic annotations + safe queue.

    The caller decides its original candidate selection before invoking this
    function. It does not NMS, resize, reclassify, join boxes, or use labels.
    """
    if policy not in {"annotate_only", "frame_guard"}:
        raise ValueError("Policy must be annotate_only or frame_guard")
    evidence = line_evidence(image)
    ids = [row.get("candidate_id") for row in candidates]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Candidates require unique stable candidate_id values")
    rows = deepcopy(candidates)
    for row in rows:
        box = _box(row["bbox"])
        if not (0 <= box[0] < box[2] <= evidence["width"] and 0 <= box[1] < box[3] <= evidence["height"]):
            raise ValueError("Candidate must lie inside original image")
        row["structure"] = {"frame": frame_evidence(box, evidence), "review_reasons": [],
                            "nearby_reference_candidates": _text_refs(text_rows or [], box)}
        if row["structure"]["frame"]["four_sides_supported"]:
            row["structure"]["review_reasons"].append("four_sided_rectangle_not_necessarily_frame")
    pairs, child_ids = [], {value: [] for value in ids}
    adjacency = {value: set() for value in ids}
    for index, a in enumerate(rows):
        for b in rows[index + 1:]:
            geometry = box_geometry(a["bbox"], b["bbox"])
            reasons = []
            if (geometry["intersection_over_smaller"] >= .85 and
                    geometry["smaller_to_larger_area"] <= .70):
                reasons.append("contained_geometry_split_or_merge_ambiguous")
                area_a = (a["bbox"][2] - a["bbox"][0]) * (a["bbox"][3] - a["bbox"][1])
                area_b = (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
                larger, smaller = (a, b) if area_a > area_b else (b, a)
                child_ids[larger["candidate_id"]].append(smaller["candidate_id"])
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]
            gap_x = max(0., max(ax1, bx1) - min(ax2, bx2))
            gap_y = max(0., max(ay1, by1) - min(ay2, by2))
            small_side = min(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1)
            near = math.hypot(gap_x, gap_y) <= max(6., .18 * small_side)
            seam = a.get("edge", {}).get("touches_internal") or b.get("edge", {}).get("touches_internal")
            cross_pass = a.get("pass_id") != b.get("pass_id")
            if near and (seam or cross_pass) and geometry["iou"] < .5:
                reasons.append("nearby_cross_pass_or_tile_seam_disagreement")
            if not reasons:
                continue
            a_id, b_id = a["candidate_id"], b["candidate_id"]
            pairs.append({"candidate_ids": [a_id, b_id], **geometry, "reasons": reasons,
                          "action": "retain_both_no_automatic_merge_or_split"})
            adjacency[a_id].add(b_id)
            adjacency[b_id].add(a_id)
            for row in (a, b):
                row["structure"]["review_reasons"].extend(reason for reason in reasons
                                                           if reason not in row["structure"]["review_reasons"])
    for row in rows:
        children = child_ids[row["candidate_id"]]
        row["structure"]["contained_candidate_ids"] = children
        if len(children) >= 2:
            row["structure"]["review_reasons"].append("one_large_box_multiple_smaller_candidates_unresolved")
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
        member_rows = [row for row in rows if row["candidate_id"] in members]
        region = [min(row["bbox"][0] for row in member_rows), min(row["bbox"][1] for row in member_rows),
                  max(row["bbox"][2] for row in member_rows), max(row["bbox"][3] for row in member_rows)]
        pad = max(16., .15 * max(region[2] - region[0], region[3] - region[1]))
        context = [max(0., region[0] - pad), max(0., region[1] - pad),
                   min(float(evidence["width"]), region[2] + pad),
                   min(float(evidence["height"]), region[3] + pad)]
        group_id = hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:20]
        reasons = sorted({reason for row in member_rows for reason in row["structure"]["review_reasons"]})
        groups.append({"group_id": group_id, "candidate_ids": sorted(members),
                       "review_region_bbox": region, "context_bbox": context,
                       "region_image_area_fraction": ((region[2] - region[0]) * (region[3] - region[1]) /
                                                      (evidence["width"] * evidence["height"])),
                       "reasons": reasons, "is_component_geometry": False,
                       "action": "context_review_only_not_one_component"})
    suppressed, kept = [], []
    for row in rows:
        remove = policy == "frame_guard" and row["structure"]["frame"]["corroborated_page_frame_hypothesis"]
        row["structure"]["decision"] = "suppress_corroborated_page_frame" if remove else "retain"
        if remove:
            suppressed.append({"candidate_id": row["candidate_id"], "reason": "four_sided_near_page_frame_without_repeated_pin_stubs",
                               "bbox": row["bbox"], "evidence": row["structure"]["frame"],
                               "reversible": True, "original_retained": True})
        else:
            kept.append(row)
    return {"policy_version": POLICY_VERSION, "policy": policy, "settings": dict(SETTINGS),
            "original_candidates": deepcopy(candidates), "annotated_candidates": rows, "predictions": kept,
            "suppressed": suppressed, "conflict_pairs": pairs, "review_groups": groups,
            "coordinates_unchanged": True, "line_kernels": evidence["kernels"],
            "counts": {"input": len(rows), "retained": len(kept), "suppressed": len(suppressed),
                       "review_candidates": sum(bool(row["structure"]["review_reasons"]) for row in rows),
                       "conflict_pairs": len(pairs), "conflict_groups": len(groups)}}
