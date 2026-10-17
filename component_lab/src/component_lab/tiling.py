"""Model-free tile geometry shared by the production inference path.

Coordinates and merge policy match scripts/evaluate_tiled_yolo.py: complete
whole-image prediction, overlapping windows, seam filtering, then class NMS.
"""
import math


def validate_settings(tile_size=None, overlap=.25, max_tiles=100):
    if tile_size is not None and (not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0):
        raise ValueError("tile_size must be a positive integer, or None to disable tiling")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("tile_overlap must be finite and in [0, 1)")
    if not isinstance(max_tiles, int) or isinstance(max_tiles, bool) or max_tiles <= 0:
        raise ValueError("tile_max_tiles must be a positive integer")


def axis_starts(length, tile_size, overlap):
    if length <= 0 or tile_size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Image dimensions and tile size must be positive; overlap must be in [0,1).")
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def tile_plan(width, height, tile_size=1024, overlap=.25, max_tiles=100):
    """Skip the entire tile pass above the cap; never use partial coverage."""
    validate_settings(tile_size, overlap, max_tiles)
    xs, ys = axis_starts(width, tile_size, overlap), axis_starts(height, tile_size, overlap)
    required = len(xs) * len(ys)
    skipped = required > max_tiles
    windows = [] if skipped else [[x, y, min(width, x + tile_size), min(height, y + tile_size)]
                                  for y in ys for x in xs]
    return {"windows": windows, "required_tile_count": required, "tile_count": len(windows),
            "skipped": skipped, "truncated": False,
            "skip_reason": "required_tile_count_exceeds_max_tiles" if skipped else None}


def touches_internal_edge(bbox, window, width, height, margin=2):
    if margin < 0:
        raise ValueError("edge margin must be nonnegative")
    left, top, right, bottom = window
    x1, y1, x2, y2 = bbox
    return ((left > 0 and x1 <= margin) or (top > 0 and y1 <= margin) or
            (right < width and x2 >= right - left - margin) or
            (bottom < height and y2 >= bottom - top - margin))


def translate_prediction(prediction, window, width, height):
    left, top, right, bottom = window
    x1, y1, x2, y2 = prediction["bbox"]
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    bbox = [max(0.0, min(float(width), x1 + left)), max(0.0, min(float(height), y1 + top)),
            max(0.0, min(float(width), x2 + left)), max(0.0, min(float(height), y2 + top))]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return {**prediction, "bbox": bbox, "source": "tile", "tile_window": [left, top, right, bottom]}


def box_iou(a, b):
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.


def class_aware_nms(predictions, iou_threshold=.5):
    if not 0 < iou_threshold <= 1:
        raise ValueError("NMS IoU must be in (0,1].")
    ranked = sorted(enumerate(predictions), key=lambda entry: (-entry[1]["score"], entry[0]))
    kept = []
    for _, prediction in ranked:
        if not any(prediction["class_id"] == previous["class_id"] and
                   box_iou(prediction["bbox"], previous["bbox"]) > iou_threshold for previous in kept):
            kept.append(prediction)
    return kept
