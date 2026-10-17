"""Conversion between top-left image pixels and the contest's bottom-left frame.

Spec (appendix 1.2, appendix 3): x grows right, y grows up, bbox is
[x1, y1, x2, y2] with (x1, y1) the bottom-left corner. For an image of height
H, y' = H - y, and a bbox's y1/y2 swap. The map is its own inverse, so one
function serves both directions. Unlike geometry.convert_bbox it never clips.
"""
from __future__ import annotations

import copy


def flip_point(point, height):
    return {"x": point["x"], "y": height - point["y"]}


def flip_bbox(bbox, height):
    x1, y1, x2, y2 = bbox
    return [x1, height - y2, x2, height - y1]


def flip_case(payload, height):
    """Return a copy of a contest JSON with every bbox, pin point and edge flipped."""
    out = copy.deepcopy(payload)
    for item in out.get("components", {}).values():
        item["bbox"] = flip_bbox(item["bbox"], height)
    for pins in out.get("pins", {}).values():
        for pin in pins.values():
            pin["point"] = flip_point(pin["point"], height)
    for net in out.get("nets", {}).values():
        net["edges"] = {k: [flip_point(p, height) for p in edge] for k, edge in net["edges"].items()}
    return out


topleft_to_bottomleft = bottomleft_to_topleft = flip_case


def out_of_bounds(payload, width, height, slack=0.0):
    """Coordinates of a case that fall outside [0, W] x [0, H] (frame-independent check)."""
    bad = []
    inside = lambda x, y: -slack <= x <= width + slack and -slack <= y <= height + slack
    for key, item in payload["components"].items():
        x1, y1, x2, y2 = item["bbox"]
        if not (inside(x1, y1) and inside(x2, y2)):
            bad.append(("bbox", key))
    for key, pins in payload["pins"].items():
        for pin_key, pin in pins.items():
            if not inside(pin["point"]["x"], pin["point"]["y"]):
                bad.append(("pin", f"{key}.{pin_key}"))
    for net_id, net in payload["nets"].items():
        for edge_id, edge in net["edges"].items():
            if not all(inside(p["x"], p["y"]) for p in edge):
                bad.append(("edge", f"{net_id}.{edge_id}"))
    return bad
