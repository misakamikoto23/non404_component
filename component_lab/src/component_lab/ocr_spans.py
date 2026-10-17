"""Conservative substring candidates; never repair characters or use target labels."""
from copy import deepcopy
import math
import re

from .text_roles import REFERENCE, parse_text_role

DECORATION = " \t\r\n$\\{}[]().,:;"


def _interval(raw):
    left = len(raw) - len(raw.lstrip(DECORATION))
    right = len(raw.rstrip(DECORATION))
    return left, right


def candidate_spans(raw):
    """Allow whole decorated references or ref + explicit value/model only.

    Arbitrary interior regex matches are forbidden: R1 inside R10 or D2 in a
    signal expression must never be promoted into a component identity.
    """
    left, right = _interval(raw)
    if left >= right:
        return []
    token = raw[left:right]
    if REFERENCE.fullmatch(token) and (left or right != len(raw)):
        return [(left, right, "decorated_reference")]
    marker = re.fullmatch(r"[-+]?\s*[O0o○◯]\s+(TP\d+[A-Za-z]?)", token)
    if marker:
        a,b = marker.span(1)
        return [(left+a, left+b, "testpoint_marker_prefix")]
    parts = list(re.finditer(r"\S+", token))
    if len(parts) < 2 or not REFERENCE.fullmatch(parts[0].group()):
        return []
    ref = parts[0]
    remainder_start = parts[1].start()
    remainder = token[remainder_start:]
    parsed = parse_text_role({"raw": remainder, "text": remainder, "bbox": [0,0,100,10]})
    roles = [c for c in parsed["role_candidates"] if c["role"] in {"value", "part_number"}
             and not c.get("requires_review") and not c.get("requires_context")]
    if not roles:
        return []
    return [(left+ref.start(), left+ref.end(), "compound_reference"),
            (left+remainder_start, right, "compound_value_or_model")]


def span_polygon(row, start, end):
    raw = str(row.get("raw", row.get("text", "")))
    n = len(raw)
    if not 0 <= start < end <= n:
        raise ValueError("invalid character span")
    polygon = row.get("polygon")
    if not polygon or len(polygon) != 4:
        x1,y1,x2,y2 = row["bbox"]
        angle = float(row.get("orientation", 0))
        if 45 < abs(angle) <= 135:
            polygon = [[x1,y1],[x1,y2],[x2,y2],[x2,y1]]
            if angle < 0:
                polygon = [[x2,y2],[x2,y1],[x1,y1],[x1,y2]]
        else:
            polygon = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    p0,p1,p2,p3 = polygon
    lerp = lambda a,b,t: [float(a[0]+(b[0]-a[0])*t), float(a[1]+(b[1]-a[1])*t)]
    a,b = start/n,end/n
    return [lerp(p0,p1,a),lerp(p0,p1,b),lerp(p3,p2,b),lerp(p3,p2,a)]


def split_ocr_spans(texts):
    """Return mutually exclusive replacement view plus immutable parent audit.

    Geometry is a proportional-character approximation, never a character-level
    detector output. Parent records are preserved in provenance, not duplicated
    into the candidate pool alongside their children.
    """
    result, parents = [], []
    ids = {str(row.get("id", f"text_{i:04d}")) for i,row in enumerate(texts,1)}
    for i,original in enumerate(texts,1):
        row = deepcopy(original)
        row.setdefault("id", f"text_{i:04d}")
        raw = str(row.get("raw", row.get("text", "")))
        spans = candidate_spans(raw)
        if not spans:
            result.append(row)
            continue
        parent = {"source": deepcopy(row), "children": []}
        for a,b,reason in spans:
            polygon = span_polygon(row,a,b)
            xs,ys = zip(*polygon)
            child = deepcopy(row)
            # Parsed roles must be recomputed for the substring, not inherited.
            for key in ("role_candidates","primary_role","normalized_whitespace","role_score_kind"):
                child.pop(key,None)
            identifier = f"{row['id']}__span_{a}_{b}"
            while identifier in ids:
                identifier += "_"
            ids.add(identifier)
            child.update(id=identifier, raw=raw[a:b], text=raw[a:b], polygon=polygon,
                         bbox=[min(xs),min(ys),max(xs),max(ys)],
                         span_provenance={"parent_id":row["id"],"parent_raw":raw,"start":a,"end":b,
                                          "reason":reason,"geometry":"proportional_character_approximation",
                                          "uses_ground_truth":False})
            if not all(math.isfinite(v) for v in child["bbox"]):
                raise ValueError("nonfinite span geometry")
            result.append(child)
            parent["children"].append({"id":identifier,"start":a,"end":b,"reason":reason})
        parents.append(parent)
    return result, {"parents":parents,"original_count":len(texts),"replacement_count":len(result),
                    "split_parent_count":len(parents),"ground_truth_used":False}
