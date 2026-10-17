"""Local re-implementation of the Contest 6 single-case accuracy score.

Score = 100 * (0.30 ComponentF1 + 0.25 PinF1 + 0.35 NetHypergraphF1 + 0.10 NetLineF1),
following the public spec (0803Rev1, section 5 and appendix 4). The official
evaluator has not shipped; every choice the spec leaves open is a field of
``ScoreConfig`` and is echoed in ``assumptions()`` so reports stay honest.

Anonymous ``∅<n>`` keys: ``anon_policy="strict"`` (headline) scores them like
any key. ``"observable_diagnostic"`` drops them from BOTH GT and prediction
and answers "how good is everything a reader could name?"; it is not the
contest metric. A pipeline that simply omits ∅ objects from its output is a
prediction-side choice and must be scored under ``strict``, where every
omitted GT ∅ object is a false negative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from .matching import max_cardinality, sparse_max_matching

WEIGHTS = {"component_f1": .30, "pin_f1": .25, "net_hypergraph_f1": .35, "net_line_f1": .10}
_VOCAB = json.loads((Path(__file__).with_name("type_vocab.json")).read_text(encoding="utf-8"))
VOCAB_VERSION = _VOCAB["version"]
CANONICAL_TYPES = _VOCAB["canonical"]
NAME_TEXT_TYPES = {"ic"}  # canonical types
VALUE_TEXT_TYPES = {"resistor", "capacitor"}
ANON_PREFIX = "∅"
ORACLE_STAGES = ("components", "pins", "nets")


@dataclass(frozen=True)
class ScoreConfig:
    bbox_tolerance: float = 20.0
    bbox_metric: str = "max_corner"  # or "center"
    point_tolerance: float = 5.0
    anon_policy: str = "strict"  # or "observable_diagnostic"
    type_policy: str = "strict"  # or "mapped"
    net_tiebreak: str = "segments"  # or "none"
    oracle: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        if self.bbox_metric not in ("max_corner", "center"):
            raise ValueError(f"unknown bbox_metric {self.bbox_metric!r}")
        if self.anon_policy not in ("strict", "observable_diagnostic"):
            raise ValueError(f"unknown anon_policy {self.anon_policy!r}")
        if self.type_policy not in ("strict", "mapped"):
            raise ValueError(f"unknown type_policy {self.type_policy!r}")
        if self.net_tiebreak not in ("segments", "none"):
            raise ValueError(f"unknown net_tiebreak {self.net_tiebreak!r}")
        unknown = set(self.oracle) - set(ORACLE_STAGES)
        if unknown:
            raise ValueError(f"unknown oracle stage(s) {sorted(unknown)}")

    @property
    def oracle_label(self):
        return "+".join(s for s in ORACLE_STAGES if s in self.oracle) or "none"

    def header(self):
        return (f"type_policy={self.type_policy} anon_policy={self.anon_policy} bbox_metric={self.bbox_metric} "
                f"bbox_tol={self.bbox_tolerance:g}px point_tol={self.point_tolerance:g}px "
                f"net_tiebreak={self.net_tiebreak} vocab={VOCAB_VERSION}")

    def assumptions(self):
        return {
            "evaluator": "local re-implementation of the public spec, NOT the official evaluator",
            "bbox_error": (f"{self.bbox_metric}: " + ("max |pred-gt| over the four bbox coordinates"
                           if self.bbox_metric == "max_corner" else "Euclidean distance between bbox centers")
                           + f" <= {self.bbox_tolerance} px (ASSUMPTION until the official evaluator ships)"),
            "point_tolerance": f"Euclidean distance <= {self.point_tolerance} px for pins and both segment endpoints",
            "component_text": ("keyed on the CANONICAL GT type in both type policies: Name for ic (box/boxic/ic), "
                               "value for resistor/capacitor (r/c), nothing otherwise; strict string equality; "
                               "GT None vs pred None counts as equal (ASSUMPTION)"),
            "type_policy": ("strict: raw type strings must be equal (headline)" if self.type_policy == "strict" else
                            f"mapped (DIAGNOSTIC): types compared after canonicalization with {VOCAB_VERSION}; "
                            "types missing from the table keep their raw string and are reported as unmapped"),
            "anon_policy": ("strict: ∅ keys scored like any key (headline)" if self.anon_policy == "strict" else
                            "observable_diagnostic (DIAGNOSTIC, not the contest metric): ∅ components, their pins "
                            "and ∅ net members dropped from GT and prediction; nets left empty by that removal "
                            "are dropped with their edges"),
            "net_pairing": ("Hungarian maximum of pin-set intersection; zero-weight pairs never match; " + (
                "net_tiebreak=segments (ASSUMPTION, not in the spec): equal-overlap ties broken by matched-segment "
                "count, and pin-less nets may pair with pin-less nets by segments"
                if self.net_tiebreak == "segments" else
                "net_tiebreak=none: ties resolved by solver order over sorted net ids; pin-less nets never match")),
            "segment_pairing": "maximum-cardinality matching, endpoints compared in either orientation",
            "empty_metric": "F1 is 1.0 when GT and prediction are both empty for that metric",
            "invalid_prediction": "missing, unreadable or schema-violating prediction scores 0",
            "oracle": (f"output substitution ({self.oracle_label}): that JSON section is replaced by GT after "
                       "the pipeline ran; downstream stages are NOT rerun, so this is not a pipeline oracle"),
        }


class InvalidCase(ValueError):
    """Raised when a JSON payload violates the output contract."""


# ---------------------------------------------------------------- validation

def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _point(value, label):
    if not isinstance(value, dict) or not _number(value.get("x")) or not _number(value.get("y")):
        raise InvalidCase(f"{label}: point must be {{'x': number, 'y': number}}")
    return float(value["x"]), float(value["y"])


def parse_hypergraph(text, component_ids, label="hyperGraph"):
    """Parse ``(U1.A5,R1.1)`` into ``{(component, pin_number)}``.

    Component keys may contain dots, so the longest known key prefix wins;
    otherwise the item splits at its first dot.
    """
    if not isinstance(text, str) or not (text.startswith("(") and text.endswith(")")):
        raise InvalidCase(f"{label}: must be a string like '(U1.1,R1.2)'")
    inner = text[1:-1].strip()
    if not inner:
        return set()
    known = component_ids if isinstance(component_ids, list) else sorted(component_ids, key=len, reverse=True)
    members = set()
    for raw in inner.split(","):
        item = raw.strip()
        component = next((key for key in known if item.startswith(key + ".")), None)
        if component is None:
            if "." not in item:
                raise InvalidCase(f"{label}: member {item!r} has no '.'")
            component = item.split(".", 1)[0]
        number = item[len(component) + 1:]
        if not component or not number:
            raise InvalidCase(f"{label}: member {item!r} is malformed")
        members.add((component, number))
    return members


def normalize_case(payload):
    """Validate one case JSON and return a compact internal form.

    Internal form: components {key: (type, Name, value, bbox)}, pins
    {(key, pin_key): (pinname, (x, y))}, nets {id: (members, [((x,y),(x,y))])}.
    """
    if not isinstance(payload, dict):
        raise InvalidCase("top level must be an object")
    for section in ("components", "pins", "nets"):
        if not isinstance(payload.get(section), dict):
            raise InvalidCase(f"missing or non-object top-level field {section!r}")
    components = {}
    for key, item in payload["components"].items():
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise InvalidCase(f"component {key}: needs a string 'type'")
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(_number(v) for v in bbox):
            raise InvalidCase(f"component {key}: bbox must be four numbers")
        components[key] = (item["type"], item.get("Name"), item.get("value"), tuple(float(v) for v in bbox))
    pins = {}
    for key, group in payload["pins"].items():
        if not isinstance(group, dict):
            raise InvalidCase(f"pins.{key} must be an object")
        for pin_key, pin in group.items():
            if not isinstance(pin, dict):
                raise InvalidCase(f"pins.{key}.{pin_key} must be an object")
            pins[(key, pin_key)] = (pin.get("pinname"), _point(pin.get("point"), f"pins.{key}.{pin_key}"))
    nets = {}
    known = sorted(components, key=len, reverse=True)
    for net_id, net in payload["nets"].items():
        if not isinstance(net, dict) or not isinstance(net.get("edges"), dict):
            raise InvalidCase(f"net {net_id}: needs 'hyperGraph' and an 'edges' object")
        members = {(c, "pin_" + n) for c, n in parse_hypergraph(net.get("hyperGraph"), known, f"net {net_id}")}
        edges = []
        for edge_id, edge in net["edges"].items():
            if not isinstance(edge, list) or len(edge) != 2:
                raise InvalidCase(f"net {net_id}.{edge_id}: an edge must hold exactly two points")
            edges.append((_point(edge[0], f"{net_id}.{edge_id}[0]"), _point(edge[1], f"{net_id}.{edge_id}[1]")))
        nets[net_id] = (members, edges)
    return {"components": components, "pins": pins, "nets": nets}


def canonical_type(raw):
    """Canonical type from type_vocab.json, or the raw string when the table lacks it."""
    return CANONICAL_TYPES.get(raw, CANONICAL_TYPES.get(raw.lower(), raw))


def unmapped_types(case):
    return sorted({item[0] for item in case["components"].values()
                   if item[0] not in CANONICAL_TYPES and item[0].lower() not in CANONICAL_TYPES})


def exclude_anonymous(case):
    """Drop ∅-keyed components, their pins and ∅ net members (anon_policy=observable_diagnostic)."""
    anon = lambda key: str(key).startswith(ANON_PREFIX)
    nets = {}
    for net_id, (members, edges) in case["nets"].items():
        kept = {m for m in members if not anon(m[0])}
        if kept or not members:
            nets[net_id] = (kept, edges)
    return {"components": {k: v for k, v in case["components"].items() if not anon(k)},
            "pins": {k: v for k, v in case["pins"].items() if not anon(k[0])},
            "nets": nets}


# ------------------------------------------------------------------- metrics

def _f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2 * tp / denominator


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bbox_ok(pred, gt, config):
    if config.bbox_metric == "center":
        return _distance(((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2),
                         ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)) <= config.bbox_tolerance
    return max(abs(p - g) for p, g in zip(pred, gt)) <= config.bbox_tolerance


def text_ok(pred, gt):
    kind = canonical_type(gt[0])
    if kind in NAME_TEXT_TYPES:
        return pred[1] == gt[1]
    if kind in VALUE_TEXT_TYPES:
        return pred[2] == gt[2]
    return True


def component_counts(pred, gt, config):
    diag = {"key_hit": 0, "type_wrong": 0, "bbox_wrong": 0, "text_wrong": 0}
    tp = 0
    for key, expected in gt.items():
        actual = pred.get(key)
        if actual is None:
            continue
        diag["key_hit"] += 1
        good_type = (actual[0] == expected[0] if config.type_policy == "strict"
                     else canonical_type(actual[0]) == canonical_type(expected[0]))
        good_box = bbox_ok(actual[3], expected[3], config)
        good_text = text_ok(actual, expected)
        diag["type_wrong"] += not good_type
        diag["bbox_wrong"] += not good_box
        diag["text_wrong"] += not good_text
        tp += good_type and good_box and good_text
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp}, diag


def pin_counts(pred, gt, config):
    diag = {"key_hit": 0, "pinname_wrong": 0, "point_wrong": 0}
    tp = 0
    for key, (name, point) in gt.items():
        actual = pred.get(key)
        if actual is None:
            continue
        diag["key_hit"] += 1
        good_name = actual[0] == name
        good_point = _distance(actual[1], point) <= config.point_tolerance
        diag["pinname_wrong"] += not good_name
        diag["point_wrong"] += not good_point
        tp += good_name and good_point
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp}, diag


def _segment_ok(a, b, tolerance):
    direct = max(_distance(a[0], b[0]), _distance(a[1], b[1]))
    swapped = max(_distance(a[0], b[1]), _distance(a[1], b[0]))
    return min(direct, swapped) <= tolerance


def matched_segments(gt_edges, pred_edges, tolerance):
    return max_cardinality(len(gt_edges), len(pred_edges),
                           lambda i, j: _segment_ok(gt_edges[i], pred_edges[j], tolerance))


def net_counts(pred, gt, config):
    """Pair nets by pin overlap, then count hypergraph members and segments."""
    gt_ids, pred_ids = sorted(gt), sorted(pred)
    member_index = {}
    for pid in pred_ids:
        for member in pred[pid][0]:
            member_index.setdefault(member, set()).add(pid)
    overlap = {}
    for gid in gt_ids:
        for member in gt[gid][0]:
            for pid in member_index.get(member, ()):
                overlap[(gid, pid)] = overlap.get((gid, pid), 0) + 1
    segment_hits = lambda pair: matched_segments(gt[pair[0]][1], pred[pair[1]][1], config.point_tolerance)
    if config.net_tiebreak == "segments":
        # Pin-less nets can only pair with pin-less nets, via geometry.
        for gid in gt_ids:
            if not gt[gid][0]:
                for pid in pred_ids:
                    if not pred[pid][0]:
                        overlap.setdefault((gid, pid), 0)
        segments = {pair: segment_hits(pair) for pair in overlap}
        # Preserve primary pin overlap across the WHOLE assignment. Bounding a
        # single pair is insufficient: several segment-rich pairs could otherwise
        # outweigh one lost pin match. Each segment can match at most once, so
        # this bounds the total secondary objective under any net assignment.
        scale = 1 + min(sum(len(n[1]) for n in gt.values()),
                        sum(len(n[1]) for n in pred.values()))
        weight = {pair: count * scale + segments[pair] for pair, count in overlap.items()}
        matches = sparse_max_matching(gt_ids, pred_ids, weight)
    else:
        matches = sparse_max_matching(gt_ids, pred_ids, overlap)
        segments = {pair: segment_hits(pair) for pair in matches}

    hyper = {"tp": 0, "fp": 0, "fn": 0}
    lines = {"tp": 0, "fp": 0, "fn": 0}
    matched_gt = {g for g, _ in matches}
    matched_pred = {p for _, p in matches}
    for gid, pid in matches:
        expected, actual = gt[gid][0], pred[pid][0]
        hyper["tp"] += len(expected & actual)
        hyper["fp"] += len(actual - expected)
        hyper["fn"] += len(expected - actual)
        hit = segments[(gid, pid)]
        lines["tp"] += hit
        lines["fp"] += len(pred[pid][1]) - hit
        lines["fn"] += len(gt[gid][1]) - hit
    for pid in pred_ids:
        if pid not in matched_pred:
            hyper["fp"] += len(pred[pid][0])
            lines["fp"] += len(pred[pid][1])
    for gid in gt_ids:
        if gid not in matched_gt:
            hyper["fn"] += len(gt[gid][0])
            lines["fn"] += len(gt[gid][1])
    diag = {"gt_nets": len(gt), "pred_nets": len(pred), "matched_nets": len(matches)}
    return hyper, lines, diag


def _assemble(counts, diagnostics, status, reason=""):
    metrics = {name: _f1(**counts[name]) for name in WEIGHTS}
    return {"status": status, "reason": reason,
            "score": 100.0 * sum(WEIGHTS[name] * metrics[name] for name in WEIGHTS),
            **metrics, "counts": counts, "diagnostics": diagnostics}


def score_case(prediction, ground_truth, config=ScoreConfig()):
    """Score one case. ``prediction`` may be ``None`` (missing) or any JSON value.

    Invalid ground truth raises ``InvalidCase``; an invalid prediction scores 0.
    Oracle stages replace that section of a valid prediction with GT.
    """
    gt = normalize_case(ground_truth)
    status, reason = "ok", ""
    if prediction is None:
        status, reason = "missing", "no prediction file"
    else:
        try:
            if config.oracle and isinstance(prediction, dict):
                prediction = dict(prediction)
                for stage in config.oracle:
                    prediction[stage] = ground_truth[stage]
            pred = normalize_case(prediction)
        except InvalidCase as exc:
            status, reason = "invalid", str(exc)
    if status != "ok":
        pred = {"components": {}, "pins": {}, "nets": {}}
    if config.anon_policy == "observable_diagnostic":
        gt, pred = exclude_anonymous(gt), exclude_anonymous(pred)
    types = {"gt_unmapped": unmapped_types(gt), "pred_unmapped": unmapped_types(pred)}
    comp, comp_diag = component_counts(pred["components"], gt["components"], config)
    pins, pin_diag = pin_counts(pred["pins"], gt["pins"], config)
    hyper, lines, net_diag = net_counts(pred["nets"], gt["nets"], config)
    counts = {"component_f1": comp, "pin_f1": pins, "net_hypergraph_f1": hyper, "net_line_f1": lines}
    result = _assemble(counts, {"components": comp_diag, "pins": pin_diag, "nets": net_diag}, status, reason)
    result["types"] = types
    if status != "ok":
        result["score"] = 0.0
        for name in WEIGHTS:
            result[name] = 0.0
    return result
