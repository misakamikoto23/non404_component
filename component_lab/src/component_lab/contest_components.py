"""Prediction-only component identity refinement and contest export.

No ground truth, library IDs or evaluation matches are inputs to these functions.
Raw OCR evidence is preserved in the final text; priors only affect type choices.
"""
from collections import Counter
from copy import deepcopy

from .eval.coords import flip_bbox
from .text_matching import associate_structured
from .text_roles import REFERENCE, NUMERIC_UNIT_REFERENCE

TYPE_MAP = {
    "resistor": "r", "capacitor": "c", "inductor": "l", "ground": "gnd",
    "ic_or_module": "box", "ic": "box", "diode": "d", "led": "led",
    "connector": "connector", "motor": "motor", "switch": "switch",
    "testpoint": "testpoint", "crystal": "crystal", "fuse": "fuse",
    "battery": "battary", "transformer": "transfomer", "antenna": "ant", "optocoupler": "opt",
}
PREFIX_PRIOR = {"R": "resistor", "C": "capacitor", "L": "inductor", "SW": "switch",
                "F": "fuse", "J": "connector", "CN": "connector", "CON": "connector",
                "U": "ic_or_module", "IC": "ic_or_module"}


def raw_field(component, field):
    value = component.get("association_evidence", {}).get(field, {}).get("raw")
    return value if isinstance(value, str) else None


def refine_types(association, *, numeric_unit_refs=False):
    """Apply explicit prefix priors only to classifier-abstained predictions.

    Conflicts with strong predictions remain visible instead of silently fixing
    every D/Q/U component according to a naming convention.
    """
    rows = deepcopy(association["components"])
    changes = []
    for row in rows:
        reference = raw_field(row, "reference")
        match = REFERENCE.fullmatch(reference or "")
        if match is None and numeric_unit_refs:
            match = NUMERIC_UNIT_REFERENCE.fullmatch(reference or '')
        prior = PREFIX_PRIOR.get(match.group(1).upper()) if match else None
        if not prior:
            continue
        before = row.get("type", "component")
        classifier = row.get("classification") or {}
        compatible = TYPE_MAP.get(before, before) == TYPE_MAP.get(prior, prior)
        if not compatible and (before == "component" or classifier.get("score", 0) < .8):
            row["type"] = prior
            changes.append({"id": row["id"], "reference_raw": reference, "before": before,
                            "after": prior, "reason": "explicit_prefix_on_classifier_abstention"})
    return rows, changes


def associate_variants(components, texts, *, numeric_unit_refs=False,
                       reference_cost_adjuster=None, ambiguity_gap=.10):
    """Small predeclared ablation: identical detections/OCR in all variants."""
    options = {'numeric_unit_refs': numeric_unit_refs, 'ambiguity_gap': ambiguity_gap}
    baseline = associate_structured(components, texts, **options)
    typed, changes = refine_types(baseline, numeric_unit_refs=numeric_unit_refs)
    prefix = associate_structured(typed, texts, **options)
    corrected = associate_structured(typed, texts, clamp_costs=False,
                                     reference_cost_adjuster=reference_cost_adjuster, **options)
    # A newly resolved reference can now justify a type on an abstained crop.
    refined, extra = refine_types(corrected, numeric_unit_refs=numeric_unit_refs)
    corrected = associate_structured(refined, texts, clamp_costs=False,
                                     reference_cost_adjuster=reference_cost_adjuster, **options)
    return {"baseline": baseline, "prefix": prefix, "identity_v1": corrected}, {
        "prefix_changes": changes, "second_pass_changes": extra,
        "settings": {"classifier_abstention_threshold": .8, "ambiguity_gap": ambiguity_gap,
                     "max_distance_heights": 8, "unmatched_cost": 1.75,
                     "numeric_unit_refs": numeric_unit_refs,
                     "reference_group_adjustment_enabled": reference_cost_adjuster is not None,
                     "final_negative_costs_preserved": True}}


def export_contest(association, height, *, ic_value_as_name=False, display_names=False):
    """Export all detections; unresolved identities remain explicit false positives."""
    out = {"components": {}, "pins": {}, "nets": {}}
    refs = Counter(raw_field(c, "reference") for c in association["components"])
    trace = []
    for i, row in enumerate(association["components"], 1):
        reference = raw_field(row, "reference")
        key = reference if reference and refs[reference] == 1 else f"__unresolved_detection_{i:04d}"
        while key in out["components"]:
            key += "_"
        x1, y1, x2, y2 = row["bbox"]
        if not 0 <= y1 < y2 <= height:
            raise ValueError("component bbox outside image")
        kind = TYPE_MAP.get(row.get("type"), row.get("type", "component"))
        name, name_source = raw_field(row, "part_number"), "part_number"
        if display_names and kind == "box" and name is None:
            name, name_source = raw_field(row, "display_name"), "display_name"
        if ic_value_as_name and kind == "box" and name is None:
            # A box may be titled by a frequency or another token parsed as a
            # value. Reuse only its OWN assigned token, never invent a model.
            name, name_source = raw_field(row, "value"), "value_fallback"
        out["components"][key] = {"type": kind, "bbox": flip_bbox([x1, y1, x2, y2], height),
                                  "Name": name, "value": raw_field(row, "value")}
        trace.append({"key": key, "source_id": row["id"], "type": kind, "reference_raw": reference,
                      "name_source": name_source if name is not None else None,
                      "evidence": row.get("association_evidence", {}), "review": row.get("review", [])})
    return out, trace
