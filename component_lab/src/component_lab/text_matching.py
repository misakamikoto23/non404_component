"""Text-role / component association with global slots and abstention.

This is a deterministic candidate generator. It does not validate OCR readings,
learn from ground truth, or rewrite the caller's components/texts in place.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from .text_roles import PREFIX_TYPES, parse_texts


NO_REFERENCE_TYPES = {"ground", "power_port", "net_port", "port", "no_connect"}
NUMERIC_VALUE_TYPES = {"resistor", "capacitor", "inductor", "potentiometer"}
PRIMARY_VALUE_KIND = {"resistor": "resistance", "potentiometer": "resistance",
                      "capacitor": "capacitance", "inductor": "inductance"}
ALIASES = {"res": "resistor", "cap": "capacitor", "operational_amplifier": "opamp",
           "integrated_circuit": "ic", "nmos": "mosfet", "pmos": "mosfet",
           "bipolar_transistor": "bjt", "gnd": "ground", "amplifier": "opamp",
           "ic_or_module": "ic", "testpoint": "test_point"}


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _height(text):
    box = text["bbox"]
    # For rotated OCR use its narrow dimension, avoiding a long word as scale.
    return max(3., min(abs(box[2] - box[0]), abs(box[3] - box[1])))


def _distance(a, b):
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0),
                      max(a[1] - b[3], b[1] - a[3], 0))


def _type(component):
    value = str(component.get("type", "component")).lower().replace(" ", "_")
    return ALIASES.get(value, value)


def _role(text, role, allow_context=False):
    matches = [candidate for candidate in text["role_candidates"]
               if candidate["role"] == role and not candidate.get("requires_review")
               and (allow_context or not candidate.get("requires_context"))]
    return max(matches, key=lambda row: row["heuristic_score"], default=None)


def build_text_groups(texts):
    """Link an aligned nearby value/model to a reference, without an owner yet."""
    refs = [text for text in texts if _role(text, "reference")]
    groups = [{"id": f"group_{i:04d}", "reference_text_id": ref["id"],
               "text_ids": [ref["id"]], "edges": []} for i, ref in enumerate(refs, 1)]
    for role in ("value", "part_number"):
        fields = [text for text in texts if _role(text, role, allow_context=True)]
        if not refs or not fields:
            continue
        costs = np.full((len(fields), len(refs) + len(fields)), 2.1)
        for i, field in enumerate(fields):
            for j, ref in enumerate(refs):
                if field["id"] == ref["id"]:
                    continue
                if role == "value":
                    field_role = _role(field, role, allow_context=True)
                    prefix = _role(ref, "reference").get("prefix")
                    expected = {"R": "resistance", "C": "capacitance", "L": "inductance"}.get(prefix)
                    kind = field_role.get("value_kind")
                    if expected and kind not in {expected, "numeric"}:
                        continue
                    if prefix in {"U", "IC", "J", "P", "TP"} and kind in {"resistance", "capacitance", "inductance", "numeric", "tolerance"}:
                        continue
                scale = max(_height(field), _height(ref))
                gap = _distance(field["bbox"], ref["bbox"]) / scale
                fx, fy = _center(field["bbox"])
                rx, ry = _center(ref["bbox"])
                aligned = min(abs(field["bbox"][0] - ref["bbox"][0]),
                              abs(field["bbox"][2] - ref["bbox"][2]), abs(fx-rx)) / scale
                row_aligned = abs(fy-ry) / scale
                if gap > 2.0 or min(aligned, row_aligned) > 1.1:
                    continue
                # A reference lying between two candidate fields should not
                # pull across another reference in a dense repeated array.
                midpoint = ((fx+rx)/2, (fy+ry)/2)
                blocked = any(other["id"] != ref["id"] and
                              math.dist(_center(other["bbox"]), midpoint) < math.dist((rx, ry), midpoint) * .7
                              for other in refs)
                if blocked:
                    continue
                costs[i, j] = gap * .55 + min(aligned, row_aligned) * .5
        left, right = linear_sum_assignment(costs)
        for i, j in zip(left, right):
            if j < len(refs) and costs[i, j] < 1.9:
                groups[j]["text_ids"].append(fields[i]["id"])
                groups[j]["edges"].append({"text_id": fields[i]["id"], "role": role,
                                           "normalized_geometry_cost": float(costs[i, j])})
    return groups


def associate_structured(components, texts, *, max_distance_heights=8., ambiguity_gap=.10,
                         unmatched_cost=1.75, clamp_costs=True, numeric_unit_refs=False,
                         reference_cost_adjuster=None, allowed_text_owners=None):
    """Assign ref/value/model slots globally, keeping ambiguous texts unmatched.

    Returns independent copies. Component IDs and OCR IDs must be unique. Inputs
    use original-image top-left xyxy coordinates. Optional region_id is soft
    evidence only. OCR score is recorded, never interpreted as truth.
    Set clamp_costs=False to preserve negative compatibility rewards: clipping
    several distinct good candidates to zero can create artificial owner ties.
    allowed_text_owners optionally restricts specified OCR IDs to existing
    component IDs; the unmatched path remains available. Default is unrestricted.
    """
    if max_distance_heights <= 0 or ambiguity_gap < 0 or unmatched_cost <= 0:
        raise ValueError("Distance/cost must be positive and ambiguity gap nonnegative")
    comps = deepcopy(components)
    for i, component in enumerate(comps, 1):
        component.setdefault("id", f"component_{i:04d}")
        component.update(reference=None, value=None, part_number=None, Name=None,
                         physical_reference=None, unit=None, texts=[], association_evidence={}, review=[])
    parsed = parse_texts(texts, numeric_unit_refs=numeric_unit_refs)
    if allowed_text_owners is not None:
        component_ids={row['id'] for row in comps};text_ids={row['id'] for row in parsed}
        if set(allowed_text_owners)-text_ids or any(set(owners)-component_ids for owners in allowed_text_owners.values()):
            raise ValueError('OCR owner allowlist contains unknown IDs')
    for collection in (comps, parsed):
        if len({row["id"] for row in collection}) != len(collection):
            raise ValueError("Component and text IDs must be unique within each collection")
        for row in collection:
            box = row.get("bbox", [])
            if len(box) != 4 or not all(math.isfinite(float(v)) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("All bbox entries must be finite positive-area xyxy boxes")
    groups = build_text_groups(parsed)
    by_text = {row["id"]: row for row in parsed}
    group_for = {identifier: group for group in groups for identifier in group["text_ids"]}
    assignments = []
    assigned_ids = set()
    ref_owner = {}
    ambiguity = {}
    candidates_log = {}
    for slot in ("reference", "value", "part_number"):
        relevant = [row for row in parsed if row["id"] not in assigned_ids
                    and _role(row, slot, allow_context=True)]
        if not relevant or not comps:
            continue
        costs = np.full((len(relevant), len(comps) + len(relevant)), unmatched_cost)
        details = {}
        for i, text in enumerate(relevant):
            candidate = _role(text, slot, allow_context=True)
            scale = _height(text)
            group = group_for.get(text["id"])
            anchor = ref_owner.get(group["reference_text_id"]) if group else None
            for j, component in enumerate(comps):
                if allowed_text_owners is not None and text['id'] in allowed_text_owners and component['id'] not in allowed_text_owners[text['id']]:
                    continue
                dist = _distance(text["bbox"], component["bbox"]) / scale
                if dist > max_distance_heights:
                    continue
                ctype = _type(component)
                if candidate.get("requires_context") and (anchor != j or ctype not in NUMERIC_VALUE_TYPES):
                    continue
                tx, ty = _center(text["bbox"])
                cx, cy = _center(component["bbox"])
                center = math.hypot(tx-cx, ty-cy) / scale
                cost = .23 * dist + .015 * center + .35 * (1-candidate["heuristic_score"])
                evidence = {"distance_text_heights": dist, "center_distance_text_heights": center,
                            "text_height_pixels": scale, "ocr_score": text.get("score"),
                            "ocr_score_is_ground_truth": False, "role_reason": candidate["reason"]}
                if slot == "reference":
                    expected = PREFIX_TYPES.get(candidate.get("prefix"), set())
                    compatibility = -.30 if ctype in expected else (.30 if ctype != "component" else 0.)
                    if ctype in NO_REFERENCE_TYPES:
                        compatibility += 1.55
                    cost += compatibility
                    evidence["type_compatibility_cost"] = compatibility
                elif ctype in NO_REFERENCE_TYPES:
                    # A voltage beside a power marker is usually a net name,
                    # not a physical component's value or part number. Keep it
                    # for the net stage instead of rewarding mere proximity.
                    cost += 1.9
                    evidence["port_text_belongs_to_net_stage_cost"] = 1.9
                if slot == "value":
                    kind = candidate.get("value_kind", "other")
                    effective_type = ctype
                    # A nearby reference is additional soft evidence when the
                    # classifier abstains, not an oracle class replacement.
                    if ctype == "component" and component.get("reference"):
                        prefix = component["association_evidence"]["reference"]["candidate"].get("prefix")
                        effective_type = {"R": "resistor", "C": "capacitor", "L": "inductor", "U": "ic", "IC": "ic"}.get(prefix, ctype)
                    expected = PRIMARY_VALUE_KIND.get(effective_type)
                    if expected:
                        # Ratings remain raw OCR evidence for a future rating
                        # field, rather than filling primary resistance/C/L.
                        adjustment = -.28 if kind == expected else (1.9 if kind in {"voltage", "current", "power", "tolerance"} else (.75 if kind not in {"numeric", "other"} else 0.))
                    elif effective_type in {"ic", "opamp", "diode", "led", "bjt", "mosfet", "connector", "test_point", "switch"}:
                        adjustment = 1.35 if kind in {"resistance", "capacitance", "inductance", "tolerance"} else .55
                    else:
                        adjustment = .8 if kind == "tolerance" else 0.
                    cost += adjustment
                    evidence["value_kind"] = kind
                    evidence["primary_value_compatibility_cost"] = adjustment
                elif slot == "part_number":
                    # A generic model-like label must not beat a nearby IC just
                    # because its capacitor's outline is slightly closer.
                    adjustment = 1.0 if ctype in {"resistor", "capacitor", "inductor", "test_point"} else (-.18 if ctype in {"ic", "opamp"} else 0.)
                    cost += adjustment
                    evidence["model_type_compatibility_cost"] = adjustment
                if anchor is not None:
                    cost += -.32 if anchor == j else .55
                    evidence["text_group_anchor"] = comps[anchor]["id"]
                if text.get("region_id") is not None and component.get("region_id") is not None and text["region_id"] != component["region_id"]:
                    cost += .8
                    evidence["different_region_cost"] = .8
                costs[i, j] = max(0., cost) if clamp_costs else cost
                evidence["cost"] = float(costs[i, j])
                details[(i, j)] = evidence
            real = sorted([(float(costs[i, j]), j) for j in range(len(comps)) if (i, j) in details])
            if slot == 'reference' and reference_cost_adjuster is not None:
                adjustments = reference_cost_adjuster(text, comps, parsed, group, real, ambiguity_gap)
                for j, adjustment in adjustments.items():
                    if (i, j) not in details:
                        raise ValueError('Reference adjustment cannot expand candidate search')
                    delta = adjustment['delta']
                    if not math.isfinite(delta):
                        raise ValueError('Reference adjustment must be finite')
                    costs[i, j] += delta
                    details[(i, j)]['reference_group_adjustment'] = adjustment
                    details[(i, j)]['cost'] = float(costs[i, j])
                real = sorted([(float(costs[i, j]), j) for j in range(len(comps)) if (i, j) in details])
            candidates_log[text["id"]] = [{"component_id": comps[j]["id"], **details[(i, j)]}
                                          for _, j in real[:5]]
            if len(real) >= 2 and real[1][0] - real[0][0] < ambiguity_gap:
                # Context may resolve equal distances, but the global solver
                # must not invent evidence just because another slot is busy.
                costs[i, :len(comps)] = unmatched_cost + 1
                ambiguity[text["id"]] = "ambiguous_owner"
        left, right = linear_sum_assignment(costs)
        for i, j in zip(left, right):
            if j >= len(comps) or costs[i, j] >= unmatched_cost:
                continue
            text, component = relevant[i], comps[j]
            candidate = _role(text, slot, allow_context=True)
            component[slot] = candidate["value"]
            if slot == "part_number":
                component["Name"] = candidate["value"]
            if slot == "reference":
                component["physical_reference"] = candidate.get("physical_reference", candidate["value"])
                component["unit"] = candidate.get("unit")
                ref_owner[text["id"]] = j
            component["texts"].append(text)
            evidence = {"text_id": text["id"], "raw": text["raw"], "role": slot,
                        "candidate": candidate, **details[(i, j)]}
            component["association_evidence"][slot] = evidence
            assignments.append({"component_id": component["id"], **evidence})
            assigned_ids.add(text["id"])
    # Physical references may recur in multi-unit drawings. Preserve them and
    # flag exact duplicates; never erase the only evidence automatically.
    refs = {}
    for component in comps:
        if component["reference"]:
            refs.setdefault(component["reference"].upper(), []).append(component)
        elif _type(component) not in NO_REFERENCE_TYPES:
            component["review"].append("reference_unresolved")
    for same in refs.values():
        if len(same) > 1:
            for component in same:
                component["review"].append("repeated_reference_requires_topology_check")
    unassigned = []
    for text in parsed:
        if text["id"] in assigned_ids:
            continue
        if text["id"] in ambiguity:
            reason = ambiguity[text["id"]]
        elif text["primary_role"] in {"net", "pin", "description"}:
            reason = "retained_for_pin_net_or_description_stage"
        elif all(item.get("requires_review") for item in text["role_candidates"]):
            reason = "unverified_ocr_alternative"
        else:
            reason = "no_sufficient_owner_or_slot_conflict"
        unassigned.append({**text, "reason": reason, "owner_candidates": candidates_log.get(text["id"], [])})
    return {"schema_version": "0.1", "components": comps, "texts": parsed,
            "groups": groups, "assignments": assignments, "unassigned_texts": unassigned,
            "diagnostics": {"method": "normalized_geometry_global_slots_with_abstention",
                            "max_distance_text_heights": max_distance_heights,
                            "ambiguity_gap": ambiguity_gap, "unmatched_cost": unmatched_cost,
                            "clamp_costs": clamp_costs,
                            "role_and_owner_scores_are_uncalibrated": True,
                            "uses_ground_truth": False}}
