"""Experimental OCR-ID adjudication. Never accept generated replacement strings."""
from collections import Counter
from copy import deepcopy
import json

from .contest_components import PREFIX_PRIOR, TYPE_MAP
from .text_roles import REFERENCE


def parse_choices(content, targets):
    data = json.loads(content)
    if not isinstance(data, dict) or set(data) != {"items"} or not isinstance(data["items"], list):
        raise ValueError("Expected items object")
    expected = {t["id"]: t for t in targets}
    seen = set()
    for row in data["items"]:
        if not isinstance(row, dict) or set(row) != {"id", "text_id", "uncertain"}:
            raise ValueError("Invalid choice schema")
        ident = row["id"]
        if not isinstance(ident, str) or ident not in expected or ident in seen:
            raise ValueError("Unknown or repeated target")
        if type(row["uncertain"]) is not bool:
            raise ValueError("uncertain must be boolean")
        allowed = {c["id"] for c in expected[ident]["candidates"]}
        if row["text_id"] is not None and (not isinstance(row["text_id"], str) or row["text_id"] not in allowed):
            raise ValueError("Unknown OCR candidate")
        seen.add(ident)
    if seen != set(expected):
        raise ValueError("Missing target")
    return data["items"]


def overlaps(a, b):
    if a["text_id"] == b["text_id"]:
        return True
    pa, pb = a.get("span_provenance"), b.get("span_provenance")
    return bool(pa and pb and pa["parent_id"] == pb["parent_id"] and
                max(pa["start"], pb["start"]) < min(pa["end"], pb["end"]))


def apply_choices(baselines, targets, choices, *, empty_only=False):
    """Freeze named keys, boxes and types; reject shared-token/key conflicts.

    The only type fallback is the existing prefix prior on a newly named
    classifier-abstained component. All selected field strings come from OCR.
    """
    out = deepcopy(baselines)
    by_id = {t["id"]: t for t in targets}
    proposals, changes, blocked = [], [], []
    for choice in choices:
        target = by_id[choice["id"]]
        if choice["uncertain"]:
            continue
        candidate = next((c for c in target["candidates"] if c["id"] == choice["text_id"]), None)
        if choice["text_id"] is not None and candidate is None:
            raise ValueError("Candidate not in frozen catalog")
        field, case, key = target["field"], target["case"], target["baseline_key"]
        component = baselines[case]["components"][key]
        raw = candidate["raw"] if candidate else None
        if field == "reference":
            if raw is None:
                continue
            if not key.startswith("__unresolved_detection_"):
                blocked.append({"id": target["id"], "reason": "named_key_locked"}); continue
            if raw in baselines[case]["components"]:
                blocked.append({"id": target["id"], "reason": "key_already_exported"}); continue
        else:
            if empty_only and component[field] is not None:
                continue
            if raw == component[field]:
                continue
        if candidate and candidate.get("owner_source_id") not in {None, target["source_id"]}:
            blocked.append({"id": target["id"], "reason": "other_owner_locked"}); continue
        proposals.append({"target": target, "candidate": candidate, "raw": raw})
    bad = set()
    for i, a in enumerate(proposals):
        for j, b in enumerate(proposals[:i]):
            ta, tb = a["target"], b["target"]
            if ta["case"] != tb["case"]:
                continue
            shared = a["candidate"] and b["candidate"] and overlaps(a["candidate"], b["candidate"])
            same_key = ta["field"] == tb["field"] == "reference" and a["raw"] == b["raw"]
            if shared or same_key:
                bad.update([i, j])
    for i, proposal in enumerate(proposals):
        target, raw = proposal["target"], proposal["raw"]
        if i in bad:
            blocked.append({"id": target["id"], "reason": "cross_target_token_or_key_conflict"}); continue
        component = out[target["case"]]["components"][target["baseline_key"]]
        field = target["field"]
        before = target["baseline_key"] if field == "reference" else component[field]
        if field == "reference":
            del out[target["case"]]["components"][target["baseline_key"]]
            out[target["case"]]["components"][raw] = component
            match = REFERENCE.fullmatch(raw)
            prior = PREFIX_PRIOR.get(match.group(1).upper()) if match else None
            if component["type"] == "component" and prior:
                component["type"] = TYPE_MAP.get(prior, prior)
        else:
            component[field] = raw
        changes.append({"id": target["id"], "case": target["case"], "source_id": target["source_id"],
                        "field": field, "baseline_key": target["baseline_key"], "before": before, "after": raw,
                        "ocr_text_id": proposal["candidate"]["text_id"] if proposal["candidate"] else None})
    return out, {"changes": changes, "blocked": blocked, "empty_only": empty_only}
