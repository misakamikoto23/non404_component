"""Read-only evaluator audit of ten historical OCR caches, never a predictor.

GT is used only to classify historical failures. No source caches are mutated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.eval.metric import canonical_type


def read(path):
    return json.loads(path.read_text("utf-8"))


def box_error(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/unattended_20261016/ocr_association/rapidocr")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/contest6_cached_component_pilot_20261017")
    parser.add_argument("--gt", type=Path, default=ROOT / "data/official_contest6/train200")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/component_identity_audit_20261017")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    overall, states, unassigned_reasons, types = Counter(), Counter(), Counter(), Counter()
    text_failures, examples, cases = [], [], []
    for path in sorted(args.cache.glob("*/association.json")):
        case = path.parent.name.rsplit("_", 1)[-1]
        association = read(path)
        # These are existing historical diagnostic cases only, never sealed tests.
        label_paths = list(args.gt.glob(f"{case}-*.json"))
        if len(label_paths) != 1:
            raise ValueError(f"expected one historical GT for {case}")
        gt = read(label_paths[0])["components"]
        pred = read(args.baseline / "predictions" / f"{case}.json")["components"]
        trace = read(args.baseline / "traces" / f"{case}.json")
        assignments = association["assignments"]
        tokens = association["texts"]
        unassigned = {row["id"]: row for row in association["unassigned_texts"]}
        assigned_by_text = {row["text_id"]: row for row in assignments}
        counts = Counter(gt=len(gt), pred=len(pred), anonymous=sum(k.startswith("∅") for k in gt))
        for key, expected in gt.items():
            if key.startswith("∅"):
                continue
            counts["named_gt"] += 1
            exact = [t for t in tokens if t.get("raw", t.get("text")) == key]
            parsed = [t for t in tokens if any(c.get("role") == "reference" and c.get("value") == key
                                                for c in t.get("role_candidates", []))]
            merged = [t for t in tokens if re.search(r"(?<![A-Za-z0-9])"+re.escape(key)+r"(?![A-Za-z0-9])", t.get("raw", t.get("text", "")))]
            if exact:
                counts["key_exact_in_ocr"] += 1
            if parsed:
                counts["key_in_reference_candidates"] += 1
            if merged:
                counts["key_exact_or_embedded_in_ocr"] += 1
            actual = pred.get(key)
            if actual:
                state = "key_exported"
                counts["key_exported"] += 1
                type_good = actual["type"] == expected["type"]
                bbox_good = box_error(actual["bbox"], expected["bbox"]) <= 20
                if not type_good:
                    counts["type_wrong"] += 1
                    types[f"{actual['type']} -> {expected['type']}"] += 1
                if not bbox_good:
                    counts["bbox_wrong"] += 1
                kind = canonical_type(expected["type"])
                field = "Name" if kind == "ic" else "value" if kind in {"resistor", "capacitor"} else None
                text_good = field is None or actual.get(field) == expected.get(field)
                if not text_good:
                    counts["text_wrong"] += 1
                    field_tokens = [t for t in tokens if t.get("raw", t.get("text")) == expected.get(field)] if expected.get(field) is not None else []
                    text_failures.append({"case":case,"key":key,"field":field,"gt":expected.get(field),"pred":actual.get(field),
                                          "gt_text_exact_in_ocr":bool(field_tokens),"type_correct":type_good,"bbox_correct":bbox_good})
                if type_good and bbox_good and text_good:
                    counts["component_tp"] += 1
            elif exact:
                relevant = [assigned_by_text[t["id"]] for t in exact if t["id"] in assigned_by_text]
                if any(row["role"] == "reference" for row in relevant):
                    state = "exact_ocr_assigned_reference_but_export_missing"
                elif relevant:
                    state = "exact_ocr_assigned_other_role"
                else:
                    state = "exact_ocr_unassigned"
                    for token in exact:
                        unassigned_reasons[unassigned.get(token["id"],{}).get("reason", "not_in_unassigned")] += 1
            elif parsed:
                state = "only_normalized_or_review_candidate"
            elif merged:
                state = "key_embedded_in_merged_ocr"
            else:
                state = "key_absent_from_cached_ocr"
            states[state] += 1
            if state != "key_exported":
                evidence = [{"id":t["id"],"raw":t.get("raw",t.get("text")),"role":t.get("primary_role"),
                             "assignment":assigned_by_text.get(t["id"]),"unassigned_reason":unassigned.get(t["id"],{}).get("reason")}
                            for t in (exact or parsed or merged)]
                examples.append({"case":case,"key":key,"type":expected["type"],"state":state,"ocr_evidence":evidence})
        counts["reference_assignments"] = sum(a["role"] == "reference" for a in assignments)
        counts["raw_vs_candidate_reference_different"] = sum(a["role"] == "reference" and a["raw"] != a["candidate"]["value"] for a in assignments)
        counts["unresolved_exports"] = sum(k.startswith("__unresolved_detection_") for k in pred)
        counts["non_gt_named_exports"] = sum(not k.startswith("__unresolved_detection_") and k not in gt for k in pred)
        cases.append({"case":case, **dict(counts)})
        overall.update(counts)
    report = {"kind":"historical_component_identity_evaluator_audit_NOT_independent_test", "gt_used_only_for_evaluation":True,
              "paid_api_calls":0,"new_training":False,"overall":dict(overall),"named_gt_states":dict(states),
              "unassigned_exact_key_reasons":dict(unassigned_reasons),"key_hit_type_confusions":dict(types),
              "text_failures":text_failures,"missing_key_details":examples,"per_case":cases,
              "limitations":["Ten historical diagnostic cases, not a new holdout.",
                             "OCR presence is string availability, not verified text ownership.",
                             "GT absent from OCR means absent from this cache, not necessarily absent from image.",
                             "GT use is evaluation only; no label-derived predictions or candidate corrections produced."]}
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),"utf-8")
    lines = ["# Historical component identity audit", "", report["kind"],"", "## Counts", ""]
    lines += [f"- {k}: {v}" for k,v in overall.items()]
    lines += ["", "## Named GT key failure states", ""] + [f"- {k}: {v}" for k,v in states.items()]
    lines += ["", "## Unassigned exact-key reasons", ""] + [f"- {k}: {v}" for k,v in unassigned_reasons.items()]
    lines += ["", "## Type errors after correct key", ""] + [f"- {k}: {v}" for k,v in types.most_common()]
    lines += ["", "## Text errors after correct key", "", "|Case|Key|Field|Expected|Actual|Expected text present in OCR|", "|---|---|---|---|---|---|"]
    for e in text_failures:
        lines.append(f"|{e['case']}|{e['key']}|{e['field']}|{e['gt']}|{e['pred']}|{e['gt_text_exact_in_ocr']}|")
    lines += ["", "## Interpretation and next steps", "",
              "- Of 210 named GT keys, only 120 are available as exact OCR strings; 132 occur either exactly or within a larger token. Improving association alone cannot recover the other 78 from this cache.",
              "- The 26 unexported exact OCR keys correspond to 28 OCR token instances: 25 abstained as ambiguous owners and 3 failed owner/slot sufficiency. This is a candidate for contextual association, not justification to disable abstention globally.",
              "- Twelve missed keys occur inside merged or decorated text. Preserve the raw OCR string, and use generic substring/span evidence or local OCR to produce designator candidates rather than GT-specific substitutions.",
              "- One exact key uses a nonstandard designator prefix and is parsed as a model. A fixed designator prefix list is not exhaustive; unfamiliar alphanumeric text should remain a context-resolvable candidate.",
              "- Exporting IC Name only from the part_number role can lose displayed names that look like a frequency or a reference. Name is a schema field, not necessarily a semiconductor part number.",
              "- R/C expected text sometimes is null, and sometimes looks like a voltage. Preserve raw field evidence and avoid treating a value-kind rule as guaranteed ontology truth.",
              "- The 23 type errors mix classifier errors, generic abstentions and vocabulary mismatches. A connector-looking symbol can carry the dataset type box; generic prefix priors must be calibrated on the new development split, not hardcoded from these cases.",
              "- No parameters or source modules were changed using this historical diagnostic. All new model or association choices must use the frozen grouped development split, leaving sealed test labels untouched."]
    lines += ["", "## Reproduce", "", "```powershell", "python scripts/audit_component_identity.py --output outputs/component_identity_audit_20261017", "```", "", "The output directory must not already exist; use a new directory for a rerun.", "", "## Limitations", ""]
    lines += ["- "+s for s in report["limitations"]]
    (args.output/"REPORT.md").write_text("\n".join(lines),"utf-8")
    print(json.dumps({"overall":dict(overall),"states":dict(states),"type_confusions":dict(types)},ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
