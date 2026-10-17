"""Export cached model/OCR outputs, then score them; no GT enters export.

Development diagnostic only: cached cases were previously used in experiments.
Pins/nets are empty because this pilot evaluates the component stage only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.eval.metric import ScoreConfig, score_case

# Explicit output vocabulary adapter, not a GT-derived per-case correction.
# Broad/ambiguous classes stay unresolved instead of inventing polarity/subtype.
TYPE_MAP = {
    "resistor": "r", "capacitor": "c", "inductor": "l",
    "ground": "gnd", "ic_or_module": "box", "diode": "d",
    "led": "led", "connector": "connector", "motor": "motor",
    "switch": "switch", "testpoint": "testpoint", "crystal": "crystal",
}


def raw_field(component, field):
    """Only emit OCR evidence verbatim, never the parser's normalized value."""
    evidence = component.get("association_evidence", {}).get(field, {})
    return evidence.get("raw") if isinstance(evidence.get("raw"), str) else None


def export_components(association, height):
    """Pure prediction-side adapter. No labels, matching or inferred hidden IDs."""
    output = {"components": {}, "pins": {}, "nets": {}}
    trace, issues = [], Counter()
    references = Counter(raw_field(row, "reference") for row in association["components"])
    for index, component in enumerate(association["components"], 1):
        reference = raw_field(component, "reference")
        if reference and references[reference] == 1:
            key = reference
        else:
            # Retain every detection, so unidentified objects still count as FP.
            key = f"__unresolved_detection_{index:04d}"
            issues["missing_reference" if not reference else "duplicate_reference"] += 1
        while key in output["components"]:
            key += "_"
        raw_type = component.get("type", "component")
        exported_type = TYPE_MAP.get(raw_type, raw_type)
        if raw_type not in TYPE_MAP:
            issues["unresolved_type"] += 1
        x1, y1, x2, y2 = component["bbox"]
        if not (0 <= y1 <= y2 <= height):
            raise ValueError("cached bbox outside source image")
        output["components"][key] = {
            "type": exported_type,
            "bbox": [x1, height-y2, x2, height-y1],
            "Name": raw_field(component, "part_number"),
            "value": raw_field(component, "value"),
        }
        trace.append({"prediction_key": key, "source_id": component.get("id"),
                      "raw_type": raw_type, "exported_type": exported_type,
                      "reference": reference, "association_evidence": component.get("association_evidence", {})})
    return output, {"issues": dict(issues), "components": trace}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/unattended_20261016/ocr_association/rapidocr")
    parser.add_argument("--gt", type=Path, default=ROOT / "data/official_contest6/train200")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/contest6_cached_component_pilot_20261017")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; existing experiments are immutable")
    associations = sorted(args.cache.glob("*/association.json"))
    if not associations:
        raise ValueError("No association caches found")
    predictions, provenance = [], []
    # Export ALL predictions before reading any GT, enforcing the boundary.
    for path in associations:
        case = path.parent.name.split("_")[-1]
        with Image.open(path.with_name("source.png")) as image:
            height = image.height
        prediction, trace = export_components(json.loads(path.read_text("utf-8")), height)
        predictions.append((case, prediction))
        target = args.output / "predictions" / f"{case}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), "utf-8")
        target = args.output / "traces" / f"{case}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(trace, ensure_ascii=False, indent=2), "utf-8")
        provenance.append({"case": case, "cache": str(path.resolve()), "cache_sha256": sha256(path),
                           "source_sha256": sha256(path.with_name("source.png")), "height": height})
    results = []
    for case, prediction in predictions:
        files = list(args.gt.glob(f"{case}-*.json"))
        if len(files) != 1:
            raise ValueError(f"Need exactly one GT for {case}")
        gt = json.loads(files[0].read_text("utf-8"))
        for anon in ("strict", "observable_diagnostic"):
            for bbox in ("max_corner", "center"):
                result = score_case(prediction, gt, ScoreConfig(anon_policy=anon, bbox_metric=bbox))
                results.append({"case": case, "anon_policy": anon, "bbox_metric": bbox, **result})
    groups = []
    for anon in ("strict", "observable_diagnostic"):
        for bbox in ("max_corner", "center"):
            rows = [r for r in results if r["anon_policy"] == anon and r["bbox_metric"] == bbox]
            counts = Counter()
            diagnostics = Counter()
            for row in rows:
                counts.update(row["counts"]["component_f1"])
                diagnostics.update(row["diagnostics"]["components"])
            groups.append({"anon_policy": anon, "bbox_metric": bbox, "cases": len(rows),
                           "macro_component_f1": sum(r["component_f1"] for r in rows)/len(rows),
                           "component_counts": dict(counts), "component_diagnostics": dict(diagnostics)})
    report = {"kind": "cached_component_development_diagnostic_NOT_independent_test",
              "paid_api_calls": 0, "new_training": False, "gt_used_for_export": False,
              "type_adapter": TYPE_MAP, "provenance": provenance, "groups": groups, "per_case": results,
              "limitations": ["Historical model-selection exposure; ten cached cases only, not the full dev split.",
                              "No pin or net prediction. Weighted score is not an end-to-end system result.",
                              "Unknown keys retained as unresolved FPs; no attempt to recover hidden anonymous IDs.",
                              "Existing classifier is a weak-label baseline; no prefix override or library lookup.",
                              "Absent field evidence is null; this does not establish that text is absent on the image.",
                              "Existing association candidates were normalized internally; final fields use raw evidence only."]}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    lines = ["# Cached component pilot", "", report["kind"], "", "| Anonymous policy | Bbox | Cases | Macro Component F1 | TP | FP | FN |", "|---|---|---:|---:|---:|---:|---:|"]
    for group in groups:
        c = group["component_counts"]
        lines.append(f"| {group['anon_policy']} | {group['bbox_metric']} | {group['cases']} | {group['macro_component_f1']:.2%} | {c['tp']} | {c['fp']} | {c['fn']} |")
    lines += ["", "## Limitations", ""] + [f"- {s}" for s in report["limitations"]]
    lines += ["", "No paid API calls, no training, no changes to source labels/caches. Details: report.json and traces/."]
    (args.output / "REPORT.md").write_text("\n".join(lines), "utf-8")
    print(json.dumps(groups, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
