"""Frozen DEV20 component ablation; model/OCR inputs have no target access."""
from collections import Counter
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.contest_components import associate_variants, export_contest
from component_lab.eval.metric import ScoreConfig, score_case, bbox_ok, text_ok
from component_lab.eval.coords import flip_bbox


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest, allowed):
    jobs = manifest.get("jobs", [])
    ids = [job["id"] for job in jobs]
    if len(jobs) != len(allowed) or len(set(ids)) != len(ids) or set(ids) != allowed or not manifest.get("complete"):
        raise ValueError("Need every frozen dev case exactly once")
    return jobs


def diagnose(pred, gt, texts, height):
    """Evaluator ONLY: no diagnostic output goes back into inference."""
    raw = {t.get("raw", t.get("text", "")) for t in texts}
    named, counts = [], Counter()
    for key, expected in gt["components"].items():
        if key.startswith("∅"):
            continue
        actual = pred["components"].get(key)
        available = key in raw
        covered = any(bbox_ok(p["bbox"], expected["bbox"], ScoreConfig()) for p in pred["components"].values())
        if actual:
            good = {"type": actual["type"] == expected["type"],
                    "bbox": bbox_ok(actual["bbox"], expected["bbox"], ScoreConfig()),
                    "text": text_ok((actual["type"], actual["Name"], actual["value"], actual["bbox"]),
                                    (expected["type"], expected.get("Name"), expected.get("value"), expected["bbox"]))}
            state = "correct" if all(good.values()) else "key_hit_other_field_error"
        else:
            good = None
            state = ("no_bbox_within_official_tolerance" if not covered else
                     "no_exact_key_in_ocr" if not available else "exact_key_available_but_not_exported")
        counts[state] += 1
        named.append({"key": key, "state": state, "key_in_ocr": available, "bbox_covered": covered,
                      "checks": good, "gt": expected, "prediction": actual})
    return {"named_state_counts": dict(counts), "named_components": named,
            "warning": "bbox coverage uses official coordinate tolerance, not proof of detector recall; states depend on ordering"}


def svg(pred, gt, source, width, height):
    elements = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"><image href="{html.escape(source)}" width="{width}" height="{height}"/>']
    for key, item in pred["components"].items():
        x1,y1,x2,y2 = flip_bbox(item["bbox"], height)
        expected = gt["components"].get(key)
        color = "#888" if key.startswith("__unresolved") else "#d52a18"
        if expected:
            checks = (item["type"] == expected["type"], bbox_ok(item["bbox"], expected["bbox"], ScoreConfig()),
                      text_ok((item["type"], item["Name"], item["value"], item["bbox"]),
                              (expected["type"], expected.get("Name"), expected.get("value"), expected["bbox"])))
            color = "#00845b" if all(checks) else "#b57000"
        title = f'{key}: {item["type"]}, Name={item["Name"]}, value={item["value"]}'
        elements.append(f'<g><title>{html.escape(title)}</title><rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width="1.8"/></g>')
    return "".join(elements) + "</svg>"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=ROOT / "outputs/component_identity_round_001_20261017/inputs")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/component_identity_round_001_20261017/experiment_v1")
    p.add_argument("--split", type=Path, default=ROOT / "configs/contest6_group_split_v1.json")
    p.add_argument("--display-names", action="store_true", help="Experimental box-only raw description Name slot")
    p.add_argument("--local-values", action="store_true", help="Experimental ref-local empty-value fill")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output; previous experiments are immutable")
    manifest = json.loads((args.inputs / "manifest.json").read_text("utf-8"))
    allowed = {c["case"] for c in json.loads(args.split.read_text("utf-8"))["partitions"]["dev"]}
    jobs = validate_manifest(manifest, allowed)
    for job in jobs:
        if not Path(job["ocr"]).exists():
            raise ValueError(f"OCR incomplete: {job['id']}")
    predictions, metadata = {}, []
    # Stage 1: freeze all variants for all cases before any target JSON reads.
    for job in jobs:
        start = time.perf_counter()
        components = json.loads(Path(job["detections"]).read_text("utf-8"))["components"]
        ocr = json.loads(Path(job["ocr"]).read_text("utf-8"))
        with Image.open(job["image"]) as image:
            width, height = image.size
        variants, edits = associate_variants(components, ocr["texts"])
        # Separate the cost correction from the IC Name export change.
        variants["unclamped"] = variants["identity_v1"]
        per_case = {}
        for name in ("baseline", "prefix", "unclamped", "identity_v1"):
            association = variants[name]
            if args.display_names and name == "identity_v1":
                from component_lab.display_names import assign_display_names
                association = assign_display_names(association)
            if args.local_values and name == "identity_v1":
                from component_lab.local_values import fill_local_values
                association = fill_local_values(association)
            pred, trace = export_contest(association, height, ic_value_as_name=name == "identity_v1",
                                        display_names=args.display_names and name == "identity_v1")
            per_case[name] = pred
            save(args.output / name / "predictions" / f"{job['id']}.json", pred)
            save(args.output / name / "traces" / f"{job['id']}.json", {"changes": edits, "components": trace})
            save(args.output / name / "associations" / f"{job['id']}.json", association)
        predictions[job["id"]] = {"variants": per_case, "texts": ocr["texts"], "width": width, "height": height, "image": job["image"]}
        metadata.append({"case": job["id"], "detector_cache_sha256": hash_file(Path(job["detections"])),
                         "ocr_cache_sha256": hash_file(Path(job["ocr"])), "association_seconds": time.perf_counter()-start})
        print(f"{job['id']}: all prediction variants frozen", flush=True)
    rows, diagnoses, gallery = [], {}, []
    for case, cached in predictions.items():
        gtfiles = list((ROOT / "data/official_contest6/train200").glob(f"{case}-*.json"))
        if len(gtfiles) != 1:
            raise ValueError(f"Need one GT: {case}")
        gt = json.loads(gtfiles[0].read_text("utf-8"))
        for name, pred in cached["variants"].items():
            diagnoses[f"{case}/{name}"] = diagnose(pred, gt, cached["texts"], cached["height"])
            for bbox in ("max_corner", "center"):
                result = score_case(pred, gt, ScoreConfig(bbox_metric=bbox))
                rows.append({"case": case, "variant": name, "bbox_metric": bbox, **result})
        pages = args.output / "gallery"
        pages.mkdir(exist_ok=True)
        source = Path(os.path.relpath(cached["image"], pages)).as_posix()
        before = svg(cached["variants"]["baseline"], gt, source, cached["width"], cached["height"])
        after = svg(cached["variants"]["identity_v1"], gt, source, cached["width"], cached["height"])
        case_rows = [r for r in rows if r["case"] == case and r["bbox_metric"] == "max_corner"]
        table = "".join(f"<tr><td>{r['variant']}</td><td>{r['component_f1']:.2%}</td><td>{html.escape(str(r['diagnostics']['components']))}</td></tr>" for r in case_rows)
        page = f'<!doctype html><meta charset="utf-8"><title>{case} component identity</title><style>body{{font-family:system-ui;margin:20px}}.compare{{display:grid;grid-template-columns:1fr 1fr}}svg{{width:100%}}td{{padding:8px}}</style><a href="../gallery.html">All cases</a><h1>{case}</h1><p>Historical-exposure DEV; component-only diagnostic. Green=full correct; amber=key hit with field errors; red=wrong key; grey=unresolved key. Colors use GT ONLY for display.</p><table>{table}</table><div class="compare"><div><h2>Baseline</h2>{before}</div><div><h2>Identity v1</h2>{after}</div></div>'
        (pages / f"{case}.html").write_text(page, "utf-8")
        gallery.append((case, case_rows))
    summaries = []
    for variant in ("baseline", "prefix", "unclamped", "identity_v1"):
        for bbox in ("max_corner", "center"):
            selected = [r for r in rows if r["variant"] == variant and r["bbox_metric"] == bbox]
            counts, diag, states = Counter(), Counter(), Counter()
            for r in selected:
                counts.update(r["counts"]["component_f1"])
                diag.update(r["diagnostics"]["components"])
                states.update(diagnoses[f"{r['case']}/{variant}"]["named_state_counts"])
            summaries.append({"variant": variant, "bbox_metric": bbox, "cases": len(selected),
                              "macro_component_f1": float(np.mean([r["component_f1"] for r in selected])),
                              "counts": dict(counts), "diagnostics": dict(diag), "named_states": dict(states)})
    pairs = [[next(r["component_f1"] for r in rows if r["case"] == case and r["variant"] == v and r["bbox_metric"] == "max_corner")
              for v in ("baseline", "identity_v1")] for case in predictions]
    differences = np.array([b-a for a,b in pairs])
    rng = np.random.default_rng(20261017)
    ci = np.percentile(differences[rng.integers(0, len(pairs), (10000,len(pairs)))].mean(axis=1), [2.5,97.5]).tolist()
    report = {"kind": "fixed_dev20_component_ablation_existing_contaminated_checkpoints",
              "api_calls": 0, "training_runs": 0, "gt_used_for_predictions": False,
              "experimental_display_names": args.display_names,
              "experimental_local_values": args.local_values,
              "split_sha256": hash_file(args.split), "summaries": summaries, "metadata": metadata,
              "per_case": rows, "diagnostics": diagnoses,
              "paired_delta_ci95": {"macro_component_f1_delta": float(differences.mean()), "case_bootstrap_ci95": ci,
                                    "limitation": "case resampling on development data, not independent confirmation; scorer uncertainty excluded"},
              "limitations": ["Checkpoints have historical exposure to this split. Not independent generalization.",
                              "Only components implemented; pins/nets empty, weighted total is not end-to-end capability.",
                              "No recovery of hidden anonymous IDs; unresolved detections retained.",
                              "Raw OCR text is exact output; absence of extracted text does not prove absence on image.",
                              "Class priors and classifier scores are uncalibrated; no new threshold search.",
                              "Named-state coverage uses official bbox protocol, not detector-only recall."]}
    save(args.output / "report.json", report)
    lines = ["# Component identity DEV20 experiment", "", report["kind"], "", "| Variant | Bbox | Macro Component F1 | TP | FP | FN | Key hits |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        c = r["counts"]
        lines.append(f"| {r['variant']} | {r['bbox_metric']} | {r['macro_component_f1']:.2%} | {c['tp']} | {c['fp']} | {c['fn']} | {r['diagnostics']['key_hit']} |")
    lines += ["", "## Paired development change", "", f"Identity minus baseline: {differences.mean():+.2%}; 95% case-bootstrap interval [{ci[0]:+.2%}, {ci[1]:+.2%}]. This is development evidence only.", "", "## Limitations", ""]
    lines += [f"- {x}" for x in report["limitations"]]
    lines += ["", "## Reproduce", "", "python scripts/run_component_identity_experiment.py --output <fresh-directory>", "", "Inputs are frozen before reading targets. See report.json for hashes, traces, all failures, and individual cases. No API calls or new training."]
    (args.output / "REPORT.md").write_text("\n".join(lines), "utf-8")
    links = "".join(f'<li><a href="gallery/{case}.html">{case}</a> — ' + "; ".join(f"{r['variant']}: {r['component_f1']:.1%}" for r in rs) + "</li>" for case,rs in gallery)
    (args.output / "gallery.html").write_text(f'<!doctype html><meta charset="utf-8"><title>DEV20 component identity</title><style>body{{font-family:system-ui;max-width:1150px;margin:30px auto}}li{{margin:12px}}</style><h1>DEV20 component identity</h1><p>Development diagnostic, existing contaminated weights. No API or training. Component-only; no pin/net score claimed.</p><pre>{html.escape(chr(10).join(lines[:14]))}</pre><ul>{links}</ul>', "utf-8")
    print(json.dumps(summaries, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
