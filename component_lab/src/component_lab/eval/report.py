"""Batch scoring over prediction/GT directories, CSV and Markdown reports."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import random

from .cases import discover, load_split, source_tag
from .metric import ANON_PREFIX, ORACLE_STAGES, WEIGHTS, ScoreConfig, score_case

METRICS = ("score", *WEIGHTS)
GROUP_KEYS = ("type_policy", "anon_policy", "oracle")


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return exc


def _gt_files(gt_dir, split, partition):
    gt_files = discover(gt_dir)
    if split is None:
        return gt_files
    wanted = load_split(split, partition)
    missing = [case for case in wanted if case not in gt_files]
    if missing:
        raise ValueError(f"split cases absent from GT dir: {missing}")
    return {case: gt_files[case] for case in wanted}


def score_directory(pred_dir, gt_dir, configs, split=None, partition=None):
    """Score every GT case (optionally restricted to a split) under each config.

    Returns rows: one per (config, case). GT cases without a prediction score 0.
    """
    gt_files = _gt_files(gt_dir, split, partition)
    pred_files = discover(pred_dir)
    rows = []
    for case in sorted(gt_files, key=lambda c: (len(c), c)):
        ground_truth = _read(gt_files[case])
        if isinstance(ground_truth, Exception):
            raise ValueError(f"unreadable GT {gt_files[case]}: {ground_truth}")
        prediction = _read(pred_files[case]) if case in pred_files else None
        for config in configs:
            if isinstance(prediction, Exception):
                result = score_case({}, ground_truth, config)
                result.update(status="invalid", reason=f"unreadable JSON: {prediction}")
            else:
                result = score_case(prediction, ground_truth, config)
            rows.append({"case": case, "stem": gt_files[case].stem, "tag": source_tag(gt_files[case].stem),
                         "type_policy": config.type_policy, "anon_policy": config.anon_policy,
                         "oracle": config.oracle_label, "bbox_metric": config.bbox_metric,
                         "net_tiebreak": config.net_tiebreak, **result})
    return rows


def anonymous_statistics(gt_dir, split=None, partition=None):
    """Count ∅ objects in the scored GT so reports show what observable_diagnostic removes."""
    by_type, totals = Counter(), Counter()
    for path in _gt_files(gt_dir, split, partition).values():
        case = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        for key, item in case["components"].items():
            totals["components"] += 1
            if key.startswith(ANON_PREFIX):
                totals["anon_components"] += 1
                by_type[item["type"]] += 1
        for key, pins in case["pins"].items():
            totals["pins"] += len(pins)
            totals["anon_pins"] += len(pins) if key.startswith(ANON_PREFIX) else 0
        for net in case["nets"].values():
            members = [m for m in net["hyperGraph"].strip("()").split(",") if m]
            totals["net_members"] += len(members)
            totals["anon_net_members"] += sum(m.startswith(ANON_PREFIX) for m in members)
    return {"totals": dict(totals), "anon_components_by_type": dict(by_type.most_common())}


def build_configs(anon_policies, bbox_metric, bbox_tolerance, point_tolerance, oracle, ablate,
                  net_tiebreak="segments", type_policies=("strict",)):
    oracle_sets = [frozenset(oracle)]
    if ablate:
        oracle_sets = [frozenset()] + [frozenset([stage]) for stage in ORACLE_STAGES]
    return [ScoreConfig(bbox_tolerance=bbox_tolerance, bbox_metric=bbox_metric, point_tolerance=point_tolerance,
                        anon_policy=policy, type_policy=types, net_tiebreak=net_tiebreak, oracle=stages)
            for types in type_policies for policy in anon_policies for stages in oracle_sets]


def _flatten(row):
    flat = {k: row[k] for k in ("case", "stem", "tag", "type_policy", "anon_policy", "oracle", "bbox_metric",
                                "net_tiebreak", "status", "reason")}
    flat.update({k: round(row[k], 6) for k in METRICS})
    for metric, counts in row["counts"].items():
        for name, value in counts.items():
            flat[f"{metric.removesuffix('_f1')}_{name}"] = value
    for section, values in row["diagnostics"].items():
        for name, value in values.items():
            flat[f"diag_{section}_{name}"] = value
    flat["gt_unmapped_types"] = " ".join(row["types"]["gt_unmapped"])
    flat["pred_unmapped_types"] = " ".join(row["types"]["pred_unmapped"])
    return flat


def write_csv(rows, path):
    flat = [_flatten(row) for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]) if flat else ["case"])
        writer.writeheader()
        writer.writerows(flat)


def _groups(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return groups


def bootstrap_ci(values, resamples=10000, seed=0, level=.95):
    """Percentile bootstrap CI of the mean, resampling cases with replacement."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    low = means[int((1 - level) / 2 * resamples)]
    high = means[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return (low, high)


def summarize(rows, resamples=10000):
    """Mean of each metric over cases (missing/invalid cases count as 0) with a 95% case bootstrap CI."""
    out = []
    for key, group in _groups(rows, lambda r: tuple(r[k] for k in GROUP_KEYS)).items():
        entry = dict(zip(GROUP_KEYS, key))
        entry.update(cases=len(group), failed=sum(r["status"] != "ok" for r in group))
        entry.update({m: sum(r[m] for r in group) / len(group) for m in METRICS})
        entry["score_ci95"] = bootstrap_ci([r["score"] for r in group], resamples)
        out.append(entry)
    return out


def _table(header, lines):
    text = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    text += ["| " + " | ".join(str(c) for c in line) + " |" for line in lines]
    return "\n".join(text)


def markdown(rows, configs, pred_dir, gt_dir, split, partition=None, anon_stats=None):
    summary = summarize(rows)
    base = {(e["type_policy"], e["anon_policy"]): e for e in summary if e["oracle"] == "none"}
    lines = []
    for e in summary:
        delta = ""
        reference = base.get((e["type_policy"], e["anon_policy"]))
        if e["oracle"] != "none" and reference:
            delta = f"{e['score'] - reference['score']:+.2f}"
        low, high = e["score_ci95"]
        label = lambda value, diagnostic: value + (" (diagnostic)" if value == diagnostic else "")
        lines.append([label(e["type_policy"], "mapped"), label(e["anon_policy"], "observable_diagnostic"),
                      e["oracle"], e["cases"], e["failed"], f"{e['score']:.2f}", f"[{low:.2f}, {high:.2f}]", delta,
                      *(f"{100 * e[m]:.1f}" for m in WEIGHTS)])
    headers = sorted({c.header() for c in configs})
    parts = [
        "# Contest 6 local score",
        "",
        f"- pred: `{pred_dir}`\n- gt: `{gt_dir}`\n- split: `{split or 'all GT cases'}`"
        + (f" partition `{partition}`" if partition else ""),
        "- Local re-implementation of the public scoring rules; not the official evaluator.",
        "- Headline = type_policy strict + anon_policy strict + no oracle. Rows marked (diagnostic) are not the contest metric.",
        "- Configurations scored:",
        *[f"  - `{h}`" for h in headers],
        "",
        _table(["type policy", "anon policy", "substitution oracle", "cases", "failed", "score", "95% CI (case bootstrap)",
                "delta vs none", "Component", "Pin", "NetHypergraph", "NetLine"], lines),
        "",
        "## By source tag (no oracle)",
        "",
    ]
    tag_lines = []
    for (types, policy, tag), group in sorted(_groups([r for r in rows if r["oracle"] == "none"],
                                                      lambda r: (r["type_policy"], r["anon_policy"], r["tag"])).items()):
        tag_lines.append([types, policy, tag, len(group), f"{sum(r['score'] for r in group) / len(group):.2f}",
                          *(f"{100 * sum(r[m] for r in group) / len(group):.1f}" for m in WEIGHTS)])
    parts += [_table(["type policy", "anon policy", "tag", "cases", "score", "Component", "Pin", "NetHypergraph",
                      "NetLine"], tag_lines), ""]
    gt_unmapped = sorted({t for r in rows for t in r["types"]["gt_unmapped"]})
    pred_unmapped = sorted({t for r in rows for t in r["types"]["pred_unmapped"]})
    parts += ["## Types not covered by the type vocabulary", "",
              f"- GT: {', '.join(gt_unmapped) or 'none'}", f"- prediction: {', '.join(pred_unmapped) or 'none'}", ""]
    if anon_stats:
        totals = anon_stats["totals"]
        parts += ["## ∅ (anonymous-key) objects in the scored GT", "",
                  f"- components {totals.get('anon_components', 0)}/{totals.get('components', 0)}, "
                  f"pins {totals.get('anon_pins', 0)}/{totals.get('pins', 0)}, "
                  f"net members {totals.get('anon_net_members', 0)}/{totals.get('net_members', 0)}",
                  "- ∅ components by type: " + ", ".join(f"{t} {n}" for t, n in anon_stats["anon_components_by_type"].items()),
                  ""]
    parts += ["## Assumptions", ""]
    parts += [f"- **{k}**: {v}" for k, v in configs[0].assumptions().items()
              if k not in ("oracle", "anon_policy", "type_policy")]
    for field in ("type_policy", "anon_policy"):
        parts += [f"- **{field}={getattr(c, field)}**: {c.assumptions()[field]}"
                  for c in {getattr(c, field): c for c in configs}.values()]
    parts += ["- **substitution oracle**: replaces that section of each valid prediction's final JSON with GT; "
              "later stages are not rerun, so the delta is the score lost in that section's own fields, not the "
              "end-to-end gain a better upstream stage would bring (that needs a pipeline oracle: rerun pins/nets "
              "from GT components). Missing/invalid cases stay 0.",
              "- **95% CI**: percentile bootstrap over cases (10,000 resamples, seed 0); it covers case sampling only, "
              "not scorer-assumption uncertainty."]
    return "\n".join(parts) + "\n"


def run(pred_dir, gt_dir, configs, split=None, out=None, partition=None):
    rows = score_directory(pred_dir, gt_dir, configs, split, partition)
    anon_stats = anonymous_statistics(gt_dir, split, partition)
    report = markdown(rows, configs, pred_dir, gt_dir, split, partition, anon_stats)
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "per_case.csv")
        (out / "summary.md").write_text(report, encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(
            {"summary": summarize(rows), "configs": sorted({c.header() for c in configs}),
             "assumptions": configs[0].assumptions(), "anonymous_gt": anon_stats,
             "split": str(split) if split else None, "partition": partition},
            ensure_ascii=False, indent=1), encoding="utf-8")
    return rows, report
