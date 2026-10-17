"""E1: does dropping unmatchable predictions raise strict ComponentF1 at no recall cost?

Frozen predictions are never modified. Variants only delete entries, then the
official-rule scorer re-runs. No GT is used to choose what to delete.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from component_lab.eval.metric import ScoreConfig, score_case

SPLIT = ROOT / "configs/contest6_group_split_v1.json"
PRED = ROOT / "outputs/advance_until_bottleneck_20261017/lexical_experiment/identity_v1/predictions"
GT = ROOT / "data/official_contest6/train200"
OUT = ROOT / "outputs/unattended_20261018_e1_drop_unresolved"

UNRESOLVED = "__unresolved_detection_"


def dev_cases():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    out = []
    for entry in split["partitions"]["dev"]:
        case = entry["case"] if isinstance(entry, dict) else entry
        out.append(str(case).zfill(4) if str(case).isdigit() else str(case))
    return out


def gt_index():
    index = {}
    for path in GT.glob("*.json"):
        match = re.match(r"(\d{4})", path.name)
        if match:
            index[match.group(1)] = path
    return index


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def drop_unresolved(pred):
    return {k: v for k, v in pred["components"].items() if not k.startswith(UNRESOLVED)}


def drop_untyped(components):
    return {k: v for k, v in components.items() if str(v.get("type", "")).lower() != "component"}


def variants(pred):
    allc = pred["components"]
    named = drop_unresolved(pred)
    return {
        "v0_all": dict(allc),
        "v1_drop_unresolved": named,
        "v2_drop_unresolved_and_untyped": drop_untyped(named),
        "v3_drop_untyped_only": drop_untyped(allc),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    cases = dev_cases()
    gts = gt_index()
    config = ScoreConfig()
    rows, missing = [], []
    for case in cases:
        pred_path = PRED / f"{case}.json"
        if not pred_path.exists() or case not in gts:
            missing.append(case)
            continue
        pred = load(pred_path)
        gt = load(gts[case])
        for name, components in variants(pred).items():
            view = {"components": components, "pins": {}, "nets": {}}
            result = score_case(view, gt, config)
            counts = result["counts"]["component_f1"]
            rows.append({"case": case, "variant": name, "f1": result["component_f1"],
                         "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
                         "predicted": len(components)})
    names = sorted({r["variant"] for r in rows})
    summary = []
    for name in names:
        sel = [r for r in rows if r["variant"] == name]
        tp = sum(r["tp"] for r in sel); fp = sum(r["fp"] for r in sel); fn = sum(r["fn"] for r in sel)
        macro = sum(r["f1"] for r in sel) / len(sel)
        micro = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        summary.append({"variant": name, "macro_f1": macro, "micro_f1": micro, "tp": tp, "fp": fp,
                        "fn": fn, "predicted": sum(r["predicted"] for r in sel),
                        "cases": len(sel)})
    (OUT / "variants.json").write_text(json.dumps({"rows": rows, "summary": summary,
                                                   "missing": missing, "config": str(config)},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    base = next(s for s in summary if s["variant"] == "v0_all")
    lines = ["# E1 少报换精确：删除不可命中预测的操作点扫描", "",
             "固定 dev20、冻结预测（`lexical_experiment/identity_v1/predictions`）、官方规则评分（strict type / strict anon / max_corner 20px）。",
             "变体只**删除**预测条目，不新增、不改字段；不读 GT 决定删谁。", "",
             "| 变体 | 逐图宏 F1 | micro F1 | TP | FP | FN | 预测数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['variant']} | {s['macro_f1']*100:.2f}% | {s['micro_f1']*100:.2f}% | {s['tp']} | {s['fp']} | {s['fn']} | {s['predicted']} |")
    lines += ["", f"基线 v0_all 宏 F1 {base['macro_f1']*100:.2f}%。"]
    if missing:
        lines += ["", f"缺失 case: {missing}"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({
        "experiment": "e1_drop_unresolved", "api_calls": 0, "training": 0,
        "cases": len(cases), "missing": missing, "summary": summary}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
