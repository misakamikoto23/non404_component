"""E2c: an IMPLEMENTABLE abstain rule for the null protocol (no GT used).

Rule (applied to prediction type r/c only, where the scorer checks `value`):
  1. value containing '/'            -> null   (compound label such as 12pF/50V)
  2. value that is a rating only     -> null   (25V / 50V / 1A / 0.25W / 5%)
  3. everything else                 -> unchanged
Then optionally drop unresolved predictions (E1 policy) and rescore.

GT is used only for scoring. The rule is fixed before any GT is read.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.eval.metric import ScoreConfig, score_case

SPLIT = ROOT / "configs/contest6_group_split_v1.json"
PRED = ROOT / "outputs/advance_until_bottleneck_20261017/lexical_experiment/identity_v1/predictions"
GT = ROOT / "data/official_contest6/train200"
OUT = ROOT / "outputs/unattended_20261018_e2c_abstain_rule"

RATING = re.compile(r"^\d+(?:\.\d+)?\s*(?:V|A|W|%|mA|uA|kV|mV)$")
VALUE_TYPES = {"r", "c"}
UNRESOLVED = "__unresolved_detection_"


def abstain(components):
    out, changed = {}, []
    for key, comp in components.items():
        comp = dict(comp)
        value = comp.get("value")
        if value and str(comp.get("type", "")).lower() in VALUE_TYPES:
            text = str(value).strip()
            if "/" in text or RATING.fullmatch(text):
                changed.append({"key": key, "old": text})
                comp["value"] = None
        out[key] = comp
    return out, changed


def drop_unresolved(components):
    return {k: v for k, v in components.items() if not k.startswith(UNRESOLVED)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    cases = [str(e["case"] if isinstance(e, dict) else e).zfill(4) for e in split["partitions"]["dev"]]
    index = {p.name[:4]: p for p in GT.glob("*.json")}
    config = ScoreConfig()
    rows, total_changed = [], []
    for case in cases:
        pred_path = PRED / f"{case}.json"
        if not pred_path.exists() or case not in index:
            continue
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        gt = json.loads(index[case].read_text(encoding="utf-8"))
        ab, changed = abstain(pred["components"])
        total_changed.extend([{"case": case, **c} for c in changed])
        variants = {
            "v0_all": pred["components"],
            "v2_drop_unresolved": drop_unresolved(pred["components"]),
            "abstain_only": ab,
            "abstain_plus_drop": drop_unresolved(ab),
        }
        for name, comps in variants.items():
            result = score_case({"components": comps, "pins": {}, "nets": {}}, gt, config)
            c = result["counts"]["component_f1"]
            rows.append({"case": case, "variant": name, "f1": result["component_f1"],
                         "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "predicted": len(comps)})
    summary = []
    for name in sorted({r["variant"] for r in rows}):
        sel = [r for r in rows if r["variant"] == name]
        tp = sum(r["tp"] for r in sel); fp = sum(r["fp"] for r in sel); fn = sum(r["fn"] for r in sel)
        summary.append({"variant": name, "macro_f1": sum(r["f1"] for r in sel) / len(sel),
                        "tp": tp, "fp": fp, "fn": fn, "predicted": sum(r["predicted"] for r in sel)})
    base = next(s for s in summary if s["variant"] == "v0_all")
    lines = ["# E2c：可实现的「null 弃权」规则（不使用 GT）", "",
             "对 r/c 元件的 value 加两条通用规则：含 `/` 视为组合参数、纯额定值（V/A/W/%）视为非主参数；命中则输出 `null`。",
             "规则在看 GT 之前固定，GT 只用于评分。", "",
             "| 变体 | 逐图宏 F1 | TP | FP | 预测数 | 相对基线 |", "|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['variant']} | {s['macro_f1']*100:.2f}% | {s['tp']} | {s['fp']} | {s['predicted']} | "
                     f"{(s['macro_f1']-base['macro_f1'])*100:+.2f}pt |")
    lines += ["", f"规则共改动 **{len(total_changed)}** 个字段。", "",
              "## 判读", "",
              "- `abstain_only` 与 `v0_all` 的差 = 该规则单独价值；",
              "- `abstain_plus_drop` = 与 E1「删未解析」叠加后的最终可交付分数；",
              "- 若 `abstain_only` 为负，说明规则误伤了合法值，应放弃而非调参硬凑。",
              "", "## 复现", "", "```powershell", "python scripts/unattended_e2c_abstain_rule.py", "```"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "variants.json").write_text(json.dumps({"rows": rows, "summary": summary,
                                                   "changed": total_changed}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e2c_abstain_rule", "api_calls": 0,
                                                 "changed_fields": len(total_changed),
                                                 "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"summary": summary, "changed": len(total_changed)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
