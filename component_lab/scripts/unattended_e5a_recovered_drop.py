"""E5a: apply the drop-unresolved export policy to the recovered-identity set.

The recovered set (`unresolved_identity_recovery_20261017/cache_plus_new_keys`)
is the candidate production configuration (17.77% / 177TP as exported today).
This measures what it scores once unresolved entries are not submitted.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.eval.metric import ScoreConfig, score_case

SPLIT = ROOT / "configs/contest6_group_split_v1.json"
PRED = ROOT / "outputs/unresolved_identity_recovery_20261017/cache_plus_new_keys/predictions"
GT = ROOT / "data/official_contest6/train200"
OUT = ROOT / "outputs/unattended_20261018_e5a_recovered_drop"
UNRESOLVED = "__unresolved_detection_"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    cases = [str(e["case"] if isinstance(e, dict) else e).zfill(4) for e in split["partitions"]["dev"]]
    index = {p.name[:4]: p for p in GT.glob("*.json")}
    config = ScoreConfig()
    rows = []
    for case in cases:
        pred_path = PRED / f"{case}.json"
        if not pred_path.exists() or case not in index:
            continue
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        gt = json.loads(index[case].read_text(encoding="utf-8"))
        named = {k: v for k, v in pred["components"].items() if not k.startswith(UNRESOLVED)}
        for name, comps in {"v0_all": pred["components"], "v1_drop_unresolved": named}.items():
            result = score_case({"components": comps, "pins": {}, "nets": {}}, gt, config)
            c = result["counts"]["component_f1"]
            rows.append({"case": case, "variant": name, "f1": result["component_f1"],
                         "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "predicted": len(comps)})
    summary = []
    for name in ("v0_all", "v1_drop_unresolved"):
        sel = [r for r in rows if r["variant"] == name]
        tp = sum(r["tp"] for r in sel); fp = sum(r["fp"] for r in sel); fn = sum(r["fn"] for r in sel)
        summary.append({"variant": name, "macro_f1": sum(r["f1"] for r in sel) / len(sel),
                        "micro_f1": 2 * tp / (2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn,
                        "predicted": sum(r["predicted"] for r in sel)})
    v0, v1 = summary
    lines = ["# E5a：恢复身份后的正式候选配置，再叠加「不提交未解析」", "",
             "预测集：`unresolved_identity_recovery_20261017/cache_plus_new_keys/predictions`（这是当前最优、已固化的恢复结果）。",
             "只改导出策略（是否提交未解析条目），不改任何字段。", "",
             "| 导出策略 | 逐图宏 F1 | micro F1 | TP | FP | FN | 提交条目 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['variant']} | {s['macro_f1']*100:.2f}% | {s['micro_f1']*100:.2f}% | {s['tp']} | {s['fp']} | {s['fn']} | {s['predicted']} |")
    lines += ["", "## 结论", "",
              f"- 未解析条目不提交后，宏 F1 从 {v0['macro_f1']*100:.2f}% 升到 **{v1['macro_f1']*100:.2f}%**"
              f"（**+{(v1['macro_f1']-v0['macro_f1'])*100:.2f} 个百分点**），TP 由 {v0['tp']} → {v1['tp']}。",
              f"- 提交条目从 {v0['predicted']} 降到 {v1['predicted']}，精确率由 "
              f"{v0['tp']/v0['predicted']*100:.1f}% 升到 {v1['tp']/v1['predicted']*100:.1f}%。",
              "- 这是当前**可直接交付的最佳配置**的分数；不涉及训练、API 或标签修改。",
              "", "## 复现", "", "```powershell", "python scripts/unattended_e5a_recovered_drop.py", "```"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "variants.json").write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e5a_recovered_drop", "api_calls": 0,
                                                 "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
