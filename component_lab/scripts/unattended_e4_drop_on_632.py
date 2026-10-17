"""E4: does the drop-unresolved gain hold on the second baseline (632 boxes)?

`fullpage_component_identity_20261017/baselines.json` is a different detector
(37-class short-trained, whole image, 632 boxes). Same variants, same scorer.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.eval.metric import ScoreConfig, score_case

BASE = ROOT / "outputs/fullpage_component_identity_20261017/baselines.json"
GT = ROOT / "data/official_contest6/train200"
OUT = ROOT / "outputs/unattended_20261018_e4_drop_on_632"
UNRESOLVED = "__unresolved_detection_"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE.read_text(encoding="utf-8"))
    index = {}
    for path in GT.glob("*.json"):
        index[path.name[:4]] = path
    config = ScoreConfig()
    rows = []
    for case, payload in sorted(base.items()):
        gt_path = index.get(case)
        if gt_path is None:
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        variants = {"v0_all": dict(payload["components"])}
        named = {k: v for k, v in payload["components"].items() if not k.startswith(UNRESOLVED)}
        variants["v1_drop_unresolved"] = named
        variants["v2_drop_unresolved_and_untyped"] = {
            k: v for k, v in named.items() if str(v.get("type", "")).lower() != "component"}
        for name, comps in variants.items():
            result = score_case({"components": comps, "pins": {}, "nets": {}}, gt, config)
            counts = result["counts"]["component_f1"]
            rows.append({"case": case, "variant": name, "f1": result["component_f1"],
                         "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "predicted": len(comps)})
    summary = []
    for name in sorted({r["variant"] for r in rows}):
        sel = [r for r in rows if r["variant"] == name]
        tp = sum(r["tp"] for r in sel); fp = sum(r["fp"] for r in sel); fn = sum(r["fn"] for r in sel)
        summary.append({"variant": name, "macro_f1": sum(r["f1"] for r in sel) / len(sel),
                        "tp": tp, "fp": fp, "fn": fn, "predicted": sum(r["predicted"] for r in sel)})
    lines = ["# E4：删未解析策略在第二个基线（632 框）上是否同样有效", "",
             "基线来自 `fullpage_component_identity_20261017/baselines.json`（37 类短训整图检测器，dev20 共 632 框）。",
             "评分口径与 E1 完全一致（strict type / strict anon / max-corner 20px / 逐图宏平均）。", "",
             "| 变体 | 逐图宏 F1 | TP | FP | FN | 预测数 |", "|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['variant']} | {s['macro_f1']*100:.2f}% | {s['tp']} | {s['fp']} | {s['fn']} | {s['predicted']} |")
    best = max(summary, key=lambda s: s["macro_f1"]); worst = min(summary, key=lambda s: s["macro_f1"])
    lines += ["", "## 结论", "",
              f"- 该基线上，最优变体比最差变体高 **{(best['macro_f1']-worst['macro_f1'])*100:.2f} 个百分点**，"
              f"TP 变化 {worst['tp']}→{best['tp']}。",
              "- 与 E1 一致：删除无法命名的预测只减少 FP，不损伤 TP（或损伤极小）。" if best["tp"] >= worst["tp"] - 2
              else "- 注意：该基线上删除未解析预测会损失部分 TP，说明结论依赖检测器。",
              "", "## 复现", "", "```powershell", "python scripts/unattended_e4_drop_on_632.py", "```",
              "", "API 0、训练 0、未改权重/标签。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "variants.json").write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e4_drop_on_632", "api_calls": 0,
                                                 "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
