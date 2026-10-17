"""E0: interpret whatever arms the running dev20_v6_qwen_assignment has produced.

Read-only over that directory: every `contest_components.json` is scored against
the official GT with the local strict scorer, grouped by (case, arm) derived from
the path. Safe to run while the experiment is still writing: it only reads files
that already exist and never writes into the experiment directory.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.eval.metric import ScoreConfig, score_case

EXP = ROOT / "outputs/dev20_v6_qwen_assignment_20261018"
GT = ROOT / "data/official_contest6/train200"
SPLIT = ROOT / "configs/contest6_group_split_v1.json"
OUT = ROOT / "outputs/unattended_20261018_e0_interpret"

UNRESOLVED = "__unresolved"
SKIP_DIRS = {"local", "inputs", "qwen_inputs", "qwen_responses", "qwen_audits", "qwen_attempts"}


def dev_stems():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    return {str(e["case"]).zfill(4): e["stem"] for e in split["partitions"]["dev"]}


def collect():
    found = defaultdict(dict)
    for path in EXP.rglob("contest_components.json"):
        match = re.search(r"(\d{4})", str(path.relative_to(EXP)))
        if not match:
            continue
        found[match.group(1)][path.parent.name] = path
    for path in EXP.glob("*/*.json"):
        if path.parent.name in SKIP_DIRS or not re.fullmatch(r"\d{4}\.json", path.name):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("components"), dict):
            found[path.stem][path.parent.name] = path
    return found


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stems = dev_stems()
    config = ScoreConfig()
    found = collect()
    rows = []
    for case, arms in sorted(found.items()):
        gt_path = GT / f"{stems[case]}.json" if case in stems else None
        if gt_path is None or not gt_path.exists():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        for arm, path in arms.items():
            try:
                pred = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue  # still being written
            comps = pred.get("components", {})
            named = {k: v for k, v in comps.items() if not k.startswith(UNRESOLVED)}
            for policy, subset in (("all", comps), ("submitted", named)):
                result = score_case({"components": subset, "pins": {}, "nets": {}}, gt, config)
                c = result["counts"]["component_f1"]
                rows.append({"case": case, "arm": arm, "policy": policy,
                             "f1": result["component_f1"], "tp": c["tp"], "fp": c["fp"],
                             "fn": c["fn"], "predicted": len(subset)})
    agg = defaultdict(lambda: {"cases": 0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0, "predicted": 0})
    for row in rows:
        key = (row["arm"], row["policy"])
        a = agg[key]
        a["cases"] += 1
        a["f1"] += row["f1"]
        for field in ("tp", "fp", "fn", "predicted"):
            a[field] += row[field]
    summary = []
    for (arm, policy), a in sorted(agg.items()):
        summary.append({"arm": arm, "policy": policy, "cases": a["cases"],
                        "macro_f1": a["f1"] / a["cases"] if a["cases"] else 0.0,
                        "tp": a["tp"], "fp": a["fp"], "fn": a["fn"], "predicted": a["predicted"]})
    lines = ["# E0：dev20_v6_qwen_assignment 判读（自动生成）", "",
             f"扫描 `{EXP.name}`，找到 {len(found)} 个 case。每个物理目录名视为一个 arm。", "",
             "| arm | 导出策略 | 图数 | 逐图宏 F1 | TP | FP | 提交条目 |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['arm']} | {s['policy']} | {s['cases']} | {s['macro_f1']*100:.2f}% | "
                     f"{s['tp']} | {s['fp']} | {s['predicted']} |")
    def get(arm, policy="submitted"):
        return next((s for s in summary if s["arm"] == arm and s["policy"] == policy), None)

    def paired(a_arm, b_arm, policy="submitted"):
        """Macro F1 over the cases both arms actually produced (fair comparison)."""
        a = {r["case"]: r["f1"] for r in rows if r["arm"] == a_arm and r["policy"] == policy}
        b = {r["case"]: r["f1"] for r in rows if r["arm"] == b_arm and r["policy"] == policy}
        shared = sorted(set(a) & set(b))
        if not shared:
            return None
        ma = sum(a[c] for c in shared) / len(shared)
        mb = sum(b[c] for c in shared) / len(shared)
        return {"cases": len(shared), "a": ma, "b": mb, "delta": ma - mb}

    primary, local = get("primary"), get("local_v6")
    lines += ["", "## 三个关键差值（只在两臂都有的图上配对比较）", ""]
    pair_local = paired("local_v6", "primary") if local and primary else None
    if pair_local:
        lines.append(f"- **v6 局部补读净增益**：{pair_local['b']*100:.2f}% → {pair_local['a']*100:.2f}%，"
                     f"**{pair_local['delta']*100:+.2f}pt**（配对 {pair_local['cases']} 图）")
    hungarian, qwen = get("hungarian_all_ocr"), get("qwen_replace")
    pair_alloc = paired("qwen_replace", "hungarian_all_ocr") if hungarian and qwen else None
    if pair_alloc:
        lines.append(f"- **纯分配算法增益（Qwen 替代匈牙利）**：{pair_alloc['b']*100:.2f}% → {pair_alloc['a']*100:.2f}%，"
                     f"**{pair_alloc['delta']*100:+.2f}pt**（配对 {pair_alloc['cases']} 图）")
    fill = get("qwen_fill")
    pair_fill = paired("qwen_fill", "primary") if primary and fill else None
    if pair_fill:
        lines.append(f"- **保守补空增益**：{pair_fill['b']*100:.2f}% → {pair_fill['a']*100:.2f}%，"
                     f"**{pair_fill['delta']*100:+.2f}pt**（配对 {pair_fill['cases']} 图）")
    if len(summary) < 4:
        lines.append("- ⚠️ 仅发现部分 arm，实验可能仍在运行；重跑本脚本可更新。")
    lines += ["", "## 判读规则（预声明）", "",
              "- 分配算法增益 ≥ +1.0pt → 值得考虑；区间跨 0 或为负 → 结案，不要再投 API。",
              "- v6 补读增益 ≥ +1.0pt → 保持默认开启。",
              "- 所有比较均为同框同 OCR 池，仅切换分配方式。",
              "", "## 复现", "", "```powershell", "python scripts/unattended_e0_interpret.py", "```",
              "", "API 0、训练 0、只读该实验目录。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"rows": rows, "summary": summary,
                                                  "cases_found": sorted(found)}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e0_interpret", "api_calls": 0,
                                                 "arms": sorted({s['arm'] for s in summary})},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
