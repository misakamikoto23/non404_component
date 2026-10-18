"""E12: the official text rule may be stricter than our local scorer.

The published rules list five conditions per component (id, type, Name, value,
bbox). Our local scorer only checks Name for box/ic and value for r/c. This
script re-scores the best available predictions under both readings and tests
whether filling Name with the designator for passives recovers the difference.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPLIT = ROOT / "configs/contest6_group_split_v1.json"
GT = ROOT / "data/official_contest6/train200"
BEST = ROOT / "outputs/unresolved_identity_recovery_20261017/cache_plus_new_keys/predictions"
OUT = ROOT / "outputs/unattended_20261018_e12_text_rule"
UNRESOLVED = "__unresolved"
TOL = 20.0
PASSIVE = {"r", "c", "l", "resistor", "capacitor", "inductor"}


def bbox_ok(a, b):
    return len(a) == 4 and all(abs(float(x) - float(y)) <= TOL for x, y in zip(a, b))


def score(pred, gt, name_for_all):
    tp = 0
    for key, expected in gt["components"].items():
        actual = pred.get(key)
        if actual is None:
            continue
        if str(actual.get("type")) != str(expected.get("type")):
            continue
        if not bbox_ok(actual.get("bbox", []), expected["bbox"]):
            continue
        ctype = str(expected.get("type")).lower()
        if name_for_all or ctype in {"box", "ic"}:
            if actual.get("Name") != expected.get("Name"):
                continue
        if ctype in PASSIVE and actual.get("value") != expected.get("value"):
            continue
        tp += 1
    fp = len(pred) - tp
    fn = len(gt["components"]) - tp
    denom = 2 * tp + fp + fn
    return (2 * tp / denom if denom else 0.0), tp, fp, fn


def cases():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    return [(str(e["case"]).zfill(4), e["stem"]) for e in split["partitions"]["dev"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, stem in cases():
        pred_path = BEST / f"{case}.json"
        gt_path = GT / f"{stem}.json"
        if not pred_path.exists() or not gt_path.exists():
            continue
        pred = json.loads(pred_path.read_text(encoding="utf-8"))["components"]
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        submitted = {k: v for k, v in pred.items() if not k.startswith(UNRESOLVED)}
        named = {k: v for k, v in submitted.items() if str(v.get("type", "")).lower() != "component"}
        variants = {"submitted": submitted, "submitted_name_key": None, "name_key_no_filter": None}
        name_key = {}
        for k, v in submitted.items():
            item = dict(v)
            if str(item.get("type", "")).lower() in PASSIVE:
                item["Name"] = k          # GT stores the designator in Name for passives
            name_key[k] = item
        variants["submitted_name_key"] = name_key
        name_key_all = {}
        for k, v in named.items():
            item = dict(v)
            if str(item.get("type", "")).lower() in PASSIVE:
                item["Name"] = k
            name_key_all[k] = item
        variants["name_key_no_filter"] = name_key_all
        for vname, comps in variants.items():
            if comps is None:
                continue
            for rule in (False, True):
                f1, tp, fp, fn = score(comps, gt, rule)
                rows.append({"case": case, "variant": vname, "name_for_all": rule,
                             "f1": f1, "tp": tp, "fp": fp, "fn": fn, "predicted": len(comps)})
    summary = []
    for vname in ("submitted", "submitted_name_key", "name_key_no_filter"):
        for rule in (False, True):
            sel = [r for r in rows if r["variant"] == vname and r["name_for_all"] == rule]
            if not sel:
                continue
            tp = sum(r["tp"] for r in sel); fp = sum(r["fp"] for r in sel); fn = sum(r["fn"] for r in sel)
            summary.append({"variant": vname, "name_for_all": rule,
                            "macro_f1": sum(r["f1"] for r in sel) / len(sel),
                            "tp": tp, "fp": fp, "fn": fn,
                            "predicted": sum(r["predicted"] for r in sel)})
    lines = ["# E12：官方文本规则可能比本地评分器更严格", "",
             "官方说明把「元件ID / 类型 / 位号(Name) / 参数值 / bbox」列为并列的 TP 条件；"
             "本地评分器目前只对 box/ic 检查 Name、只对 r/c 检查 value。本实验两种读法都算一遍。", "",
             "| 预测版本 | Name 规则 | 逐图宏 F1 | TP | FP | FN |", "|---|---|---:|---:|---:|---:|"]
    for s in summary:
        rule = "所有类型都比 Name" if s["name_for_all"] else "Name 只比 box/ic（现状）"
        lines.append(f"| {s['variant']} | {rule} | {s['macro_f1']*100:.2f}% | {s['tp']} | {s['fp']} | {s['fn']} |")
    lines += ["", "## 关键对照", "",
              "- `submitted` + 现状规则 = 我们现在汇报的分数；",
              "- `submitted` + 严格规则 = 若官方按字面执行，我们实际会拿到的分数；",
              "- `submitted_name_key` + 严格规则 = 把无源器件的 Name 填成位号后的分数（一行导出改动）。",
              "",
              "## 判读", "",
              "- 若「严格规则」明显低于「现状规则」，说明我们的本地分数被高估，必须修正导出。",
              "- 若「Name 填位号」能补回大部分差距，那这是一个**零成本、纯导出层**的修复。",
              "- 注意：GT 里 87/91 的 Name 等于 key，IC 的 Name 是型号——所以对无源器件把 Name 设成 key 是有据可依的。",
              "", "## 复现", "", "```powershell", "python scripts/unattended_e12_text_rule.py", "```"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "rows.json").write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e12_text_rule", "api_calls": 0,
                                                 "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
