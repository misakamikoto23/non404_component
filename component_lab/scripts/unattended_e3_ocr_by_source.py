"""E3: OCR coverage and key recovery, grouped by schematic source tag.

Read-only. For each dev20 case we test, for every non-anonymous GT key, whether
the string is present in the cached RapidOCR output (exact token or ASCII
boundary substring), and whether the frozen prediction exported that key.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.text_roles import REFERENCE

SPLIT = ROOT / "configs/contest6_group_split_v1.json"
PRED = ROOT / "outputs/advance_until_bottleneck_20261017/lexical_experiment/identity_v1/predictions"
GT = ROOT / "data/official_contest6/train200"
OCR = ROOT / "outputs/component_identity_round_001_20261017/inputs"
OUT = ROOT / "outputs/unattended_20261018_e3_ocr_by_source"


def dev_cases():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    for entry in split["partitions"]["dev"]:
        yield str(entry["case"]).zfill(4), entry["stem"], entry.get("tag", "unknown"), entry.get("group")


def is_observable(key):
    return bool(REFERENCE.fullmatch(key))


def boundary_hit(key, texts):
    pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])")
    for raw in texts:
        if raw == key or pattern.search(raw):
            return True
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    agg = defaultdict(lambda: {"cases": 0, "gt_all": 0, "gt_visible": 0, "exact": 0,
                               "boundary": 0, "key_hit": 0})
    rows = []
    for case, stem, tag, group in dev_cases():
        gt_path = GT / f"{stem}.json"
        ocr_path = OCR / case / "ocr_rapidocr.json"
        pred_path = PRED / f"{case}.json"
        if not gt_path.exists() or not ocr_path.exists():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        texts = [str(t.get("raw", t.get("text", ""))) for t in ocr.get("texts", [])]
        preds = json.loads(pred_path.read_text(encoding="utf-8"))["components"] if pred_path.exists() else {}
        stat = {"case": case, "tag": tag, "group": group, "gt_all": len(gt["components"]),
                "gt_visible": 0, "exact": 0, "boundary": 0, "key_hit": 0}
        for key in gt["components"]:
            if not is_observable(key):
                continue
            stat["gt_visible"] += 1
            if key in texts:
                stat["exact"] += 1
            if boundary_hit(key, texts):
                stat["boundary"] += 1
            if key in preds:
                stat["key_hit"] += 1
        rows.append(stat)
        bucket = agg[tag]
        bucket["cases"] += 1
        for field in ("gt_all", "gt_visible", "exact", "boundary", "key_hit"):
            bucket[field] += stat[field]

    lines = ["# E3：OCR 覆盖率与 key 恢复率（按原理图来源）", "",
             "只读：固定 dev20、既有 RapidOCR 缓存、冻结预测。不使用模型推理、不调 API。",
             "口径：`GT 可见 key` = 能通过位号语法（如 `R12`/`U1`）的 key；匿名 `∅` 类不可观测，已排除。",
             "`原文命中` = OCR 原文里出现与 key 完全相同的 token；`边界命中` = 作为独立子串出现（排除 `R1`⊂`R10`）。", "",
             "| 来源 | 图数 | GT 元件 | 可见 key | 原文命中 | 边界命中 | 导出 key 命中 |", "|---|---:|---:|---:|---:|---:|---:|"]
    order = sorted(agg.items(), key=lambda kv: -kv[1]["gt_visible"])
    for tag, v in order:
        lines.append(f"| {tag} | {v['cases']} | {v['gt_all']} | {v['gt_visible']} | {v['exact']} | {v['boundary']} | {v['key_hit']} |")
    lines.append("")
    lines.append("| 来源 | 原文覆盖率 | 边界覆盖率 | 导出命中率 |")
    lines.append("|---|---:|---:|---:|")
    for tag, v in order:
        if v["gt_visible"]:
            lines.append(f"| {tag} | {v['exact']/v['gt_visible']*100:.1f}% | "
                         f"{v['boundary']/v['gt_visible']*100:.1f}% | {v['key_hit']/v['gt_visible']*100:.1f}% |")
    total = {k: sum(v[k] for _, v in order) for k in ("cases", "gt_all", "gt_visible", "exact", "boundary", "key_hit")}
    lines += ["", "## 合计", "",
              f"- 20 图共 {total['gt_all']} 个 GT 元件，其中 **{total['gt_visible']} 个可观测 key**"
              f"（{total['gt_visible']/total['gt_all']*100:.1f}%）。",
              f"- OCR 原文命中 {total['exact']}（{total['exact']/total['gt_visible']*100:.1f}%），"
              f"边界命中 {total['boundary']}（{total['boundary']/total['gt_visible']*100:.1f}%）。",
              f"- 最终导出 key 命中 {total['key_hit']}（{total['key_hit']/total['gt_visible']*100:.1f}%）。",
              "", "## 读法", "",
              "- `原文命中 → 导出命中` 的落差 = **归属/角色/竞争** 造成的损失（可优化）。",
              "- `可见 key → 原文命中` 的落差 = **OCR 未读出**（换识别器/放大/旋转可救）。",
              "- 不同来源的落差结构不同，下一轮 OCR 投入应按来源排序，而不是全图统一加码。",
              "", "## 复现", "", "```powershell", "python scripts/unattended_e3_ocr_by_source.py", "```",
              "", "明细：`per_case.json`。API 0、训练 0。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "per_case.json").write_text(json.dumps({"rows": rows, "by_tag": agg}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e3_ocr_by_source", "api_calls": 0,
                                                 "totals": total}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"by_tag": agg, "total": total}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
