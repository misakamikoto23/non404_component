"""E2: how much does the GT-null protocol conflict with visible text?

Two independent measurements on frozen artifacts:
  A. Surface: GT value=null resistors/capacitors that have a value-looking OCR
     token nearby (image evidence exists, GT says null).
  B. Realised loss: in the frozen predictions, keys present on both sides where
     GT value is null but the prediction filled a non-null value.
No inference, no API, no GT-driven edits.
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.text_roles import VALUE, value_kind
from component_lab.eval.coords import flip_bbox

SPLIT = ROOT / "configs/contest6_group_split_v1.json"
PRED = ROOT / "outputs/advance_until_bottleneck_20261017/lexical_experiment/identity_v1/predictions"
GT = ROOT / "data/official_contest6/train200"
OCR = ROOT / "outputs/component_identity_round_001_20261017/inputs"
OUT = ROOT / "outputs/unattended_20261018_e2_null_conflict"

VALUE_TYPES = {"r", "c", "resistor", "capacitor", "l", "inductor"}
DIST_TEXT_HEIGHTS = 3.0


def dev_cases():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    out = []
    for entry in split["partitions"]["dev"]:
        case = entry["case"] if isinstance(entry, dict) else entry
        out.append((str(case).zfill(4), entry.get("stem") if isinstance(entry, dict) else str(case),
                    entry.get("tag") if isinstance(entry, dict) else "unknown"))
    return out


def gap(a, b):
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0))


def height(box):
    return max(3.0, min(abs(box[2] - box[0]), abs(box[3] - box[1])))


def looks_like_value(raw):
    token = str(raw).strip()
    compact = re.sub(r"\s+", "", token)
    if VALUE.fullmatch(token) or VALUE.fullmatch(compact):
        return value_kind(token)
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rows, examples = [], []
    totals = {k: 0 for k in ("gt_value_null", "gt_value_set", "surface_conflicts",
                             "lost_filled", "missed_empty", "ocr_missing")}
    for case, stem, tag in dev_cases():
        gt_path = GT / f"{stem}.json"
        pred_path = PRED / f"{case}.json"
        ocr_path = OCR / case / "ocr_rapidocr.json"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        texts = []
        image_height = None
        if ocr_path.exists():
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            texts = ocr.get("texts", [])
            views = ocr.get("views") or []
            if views:
                image_height = views[0][3]
        if image_height is None:
            totals["ocr_missing"] += 1
            continue
        stat = {"case": case, "tag": tag, "gt_value_null": 0, "gt_value_set": 0,
                "surface_conflicts": 0, "lost_filled": 0, "missed_empty": 0}
        for key, comp in gt["components"].items():
            ctype = str(comp.get("type", "")).lower()
            if ctype not in VALUE_TYPES:
                continue
            is_null = comp.get("value") is None
            stat["gt_value_null" if is_null else "gt_value_set"] += 1
            if is_null:
                box = flip_bbox(comp["bbox"], image_height)
                for text in texts:
                    kind = looks_like_value(text.get("raw", text.get("text", "")))
                    if kind in (None, "tolerance", "voltage", "current", "power"):
                        continue
                    scale = max(height(text["bbox"]), height(box))
                    if gap(box, text["bbox"]) / scale > DIST_TEXT_HEIGHTS:
                        continue
                    stat["surface_conflicts"] += 1
                    if len(examples) < 40:
                        examples.append({"case": case, "key": key, "type": ctype,
                                         "ocr_text": text.get("raw"), "kind": kind,
                                         "pred_value": pred["components"].get(key, {}).get("value")})
                    break
            elif key in pred["components"] and pred["components"][key].get("value") is None:
                stat["missed_empty"] += 1
        for key, comp in gt["components"].items():
            if key not in pred["components"] or comp.get("value") is not None:
                continue
            if str(comp.get("type", "")).lower() in VALUE_TYPES and pred["components"][key].get("value") is not None:
                stat["lost_filled"] += 1
        rows.append(stat)
        for k in stat:
            if k in totals:
                totals[k] += stat[k]
    payload = {"rows": rows, "totals": totals, "examples": examples,
               "distance_threshold_text_heights": DIST_TEXT_HEIGHTS,
               "value_types": sorted(VALUE_TYPES), "api_calls": 0, "uses_ground_truth": "evaluation only"}
    (OUT / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    null = totals["gt_value_null"]; surface = totals["surface_conflicts"]
    lines = ["# E2：GT null 与图面文字的冲突规模", "",
             "只读审计：固定 dev20、冻结预测、既有 OCR 缓存；不改 GT、不推理、不调 API。",
             "判据：GT 的 r/c/l 元件 `value is null`，且其框外 3 个文字高度内存在符合数值语法的 OCR 原文。", "",
             "| 指标 | 数量 |", "|---|---:|",
             f"| r/c/l 元件总数（value 为 null） | {null} |",
             f"| r/c/l 元件总数（value 非 null） | {totals['gt_value_set']} |",
             f"| **图上有数值文字但 GT 为 null** | **{surface}** |",
             f"| 冻结预测中因此被填非空（已实际丢 TP 的候选） | {totals['lost_filled']} |",
             f"| GT 有值但预测为空（漏填） | {totals['missed_empty']} |",
             f"| 缺 OCR 缓存而跳过的图 | {totals['ocr_missing']} |",
             "",
             f"冲突率：{surface}/{null} = {surface/null*100:.1f}%（占全部 r/c/l 空值元件）。" if null else ""]
    lines += ["", "## 含义", "",
              "- 这些元件在图上**确实印着数值**，但官方 GT 记为 `null`；任何正确读取该文字并填写的系统都会**被扣分**。",
              "- 因此「value 字段正确率」并不完全由识别能力决定，其中一部分是**评分协议与标注定义冲突**。",
              "- 该冲突无法靠模型修正，只能向主办方确认 `value` 的取值规则（组合值、额定值、null 的判定边界）。",
              "", "## 样例（前 20）", "", "| case | key | OCR 原文 | 类别 | 当前预测值 |", "|---|---|---|---|---|"]
    for e in examples[:20]:
        lines.append(f"| {e['case']} | {e['key']} | `{e['ocr_text']}` | {e['kind']} | {e['pred_value']} |")
    lines += ["", "## 复现", "", "```powershell", "python scripts/unattended_e2_null_conflict.py", "```",
              "", "明细：`audit.json`（逐图计数与样例）。API 0、训练 0、GT 未修改。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e2_null_conflict", "api_calls": 0,
                                                 "totals": totals}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
