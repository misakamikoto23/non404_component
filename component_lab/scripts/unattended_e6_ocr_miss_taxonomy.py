"""E6: taxonomy of unread GT designators.

For every observable GT key that never appears in the cached OCR text, classify:
  - no_text_nearby : OCR detected nothing within 2 text-heights of the symbol
                     (text-detection / resolution problem)
  - text_nearby    : OCR has text there but the string is wrong
                     (recognition / merge problem)
and bucket by the GT symbol's short side in original pixels.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.text_roles import REFERENCE
from component_lab.eval.coords import flip_bbox

SPLIT = ROOT / "configs/contest6_group_split_v1.json"
GT = ROOT / "data/official_contest6/train200"
OCR = ROOT / "outputs/component_identity_round_001_20261017/inputs"
OUT = ROOT / "outputs/unattended_20261018_e6_ocr_miss_taxonomy"

BUCKETS = ((0, 20, "<20px"), (20, 40, "20-40px"), (40, 10**6, ">=40px"))


def bucket(short):
    for lo, hi, name in BUCKETS:
        if lo <= short < hi:
            return name
    return ">=40px"


def gap(a, b):
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0))


def height(box):
    return max(3.0, min(abs(box[2] - box[0]), abs(box[3] - box[1])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    entries = [(str(e["case"]).zfill(4), e["stem"], e.get("tag", "unknown")) for e in split["partitions"]["dev"]]
    agg = defaultdict(lambda: {"missed": 0, "no_text_nearby": 0, "text_nearby": 0})
    by_tag = defaultdict(lambda: {"visible": 0, "read": 0, "missed": 0, "no_text": 0, "text_nearby": 0})
    rows = []
    for case, stem, tag in entries:
        gt_path = GT / f"{stem}.json"
        ocr_path = OCR / case / "ocr_rapidocr.json"
        if not gt_path.exists() or not ocr_path.exists():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        texts = ocr.get("texts", [])
        raws = {str(t.get("raw", t.get("text", ""))) for t in texts}
        image_height = (ocr.get("views") or [[0, 0, 0, 0]])[0][3] or max(
            [t["bbox"][3] for t in texts] + [1])
        for key, comp in gt["components"].items():
            if not REFERENCE.fullmatch(key):
                continue
            by_tag[tag]["visible"] += 1
            if key in raws:
                by_tag[tag]["read"] += 1
                continue
            by_tag[tag]["missed"] += 1
            box = flip_bbox(comp["bbox"], image_height)
            short = min(box[2] - box[0], box[3] - box[1])
            near = False
            for text in texts:
                tb = text["bbox"]
                if gap(box, tb) / max(height(tb), short, 1.0) <= 2.0:
                    near = True
                    break
            b = bucket(short)
            agg[b]["missed"] += 1
            agg[b]["text_nearby" if near else "no_text_nearby"] += 1
            by_tag[tag]["text_nearby" if near else "no_text"] += 1
            if len(rows) < 60:
                rows.append({"case": case, "key": key, "tag": tag, "short_px": round(short, 1),
                             "text_nearby": near, "bucket": b})
    lines = ["# E6：未读出位号的失败类型（检测漏 vs 识别错）", "",
             "只读：dev20 既有 RapidOCR 缓存 + 官方 GT。对每个「可见但未原文命中」的位号，"
             "判断其符号附近 2 个文字高度内是否存在任何 OCR 文本。", "",
             "| 符号短边（原图 px） | 未读出 | 附近无任何文本（检测/分辨率问题） | 附近有文本但读错（识别/合并问题） |",
             "|---|---:|---:|---:|"]
    for _, _, name in BUCKETS:
        v = agg[name]
        if v["missed"]:
            lines.append(f"| {name} | {v['missed']} | {v['no_text_nearby']} | {v['text_nearby']} |")
    total = {"missed": sum(v["missed"] for v in agg.values()),
             "no_text_nearby": sum(v["no_text_nearby"] for v in agg.values()),
             "text_nearby": sum(v["text_nearby"] for v in agg.values())}
    lines += ["", f"合计：未读出 **{total['missed']}** 个；其中 **{total['no_text_nearby']}** 个附近根本没有文本，"
              f"**{total['text_nearby']}** 个附近有文本但没读对。", "",
              "## 按来源", "", "| 来源 | 可见 key | 读出 | 未读出 | 其中无文本 | 其中有文本 |", "|---|---:|---:|---:|---:|---:|"]
    for tag, v in sorted(by_tag.items(), key=lambda kv: -kv[1]["visible"]):
        lines.append(f"| {tag} | {v['visible']} | {v['read']} | {v['missed']} | {v['no_text']} | {v['text_nearby']} |")
    lines += ["", "## 判读", "",
              "- `无文本` 占多数 → 应该提高**文字检测分辨率**（放大、切块、更小 scale）或换更强的 det 模型；",
              "- `有文本` 占多数 → 应该改**识别/合并**（多行合并、竖排、字符集）；",
              "- 分桶还能看出是不是小符号问题：`<20px` 占比高说明是小目标，指向放大而非换模型。",
              "", "## 复现", "", "```powershell", "python scripts/unattended_e6_ocr_miss_taxonomy.py", "```",
              "", "明细：`misses.json`。API 0、训练 0。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "misses.json").write_text(json.dumps({"buckets": agg, "by_tag": by_tag,
                                                 "totals": total, "examples": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e6_ocr_miss_taxonomy", "api_calls": 0,
                                                 "totals": total}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"totals": total, "buckets": agg}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
