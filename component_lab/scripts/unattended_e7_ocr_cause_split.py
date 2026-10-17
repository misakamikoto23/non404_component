"""E7: split the 220 unread designators into three actionable causes.

For each observable GT key that is absent from the OCR raw set, look at the OCR
tokens near the symbol and classify:
  contained : the key sits inside a longer token (merge/segmentation problem)
  near_miss : a nearby token is within edit distance 1-2     (recognition error)
  absent    : nothing resembling it                          (detection/scale)
"""
import json
import math
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.text_roles import REFERENCE
from component_lab.eval.coords import flip_bbox

SPLIT = ROOT / "configs/contest6_group_split_v1.json"
GT = ROOT / "data/official_contest6/train200"
OCR = ROOT / "outputs/component_identity_round_001_20261017/inputs"
OUT = ROOT / "outputs/unattended_20261018_e7_ocr_cause_split"


def gap(a, b):
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0))


def height(box):
    return max(3.0, min(abs(box[2] - box[0]), abs(box[3] - box[1])))


def lev(a, b, cap=3):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
        if min(prev) > cap:
            return cap + 1
    return prev[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    entries = [(str(e["case"]).zfill(4), e["stem"], e.get("tag", "unknown")) for e in split["partitions"]["dev"]]
    agg = defaultdict(lambda: {"missed": 0, "contained": 0, "near_miss": 0, "absent": 0})
    by_tag = defaultdict(lambda: {"missed": 0, "contained": 0, "near_miss": 0, "absent": 0})
    examples = []
    for case, stem, tag in entries:
        gt_path = GT / f"{stem}.json"
        ocr_path = OCR / case / "ocr_rapidocr.json"
        if not gt_path.exists() or not ocr_path.exists():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        texts = ocr.get("texts", [])
        raws = {str(t.get("raw", t.get("text", ""))) for t in texts}
        image_height = (ocr.get("views") or [[0, 0, 0, 0]])[0][3] or max([t["bbox"][3] for t in texts] + [1])
        for key, comp in gt["components"].items():
            if not REFERENCE.fullmatch(key) or key in raws:
                continue
            box = flip_bbox(comp["bbox"], image_height)
            short = min(box[2] - box[0], box[3] - box[1])
            nearby = [str(t.get("raw", t.get("text", ""))) for t in texts
                      if gap(box, t["bbox"]) / max(height(t["bbox"]), short, 1.0) <= 2.0]
            kind = "absent"
            detail = None
            for raw in nearby:
                token = raw.strip()
                if key in token:
                    kind, detail = "contained", token
                    break
                for part in re.split(r"[^A-Za-z0-9]+", token):
                    if part and lev(key.upper(), part.upper()) <= max(1, len(key) // 3):
                        kind, detail = "near_miss", token
                        break
                if kind != "absent":
                    break
            if kind == "absent":
                best = max(nearby, key=SequenceMatcher(None, "", "").ratio) if False else None
            agg[kind if kind else "absent"]["missed"] += 1
            agg[kind if kind else "absent"][kind if kind else "absent"] += 1
            by_tag[tag]["missed"] += 1
            by_tag[tag][kind] += 1
            if len(examples) < 40:
                examples.append({"case": case, "key": key, "tag": tag, "kind": kind,
                                 "ocr": detail, "short_px": round(short, 1)})
    totals = {k: sum(v[k] for v in agg.values()) for k in ("missed", "contained", "near_miss", "absent")}
    lines = ["# E7：220 个漏读位号的三个可行动成因", "",
             "对每个「可见但 OCR 原文里没有」的位号，看它符号附近 2 个文字高度内的 OCR token：", "",
             "- `contained`：位号被包在更长的串里（如 `R8 5.1kΩ`）→ **切分问题**",
             "- `near_miss`：附近有近似串（编辑距离小）→ **识别错字**",
             "- `absent`：附近没有像的 → **没检出/太小**", "",
             "| 成因 | 数量 | 占比 | 对应手段 |", "|---|---:|---:|---|",
             f"| contained（合并/切分） | {totals['contained']} | {totals['contained']/max(totals['missed'],1)*100:.1f}% | 字符 span 切分（零模型成本） |",
             f"| near_miss（读错字） | {totals['near_miss']} | {totals['near_miss']/max(totals['missed'],1)*100:.1f}% | 放大重识别 / 换 rec / 微调 |",
             f"| absent（没读到） | {totals['absent']} | {totals['absent']/max(totals['missed'],1)*100:.1f}% | ROI 重读 / 提高检测分辨率 |",
             f"| **合计** | **{totals['missed']}** | 100% | |", "",
             "## 按来源", "", "| 来源 | 漏读 | contained | near_miss | absent |", "|---|---:|---:|---:|---:|"]
    for tag, v in sorted(by_tag.items(), key=lambda kv: -kv[1]["missed"]):
        lines.append(f"| {tag} | {v['missed']} | {v['contained']} | {v['near_miss']} | {v['absent']} |")
    lines += ["", "## 样例", "", "| case | key | 成因 | 附近 OCR | 短边 px |", "|---|---|---|---|---:|"]
    for e in examples[:25]:
        lines.append(f"| {e['case']} | {e['key']} | {e['kind']} | `{e['ocr']}` | {e['short_px']} |")
    lines += ["", "## 复现", "", "```powershell", "python scripts/unattended_e7_ocr_cause_split.py", "```",
              "", "API 0、训练 0、只读缓存。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "causes.json").write_text(json.dumps({"totals": totals, "by_tag": by_tag,
                                                 "examples": examples}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e7_ocr_cause_split", "api_calls": 0,
                                                 "totals": totals}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
