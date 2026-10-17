"""E9: inside each missed designator's ROI, is there any text at all?

Complements E8 (which only counted exact-key recoveries). For every observable
GT key the full-page OCR missed, crop the same ROI at 4x and record every string
RapidOCR returns. Then classify:
    recovered      : the key itself came back
    wrong_text     : the ROI has text, but nothing close to the key
    no_text        : the ROI yields no text at all
Run with the OCR env:
    .venv-ocr\\Scripts\\python.exe scripts/unattended_e9_roi_text_presence.py
"""
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data/official_contest6/train200"
OCR = ROOT / "outputs/component_identity_round_001_20261017/inputs"
SPLIT = ROOT / "configs/contest6_group_split_v1.json"
OUT = ROOT / "outputs/unattended_20261018_e9_roi_text_presence"
WEIGHTS = ROOT / "weights/ocr_rapid"

REF = re.compile(r"^[A-Za-z]{1,8}\d+(?:[A-Za-z]|\.\d+)?$")


def build_engine():
    from rapidocr import RapidOCR, OCRVersion, ModelType
    return RapidOCR(params={
        "Global.model_root_dir": str(WEIGHTS), "Global.max_side_len": 4096,
        "Global.text_score": 0.20,
        "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
        "Det.limit_side_len": 2048, "Det.limit_type": "max",
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False})


def lev(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    entries = [(str(e["case"]).zfill(4), e["stem"], e.get("tag", "?")) for e in split["partitions"]["dev"]]
    engine = build_engine()
    tallies = defaultdict(int)
    per_tag = defaultdict(lambda: defaultdict(int))
    examples = {"wrong_text": [], "near_text": [], "recovered": [], "no_text": []}
    seconds = 0.0
    for case, stem, tag in entries:
        gt_path = GT / f"{stem}.json"
        ocr_path = OCR / case / "ocr_rapidocr.json"
        image_path = OCR / case / "source.png"
        if not (gt_path.exists() and ocr_path.exists() and image_path.exists()):
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        page_raws = {str(t.get("raw", t.get("text", ""))) for t in ocr.get("texts", [])}
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        for key, comp in gt["components"].items():
            if not REF.fullmatch(key) or key in page_raws:
                continue
            x1, y1, x2, y2 = comp["bbox"]
            box = [x1, H - y2, x2, H - y1]
            w, h = box[2] - box[0], box[3] - box[1]
            margin = max(12.0, 2.5 * max(w, h))
            crop = [max(0, int(box[0] - margin)), max(0, int(box[1] - margin)),
                    min(W, int(box[2] + margin)), min(H, int(box[3] + margin))]
            if crop[2] - crop[0] < 8 or crop[3] - crop[1] < 8:
                continue
            region = image.crop(tuple(crop))
            scaled = region.resize((region.width * 4, region.height * 4), Image.Resampling.LANCZOS)
            tick = time.perf_counter()
            out = engine(np.asarray(scaled))
            seconds += time.perf_counter() - tick
            texts = [str(t).strip() for t in (out.txts or [])] if out is not None else []
            texts = [t for t in texts if t]
            if not texts:
                kind = "no_text"
            elif any(t == key or re.search(r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])", t) for t in texts):
                kind = "recovered"
            else:
                best = min(lev(key.upper(), t.upper()) for t in texts)
                kind = "near_text" if best <= 2 else "wrong_text"
            tallies[kind] += 1
            per_tag[tag][kind] += 1
            if len(examples[kind]) < 15:
                examples[kind].append({"case": case, "key": key, "tag": tag,
                                       "texts": texts[:6], "short_px": round(min(w, h), 1)})
    total = sum(tallies.values())
    lines = ["# E9：漏读位号的 ROI 里到底有没有字", "",
             "对全页 OCR 未读出的可见位号，以其 GT 框为中心裁 2.5 倍长边外扩的区域，4× 放大后单独识别（text_score 0.20）。",
             "识别盲测；只有 ROI 选区用了 GT，用于给出上界。", "",
             "| 类别 | 数量 | 占比 | 含义 |", "|---|---:|---:|---|",
             f"| recovered | {tallies['recovered']} | {tallies['recovered']/max(total,1)*100:.1f}% | 重读直接读回位号 |",
             f"| near_text | {tallies['near_text']} | {tallies['near_text']/max(total,1)*100:.1f}% | 有文本且与位号编辑距离 ≤2 |",
             f"| wrong_text | {tallies['wrong_text']} | {tallies['wrong_text']/max(total,1)*100:.1f}% | 有文本但完全不像 |",
             f"| no_text | {tallies['no_text']} | {tallies['no_text']/max(total,1)*100:.1f}% | ROI 内一个字都没检出 |",
             f"| **合计** | **{total}** | 100% | |", "",
             f"耗时：{total} 次识别，{seconds:.1f} 秒（平均 {seconds/max(total,1)*1000:.0f} ms）。", "",
             "## 按来源", "", "| 来源 | recovered | near_text | wrong_text | no_text |", "|---|---:|---:|---:|---:|"]
    for tag, v in sorted(per_tag.items(), key=lambda kv: -sum(kv[1].values())):
        lines.append(f"| {tag} | {v['recovered']} | {v['near_text']} | {v['wrong_text']} | {v['no_text']} |")
    lines += ["", "## 判读", "",
              "- `no_text` 高：说明该区域**根本没有可读墨迹**（位号不在框附近，或图上是别的名字）→ 再怎么重读也没用，属于协议/标注问题。",
              "- `wrong_text` 高：说明有字但读错（小字、连线干扰）→ 换 rec、放大、二值化有空间。",
              "- `near_text` 高：说明差一两个字 → 最容易救，靠 rec 升级或字符级后处理。",
              "", "## 样例", ""]
    for kind in ("no_text", "wrong_text", "near_text", "recovered"):
        if examples[kind]:
            lines.append(f"### {kind}")
            lines.append("")
            for e in examples[kind][:8]:
                lines.append(f"- `{e['case']}/{e['key']}`（短边 {e['short_px']}px）：{e['texts']}")
            lines.append("")
    lines += ["## 复现", "", "```powershell",
              ".\\.venv-ocr\\Scripts\\python.exe scripts/unattended_e9_roi_text_presence.py",
              "```", "", "明细：`presence.json`。API 0、训练 0、GT 未修改。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "presence.json").write_text(json.dumps({"tallies": tallies, "by_tag": per_tag,
                                                   "examples": examples, "seconds": seconds},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e9_roi_text_presence", "api_calls": 0,
                                                 "tallies": tallies, "seconds": round(seconds, 1)},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"tallies": tallies, "seconds": round(seconds, 1)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
