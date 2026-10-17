"""E8: how much can an ROI re-read recover? (selection-oracle, blind recognition)

For every observable GT designator that the full-page OCR missed, crop a region
around the symbol, upscale 2x and 4x, and let RapidOCR read it. The ROI *choice*
uses GT boxes (oracle for selection only); the recognition never sees GT.

Run with the OCR env:
    .venv-ocr\Scripts\python.exe scripts/unattended_e8_roi_reread_probe.py
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data/official_contest6/train200"
OCR = ROOT / "outputs/component_identity_round_001_20261017/inputs"
SPLIT = ROOT / "configs/contest6_group_split_v1.json"
OUT = ROOT / "outputs/unattended_20261018_e8_roi_reread"
WEIGHTS = ROOT / "weights/ocr_rapid"

REF = re.compile(r"^[A-Za-z]{1,8}\d+(?:[A-Za-z]|\.\d+)?$")
BUCKETS = ((0, 20, "<20px"), (20, 40, "20-40px"), (40, 10**6, ">=40px"))


def bucket(px):
    for lo, hi, name in BUCKETS:
        if lo <= px < hi:
            return name
    return ">=40px"


def build_engine():
    from rapidocr import RapidOCR, OCRVersion, ModelType
    return RapidOCR(params={
        "Global.model_root_dir": str(WEIGHTS), "Global.max_side_len": 4096,
        "Global.text_score": 0.30,
        "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
        "Det.limit_side_len": 2048, "Det.limit_type": "max",
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False})


def hit(key, texts):
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])")
    for raw in texts:
        if raw == key or pat.search(raw):
            return raw
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    entries = [(str(e["case"]).zfill(4), e["stem"], e.get("tag", "?")) for e in split["partitions"]["dev"]]
    engine = build_engine()
    stats = defaultdict(lambda: {"missed": 0, "hit2": 0, "hit4": 0, "either": 0})
    by_tag = defaultdict(lambda: {"missed": 0, "hit2": 0, "hit4": 0})
    details = []
    total_seconds = 0.0
    rois = 0
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
            box = [x1, H - y2, x2, H - y1]  # bottom-left -> top-left
            w, h = box[2] - box[0], box[3] - box[1]
            short = min(w, h)
            margin = max(12.0, 2.5 * max(w, h))
            crop = [max(0, int(box[0] - margin)), max(0, int(box[1] - margin)),
                    min(W, int(box[2] + margin)), min(H, int(box[3] + margin))]
            if crop[2] - crop[0] < 8 or crop[3] - crop[1] < 8:
                continue
            region = image.crop(tuple(crop))
            got = {}
            for scale in (2, 4):
                scaled = region.resize((region.width * scale, region.height * scale), Image.Resampling.LANCZOS)
                tick = time.perf_counter()
                out = engine(np.asarray(scaled))
                total_seconds += time.perf_counter() - tick
                rois += 1
                texts = [str(t) for t in (out.txts or [])] if out is not None else []
                got[scale] = hit(key, texts)
            b = bucket(short)
            stats[b]["missed"] += 1
            by_tag[tag]["missed"] += 1
            if got[2]:
                stats[b]["hit2"] += 1
                by_tag[tag]["hit2"] += 1
            if got[4]:
                stats[b]["hit4"] += 1
                by_tag[tag]["hit4"] += 1
            if got[2] or got[4]:
                stats[b]["either"] += 1
            details.append({"case": case, "key": key, "tag": tag, "short_px": round(short, 1),
                            "bucket": b, "hit_2x": got[2], "hit_4x": got[4]})
    total = {k: sum(v[k] for v in stats.values()) for k in ("missed", "hit2", "hit4", "either")}
    lines = ["# E8：ROI 放大重读能救回多少漏读位号（选区 oracle，识别盲测）", "",
             "对 220 个「全页 OCR 未读出」的可见位号：以 GT 框为中心裁一块（2.5 倍长边外扩），"
             "分别放大 2× 和 4× 后交给同一个 RapidOCR v5 mobile 识别。",
             "**选区用了 GT（仅此一处），识别过程完全不看 GT**，因此结果是「完美选区下 ROI 重读的上界」。", "",
             "| 符号短边 | 漏读 | 2× 救回 | 4× 救回 | 任一救回 | 救回率 |", "|---|---:|---:|---:|---:|---:|"]
    for _, _, name in BUCKETS:
        v = stats[name]
        if v["missed"]:
            lines.append(f"| {name} | {v['missed']} | {v['hit2']} | {v['hit4']} | {v['either']} | "
                         f"{v['either']/v['missed']*100:.1f}% |")
    lines += ["", f"**合计：{total['missed']} 个漏读，2× 救回 {total['hit2']}，4× 救回 {total['hit4']}，"
              f"任一救回 {total['either']}（{total['either']/max(total['missed'],1)*100:.1f}%）。**", "",
              f"耗时：{rois} 次识别，合计 {total_seconds:.1f} 秒，平均 {total_seconds/max(rois,1)*1000:.0f} ms/次。", "",
              "## 按来源", "", "| 来源 | 漏读 | 2× 救回 | 4× 救回 |", "|---|---:|---:|---:|"]
    for tag, v in sorted(by_tag.items(), key=lambda kv: -kv[1]["missed"]):
        lines.append(f"| {tag} | {v['missed']} | {v['hit2']} | {v['hit4']} |")
    lines += ["", "## 判读", "",
              "- 救回率就是**ROI 重读的天花板**（选区完美、识别只有一次机会）。实际系统选区会打折，所以真实收益低于此值。",
              "- 2× 与 4× 的差说明「放大」是否还有收益；若 4× 明显更高，说明主 OCR 的 2× 不够。",
              "- 若救回率很低，说明这些字**不是分辨率问题**，而是根本没有可读的墨迹（或被导线压住、或 GT 框与文字错位）。",
              "", "## 复现", "",
              "```powershell",
              ".\\.venv-ocr\\Scripts\\python.exe scripts/unattended_e8_roi_reread_probe.py",
              "```", "", "明细：`recovered.json`。API 0、训练 0、GT 未修改。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "recovered.json").write_text(json.dumps({"totals": total, "buckets": stats,
                                                     "by_tag": by_tag, "details": details,
                                                     "seconds": total_seconds, "engine_calls": rois},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e8_roi_reread", "api_calls": 0,
                                                 "totals": total, "engine_calls": rois,
                                                 "seconds": round(total_seconds, 1)},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"totals": total, "buckets": {k: dict(v) for k, v in stats.items()},
                      "seconds": round(total_seconds, 1), "calls": rois}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
