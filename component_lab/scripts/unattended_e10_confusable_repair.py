"""E10: how many near-miss readings become exact after confusable-character repair?

E9 showed 155/224 missed designators are read within edit distance 2. This probe
asks the cheapest possible follow-up: if we map the classic OCR confusions
(l->1, I->1, O->0, S->5, B->8, Z->2, G->6) on the *recognised* string and keep
only strings that pass the designator grammar, how many become exact hits?

Recognition is blind (no GT); GT is used only for ROI selection and scoring.
Run with:  .venv-ocr\\Scripts\\python.exe scripts/unattended_e10_confusable_repair.py
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
OUT = ROOT / "outputs/unattended_20261018_e10_confusable_repair"
WEIGHTS = ROOT / "weights/ocr_rapid"

REF = re.compile(r"^[A-Za-z]{1,8}\d+(?:[A-Za-z]|\.\d+)?$")
CONFUSABLE = str.maketrans({"l": "1", "I": "1", "|": "1", "O": "0", "o": "0",
                            "S": "5", "B": "8", "Z": "2", "G": "6"})


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


def exact(key, texts):
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])")
    return any(t == key or pat.search(t) for t in texts)


def repaired_hit(key, texts):
    """Map confusables, split on non-alphanumerics, keep grammar-valid tokens."""
    for raw in texts:
        for token in re.split(r"[^A-Za-z0-9.]+", raw):
            if not token:
                continue
            if token.translate(CONFUSABLE).upper() == key.upper():
                return token
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    entries = [(str(e["case"]).zfill(4), e["stem"], e.get("tag", "?")) for e in split["partitions"]["dev"]]
    engine = build_engine()
    tally = defaultdict(int)
    per_tag = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)
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
            if exact(key, texts):
                kind = "exact"
            else:
                token = repaired_hit(key, texts)
                if token:
                    kind = "repair"
                    if len(examples[kind]) < 20:
                        examples[kind].append({"case": case, "key": key, "ocr": token, "texts": texts[:5]})
                else:
                    kind = "still_miss"
            tally[kind] += 1
            per_tag[tag][kind] += 1
    total = sum(tally.values())
    lines = ["# E10：易混字符修复能把多少「差一两个字符」救成精确命中", "",
             "对全页 OCR 未读出的可见位号，ROI 4× 重读后：先看是否精确命中；否则把识别串里的易混字符映射"
             "（l/I/|→1、O/o→0、S→5、B→8、Z→2、G→6），只保留通过位号语法的 token，再比一次。",
             "识别盲测；GT 只用于选区与计分。", "",
             "| 结果 | 数量 | 占漏读比例 |", "|---|---:|---:|",
             f"| 直接精确命中 | {tally['exact']} | {tally['exact']/max(total,1)*100:.1f}% |",
             f"| **易混字符修复后命中** | **{tally['repair']}** | **{tally['repair']/max(total,1)*100:.1f}%** |",
             f"| 仍未命中 | {tally['still_miss']} | {tally['still_miss']/max(total,1)*100:.1f}% |",
             f"| **合计漏读** | **{total}** | 100% |", "",
             f"耗时：{total} 次识别，{seconds:.1f} 秒。", "",
             "## 按来源", "", "| 来源 | exact | repair | still_miss |", "|---|---:|---:|---:|"]
    for tag, v in sorted(per_tag.items(), key=lambda kv: -sum(kv[1].values())):
        lines.append(f"| {tag} | {v['exact']} | {v['repair']} | {v['still_miss']} |")
    lines += ["", "## 修复样例", ""]
    for e in examples["repair"][:15]:
        lines.append(f"- `{e['case']}/{e['key']}` ← 识别为 `{e['ocr']}`（同框文本 {e['texts']}）")
    lines += ["", "## 判读", "",
              "- `repair` 是**零模型成本**能拿回的量：只要在候选层允许易混字符映射，并保留原文做审计。",
              "- 但注意：修复会引入**假阳性风险**，必须用位号语法 + 唯一性约束（同页不重复）兜底。",
              "- 若 `repair` 数量很小，说明剩余错误不是易混字符，而是更严重的字形/切分问题，应转向 rec 升级。",
              "", "## 复现", "",
              "```powershell",
              ".\\.venv-ocr\\Scripts\\python.exe scripts/unattended_e10_confusable_repair.py",
              "```", "", "明细：`repair.json`。API 0、训练 0、GT 未修改。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "repair.json").write_text(json.dumps({"tally": tally, "by_tag": per_tag,
                                                 "examples": examples, "seconds": seconds},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e10_confusable_repair", "api_calls": 0,
                                                 "tally": tally, "seconds": round(seconds, 1)},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"tally": tally, "seconds": round(seconds, 1)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
