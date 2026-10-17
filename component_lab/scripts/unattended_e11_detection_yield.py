"""E11: would a finer full-page text-detection setting read more designators?

Three pages (one per source), three settings. Only string presence is measured:
for every observable GT designator, does it appear anywhere in the OCR output?
Coordinate bookkeeping is skipped on purpose - this is a detection-yield probe.
Run with the OCR env.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data/official_contest6/train200"
OCR = ROOT / "outputs/component_identity_round_001_20261017/inputs"
OUT = ROOT / "outputs/unattended_20261018_e11_detection_yield"
WEIGHTS = ROOT / "weights/ocr_rapid"

PAGES = [("0001", "0001-KiCad"), ("0058", "0058-Datasheet"), ("0200", "0200-jlc")]
REF = re.compile(r"^[A-Za-z]{1,8}\d+(?:[A-Za-z]|\.\d+)?$")
CONFIGS = [
    ("baseline_1200_2x_035", dict(tile=1200, stride=1040, scale=2.0, score=0.35)),
    ("A_800_3x_020", dict(tile=800, stride=640, scale=3.0, score=0.20)),
    ("B_600_3x_020", dict(tile=600, stride=480, scale=3.0, score=0.20)),
]


def engine(score):
    from rapidocr import RapidOCR, OCRVersion, ModelType
    return RapidOCR(params={
        "Global.model_root_dir": str(WEIGHTS), "Global.max_side_len": 4096,
        "Global.text_score": score,
        "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
        "Det.limit_side_len": 2048, "Det.limit_type": "max",
        "EngineConfig.onnxruntime.intra_op_num_threads": 4,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False})


def starts(n, tile, stride):
    if n <= tile:
        return [0]
    out = list(range(0, n - tile + 1, stride))
    if out[-1] != n - tile:
        out.append(n - tile)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for case, stem in PAGES:
        gt_path = GT / f"{stem}.json"
        image_path = OCR / case / "source.png"
        if not (gt_path.exists() and image_path.exists()):
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        keys = [k for k in gt["components"] if REF.fullmatch(k)]
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        for name, cfg in CONFIGS:
            tick = time.perf_counter()
            eng = engine(cfg["score"])
            found = set()
            xs = starts(W, cfg["tile"], cfg["stride"])
            ys = starts(H, cfg["tile"], cfg["stride"])
            for y in ys:
                for x in xs:
                    crop = image.crop((x, y, min(W, x + cfg["tile"]), min(H, y + cfg["tile"])))
                    scaled = crop.resize((round(crop.width * cfg["scale"]), round(crop.height * cfg["scale"])),
                                         Image.Resampling.LANCZOS)
                    out = eng(np.asarray(scaled))
                    if out is not None and out.txts:
                        for txt in out.txts:
                            for token in re.split(r"[^A-Za-z0-9.]+", str(txt)):
                                if token:
                                    found.add(token.upper())
            hit = sum(1 for k in keys if k.upper() in found)
            results.append({"case": case, "config": name, "keys": len(keys), "hit": hit,
                            "coverage": hit / len(keys) if keys else 0.0,
                            "tiles": len(xs) * len(ys), "size": [W, H],
                            "seconds": round(time.perf_counter() - tick, 1), "unique_tokens": len(found)})
            print(case, name, f"{hit}/{len(keys)}", f"{results[-1]['coverage']*100:.1f}%",
                  f"{results[-1]['seconds']}s", flush=True)
    lines = ["# E11：更细的整页文字检测能多读出多少位号", "",
             "三张图（kicad / datasheet / jlc），三组设置，只看**字符串是否出现**，不做坐标与归属。",
             "GT 位号仅用于计分；识别过程不看 GT。", "",
             "| case | 配置 | 切片/stride/放大/阈值 | 可见位号 | 读出 | 覆盖率 | 耗时 |", "|---|---|---|---:|---:|---:|---:|"]
    for r in results:
        cfg = next(c for n, c in CONFIGS if n == r["config"])
        lines.append(f"| {r['case']} | {r['config']} | {cfg['tile']}/{cfg['stride']}/{cfg['scale']}×/{cfg['score']} | "
                     f"{r['keys']} | {r['hit']} | {r['coverage']*100:.1f}% | {r['seconds']}s |")
    lines += ["", "## 判读", "",
              "- 若 A/B 的覆盖率明显高于 baseline，说明**现有 det 设置确实吃掉了小字**，值得整页重跑。",
              "- 若三者接近，说明这些位号在当前 det 模型下就是检不出，应换 det 模型（server/v6），而不是调切片。",
              "- 耗时差同时给出成本上界：整页重跑 20 图要多花多少。",
              "", "## 复现", "",
              "```powershell",
              ".\\.venv-ocr\\Scripts\\python.exe scripts/unattended_e11_detection_yield.py",
              "```", "", "API 0、训练 0、GT 未修改。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "yield.json").write_text(json.dumps({"results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"experiment": "e11_detection_yield", "api_calls": 0,
                                                 "pages": len(PAGES), "results": results},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
