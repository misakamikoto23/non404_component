"""Run frozen model inputs on grouped DEV images only; never read target JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.models import YoloDetector, ResnetClassifier


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", type=Path, default=ROOT / "configs/contest6_group_split_v1.json")
    p.add_argument("--archive", type=Path, default=next(ROOT.parent.glob("*.zip")))
    p.add_argument("--device", default="0")
    args = p.parse_args()
    split = json.loads(args.split.read_text("utf-8"))
    cases = split["partitions"]["dev"]  # no test inference or test GT reads
    args.output.mkdir(parents=True, exist_ok=True)
    yolo = ROOT / "weights/pcbnet_component_yolo26n_review_round_004.pt"
    resnet = ROOT / "weights/resnet18_component_balanced_v1.pt"
    settings = {"yolo_sha256": digest(yolo), "resnet_sha256": digest(resnet),
                "imgsz": 1280, "conf": .25, "tile_size": 1024, "tile_overlap": .25,
                "classifier_abstain": .8, "crop_margin": "8_percent_min2px",
                "partition": "dev", "split_sha256": digest(args.split), "uses_gt_for_prediction": False,
                "checkpoint_contamination": "existing weights are contaminated on this new split"}
    detector, classifier = None, None
    jobs = []
    with zipfile.ZipFile(args.archive) as archive:
        for case in cases:
            ident = case["case"]
            directory = args.output.resolve() / ident
            directory.mkdir(exist_ok=True)
            image_path = directory / "source.png"
            matches = [n for n in archive.namelist() if f"/200_train_cases/{ident}/" in n and n.lower().endswith(".png")]
            if len(matches) != 1:
                raise ValueError(f"Need unique official image: {ident}: {matches}")
            if not image_path.exists():
                image_path.write_bytes(archive.read(matches[0]))
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            width, height = image.size
            pixel_hash = hashlib.sha256(f"{width}x{height}:RGB:".encode() + image.tobytes()).hexdigest()
            if pixel_hash != case["pixel_sha256"]:
                raise ValueError(f"Source pixels disagree with frozen split for {ident}")
            target = directory / "components.json"
            provenance = {**settings, "image_sha256": digest(image_path), "pixel_sha256": pixel_hash}
            if target.exists():
                cached = json.loads(target.read_text("utf-8"))
                if cached["provenance"] != provenance:
                    raise ValueError(f"Refusing incompatible model cache: {target}")
            else:
                if detector is None:
                    detector = YoloDetector(str(yolo), args.device, 1280, .25, tile_size=1024, tile_overlap=.25)
                    classifier = ResnetClassifier(str(resnet), args.device)
                started = time.perf_counter()
                rows = detector.predict(image)
                for i, row in enumerate(rows, 1):
                    row["id"] = f"component_{i:04d}"
                    x1, y1, x2, y2 = row["bbox"]
                    dx, dy = max(2, .08*(x2-x1)), max(2, .08*(y2-y1))
                    box = [max(0, int(x1-dx)), max(0, int(y1-dy)), min(width, int(x2+dx+1)), min(height, int(y2+dy+1))]
                    prediction = classifier.predict(image.crop(box))
                    row.update(classification=prediction, classification_crop=box,
                               type=prediction["type"] if prediction["score"] >= .8 else "component")
                save(target, {"image": str(image_path), "components": rows, "provenance": provenance,
                              "inference": detector.inference_metadata, "seconds": time.perf_counter()-started})
            jobs.append({"id": ident, "image": str(image_path), "detections": str(target),
                         "ocr": str(directory / "ocr_rapidocr.json")})
            save(args.output / "manifest.json", {"jobs": jobs, "settings": settings,
                                                  "expected_cases": len(cases), "complete": len(jobs) == len(cases)})
            print(f"{ident}: cached model input ready ({len(jobs)}/{len(cases)})", flush=True)


if __name__ == "__main__":
    main()
