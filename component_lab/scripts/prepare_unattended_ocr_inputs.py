"""Prepare model-only OCR association inputs; no component/text GT is read."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from PIL import Image
    from component_lab.models import ResnetClassifier
    from component_lab.review_workflow import atomic_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    source = json.loads(args.cache.read_text(encoding="utf-8-sig"))
    weights = LAB / "weights/resnet18_component_balanced_v1.pt"
    classifier = ResnetClassifier(weights, args.device)
    jobs = []
    for key in sorted(source["predictions"])[:args.count]:
        image_path = Path(source["data_root"]) / key
        ident = Path(key).stem
        destination = args.output.resolve() / ident
        destination.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            rows = []
            for number, old in enumerate(source["predictions"][key], 1):
                x1, y1, x2, y2 = old["bbox"]
                dx, dy = max(2, .08 * (x2-x1)), max(2, .08 * (y2-y1))
                box = [max(0, int(x1-dx)), max(0, int(y1-dy)), min(im.width, int(x2+dx+1)), min(im.height, int(y2+dy+1))]
                prediction = classifier.predict(im.crop(box))
                rows.append({"id": f"component_{number:04d}", "bbox": old["bbox"],
                             "type": prediction["type"] if prediction["score"] >= .8 else "component",
                             "classification": prediction, "detection": {"score": old["score"], "type": "component"},
                             "candidate_id": old.get("candidate_id"), "classification_crop": box})
        atomic_json(destination / "components.json", {"image": str(image_path), "components": rows,
                    "provenance": {"detector_cache_sha256": sha(args.cache), "image_sha256": sha(image_path),
                                   "classifier_sha256": sha(weights), "crop_policy": "8_percent_min2px_context",
                                   "ground_truth_used": False}})
        jobs.append({"id": ident, "image": str(image_path.resolve()),
                     "detections": str(destination / "components.json"),
                     "ocr": str(destination / "ocr_rapidocr.json")})
        print(f"{ident}: {len(rows)} model candidates classified", flush=True)
    atomic_json(args.output / "manifest.json", {"jobs": jobs, "selection": "first count sorted val image keys, no labels",
                                               "test_used": False, "ground_truth_used_for_predictions": False})


if __name__ == "__main__":
    main()
