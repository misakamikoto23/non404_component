"""Evaluate a selected component detector on a declared PCBnet split."""
import argparse
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=root / "weights/pcbnet_component_yolo26n_v1.pt")
    parser.add_argument("--data", type=Path, default=root / "data/yolo/pcbnet_component_v1/dataset.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=root / "outputs/pcbnet_component_final_test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if args.imgsz <= 0 or args.batch <= 0:
        parser.error("imgsz and batch must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "outputs/ultralytics_settings"))
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    if list(model.names.values()) != ["component"]:
        raise ValueError("Selected weights must detect only the component class")
    metrics = model.val(data=str(args.data.resolve()), split=args.split, device=args.device,
                        imgsz=args.imgsz, batch=args.batch, workers=0, plots=True, max_det=1000,
                        project=str(args.output.parent.resolve()), name=args.output.name, exist_ok=True)
    payload = {"split": args.split, "weights": str(args.weights.resolve()),
               "data": str(args.data.resolve()), "imgsz": args.imgsz, "batch": args.batch,
               "results": {name: float(value) for name, value in metrics.results_dict.items()}}
    (args.output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
