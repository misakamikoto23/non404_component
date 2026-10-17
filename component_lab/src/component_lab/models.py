"""Optional model dependencies are loaded only when their stage is invoked."""
import json
import os
from pathlib import Path
from PIL import ImageOps
from .tiling import validate_settings, tile_plan, touches_internal_edge, translate_prediction, class_aware_nms


def configure_yolo():
    directory = Path(__file__).resolve().parents[2] / "outputs/ultralytics_settings"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(directory))


def transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Lambda(lambda im: ImageOps.pad(im.convert("RGB"), (224, 224), color="white")),
        transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


class YoloDetector:
    def __init__(self, weights, device="cpu", imgsz=1280, confidence=.25,
                 tile_size=None, tile_overlap=.25, tile_max_tiles=100):
        validate_settings(tile_size, tile_overlap, tile_max_tiles)
        configure_yolo()
        from ultralytics import YOLO
        self.model, self.device, self.imgsz, self.confidence = YOLO(weights), device, imgsz, confidence
        self.tile_size, self.tile_overlap, self.tile_max_tiles = tile_size, tile_overlap, tile_max_tiles
        self.inference_metadata = {"mode": "whole_image"}

    def _predict_rows(self, image, source="whole_image"):
        options = {"max_det": 1000} if self.tile_size is not None else {}
        result = self.model.predict(image, device=self.device, imgsz=self.imgsz, conf=self.confidence,
                                    verbose=False, **options)[0]
        rows = []
        for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            label = result.names[int(cls)]
            rows.append({"bbox": box, "class_id": int(cls), "type": label, "score": score, "source": source})
        return rows

    def predict(self, image):
        whole = self._predict_rows(image)
        if self.tile_size is None:
            predictions = whole
            self.inference_metadata = {"mode": "whole_image"}
        else:
            width, height = image.size
            plan = tile_plan(width, height, self.tile_size, self.tile_overlap, self.tile_max_tiles)
            candidates = list(whole)
            edge_discarded, invalid_discarded = 0, 0
            for window in plan["windows"]:
                for row in self._predict_rows(image.crop(tuple(window)), "tile"):
                    if touches_internal_edge(row["bbox"], window, width, height):
                        edge_discarded += 1
                        continue
                    translated = translate_prediction(row, window, width, height)
                    if translated is None:
                        invalid_discarded += 1
                    else:
                        candidates.append(translated)
            # Preserve the exact baseline on capped images, including duplicates.
            predictions = whole if plan["skipped"] else class_aware_nms(candidates)
            self.inference_metadata = {"mode": "whole_image_plus_tiles", **plan,
                "tile_size_original_pixels": self.tile_size, "overlap": self.tile_overlap,
                "max_tiles": self.tile_max_tiles, "model_max_det_per_call": 1000,
                "merge_nms_iou": .5, "edge_margin_original_tile_pixels": 2,
                "whole_detection_count": len(whole), "pre_nms_detection_count": len(candidates),
                "combined_detection_count": len(predictions), "internal_edge_discarded": edge_discarded,
                "invalid_box_discarded": invalid_discarded}
        return [{"bbox": row["bbox"], "type": row["type"],
                 "detection": {"type": row["type"], "score": row["score"]}, "classification": None}
                for row in predictions]


class ResnetClassifier:
    def __init__(self, weights, device="cpu"):
        import torch
        from torchvision.models import resnet18
        device = f"cuda:{device}" if str(device).isdigit() else device
        self.torch, self.device = torch, device
        checkpoint = torch.load(weights, map_location=device, weights_only=True)
        self.classes = checkpoint["classes"]
        self.model = resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()
        self.preprocess = transform()

    def predict(self, image):
        with self.torch.inference_mode():
            probs = self.model(self.preprocess(image).unsqueeze(0).to(self.device)).softmax(1)[0]
        score, index = probs.max(0)
        return {"type": self.classes[int(index)], "score": float(score)}


def paddle_texts(image_path):
    from paddleocr import PaddleOCR
    engine = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    rows = []
    for result in engine.predict(str(image_path)):
        payload = result.json
        if isinstance(payload, str):
            payload = json.loads(payload)
        payload = payload.get("res", payload)
        for text, score, poly in zip(payload["rec_texts"], payload["rec_scores"], payload["rec_polys"]):
            xs, ys = zip(*poly)
            rows.append({"text": text, "score": float(score), "bbox": [min(xs), min(ys), max(xs), max(ys)]})
    return rows
