"""Production tiled inference tests: fake results only, no torch or GPU."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from component_lab.models import YoloDetector
from component_lab import tiling


def result(rows):
    def tensor(values):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))
    boxes = SimpleNamespace(xyxy=tensor([row[0] for row in rows]),
                            cls=tensor([row[1] for row in rows]),
                            conf=tensor([row[2] for row in rows]))
    return [SimpleNamespace(boxes=boxes, names={0: "R", 1: "C"})]


class InferTilingTests(unittest.TestCase):
    def detector(self, predictions, **options):
        model = SimpleNamespace(predict=Mock(side_effect=predictions))
        factory = Mock(return_value=model)
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=factory)}), \
             patch("component_lab.models.configure_yolo"):
            detector = YoloDetector("mock.pt", **options)
        factory.assert_called_once_with("mock.pt")
        return detector, model

    def test_whole_image_default_keeps_existing_output_and_one_call(self):
        detector, model = self.detector([result([([10, 20, 30, 40], 0, .9)])])
        predictions = detector.predict(Image.new("RGB", (1792, 600)))
        self.assertEqual(predictions, [{"bbox": [10, 20, 30, 40], "type": "R",
            "detection": {"type": "R", "score": .9}, "classification": None}])
        model.predict.assert_called_once()
        self.assertNotIn("max_det", model.predict.call_args.kwargs)
        self.assertEqual(detector.inference_metadata, {"mode": "whole_image"})

    def test_tiles_translate_filter_seams_and_merge_by_class(self):
        detector, model = self.detector([
            result([([100, 100, 140, 140], 0, .8)]),
            result([([100, 100, 140, 140], 0, .9), ([1000, 10, 1023, 40], 0, .99),
                    ([100, 100, 140, 140], 1, .7)]),
            result([([100, 100, 140, 140], 0, .95), ([1, 30, 20, 50], 0, .99),
                    ([100, 100, float("nan"), 140], 0, .6)])], tile_size=1024)
        predictions = detector.predict(Image.new("RGB", (1792, 600)))
        self.assertEqual([row["bbox"] for row in predictions],
                         [[868, 100, 908, 140], [100, 100, 140, 140], [100, 100, 140, 140]])
        self.assertEqual([row["type"] for row in predictions], ["R", "R", "C"])
        self.assertEqual(model.predict.call_count, 3)
        self.assertEqual([call.args[0].size for call in model.predict.call_args_list],
                         [(1792, 600), (1024, 600), (1024, 600)])
        for call in model.predict.call_args_list:
            self.assertEqual(call.kwargs["imgsz"], 1280)
            self.assertEqual(call.kwargs["max_det"], 1000)
        metadata = detector.inference_metadata
        self.assertEqual(metadata["tile_count"], 2)
        self.assertEqual(metadata["internal_edge_discarded"], 2)
        self.assertEqual(metadata["invalid_box_discarded"], 1)
        self.assertEqual(metadata["combined_detection_count"], 3)

    def test_tile_limit_keeps_exact_whole_predictions_without_extra_nms(self):
        whole = [([10, 20, 30, 40], 0, .9), ([10, 20, 30, 40], 0, .8)]
        detector, model = self.detector([result(whole)], tile_size=1024, tile_max_tiles=5)
        predictions = detector.predict(Image.new("RGB", (2560, 1280)))
        self.assertEqual(len(predictions), 2)
        self.assertEqual([r["detection"]["score"] for r in predictions], [.9, .8])
        model.predict.assert_called_once()
        self.assertTrue(detector.inference_metadata["skipped"])
        self.assertFalse(detector.inference_metadata["truncated"])
        self.assertEqual(detector.inference_metadata["required_tile_count"], 6)
        self.assertEqual(detector.inference_metadata["tile_count"], 0)

    def test_invalid_settings_fail_before_loading_model(self):
        for options in ({"tile_size": 0}, {"tile_size": -1}, {"tile_size": True},
                        {"tile_overlap": 1}, {"tile_overlap": -.1}, {"tile_overlap": float("nan")},
                        {"tile_max_tiles": 0}):
            with self.subTest(options=options), patch("component_lab.models.configure_yolo") as configure:
                with self.assertRaises(ValueError):
                    YoloDetector("must-not-load.pt", **options)
                configure.assert_not_called()

    def test_geometry_matches_evaluation_implementation(self):
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_tiled_yolo.py"
        spec = importlib.util.spec_from_file_location("tiled_evaluation_reference", script)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        for width, height, cap in ((100, 80, 100), (3000, 2000, 100), (2560, 1280, 5)):
            self.assertEqual(tiling.tile_plan(width, height, max_tiles=cap),
                             reference.tile_plan(width, height, max_tiles=cap))
        for window in ([0, 0, 1024, 1024], [768, 768, 1792, 1792], [976, 976, 2000, 2000]):
            for box in ([0, 0, 20, 20], [2, 20, 50, 50], [1000, 1000, 1024, 1024]):
                self.assertEqual(tiling.touches_internal_edge(box, window, 2000, 2000),
                                 reference.touches_internal_edge(box, window, 2000, 2000))
                prediction = {"bbox": box, "score": .9, "class_id": 0}
                self.assertEqual(tiling.translate_prediction(prediction, window, 2000, 2000),
                                 reference.translate_prediction(prediction, window, 2000, 2000))
        rows = [{"bbox": [0, 0, 10, 10], "score": .9, "class_id": 0},
                {"bbox": [0, 0, 10, 10], "score": .9, "class_id": 0},
                {"bbox": [0, 0, 10, 10], "score": .8, "class_id": 1}]
        self.assertEqual(tiling.class_aware_nms(rows), reference.class_aware_nms(rows))

    def test_cli_rejects_invalid_settings_and_cached_tile_mode(self):
        from component_lab.cli import main
        base = ["component-lab", "infer", "--image", "unused.png", "--output", "unused", "--yolo", "mock.pt"]
        for extras in (["--tile-size", "0"], ["--tile-overlap", "nan"],
                       ["--tile-overlap", "1"], ["--tile-max-tiles", "0"]):
            with self.subTest(extras=extras), patch.object(sys, "argv", base + extras), \
                 patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 2)
        cached = ["component-lab", "infer", "--image", "unused.png", "--output", "unused",
                  "--detections", "unused.json", "--tile-size", "1024"]
        with patch.object(sys, "argv", cached), patch("sys.stderr"), self.assertRaises(SystemExit):
            main()

    def test_pipeline_wires_tiles_and_preserves_crops_and_report(self):
        from component_lab.pipeline import infer
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            args = SimpleNamespace(image=str(image_path), output=str(root / "out"), detections=None,
                yolo="mock.pt", device="cpu", imgsz=1280, conf=.25, tile_size=1024, tile_overlap=.25,
                tile_max_tiles=100, resnet=None, classifier_threshold=.8, ocr_json=None, ocr="none",
                text_margin=60, qwen=False, qwen_max_calls=10, context_margin=48)
            detector = SimpleNamespace(predict=Mock(return_value=[{"bbox": [10, 20, 30, 40], "type": "R",
                "detection": {"type": "R", "score": .9}, "classification": None}]),
                inference_metadata={"mode": "whole_image_plus_tiles", "skipped": False})
            with patch("component_lab.pipeline.YoloDetector", return_value=detector) as factory:
                summary = infer(args)
            factory.assert_called_once_with("mock.pt", "cpu", 1280, .25,
                tile_size=1024, tile_overlap=.25, tile_max_tiles=100)
            detector.predict.assert_called_once()
            report = json.loads((root / "out/components.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["components"], 1)
            self.assertEqual(report["inference"]["mode"], "whole_image_plus_tiles")
            component = report["components"][0]
            self.assertEqual(component["id"], "component_0001")
            self.assertTrue(Path(component["crops"]["symbol"]).exists())


if __name__ == "__main__":
    unittest.main()
