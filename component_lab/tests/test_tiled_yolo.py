"""Pure geometry tests, without model imports, prediction or GPU use."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_tiled_yolo.py"
SPEC = importlib.util.spec_from_file_location("evaluate_tiled_yolo", SCRIPT)
tiled = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tiled)


def pred(box, score=.9, cls=0, source="tile"):
    return {"bbox": box, "score": score, "class_id": cls, "source": source}


class TiledYoloTests(unittest.TestCase):
    def test_axes_cover_far_edge_with_fixed_tile_and_overlap(self):
        self.assertEqual(tiled.axis_starts(512, 1024, .25), [0])
        self.assertEqual(tiled.axis_starts(2560, 1024, .25), [0, 768, 1536])
        starts = tiled.axis_starts(3000, 1024, .25)
        self.assertEqual(starts, [0, 768, 1536, 1976])
        self.assertEqual(starts[-1] + 1024, 3000)

    def test_plan_covers_image_and_cap_skips_entire_tiling(self):
        plan = tiled.tile_plan(2560, 1280)
        self.assertEqual(plan["required_tile_count"], 6)
        self.assertEqual(plan["windows"][-1], [1536, 256, 2560, 1280])
        self.assertFalse(plan["skipped"])
        capped = tiled.tile_plan(2560, 1280, max_tiles=5)
        self.assertEqual(capped["required_tile_count"], 6)
        self.assertEqual(capped["windows"], [])
        self.assertEqual(capped["tile_count"], 0)
        self.assertTrue(capped["skipped"])
        self.assertFalse(capped["truncated"])
        self.assertEqual(capped["skip_reason"], "required_tile_count_exceeds_max_tiles")

    def test_outer_boundaries_allowed_and_internal_edges_filtered(self):
        outer = [0, 0, 1024, 1024]
        self.assertFalse(tiled.touches_internal_edge([0, 0, 20, 20], outer, 2000, 2000))
        self.assertTrue(tiled.touches_internal_edge([1000, 10, 1023, 30], outer, 2000, 2000))
        inner = [768, 768, 1792, 1792]
        self.assertTrue(tiled.touches_internal_edge([2, 20, 50, 50], inner, 2000, 2000))
        self.assertFalse(tiled.touches_internal_edge([2.01, 20, 50, 50], inner, 2000, 2000))
        bottom_right = [976, 976, 2000, 2000]
        self.assertFalse(tiled.touches_internal_edge([1000, 1000, 1024, 1024], bottom_right, 2000, 2000))

    def test_translation_uses_crop_coordinates_not_model_input(self):
        translated = tiled.translate_prediction(pred([10, 20, 30, 40]), [768, 256, 1792, 1280], 2560, 1280)
        self.assertEqual(translated["bbox"], [778, 276, 798, 296])
        self.assertEqual(translated["tile_window"], [768, 256, 1792, 1280])
        self.assertEqual(translated["source"], "tile")
        clipped = tiled.translate_prediction(pred([-5, -8, 105, 110]), [0, 0, 100, 100], 100, 100)
        self.assertEqual(clipped["bbox"], [0, 0, 100, 100])
        self.assertIsNone(tiled.translate_prediction(pred([0, 0, 0, 5]), [0, 0, 100, 100], 100, 100))
        self.assertIsNone(tiled.translate_prediction(pred([0, 0, float("nan"), 5]), [0, 0, 100, 100], 100, 100))

    def test_comparison_keeps_fp_cost_and_undefined_recall(self):
        whole = {"overall": {"tp": 10, "fp": 2, "fn": 10, "precision": 10/12, "recall": .5},
                 "by_class": [{"class_id": 0, "name": "missing", "tp": 0, "fp": 0, "fn": 0,
                               "precision": None, "recall": None}],
                 "by_original_area": [], "by_estimated_resized_short_side": []}
        combined = {**whole, "overall": {"tp": 15, "fp": 8, "fn": 5, "precision": 15/23, "recall": .75}}
        comparison = tiled.compare_reports(whole, combined)
        self.assertEqual(comparison["delta"]["tp"], 5)
        self.assertEqual(comparison["delta"]["fp"], 6)
        self.assertEqual(comparison["delta"]["recall"], .25)
        self.assertIsNone(comparison["by_class"][0]["delta"]["recall"])

    def test_nms_is_class_aware_and_uses_confidence(self):
        whole = pred([0, 0, 100, 100], score=.8, source="whole_image")
        tile = pred([0, 0, 100, 100], score=.9)
        different = pred([0, 0, 100, 100], score=.7, cls=1)
        far = pred([200, 200, 300, 300], score=.6)
        self.assertEqual(tiled.class_aware_nms([whole, tile, different, far]), [tile, different, far])
        tie = pred([0, 0, 100, 100], score=.8)
        self.assertEqual(tiled.class_aware_nms([whole, tie]), [whole])

    def test_nms_threshold_is_strict_and_empty_valid(self):
        # IoU is exactly 1/3: suppression happens only above threshold.
        a, b = pred([0, 0, 10, 10]), pred([5, 0, 15, 10], score=.8)
        self.assertEqual(tiled.class_aware_nms([a, b], 1/3), [a, b])
        self.assertEqual(tiled.class_aware_nms([a, b], .3), [a])
        self.assertEqual(tiled.class_aware_nms([]), [])

    def test_invalid_geometry_settings_are_rejected(self):
        for args in ((0, 100, .25), (100, 0, .25), (100, 50, 1)):
            with self.assertRaises(ValueError):
                tiled.axis_starts(*args)
        with self.assertRaises(ValueError):
            tiled.tile_plan(100, 100, max_tiles=0)
        with self.assertRaises(ValueError):
            tiled.class_aware_nms([], 0)

    def test_mocked_cli_defaults_val_and_preserves_test_metadata(self):
        from PIL import Image

        analysis_spec = importlib.util.spec_from_file_location("analyze_yolo_errors", SCRIPT.with_name("analyze_yolo_errors.py"))
        analysis = importlib.util.module_from_spec(analysis_spec)
        analysis_spec.loader.exec_module(analysis)
        fake_model = SimpleNamespace(names={0: "component"}, predict=Mock(return_value=[None]))
        fake_module = SimpleNamespace(YOLO=Mock(return_value=fake_model))
        row = {"bbox": [25, 20, 75, 60], "class_id": 0, "type": "component", "score": .9, "source": "whole_image"}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "dataset.yaml").write_text("names: [component]\nval: images/val\ntest: images/test\n", encoding="utf-8")
            for split in ("val", "test"):
                (root / f"images/{split}").mkdir(parents=True)
                (root / f"labels/{split}").mkdir(parents=True)
                Image.new("RGB", (100, 80), "white").save(root / f"images/{split}/{split}.png")
                (root / f"labels/{split}/{split}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
                output = root / f"output_{split}"
                argv = ["--data-root", str(root), "--output", str(output), "--visualizations", "0"]
                if split == "test":
                    argv += ["--split", "test"]
                with patch.dict(sys.modules, {"ultralytics": fake_module, "analyze_yolo_errors": analysis}), \
                     patch.dict("os.environ"), \
                     patch.object(tiled, "result_rows", return_value=[dict(row)]):
                    self.assertEqual(tiled.main(argv), 0)
                for path in ("metadata.json", "predictions.json", "whole_image_predictions.json", "comparison.json",
                             "whole_image/metrics.json", "combined/metrics.json"):
                    self.assertEqual(json.loads((output / path).read_text(encoding="utf-8"))["split"], split, path)
                self.assertEqual(json.loads((output / "metadata.json").read_text(encoding="utf-8"))["images"][0]["split"], split)
                self.assertIn(f"Split: {split};", (output / "README.md").read_text(encoding="utf-8"))
                self.assertIn(f"images/{split}/{split}.png", json.loads((output / "predictions.json").read_text(encoding="utf-8"))["predictions"])


if __name__ == "__main__":
    unittest.main()
