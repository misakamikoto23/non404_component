"""Pure CPU tests of diagnostic matching; no model inference or training."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_yolo_errors.py"
SPEC = importlib.util.spec_from_file_location("analyze_yolo_errors", SCRIPT)
errors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(errors)


def target(box, cls=0):
    return {"bbox": box, "class_id": cls}


def prediction(box, cls=0, score=.9):
    return {**target(box, cls), "score": score}


class YoloErrorTests(unittest.TestCase):
    def test_iou_and_original_area_boundaries(self):
        self.assertEqual(errors.box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1/3)
        self.assertEqual(errors.box_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0)
        self.assertEqual(errors.area_bucket([0, 0, 31, 32]), errors.AREA_BUCKETS[0])
        self.assertEqual(errors.area_bucket([0, 0, 32, 32]), errors.AREA_BUCKETS[1])
        self.assertEqual(errors.area_bucket([0, 0, 96, 96]), errors.AREA_BUCKETS[2])

    def test_duplicate_prediction_is_one_tp_and_one_fp(self):
        gt = [target([0, 0, 20, 20])]
        preds = [prediction([0, 0, 20, 20], score=.5), prediction([0, 0, 20, 20], score=.9)]
        result = errors.match_detections(gt, preds)
        self.assertEqual(result["matches"][0]["prediction_index"], 1)
        self.assertEqual(result["false_positive_indices"], [0])
        self.assertEqual(result["false_negative_indices"], [])

    def test_wrong_class_is_fp_and_fn_low_conf_is_ignored(self):
        gt = [target([0, 0, 20, 20])]
        preds = [prediction([0, 0, 20, 20], cls=1), prediction([0, 0, 20, 20], score=.2)]
        result = errors.match_detections(gt, preds)
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["false_positive_indices"], [0])
        self.assertEqual(result["false_negative_indices"], [0])

    def test_greedy_matches_best_unclaimed_gt_and_respects_iou(self):
        gt = [target([0, 0, 10, 10]), target([2, 0, 12, 10])]
        preds = [prediction([0, 0, 10, 10]), prediction([1, 0, 11, 10], score=.8)]
        result = errors.match_detections(gt, preds)
        self.assertEqual([m["ground_truth_index"] for m in result["matches"]], [0, 1])
        stricter = errors.match_detections(gt, preds, iou_threshold=.9)
        self.assertEqual(stricter["false_negative_indices"], [1])
        self.assertEqual(stricter["false_positive_indices"], [1])

    def test_summary_counts_by_class_and_gt_size(self):
        gt = [target([0, 0, 16, 16]), target([0, 0, 40, 40], 1)]
        preds = [prediction([0, 0, 16, 16]), prediction([100, 100, 120, 120], 1)]
        records = [{"width": 1280, "height": 1280, "ground_truth": gt, "predictions": preds,
                    "matching": errors.match_detections(gt, preds)}]
        summary = errors.summarize(records, ["R", "C", "absent"])
        self.assertEqual(summary["overall"], errors.count_summary(1, 1, 1))
        self.assertEqual(summary["by_class"][2]["recall"], None)
        self.assertEqual(summary["by_original_area"][0]["recall"], 1)
        self.assertEqual(summary["by_original_area"][1]["recall"], 0)
        self.assertEqual(summary["by_original_area"][2]["recall"], None)
        self.assertEqual(summary["by_estimated_resized_short_side"][3]["recall"], .5)

    def test_scaled_side_boundaries_follow_original_aspect_ratio(self):
        # 2560x1280 to imgsz=1280 gives scale .5, regardless of padding.
        for original_side, bucket in ((7, 0), (8, 1), (16, 2), (32, 3)):
            self.assertEqual(errors.scaled_side_bucket([0, 0, 100, original_side], 2560, 1280, 1280),
                             errors.SCALED_SIDE_BUCKETS[bucket])

    def test_cache_must_cover_all_val_images_and_class_order(self):
        with patch.object(Path, "read_text") as read_text:
            path = Path("cache.json")
            record = {"image": "images/val/a.png"}
            read_text.return_value = json.dumps({"predictions": {}})
            with self.assertRaisesRegex(ValueError, "missing validation"):
                errors.load_predictions(path, [record], ["R"], .25)
            read_text.return_value = json.dumps({"names": ["C"], "predictions": {record["image"]: []}})
            with self.assertRaisesRegex(ValueError, "class order"):
                errors.load_predictions(path, [record], ["R"], .25)
            read_text.return_value = json.dumps({"confidence": .5, "predictions": {record["image"]: []}})
            with self.assertRaisesRegex(ValueError, "higher confidence"):
                errors.load_predictions(path, [record], ["R"], .25)
            read_text.return_value = json.dumps({"split": "test", "predictions": {record["image"]: []}})
            with self.assertRaisesRegex(ValueError, "cache split"):
                errors.load_predictions(path, [record], ["R"], .25)
            read_text.return_value = json.dumps({record["image"]: []})
            self.assertEqual(errors.load_predictions(path, [record], ["R"], .25), {record["image"]: []})

    def test_split_loader_and_cached_cli_mark_val_and_test_correctly(self):
        from PIL import Image

        # Use the writable test directory rather than the Windows sandbox's
        # temporary directory, which can deny writes to generated children.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "dataset.yaml").write_text("names: [component]\nval: images/val\ntest: images/test\n", encoding="utf-8")
            for split in ("val", "test"):
                (root / f"images/{split}").mkdir(parents=True)
                (root / f"labels/{split}").mkdir(parents=True)
                Image.new("RGB", (100, 80), "white").save(root / f"images/{split}/{split}.png")
                (root / f"labels/{split}/{split}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            names, val_records = errors.load_validation(root)
            self.assertEqual(names, ["component"])
            self.assertEqual([row["image"] for row in val_records], ["images/val/val.png"])
            for split in ("val", "test"):
                cache = root / f"{split}_cache.json"
                cache.write_text(json.dumps({"split": split, "names": ["component"],
                                            "predictions": {f"images/{split}/{split}.png": [
                                                {"bbox": [25, 20, 75, 60], "class_id": 0, "score": .9}]}}), encoding="utf-8")
                output = root / f"output_{split}"
                argv = ["--data-root", str(root), "--predictions-json", str(cache), "--output", str(output), "--visualizations", "0"]
                if split == "test":
                    argv += ["--split", "test"]
                with patch.object(errors, "predict_validation", side_effect=AssertionError("No model should run")):
                    self.assertEqual(errors.main(argv), 0)
                self.assertEqual(json.loads((output / "metrics.json").read_text(encoding="utf-8"))["split"], split)
                self.assertEqual(json.loads((output / "predictions.json").read_text(encoding="utf-8"))["split"], split)
                readme = (output / "README.md").read_text(encoding="utf-8")
                self.assertIn(f"Split: {split}.", readme)
                self.assertIn(f"Only {split} images and labels", readme)
            with self.assertRaisesRegex(ValueError, "Only val and test"):
                errors.load_split(root, "train")

    def test_invalid_boxes_and_thresholds_are_rejected(self):
        for box in ([0, 0, 0, 2], [0, 0, float("nan"), 2], [0, 1]):
            with self.assertRaises(ValueError):
                errors.valid_box(box)
        with self.assertRaises(ValueError):
            errors.match_detections([], [], iou_threshold=0)


if __name__ == "__main__":
    unittest.main()
