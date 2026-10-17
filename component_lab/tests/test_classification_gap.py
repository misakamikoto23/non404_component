"""CPU matching tests; no image inference or model dependencies."""
import importlib.util
from pathlib import Path
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gap, aware = load("analyze_classification_gap"), load("analyze_yolo_errors")


def target(box, cls=0):
    return {"bbox": box, "class_id": cls}


def pred(box, cls=0, score=.9):
    return {**target(box, cls), "score": score}


class ClassificationGapTests(unittest.TestCase):
    def test_wrong_class_can_still_be_localized(self):
        gt, predictions = [target([0, 0, 10, 10])], [pred([0, 0, 10, 10], cls=1)]
        geometry = gap.location_matches(gt, predictions)
        self.assertEqual(len(geometry["matches"]), 1)
        self.assertFalse(geometry["matches"][0]["class_correct"])
        self.assertEqual(len(aware.match_detections(gt, predictions)["matches"]), 0)

    def test_confidence_order_duplicate_and_threshold(self):
        gt = [target([0, 0, 10, 10])]
        predictions = [pred([0, 0, 10, 10], score=.8), pred([0, 0, 10, 10], cls=1, score=.95),
                       pred([0, 0, 10, 10], score=.2)]
        result = gap.location_matches(gt, predictions)
        self.assertEqual(result["matches"][0]["prediction_index"], 1)
        self.assertEqual(result["false_positive_indices"], [0])
        self.assertEqual(result["false_negative_indices"], [])
        self.assertEqual(result["evaluated_predictions"], [1, 0])

    def test_class_agnostic_competition_can_reduce_total_matches(self):
        gt = [target([0, 0, 10, 10], 0), target([2, 0, 12, 10], 1)]
        predictions = [pred([0, 0, 10, 10], cls=1, score=.9), pred([-2, 0, 8, 10], cls=0, score=.8)]
        # First geometry prediction claims class-0 GT at IoU 1, leaving no
        # adequate GT for the second. Class-aware matching can match both.
        geometry = gap.location_matches(gt, predictions)
        class_result = aware.match_detections(gt, predictions)
        self.assertEqual(len(geometry["matches"]), 1)
        self.assertEqual(len(class_result["matches"]), 2)
        self.assertFalse(geometry["matches"][0]["class_correct"])

    def test_confusion_summary_and_absent_class(self):
        gt = [target([0, 0, 10, 10], 0), target([30, 0, 40, 10], 1)]
        predictions = [pred([0, 0, 10, 10], cls=1), pred([30, 0, 40, 10], cls=1)]
        record = {"image": "images/val/a.png", "ground_truth": gt, "predictions": predictions,
                  "location_matching": gap.location_matches(gt, predictions),
                  "class_aware_matching": aware.match_detections(gt, predictions)}
        report = gap.build_report([record], ["R", "C", "absent"])
        self.assertEqual(report["overall"]["location_only_recall"], 1)
        self.assertEqual(report["overall"]["class_aware_recall"], .5)
        self.assertEqual(report["overall"]["class_accuracy_given_geometry_match"], .5)
        self.assertEqual(report["confusion_matrix_gt_rows_prediction_columns"], [[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertIsNone(report["by_class"][2]["location_recall"])
        self.assertEqual(len(report["wrong_class_examples"]), 1)

    def test_no_match_accuracy_is_undefined_and_bad_iou_rejected(self):
        self.assertIsNone(gap.build_report([], ["R"])["overall"]["class_accuracy_given_geometry_match"])
        with self.assertRaises(ValueError):
            gap.location_matches([], [], iou_threshold=0)


if __name__ == "__main__":
    unittest.main()
