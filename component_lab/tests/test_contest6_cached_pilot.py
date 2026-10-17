"""Prediction export boundaries: raw OCR, coordinate conversion, no dropped boxes."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/contest6_cached_component_pilot.py"
spec = importlib.util.spec_from_file_location("cached_pilot", path)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class ExportTests(unittest.TestCase):
    def component(self, reference="R1"):
        return {"type": "resistor", "bbox": [10, 20, 30, 40], "value": "10k",
                "association_evidence": {"reference": {"raw": reference}, "value": {"raw": "10 kΩ"}}}

    def test_raw_text_and_bottom_left(self):
        result, _ = pilot.export_components({"components": [self.component()]}, 100)
        self.assertEqual(result["components"]["R1"]["bbox"], [10, 60, 30, 80])
        self.assertEqual(result["components"]["R1"]["value"], "10 kΩ")

    def test_duplicate_and_missing_references_keep_all_detections(self):
        result, trace = pilot.export_components({"components": [self.component(), self.component(), self.component(None)]}, 100)
        self.assertEqual(len(result["components"]), 3)
        self.assertNotIn("R1", result["components"])
        self.assertEqual(trace["issues"]["duplicate_reference"], 2)

    def test_no_evidence_does_not_copy_normalized_value(self):
        row = self.component()
        row["association_evidence"].pop("value")
        result, _ = pilot.export_components({"components": [row]}, 100)
        self.assertIsNone(result["components"]["R1"]["value"])

    def test_invalid_coordinate_rejected(self):
        with self.assertRaises(ValueError):
            pilot.export_components({"components": [self.component()]}, 30)
