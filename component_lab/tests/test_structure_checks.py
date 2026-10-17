"""Conservative structural evidence tests; synthetic geometry, CPU only."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from component_lab.structure_checks import annotate_image
from recheck_structure_regions import fuse_preserving_baseline, select_regions


def row(identifier, box, pass_id="whole_image"):
    return {"candidate_id": identifier, "bbox": box, "score": .8, "class_id": 0,
            "pass_id": pass_id, "source_pass": "whole_image" if pass_id == "whole_image" else "tile",
            "edge": {"touches_internal": False}, "provenance": {"keep": "original"}}


class StructureChecksTests(unittest.TestCase):
    def canvas(self):
        return np.full((600, 800, 3), 255, np.uint8)

    def test_default_retains_even_corroborated_page_frame(self):
        image = self.canvas()
        cv2.rectangle(image, (20, 20), (780, 580), (0, 0, 0), 2)
        candidates = [row("frame", [20, 20, 780, 580])]
        original = deepcopy(candidates)
        result = annotate_image(image, candidates)
        self.assertEqual(candidates, original)
        self.assertEqual(len(result["predictions"]), 1)
        self.assertTrue(result["predictions"][0]["structure"]["frame"]["corroborated_page_frame_hypothesis"])
        self.assertEqual(result["original_candidates"], original)

    def test_optional_guard_requires_four_actual_long_sides(self):
        image = self.canvas()
        candidates = [row("large", [20, 20, 780, 580])]
        self.assertEqual(len(annotate_image(image, candidates, "frame_guard")["predictions"]), 1)
        cv2.rectangle(image, (20, 20), (780, 580), (0, 0, 0), 2)
        result = annotate_image(image, candidates, "frame_guard")
        self.assertEqual(result["predictions"], [])
        self.assertEqual(len(result["annotated_candidates"]), 1)
        self.assertEqual(result["original_candidates"], candidates)
        self.assertTrue(result["suppressed"][0]["reversible"])

    def test_ic_rectangle_interior_is_protected(self):
        image = self.canvas()
        cv2.rectangle(image, (220, 160), (560, 440), (0, 0, 0), 2)
        result = annotate_image(image, [row("ic", [220, 160, 560, 440])], "frame_guard")
        self.assertEqual(len(result["predictions"]), 1)
        frame = result["predictions"][0]["structure"]["frame"]
        self.assertTrue(frame["four_sides_supported"])
        self.assertTrue(frame["rectangular_symbol_protection"])

    def test_large_ic_with_repeated_boundary_pins_is_protected(self):
        image = self.canvas()
        cv2.rectangle(image, (50, 50), (750, 550), (0, 0, 0), 2)
        for y in (100, 150, 200, 250):
            cv2.line(image, (25, y), (75, y), (0, 0, 0), 2)
        result = annotate_image(image, [row("largeic", [50, 50, 750, 550])], "frame_guard")
        self.assertEqual(len(result["predictions"]), 1)
        self.assertTrue(result["predictions"][0]["structure"]["frame"]["boundary_strokes"]["repeated_boundary_strokes"])

    def test_connector_pin_row_never_unioned(self):
        image = self.canvas()
        candidates = [row(f"pin_{i}", [100 + i * 22, 100, 118 + i * 22, 160]) for i in range(6)]
        result = annotate_image(image, candidates, "frame_guard")
        self.assertEqual([item["bbox"] for item in result["predictions"]], [item["bbox"] for item in candidates])
        self.assertEqual(result["counts"]["suppressed"], 0)

    def test_containment_multiple_small_boxes_is_ambiguous_and_retained(self):
        candidates = [row("big", [100, 100, 400, 400]), row("a", [120, 120, 170, 170]),
                      row("b", [220, 220, 270, 270])]
        result = annotate_image(self.canvas(), candidates, "frame_guard")
        self.assertEqual(len(result["predictions"]), 3)
        self.assertEqual(len(result["review_groups"]), 1)
        self.assertIn("one_large_box_multiple_smaller_candidates_unresolved",
                      result["predictions"][0]["structure"]["review_reasons"])

    def test_one_ocr_reference_does_not_remove_boxes(self):
        candidates = [row("a", [100, 100, 200, 200]), row("b", [120, 120, 180, 180])]
        texts = [{"text": "U1", "bbox": [105, 90, 130, 99]}]
        result = annotate_image(self.canvas(), candidates, "frame_guard", texts)
        self.assertEqual(len(result["predictions"]), 2)
        self.assertEqual(result["predictions"][0]["structure"]["nearby_reference_candidates"][0]["text"], "U1")

    def test_subregion_border_retained(self):
        image = self.canvas()
        cv2.rectangle(image, (100, 50), (650, 450), (100, 180, 100), 2)
        result = annotate_image(image, [row("region", [100, 50, 650, 450])], "frame_guard")
        self.assertEqual(len(result["predictions"]), 1)

    def test_invalid_policy_or_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            annotate_image(self.canvas(), [], "delete_all")
        with self.assertRaises(ValueError):
            annotate_image(self.canvas(), [row("x", [1, 1, 10, 10]), row("x", [20, 20, 30, 30])])

    def test_fusion_does_not_drop_original_for_higher_score_local(self):
        base = row("base", [100, 100, 200, 200])
        new = row("local", [101, 101, 201, 201], "region")
        base["valid"] = new["valid"] = True
        new["score"] = .99
        fused, suppressed = fuse_preserving_baseline([base], [new], .25)
        self.assertEqual([entry["candidate_id"] for entry in fused], ["base"])
        self.assertEqual(suppressed[0]["candidate_id"], "local")
        self.assertEqual(fused[0]["bbox"], base["bbox"])

    def test_local_edge_fragments_retained_raw_but_not_added(self):
        fragment = row("local", [100, 100, 200, 200], "region")
        fragment["valid"] = True
        fragment["edge"]["touches_internal"] = True
        fused, suppressed = fuse_preserving_baseline([], [fragment], .10)
        self.assertEqual(fused, [])
        self.assertEqual(fragment["bbox"], [100, 100, 200, 200])
        self.assertIn("raw_retained", suppressed[0]["reason"])

    def test_local_region_selection_bounded_without_ground_truth(self):
        raw = {"images": {}, "predictions": {}}
        audit = {}
        for image_index in range(16):
            key = f"images/val/{image_index:04d}.png"
            raw["images"][key] = {"width": 1000, "height": 1000}
            rows = []
            for group_index in range(4):
                x = 100 + 180 * group_index
                rows.extend([row(f"{image_index}_{group_index}_a", [x, 100, x + 100, 250]),
                             row(f"{image_index}_{group_index}_b", [x + 10, 110, x + 70, 180], "tile")])
            raw["predictions"][key] = rows
            audit[key] = annotate_image(np.full((1000, 1000, 3), 255, np.uint8), rows)
        selected = select_regions(raw, audit)
        self.assertEqual(len(selected), 24)
        self.assertTrue(all(sum(region["image"] == key for region in selected) <= 2 for key in raw["images"]))
        self.assertTrue(all(not region["selection_uses_ground_truth"] for region in selected))
        self.assertTrue(all(region["window_image_area_fraction"] <= .8 for region in selected))

    def test_local_selection_rejects_mismatched_audit_geometry(self):
        candidates = [row("a", [100, 100, 300, 300]), row("b", [120, 120, 150, 150])]
        raw = {"images": {"images/val/x.png": {"width": 800, "height": 600}},
               "predictions": {"images/val/x.png": candidates}}
        result = annotate_image(self.canvas(), candidates)
        result["original_candidates"][0]["bbox"][0] = 99
        with self.assertRaises(ValueError):
            select_regions(raw, {"images/val/x.png": result})


if __name__ == "__main__":
    unittest.main()
