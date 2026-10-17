"""Traceability/geometry tests with synthetic data; no GPU or API access."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch


LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
sys.path.insert(0, str(LAB / "scripts"))
from component_lab.candidate_postprocess import (COORDINATES, RAW_SCHEMA, conservative_policy, context_neighbors,
                                               dump_json, edge_flags, make_candidate,
                                               maximum_matching_coverage, sha256_file, validate_raw_cache)
import cache_component_candidates as pipeline
from cache_component_candidates import main as pipeline_main, load_plain_images


def candidate(box, *, pass_id="whole_image", source="whole_image", ordinal=0,
              window=None, width=200, height=100, score=.9):
    return make_candidate("images/val/sample.png", pass_id, ordinal,
                          {"bbox": box, "class_id": 0, "type": "component", "score": score},
                          source, window or [0, 0, width, height], width, height, 1280)


class CandidateGeometryTests(unittest.TestCase):
    def test_internal_edge_is_marked_and_never_discarded_from_raw(self):
        row = candidate([0, 20, 10, 40], source="tile", pass_id="tile_0000", window=[50, 0, 150, 100])
        self.assertTrue(row["valid"])
        self.assertEqual(row["bbox"], [50., 20., 60., 40.])
        self.assertEqual(row["edge"]["internal_sides"], ["left"])
        policy = conservative_policy([row])
        self.assertEqual(policy["predictions"][0]["candidate_id"], row["candidate_id"])
        self.assertEqual(policy["counts"]["pending"], 1)

    def test_outer_image_boundary_is_not_an_internal_seam(self):
        flags = edge_flags([0, 0, 10, 20], [0, 0, 100, 100], 200, 100)
        self.assertFalse(flags["touches_internal"])
        self.assertEqual(flags["outer_image_sides"], ["left", "top"])

    def test_coordinate_mapping_uses_native_crop_pixels(self):
        row = candidate([5, 10, 25, 30], source="tile", pass_id="tile_0000", window=[50, 0, 150, 100])
        self.assertEqual(row["bbox"], [55., 10., 75., 30.])
        self.assertEqual(row["coordinate_mapping"]["scale"], [1., 1.])
        self.assertEqual(row["coordinate_mapping"]["offset"], [50, 0])

    def test_invalid_box_is_preserved_as_strict_json_audit_row(self):
        row = candidate([1, 2, float("nan"), 30])
        self.assertFalse(row["valid"])
        self.assertIn("invalid_local_bbox", row["invalid_reasons"])
        json.dumps(row, allow_nan=False)
        policy = conservative_policy([row])
        self.assertEqual(policy["counts"]["raw"], 1)
        self.assertEqual(policy["counts"]["invalid"], 1)
        self.assertEqual(policy["invalid_retained_in_raw"], [row["candidate_id"]])

    def test_exact_duplicate_suppression_keeps_a_provenance_record(self):
        whole = candidate([20, 20, 40, 40], score=.2)
        tile = candidate([20, 20, 40, 40], source="tile", pass_id="tile_0000", score=.99)
        result = conservative_policy([tile, whole])
        self.assertEqual(len(result["predictions"]), 1)
        self.assertEqual(result["predictions"][0]["candidate_id"], whole["candidate_id"])
        self.assertEqual(result["suppressed"][0]["candidate_id"], tile["candidate_id"])
        self.assertEqual(result["suppressed"][0]["reason"], "exact_coordinates_same_class")
        self.assertEqual(tile["bbox"], [20., 20., 40., 40.])

    def test_contained_pin_and_whole_ic_remain_distinct(self):
        ic = candidate([10, 10, 180, 90])
        pin = candidate([20, 20, 30, 30], source="tile", pass_id="tile_0000")
        result = conservative_policy([ic, pin])
        self.assertEqual(len(result["predictions"]), 2)
        self.assertEqual(len(result["suppressed"]), 0)
        self.assertIn("containment_or_near_containment", result["conflict_pairs"][0]["reasons"])
        self.assertEqual(result["review_groups"][0]["action"], "review_only_not_a_merged_component")

    def test_neighboring_ports_are_not_merged(self):
        ports = [candidate([10 + index * 11, 10, 20 + index * 11, 50], ordinal=index) for index in range(5)]
        result = conservative_policy(ports)
        self.assertEqual(len(result["predictions"]), 5)
        self.assertEqual(result["suppressed"], [])
        self.assertEqual(result["review_groups"], [])

    def test_boundary_fragment_cannot_replace_or_suppress_full_box(self):
        whole = candidate([50, 20, 100, 50], score=.1)
        fragment = candidate([0, 20, 49, 50], source="tile", pass_id="tile_0000", window=[50, 0, 150, 100], score=.99)
        result = conservative_policy([fragment, whole])
        self.assertEqual(len(result["predictions"]), 2)
        self.assertTrue(any("boundary_overlap" in pair["reasons"] for pair in result["conflict_pairs"]))

    def test_review_cluster_does_not_union_chained_boxes(self):
        rows = [candidate([10 + index * 15, 10, 40 + index * 15, 40], ordinal=index) for index in range(3)]
        result = conservative_policy(rows)
        self.assertEqual(len(result["predictions"]), 3)
        self.assertEqual(len(result["review_groups"]), 1)
        self.assertEqual(len(result["review_groups"][0]["candidate_ids"]), 3)

    def test_same_pass_near_overlaps_remain_even_when_iou_is_high(self):
        a = candidate([10, 10, 110, 90])
        b = candidate([10.1, 10, 110.1, 90], ordinal=1)
        self.assertEqual(len(conservative_policy([a, b])["predictions"]), 2)

    def test_cross_pass_almost_identical_boxes_can_be_deduplicated(self):
        a = candidate([10, 10, 110, 90])
        b = candidate([10.1, 10, 110.1, 90], source="tile", pass_id="tile_0000")
        result = conservative_policy([a, b])
        self.assertEqual(len(result["predictions"]), 1)
        self.assertEqual(result["suppressed"][0]["reason"], "near_identical_cross_pass_geometry")

    def test_neighbor_limit_prioritizes_containment_overlap_and_distance(self):
        target = candidate([40, 20, 80, 60])
        contained = candidate([45, 25, 50, 30], ordinal=1)
        overlap = candidate([70, 30, 90, 50], ordinal=2)
        nearby = candidate([81, 40, 85, 44], ordinal=3)
        selected, truncated = context_neighbors(target, [nearby, overlap, target, contained], limit=2)
        self.assertEqual([row["id"] for row in selected], [contained["candidate_id"], overlap["candidate_id"]])
        self.assertEqual(truncated, 1)

    def test_diagnostic_coverage_is_one_to_one_and_can_reassign(self):
        gt = [{"bbox": [0, 0, 10, 10], "class_id": 0}, {"bbox": [5, 0, 15, 10], "class_id": 0}]
        # Broad prediction can cover either GT; narrow prediction only the first.
        predictions = [{"bbox": [1, 0, 11, 10], "class_id": 0}, {"bbox": [0, 0, 10, 10], "class_id": 0}]
        report = maximum_matching_coverage(gt, predictions, .4)
        self.assertEqual(report["matched_maximum"], 2)
        self.assertNotIn("matches", report)
        self.assertEqual(maximum_matching_coverage(gt, predictions[:1], .4)["matched_maximum"], 1)
        predictions[1]["class_id"] = 1
        self.assertEqual(maximum_matching_coverage(gt, predictions, .4)["matched_maximum"], 1)


class CandidateReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture_parent = (LAB / "outputs/automation_round_001").resolve()
        self.root = self.fixture_parent / f"test_fixture_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        from PIL import Image

        (self.root / "images/val").mkdir(parents=True)
        (self.root / "labels/val").mkdir(parents=True)
        self.image_path = self.root / "images/val/sample.png"
        Image.new("RGB", (200, 100), "white").save(self.image_path)
        (self.root / "labels/val/sample.txt").write_text("0 .25 .3 .2 .2\n", encoding="utf-8")
        (self.root / "dataset.yaml").write_text("names: [component]\nval: images/val\n", encoding="utf-8")
        rows = [candidate([30, 20, 70, 40]), candidate([30, 20, 70, 40], source="tile", pass_id="tile_0000")]
        self.raw = {"schema": RAW_SCHEMA, "coordinate_system": COORDINATES, "names": ["component"],
                    "split": "val", "capture_complete": True, "data_root": str(self.root),
                    "weights": {"path": str(self.root / "nonexistent_model.pt"), "sha256": "a" * 64},
                    "generation": {"confidence": .03, "imgsz": 1280, "max_det_per_call": 1000},
                    "images": {"images/val/sample.png": {"width": 200, "height": 100,
                                                          "sha256": sha256_file(self.image_path),
                                                          "tile_plan": {"skipped": False}}},
                    "predictions": {"images/val/sample.png": rows}}
        self.raw_path = self.root / "raw_candidates.json"
        dump_json(self.raw_path, self.raw)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.fixture_parent))
        shutil.rmtree(self.root)

    def test_schema_rejects_mapped_coordinate_tampering(self):
        validate_raw_cache(self.raw)
        self.raw["predictions"]["images/val/sample.png"][0]["bbox"][0] += 1
        with self.assertRaisesRegex(ValueError, "Inconsistent coordinate"):
            validate_raw_cache(self.raw)

    def test_schema_rejects_missing_image_hash_and_duplicate_identity(self):
        self.raw["images"]["images/val/sample.png"]["sha256"] = ""
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_raw_cache(self.raw)
        self.raw["images"]["images/val/sample.png"]["sha256"] = "a" * 64
        self.raw["predictions"]["images/val/sample.png"].append(self.raw["predictions"]["images/val/sample.png"][0])
        with self.assertRaisesRegex(ValueError, "globally unique"):
            validate_raw_cache(self.raw)

    def test_replay_evaluates_all_raw_rows_without_loading_model(self):
        output = self.root / "replay"
        with patch.dict(sys.modules, {"torch": None, "ultralytics": None}):
            self.assertEqual(pipeline_main(["replay", "--raw-cache", str(self.raw_path), "--output", str(output)]), 0)
        summary = json.loads((output / "pipeline_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["metrics"]["raw"]["predictions"], 2)
        self.assertEqual(summary["metrics"]["raw"]["fp"], 1)
        self.assertEqual(summary["metrics"]["policy_pre_qwen"]["predictions"], 1)
        final = json.loads((output / "final_predictions.json").read_text(encoding="utf-8"))
        baseline = json.loads((output / "baseline_predictions.json").read_text(encoding="utf-8"))
        self.assertEqual(final["predictions"], baseline["predictions"])
        self.assertFalse(final["qwen_decisions_applied"])
        self.assertEqual(summary["api_calls_performed"], 0)
        queue = [json.loads(line) for line in (output / "qwen_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(queue)
        self.assertEqual(queue[0]["image_sha256"], sha256_file(self.image_path))
        self.assertNotIn("ground_truth", json.dumps(queue))
        self.assertTrue(Path(queue[0]["image_path"]).is_absolute())

    def test_supplied_baseline_rows_are_preserved_even_with_different_cli_threshold(self):
        supplied = {"names": ["component"], "split": "val", "confidence": .05,
                    "predictions": {"images/val/sample.png": [{"bbox": [30, 20, 70, 40], "class_id": 0,
                                                              "score": .06, "original_marker": "retained"}]}}
        stable = self.root / "stable.json"
        dump_json(stable, supplied)
        output = self.root / "stable_replay"
        pipeline_main(["replay", "--raw-cache", str(self.raw_path), "--output", str(output),
                       "--baseline-cache", str(stable), "--baseline-confidence", ".8"])
        final = json.loads((output / "final_predictions.json").read_text(encoding="utf-8"))
        self.assertEqual(final["predictions"], supplied["predictions"])
        self.assertEqual(final["confidence"], .05)

    def test_changed_image_blocks_cache_reuse(self):
        self.image_path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Captured image changed"):
            pipeline_main(["replay", "--raw-cache", str(self.raw_path), "--output", str(self.root / "replay")])

    def test_ordinary_input_does_not_require_dataset_or_claim_PR(self):
        source, records = load_plain_images(self.image_path)
        self.assertEqual(source, self.image_path.parent.resolve())
        self.assertEqual(records[0]["source_relative_path"], "sample.png")
        self.raw["evaluation_available"] = False
        dump_json(self.raw_path, self.raw)
        output = self.root / "no_gt"
        pipeline_main(["replay", "--raw-cache", str(self.raw_path), "--output", str(output)])
        summary = json.loads((output / "pipeline_summary.json").read_text(encoding="utf-8"))
        self.assertIsNone(summary["metrics"])
        self.assertEqual(summary["evaluation_status"], "unavailable_or_explicitly_skipped_no_PR_claim")

    def test_dense_qwen_queue_keeps_candidates_but_caps_neighbors_at40(self):
        target = candidate([30, 20, 170, 90])
        others = [candidate([40 + (i % 10) * 10, 25 + (i // 10) * 7,
                             45 + (i % 10) * 10, 30 + (i // 10) * 7], ordinal=i + 1)
                  for i in range(80)]
        self.raw["predictions"]["images/val/sample.png"] = [target] + others
        dump_json(self.raw_path, self.raw)
        output = self.root / "dense_queue"
        pipeline_main(["replay", "--raw-cache", str(self.raw_path), "--output", str(output), "--skip-evaluation"])
        queue = [json.loads(line) for line in (output / "qwen_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(queue), 81)
        self.assertLessEqual(max(len(row["neighbors"]) for row in queue), 40)
        target_row = next(row for row in queue if row["id"] == target["candidate_id"])
        self.assertEqual(len(target_row["neighbors"]), 40)
        self.assertEqual(target_row["neighbors_truncated_count"], 40)

    def test_capture_ordinary_images_keeps_all_returned_rows_without_labels(self):
        weights = self.root / "fake_weights.pt"
        weights.write_bytes(b"model fixture for hash; no model loaded")
        fake_model = SimpleNamespace(names={0: "component"}, predict=Mock(return_value=[None]))
        fake_module = SimpleNamespace(YOLO=Mock(return_value=fake_model))
        output = self.root / "plain_capture"
        rows = [{"bbox": [30, 20, 70, 40], "class_id": 0, "type": "component", "score": .9}]
        with patch.dict(sys.modules, {"ultralytics": fake_module}), patch.object(pipeline, "result_rows", return_value=rows):
            self.assertEqual(pipeline_main(["capture", "--images", str(self.image_path), "--weights", str(weights),
                                           "--output", str(output)]), 0)
        raw = json.loads((output / "raw_candidates.json").read_text(encoding="utf-8"))
        self.assertFalse(raw["evaluation_available"])
        self.assertEqual(raw["generation"]["confidence"], .03)
        self.assertEqual(raw["generation"]["max_det_per_call"], 1000)
        self.assertEqual(len(next(iter(raw["predictions"].values()))), 2)
        self.assertEqual(raw["weights"]["sha256"], sha256_file(weights))
        self.assertTrue((output / "raw_capture_journal.jsonl").is_file())
        summary = json.loads((output / "pipeline_summary.json").read_text(encoding="utf-8"))
        self.assertIsNone(summary["metrics"])


if __name__ == "__main__":
    unittest.main()
