"""Checks on the extracted official GT and the committed group split (skipped when the data is absent)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
from component_lab.eval.cases import load_split
from component_lab.eval.coords import out_of_bounds

GT = LAB / "data/official_contest6"
SPLIT = LAB / "configs/contest6_group_split_v1.json"
GROUPS = LAB / "outputs/eval_split_audit_20261017/groups.json"
FORBIDDEN = LAB / "outputs/eval_split_audit_20261017/forbidden_for_clean_retrain.json"


@unittest.skipUnless((GT / "_manifest.json").exists(), "official GT not extracted")
class OfficialCoordinates(unittest.TestCase):
    def test_points_inside_image_and_boxes_at_most_slightly_outside(self):
        manifest = json.loads((GT / "_manifest.json").read_text(encoding="utf-8"))["cases"]
        boxes_outside = 0
        for row in manifest:
            case = json.loads((GT / row["split"] / f"{row['stem']}.json").read_text(encoding="utf-8"))
            exact = out_of_bounds(case, row["width"], row["height"])
            self.assertFalse([b for b in exact if b[0] != "bbox"], row["stem"])  # pins and edges always inside
            boxes_outside += len(exact)
            self.assertEqual(out_of_bounds(case, row["width"], row["height"], slack=40), [], row["stem"])
        # 58 boxes (mostly gnd symbols at the page edge) overhang by 1-35 px; flipped frames would not fit at all.
        self.assertEqual(boxes_outside, 58)


@unittest.skipUnless(SPLIT.exists() and GROUPS.exists(), "group split not built")
class GroupSplit(unittest.TestCase):
    def setUp(self):
        self.split = json.loads(SPLIT.read_text(encoding="utf-8"))
        self.groups = json.loads(GROUPS.read_text(encoding="utf-8"))

    def test_partitions_are_disjoint_by_case_and_group(self):
        parts = self.split["partitions"]
        self.assertEqual({p: len(v) for p, v in parts.items()}, {"test": 40, "dev": 20, "train": 140})
        groups = {p: {row["group"] for row in rows} for p, rows in parts.items()}
        self.assertFalse(groups["test"] & groups["dev"] or groups["test"] & groups["train"] or groups["dev"] & groups["train"])
        cases = [row["case"] for rows in parts.values() for row in rows]
        self.assertEqual(len(cases), len(set(cases)))

    def test_gt10_groups_are_train_and_held_out_images_are_forbidden(self):
        held = {row["group"] for p in ("test", "dev") for row in self.split["partitions"][p]}
        for gid in held:
            self.assertFalse(self.groups[gid]["gt10"], gid)
        if FORBIDDEN.exists():
            forbidden = json.loads(FORBIDDEN.read_text(encoding="utf-8"))
            self.assertEqual(set(forbidden["held_out_groups"]), held)
            listed = {i["id"] for i in forbidden["images"]}
            expected = {m for gid in held for m in self.groups[gid]["members"] if not m.startswith("official:")}
            self.assertEqual(listed, expected)

    def test_load_split_partition(self):
        self.assertEqual(len(load_split(SPLIT, "test")), 40)
        with self.assertRaises(ValueError):
            load_split(SPLIT)


class LoadSplitFormats(unittest.TestCase):
    def test_flat_and_partitioned(self):
        with tempfile.TemporaryDirectory() as tmp:
            flat, grouped = Path(tmp, "flat.json"), Path(tmp, "grouped.json")
            flat.write_text(json.dumps({"cases": [{"case": "0007"}, "0010"]}), encoding="utf-8")
            grouped.write_text(json.dumps({"partitions": {"test": [{"case": "0003"}]}}), encoding="utf-8")
            self.assertEqual(load_split(flat), ["7", "10"])
            self.assertEqual(load_split(grouped, "test"), ["3"])
            with self.assertRaises(ValueError):
                load_split(flat, "test")


if __name__ == "__main__":
    unittest.main()
