"""Safety and geometry tests for gold crop export, using only synthetic fixtures."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_official_gold_crops.py"
SPEC = importlib.util.spec_from_file_location("official_gold_preparer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OfficialGoldCropTests(unittest.TestCase):
    def temp_directory(self):
        # Keep fixture creation/cleanup inside the explicitly writable workspace.
        workspace = SCRIPT.parents[1].resolve()
        base = (workspace / "outputs/_test_tmp_official_gold").resolve()
        if not base.is_relative_to(workspace):
            raise ValueError("Unsafe fixture root")
        base.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=base)

    def test_bottom_left_box_and_padding(self):
        self.assertEqual(MODULE.crop_box([10, 20, 30, 40], 100, 80, 0, 0),
                         ([10, 40, 30, 60], [10, 40, 30, 60], False))
        self.assertEqual(MODULE.crop_box([0, 0, 10, 10], 100, 80, 2, .1),
                         ([0, 70, 10, 80], [0, 67, 13, 80], True))

    def test_invalid_geometry(self):
        for box in ([1, 2, 1, 3], [1, 3, 2, 1], [1, 2, float("nan"), 4], [-30, 0, -20, 10]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                MODULE.crop_box(box, 100, 80)

    def test_safe_raw_types(self):
        self.assertEqual(MODULE.validate_type_name("m3螺丝"), "m3螺丝")
        for name in ("../r", "a/b", "CON", "r ", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                MODULE.validate_type_name(name)

    def test_blank_crops_flagged_without_relabeling(self):
        quality = MODULE.crop_quality(Image.new("RGB", (20, 20), "white"))
        self.assertIn("near_blank_or_low_contrast", quality["flags"])
        self.assertEqual(quality["dark_fraction_below_220"], 0)

    def make_fixture(self, root):
        groups, partitions, image_bytes = {}, {}, {}
        for index, partition in enumerate(("train", "dev", "test"), start=1):
            image = Image.new("RGB", (30, 40), (index * 30, 0, 0))
            buff = io.BytesIO()
            image.save(buff, format="PNG")
            stem = f"{index:04d}-fixture"
            row = {"case": f"{index:04d}", "stem": stem, "group": f"G{index}",
                   "tag": "fixture", "pixel_sha256": MODULE.pixel_sha256(image)}
            partitions[partition] = [row]
            groups[row["group"]] = {"members": [f"official:train200:{stem}"],
                                     "pixel_sha256": [row["pixel_sha256"]]}
            image_bytes[stem] = buff.getvalue()
        forbidden = {"held_out_groups": ["G2", "G3"], "images": []}
        split = {"name": "fixture", "partitions": partitions, "groups_file": "groups.json",
                 "forbidden_manifest": "forbidden.json"}
        for name, value in (("split.json", split), ("groups.json", groups), ("forbidden.json", forbidden)):
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        archive_path = root / "fixture.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for partition, rows in partitions.items():
                row = rows[0]
                prefix = f"official/200_train_cases/{row['case']}/{row['stem']}"
                archive.writestr(prefix + ".png", image_bytes[row["stem"]])
                label = {"components": {"R1": {"type": "r", "Name": "R1", "value": "10 KΩ",
                                                        "bbox": [3, 4, 13, 14]}}}
                # The test JSON is intentionally invalid: successful export proves it was not parsed.
                archive.writestr(prefix + "_target_named.json", "NOT JSON" if partition == "test" else json.dumps(label))
        return split, groups, forbidden, archive_path

    def test_full_export_never_reads_test_members_and_preserves_text(self):
        with self.temp_directory() as tmp:
            root = Path(tmp)
            split, _, _, archive_path = self.make_fixture(root)
            original_read = zipfile.ZipFile.read
            opened = []

            def tracked_read(archive, name, *args, **kwargs):
                opened.append(name)
                self.assertNotIn("/0003/", name)
                return original_read(archive, name, *args, **kwargs)

            with mock.patch.object(zipfile.ZipFile, "read", tracked_read):
                summary = MODULE.prepare(root, root / "split.json", archive_path, root / "dataset", root / "report", 0, 0)
            self.assertTrue(opened)
            self.assertEqual(summary["crops"], {"train": 1, "val": 1})
            self.assertEqual(summary["test_labels_read"], 0)
            manifest = json.loads((root / "dataset/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["crops"][0]["value"], "10 KΩ")
            self.assertEqual(manifest["crops"][0]["bbox_top_left"], [3, 26, 13, 36])
            self.assertFalse((root / "dataset/test").exists())

    def test_reject_group_leak_and_stale_ban_manifest(self):
        with self.temp_directory() as tmp:
            split, groups, forbidden, _ = self.make_fixture(Path(tmp))
            bad = json.loads(json.dumps(split))
            bad["partitions"]["dev"][0]["group"] = "G1"
            with self.assertRaises(ValueError):
                MODULE.validate_split(bad, groups, forbidden)
            with self.assertRaises(ValueError):
                MODULE.validate_split(split, groups, {"held_out_groups": ["G2"]})

    def test_hash_mismatch_does_not_create_output(self):
        with self.temp_directory() as tmp:
            root = Path(tmp)
            split, groups, _, archive_path = self.make_fixture(root)
            split["partitions"]["train"][0]["pixel_sha256"] = "wrong"
            groups["G1"]["pixel_sha256"] = ["wrong"]
            (root / "split.json").write_text(json.dumps(split), encoding="utf-8")
            (root / "groups.json").write_text(json.dumps(groups), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                MODULE.prepare(root, root / "split.json", archive_path, root / "dataset", root / "report")
            self.assertFalse((root / "dataset").exists())

    def test_duplicate_archive_source_is_rejected(self):
        with self.temp_directory() as tmp:
            root = Path(tmp)
            split, _, _, archive_path = self.make_fixture(root)
            with zipfile.ZipFile(archive_path, "a") as archive:
                archive.writestr("other/200_train_cases/0001/0001-fixture.png", b"duplicate")
            with zipfile.ZipFile(archive_path) as archive, self.assertRaisesRegex(ValueError, "Ambiguous"):
                MODULE.archive_members(archive, split["partitions"]["train"])


if __name__ == "__main__":
    unittest.main()
