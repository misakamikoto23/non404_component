"""Review safety tests using synthetic files only; no models or remote APIs."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
import uuid

from PIL import Image

from component_lab.review_workflow import (
    RevisionConflict,
    export_reviewed_dataset,
    validate_review_update,
)


CATEGORIES = ["component", "resistor", "capacitor"]


def make_box(box_id="seed_1", **changes):
    box = {
        "id": box_id,
        "bbox": [4, 5, 14, 15],
        "category": "resistor",
        "reference": "R1",
        "source": "official",
        "review_status": "accepted",
        "category_confirmed": False,
    }
    box.update(changes)
    return box


def make_item(case="train_a", split="train", **changes):
    item = {
        "id": case,
        "case": case,
        "split": split,
        "width": 40,
        "height": 30,
        "revision": 0,
        "boxes": [make_box()],
        "review": {"whole_image_checked": False, "completed": False},
    }
    item.update(changes)
    return item


def make_payload(item):
    return deepcopy({key: item[key] for key in ("revision", "boxes", "review")})


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class ReviewUpdateTests(unittest.TestCase):
    def test_success_is_pure_and_increments_revision(self):
        item = make_item(revision=7)
        payload = make_payload(item)
        payload["review"] = {"whole_image_checked": True, "completed": True}
        original_item, original_payload = deepcopy(item), deepcopy(payload)

        updated = validate_review_update(item, payload, CATEGORIES)

        self.assertEqual(item, original_item)
        self.assertEqual(payload, original_payload)
        self.assertEqual(updated["revision"], 8)
        self.assertTrue(updated["review"]["completed"])
        updated["boxes"][0]["bbox"][0] = 9
        self.assertEqual(item, original_item, "Returned data must not alias stored history")

    def test_stale_revision_has_a_distinct_value_error(self):
        self.assertTrue(issubclass(RevisionConflict, ValueError))
        item = make_item(revision=2)
        payload = make_payload(item)
        payload["revision"] = 1
        with self.assertRaises(RevisionConflict):
            validate_review_update(item, payload, CATEGORIES)

    def test_invalid_geometry_is_rejected_without_mutating_input(self):
        bad_boxes = (
            [4, 5, 4, 15],
            [14, 5, 4, 15],
            [4, 15, 14, 5],
            [-1, 5, 14, 15],
            [4, 5, 41, 15],
            [4, 5, 14, 31],
            [4, 5, 14],
            [4, 5, float("nan"), 15],
            [4, 5, float("inf"), 15],
            [4, 5, "14", 15],
            [True, 5, 14, 15],
        )
        for bbox in bad_boxes:
            with self.subTest(bbox=bbox):
                item = make_item()
                original = deepcopy(item)
                payload = make_payload(item)
                payload["boxes"][0]["bbox"] = bbox
                with self.assertRaises(ValueError):
                    validate_review_update(item, payload, CATEGORIES)
                self.assertEqual(item, original)

    def test_unknown_category_and_review_status_are_rejected(self):
        for key, value in (("category", "unknown-part"), ("review_status", "approved")):
            with self.subTest(key=key):
                item = make_item()
                payload = make_payload(item)
                payload["boxes"][0][key] = value
                with self.assertRaises(ValueError):
                    validate_review_update(item, payload, CATEGORIES)

    def test_known_boxes_cannot_disappear_and_rejections_keep_history(self):
        item = make_item()
        payload = make_payload(item)
        payload["boxes"] = []
        with self.assertRaises(ValueError):
            validate_review_update(item, payload, CATEGORIES)

        payload = make_payload(item)
        payload["boxes"][0]["review_status"] = "rejected"
        payload["boxes"][0]["source"] = "human"
        updated = validate_review_update(item, payload, CATEGORIES)
        self.assertEqual(len(updated["boxes"]), 1)
        self.assertEqual(updated["boxes"][0]["id"], "seed_1")
        self.assertEqual(updated["boxes"][0]["bbox"], item["boxes"][0]["bbox"])
        self.assertEqual(updated["boxes"][0]["review_status"], "rejected")
        self.assertEqual(updated["boxes"][0]["source"], "official")

    def test_new_boxes_need_human_ids_and_server_owned_provenance(self):
        item = make_item()
        payload = make_payload(item)
        payload["boxes"].append(make_box("injected", source="official"))
        with self.assertRaises(ValueError):
            validate_review_update(item, payload, CATEGORIES)

        payload["boxes"][-1]["id"] = "human_added"
        updated = validate_review_update(item, payload, CATEGORIES)
        added = next(box for box in updated["boxes"] if box["id"] == "human_added")
        self.assertEqual(added["source"], "human")

    def test_duplicate_box_ids_are_rejected(self):
        item = make_item()
        payload = make_payload(item)
        payload["boxes"].append(deepcopy(payload["boxes"][0]))
        with self.assertRaises(ValueError):
            validate_review_update(item, payload, CATEGORIES)

    def assert_not_completed(self, item, payload):
        """A rejected completion and a cleared completion are equally safe."""
        try:
            updated = validate_review_update(item, payload, CATEGORIES)
        except ValueError:
            return
        self.assertFalse(updated["review"]["completed"])

    def test_completion_requires_whole_image_check_and_no_pending_boxes(self):
        item = make_item()
        payload = make_payload(item)
        payload["review"] = {"whole_image_checked": False, "completed": True}
        self.assert_not_completed(item, payload)

        payload["review"] = {"whole_image_checked": True, "completed": True}
        payload["boxes"][0]["review_status"] = "pending"
        self.assert_not_completed(item, payload)

    def test_edits_require_an_explicit_new_whole_image_confirmation(self):
        item = make_item(review={"whole_image_checked": True, "completed": True})
        for key, value in (("bbox", [5, 5, 15, 15]), ("category", "capacitor"), ("reference", "R2")):
            with self.subTest(key=key):
                payload = make_payload(item)
                payload["boxes"][0][key] = value
                payload["review"] = {"completed": True}
                self.assert_not_completed(item, payload)

                payload["review"] = {"whole_image_checked": True, "completed": True}
                updated = validate_review_update(item, payload, CATEGORIES)
                self.assertTrue(updated["review"]["completed"])

    def test_classification_confirmation_requires_an_accepted_specific_category(self):
        for category, status in (("component", "accepted"), ("resistor", "pending"), ("resistor", "rejected")):
            with self.subTest(category=category, status=status):
                item = make_item()
                payload = make_payload(item)
                payload["boxes"][0].update(category=category, review_status=status, category_confirmed=True)
                with self.assertRaises(ValueError):
                    validate_review_update(item, payload, CATEGORIES)

        item = make_item()
        payload = make_payload(item)
        payload["boxes"][0]["category_confirmed"] = True
        updated = validate_review_update(item, payload, CATEGORIES)
        self.assertTrue(updated["boxes"][0]["category_confirmed"])


class ReviewedExportTests(unittest.TestCase):
    def setUp(self):
        # Inherit the workspace ACL: Windows Python's mode-0700 temporary
        # directories can exclude the restricted test process that made them.
        test_directory = Path(__file__).resolve().parent
        self.root = test_directory / f"review-workflow-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.assertEqual(self.root.resolve().parent, test_directory)
        self.addCleanup(shutil.rmtree, self.root)
        self.data_root = self.root / "source_dataset"
        self.batch_dir = self.root / "review_batch"
        self.output = self.root / "export"
        samples = []
        for split, case in (("train", "train_a"), ("val", "val_a"), ("test", "test_a")):
            image_path = self.data_root / "images" / split / f"{case}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 30), color=(23, 90, 140)).save(image_path)
            label_path = self.data_root / "labels" / split / f"{case}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_bytes(b"0  .225000  .333333  .250000  .333333\r\n")
            samples.append({
                "case": case, "split": split, "width": 40, "height": 30,
                "image": f"images/{split}/{case}.png",
                "labels": [{"category": "resistor", "bbox": [4, 5, 14, 15], "component_id": "seed_1"}],
            })
        write_json(self.data_root / "manifest.json", {
            "classes": ["component"],
            "original_category_names": ["resistor", "capacitor"],
            "samples": samples,
        })
        self.initial_item = make_item(boxes=[make_box(review_status="pending")])
        write_json(self.batch_dir / "batch.json", {"items": [self.initial_item]})
        (self.batch_dir / "items").mkdir(exist_ok=True)

    def save_item(self, item):
        write_json(self.batch_dir / "items" / f"{item['id']}.json", item)

    def completed_item(self, **box_changes):
        accepted = make_box("human_added", bbox=[10, 6, 30, 24], source="human", category_confirmed=True)
        accepted.update(box_changes)
        return make_item(
            revision=1,
            boxes=[make_box(review_status="rejected"), accepted],
            review={"whole_image_checked": True, "completed": True},
        )

    def assert_source_copy(self, split, case):
        for directory, suffix in (("images", ".png"), ("labels", ".txt")):
            source = self.data_root / directory / split / f"{case}{suffix}"
            target = self.output / "yolo" / directory / split / f"{case}{suffix}"
            self.assertEqual(target.read_bytes(), source.read_bytes())

    def classifier_images(self):
        directory = self.output / "resnet_reviewed"
        return [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg")]

    def test_saved_train_review_overrides_seed_and_holdouts_are_byte_identical(self):
        self.save_item(self.completed_item())
        # Unlisted files must not turn the batch loader into a holdout scan.
        self.save_item(make_item("val_a", "val", boxes=[make_box(category_confirmed=True)],
                                 review={"whole_image_checked": True, "completed": True}))
        before = {path: path.read_bytes() for directory in (self.data_root, self.batch_dir)
                  for path in directory.rglob("*") if path.is_file()}

        summary = export_reviewed_dataset(self.batch_dir, self.data_root, self.output)

        label = self.output / "yolo" / "labels" / "train" / "train_a.txt"
        rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 1, "Rejected original boxes must stay out of training labels")
        self.assertEqual(int(rows[0][0]), 0, "Detection export must remain single class")
        for actual, expected in zip(map(float, rows[0][1:]), (.5, .5, .5, .6)):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assert_source_copy("val", "val_a")
        self.assert_source_copy("test", "test_a")
        crops = self.classifier_images()
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].parent.name, "resistor")
        self.assertEqual(crops[0].parent.parent.name, "train")
        with Image.open(crops[0]) as crop:
            self.assertEqual(crop.size, (20, 18))
        self.assertFalse((self.output / "resnet_reviewed" / "val").exists())
        self.assertFalse((self.output / "resnet_reviewed" / "test").exists())
        self.assertIn("component", (self.output / "yolo" / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(summary, json.loads((self.output / "summary.json").read_text(encoding="utf-8")))
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_all_incomplete_batches_do_not_create_an_export(self):
        variants = (
            {"whole_image_checked": False, "completed": False},
            {"whole_image_checked": True, "completed": False},
            {"whole_image_checked": False, "completed": True},
        )
        for number, review in enumerate(variants):
            with self.subTest(review=review):
                self.output = self.root / f"pending_export_{number}"
                item = self.completed_item()
                item["review"] = review
                self.save_item(item)
                with self.assertRaises(ValueError):
                    export_reviewed_dataset(self.batch_dir, self.data_root, self.output)
                self.assertFalse(self.output.exists())

    def test_mixed_batch_preserves_pending_train_labels_and_excludes_pending_crops(self):
        self.save_item(self.completed_item())
        pending = make_item("train_pending", boxes=[
            make_box(bbox=[20, 10, 35, 25], review_status="pending")
        ])
        write_json(self.batch_dir / "batch.json", {"items": [self.initial_item, pending]})
        manifest_path = self.data_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sample = deepcopy(manifest["samples"][0])
        sample.update(case="train_pending", image="images/train/train_pending.png")
        manifest["samples"].append(sample)
        write_json(manifest_path, manifest)
        for directory, suffix in (("images", ".png"), ("labels", ".txt")):
            base = self.data_root / directory / "train"
            shutil.copyfile(base / f"train_a{suffix}", base / f"train_pending{suffix}")

        summary = export_reviewed_dataset(self.batch_dir, self.data_root, self.output)

        self.assert_source_copy("train", "train_pending")
        self.assertEqual(len(self.classifier_images()), 1)
        self.assertEqual(summary["reviewed_train_images"], 1)
        self.assertEqual(summary["excluded_pending_images"], 1)

    def test_pending_boxes_cannot_be_exported_by_forging_completed_flags(self):
        item = self.completed_item(category_confirmed=False, review_status="pending")
        self.save_item(item)
        with self.assertRaises(ValueError):
            export_reviewed_dataset(self.batch_dir, self.data_root, self.output)
        self.assertFalse(self.output.exists())

    def add_unlabelled_external_image(self, complete=False):
        manifest_path=self.data_root/'manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        pending=self.data_root/'images/pending/hf_new.png'
        pending.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(40,30),'white').save(pending)
        manifest['samples'].append({'case':'hf_new','split':'train','width':40,'height':30,'labels':[],
            'annotation_status':'pending_review','pending_image_file':'images/pending/hf_new.png',
            'source_project':'example/project'})
        write_json(manifest_path,manifest)
        item=make_item('hf_new',boxes=[make_box(review_status='accepted' if complete else 'pending')],
            review={'whole_image_checked':complete,'completed':complete})
        write_json(self.batch_dir/'batch.json',{'items':[self.initial_item,item]})
        self.save_item(self.completed_item())

    def test_unlabelled_external_pending_image_never_becomes_negative_training_data(self):
        self.add_unlabelled_external_image()
        self.assertFalse((self.data_root/'labels/train/hf_new.txt').exists())
        summary=export_reviewed_dataset(self.batch_dir,self.data_root,self.output)
        self.assertEqual(summary['excluded_unlabelled_new_images'],['hf_new'])
        self.assertFalse((self.output/'yolo/images/train/hf_new.png').exists())
        self.assertFalse((self.output/'yolo/labels/train/hf_new.txt').exists())
        manifest=json.loads((self.output/'yolo/manifest.json').read_text(encoding='utf-8'))
        self.assertNotIn('hf_new',[s['case'] for s in manifest['samples']])
        self.assertNotIn('hf_new',manifest['splits']['train'])
        self.assert_source_copy('val','val_a')
        self.assert_source_copy('test','test_a')

    def test_external_image_enters_training_only_with_completed_review_and_real_labels(self):
        from component_lab.review_workflow import image_path
        self.add_unlabelled_external_image(complete=True)
        export_reviewed_dataset(self.batch_dir,self.data_root,self.output)
        root=self.output/'yolo'
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        sample=next(s for s in manifest['samples'] if s['case']=='hf_new')
        self.assertEqual(sample['annotation_status'],'human_reviewed')
        self.assertNotIn('pending_image_file',sample)
        self.assertEqual(sample['source_project'],'example/project')
        self.assertEqual(image_path(root,sample),root/'images/train/hf_new.png')
        self.assertEqual(len((root/'labels/train/hf_new.txt').read_text().strip().splitlines()),1)
        self.assertIn('hf_new',manifest['splits']['train'])

    def test_explicit_holdout_review_in_batch_is_rejected_before_any_export(self):
        self.save_item(self.completed_item())
        holdout = make_item("val_a", "val", review={"whole_image_checked": True, "completed": True})
        write_json(self.batch_dir / "batch.json", {"items": [self.initial_item, holdout]})
        with self.assertRaises(ValueError):
            export_reviewed_dataset(self.batch_dir, self.data_root, self.output)
        self.assertFalse(self.output.exists())

    def test_completed_all_rejected_image_has_an_empty_detection_label(self):
        item = make_item(boxes=[make_box(review_status="rejected")],
                         review={"whole_image_checked": True, "completed": True})
        self.save_item(item)
        export_reviewed_dataset(self.batch_dir, self.data_root, self.output)
        label = self.output / "yolo" / "labels" / "train" / "train_a.txt"
        self.assertEqual(label.read_bytes(), b"")
        self.assertEqual(self.classifier_images(), [])

    def test_unconfirmed_categories_do_not_create_classification_training_data(self):
        self.save_item(self.completed_item(category_confirmed=False))
        export_reviewed_dataset(self.batch_dir, self.data_root, self.output)
        self.assertEqual(self.classifier_images(), [])

    def test_output_must_be_empty_and_outside_both_input_trees(self):
        occupied = self.root / "occupied"
        occupied.mkdir()
        marker = occupied / "do-not-touch.txt"
        marker.write_bytes(b"existing result")
        targets = (occupied, self.data_root / "new_export", self.batch_dir / "new_export")
        for target in targets:
            with self.subTest(target=target), self.assertRaises(ValueError):
                export_reviewed_dataset(self.batch_dir, self.data_root, target)
        self.assertEqual(marker.read_bytes(), b"existing result")
        self.assertFalse((self.data_root / "new_export").exists())
        self.assertFalse((self.batch_dir / "new_export").exists())


if __name__ == "__main__":
    unittest.main()
