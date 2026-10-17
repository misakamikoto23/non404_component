"""Pure category-audit tests; no inference, network, or training dependencies."""
from collections import Counter
from copy import deepcopy
import unittest

from component_lab.category_audit import (
    build_category_report,
    classification_balance_plan,
    select_review_batch,
)


def sample(case, categories, split="train", width=1000, height=1000):
    return {
        "case": case, "image": f"{case}.png", "split": split,
        "width": width, "height": height,
        "labels": [
            {"category": name, "bbox": [10, 10, 30, 30], "component_id": f"C_{i}"}
            for i, name in enumerate(categories)
        ],
    }


class CategoryAuditTests(unittest.TestCase):
    def test_box_counts_and_image_counts_do_not_double_count_categories(self):
        manifest = {"original_category_names": ["R", "C", "absent"], "samples": [
            sample("a", ["R", "R", "C"]), sample("b", ["R"]),
            sample("v", ["C", "C"], "val"), sample("t", ["R"], "test"),
        ]}
        result = build_category_report(manifest)
        categories = {row["category"]: row for row in result["categories"]}
        self.assertEqual(result["totals"], {
            "images": 4, "boxes": 7,
            "by_split": {"train": {"images": 2, "boxes": 4},
                         "val": {"images": 1, "boxes": 2}, "test": {"images": 1, "boxes": 1}},
        })
        self.assertEqual(categories["R"]["train_boxes"], 3)
        self.assertEqual(categories["R"]["train_images"], 2)
        self.assertEqual(categories["C"]["total_boxes"], 3)
        self.assertEqual(categories["C"]["total_images"], 2)
        self.assertEqual(categories["absent"]["total_images"], 0)

    def test_rarity_uses_train_source_images_at_boundaries(self):
        manifest = {"samples": [
            sample(str(i), (["four"] if i < 4 else []) + (["five"] if i < 5 else [])
                   + (["fourteen"] if i < 14 else []) + ["fifteen"])
            for i in range(15)
        ] + [sample("val", ["four"] * 100, "val")]}
        rows = {row["category"]: row for row in build_category_report(manifest)["categories"]}
        self.assertEqual(rows["four"]["rarity_tier"], "very_rare")
        self.assertEqual(rows["five"]["rarity_tier"], "rare")
        self.assertEqual(rows["fourteen"]["rarity_tier"], "rare")
        self.assertEqual(rows["fifteen"]["rarity_tier"], "common")

    def test_review_is_deterministic_unique_copied_and_train_only(self):
        manifest = {"samples": [sample(str(i), ["R", f"rare_{i % 5}"])
                                for i in range(40)] + [sample("v", ["val-only"], "val")]}
        before = deepcopy(manifest)
        batch = select_review_batch(manifest)
        self.assertEqual(batch, select_review_batch(manifest))
        self.assertEqual(len(batch), 20)
        self.assertEqual(len({row["case"] for row in batch}), 20)
        self.assertEqual({row["split"] for row in batch}, {"train"})
        self.assertEqual(Counter(row["selection_group"] for row in batch),
                         {"random": 8, "rare_coverage": 6, "hard_geometry": 6})
        self.assertEqual([row["priority"] for row in batch], list(range(1, 21)))
        self.assertTrue(all(row["selection_reasons"] for row in batch))
        self.assertEqual(manifest, before)
        batch[0]["labels"][0]["category"] = "edited"
        self.assertEqual(manifest, before)

    def test_held_out_changes_cannot_change_review_or_training_weights(self):
        training = [sample(str(i), ["R"] * (i + 1) + (["rare"] if i < 3 else []))
                    for i in range(25)]
        first = {"samples": training + [sample("v", ["rare"], "val")]}
        second = {"original_category_names": ["held-only"], "samples": training + [
            sample("v", ["rare"] * 1000, "val", 999999, 999999),
            sample("t", ["held-only"] * 2000, "test", 999999, 999999),
        ]}
        self.assertEqual(select_review_batch(first), select_review_batch(second))
        a = classification_balance_plan(build_category_report(first))["class_weights"]
        b = classification_balance_plan(build_category_report(second))["class_weights"]
        self.assertEqual({k: b[k] for k in a}, a)
        self.assertIsNone(b["held-only"])

    def test_selection_scales_caps_deduplicates_and_fills(self):
        rows = [sample(str(i), ["common"]) for i in range(25)]
        rows.append(deepcopy(rows[0]))
        manifest = {"samples": rows}
        self.assertEqual(select_review_batch(manifest, 0), [])
        for count in (1, 2, 5, 11, 20, 24, 25, 99):
            batch = select_review_batch(manifest, count)
            self.assertEqual(len(batch), min(count, 25))
            self.assertEqual(len({row["case"] for row in batch}), len(batch))
        # No rare categories: the default rare quota becomes a random fill.
        groups = Counter(row["selection_group"] for row in select_review_batch(manifest))
        self.assertEqual(groups, {"random": 14, "hard_geometry": 6})
        with self.assertRaises(ValueError):
            select_review_batch(manifest, -1)
        with self.assertRaises(ValueError):
            select_review_batch(manifest, 1.5)

    def test_rare_coverage_prefers_multiple_uncovered_categories(self):
        # count=2 allocates one random and one rare example. Choose a seed whose
        # first random choice leaves the multi-category sample available.
        rows = [sample("multi", ["A", "B", "C"])] + [sample(f"single{i}", ["A"])
                                                            for i in range(8)]
        for seed in range(10):
            batch = select_review_batch({"samples": rows}, 2, seed)
            if batch[0]["case"] != "multi":
                self.assertEqual(batch[1]["case"], "multi")
                self.assertEqual(batch[1]["selection_group"], "rare_coverage")
                break
        else:
            self.fail("expected at least one seed leaving the multi-category sample")

    def test_weights_use_train_instances_cap_and_reject_absent_training_class(self):
        manifest = {"samples": [sample("a", ["R"] * 100 + ["C"] * 25 + ["rare"]),
                                sample("v", ["held-only"] * 10000, "val")]}
        before = deepcopy(manifest)
        report = build_category_report(manifest)
        report_before = deepcopy(report)
        plan = classification_balance_plan(report)
        self.assertEqual(plan["class_weights"], {"C": 2.0, "R": 1.0, "held-only": None, "rare": 5.0})
        self.assertEqual(plan["untrainable_categories"], ["held-only"])
        missing = next(row for row in plan["classes"] if row["category"] == "held-only")
        self.assertFalse(missing["trainable"])
        self.assertTrue(missing["needs_source_diversity"])
        self.assertFalse(plan["is_ready_sampler"])
        self.assertEqual(report, report_before)
        self.assertEqual(manifest, before)

    def test_empty_manifest_is_supported(self):
        report = build_category_report({})
        self.assertEqual(report["categories"], [])
        self.assertEqual(report["totals"]["images"], 0)
        self.assertEqual(select_review_batch({}), [])
        self.assertEqual(classification_balance_plan(report)["class_weights"], {})


if __name__ == "__main__":
    unittest.main()
