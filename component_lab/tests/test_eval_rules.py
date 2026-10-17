"""Discriminating scorer tests: each case fails under a plausible wrong reading of the spec."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from component_lab.eval.coords import flip_bbox, flip_case, flip_point, out_of_bounds
from component_lab.eval.dedup import UnionFind, assign_partitions, multiset_jaccard, label_multiset
from component_lab.eval.metric import ScoreConfig, canonical_type, score_case
from component_lab.eval.report import bootstrap_ci

STRICT = ScoreConfig()
MAPPED = ScoreConfig(type_policy="mapped")


def one(type_, name=None, value=None, bbox=(100.0, 100.0, 140.0, 160.0), key="X1"):
    return {"components": {key: {"type": type_, "Name": name, "value": value, "bbox": list(bbox)}},
            "pins": {}, "nets": {}}


def component_ok(pred, gt, config=STRICT):
    counts = score_case(pred, gt, config)["counts"]["component_f1"]
    return counts["tp"] == 1


def edit(case, **fields):
    case = deepcopy(case)
    case["components"]["X1"].update(fields)
    return case


class ComponentText(unittest.TestCase):
    # (description, gt, pred, expected under the spec, expected if Name were checked for every type)
    CASES = [
        ("resistor: wrong Name, right value", one("r", "R1", "10k"), edit(one("r", "R1", "10k"), Name="R9"), True, False),
        ("ic(box): wrong value, right Name", one("box", "LM358", None), edit(one("box", "LM358", None), value="5V"), True, True),
        ("led: wrong Name and value", one("led", "D1", "red"), edit(one("led", "D1", "red"), Name="D2", value="blue"), True, False),
        ("gnd: wrong Name and value", one("gnd", "GND", None), edit(one("gnd", "GND", None), Name="AGND", value="0"), True, False),
        ("capacitor: 10k vs 10K", one("c", "C1", "10K"), edit(one("c", "C1", "10K"), value="10k"), False, False),
        ("resistor: '10 k' vs '10k'", one("r", "R1", "10k"), edit(one("r", "R1", "10k"), value="10 k"), False, False),
        ("resistor: GT null, pred null", one("r", "R1", None), one("r", "R1", None), True, True),
        ("resistor: GT null, pred 10k", one("r", "R1", None), edit(one("r", "R1", None), value="10k"), False, False),
        ("ic(box): wrong Name", one("box", "LM358"), edit(one("box", "LM358"), Name="LM358A"), False, False),
    ]

    def test_text_rules(self):
        for description, gt, pred, expected, _ in self.CASES:
            self.assertEqual(component_ok(pred, gt), expected, description)

    def test_differs_from_name_for_every_type_matcher(self):
        """non404 accuracy_proxy compared Name for all official types; these rows separate the two readings."""
        differing = [d for d, _, _, ours, name_everywhere in self.CASES if ours != name_everywhere]
        self.assertEqual(differing, ["resistor: wrong Name, right value", "led: wrong Name and value",
                                     "gnd: wrong Name and value"])

    def test_text_rule_uses_canonical_type_under_both_policies(self):
        for config in (STRICT, MAPPED):
            self.assertFalse(component_ok(edit(one("ic", "U"), Name="V"), one("ic", "U"), config))
            self.assertFalse(component_ok(edit(one("resistor", value="1k"), value="2k"), one("resistor", value="1k"), config))
            self.assertFalse(component_ok(edit(one("box", "U"), Name="V"), one("box", "U"), config))


class TypePolicy(unittest.TestCase):
    def test_box_vs_ic_and_polarity(self):
        for gt_type, pred_type in (("box", "ic"), ("mosfet", "mosfet_npn"), ("bjt", "bjt_pnp"), ("r", "resistor")):
            gt = one(gt_type, "U", "1k")
            pred = edit(gt, type=pred_type)
            self.assertFalse(component_ok(pred, gt, STRICT), (gt_type, pred_type))
            self.assertTrue(component_ok(pred, gt, MAPPED), (gt_type, pred_type))

    def test_unmapped_types_are_reported_not_mapped(self):
        gt = one("m3螺丝")
        result = score_case(edit(gt, type="m3"), gt, MAPPED)
        self.assertEqual(result["counts"]["component_f1"]["tp"], 0)
        self.assertEqual(result["types"]["gt_unmapped"], ["m3螺丝"])
        self.assertEqual(result["types"]["pred_unmapped"], ["m3"])
        self.assertEqual(canonical_type("m3螺丝"), "m3螺丝")


class BoxTolerance(unittest.TestCase):
    def test_one_coordinate_at_20_and_20_01(self):
        gt = one("d")
        self.assertTrue(component_ok(edit(gt, bbox=[120.0, 100.0, 140.0, 160.0]), gt))
        self.assertFalse(component_ok(edit(gt, bbox=[120.01, 100.0, 140.0, 160.0]), gt))

    def test_center_passes_where_max_corner_fails(self):
        gt = one("d")
        grown = edit(gt, bbox=[75.0, 75.0, 165.0, 185.0])  # 25 px larger on every side, same center
        self.assertFalse(component_ok(grown, gt, ScoreConfig(bbox_metric="max_corner")))
        self.assertTrue(component_ok(grown, gt, ScoreConfig(bbox_metric="center")))


class Pins(unittest.TestCase):
    def case(self, name="VCC", x=10.0):
        return {"components": {"U1": {"type": "box", "Name": "U", "value": None, "bbox": [0, 0, 50, 50]}},
                "pins": {"U1": {"pin_1": {"pinname": name, "point": {"x": x, "y": 0.0}}}}, "nets": {}}

    def pin_ok(self, pred):
        return score_case(pred, self.case())["counts"]["pin_f1"]["tp"] == 1

    def test_pinname_case_and_distance_edge(self):
        self.assertFalse(self.pin_ok(self.case(name="vcc")))
        self.assertTrue(self.pin_ok(self.case(x=15.0)))
        self.assertFalse(self.pin_ok(self.case(x=15.01)))


class AnonymousModes(unittest.TestCase):
    def gt(self):
        return {"components": {"R1": {"type": "r", "Name": "R1", "value": "1k", "bbox": [0, 0, 10, 30]},
                               "∅3": {"type": "battary", "Name": "BAT", "value": None, "bbox": [50, 0, 70, 30]}},
                "pins": {"R1": {"pin_1": {"pinname": "1", "point": {"x": 5, "y": 30}}},
                         "∅3": {"pin_1": {"pinname": "+", "point": {"x": 60, "y": 30}}}},
                "nets": {"N": {"hyperGraph": "(R1.1,∅3.1)", "edges": {"edge_1": [{"x": 5, "y": 30}, {"x": 60, "y": 30}]}}}}

    def test_strict_vs_observable_diagnostic(self):
        pred = self.gt()
        del pred["components"]["∅3"], pred["pins"]["∅3"]
        pred["nets"]["N"]["hyperGraph"] = "(R1.1)"  # the only scorable member is non-∅
        strict = score_case(pred, self.gt(), ScoreConfig(anon_policy="strict"))
        observable = score_case(pred, self.gt(), ScoreConfig(anon_policy="observable_diagnostic"))
        self.assertEqual(strict["counts"]["component_f1"], {"tp": 1, "fp": 0, "fn": 1})
        self.assertEqual(strict["counts"]["net_hypergraph_f1"], {"tp": 1, "fp": 0, "fn": 1})
        self.assertAlmostEqual(observable["score"], 100.0)


SAMPLE = {
    "components": {"U1": {"type": "box", "Name": "U", "value": None, "bbox": [100.0, 600.0, 300.0, 900.0]},
                   "R1": {"type": "r", "Name": "R1", "value": "1k", "bbox": [400.0, 100.0, 420.0, 160.0]}},
    "pins": {"U1": {"pin_1": {"pinname": "A", "point": {"x": 300.0, "y": 700.0}}},
             "R1": {"pin_1": {"pinname": "1", "point": {"x": 410.0, "y": 160.0}}}},
    "nets": {"N": {"hyperGraph": "(U1.1,R1.1)",
                   "edges": {"edge_1": [{"x": 300.0, "y": 700.0}, {"x": 410.0, "y": 700.0}],
                             "edge_2": [{"x": 410.0, "y": 700.0}, {"x": 410.0, "y": 160.0}]}}},
}


class YFlip(unittest.TestCase):
    def test_round_trip_is_identity(self):
        self.assertEqual(flip_case(flip_case(SAMPLE, 1000), 1000), SAMPLE)

    def test_known_conversion(self):
        self.assertEqual(flip_bbox([10, 20, 30, 40], 100), [10, 60, 30, 80])  # top-left rows 20..40
        self.assertEqual(flip_point({"x": 5, "y": 10}, 100), {"x": 5, "y": 90})

    def test_unflipped_prediction_loses_every_geometric_metric(self):
        result = score_case(flip_case(SAMPLE, 1000), SAMPLE)
        self.assertEqual((result["component_f1"], result["pin_f1"], result["net_line_f1"]), (0, 0, 0))
        # Hypergraph membership has no coordinates, so 35 points survive a frame error.
        self.assertAlmostEqual(result["score"], 35.0)

    def test_bounds_check(self):
        self.assertEqual(out_of_bounds(SAMPLE, 500, 1000), [])
        self.assertEqual(out_of_bounds(SAMPLE, 500, 800), [("bbox", "U1")])


class Bootstrap(unittest.TestCase):
    def test_ci(self):
        values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        low, high = bootstrap_ci(values, resamples=2000, seed=1)
        self.assertLess(low, 45)
        self.assertGreater(high, 45)
        self.assertEqual(bootstrap_ci(values, resamples=2000, seed=1), (low, high))
        self.assertEqual(bootstrap_ci([7.0, 7.0], resamples=100), (7.0, 7.0))


class Dedup(unittest.TestCase):
    def test_union_find(self):
        uf = UnionFind("abcde")
        uf.union("a", "b")
        uf.union("d", "e")
        uf.union("b", "e")
        self.assertEqual(sorted(uf.groups().values()), [["a", "b", "d", "e"], ["c"]])

    def test_label_jaccard_ignores_anonymous_keys(self):
        a = {"components": {"R1": {"value": "1k"}, "C1": {"value": "1u"}, "∅1": {"value": None}}}
        b = {"components": {"R1": {"value": "1k"}, "C1": {"value": "2u"}, "∅9": {"value": None}}}
        self.assertAlmostEqual(multiset_jaccard(label_multiset(a), label_multiset(b)), 1 / 3)

    def test_partitions_never_split_groups(self):
        candidates = {f"g{i}": {"cases": [f"c{i}a", f"c{i}b"] if i % 5 == 0 else [f"c{i}"],
                                "tags": Counter({"jlc" if i % 2 else "datasheet": 1})} for i in range(40)}
        parts = assign_partitions(candidates, {"test": 12, "dev": 6}, seed=3)
        self.assertEqual(parts, assign_partitions(candidates, {"test": 12, "dev": 6}, seed=3))
        self.assertFalse(set(parts["test"]) & set(parts["dev"]))
        sizes = {p: sum(len(candidates[g]["cases"]) for g in gids) for p, gids in parts.items()}
        self.assertEqual(sizes, {"test": 12, "dev": 6})
        tags = Counter(candidates[g]["tags"].most_common(1)[0][0] for g in parts["test"])
        self.assertEqual(set(tags), {"jlc", "datasheet"})


if __name__ == "__main__":
    unittest.main()
