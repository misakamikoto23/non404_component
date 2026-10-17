"""Contest 6 local scorer: synthetic cases, CPU only, no dataset needed."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from component_lab.eval.cases import case_id, make_split, source_tag
from component_lab.eval.matching import hungarian_max
from component_lab.eval.metric import ScoreConfig, score_case
from component_lab.eval.report import build_configs, markdown, score_directory, summarize


def pt(x, y):
    return {"x": x, "y": y}


def sample_case():
    return {
        "components": {
            "U1": {"Name": "RP2350A", "type": "box", "value": None, "bbox": [100.0, 100.0, 300.0, 400.0]},
            "R1": {"Name": "R1", "type": "r", "value": "10k", "bbox": [400.0, 200.0, 420.0, 260.0]},
            "D.1": {"Name": "D.1", "type": "led", "value": None, "bbox": [500.0, 200.0, 520.0, 240.0]},
            "∅7": {"Name": "GND", "type": "gnd", "value": None, "bbox": [400.0, 100.0, 420.0, 120.0]},
        },
        "pins": {
            "U1": {"pin_1": {"pinname": "VCC", "point": pt(300, 300)},
                   "pin_A5": {"pinname": "CC1", "point": pt(300, 250)}},
            "R1": {"pin_1": {"pinname": "1", "point": pt(410, 260)},
                   "pin_2": {"pinname": "2", "point": pt(410, 200)}},
            "D.1": {"pin_1": {"pinname": "A", "point": pt(510, 240)}},
            "∅7": {"pin_1": {"pinname": "GND", "point": pt(410, 120)}},
        },
        "nets": {
            "N1": {"hyperGraph": "(U1.1,R1.1,D.1.1)",
                   "edges": {"edge_1": [pt(300, 300), pt(410, 300)], "edge_2": [pt(410, 300), pt(410, 260)],
                             "edge_3": [pt(410, 300), pt(510, 300)]}},
            "GND": {"hyperGraph": "(R1.2,∅7.1)", "edges": {"edge_1": [pt(410, 200), pt(410, 120)]}},
            "CC": {"hyperGraph": "(U1.A5)", "edges": {"edge_1": [pt(300, 250), pt(350, 250)]}},
        },
    }


EMPTY = {"components": {}, "pins": {}, "nets": {}}


class PerfectAndEmpty(unittest.TestCase):
    def test_perfect_prediction_scores_100_under_every_policy(self):
        for policy, metric in itertools.product(("strict", "observable_diagnostic"), ("max_corner", "center")):
            result = score_case(sample_case(), sample_case(), ScoreConfig(anon_policy=policy, bbox_metric=metric))
            self.assertAlmostEqual(result["score"], 100.0, msg=(policy, metric))

    def test_empty_missing_and_invalid_predictions_score_0(self):
        self.assertEqual(score_case(EMPTY, sample_case())["score"], 0.0)
        missing = score_case(None, sample_case())
        self.assertEqual((missing["status"], missing["score"]), ("missing", 0.0))
        broken = sample_case()
        broken["nets"]["N1"]["edges"]["edge_1"].append(pt(1, 1))  # three-point edge = format error
        invalid = score_case(broken, sample_case())
        self.assertEqual((invalid["status"], invalid["score"]), ("invalid", 0.0))
        self.assertEqual(score_case({"components": {}}, sample_case())["score"], 0.0)

    def test_weights(self):
        pred = sample_case()
        pred["nets"] = {}
        result = score_case(pred, sample_case())
        self.assertAlmostEqual(result["score"], 55.0)  # component 0.30 + pin 0.25


class ComponentRules(unittest.TestCase):
    def component_f1(self, pred, config=ScoreConfig()):
        return score_case(pred, sample_case(), config)["component_f1"]

    def test_strict_value_for_resistor_and_name_for_ic(self):
        pred = sample_case()
        pred["components"]["R1"]["value"] = "10K"
        self.assertLess(self.component_f1(pred), 1)
        pred = sample_case()
        pred["components"]["U1"]["Name"] = "RP2350B"
        self.assertLess(self.component_f1(pred), 1)
        pred = sample_case()
        pred["components"]["R1"]["Name"] = "anything"  # not compared for r
        pred["components"]["D.1"]["Name"] = "anything"  # other types: no text
        self.assertEqual(self.component_f1(pred), 1)

    def test_key_type_and_bbox_tolerance(self):
        pred = sample_case()
        pred["components"]["R2"] = pred["components"].pop("R1")
        self.assertLess(self.component_f1(pred), 1)
        pred = sample_case()
        pred["components"]["R1"]["type"] = "c"
        self.assertLess(self.component_f1(pred), 1)
        pred = sample_case()
        pred["components"]["U1"]["bbox"] = [120.0, 80.0, 320.0, 380.0]  # every coordinate off by exactly 20
        self.assertEqual(self.component_f1(pred), 1)
        pred["components"]["U1"]["bbox"] = [121.0, 80.0, 320.0, 380.0]
        self.assertLess(self.component_f1(pred), 1)

    def test_bbox_metric_is_configurable(self):
        pred = sample_case()
        pred["components"]["U1"]["bbox"] = [70.0, 70.0, 330.0, 430.0]  # grown by 30 px, same center
        self.assertLess(self.component_f1(pred, ScoreConfig(bbox_metric="max_corner")), 1)
        self.assertEqual(self.component_f1(pred, ScoreConfig(bbox_metric="center")), 1)


class PinRules(unittest.TestCase):
    def test_pinname_and_point_tolerance(self):
        pred = sample_case()
        pred["pins"]["U1"]["pin_1"]["point"] = pt(303, 304)  # distance 5
        self.assertEqual(score_case(pred, sample_case())["pin_f1"], 1)
        pred["pins"]["U1"]["pin_1"]["point"] = pt(304, 304)
        self.assertLess(score_case(pred, sample_case())["pin_f1"], 1)
        pred = sample_case()
        pred["pins"]["U1"]["pin_1"]["pinname"] = "vcc"
        self.assertLess(score_case(pred, sample_case())["pin_f1"], 1)


class NetRules(unittest.TestCase):
    def test_net_keys_free_and_edges_orientation_free(self):
        pred = sample_case()
        pred["nets"] = {f"renamed_{k}": v for k, v in pred["nets"].items()}
        edge = pred["nets"]["renamed_N1"]["edges"]["edge_1"]
        edge.reverse()
        result = score_case(pred, sample_case())
        self.assertEqual((result["net_hypergraph_f1"], result["net_line_f1"]), (1, 1))

    def test_missing_member_and_extra_edge(self):
        pred = sample_case()
        pred["nets"]["N1"]["hyperGraph"] = "(U1.1,R1.1)"
        pred["nets"]["N1"]["edges"]["edge_4"] = [pt(0, 0), pt(1, 1)]
        result = score_case(pred, sample_case())
        hyper, lines = result["counts"]["net_hypergraph_f1"], result["counts"]["net_line_f1"]
        self.assertEqual((hyper["tp"], hyper["fp"], hyper["fn"]), (5, 0, 1))
        self.assertEqual((lines["tp"], lines["fp"], lines["fn"]), (5, 1, 0))

    def test_lines_in_wrong_net_do_not_count(self):
        pred = sample_case()
        moved = pred["nets"]["CC"]["edges"].pop("edge_1")
        pred["nets"]["N1"]["edges"]["edge_9"] = moved
        lines = score_case(pred, sample_case())["counts"]["net_line_f1"]
        self.assertEqual((lines["tp"], lines["fp"], lines["fn"]), (4, 1, 1))

    def test_merged_net_matches_one_gt_net(self):
        pred = sample_case()
        gnd = pred["nets"].pop("GND")
        pred["nets"]["N1"]["hyperGraph"] = "(U1.1,R1.1,D.1.1,R1.2,∅7.1)"
        pred["nets"]["N1"]["edges"].update({"edge_9": gnd["edges"]["edge_1"]})
        hyper = score_case(pred, sample_case())["counts"]["net_hypergraph_f1"]
        self.assertEqual((hyper["tp"], hyper["fp"], hyper["fn"]), (4, 2, 2))


class NetTiebreak(unittest.TestCase):
    def test_pinless_nets_pair_only_with_segment_tiebreak(self):
        gt = sample_case()
        gt["nets"]["floating"] = {"hyperGraph": "()", "edges": {"edge_1": [pt(0, 0), pt(50, 0)]}}
        segments = score_case(deepcopy(gt), gt, ScoreConfig(net_tiebreak="segments"))
        none = score_case(deepcopy(gt), gt, ScoreConfig(net_tiebreak="none"))
        self.assertEqual(segments["counts"]["net_line_f1"]["fn"], 0)
        self.assertEqual(none["counts"]["net_line_f1"]["fn"], 1)
        self.assertEqual(segments["net_hypergraph_f1"], none["net_hypergraph_f1"])

    def test_equal_overlap_tie_uses_segments_only_when_asked(self):
        pred = sample_case()
        cc = pred["nets"].pop("CC")
        pred["nets"]["CC_a"] = {"hyperGraph": "(U1.A5)", "edges": {"edge_1": [pt(900, 900), pt(950, 900)]}}
        pred["nets"]["CC_b"] = cc
        segments = score_case(pred, sample_case(), ScoreConfig(net_tiebreak="segments"))
        none = score_case(pred, sample_case(), ScoreConfig(net_tiebreak="none"))
        self.assertEqual(segments["counts"]["net_line_f1"]["tp"], 5)
        self.assertEqual(none["counts"]["net_line_f1"]["tp"], 4)  # solver keeps the first sorted net, CC_a


class AnonymousPolicy(unittest.TestCase):
    def test_observable_diagnostic_drops_anonymous_items_from_both_sides(self):
        pred = sample_case()  # a predictor that cannot know internal ∅ ids
        del pred["components"]["∅7"], pred["pins"]["∅7"]
        pred["nets"]["GND"]["hyperGraph"] = "(R1.2)"
        include = score_case(pred, sample_case(), ScoreConfig(anon_policy="strict"))
        exclude = score_case(pred, sample_case(), ScoreConfig(anon_policy="observable_diagnostic"))
        self.assertLess(include["score"], 100)
        self.assertAlmostEqual(exclude["score"], 100)
        self.assertEqual(include["counts"]["component_f1"]["fn"], 1)
        self.assertEqual(exclude["counts"]["component_f1"], {"tp": 3, "fp": 0, "fn": 0})

    def test_observable_diagnostic_ignores_guessed_anonymous_keys_in_prediction(self):
        pred = sample_case()
        pred["components"]["∅99"] = pred["components"].pop("∅7")
        pred["pins"]["∅99"] = pred["pins"].pop("∅7")
        pred["nets"]["GND"]["hyperGraph"] = "(R1.2,∅99.1)"
        self.assertLess(score_case(pred, sample_case(), ScoreConfig(anon_policy="strict"))["score"], 100)
        self.assertAlmostEqual(score_case(pred, sample_case(), ScoreConfig(anon_policy="observable_diagnostic"))["score"], 100)

    def test_net_with_only_anonymous_members_is_dropped_with_its_edges(self):
        gt = sample_case()
        gt["nets"]["GND"]["hyperGraph"] = "(∅7.1)"
        pred = deepcopy(gt)
        del pred["nets"]["GND"]
        self.assertAlmostEqual(score_case(pred, gt, ScoreConfig(anon_policy="observable_diagnostic"))["score"], 100)


class Oracle(unittest.TestCase):
    def test_oracle_replaces_stage_with_gt(self):
        pred = sample_case()
        pred["components"] = {}
        pred["nets"] = {}
        self.assertEqual(score_case(pred, sample_case(), ScoreConfig(oracle=frozenset({"components"})))["component_f1"], 1)
        both = score_case(pred, sample_case(), ScoreConfig(oracle=frozenset({"components", "nets"})))
        self.assertAlmostEqual(both["score"], 100)
        self.assertEqual(score_case(None, sample_case(), ScoreConfig(oracle=frozenset({"nets"})))["score"], 0)


class Matching(unittest.TestCase):
    def test_hungarian_matches_brute_force(self):
        rng = random.Random(0)
        for _ in range(200):
            rows, cols = rng.randint(1, 5), rng.randint(1, 5)
            weights = [[rng.choice([0, 0, 1, 2, 3, 7]) for _ in range(cols)] for _ in range(rows)]
            got = sum(weights[i][j] for i, j in hungarian_max(weights))
            best = 0
            for perm in itertools.permutations(range(max(rows, cols)), rows):
                best = max(best, sum(weights[i][j] for i, j in enumerate(perm) if j < cols))
            self.assertEqual(got, best)


class CasesAndSplit(unittest.TestCase):
    def test_case_ids(self):
        for stem in ("0001-KiCad_target_named", "0001-KiCad", "0001", "case_001", "case_1"):
            self.assertEqual(case_id(stem), "1")
        self.assertEqual(source_tag("0003-Altium Designer_target"), "altium designer")

    def test_directory_scoring_counts_missing_cases_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt_dir, pred_dir = Path(tmp, "gt"), Path(tmp, "pred")
            for case in ("0001-KiCad", "0002-jlc"):
                (gt_dir / case.split("-")[0]).mkdir(parents=True)
                (gt_dir / case.split("-")[0] / f"{case}_target_named.json").write_text(
                    json.dumps(sample_case(), ensure_ascii=False), encoding="utf-8")
            (pred_dir / "case_001").mkdir(parents=True)
            (pred_dir / "case_001" / "case_001.json").write_text(json.dumps(sample_case()), encoding="utf-8")
            rows = score_directory(pred_dir, gt_dir, build_configs(["strict"], "max_corner", 20, 5, [], False))
            self.assertEqual([r["status"] for r in rows], ["ok", "missing"])
            self.assertAlmostEqual(summarize(rows)[0]["score"], 50.0)

            broken = sample_case()
            broken["nets"] = {}
            (pred_dir / "case_001" / "case_001.json").write_text(json.dumps(broken), encoding="utf-8")
            configs = build_configs(["strict"], "max_corner", 20, 5, [], True)
            rows = score_directory(pred_dir, gt_dir, configs)
            report = markdown(rows, configs, pred_dir, gt_dir, None)
            nets_row = next(line for line in report.splitlines() if line.startswith("| strict | strict | nets |"))
            self.assertIn("| 2 | 1 | 50.00 |", nets_row)
            self.assertTrue(nets_row.split(" | ")[7] == "+22.50")  # (100 - 55) / 2 cases

    def test_split_is_deterministic_stratified_and_skips_duplicates(self):
        tags = ["jlc"] * 50 + ["datasheet"] * 30 + ["altium designer"] * 10 + ["kicad"] * 3 + ["other"] * 7
        cases = [{"case": str(i + 1), "stem": f"{i + 1:04d}-{tag}", "split": "train200", "tag": tag,
                  "pixel_sha256": f"h{i}"} for i, tag in enumerate(tags)]
        cases[1]["pixel_sha256"] = cases[0]["pixel_sha256"]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp, "_manifest.json")
            manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            first, second = make_split(manifest, 20, 7), make_split(manifest, 20, 7)
            other_seed = make_split(manifest, 20, 8)
        self.assertEqual(first, second)
        self.assertNotEqual(first["cases"], other_seed["cases"])
        self.assertEqual(first["size"], 20)
        self.assertEqual({t: s["selected"] for t, s in first["strata"].items()},
                         {"jlc": 10, "datasheet": 6, "altium designer": 2, "kicad": 1, "other": 1})
        self.assertFalse({"0001", "0002"} & {c["case"] for c in first["cases"]})


if __name__ == "__main__":
    unittest.main()
