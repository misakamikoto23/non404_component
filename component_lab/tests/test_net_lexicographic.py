"""Segment tie-breaking must never sacrifice total matched pin membership."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from component_lab.eval.metric import ScoreConfig, net_counts


def edges(offset):
    return [((offset, 0.0), (offset + 10.0, 0.0)),
            ((offset, 20.0), (offset + 10.0, 20.0))]


class NetLexicographic(unittest.TestCase):
    def test_global_pin_overlap_beats_accumulated_segment_tiebreaks(self):
        # Pin intersections: [[2, 1, 0], [0, 1, 1], [1, 0, 1]].
        # Diagonal: four pin matches, zero segment matches. The three-cycle has
        # only three pins but six segments. A per-net bound chooses the cycle.
        gt = {"g0": (set("abc"), edges(0.0)),
              "g1": (set("de"), edges(100.0)),
              "g2": (set("fg"), edges(200.0))}
        pred = {"p0": (set("abf"), edges(200.0)),
                "p1": (set("cd"), edges(0.0)),
                "p2": (set("eg"), edges(100.0))}
        for mode in ("segments", "none"):
            with self.subTest(mode=mode):
                hyper, lines, diag = net_counts(pred, gt, ScoreConfig(net_tiebreak=mode))
                self.assertEqual(hyper, {"tp": 4, "fp": 3, "fn": 3})
                self.assertEqual(lines, {"tp": 0, "fp": 6, "fn": 6})
                self.assertEqual(diag["matched_nets"], 3)

    def test_segments_still_break_equal_total_pin_overlap_ties(self):
        gt = {"g0": (set("ab"), edges(0.0)),
              "g1": (set("cd"), edges(100.0))}
        pred = {"p0": (set("ac"), edges(100.0)),
                "p1": (set("bd"), edges(0.0))}
        hyper, lines, _ = net_counts(pred, gt, ScoreConfig(net_tiebreak="segments"))
        self.assertEqual(hyper, {"tp": 2, "fp": 2, "fn": 2})
        self.assertEqual(lines, {"tp": 4, "fp": 0, "fn": 0})


if __name__ == "__main__":
    unittest.main()
