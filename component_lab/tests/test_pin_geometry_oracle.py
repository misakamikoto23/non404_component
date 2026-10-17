"""Non-greedy location matching and inference input separation for the pilot."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_pin_geometry_oracle import match_locations, boundary_offset


class LocationMatching(unittest.TestCase):
    def test_max_cardinality_beats_nearest_pair_greedy(self):
        pairs = match_locations([(0, 0), (3, 0)], [(1, 0), (-2, 0)], tolerance=3)
        self.assertEqual({(i, j) for i, j, _ in pairs}, {(0, 1), (1, 0)})

    def test_inclusive_distance_and_unmatched_outlier(self):
        self.assertEqual(len(match_locations([(0, 0)], [(3, 4)])), 1)
        self.assertEqual(match_locations([(0, 0)], [(3, 4.01)]), [])
        self.assertEqual(match_locations([], [(3, 4)]), [])

    def test_signed_box_distance(self):
        self.assertEqual(boundary_offset((5, 5), [0, 0, 10, 10]), -5)
        self.assertEqual(boundary_offset((10, 5), [0, 0, 10, 10]), 0)
        self.assertEqual(boundary_offset((13, 14), [0, 0, 10, 10]), 5)


if __name__ == "__main__":
    unittest.main()
