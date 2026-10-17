"""Guard against ambiguous source identity associations becoming class labels."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import native_reviewed_alignment_20261017 as alignment


class NativeReviewedMatchingTests(unittest.TestCase):
    def test_empty_sets_are_valid_and_unmatched(self):
        _, pairs = alignment.match_boxes([], [[0,0,10,10]])
        self.assertEqual(pairs, [])

    def test_duplicate_sources_cannot_claim_same_human_box(self):
        matrix, pairs = alignment.match_boxes([[0,0,10,10],[0,0,10,10]], [[0,0,10,10]])
        self.assertEqual(len(pairs), 1)
        self.assertFalse(alignment.association_gate(matrix, *pairs[0]))

    def test_duplicate_human_boxes_make_identity_ambiguous(self):
        matrix, pairs = alignment.match_boxes([[0,0,10,10]], [[0,0,10,10],[0,0,10,10]])
        self.assertEqual(len(pairs), 1)
        self.assertFalse(alignment.association_gate(matrix, *pairs[0]))

    def test_unique_shuffled_pairs_are_recovered(self):
        matrix, pairs = alignment.match_boxes([[0,0,10,10],[20,0,30,10]],
                                             [[20,0,30,10],[0,0,10,10]])
        self.assertEqual(set(pairs), {(0,1),(1,0)})
        self.assertTrue(all(alignment.association_gate(matrix,i,j) for i,j in pairs))

    def test_partial_overlap_is_diagnostic_but_not_semantic_label(self):
        matrix, pairs = alignment.match_boxes([[0,0,10,10]], [[2,0,12,10]])
        self.assertEqual(len(pairs), 1)
        self.assertFalse(alignment.association_gate(matrix,0,0))

    def test_competition_margin_applies_both_directions(self):
        self.assertFalse(alignment.association_gate(np.array([[.9,.8],[.1,.9]]),0,0))
        self.assertFalse(alignment.association_gate(np.array([[.9,.1],[.8,.9]]),0,0))

    def test_type_whitelist_does_not_match_by_prefix(self):
        for lib in ['Device:Crystal','Device:R_Potentiometer','Device:R_Pack04','Device:Rotary_Encoder']:
            self.assertNotIn(lib, alignment.TYPE_MAP)


if __name__ == '__main__':
    unittest.main()
