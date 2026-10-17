import copy
import json
import unittest

from component_lab.qwen_identity import apply_choices, parse_choices


def fixture():
    baseline = {"a": {"components": {
        "__unresolved_detection_0001": {"type": "component", "bbox": [1, 2, 3, 4], "Name": None, "value": None},
        "R2": {"type": "r", "bbox": [5, 6, 7, 8], "Name": None, "value": "10 kΩ"}}, "pins": {}, "nets": {}}}
    targets = [{"id": "T1", "case": "a", "source_id": "c1", "baseline_key": "__unresolved_detection_0001",
                "field": "reference", "candidates": [{"id": "O1", "text_id": "o1", "raw": "R1", "owner_source_id": None}]},
               {"id": "T2", "case": "a", "source_id": "c2", "baseline_key": "R2", "field": "value",
                "candidates": [{"id": "O2", "text_id": "o2", "raw": "4.7K", "owner_source_id": None}]}]
    return baseline, targets


class QwenIdentityTests(unittest.TestCase):
    def test_raw_literals_and_no_input_mutation(self):
        b, t = fixture(); before = copy.deepcopy(b)
        choices = [{"id": "T1", "text_id": "O1", "uncertain": False}, {"id": "T2", "text_id": "O2", "uncertain": False}]
        out, _ = apply_choices(b, t, parse_choices(json.dumps({"items": choices}), t))
        self.assertEqual(b, before)
        self.assertEqual(out['a']['components']['R2']['value'], '4.7K')
        self.assertEqual(out['a']['components']['R1']['type'], 'r')
        self.assertEqual(out['a']['components']['R1']['bbox'], [1, 2, 3, 4])

    def test_generated_text_is_rejected(self):
        _, t = fixture()
        for items in [[{"id": "T1", "text_id": "R1", "uncertain": False}],
                      [{"id": "T1", "text_id": None, "uncertain": "false"}],
                      [{"id": "T1", "text_id": None, "uncertain": False}] * 2]:
            with self.assertRaises(ValueError): parse_choices(json.dumps({"items": items}), t)

    def test_empty_only_does_not_rewrite_value(self):
        b, t = fixture()
        out, _ = apply_choices(b, t, [{"id": "T2", "text_id": "O2", "uncertain": False}], empty_only=True)
        self.assertEqual(out, b)

    def test_existing_key_is_locked(self):
        b, t = fixture(); t[0]['candidates'][0]['raw'] = 'R2'
        out, meta = apply_choices(b, t, [{"id": "T1", "text_id": "O1", "uncertain": False}])
        self.assertEqual(out, b); self.assertEqual(meta['blocked'][0]['reason'], 'key_already_exported')

    def test_other_owner_is_locked(self):
        b, t = fixture(); t[1]['candidates'][0]['owner_source_id'] = 'c3'
        out, meta = apply_choices(b, t, [{"id": "T2", "text_id": "O2", "uncertain": False}])
        self.assertEqual(out, b); self.assertEqual(meta['blocked'][0]['reason'], 'other_owner_locked')

    def test_cross_target_token_conflict_abstains(self):
        b, t = fixture(); t[1]['candidates'][0]['text_id'] = 'o1'
        out, meta = apply_choices(b, t, [{"id": "T1", "text_id": "O1", "uncertain": False},
                                     {"id": "T2", "text_id": "O2", "uncertain": False}])
        self.assertEqual(out, b); self.assertEqual(len(meta['blocked']), 2)

    def test_uncertain_does_not_apply(self):
        b, t = fixture()
        out, _ = apply_choices(b, t, [{"id": "T2", "text_id": "O2", "uncertain": True}])
        self.assertEqual(out, b)
