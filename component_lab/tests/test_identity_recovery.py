from copy import deepcopy
import json
import unittest

from component_lab.identity_recovery import apply_recovery, parse_recovery, unresolved_key, record_fingerprint


class IdentityRecoveryTests(unittest.TestCase):
    def fixture(self):
        named = {'type': 'r', 'bbox': [1, 2, 8, 9], 'Name': None, 'value': None}
        unknown = {'type': 'component', 'bbox': [20, 2, 28, 9], 'Name': 'old', 'value': 'old'}
        baseline = {'a': {'components': {'R1': named, '__unresolved_detection_0001': unknown},
                          'pins': {'P': {'point': [1, 2]}}, 'nets': {'N': {'edges': []}}}}
        targets = [{'id': 'T1', 'case': 'a', 'baseline_key': '__unresolved_detection_0001', 'source_id': 'c1'}]
        readings = [{'id': 'T1', 'reference': 'c2', 'Name': None, 'value': '0.01 μF',
                     'family': 'capacitor', 'target_complete': True, 'uncertain_fields': []}]
        return baseline, targets, readings

    def test_new_identity_preserves_old_null_raw_text_geometry_and_input(self):
        baseline, targets, readings = self.fixture()
        before = deepcopy(baseline)
        out, audit = apply_recovery(baseline, targets, readings)
        self.assertEqual(baseline, before)
        self.assertEqual(out['a']['components']['R1'], before['a']['components']['R1'])
        self.assertEqual(out['a']['components']['c2']['value'], '0.01 μF')
        self.assertEqual(out['a']['components']['c2']['type'], 'c')
        self.assertEqual(out['a']['components']['c2']['bbox'], before['a']['components']['__unresolved_detection_0001']['bbox'])
        self.assertEqual(out['a']['pins'], before['a']['pins'])
        self.assertEqual(out['a']['nets'], before['a']['nets'])
        self.assertEqual(audit['named_before_sha256'], audit['original_named_after_sha256'])

    def test_named_source_is_locked_even_if_directly_targeted(self):
        baseline, targets, readings = self.fixture()
        targets[0]['baseline_key'] = 'R1'
        out, audit = apply_recovery(baseline, targets, readings)
        self.assertEqual(out, baseline)
        self.assertEqual(audit['blocked'][0]['reason'], 'named_source_locked')

    def test_existing_key_cannot_be_overwritten(self):
        baseline, targets, readings = self.fixture()
        readings[0]['reference'] = 'R1'
        out, audit = apply_recovery(baseline, targets, readings)
        self.assertEqual(out, baseline)
        self.assertEqual(audit['blocked'][0]['reason'], 'existing_or_competing_key')

    def test_competing_new_keys_are_both_rejected_order_independently(self):
        baseline, targets, readings = self.fixture()
        baseline['a']['components']['__unresolved_detection_0002'] = deepcopy(baseline['a']['components']['__unresolved_detection_0001'])
        targets.append(dict(targets[0], id='T2', baseline_key='__unresolved_detection_0002'))
        readings.append(dict(readings[0], id='T2'))
        for ordered in [readings, list(reversed(readings))]:
            out, audit = apply_recovery(baseline, targets, ordered)
            self.assertEqual(out, baseline)
            self.assertEqual(len(audit['blocked']), 2)

    def test_incomplete_symbol_or_uncertain_reference_abstains(self):
        baseline, targets, readings = self.fixture()
        for extra in [{'target_complete': False}, {'family': 'not_component'}, {'uncertain_fields': ['reference']}]:
            out, _ = apply_recovery(baseline, targets, [dict(readings[0], **extra)])
            self.assertEqual(out, baseline)

    def test_key_only_replay_preserves_unresolved_fields(self):
        baseline, targets, readings = self.fixture()
        out, _ = apply_recovery(baseline, targets, readings, include_fields=False)
        self.assertEqual(out['a']['components']['c2']['value'], 'old')

    def test_uncertain_text_does_not_replace_existing_machine_field(self):
        baseline, targets, readings = self.fixture()
        readings[0]['uncertain_fields'] = ['value']
        out, _ = apply_recovery(baseline, targets, readings)
        self.assertEqual(out['a']['components']['c2']['value'], 'old')

    def test_unsupported_or_hidden_id_is_not_created(self):
        baseline, targets, readings = self.fixture()
        for reference in ['∅75', 'D2_FLASH', 'A5', ' R3 ', '__unresolved_detection_2']:
            out, _ = apply_recovery(baseline, targets, [dict(readings[0], reference=reference)])
            self.assertEqual(out, baseline)

    def test_duplicate_source_and_duplicate_reading_are_rejected(self):
        baseline, targets, readings = self.fixture()
        with self.assertRaises(ValueError):
            apply_recovery(baseline, targets + [dict(targets[0], id='T2')], readings)
        with self.assertRaises(ValueError):
            apply_recovery(baseline, targets, readings + readings)

    def test_response_requires_exact_schema_and_every_target(self):
        _, targets, readings = self.fixture()
        self.assertEqual(parse_recovery(json.dumps({'items': readings}), targets), readings)
        bad = [[], readings + readings, [dict(readings[0], family=[])],
               [dict(readings[0], target_complete='true')], [dict(readings[0], uncertain_fields=['bbox'])],
               [dict(readings[0], value=0)], [dict(readings[0], extra='instruction')]]
        for items in bad:
            with self.subTest(items=items), self.assertRaises(ValueError):
                parse_recovery(json.dumps({'items': items}), targets)

    def test_only_reserved_detection_namespace_is_mutable(self):
        self.assertTrue(unresolved_key('__unresolved_detection_0001_'))
        for key in ['∅75', 'U1', '__unresolved_something', '__unresolved_detection_U1']:
            self.assertFalse(unresolved_key(key))

    def test_stale_baseline_record_fails_closed(self):
        baseline, targets, readings = self.fixture()
        targets[0]['expected_record_sha256'] = record_fingerprint(baseline['a']['components']['__unresolved_detection_0001'])
        baseline['a']['components']['__unresolved_detection_0001']['bbox'][0] += 1
        with self.assertRaisesRegex(ValueError, 'Stale'):
            apply_recovery(baseline, targets, readings)


if __name__ == '__main__':
    unittest.main()
