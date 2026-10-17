import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pilot_qwen_same_roi import parse_ids, request_hash


class SameRoiProtocolTests(unittest.TestCase):
    def setUp(self):
        self.targets = [{'id': 'S1', 'roi_candidates': [
            {'id': 'O1', 'raw': 'u2'}, {'id': 'O2', 'raw': '0.01 μF'}]}]
        self.item = {'id': 'S1', 'reference_id': 'O1', 'Name_id': None,
                     'value_id': 'O2', 'uncertain_fields': []}

    def parse(self, items):
        return parse_ids(json.dumps({'items': items}), self.targets)

    def test_ids_return_exact_raw_without_normalizing(self):
        self.assertEqual(self.parse([self.item])[0], {
            'id': 'S1', 'reference': 'u2', 'Name': None,
            'value': '0.01 μF', 'uncertain_fields': []})

    def test_unknown_id_and_free_literal_are_rejected(self):
        for value in ['O99', '10kΩ', 2]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse([dict(self.item, value_id=value)])

    def test_missing_duplicate_and_foreign_target_are_rejected(self):
        for items in [[], [self.item, self.item], [dict(self.item, id='S2')]]:
            with self.subTest(items=items), self.assertRaises(ValueError):
                self.parse(items)

    def test_invalid_or_duplicate_uncertainty_is_rejected(self):
        for fields in [['bbox'], ['value', 'value'], 'value']:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.parse([dict(self.item, uncertain_fields=fields)])

    def test_cache_key_includes_endpoint_pixels_catalog_and_protocol(self):
        original = {'model': 'qwen3.8-flash', 'prompt': 'select IDs',
                    'image': 'pixels', 'catalog': {'O1': 'u2'}}
        digest = request_hash(original, 'https://first.example/v1')
        self.assertEqual(digest, request_hash(dict(reversed(list(original.items()))),
                                              'https://first.example/v1'))
        self.assertNotEqual(digest, request_hash(original, 'https://second.example/v1'))
        for changed in [dict(original, image='other pixels'),
                        dict(original, catalog={'O1': 'U2'}),
                        dict(original, prompt='direct reading')]:
            self.assertNotEqual(digest, request_hash(changed, 'https://first.example/v1'))


if __name__ == '__main__':
    unittest.main()
