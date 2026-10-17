import unittest
from component_lab.geometry import convert_bbox
from component_lab.association import associate


class CoreTests(unittest.TestCase):
    def test_bottom_left_conversion(self):
        self.assertEqual(convert_bbox([10, 20, 30, 40], 100, 80, "bottom-left"), [10, 40, 30, 60])

    def test_reference_and_value(self):
        rows = [{"id": "a", "bbox": [30, 30, 50, 60]}]
        texts = [{"text": "R12", "bbox": [55, 32, 75, 42]}, {"text": "10kΩ", "bbox": [55, 48, 80, 58]}]
        self.assertEqual(associate(rows, texts), [])
        self.assertEqual(rows[0]["reference"], "R12")
        self.assertEqual(rows[0]["value"], "10kΩ")
        self.assertIsNone(rows[0]["Name"])

    def test_ambiguous_text_kept_unassigned(self):
        rows = [{"id": "a", "bbox": [0, 0, 10, 10]}, {"id": "b", "bbox": [30, 0, 40, 10]}]
        result = associate(rows, [{"text": "R1", "bbox": [18, 0, 22, 10]}])
        self.assertEqual(result[0]["reason"], "ambiguous_owner")
        self.assertTrue(all(row["reference"] is None for row in rows))

    def test_duplicate_reference_requires_review(self):
        rows = [{"id": "a", "bbox": [0, 0, 10, 10]}, {"id": "b", "bbox": [200, 0, 210, 10]}]
        associate(rows, [{"text": "R1", "bbox": [12, 0, 22, 10]}, {"text": "R1", "bbox": [212, 0, 222, 10]}])
        self.assertTrue(all(row["reference"] is None and "duplicate_reference" in row["review"] for row in rows))


if __name__ == "__main__":
    unittest.main()
