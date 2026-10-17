import copy
import unittest
from component_lab.ocr_spans import candidate_spans, split_ocr_spans, span_polygon


class OCRSpanTests(unittest.TestCase):
    def test_compound_reference_value_preserves_raw(self):
        raw = "R3  5.1kΩ"
        spans = candidate_spans(raw)
        self.assertEqual([raw[a:b] for a,b,_ in spans], ["R3","5.1kΩ"])

    def test_reference_model(self):
        raw = "u2 RP2350B"
        self.assertEqual([raw[a:b] for a,b,_ in candidate_spans(raw)],["u2","RP2350B"])

    def test_no_substring_guess_or_character_correction(self):
        for raw in ["R10", "ADC2/P1.2", "QSPIIO3/D2/ADC2", "Rl", "R1 R2", "RXD2"]:
            self.assertEqual(candidate_spans(raw), [], raw)

    def test_wrapping_and_testpoint_marker(self):
        for raw,target in [("$R2}$","R2"),("C12)","C12"),("\\R5","R5"),("-O TP5","TP5")]:
            self.assertEqual([raw[a:b] for a,b,_ in candidate_spans(raw)],[target])

    def test_mutually_exclusive_view_and_original_immutable(self):
        rows=[{"id":"a","raw":"R1 10k","bbox":[0,0,60,10]}]
        original=copy.deepcopy(rows)
        new,audit=split_ocr_spans(rows)
        self.assertEqual(rows,original)
        self.assertEqual(len(new),2)
        self.assertNotIn("a",[x['id'] for x in new])
        self.assertEqual(audit['parents'][0]['source'],original[0])
        for child in new:
            p=child['span_provenance'];self.assertEqual(child['raw'],p['parent_raw'][p['start']:p['end']])

    def test_rotated_polygon_interpolation(self):
        row={"raw":"R1 10k","bbox":[10,20,20,80],"polygon":[[10,20],[10,80],[20,80],[20,20]]}
        self.assertEqual(span_polygon(row,0,2),[[10.,20.],[10.,40.],[20.,40.],[20.,20.]])

    def test_duplicate_ids_avoided(self):
        rows=[{"id":"a","raw":"R1 10k","bbox":[0,0,60,10]},
              {"id":"a__span_0_2","raw":"R2","bbox":[60,0,80,10]}]
        new,_=split_ocr_spans(rows)
        self.assertEqual(len({x['id'] for x in new}),len(new))
