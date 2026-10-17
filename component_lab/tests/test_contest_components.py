import copy
import unittest

from component_lab.contest_components import associate_variants, export_contest, refine_types
from component_lab.text_matching import associate_structured


class ComponentIdentityTests(unittest.TestCase):
    def test_public_type_spellings_are_explicit(self):
        from component_lab.contest_components import TYPE_MAP
        for source, target in {"battery": "battary", "transformer": "transfomer", "antenna": "ant", "optocoupler": "opt"}.items():
            self.assertEqual(TYPE_MAP[source], target)
    def test_negative_costs_do_not_collapse_distinct_candidates(self):
        comps = [{"id": "near", "type": "resistor", "bbox": [20, 15, 30, 35]},
                 {"id": "far", "type": "resistor", "bbox": [26, 15, 36, 35]}]
        texts = [{"id": "t", "text": "R1", "bbox": [0, 20, 20, 30]}]
        old = associate_structured(comps, texts)
        new = associate_structured(comps, texts, clamp_costs=False)
        self.assertIsNone(old["components"][0]["reference"])
        self.assertEqual(new["components"][0]["reference"], "R1")

    def test_prior_changes_abstention_but_preserves_strong_conflict(self):
        rows = [{"id": "a", "type": "component", "association_evidence": {"reference": {"raw": "R2"}}},
                {"id": "b", "type": "capacitor", "classification": {"score": .99},
                 "association_evidence": {"reference": {"raw": "R3"}}}]
        result, changes = refine_types({"components": rows})
        self.assertEqual(result[0]["type"], "resistor")
        self.assertEqual(result[1]["type"], "capacitor")
        self.assertEqual(len(changes), 1)

    def test_export_preserves_case_spaces_and_units(self):
        row = {"id": "a", "type": "resistor", "bbox": [0, 10, 20, 30],
               "association_evidence": {"reference": {"raw": "r1"}, "value": {"raw": "10 KΩ"}}}
        pred, _ = export_contest({"components": [row]}, 100)
        self.assertEqual(pred["components"]["r1"]["value"], "10 KΩ")
        self.assertEqual(pred["components"]["r1"]["bbox"], [0, 70, 20, 90])

    def test_ic_name_fallback_keeps_own_raw_token(self):
        row = {"id": "a", "type": "ic_or_module", "bbox": [0, 10, 20, 30],
               "association_evidence": {"reference": {"raw": "U1"}, "value": {"raw": "2.4GHz"}}}
        pred, _ = export_contest({"components": [row]}, 100, ic_value_as_name=True)
        self.assertEqual(pred["components"]["U1"]["Name"], "2.4GHz")

    def test_variants_leave_inputs_immutable_and_ambiguous_Q_unforced(self):
        comps = [{"id": "q", "type": "component", "bbox": [0, 25, 20, 55]}]
        texts = [{"id": "t", "text": "Q1", "bbox": [25, 25, 45, 35]}]
        before = copy.deepcopy((comps, texts))
        results, _ = associate_variants(comps, texts)
        self.assertEqual(results["identity_v1"]["components"][0]["type"], "component")
        self.assertEqual((comps, texts), before)
