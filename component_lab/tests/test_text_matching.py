import copy
import importlib.util
from pathlib import Path
import unittest

from component_lab.text_roles import parse_text_role
from component_lab.text_matching import associate_structured


def text(raw, box, identifier="t1"):
    return {"id": identifier, "text": raw, "bbox": box, "score": .99}


class TextRolesTests(unittest.TestCase):
    def test_unit_reference_and_raw_preserved(self):
        row = parse_text_role(text(" U1B ", [0, 0, 20, 10]))
        self.assertEqual(row["raw"], " U1B ")
        ref = row["role_candidates"][0]
        self.assertEqual((ref["value"], ref["physical_reference"], ref["unit"]), ("U1B", "U1", "B"))

    def test_confusable_is_not_corrected(self):
        row = parse_text_role(text("Rl", [0, 0, 20, 10]))
        self.assertEqual(row["raw"], "Rl")
        self.assertTrue(row["role_candidates"][0]["requires_review"])
        result = associate_structured([{"bbox": [0, 15, 20, 35], "type": "resistor"}], [row])
        self.assertIsNone(result["components"][0]["reference"])

    def test_digits_are_not_unconditional_value(self):
        row = parse_text_role(text("10", [0, 0, 20, 10]))
        self.assertEqual(row["primary_role"], "pin")
        self.assertTrue(next(c for c in row["role_candidates"] if c["role"] == "value")["requires_context"])

    def test_model_and_net_distinct(self):
        self.assertEqual(parse_text_role(text("LM358", [0, 0, 20, 10]))["primary_role"], "part_number")
        self.assertEqual(parse_text_role(text("GND", [0, 0, 20, 10]))["primary_role"], "net")

    def test_pin_signals_and_ocr_artifacts_not_models(self):
        for raw in ["BOOT1", "CS_260", "RTC_I2C_SDA", "U13.", "V0.9", "SC0402", "AS\t5\t208", "GPI014", "PIO4"]:
            row = parse_text_role(text(raw, [0, 0, 20, 10]))
            self.assertFalse(any(c["role"] == "part_number" for c in row["role_candidates"]), raw)

    def test_merged_ocr_model_requires_review(self):
        for raw in ["C7AMS1117-3.3", "R10 HC-PZ254-11.5L"]:
            row = parse_text_role(text(raw, [0, 0, 80, 10]))
            self.assertTrue(next(c for c in row["role_candidates"] if c["role"] == "part_number")["requires_review"])


class TextMatchingTests(unittest.TestCase):
    def test_reference_value_group_and_inputs_immutable(self):
        comps = [{"id": "r", "bbox": [30, 30, 50, 60], "type": "resistor"}]
        texts = [text("R12", [55, 32, 75, 42]), text("10kΩ", [55, 48, 80, 58], "t2")]
        before = copy.deepcopy((comps, texts))
        result = associate_structured(comps, texts)
        self.assertEqual(result["components"][0]["reference"], "R12")
        self.assertEqual(result["components"][0]["value"], "10kΩ")
        self.assertEqual((comps, texts), before)

    def test_scale_invariance(self):
        comps = [{"id": "r", "bbox": [30, 30, 50, 60], "type": "resistor"}]
        texts = [text("R1", [55, 32, 75, 42]), text("10k", [55, 48, 80, 58], "t2")]
        small = associate_structured(comps, texts)
        for row in comps + texts:
            row["bbox"] = [v * 4 for v in row["bbox"]]
        large = associate_structured(comps, texts)
        self.assertEqual([(a["component_id"], a["text_id"]) for a in small["assignments"]],
                         [(a["component_id"], a["text_id"]) for a in large["assignments"]])

    def test_equal_geometry_abstains(self):
        comps = [{"id": "a", "bbox": [0, 0, 10, 10]}, {"id": "b", "bbox": [30, 0, 40, 10]}]
        result = associate_structured(comps, [text("R1", [18, 0, 22, 10])])
        self.assertTrue(all(row["reference"] is None for row in result["components"]))
        self.assertEqual(result["unassigned_texts"][0]["reason"], "ambiguous_owner")

    def test_type_is_soft_not_forced(self):
        comps = [{"id": "a", "bbox": [0, 0, 10, 10], "type": "capacitor"}]
        result = associate_structured(comps, [text("R1", [12, 0, 22, 10])])
        self.assertEqual(result["components"][0]["reference"], "R1")

    def test_ground_does_not_require_reference(self):
        result = associate_structured([{"bbox": [0, 15, 20, 35], "type": "ground"}],
                                      [text("GND", [0, 0, 20, 10])])
        self.assertIsNone(result["components"][0]["reference"])
        self.assertNotIn("reference_unresolved", result["components"][0]["review"])
        self.assertEqual(result["unassigned_texts"][0]["primary_role"], "net")

    def test_power_voltage_retained_for_net_stage(self):
        result = associate_structured([{"bbox": [0, 15, 20, 35], "type": "power_port"}],
                                      [text("+5V", [0, 0, 20, 10])])
        self.assertIsNone(result["components"][0]["value"])
        self.assertEqual(len(result["unassigned_texts"]), 1)

    def test_pure_digit_requires_group_and_compatible_type(self):
        component = [{"bbox": [0, 25, 20, 55], "type": "resistor"}]
        alone = associate_structured(component, [text("10", [25, 35, 45, 45])])
        self.assertIsNone(alone["components"][0]["value"])
        paired = associate_structured(component, [text("R1", [25, 20, 45, 30]),
                                                  text("10", [25, 35, 45, 45], "t2")])
        self.assertEqual(paired["components"][0]["value"], "10")

    def test_multi_unit_preserved(self):
        components = [{"id": "a", "bbox": [0, 20, 20, 50], "type": "opamp"},
                      {"id": "b", "bbox": [200, 20, 220, 50], "type": "opamp"}]
        result = associate_structured(components, [text("U1A", [0, 0, 20, 10]),
                                                   text("U1B", [200, 0, 220, 10], "t2")])
        self.assertEqual([row["reference"] for row in result["components"]], ["U1A", "U1B"])
        self.assertEqual([row["physical_reference"] for row in result["components"]], ["U1", "U1"])

    def test_reference_does_not_claim_two_components(self):
        components = [{"id": "a", "bbox": [0, 20, 20, 50], "type": "resistor"},
                      {"id": "b", "bbox": [35, 20, 55, 50], "type": "resistor"}]
        result = associate_structured(components, [text("R1", [0, 0, 20, 10])])
        self.assertEqual(sum(row["reference"] == "R1" for row in result["components"]), 1)

    def test_capacitance_preferred_over_voltage_rating(self):
        result = associate_structured([{"bbox": [0, 25, 20, 55], "type": "capacitor"}],
                                      [text("C1", [25, 20, 45, 30]), text("10uF", [25, 35, 45, 45], "t2"),
                                       text("16V", [25, 50, 45, 60], "t3")])
        self.assertEqual(result["components"][0]["value"], "10uF")
        self.assertTrue(any(t["raw"] == "16V" for t in result["unassigned_texts"]))

    def test_decoupling_value_not_grouped_to_ic_reference(self):
        result = associate_structured([{"bbox": [0, 25, 20, 55], "type": "ic_or_module"}],
                                      [text("U1", [25, 20, 45, 30]), text("100nF", [25, 35, 45, 45], "t2")])
        self.assertEqual(result["groups"][0]["text_ids"], ["t1"])

    def test_single_digit_pin_not_promoted_to_value(self):
        result = associate_structured([{"bbox": [0, 25, 20, 55], "type": "capacitor"}],
                                      [text("C1", [25, 20, 45, 30]), text("3", [25, 35, 45, 45], "t2")])
        self.assertIsNone(result["components"][0]["value"])

    def test_reference_evaluator_cannot_change_predictions(self):
        path = Path(__file__).resolve().parents[1] / "scripts/evaluate_ocr_association.py"
        spec = importlib.util.spec_from_file_location("evaluate_ocr_association", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = associate_structured([{"bbox": [0, 20, 20, 50], "type": "resistor"}],
                                      [text("R1", [0, 0, 20, 10])])
        before = copy.deepcopy(result)
        first = module.evaluate_reference(result["components"], {"components": [{"bbox": [0, 20, 20, 50], "reference": "R1"}]})
        second = module.evaluate_reference(result["components"], {"components": [{"bbox": [0, 20, 20, 50], "reference": "R9"}]})
        self.assertEqual(first["fields"]["reference"]["exact_matches"], 1)
        self.assertEqual(second["fields"]["reference"]["exact_matches"], 0)
        self.assertEqual(result, before)


if __name__ == "__main__":
    unittest.main()
