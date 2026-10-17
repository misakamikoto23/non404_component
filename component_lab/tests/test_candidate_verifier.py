"""Synthetic image/schema/accounting checks; no network, keys or GPU."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from PIL import Image

from component_lab.candidate_verifier import (
    VerifierConfig, accounted_usage, build_payload, canonical, load_manifest,
    parse_items, prepare_candidate, recover_invalid_diagnostics, request_config, run_verification, sha256_file,
)


LAB = Path(__file__).resolve().parents[1]


def valid_result(ident="c1", **changes):
    row = {"id": ident, "decision": "keep", "reason": "valid_symbol",
           "component_present": True, "box_complete": True,
           "type": "unknown", "duplicate_of": None}
    row.update(changes)
    return row


class CandidateVerifierTests(unittest.TestCase):
    def setUp(self):
        self.root = LAB / "outputs" / ("test_candidate_verifier_" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.image = self.root / "input.png"
        image = Image.new("RGB", (100, 80), "white")
        image.putpixel((25, 25), (0, 0, 0))
        image.save(self.image)
        self.manifest = self.root / "manifest.jsonl"
        self.row = {"id": "c1", "image_path": "input.png", "image_sha256": sha256_file(self.image),
                    "bbox": [20, 20, 40, 30], "score": .1, "source": "detector",
                    "neighbors": [{"id": "n1", "bbox": [18, 18, 42, 32], "score": .9}],
                    "ocr": []}
        self.config = VerifierConfig(limit=2, max_requests=2, token_budget=60000)
        self.write_manifest([self.row])

    def tearDown(self):
        if not self.root.resolve().is_relative_to((LAB/"outputs").resolve()):
            raise AssertionError("Unsafe test cleanup path")
        shutil.rmtree(self.root)

    def write_manifest(self, rows):
        self.manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def prepare(self):
        row = load_manifest(self.manifest)[0]
        return prepare_candidate(row, self.config, self.root/"prepared")

    def test_native_tight_pixels_and_marked_context(self):
        prepared = self.prepare()
        self.assertEqual(prepared["tight_size"], [20, 10])
        self.assertEqual(prepared["context_bbox"], [0, 0, 72, 62])
        with Image.open(prepared["tight_path"]) as tight:
            self.assertEqual(tight.getpixel((5, 5)), (0, 0, 0))
        with Image.open(prepared["context_path"]) as context:
            self.assertEqual(context.getpixel((18, 18)), (230, 30, 30))
            self.assertEqual(context.getpixel((25, 25)), (0, 0, 0))

    def test_padded_v2_preserves_native_views_and_meets_api_dimensions(self):
        row = load_manifest(self.manifest)[0]
        row["bbox"] = [20, 20, 60, 29]
        config = replace(self.config, preprocess="padded_v2", batch_size=1)
        config.validate()
        prepared = prepare_candidate(row, config, self.root/"padded")
        self.assertEqual(prepared["tight_size"], [40, 9])
        self.assertEqual(prepared["api_tight_size"], [256, 256])
        self.assertEqual(prepared["api_context_size"], [512, 512])
        self.assertEqual(request_config(self.config), {"model": "qwen3.8-flash", "endpoint": self.config.endpoint,
            "context_ratio": .6, "context_min_pixels": 32, "max_tokens": 1200})
        self.assertIn("preprocess", request_config(config))
        with Image.open(prepared["api_tight_path"]) as image:
            self.assertEqual(image.size, (256, 256))
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        with self.assertRaises(ValueError):
            replace(config, batch_size=2).validate()

    def test_flat_item_only_wraps_a_single_candidate_without_inventing_ids(self):
        prepared = self.prepare()
        self.assertEqual(parse_items(canonical(valid_result()), [prepared])[0]["id"], "c1")
        second = dict(prepared, candidate=dict(prepared["candidate"], id="c2"))
        with self.assertRaises(ValueError):
            parse_items(canonical(valid_result()), [prepared, second])
        with self.assertRaises(ValueError):
            parse_items(canonical(valid_result("invented")), [prepared])

    def test_partial_flat_item_offline_recovery_does_not_modify_paid_ledger_or_cache(self):
        self.write_manifest([self.row, dict(self.row, id="c2")])
        output = self.root/"partial_recover"
        with patch("component_lab.candidate_verifier.credential", return_value="fakekey"):
            run_verification(self.manifest, output, self.config, True,
                transport=lambda *_: {"choices": [{"finish_reason": "stop", "message": {"content": canonical(valid_result())}}],
                                      "usage": {"total_tokens": 60}})
        old_ledger = (output/"usage_ledger.json").read_bytes()
        summary = recover_invalid_diagnostics(output, self.root/"recovered")
        self.assertEqual(summary["recovered_ids"], ["c1"])
        self.assertEqual(summary["api_requests_added"], 0)
        self.assertEqual((output/"usage_ledger.json").read_bytes(), old_ledger)
        self.assertEqual(list((output/"cache").glob("*.json")), [])
        self.assertEqual(summary["requests_audit"][0]["missing_ids"], ["c2"])

    def test_offline_recovery_refuses_redacted_diagnostics(self):
        self.write_manifest([self.row, dict(self.row, id="c2")])
        output = self.root/"redacted_recover"
        with patch("component_lab.candidate_verifier.credential", return_value="fakekey"):
            run_verification(self.manifest, output, self.config, True,
                transport=lambda *_: {"choices": [{"finish_reason": "stop", "message": {"content": canonical(valid_result(type="sk-hidden"))}}],
                                      "usage": {"total_tokens": 60}})
        summary = recover_invalid_diagnostics(output, self.root/"not_recovered")
        self.assertEqual(summary["recovered_candidates"], 0)
        self.assertEqual(summary["requests_audit"][0]["status"], "diagnostic_was_truncated_or_redacted")

    def test_human_and_ground_truth_fields_never_reach_prompt(self):
        self.row.update({"accepted": True, "category": "LEAK_CATEGORY", "evaluation_proxy": {"GT": "LEAK_GT"},
                         "review_status": "LEAK_HUMAN", "reference_bbox": [1, 2, 3, 4]})
        self.row["neighbors"][0]["category"] = "LEAK_NEIGHBOR"
        self.write_manifest([self.row])
        prepared = self.prepare()
        payload = build_payload([prepared], self.config)
        text = canonical(payload)
        self.assertNotIn("LEAK_", text)
        self.assertNotIn("evaluation_proxy", canonical(prepared))
        self.assertEqual(sum(item["type"] == "image_url" for item in payload["messages"][0]["content"]), 2)
        self.assertNotIn("source", prepared["metadata"])
        self.assertNotIn("score", prepared["metadata"])

    def test_original_sha_and_box_bounds_are_checked(self):
        self.row["image_sha256"] = "0"*64
        self.write_manifest([self.row])
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)
        self.row["image_sha256"] = sha256_file(self.image)
        self.row["bbox"] = [20, 20, 120, 30]
        self.write_manifest([self.row])
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)

    def test_unknown_type_can_keep_but_low_resolution_cannot_reject(self):
        prepared = [self.prepare()]
        self.assertEqual(parse_items(canonical({"items": [valid_result()]}), prepared)[0]["type"], "unknown")
        bad = valid_result(decision="reject", reason="low_resolution", box_complete=None)
        with self.assertRaises(ValueError):
            parse_items(canonical({"items": [bad]}), prepared)
        bad.update({"decision": "uncertain"})
        self.assertEqual(parse_items(canonical({"items": [bad]}), prepared)[0]["decision"], "uncertain")

    def test_duplicate_must_reference_known_neighbor(self):
        prepared = [self.prepare()]
        row = valid_result(decision="reject", reason="duplicate", duplicate_of="n1")
        self.assertEqual(parse_items(canonical({"items": [row]}), prepared)[0]["duplicate_of"], "n1")
        row["duplicate_of"] = "invented"
        with self.assertRaises(ValueError):
            parse_items(canonical({"items": [row]}), prepared)

    def test_missing_or_extra_ids_and_boolean_numbers_fail_schema(self):
        prepared = [self.prepare()]
        for items in ([], [valid_result(), valid_result()], [valid_result("invented")],
                      [valid_result(component_present=1)], [dict(valid_result(), confidence=.99)]):
            with self.assertRaises(ValueError):
                parse_items(canonical({"items": items}), prepared)

    def test_local_prepare_never_loads_credentials_or_calls_transport(self):
        with patch("component_lab.candidate_verifier.credential", side_effect=AssertionError("credential read")):
            summary = run_verification(self.manifest, self.root/"dry", self.config,
                                       transport=lambda *_: self.fail("Network call"))
        self.assertEqual(summary["mode"], "local_prepare")
        self.assertEqual(summary["request_attempts"], 0)

    def test_cache_resume_is_free_and_leaves_original_manifest_untouched(self):
        original = self.manifest.read_bytes()
        output, calls = self.root/"run", []
        def fake(payload, key, config):
            calls.append(payload)
            return {"choices": [{"finish_reason": "stop", "message": {"content": canonical({"items": [valid_result()]})}}],
                    "usage": {"prompt_tokens": 70, "completion_tokens": 30, "total_tokens": 100}}
        with patch("component_lab.candidate_verifier.credential", return_value="FAKE_SECRET_NOT_A_REAL_KEY"):
            first = run_verification(self.manifest, output, self.config, True, transport=fake)
            second = run_verification(self.manifest, output, self.config, True, transport=fake)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["accounted_tokens"], 100)
        self.assertEqual(second["new_requests"], 0)
        self.assertEqual(second["cached_candidates"], 1)
        self.assertEqual(self.manifest.read_bytes(), original)
        for path in output.rglob("*.json"):
            self.assertNotIn("FAKE_SECRET_NOT_A_REAL_KEY", path.read_text(encoding="utf-8"))

    def test_timeout_is_reserved_not_retried_and_stops(self):
        calls = []
        def fake(*_):
            calls.append(True)
            raise TimeoutError("contains secret and body; must not be logged")
        output = self.root/"failed"
        with patch("component_lab.candidate_verifier.credential", return_value="test"):
            first = run_verification(self.manifest, output, self.config, True, transport=fake)
            second = run_verification(self.manifest, output, self.config, True, transport=fake)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["stop_reason"], "request_failed_no_automatic_retry")
        self.assertGreater(first["accounted_tokens"], self.config.max_tokens)
        self.assertEqual(second["accounted_tokens"], first["accounted_tokens"])
        self.assertEqual(second["stop_reason"], "previous_attempts_not_retried")
        self.assertNotIn("contains secret", (output/"usage_ledger.json").read_text())

    def test_raising_local_cap_does_not_bypass_paid_candidate_cache(self):
        output, calls = self.root/"cap_change", []
        def fake(*_):
            calls.append(True)
            return {"choices": [{"finish_reason": "stop", "message": {"content": canonical({"items": [valid_result()]})}}],
                    "usage": {"total_tokens": 100}}
        with patch("component_lab.candidate_verifier.credential", return_value="test"):
            run_verification(self.manifest, output, self.config, True, transport=fake)
        with patch("component_lab.candidate_verifier.credential", side_effect=AssertionError("No new paid work")):
            second = run_verification(self.manifest, output,
                replace(self.config, token_budget=80000, max_requests=4, limit=40), True, transport=fake)
        self.assertEqual(len(calls), 1)
        self.assertEqual(second["cached_candidates"], 1)

    def test_partial_cache_after_crash_cannot_repack_into_a_paid_retry(self):
        output = self.root/"partial"
        with patch("component_lab.candidate_verifier.credential", return_value="test"):
            run_verification(self.manifest, output, self.config, True,
                transport=lambda *_: {"choices": [{"finish_reason": "stop", "message": {"content": canonical({"items": [valid_result()]})}}],
                                      "usage": {"total_tokens": 100}})
        cache_file = next((output/"cache").glob("*.json"))
        cache_file.unlink()
        with patch("component_lab.candidate_verifier.credential", side_effect=AssertionError("Must not retry")):
            summary = run_verification(self.manifest, output, self.config, True,
                                      transport=lambda *_: self.fail("Duplicate paid request"))
        self.assertEqual(summary["new_requests"], 0)
        self.assertEqual(summary["previously_attempted_without_cache"], 1)
        self.assertEqual(summary["stop_reason"], "previous_attempts_not_retried")

    def test_output_token_reservation_prevents_request(self):
        config = replace(self.config, token_budget=1300)
        with patch("component_lab.candidate_verifier.credential", return_value="test"):
            summary = run_verification(self.manifest, self.root/"budget", config, True,
                                       transport=lambda *_: self.fail("Budget exceeded"))
        self.assertEqual(summary["request_attempts"], 0)
        self.assertEqual(summary["stop_reason"], "token_reservation_exceeds_remaining_budget")

    def test_usage_missing_and_invalid_response_still_accounted(self):
        self.assertEqual(accounted_usage({}, 123), (123, "reserved_no_usage", {}))
        with patch("component_lab.candidate_verifier.credential", return_value="test"):
            summary = run_verification(self.manifest, self.root/"invalid", self.config, True,
                transport=lambda *_: {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}],
                                     "usage": {"total_tokens": 91}})
        self.assertEqual(summary["accounted_tokens"], 91)
        self.assertEqual(summary["stop_reason"], "request_failed_no_automatic_retry")
        diagnostic = json.loads(next((self.root/"invalid"/"invalid_responses").glob("*.json")).read_text())
        self.assertEqual(diagnostic["untrusted_model_content_excerpt"], "not json")
        self.assertEqual(diagnostic["parse_error_type"], "JSONDecodeError")
        self.assertFalse(diagnostic["accepted_as_result"])

    def test_diagnostics_redact_secrets_base64_and_limit_content(self):
        secret = "EXACT_FAKE_CREDENTIAL"
        output, events = self.root/"safe_diagnostic", []
        malicious = secret + " sk-example.fake-key data:image/png;base64," + "Q"*300 + "\n" + "?"*6000
        with patch("component_lab.candidate_verifier.credential", return_value=secret):
            summary = run_verification(self.manifest, output, self.config, True,
                transport=lambda *_: {"choices": [{"finish_reason": "stop", "message": {"content": malicious}}],
                                      "usage": {"total_tokens": 43}}, progress=events.append)
        diagnostic = json.loads(next((output/"invalid_responses").glob("*.json")).read_text())
        stored = canonical(diagnostic)
        self.assertNotIn(secret, stored)
        self.assertNotIn("sk-example", stored)
        self.assertNotIn("Q"*100, stored)
        self.assertLessEqual(len(diagnostic["untrusted_model_content_excerpt"]), 4096)
        self.assertTrue(diagnostic["excerpt_truncated"])
        self.assertEqual(events[0]["status"], "invalid_response")
        self.assertNotIn("content", canonical(events))
        self.assertEqual(summary["accounted_tokens"], 43)

    def test_http_error_body_is_never_a_diagnostic(self):
        output = self.root/"http_error"
        with patch("component_lab.candidate_verifier.credential", return_value="test"):
            summary = run_verification(self.manifest, output, self.config, True,
                transport=lambda *_: {"transport_error": "http_error", "http_status": 401,
                                     "error_body": "DO_NOT_PERSIST_PROVIDER_BODY"})
        self.assertFalse((output/"invalid_responses").exists())
        self.assertNotIn("DO_NOT_PERSIST_PROVIDER_BODY", (output/"usage_ledger.json").read_text())
        self.assertEqual(summary["stop_reason"], "request_failed_no_automatic_retry")

    def test_valid_json_with_non_stop_finish_reason_is_not_accepted(self):
        with patch("component_lab.candidate_verifier.credential", return_value="test"):
            summary = run_verification(self.manifest, self.root/"truncated", self.config, True,
                transport=lambda *_: {"choices": [{"finish_reason": "length",
                    "message": {"content": canonical({"items": [valid_result()]})}}],
                    "usage": {"total_tokens": 53}})
        self.assertEqual(summary["accounted_tokens"], 53)
        self.assertEqual(summary["stop_reason"], "request_failed_no_automatic_retry")
        self.assertEqual(summary["decisions"]["keep"], 0)

    def test_changed_image_pixels_invalidate_sha_before_cache(self):
        output = self.root/"changed"
        run_verification(self.manifest, output, self.config)
        Image.new("RGB", (100, 80), "black").save(self.image)
        with self.assertRaises(ValueError):
            run_verification(self.manifest, output, self.config)
        self.assertFalse((output/"run.lock").exists())


if __name__ == "__main__":
    unittest.main()
