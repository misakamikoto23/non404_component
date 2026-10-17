"""Prepare or explicitly execute a bounded Qwen candidate-validity shadow pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))
from component_lab.candidate_verifier import DEFAULT_ENDPOINT, VerifierConfig, recover_invalid_diagnostics, run_verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Neutral candidates JSONL or JSON list")
    parser.add_argument("--recover-from", type=Path, help="Offline-only independent recovery from saved diagnostics")
    parser.add_argument("--output", type=Path, required=True, help="New shadow output/cache/usage directory")
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--context-ratio", type=float, default=.6)
    parser.add_argument("--context-min-pixels", type=int, default=32)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--preprocess", choices=("native", "padded_v2"), default="native",
                        help="padded_v2 requires batch-size=1; preserves native views and sends padded 256/512 views")
    parser.add_argument("--max-requests", type=int, default=16, help="Cumulative attempts in this output directory")
    parser.add_argument("--token-budget", type=int, default=60000, help="Cumulative conservative accounted token cap")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Reserved response tokens per request")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--key-file", type=Path, help="Read only in memory after explicit API execution switch")
    parser.add_argument("--execute-api", action="store_true", help="Actually send prepared images; absent means local preparation only")
    args = parser.parse_args(argv)
    if args.recover_from:
        if args.execute_api or args.manifest:
            parser.error("--recover-from cannot be combined with --execute-api or --manifest")
        summary = recover_invalid_diagnostics(args.recover_from, args.output)
        print(json.dumps({key: summary[key] for key in ("shadow_only", "api_requests_added", "ledger_modified",
                                                      "recovered_candidates", "recovered_ids")}, ensure_ascii=False, indent=2))
        return 0
    if args.manifest is None:
        parser.error("--manifest is required unless using --recover-from")
    config = VerifierConfig(model=args.model, endpoint=args.endpoint, context_ratio=args.context_ratio,
                            context_min_pixels=args.context_min_pixels, limit=args.limit,
                            batch_size=args.batch_size, max_requests=args.max_requests,
                            token_budget=args.token_budget, max_tokens=args.max_tokens, timeout=args.timeout,
                            preprocess=args.preprocess)
    try:
        summary = run_verification(args.manifest, args.output, config, args.execute_api, args.key_file,
            progress=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
    except (ValueError, OSError, KeyError, TypeError) as error:
        # No exception message is printed: credential/path/provider text stays local.
        print(json.dumps({"status": "error", "error_type": type(error).__name__,
                          "message": "Input/configuration failure; inspect the local manifest and configuration. No secrets were logged."}))
        return 2
    # Print only safe small accounting metadata, never the full messages or raw API response.
    keys = ("mode", "shadow_only", "selected_candidates", "cached_candidates", "new_requests",
            "planned_requests", "estimated_total_tokens", "request_attempts", "accounted_tokens",
            "pending_candidates", "decisions", "stop_reason")
    print(json.dumps({k: summary[k] for k in keys if k in summary}, ensure_ascii=False, indent=2))
    return 0 if summary.get("stop_reason") is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
