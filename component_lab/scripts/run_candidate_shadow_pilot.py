"""Finish unattempted shadow candidates; quarantine bad model JSON without repaying it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB/'src'))
from component_lab.candidate_verifier import VerifierConfig, run_verification, atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--key-file', type=Path)
    p.add_argument('--execute-api', action='store_true')
    p.add_argument('--limit', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-requests', type=int, default=16)
    p.add_argument('--token-budget', type=int, default=60000)
    p.add_argument('--max-tokens', type=int, default=1200)
    p.add_argument('--preprocess', choices=('native','padded_v2'), default='native')
    args = p.parse_args()
    config = VerifierConfig(limit=args.limit, batch_size=args.batch_size,
                            max_requests=args.max_requests, token_budget=args.token_budget,
                            max_tokens=args.max_tokens, preprocess=args.preprocess)
    history, previous_attempts = [], -1
    while True:
        result = run_verification(args.manifest, args.output, config, args.execute_api, args.key_file)
        current = {key:result.get(key) for key in ('mode','selected_candidates','cached_candidates',
                   'new_requests','request_attempts','accounted_tokens','pending_candidates','stop_reason','decisions')}
        history.append(current)
        print(json.dumps(current, ensure_ascii=False), flush=True)
        if not args.execute_api or result.get('stop_reason') != 'request_failed_no_automatic_retry':
            break
        ledger = json.loads((args.output/'usage_ledger.json').read_text(encoding='utf-8'))
        attempts = len(ledger['requests'])
        if attempts <= previous_attempts or attempts >= args.max_requests:
            break
        previous_attempts = attempts
        latest = list(ledger['requests'].values())[-1]
        if latest.get('status') != 'invalid_response':
            # HTTP/auth/timeouts are service failures; do not spend on more jobs.
            break
        # A model-schema failure is quarantined. The verifier tracks attempted
        # candidate keys, so this only submits other, never-attempted candidates.
    if args.execute_api:
        ledger_path = args.output/'usage_ledger.json'
        ledger = json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {'requests':{}}
        attempted_ids = {i for request in ledger['requests'].values() for i in request['candidate_ids']}
        returned = {r['id'] for r in result.get('results',[])}
        status = ('all_model_results_valid' if not result.get('pending_candidates') else
                  'completed_with_quarantined_candidates' if len(attempted_ids)==result['selected_candidates'] else
                  'stopped_with_unattempted_candidates')
    else:
        attempted_ids, returned, status = set(), set(), 'prepared_only'
    controller = {'status':status,'shadow_only':True,'history':history,
                  'attempted_candidates':len(attempted_ids),'valid_model_results':len(returned),
                  'quarantined_candidates':sorted(attempted_ids-returned),
                  'quarantine_action':'retain_baseline_and_flag',
                  'failed_paid_candidate_retries':0}
    atomic_json(args.output/'controller_summary.json',controller)
    print(json.dumps({k:v for k,v in controller.items() if k!='history'},ensure_ascii=False,indent=2))
    return 0 if status in {'prepared_only','all_model_results_valid','completed_with_quarantined_candidates'} else 3


if __name__=='__main__':
    raise SystemExit(main())
