"""E2b: oracle headroom of the null protocol.

Uses GT only to clear `value` on keys where GT says null but the frozen
prediction filled a non-null string, then rescores. This is an ORACLE bound,
not an implementable system: it shows how many points the protocol conflict
currently costs.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.eval.metric import ScoreConfig, score_case

SPLIT = ROOT / "configs/contest6_group_split_v1.json"
PRED = ROOT / "outputs/advance_until_bottleneck_20261017/lexical_experiment/identity_v1/predictions"
GT = ROOT / "data/official_contest6/train200"
OUT = ROOT / "outputs/unattended_20261018_e2_null_conflict"

VALUE_TYPES = {"r", "c", "resistor", "capacitor", "l", "inductor"}


def dev_cases():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    return [str(e["case"] if isinstance(e, dict) else e).zfill(4) for e in split["partitions"]["dev"]]


def main():
    config = ScoreConfig()
    base_rows, oracle_rows, cleared = [], [], []
    for case in dev_cases():
        stem = next((p.stem for p in GT.glob(f"{case}-*.json")), None)
        pred_path = PRED / f"{case}.json"
        if stem is None or not pred_path.exists():
            continue
        gt = json.loads((GT / f"{stem}.json").read_text(encoding="utf-8"))
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        base = score_case(pred, gt, config)
        fixed = json.loads(json.dumps(pred))
        for key, comp in gt["components"].items():
            if str(comp.get("type", "")).lower() not in VALUE_TYPES:
                continue
            if comp.get("value") is not None:
                continue
            target = fixed["components"].get(key)
            if target and target.get("value") is not None:
                cleared.append({"case": case, "key": key, "old": target["value"]})
                target["value"] = None
        oracle = score_case(fixed, gt, config)
        base_rows.append(base["component_f1"])
        oracle_rows.append(oracle["component_f1"])
    macro_base = sum(base_rows) / len(base_rows)
    macro_oracle = sum(oracle_rows) / len(oracle_rows)
    payload = {"cases": len(base_rows), "cleared_values": len(cleared),
               "macro_f1_baseline": macro_base, "macro_f1_null_oracle": macro_oracle,
               "delta_points": (macro_oracle - macro_base) * 100,
               "examples": cleared[:20],
               "note": "ORACLE: value clearing uses GT, not implementable; measures protocol cost only."}
    (OUT / "oracle_null.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("cases", "cleared_values", "macro_f1_baseline",
                                              "macro_f1_null_oracle", "delta_points")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
