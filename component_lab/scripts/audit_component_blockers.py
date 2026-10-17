"""Audit frozen DEV component failures without changing inference or labels.

GT fields are read from existing evaluation diagnostics, only for attribution.
Reference-stage replay uses predictions/OCR only, and must match the frozen run.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from component_lab.text_matching import (NO_REFERENCE_TYPES, PREFIX_TYPES, _center,
                                        _distance, _height, _role, _type,
                                        associate_structured)
from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ref_signature(association):
    return {c["id"]: (c.get("association_evidence", {}).get("reference", {}).get("text_id"),
                       c.get("association_evidence", {}).get("reference", {}).get("raw"))
            for c in association["components"]}


def reference_edges(text, components, settings):
    role = _role(text, "reference", allow_context=True)
    if role is None:
        return []
    scale = _height(text)
    tx, ty = _center(text["bbox"])
    edges = []
    for component in components:
        distance = _distance(text["bbox"], component["bbox"]) / scale
        if distance > settings["max_distance_text_heights"]:
            continue
        cx, cy = _center(component["bbox"])
        center = ((tx - cx) ** 2 + (ty - cy) ** 2) ** .5 / scale
        kind = _type(component)
        expected = PREFIX_TYPES.get(role.get("prefix"), set())
        compatibility = -.30 if kind in expected else (.30 if kind != "component" else 0.)
        if kind in NO_REFERENCE_TYPES:
            compatibility += 1.55
        cost = .23 * distance + .015 * center + .35 * (1 - role["heuristic_score"]) + compatibility
        if text.get("region_id") is not None and component.get("region_id") is not None:
            if text["region_id"] != component["region_id"]:
                cost += .8
        if settings["clamp_costs"]:
            cost = max(0., cost)
        edges.append({"component_id": component["id"], "cost": cost,
                      "distance_text_heights": distance})
    return sorted(edges, key=lambda x: (x["cost"], x["component_id"]))


def audit_key(key, association):
    components, settings = association["components"], association["diagnostics"]
    raw_refs = Counter(c.get("association_evidence", {}).get("reference", {}).get("raw")
                       for c in components)
    assignments = {a["text_id"]: a for a in association["assignments"]}
    tokens = []
    for text in association["texts"]:
        if text["raw"] != key:
            continue
        role = _role(text, "reference", allow_context=True)
        edges = reference_edges(text, components, settings)
        assignment = assignments.get(text["id"])
        if assignment and assignment["role"] == "reference":
            stage = "duplicate_reference_export" if raw_refs[key] > 1 else "assigned_reference_missing_export_unknown"
        elif role is None:
            stage = "reference_role_filtered"
        elif not edges:
            stage = "no_box_within_actual_distance_gate"
        elif len(edges) >= 2 and edges[1]["cost"] - edges[0]["cost"] < settings["ambiguity_gap"]:
            stage = "owner_ambiguity_abstention"
        elif edges[0]["cost"] >= settings["unmatched_cost"]:
            stage = "all_reference_costs_above_unmatched"
        else:
            stage = "global_reference_slot_competition"
        tokens.append({"text_id": text["id"], "raw": text["raw"], "bbox": text["bbox"],
                       "primary_role": text["primary_role"], "reference_role": role,
                       "stage": stage, "later_or_reference_assignment": assignment,
                       "top_reference_edges": edges[:5]})
    assert tokens, key
    # A key may have several exact occurrences. Classify by the furthest
    # viable path, rather than counting every occurrence as another GT item.
    priority = ["duplicate_reference_export", "assigned_reference_missing_export_unknown",
                "global_reference_slot_competition", "owner_ambiguity_abstention",
                "all_reference_costs_above_unmatched", "no_box_within_actual_distance_gate",
                "reference_role_filtered"]
    present = {t["stage"] for t in tokens}
    return {"key": key, "stage": next(stage for stage in priority if stage in present),
            "exact_occurrences": len(tokens), "tokens": tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "outputs/advance_until_bottleneck_20261017/lexical_experiment")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/component_blocker_audit_20261017")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory; frozen audits are not overwritten")
    report_path = args.experiment / "report.json"
    source = read(report_path)
    args.output.mkdir(parents=True)
    failures, drops, case_only, replay_checks = [], [], [], []
    input_hashes = {str(report_path): hashlib.sha256(report_path.read_bytes()).hexdigest()}
    for diag_id, diagnostic in source["diagnostics"].items():
        if not diag_id.endswith("/identity_v1"):
            continue
        case = diag_id.split("/")[0]
        path = args.experiment / "identity_v1/associations" / f"{case}.json"
        association = read(path)
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        settings = association["diagnostics"]
        replay = associate_structured(association["components"], association["texts"],
            max_distance_heights=settings["max_distance_text_heights"],
            ambiguity_gap=settings["ambiguity_gap"], unmatched_cost=settings["unmatched_cost"],
            clamp_costs=settings["clamp_costs"])
        parity = ref_signature(replay) == ref_signature(association)
        assert parity, f"Reference replay mismatch: {case}; do not attribute drop stages"
        replay_checks.append({"case": case, "reference_replay_parity": parity})
        for component in association["components"]:
            evidence = component.get("association_evidence", {}).get("reference")
            if evidence:
                text = next(t for t in association["texts"] if t["id"] == evidence["text_id"])
                edge = next(e for e in reference_edges(text, association["components"], settings)
                            if e["component_id"] == component["id"])
                assert abs(edge["cost"] - evidence["cost"]) < 1e-8
        for row in diagnostic["named_components"]:
            if row["state"] == "key_hit_other_field_error":
                failed = [field for field in ("type", "bbox", "text") if not row["checks"][field]]
                failures.append({"case": case, "key": row["key"], "failed": failed,
                                 "checks": row["checks"], "gt": row["gt"], "prediction": row["prediction"]})
            elif row["state"] == "exact_key_available_but_not_exported":
                drops.append({"case": case, **audit_key(row["key"], association)})
            elif row["state"] == "no_exact_key_in_ocr":
                matches = [t for t in association["texts"] if t["raw"].casefold() == row["key"].casefold()]
                if matches:
                    with Image.open(ROOT / "outputs/component_identity_round_001_20261017/inputs" / case / "source.png") as image:
                        gx1, gy1, gx2, gy2 = row["gt"]["bbox"]
                        gt_top_left = [gx1, image.height-gy2, gx2, image.height-gy1]
                        crop_paths = []
                        for index, text in enumerate(matches):
                            x1, y1, x2, y2 = text["bbox"]
                            crop = image.crop((max(0, int(x1)-45), max(0, int(y1)-35),
                                               min(image.width, int(x2)+45), min(image.height, int(y2)+35)))
                            name = f"{case}_{row['key']}_{index}.png"
                            crop.resize((crop.width*3, crop.height*3)).save(args.output / name)
                            crop_paths.append(name)
                        x1, y1, x2, y2 = gt_top_left
                        gt_crop = image.crop((max(0, int(x1)-65), max(0, int(y1)-65),
                                              min(image.width, int(x2)+65), min(image.height, int(y2)+65)))
                        gt_name = f"{case}_{row['key']}_GT_context.png"
                        gt_crop.resize((gt_crop.width*3, gt_crop.height*3)).save(args.output / gt_name)
                    case_only.append({"case": case, "gt_key": row["key"],
                                      "gt_bbox_top_left": gt_top_left, "gt_context_crop": gt_name,
                                      "ocr": [{"id": t["id"], "raw": t["raw"],
                                               "bbox": t["bbox"],
                                               "distance_to_gt_body_pixels": _distance(t["bbox"], gt_top_left),
                                               "reference_candidates": [r for r in t["role_candidates"] if r["role"] == "reference"]}
                                              for t in matches], "crops": crop_paths})
    assert len(drops) == 36 and len(failures) == 112 and len(case_only) == 4
    combinations = Counter("+".join(row["failed"]) for row in failures)
    stages = Counter(row["stage"] for row in drops)
    output = {"kind": "frozen_DEV_component_only_read_only_failure_attribution",
              "source_report_sha256": input_hashes[str(report_path)],
              "drop_stage_counts": dict(stages), "failure_combinations": dict(combinations),
              "unexported": drops, "key_hit_failures": failures, "case_only": case_only,
              "reference_replay": replay_checks, "input_sha256": input_hashes,
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "paid_api_calls": 0, "model_forward_calls": 0, "training_runs": 0,
              "gt_or_predictions_modified": False, "sealed_test_used": False,
              "limits": ["Exact key hit does not prove geometric owner correctness.",
                         "Drop categories describe current algorithm, not necessarily the root visual cause.",
                         "Multiple exact occurrences classified by furthest viable path; token details retained.",
                         "Case-fold matching is diagnostic only; no final output normalization."]}
    (args.output / "audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Component 阻塞审计 — 2026-10-17", "", "固定 dev20 冻结缓存，只读归因；没有模型前向/API/训练，未使用封存测试。", "",
             "## 112个已命中key失败项", "", "|失败检查组合|数量|", "|---|---:|"]
    lines += [f"|{key}|{value}|" for key, value in sorted(combinations.items())]
    lines += ["", "## 36个有原文未导出项", "", "|当前算法阻塞阶段|GT项数|", "|---|---:|"]
    lines += [f"|{key}|{value}|" for key, value in sorted(stages.items())]
    lines += ["", "距离采用代码实际的bbox间距/文字窄边≤8，不采用任意100px中心半径。20图reference重放与冻结结果全部一致；重建候选代价与已分配日志一致。候选多次出现时按最深可行阶段归类，逐token证据保存在audit.json。", "",
              "## 大小写与解释限制", "", "parser生成大写语义候选，但raw保留，contest导出使用raw_field；发现.upper()本身不代表输出被改写。4个仅大小写不同样本的原图局部如下；图片视觉结论另作复核，不自动由GT决定。"]
    for row in case_only:
        lines.append(f"\n{row['case']}: GT `{row['gt_key']}`，OCR `{row['ocr'][0]['raw']}`。")
        lines.append(f"OCR框到GT主体框的最短间距：{row['ocr'][0]['distance_to_gt_body_pixels']:.2f}px。这不是文字归属的真值。")
        lines += [f"\n![原图文字局部]({name})" for name in row["crops"]]
        lines.append(f"\n![GT位置周边原图]({row['gt_context_crop']})")
    lines += ["", "key命中只意味着字符串相同，不保证绑对物理元件；112项仍可能包含owner错误。drop stage只解释当前算法路径，不把公开编号矛盾归咎于OCR。", "",
              "复现：`python scripts/audit_component_blockers.py --output outputs/component_blocker_audit_fresh`。输出目录必须不存在。所有输入哈希见audit.json。"]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"failure_combinations": dict(combinations), "drop_stage_counts": dict(stages),
                      "reference_replay_cases": len(replay_checks), "case_only": len(case_only)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
