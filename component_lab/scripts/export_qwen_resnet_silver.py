"""Export conservative, local-only ResNet candidates from Qwen crop predictions.

No network, model inference, GPU use or edits to reviewed data occur here.
Accepted detection boxes are not human-verified classification labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from PIL import Image

MIN_CONFIDENCE = .9
MIN_PROXY_IOU = .5
LABEL_ORIGIN = "teacher_agrees_original_proxy_not_human_verified"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def reject_non_json_number(value):
    raise ValueError(f"Nonfinite numeric literal is not valid JSON: {value}")


def json_lines(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line, parse_constant=reject_non_json_number)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object: {path}:{line_number}")
                yield line_number, row


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_child(root, relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("Crop image paths must be relative to the crop root")
    path = (root / relative).resolve()
    path.relative_to(root)
    if not path.is_file():
        raise ValueError(f"Missing crop image: {path}")
    return path


def validate_prediction(prediction, allowed):
    """Validation errors become quarantine reasons rather than trusted labels."""
    reasons = []
    if prediction.get("category") not in allowed:
        reasons.append("invalid_prediction_category")
    confidence = prediction.get("confidence")
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        reasons.append("invalid_prediction_confidence")
    if not isinstance(prediction.get("ambiguous"), bool):
        reasons.append("invalid_prediction_ambiguity")
    return reasons


def export(crops, ontology_path, prediction_paths, output):
    crops, ontology_path, output = Path(crops).resolve(), Path(ontology_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output == crops or crops in output.parents or output in crops.parents:
        raise ValueError("Output must be separate from the existing crop dataset")
    source_summary = read_json(crops / "summary.json")
    if source_summary.get("source_split") != "train" or source_summary.get("val_test_images_used") is not False:
        raise ValueError("Crop summary must explicitly confirm train-only sources and no val/test images")
    ontology = read_json(ontology_path)
    allowed = ontology.get("allowed_labels", [])
    if (not allowed or len(allowed) != len(set(allowed))
            or any(not isinstance(c, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", c) for c in allowed)):
        raise ValueError("Ontology must provide unique, safe semantic family names")
    if "unknown" not in allowed:
        raise ValueError("Ontology must include the unknown abstention label")
    mapping = ontology.get("raw_to_family_for_unambiguous", {})
    ambiguous_raw = set(ontology.get("ambiguous_raw_classes", []))
    if not isinstance(mapping, dict) or any(f not in allowed or f == "unknown" for f in mapping.values()):
        raise ValueError("Invalid conservative raw-to-family mapping")
    if ambiguous_raw.intersection(mapping):
        raise ValueError("Ambiguous raw classes must never have automatic conservative mappings")
    crop_rows = [r for _, r in json_lines(crops / "manifest.jsonl")]
    crop_ids = [r["crop_id"] for r in crop_rows]
    if not crop_rows or len(crop_ids) != len(set(crop_ids)):
        raise ValueError("Crop manifest is empty or contains duplicate IDs")
    if any(not isinstance(c, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in crop_ids):
        raise ValueError("Crop IDs must be safe neutral filename identifiers")
    by_id = {r["crop_id"]: r for r in crop_rows}
    predictions, orphans = defaultdict(list), []
    prediction_sources = []
    for prediction_path in prediction_paths:
        prediction_path = Path(prediction_path).resolve()
        prediction_sources.append({"path": str(prediction_path), "sha256": sha256_file(prediction_path)})
        for line_number, prediction in json_lines(prediction_path):
            # Deliberately ignore model-added explanation and any unneeded fields.
            row = {k: prediction.get(k) for k in ("id", "category", "confidence", "ambiguous")}
            row["source_file"] = str(prediction_path)
            row["source_line"] = line_number
            if not isinstance(row["id"], str) or row["id"] not in by_id:
                orphans.append({"crop_id": row["id"], "reasons": ["prediction_id_not_in_crop_manifest"],
                                "prediction": row, "category_confirmed": False})
            else:
                predictions[row["id"]].append(row)
    main, confidence_only, quarantine = [], [], list(orphans)
    reasons_count = Counter({"prediction_id_not_in_crop_manifest": len(orphans)}) if orphans else Counter()
    for record in crop_rows:
        crop_id = record["crop_id"]
        values = predictions.get(crop_id, [])
        reason = []
        merged = None
        if not values:
            reason.append("not_classified")
        else:
            for value in values:
                reason.extend(validate_prediction(value, allowed))
            if not reason:
                categories = {value["category"] for value in values}
                if len(categories) != 1:
                    reason.append("conflicting_predictions")
                merged = {"category": next(iter(categories)) if len(categories) == 1 else None,
                          "confidence": min(value["confidence"] for value in values),
                          "ambiguous": any(value["ambiguous"] for value in values),
                          "prediction_count": len(values)}
                if "unknown" in categories:
                    reason.append("teacher_unknown")
                if merged["ambiguous"]:
                    reason.append("teacher_ambiguous")
                if merged["confidence"] < MIN_CONFIDENCE:
                    reason.append("teacher_confidence_below_0_9")
        proxy = record.get("evaluation_proxy", {})
        raw = proxy.get("matched_original_category")
        iou = proxy.get("matched_iou")
        if raw is None:
            reason.append("missing_original_proxy")
        if (isinstance(iou, bool) or not isinstance(iou, (int, float))
                or not math.isfinite(iou) or iou < MIN_PROXY_IOU):
            reason.append("original_proxy_iou_below_0_5_or_invalid")
        if proxy.get("ambiguous") is not False:
            reason.append("original_proxy_ambiguous_or_unspecified")
        expected = mapping.get(raw)
        if expected is not None and merged and merged["category"] is not None and merged["category"] != expected:
            reason.append("teacher_disagrees_conservative_original_proxy")
        if expected is None and raw not in ambiguous_raw:
            reason.append("original_raw_class_has_no_conservative_mapping")
        row = {"crop_id": crop_id, "image_id": record["image_id"],
               "source_box_id": record.get("source_box_id"),
               "source_context_image_path": record["context_image_path"],
               "source_image_sha256": record.get("source_image_sha256"),
               "source_split": "train", "category_confirmed": False,
               "human_category_verified": False, "prediction": merged,
               "original_proxy": {"raw_category": raw, "conservative_family": expected,
                                   "component_id": proxy.get("matched_original_component_id"),
                                   "iou": iou, "ambiguous": proxy.get("ambiguous")},
               "teacher_evidence": values}
        if reason:
            row["reasons"] = sorted(set(reason))
            quarantine.append(row)
            reasons_count.update(row["reasons"])
        elif raw in ambiguous_raw:
            row.update(label_origin="teacher_confidence_only_not_human_verified",
                       category=merged["category"],
                       exclusion_reason="ambiguous_original_raw_semantics_not_a_teacher_error",
                       eligible_for_main_silver_train=False)
            confidence_only.append(row)
        else:
            row.update(label_origin=LABEL_ORIGIN, category=merged["category"],
                       eligible_for_main_silver_train=True)
            main.append(row)
    families = sorted(set(allowed) - {"unknown"})
    per_class = {}
    for family in families:
        selected = [r for r in main if r["category"] == family]
        confidence_selected = [r for r in confidence_only if r["category"] == family]
        per_class[family] = {"silver_train_crops": len(selected),
                             "silver_train_source_images": len({r["image_id"] for r in selected}),
                             "source_image_ids": sorted({r["image_id"] for r in selected}),
                             "confidence_only_crops_excluded": len(confidence_selected)}
    active = sorted(c for c in families if per_class[c]["silver_train_crops"])
    active_indices = {c: i for i, c in enumerate(active)}
    # Validate every file selected for copying before creating output artifacts.
    for row in main:
        path = checked_child(crops, row["source_context_image_path"])
        with Image.open(path) as im:
            im.verify()
        if path.suffix.lower() != ".png":
            raise ValueError("Expected original-resolution PNG context crops; conversion is intentionally disabled")
    output.mkdir(parents=True)
    (output / "train").mkdir()
    for row in main:
        source = checked_child(crops, row["source_context_image_path"])
        destination = output / "train" / row["category"] / f"{row['crop_id']}.png"
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        if sha256_file(destination) != source_hash:
            raise ValueError("Crop copy hash mismatch")
        with Image.open(source) as im:
            row["size_wh"] = list(im.size)
        row.update(file=destination.relative_to(output).as_posix(),
                   crop_sha256=source_hash, class_index=active_indices[row["category"]])
    write_jsonl(output / "manifest.jsonl", main)
    write_jsonl(output / "confidence_only.jsonl", confidence_only)
    write_jsonl(output / "quarantine.jsonl", quarantine)
    write_json(output / "classes.json", {
        "schema_version": 1, "allowed_family_names": families,
        "ontology_class_to_idx": {c: i for i, c in enumerate(families)},
        "training_folder_class_to_idx": active_indices,
        "unknown_is_training_class": False,
        "class_index_note": "ImageFolder alphabetic active-folder indices; empty classes are reported but folders are not created."})
    summary = {
        "schema_version": 1, "crops_root": str(crops), "output": str(output),
        "crop_manifest_sha256": sha256_file(crops / "manifest.jsonl"),
        "ontology": str(ontology_path), "ontology_sha256": sha256_file(ontology_path),
        "prediction_sources": prediction_sources,
        "source_images": len({r["image_id"] for r in crop_rows}),
        "source_crops": len(crop_rows), "source_split": "train",
        "existing_review_or_resnet_data_modified": False,
        "predicted_crop_ids": len(predictions),
        "duplicate_prediction_ids": sum(len(values) > 1 for values in predictions.values()),
        "duplicate_policy": "Require category agreement; take minimum confidence and any ambiguity across all predictions.",
        "orphan_prediction_rows": len(orphans), "silver_train_crops": len(main),
        "confidence_only_crops_excluded_from_train": len(confidence_only),
        "quarantine_crops": len(quarantine) - len(orphans),
        "quarantine_reason_counts": dict(reasons_count),
        "silver_label_origin": LABEL_ORIGIN,
        "human_verified_category_count": 0,
        "confidence_threshold": MIN_CONFIDENCE, "proxy_iou_threshold": MIN_PROXY_IOU,
        "self_reported_confidence_is_calibrated_probability": False,
        "per_class": per_class,
        "zero_source_classes": [c for c in families if per_class[c]["silver_train_source_images"] == 0],
        "single_source_classes": [c for c in families if per_class[c]["silver_train_source_images"] == 1],
        "independent_validation_provided": False,
        "ready_to_train": False,
        "blocking_reason": "Independent source-image validation set with compatible semantic labels is required. All current sources already belong to train; splitting their crops would leak source-image information.",
        "network_calls": 0, "gpu_used": False,
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# ResNet18 候选伪标签集\n\n"
        f"从 {summary['source_images']} 张既有训练原图的 {len(crop_rows)} 个人工保留框中，"
        f"筛出 {len(main)} 个主候选、{len(confidence_only)} 个 confidence-only 候选，"
        f"隔离 {len(quarantine) - len(orphans)} 个裁剪。\n\n"
        "人工审核确认的是元件框，不是分类。本目录所有分类标签 category_confirmed=false，"
        "只能称为 Qwen 与原 PCBnet 保守代理标签一致的候选伪标签，不能称人工真值或分类准确率。\n\n"
        "主 train 只接收 Qwen 非 unknown、非 ambiguous、自报 confidence≥0.9，且与保守 raw→family 映射一致、"
        "一对一原标签几何 IoU≥0.5、代理不歧义的样本。原8类歧义 raw 不因无法映射而判为模型错误；"
        "其高置信预测仅保存在 confidence_only.jsonl，未复制到 train。其余情况见 quarantine.jsonl。\n\n"
        "图片是原分辨率 context 裁剪的字节一致副本，没有放大、重新渲染或覆盖原图。"
        "train 下只建立非空类别；classes.json 分开记录完整 ontology 与当前 ImageFolder 活跃类别索引。\n\n"
        "ready_to_train=false：此处没有建立 val/test，也没有训练。全部来源图原本属于 train，"
        "不能随机拆分同一原图的裁剪当验证集。下一步需独立来源原图的验证集、兼容的语义类别标签，"
        "并检查 summary.json 中零来源及单来源类别的覆盖。模型自报置信度不等于校准后的正确概率。\n",
        encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("output", "silver_train_crops", "confidence_only_crops_excluded_from_train",
                                             "quarantine_crops", "ready_to_train")}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True, help="Crop root containing manifest.jsonl and summary.json")
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New independent output directory; never overwritten")
    args = parser.parse_args()
    export(args.crops, args.ontology, args.predictions, args.output)
