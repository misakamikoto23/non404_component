"""Separate localization and class-label agreement using validation caches.

No model, torch, GPU or API is used. Location-only and class-aware greedy
matching are two diagnostics with potentially different assignments. Their
count difference is not an independent probability decomposition of errors.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def box_iou(a: list[float], b: list[float]) -> float:
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.


def location_matches(ground_truth: list[dict], predictions: list[dict], confidence: float = .25,
                     iou_threshold: float = .5) -> dict:
    """Highest-confidence predictions claim the highest-IoU unclaimed GT, irrespective of class."""
    if not 0 <= confidence <= 1 or not 0 < iou_threshold <= 1:
        raise ValueError("Confidence must be in [0,1] and IoU in (0,1].")
    eligible = sorted((index for index, row in enumerate(predictions) if row["score"] >= confidence),
                      key=lambda index: (-predictions[index]["score"], index))
    unclaimed = set(range(len(ground_truth)))
    matches, fp = [], []
    for prediction_index in eligible:
        prediction = predictions[prediction_index]
        candidates = [(box_iou(prediction["bbox"], ground_truth[index]["bbox"]), index)
                      for index in sorted(unclaimed)]
        overlap, gt_index = max(candidates, key=lambda pair: pair[0]) if candidates else (0., None)
        if gt_index is not None and overlap >= iou_threshold:
            unclaimed.remove(gt_index)
            matches.append({"prediction_index": prediction_index, "ground_truth_index": gt_index,
                            "iou": overlap, "class_correct": prediction["class_id"] == ground_truth[gt_index]["class_id"]})
        else:
            fp.append(prediction_index)
    return {"matches": matches, "false_positive_indices": fp, "false_negative_indices": sorted(unclaimed),
            "evaluated_predictions": eligible}


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_report(records: list[dict], names: list[str]) -> dict:
    confusion = [[0] * len(names) for _ in names]
    per_image, wrong_examples = [], []
    gt_per_class, located_per_class, correct_per_class = [0] * len(names), [0] * len(names), [0] * len(names)
    gt_total = prediction_total = aware_tp = location_tp = correct = location_fp = location_fn = 0
    aware_fp = aware_fn = 0
    for record in records:
        gt, preds = record["ground_truth"], record["predictions"]
        geometry, aware = record["location_matching"], record["class_aware_matching"]
        for target in gt:
            gt_per_class[target["class_id"]] += 1
        gt_total += len(gt)
        prediction_total += len(geometry["evaluated_predictions"])
        aware_tp += len(aware["matches"])
        aware_fp += len(aware["false_positive_indices"])
        aware_fn += len(aware["false_negative_indices"])
        location_tp += len(geometry["matches"])
        location_fp += len(geometry["false_positive_indices"])
        location_fn += len(geometry["false_negative_indices"])
        image_correct = sum(match["class_correct"] for match in geometry["matches"])
        correct += image_correct
        for match in geometry["matches"]:
            target, prediction = gt[match["ground_truth_index"]], preds[match["prediction_index"]]
            truth_class, predicted_class = target["class_id"], prediction["class_id"]
            confusion[truth_class][predicted_class] += 1
            located_per_class[truth_class] += 1
            correct_per_class[truth_class] += int(match["class_correct"])
            if not match["class_correct"]:
                wrong_examples.append({"image": record["image"], "iou": match["iou"], "score": prediction["score"],
                                       "ground_truth_class_id": truth_class, "ground_truth_type": names[truth_class],
                                       "predicted_class_id": predicted_class, "predicted_type": names[predicted_class],
                                       "ground_truth_bbox": target["bbox"], "predicted_bbox": prediction["bbox"]})
        per_image.append({"image": record["image"], "ground_truth": len(gt),
                          "evaluated_predictions": len(geometry["evaluated_predictions"]),
                          "class_aware_tp": len(aware["matches"]), "location_only_tp": len(geometry["matches"]),
                          "location_only_fp": len(geometry["false_positive_indices"]),
                          "location_only_fn": len(geometry["false_negative_indices"]),
                          "geometry_matched_class_correct": image_correct,
                          "geometry_matched_class_wrong": len(geometry["matches"]) - image_correct})
    pairs = [{"ground_truth_class_id": actual, "ground_truth_type": names[actual],
              "predicted_class_id": predicted, "predicted_type": names[predicted], "count": count,
              "class_correct": actual == predicted}
             for actual, row in enumerate(confusion) for predicted, count in enumerate(row) if count]
    pairs.sort(key=lambda row: (-row["count"], row["ground_truth_class_id"], row["predicted_class_id"]))
    per_class = [{"class_id": index, "name": name, "ground_truth": gt_per_class[index],
                  "location_only_tp": located_per_class[index], "location_recall": ratio(located_per_class[index], gt_per_class[index]),
                  "geometry_matched_class_correct": correct_per_class[index],
                  "geometry_matched_class_wrong": located_per_class[index] - correct_per_class[index],
                  "class_accuracy_given_geometry_match": ratio(correct_per_class[index], located_per_class[index])}
                 for index, name in enumerate(names)]
    return {"metric": "fixed_threshold_validation_cache_diagnostics_not_mAP", "names": names,
            "overall": {"ground_truth": gt_total, "evaluated_predictions": prediction_total,
                        "class_aware_tp": aware_tp, "class_aware_fp": aware_fp, "class_aware_fn": aware_fn,
                        "class_aware_recall": ratio(aware_tp, gt_total),
                        "location_only_tp": location_tp, "location_only_fp": location_fp, "location_only_fn": location_fn,
                        "location_only_precision": ratio(location_tp, prediction_total),
                        "location_only_recall": ratio(location_tp, gt_total),
                        "geometry_matched_class_correct": correct, "geometry_matched_class_wrong": location_tp - correct,
                        "class_accuracy_given_geometry_match": ratio(correct, location_tp),
                        "location_tp_minus_class_aware_tp": location_tp - aware_tp},
            "by_class": per_class, "confusion_matrix_gt_rows_prediction_columns": confusion,
            "confusion_pairs": pairs, "wrong_class_examples": sorted(wrong_examples, key=lambda row: -row["score"]),
            "per_image": per_image,
            "limitations": [
                "Both matchers greedily order predictions by confidence, then select the highest-IoU unclaimed GT. Neither is maximum bipartite matching.",
                "Ignoring class can change competition for ground-truth boxes and yield different assignments. Correct labels among geometry matches and class-aware TP need not be equal.",
                "Location-only TP minus class-aware TP is a diagnostic count difference, not the unique number of classification errors or an independent probability factorization.",
                "Unmatched GT can reflect an absent prediction, insufficient IoU, duplicate/assignment competition, confidence filtering, or annotation issues. This does not independently identify their causes.",
                "Agreement is measured against existing original labels. Symbol ambiguity, missing/incorrect labels and inconsistent taxonomy can contribute; no automated diagnosis proves label correctness.",
                "No model is loaded. Only validation labels and cached predictions are used; no test-set tuning or API calls."]}


def save_report(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, rows in (("per_class.csv", report["by_class"]), ("confusion_pairs.csv", report["confusion_pairs"]),
                           ("per_image.csv", report["per_image"])):
        if rows:
            with (output / filename).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    overall = report["overall"]
    percent = lambda value: "n/a" if value is None else f"{value:.1%}"
    lines = ["# Localization and classification cache diagnosis", "", "Validation only; fixed-threshold counts, not mAP.", "",
             f"Confidence {report['confidence']}; matching IoU {report['iou_threshold']}; GT {overall['ground_truth']}; predictions above threshold {overall['evaluated_predictions']}.", "",
             "| Diagnostic | Count / value |", "|---|---:|",
             f"| Class-aware TP | {overall['class_aware_tp']} |",
             f"| Class-aware recall | {percent(overall['class_aware_recall'])} |",
             f"| Location-only TP | {overall['location_only_tp']} |",
             f"| Location-only recall | {percent(overall['location_only_recall'])} |",
             f"| Location-only unmatched GT | {overall['location_only_fn']} |",
             f"| Geometry-matched predictions agreeing with GT class | {overall['geometry_matched_class_correct']} |",
             f"| Geometry-matched predictions disagreeing with GT class | {overall['geometry_matched_class_wrong']} |",
             f"| Class accuracy within geometry matches | {percent(overall['class_accuracy_given_geometry_match'])} |", "",
             "The two matchers can assign predictions differently. Do not treat location-only TP minus class-aware TP as the unique cause or count of classification errors.",
             "These are agreements with original annotations; an incorrect or ambiguous annotation can produce the same observed error as a model mistake.", "",
             "## Largest off-diagonal class confusions", "", "| GT label | Predicted label | Count |", "|---|---|---:|"]
    wrong_pairs = [row for row in report["confusion_pairs"] if not row["class_correct"]]
    lines.extend(f"| {row['ground_truth_type']} | {row['predicted_type']} | {row['count']} |" for row in wrong_pairs[:20])
    if not wrong_pairs:
        lines.append("| (none) | | 0 |")
    lines += ["", "Full confusion matrix uses GT rows and predicted-class columns; class order is recorded in metrics.json.",
              "Per-class statistics, matched wrong-class box examples, and per-image counts are also preserved.", "", "## Interpretation limits", ""]
    lines.extend(f"- {note}" for note in report["limitations"])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    lab = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=lab / "data/yolo/pcbnet34_v1")
    parser.add_argument("--predictions-json", type=Path, default=lab / "outputs/pcbnet34_v1_val_errors/predictions.json")
    parser.add_argument("--output", type=Path, default=lab / "outputs/pcbnet34_v1_classification_gap")
    parser.add_argument("--confidence", "--conf", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.5)
    args = parser.parse_args(argv)
    if not 0 <= args.confidence <= 1 or not 0 < args.iou <= 1:
        parser.error("Confidence must be in [0,1] and IoU in (0,1].")
    from analyze_yolo_errors import load_validation, load_predictions, match_detections
    names, records = load_validation(args.data_root.resolve())
    predictions = load_predictions(args.predictions_json, records, names, args.confidence)
    for record in records:
        record["predictions"] = predictions[record["image"]]
        record["class_aware_matching"] = match_detections(record["ground_truth"], record["predictions"], args.iou, args.confidence)
        record["location_matching"] = location_matches(record["ground_truth"], record["predictions"], args.confidence, args.iou)
    report = {"split": "val", "data_root": str(args.data_root.resolve()),
              "predictions_json": str(args.predictions_json.resolve()), "confidence": args.confidence,
              "iou_threshold": args.iou, "image_count": len(records), **build_report(records, names)}
    save_report(args.output.resolve(), report)
    print(json.dumps(report["overall"], indent=2))
    print(f"CPU cache diagnostic written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
