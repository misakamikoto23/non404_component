"""Reproducible, GT-complete crop/control datasets without changing source labels.

The crop closure is checked against *every* original GT box. This prevents an
unlabelled, clipped GT fragment from becoming negative background. This guarantee
is relative to the supplied labels; it cannot detect omissions in those labels.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/reviewed_versions/manual_round_003/yolo"
DEFAULT_DEST = ROOT / "data/experiments/automation_round_001"
DEFAULT_REPORT = ROOT / "outputs/automation_round_001/crop_dataset_audit.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_boxes(path, width, height):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = map(float, line.split())
        box = [max(0., (cx-bw/2)*width), max(0., (cy-bh/2)*height),
               min(float(width), (cx+bw/2)*width), min(float(height), (cy+bh/2)*height)]
        assert box[2] > box[0] and box[3] > box[1], (path, line)
        rows.append((int(cls), box))
    return rows


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) + 1e-7 and min(a[3], b[3]) > max(a[1], b[1]) + 1e-7


def contains(outer, inner):
    return (outer[0] <= inner[0] + 1e-7 and outer[1] <= inner[1] + 1e-7
            and outer[2] >= inner[2] - 1e-7 and outer[3] >= inner[3] - 1e-7)


def close_crop(rect, rows, width, height):
    """Monotonic expansion terminates after at most len(rows)+1 iterations."""
    rect = list(rect)
    for _ in range(len(rows) + 2):
        previous = rect[:]
        for _, box in rows:
            if intersects(rect, box) and not contains(rect, box):
                rect = [min(rect[0], max(0, math.floor(box[0])-2)),
                        min(rect[1], max(0, math.floor(box[1])-2)),
                        max(rect[2], min(width, math.ceil(box[2])+2)),
                        max(rect[3], min(height, math.ceil(box[3])+2))]
        if previous == rect:
            assert all(not intersects(rect, box) or contains(rect, box) for _, box in rows)
            return rect
    raise AssertionError("Crop closure did not converge")


def choose_crop(rows, width, height, seed):
    full = [0, 0, width, height]
    if not rows:
        return full, {"fallback": "no_gt", "initial_side": None}
    if max(width, height) <= 1024:
        return full, {"fallback": "image_at_most_1024", "initial_side": 1024}
    rng = random.Random(seed)
    # Smallest half, reproducibly shuffled: promote fine detail without making
    # every image's view depend on just one minimum-size annotation.
    anchors = sorted(rows, key=lambda row: min(row[1][2]-row[1][0], row[1][3]-row[1][1]))
    anchors = anchors[:max(1, (len(anchors)+1)//2)]
    rng.shuffle(anchors)
    candidates = []
    for _, anchor in anchors[:12]:
        side = 1536 if max(anchor[2]-anchor[0], anchor[3]-anchor[1]) > 768 else 1024
        cw, ch = min(width, side), min(height, side)
        x = round((anchor[0]+anchor[2])/2 - cw/2 + rng.uniform(-.1, .1)*cw)
        y = round((anchor[1]+anchor[3])/2 - ch/2 + rng.uniform(-.1, .1)*ch)
        x, y = max(0, min(width-cw, x)), max(0, min(height-ch, y))
        rect = close_crop([x, y, x+cw, y+ch], rows, width, height)
        area_ratio = (rect[2]-rect[0])*(rect[3]-rect[1])/(width*height)
        indices = [i for i, (_, b) in enumerate(rows) if intersects(rect, b)]
        # Near-full views confer little size benefit and complicate attribution.
        if area_ratio < .85 and indices:
            candidates.append((area_ratio, rect, side, indices))
    if not candidates:
        return full, {"fallback": "all_safe_crops_near_full", "initial_side": 1024}
    _, rect, side, indices = min(candidates, key=lambda row: row[0])
    return rect, {"fallback": None, "initial_side": side, "included_gt_indices": indices}


def labels_for_crop(rows, rect):
    width, height = rect[2]-rect[0], rect[3]-rect[1]
    lines = []
    for cls, box in rows:
        if not intersects(rect, box):
            continue
        assert contains(rect, box), (rect, box)
        x1, y1, x2, y2 = box
        vals = [(x1+x2-2*rect[0])/(2*width), (y1+y2-2*rect[1])/(2*height),
                (x2-x1)/width, (y2-y1)/height]
        lines.append(f"{cls} " + " ".join(f"{v:.10f}" for v in vals))
    return "\n".join(lines) + ("\n" if lines else "")


def self_test():
    # Expansion must catch a second box that the first expansion newly touches.
    rows = [(0, [40, 40, 70, 70]), (0, [68, 60, 95, 90]), (0, [110, 5, 120, 15])]
    rect = close_crop([0, 0, 50, 50], rows, 200, 200)
    assert rect == [0, 0, 97, 92], rect
    assert len(labels_for_crop(rows, rect).splitlines()) == 2
    # A touching box outside an edge has no pixel-area intersection.
    assert not intersects([0, 0, 50, 50], [50, 0, 60, 10])
    assert choose_crop(rows, 200, 200, 5)[1]["fallback"] == "image_at_most_1024"
    assert choose_crop(rows, 2000, 2000, 5) == choose_crop(rows, 2000, 2000, 5)


def audit_existing(report_path):
    import numpy as np
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source, dest = Path(report["source"]), Path(report["dest"])
    crop_count, max_error = 0, 0.
    for relative, digest in report["source_sha256"].items():
        assert sha256(source/relative) == digest, relative
    for mapping in report["mappings"]:
        for kind in ("image", "label"):
            assert sha256(dest/mapping[kind]) == mapping[kind+"_sha256"]
        if not mapping["is_crop"]:
            continue
        crop_count += 1
        original = next((source/"images/train").glob(mapping["source_id"]+".*"))
        rect = mapping["rect_xyxy"]
        with Image.open(original) as im, Image.open(dest/mapping["image"]) as crop:
            assert np.array_equal(np.array(im.crop(rect)), np.array(crop)), mapping["image"]
            rows = load_boxes(source/"labels/train"/(mapping["source_id"]+".txt"), *im.size)
            expected = [(cls, box) for cls, box in rows if intersects(rect, box)]
            assert all(not intersects(rect, box) or contains(rect, box) for _, box in rows)
            actual = load_boxes(dest/mapping["label"], rect[2]-rect[0], rect[3]-rect[1])
            assert len(expected) == len(actual)
            for (cls, box), (actual_cls, actual_box) in zip(expected, actual):
                assert cls == actual_cls
                translated = [actual_box[0]+rect[0], actual_box[1]+rect[1],
                              actual_box[2]+rect[0], actual_box[3]+rect[1]]
                max_error = max(max_error, max(abs(a-b) for a, b in zip(box, translated)))
    assert max_error < 1e-5, max_error
    for entry in report["holdout"]:
        assert sha256(dest/entry["arm"]/entry["path"]) == entry["sha256"]
    result = {"status": "passed", "source_files_sha_verified": len(report["source_sha256"]),
              "training_samples_sha_verified": len(report["mappings"]),
              "holdout_files_sha_verified": len(report["holdout"]),
              "cropped_images_pixel_exact": crop_count, "max_label_roundtrip_error_px": max_error}
    audit_path = report_path.with_name("crop_dataset_verification.json")
    audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def summarize_training(report_path):
    import yaml
    report_dir = report_path.parent
    arms = {"control_full": "pcbnet_component_yolo26n_auto_control6_v2",
            "mixed_crop": "pcbnet_component_yolo26n_auto_mixed_crop6"}
    results, settings = {}, {}
    for arm, run_name in arms.items():
        run = ROOT/"runs"/run_name
        summary = json.loads((run/"baseline_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "training_complete" and summary["epochs_completed"] == 6
        rows = list(csv.DictReader((run/"results.csv").open(encoding="utf-8")))
        assert len(rows) == 6 and int(rows[-1]["epoch"]) == 6
        settings[arm] = yaml.safe_load((run/"args.yaml").read_text(encoding="utf-8"))
        export_path = ROOT/"weights"/(run_name+".pt")
        if export_path.exists():
            assert sha256(export_path) == sha256(run/"weights/best.pt")
        else:
            shutil.copy2(run/"weights/best.pt", export_path)
        results[arm] = {"run": str(run), "weights": str(export_path), "weights_sha256": sha256(export_path),
            "initial_model_sha256": sha256(Path(summary["initial_model"])),
            "epochs_completed": 6, "training_seconds_csv": float(rows[-1]["time"]),
            "best_epoch_by_csv_map50_95": int(max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))["epoch"]),
            "peak_reserved_mib": max(row["reserved_peak_mib"] for row in summary["gpu_memory"]),
            "validation": summary["validation"], "test_evaluated": "test" in summary}
    ignored = {"data", "name", "save_dir"}
    differences = {key: [settings[a].get(key) for a in arms]
                   for key in set().union(*(s.keys() for s in settings.values()))
                   if key not in ignored and len({json.dumps(settings[a].get(key), sort_keys=True) for a in arms}) > 1}
    assert not differences, differences
    assert len({row["initial_model_sha256"] for row in results.values()}) == 1
    initial = json.loads((ROOT/"runs/pcbnet_component_yolo26n_review_round_003_v2/baseline_summary.json").read_text(encoding="utf-8"))
    payload = {"status": "complete", "same_hyperparameters_except_dataset_and_output": True,
               "same_initial_weights_sha256": True, "original_round_003_validation": initial["validation"],
               "actual_crop_fraction": 61/466, "arms": results,
               "promoted_as_default": False, "test_evaluated": False}
    (report_dir/"crop_training_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    status_file = report_dir/"crop_stage_status.json"
    status = json.loads(status_file.read_text(encoding="utf-8"))
    status.update(status="complete", mixed_complete=True, results="outputs/automation_round_001/crop_training_results.json",
                  promoted_as_default=False, gpu_task_released=True)
    status_file.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["\n## 已完成的训练结果\n", "| 模型 | P | R | mAP50 | mAP50–95 |",
             "|---|---:|---:|---:|---:|"]
    for name, metrics in [("第三轮起点", initial["validation"])]+[(arm, value["validation"]) for arm, value in results.items()]:
        lines.append("| " + name + " | " + " | ".join(f"{metrics[key]*100:.2f}%" for key in
            ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")) + " |")
    lines += ["", "以上是每组最佳checkpoint重新验证得到的标准指标，两组所有超参数除数据路径和输出路径外完全一致，初始权重SHA也一致。"]
    for arm, result in results.items():
        lines.append(f"\n{arm}：实际6轮，CSV训练耗时{result['training_seconds_csv']:.1f}秒，峰值预留显存{result['peak_reserved_mib']:.0f}MiB，CSV最佳第{result['best_epoch_by_csv_map50_95']}轮。")
    control = results["control_full"]["validation"]
    mixed = results["mixed_crop"]["validation"]
    lines += ["", "mixed相比同预算control的mAP50–95变化为"
              f"{(mixed['metrics/mAP50-95(B)']-control['metrics/mAP50-95(B)'])*100:+.2f}个百分点。"
              "本次保守裁块比例和短周期设置未显示收益，两组也都没有超过第三轮起点，不能把‘增加裁块’直接当作已验证的提升方法。"]
    lines += ["", "未替换现有默认权重。是否采用裁块策略应结合独立候选池P/R与运行成本判断；这一短周期对照不能证明达到双95%，也不支持无条件增加训练轮数。"]
    markdown_path = report_dir/"crop_training_report.md"
    base = markdown_path.read_text(encoding="utf-8").split("\n## 已完成的训练结果")[0]
    base = base.replace("状态：训练中，最终结果以 crop_stage_status.json 为准。", "状态：两组训练已完成，详见 crop_training_results.json。")
    base = base.replace("训练结果将在两组完成后追加。", "训练结束后再次执行全部数据校验，源文件和训练/holdout副本SHA均保持一致。")
    markdown_path.write_text(base + "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    parser.add_argument("--summarize-training", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("CROP_SELF_TEST_OK")
        return
    if args.audit_existing:
        audit_existing(args.report)
        return
    if args.summarize_training:
        summarize_training(args.report)
        return
    source, dest = args.source.resolve(), args.dest.resolve()
    if dest.exists():
        raise FileExistsError(f"Refuse to overwrite existing experiment: {dest}")
    source_files = sorted(p for root in (source/"images", source/"labels") for p in root.rglob("*") if p.is_file())
    before = {str(p.relative_to(source)): sha256(p) for p in source_files}
    train_images = sorted(p for p in (source/"images/train").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    assert len(train_images) == 233, len(train_images)
    mappings, crop_stats, holdout = [], Counter(), []
    for arm in ("control_full", "mixed_crop"):
        for kind in ("images", "labels"):
            for split in ("train", "val", "test"):
                (dest/arm/kind/split).mkdir(parents=True)
        (dest/arm/"dataset.yaml").write_text(
            "path: " + json.dumps(str(dest/arm), ensure_ascii=False) +
            "\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [component]\n", encoding="utf-8")
        for split in ("val", "test"):
            for kind in ("images", "labels"):
                for src in sorted((source/kind/split).iterdir()):
                    if not src.is_file():
                        continue
                    target = dest/arm/kind/split/src.name
                    shutil.copy2(src, target)
                    assert sha256(target) == before[str(src.relative_to(source))]
                    holdout.append({"arm": arm, "path": str(src.relative_to(source)), "sha256": sha256(target)})
    for image_path in train_images:
        source_label = source/"labels/train"/(image_path.stem+".txt")
        with Image.open(image_path) as im:
            width, height = im.size
            rows = load_boxes(source_label, width, height)
            rect, info = choose_crop(rows, width, height, args.seed+int(image_path.stem))
            for arm in ("control_full", "mixed_crop"):
                for view in (0, 1):
                    suffix = image_path.suffix if arm == "control_full" or view == 0 else ".png"
                    name = f"{image_path.stem}_v{view}"
                    out_image = dest/arm/"images/train"/(name+suffix)
                    out_label = dest/arm/"labels/train"/(name+".txt")
                    is_crop = arm == "mixed_crop" and view == 1 and rect != [0, 0, width, height]
                    if is_crop:
                        im.crop(rect).save(out_image)
                        out_label.write_text(labels_for_crop(rows, rect), encoding="utf-8")
                    else:
                        if suffix == image_path.suffix:
                            shutil.copy2(image_path, out_image)
                        else:
                            im.save(out_image)
                        shutil.copy2(source_label, out_label)
                    mappings.append({"arm": arm, "source_id": image_path.stem, "view": view,
                        "image": str(out_image.relative_to(dest)), "image_sha256": sha256(out_image),
                        "label": str(out_label.relative_to(dest)), "label_sha256": sha256(out_label),
                        "source_image_sha256": before[str(image_path.relative_to(source))],
                        "source_label_sha256": before[str(source_label.relative_to(source))],
                        "rect_xyxy": rect if is_crop else [0, 0, width, height],
                        "is_crop": is_crop, "objects": len(out_label.read_text(encoding="utf-8").splitlines()),
                        "crop_info": info if arm == "mixed_crop" and view == 1 else None})
            crop_stats[info["fallback"] or "valid_crop"] += 1
    after = {str(p.relative_to(source)): sha256(p) for p in source_files}
    assert before == after, "Source images/labels changed during preparation"
    for arm in ("control_full", "mixed_crop"):
        counts = Counter(m["source_id"] for m in mappings if m["arm"] == arm)
        assert len(counts) == 233 and set(counts.values()) == {2}
    assert len(holdout) == 240
    report = {"seed": args.seed, "source": str(source), "dest": str(dest),
        "samples_per_arm": 466, "source_frequency_per_arm": 2,
        "crop_stats": dict(crop_stats), "source_files_unchanged": len(before),
        "holdout_copies_sha_verified": len(holdout), "test_used_for_training_or_selection": False,
        "crop_label_guarantee": "Every intersecting supplied GT is fully contained; source annotation omissions remain possible.",
        "source_sha256": before, "holdout": holdout, "mappings": mappings}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_sha256", "holdout", "mappings"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
