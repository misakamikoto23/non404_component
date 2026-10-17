"""Export official component crops from grouped train/dev only; never read test labels.

The archive is read selectively, using the frozen split's stems. Output images
retain native pixels; labels retain the official vocabulary without mapping.
This prepares data only. It neither trains nor changes the active classifier.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import zipfile

from PIL import Image, ImageDraw, ImageOps, ImageStat


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_sha256(image):
    image = image.convert("RGB")
    digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def validate_split(split, groups, forbidden):
    """Validate metadata only, including sealed test metadata, never its labels."""
    partitions = split["partitions"]
    if set(partitions) != {"train", "dev", "test"}:
        raise ValueError("Expected exactly train/dev/test partitions")
    owners, pixel_owners, cases = {}, {}, set()
    for partition, rows in partitions.items():
        for row in rows:
            group, case, pixel = row["group"], row["case"], row["pixel_sha256"]
            if case in cases:
                raise ValueError(f"Duplicate case id: {case}")
            cases.add(case)
            for key, mapping in ((group, owners), (pixel, pixel_owners)):
                if key in mapping and mapping[key] != partition:
                    raise ValueError(f"Cross-partition group or pixel: {key}")
                mapping[key] = partition
            group_meta = groups[group]
            expected_id = f"official:train200:{row['stem']}"
            if expected_id not in group_meta["members"] or pixel not in group_meta["pixel_sha256"]:
                raise ValueError(f"Source/group mismatch: {row['stem']}")
    heldout = {row["group"] for p in ("dev", "test") for row in partitions[p]}
    if heldout != set(forbidden["held_out_groups"]):
        raise ValueError("Forbidden manifest does not match dev/test groups")
    banned_pixels = {p for g in heldout for p in groups[g]["pixel_sha256"]}
    for row in partitions["train"]:
        if row["group"] in heldout or row["pixel_sha256"] in banned_pixels:
            raise ValueError(f"Forbidden training source: {row['stem']}")
    return {"partition_cases": {p: len(rows) for p, rows in partitions.items()},
            "partition_groups": {p: len({r['group'] for r in rows}) for p, rows in partitions.items()},
            "forbidden_groups": len(heldout), "train_ban_check": "passed",
            "test_labels_read": 0, "test_crops_exported": 0}


def crop_box(bbox, width, height, padding_px=2, padding_ratio=0.1):
    """Official bottom-left continuous pixels -> clamped PIL top-left crop."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in bbox):
        raise ValueError("bbox coordinates must be finite numbers")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox has non-positive extent")
    tl = [x1, height - y2, x2, height - y1]
    px = padding_px + (x2 - x1) * padding_ratio
    py = padding_px + (y2 - y1) * padding_ratio
    requested = [math.floor(tl[0] - px), math.floor(tl[1] - py),
                 math.ceil(tl[2] + px), math.ceil(tl[3] + py)]
    actual = [max(0, requested[0]), max(0, requested[1]),
              min(width, requested[2]), min(height, requested[3])]
    if actual[2] <= actual[0] or actual[3] <= actual[1]:
        raise ValueError("bbox is entirely outside image")
    return tl, actual, actual != requested


def validate_type_name(value):
    """Use raw type as ImageFolder directory, rejecting unsafe Windows paths."""
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError("Empty or invalid official type")
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', value) or value.endswith((".", " ")):
        raise ValueError(f"Type is not a safe folder name: {value!r}")
    if value.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError(f"Reserved folder name: {value!r}")
    return value


def crop_quality(image):
    """Conservative visual flags; never relabel official annotations automatically."""
    gray = image.convert("L")
    histogram = gray.histogram()
    dark_fraction = sum(histogram[:220]) / (gray.width * gray.height)
    stddev = ImageStat.Stat(gray).stddev[0]
    flags = []
    if dark_fraction < 0.005 or stddev < 4:
        flags.append("near_blank_or_low_contrast")
    if min(image.size) < 4:
        flags.append("very_narrow_crop")
    return {"dark_fraction_below_220": round(dark_fraction, 6),
            "grayscale_stddev": round(stddev, 4), "flags": flags}


def write_context_views(dataset, output, archive_path, crops):
    """Diagnostic pairs only, outside the training directory; no GT alteration."""
    selected = {}
    for row in crops:
        selected.setdefault((row["partition"], row["type"]), row)
    metadata = []
    with zipfile.ZipFile(archive_path) as archive:
        cached_member, source = None, None
        for partition in ("train", "dev"):
            examples = [row for (p, _), row in sorted(selected.items()) if p == partition]
            canvas = Image.new("RGB", (1000, math.ceil(len(examples) / 4) * 170), "#eeeeee")
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(examples):
                if row["image_member"] != cached_member:
                    with Image.open(io.BytesIO(archive.read(row["image_member"]))) as loaded:
                        source = loaded.convert("RGB")
                    cached_member = row["image_member"]
                x1, y1, x2, y2 = row["bbox_top_left"]
                mx, my = max(12, (x2 - x1) * .5), max(12, (y2 - y1) * .5)
                box = [max(0, math.floor(x1 - mx)), max(0, math.floor(y1 - my)),
                       min(source.width, math.ceil(x2 + mx)), min(source.height, math.ceil(y2 + my))]
                with Image.open(dataset / row["path"]) as crop_source:
                    crop = crop_source.convert("RGB")
                context = source.crop(box)
                x, y = index % 4 * 250, index // 4 * 170
                for offset, view in ((0, crop), (125, context)):
                    thumb = ImageOps.contain(view, (120, 125))
                    canvas.paste(thumb, (x + offset + (125 - thumb.width) // 2, y + (125 - thumb.height) // 2))
                draw.text((x + 3, y + 128), f"{row['case']} {row['type']}".encode("ascii", "replace").decode(), fill="black")
                draw.text((x + 3, y + 145), "original      context (diagnostic)", fill="black")
                metadata.append({"partition": partition, "type": row["type"], "case": row["case"],
                                 "key": row["key"], "original_crop": row["path"],
                                 "context_box_top_left": box})
            canvas.save(output / "sheets" / f"{partition}_original_vs_context.png")
    write_json(output / "context_views.json", {"policy": "per-axis margin=max(12px, 50% GT bbox dimension); diagnostic only; neighboring symbols/text may enter", "examples": metadata})


def archive_members(archive, allowed_rows):
    """Index names, but only open train/dev image and annotation members."""
    indexed = defaultdict(list)
    for name in archive.namelist():
        if "/200_train_cases/" in name:
            indexed[Path(name).name].append(name)
    result = {}
    for row in allowed_rows:
        stem = row["stem"]
        images = [n for ext in (".png", ".jpg", ".jpeg") for n in indexed.get(stem + ext, [])]
        labels = indexed.get(stem + "_target_named.json", [])
        if len(images) != 1 or len(labels) != 1:
            raise ValueError(f"Ambiguous/missing official source for {stem}: {len(images)} images, {len(labels)} labels")
        result[stem] = (images[0], labels[0])
    return result


def make_report(output, dataset, summary, counts):
    lines = ["# Official gold component crops — 2026-10-17", "",
             "Official train200 annotations only. No weak labels, API calls, training, or test-label reads.", "",
             f"- Frozen split: `{summary['split_name']}`; train {summary['partition_cases']['train']} cases, dev {summary['partition_cases']['dev']} cases.",
             f"- Exported train {summary['crops']['train']} / val (dev) {summary['crops']['val']} native-resolution crops.",
             f"- Train has {summary['unique_source_pixels']['train']} distinct image contents in {summary['partition_groups']['train']} design groups; repeated official records are retained and identified by hash.",
             f"- Exact official types: {len(counts)}; skipped {summary['skipped']} objects; padded crops clipped at image edge: {summary['clipped_crops']}.",
             "- Official GT coordinates are bottom-left; exported image crops are top-left using y'=H-y and swapped y bounds.",
             "- Per-component manifest preserves key, Name, value, original bbox, crop bbox, image hash, design group and archive member.",
             "- Split membership is clean for a future retrain, not evidence that existing checkpoints are uncontaminated.",
             "- Duplicate grouping follows the frozen audit; undetected near-duplicate designs remain a limitation.", "",
             "## Critical crop-geometry limitation", "",
             "Visual inspection found official boxes that cover only a small interior/anchor region rather than the full symbol. For example, train case 0021 has a bjt crop containing white interior and a mosfet crop showing only a fragment. Coordinate overlays align the tiny boxes at the correct positions, so a global y-flip does not fix this issue.",
             f"Conservative pixel QA flags {summary.get('quality_flagged_crops', 0)} crops; it cannot detect all partial symbols. These are annotation-derived candidate crops, not an approved training release. Inspect geometry and consider image-only context expansion or symbol-boundary refinement before training. Keep the original GT bbox unchanged for evaluation.", "",
             "`sheets/*_original_vs_context.png` compares original crops to a diagnostic context margin=max(12px, 50% bbox dimension) on each axis. Original crops and GT are unchanged; context includes neighboring symbols/text and is not automatically approved for training. Source-specific box conventions also affect detector-to-official bbox scoring, so calibrate those conventions using train/dev only.", "",
             "## Class counts", "", "| Raw official type | Train crops | Dev crops | Train groups | Dev groups |", "|---|---:|---:|---:|---:|"]
    for name, row in counts.items():
        lines.append(f"| {name} | {row['train']} | {row['val']} | {row['train_groups']} | {row['val_groups']} |")
    lines += ["", "## Balanced training recommendation", "",
              "Use the original crops as evidence. Do not create duplicate files to claim equal independent samples.",
              "For a first classifier experiment, a WeightedRandomSampler with per-example weight 1/n_class and num_samples equal to the train crop count gives equal expected class exposure. It still repeats scarce examples and cannot replace design diversity. Use it OR a class-weighted loss initially, not both.",
              "Keep dev sampling unweighted. Report raw-type accuracy, macro F1, per-class support, and design-group support. Classes absent from train cannot be learned; classes absent from dev have no measured validation result.",
              "Do not use this set to overwrite the existing 27-family silver-label classifier. Official raw types and semantic families need an explicit, independently evaluated mapping.", "",
              "## Reproduce", "", "```powershell", summary["command"], "```", "",
              "The exporter refuses to overwrite an existing dataset/report. For another run, specify fresh --dataset and --output paths.", "",
              "Focused checks: `python -m unittest tests.test_official_gold_crops -v` (synthetic fixtures, including a guard that rejects any attempted test-member read).", "",
              f"Dataset: `{dataset}`", "", "Overlays and crop sheets show train/dev examples only. They support geometry inspection; official annotations are gold by source, not independently verified human truth."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cards = []
    for folder in ("sheets", "overlays"):
        for path in sorted((output / folder).glob("*.png")):
            src = path.relative_to(output).as_posix()
            cards.append(f'<figure><figcaption>{html.escape(path.stem)}</figcaption><a href="{src}"><img loading="lazy" src="{src}"></a></figure>')
    (output / "gallery.html").write_text('<!doctype html><meta charset="utf-8"><title>Official gold crop geometry QA</title><style>body{font:16px sans-serif;margin:24px;background:#eee}img{max-width:100%;max-height:1000px}figure{display:inline-block;vertical-align:top;max-width:900px;background:white;padding:12px;margin:8px}figcaption{font-weight:bold}</style><h1>Official gold candidate crops: train/dev only</h1><p>Some GT boxes are partial symbol anchors. These crops are not yet approved for training. See REPORT.md for limitations.</p>' + "".join(cards), encoding="utf-8")


def prepare(lab, split_path, archive_path, dataset, output, padding_px=2, padding_ratio=0.1):
    lab, split_path, archive_path, dataset, output = map(Path, (lab, split_path, archive_path, dataset, output))
    split = read_json(split_path)
    groups_path = lab / split["groups_file"]
    forbidden_path = lab / split["forbidden_manifest"]
    groups, forbidden = read_json(groups_path), read_json(forbidden_path)
    summary = validate_split(split, groups, forbidden)
    if padding_px < 0 or padding_ratio < 0:
        raise ValueError("Crop padding must be nonnegative")
    if dataset.exists() or output.exists():
        raise FileExistsError("Refusing to modify existing dataset or report directory; choose new output paths")
    rows = [(p, row) for p in ("train", "dev") for row in split["partitions"][p]]
    manifest, skipped, sources = [], [], []
    counts = defaultdict(lambda: {"train": 0, "val": 0, "train_group_ids": set(), "val_group_ids": set()})
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_members(archive, [r for _, r in rows])
        # Validate every allowed image hash before creating output. Never read test JSON or image bytes.
        for partition, row in rows:
            image_member, label_member = members[row["stem"]]
            with Image.open(io.BytesIO(archive.read(image_member))) as image:
                actual_hash = pixel_sha256(image)
            if actual_hash != row["pixel_sha256"]:
                raise ValueError(f"Official image hash mismatch: {row['stem']}")
        dataset.mkdir(parents=True)
        output.mkdir(parents=True)
        (output / "overlays").mkdir()
        (output / "sheets").mkdir()
        overlay_done, sheet_examples = set(), defaultdict(list)
        for partition, row in rows:
            export_split = "train" if partition == "train" else "val"
            image_member, label_member = members[row["stem"]]
            label_bytes = archive.read(label_member)
            gt = json.loads(label_bytes.decode("utf-8-sig"))
            with Image.open(io.BytesIO(archive.read(image_member))) as source:
                image = source.convert("RGB")
            sources.append({**row, "partition": partition, "image_member": image_member,
                            "label_member": label_member, "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
                            "width": image.width, "height": image.height})
            overlay = image.copy() if (partition, row["tag"]) not in overlay_done else None
            for index, (key, component) in enumerate(gt["components"].items(), start=1):
                raw_type = validate_type_name(component.get("type"))
                try:
                    tl, box, clipped = crop_box(component.get("bbox"), image.width, image.height, padding_px, padding_ratio)
                except ValueError as error:
                    skipped.append({"partition": partition, "case": row["case"], "key": key,
                                    "type": raw_type, "bbox": component.get("bbox"), "reason": str(error)})
                    continue
                relative = Path(export_split) / raw_type / f"{row['case']}_{index:04d}.png"
                target = dataset / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                crop = image.crop(box)
                crop.save(target)
                entry = {"path": relative.as_posix(), "partition": partition, "imagefolder_split": export_split,
                         "case": row["case"], "stem": row["stem"], "group": row["group"], "tag": row["tag"],
                         "source_pixel_sha256": row["pixel_sha256"], "image_member": image_member,
                         "label_member": label_member, "label_source": "official_train200_gold",
                         "key": key, "type": raw_type, "Name": component.get("Name"), "value": component.get("value"),
                         "bbox_bottom_left": component["bbox"], "bbox_top_left": tl, "crop_box_top_left": box,
                         "clipped_padding_or_bbox": clipped, "crop_pixel_sha256": pixel_sha256(crop),
                         "width": crop.width, "height": crop.height, "quality": crop_quality(crop)}
                manifest.append(entry)
                counts[raw_type][export_split] += 1
                counts[raw_type][export_split + "_group_ids"].add(row["group"])
                if len(sheet_examples[(partition, raw_type)]) < 2:
                    sheet_examples[(partition, raw_type)].append((crop.copy(), row["case"], raw_type))
                if overlay is not None:
                    ImageDraw.Draw(overlay).rectangle(tl, outline=(0, 160, 0), width=2)
            if overlay is not None:
                overlay.thumbnail((1800, 1800))
                overlay.save(output / "overlays" / f"{partition}_{row['stem'].replace(' ', '_')}.png")
                overlay_done.add((partition, row["tag"]))
        class_names = sorted(counts)
        # Identical directory vocabularies keep ImageFolder class indices aligned.
        for name in class_names:
            for export_split in ("train", "val"):
                (dataset / export_split / name).mkdir(parents=True, exist_ok=True)
        for partition in ("train", "dev"):
            examples = [x for k, v in sorted(sheet_examples.items()) if k[0] == partition for x in v]
            sheet = Image.new("RGB", (800, math.ceil(len(examples) / 5) * 130), "#eeeeee")
            draw = ImageDraw.Draw(sheet)
            for index, (crop, case, raw_type) in enumerate(examples):
                x, y = index % 5 * 160, index // 5 * 130
                thumb = ImageOps.contain(crop, (150, 100))
                sheet.paste(thumb, (x + (160 - thumb.width) // 2, y + (100 - thumb.height) // 2))
                label = f"{case}: {raw_type}"
                draw.text((x + 3, y + 105), label.encode("ascii", "replace").decode(), fill="black")
            sheet.save(output / "sheets" / f"{partition}_classes.png")
    class_counts = {name: {"train": counts[name]["train"], "val": counts[name]["val"],
                           "train_groups": len(counts[name]["train_group_ids"]),
                           "val_groups": len(counts[name]["val_group_ids"])} for name in sorted(counts)}
    crop_hashes = {p: {r["crop_pixel_sha256"] for r in manifest if r["imagefolder_split"] == p} for p in ("train", "val")}
    summary.update({"split_name": split["name"], "split_sha256": sha256_file(split_path),
                    "groups_sha256": sha256_file(groups_path), "forbidden_sha256": sha256_file(forbidden_path),
                    "archive": str(archive_path.resolve()), "padding_px": padding_px, "padding_ratio": padding_ratio,
                    "crops": dict(Counter(r["imagefolder_split"] for r in manifest)), "classes": len(class_counts),
                    "unique_source_pixels": {p: len({r['pixel_sha256'] for r in sources if r['partition'] == p}) for p in ("train", "dev")},
                    "clipped_crops": sum(r["clipped_padding_or_bbox"] for r in manifest), "skipped": len(skipped),
                    "quality_flagged_crops": sum(bool(r["quality"]["flags"]) for r in manifest),
                    "quality_flags": dict(Counter(flag for r in manifest for flag in r["quality"]["flags"])),
                    "training_ready": False,
                    "shared_crop_pixel_hashes_train_dev": len(crop_hashes["train"] & crop_hashes["val"]),
                    "shared_crop_hash_note": "Identical standard symbol pixels across distinct diagrams are possible; design groups, not symbol appearance, define the split.",
                    "train_classes_absent_from_dev": [c for c, n in class_counts.items() if n["train"] and not n["val"]],
                    "dev_classes_absent_from_train": [c for c, n in class_counts.items() if n["val"] and not n["train"]],
                    "command": "python scripts/prepare_official_gold_crops.py"})
    write_json(dataset / "manifest.json", {"summary": summary, "classes": class_names, "sources": sources, "crops": manifest, "skipped": skipped})
    write_json(dataset / "class_counts.json", class_counts)
    write_json(output / "summary.json", summary)
    write_json(output / "class_counts.json", class_counts)
    write_context_views(dataset, output, archive_path, manifest)
    make_report(output, dataset.resolve(), summary, class_counts)
    return summary


def main(argv=None):
    lab = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=lab / "configs/contest6_group_split_v1.json")
    parser.add_argument("--archive", type=Path, default=lab.parent / "赛题六公开数据集.zip")
    parser.add_argument("--dataset", type=Path, default=lab / "data/resnet/official_gold_v1")
    parser.add_argument("--output", type=Path, default=lab / "outputs/official_gold_crops_20261017")
    parser.add_argument("--padding-px", type=int, default=2)
    parser.add_argument("--padding-ratio", type=float, default=0.1)
    args = parser.parse_args(argv)
    summary = prepare(lab, args.split, args.archive, args.dataset, args.output, args.padding_px, args.padding_ratio)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
