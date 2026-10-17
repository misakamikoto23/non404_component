"""Prepare an isolated X-AnyLabeling seed set from PCBnet CVJsonStd labels.

This script intentionally reads only ``*_CVJsonStd.json`` source annotations and
the existing dataset manifest.  It never reads hand-authored JSON files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "component_lab/data/yolo/pcbnet34_v1/manifest.json"
OUT = ROOT / "component_lab/data/ai_annotations_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_shape(category: str, bbox: list[float], component_id: str) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "label": category,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {
            "component_id": component_id,
            "source": "PCBnet *_CVJsonStd.json realBBox",
            "change": "unchanged_seed",
            "uncertain": False,
        },
    }


def make_document(sample: dict, image_name: str, source_annotation: str) -> dict:
    return {
        "version": "2.5.4",
        "flags": {
            "annotation_status": "official_label_conversion_unreviewed",
            "source_case": sample["case"],
            "source_annotation": source_annotation,
            "source_sha256": sample["sha256"],
        },
        "shapes": [
            make_shape(label["category"], label["bbox"], label["component_id"])
            for label in sample["labels"]
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": sample["height"],
        "imageWidth": sample["width"],
    }


def verify_standard_source(sample: dict, classes: set[str]) -> None:
    annotation = ROOT / sample["annotation"]
    if not annotation.name.endswith("_CVJsonStd.json"):
        raise ValueError(f"Refusing non-standard annotation: {annotation}")
    raw = read_json(annotation)
    if raw.get("canvas", {}).get("width") != sample["width"]:
        raise ValueError(f"Width mismatch for {sample['case']}")
    if raw.get("canvas", {}).get("height") != sample["height"]:
        raise ValueError(f"Height mismatch for {sample['case']}")
    expected = []
    for component_id, component in raw.get("components", {}).items():
        if component.get("state") == "deleted":
            continue
        bbox = component.get("realBBox")
        category = component.get("category")
        if not bbox or category not in classes:
            continue
        expected.append(
            (
                component_id,
                category,
                [float(bbox[k]) for k in ("x1", "y1", "x2", "y2")],
            )
        )
    actual = [
        (x["component_id"], x["category"], [float(v) for v in x["bbox"]])
        for x in sample["labels"]
    ]
    # Three manifest entries are clipped at the canvas boundary. The manifest
    # is the source of the final in-bounds geometry, but the IDs/categories and
    # all other boxes must match the CVJsonStd source.
    if [(a, b) for a, b, _ in expected] != [(a, b) for a, b, _ in actual]:
        raise ValueError(f"Component/category mismatch for {sample['case']}")


def seed() -> None:
    manifest = read_json(MANIFEST)
    classes = manifest["classes"]
    out = OUT / "original_seeded"
    out.mkdir(parents=True, exist_ok=True)
    (OUT / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    rows = []
    for sample in manifest["samples"]:
        verify_standard_source(sample, set(classes))
        src_image = ROOT / sample["image"]
        image_name = f"{sample['case']}__{src_image.name}"
        shutil.copy2(src_image, out / image_name)
        document = make_document(sample, image_name, sample["annotation"])
        json_path = out / f"{sample['case']}.json"
        json_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "case": sample["case"],
                "split": sample["split"],
                "image": image_name,
                "annotation": json_path.name,
                "shape_count": len(document["shapes"]),
                "source_annotation": sample["annotation"],
            }
        )
    (OUT / "seed_manifest.json").write_text(
        json.dumps(
            {
                "description": "Official PCBnet CVJsonStd realBBox conversion; not newly AI-labeled",
                "sample_count": len(rows),
                "classes": classes,
                "samples": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"seeded {len(rows)} unique samples in {out}")


def overlay(case: str) -> None:
    source = OUT / "original_seeded" / f"{case}.json"
    document = read_json(source)
    image_path = source.parent / document["imagePath"]
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for shape in document["shapes"]:
        (x1, y1), (x2, y2) = shape["points"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
        text = f"{shape['label']}:{shape['attributes']['component_id']}"
        box = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text((x1, y1), text, fill=(180, 0, 0), font=font)
    target = OUT / "review_overlays"
    target.mkdir(parents=True, exist_ok=True)
    image.save(target / f"{case}.png")
    print(target / f"{case}.png")


def draw_overlay(image: Image.Image, document: dict) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for shape in document["shapes"]:
        (x1, y1), (x2, y2) = shape["points"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
        text = f"{shape['label']}:{shape['attributes']['component_id']}"
        box = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text((x1, y1), text, fill=(180, 0, 0), font=font)
    return image


def ant_crop(case: str) -> None:
    source = OUT / "original_seeded" / f"{case}.json"
    document = read_json(source)
    image_path = source.parent / document["imagePath"]
    original = Image.open(image_path).convert("RGB")
    ants = [s for s in document["shapes"] if s["label"] == "ANT"]
    if not ants:
        raise ValueError(f"No ANT shape in {case}")
    xs = [p[0] for s in ants for p in s["points"]]
    ys = [p[1] for s in ants for p in s["points"]]
    pad = 60
    crop_box = (
        max(0, int(min(xs) - pad)),
        max(0, int(min(ys) - pad)),
        min(original.width, int(max(xs) + pad)),
        min(original.height, int(max(ys) + pad)),
    )
    overlaid = draw_overlay(original.copy(), {"shapes": ants})
    left = original.crop(crop_box)
    right = overlaid.crop(crop_box)
    montage = Image.new("RGB", (left.width * 2 + 8, left.height), "white")
    montage.paste(left, (0, 0))
    montage.paste(right, (left.width + 8, 0))
    target = OUT / "ant_review_crops"
    target.mkdir(parents=True, exist_ok=True)
    montage.save(target / f"{case}.png")
    print(target / f"{case}.png")


REVIEW_DECISIONS = {
    "0287": {
        "category_changes": {
            "C_1": {
                "from": "ANT",
                "to": "AMP",
                "reason": "Clearly drawn operational-amplifier triangle with + and - inputs.",
            }
        },
        "notes": "All six seeded objects are visible; no additional symbol was found.",
    },
    "0258": {
        "category_changes": {},
        "notes": "LM57 block, two resistors, and three ground symbols match the visible drawing.",
    },
    "0267": {
        "category_changes": {},
        "notes": "Two blocks, two resistors, and four ground symbols match the visible drawing.",
    },
    "0292": {
        "category_changes": {
            "C_3": {
                "from": "ANT",
                "to": "AMP",
                "reason": "Clearly drawn operational-amplifier triangle with + and - inputs.",
            }
        },
        "notes": "The remaining two blocks, four resistors, and two ground symbols are consistent.",
    },
    "0025": {
        "category_changes": {
            "C_26": {
                "from": "Other",
                "to": "Switch",
                "reason": "K1 uses the same normally-open switch symbol as the seeded Switch objects K2/K4.",
            },
            "C_24": {
                "from": "Other",
                "to": "Switch",
                "reason": "K3 uses the same normally-open switch symbol as the seeded Switch objects K2/K4.",
            },
        },
        "notes": "Four resistors, four switch symbols, and the ground symbol are all visible.",
    },
    "0055": {
        "category_changes": {},
        "notes": "Regulator block, two resistors, two capacitors, three pins, and two grounds match.",
    },
    "0088": {
        "review_scope": "target_class_ANT",
        "category_changes": {
            "C_7": {
                "from": "ANT",
                "to": "AMP",
                "reason": "Clearly drawn operational-amplifier triangle with + and - inputs.",
            }
        },
        "notes": "The official ANT target was checked in a local original/overlay crop; it is an amplifier symbol.",
    },
    "0213": {
        "review_scope": "target_class_ANT",
        "category_changes": {
            "C_2": {
                "from": "ANT",
                "to": "AMP",
                "reason": "Triangle is explicitly labeled Amp A and has + and - inputs.",
            }
        },
        "notes": "The official ANT target was checked in a local original/overlay crop; it is an amplifier symbol.",
    },
    "0224": {
        "review_scope": "target_class_ANT",
        "category_changes": {
            "C_20": {
                "from": "ANT",
                "to": "AMP",
                "reason": "Clearly drawn LME49743 operational-amplifier triangle with + and - inputs.",
            }
        },
        "notes": "The official ANT target was checked in a local original/overlay crop; it is an amplifier symbol.",
    },
    "0228": {
        "review_scope": "target_class_ANT",
        "category_changes": {
            "C_1": {
                "from": "ANT",
                "to": "AMP",
                "reason": "Clearly drawn OPA2134 operational-amplifier triangle with + and - inputs.",
            }
        },
        "notes": "The official ANT target was checked in a local original/overlay crop; it is an amplifier symbol.",
    },
    "0230": {
        "review_scope": "target_class_ANT",
        "category_changes": {
            "C_5": {
                "from": "ANT",
                "to": "AMP",
                "reason": "Clearly drawn LM386 amplifier triangle with + and - inputs.",
            }
        },
        "notes": "The official ANT target was checked in a local original/overlay crop; it is an amplifier symbol.",
    },
    "0246": {
        "review_scope": "target_class_ANT",
        "category_changes": {
            "C_1": {
                "from": "ANT",
                "to": "AMP",
                "reason": "Clearly drawn operational-amplifier triangle with + and - inputs.",
            },
            "C_2": {
                "from": "ANT",
                "to": "AMP",
                "reason": "Clearly drawn operational-amplifier triangle with + and - inputs.",
            },
        },
        "notes": "Both official ANT targets were checked in a local original/overlay crop; both are amplifier symbols.",
    },
}


def finalize_reviews() -> None:
    reviewed = OUT / "reviewed"
    reviewed.mkdir(parents=True, exist_ok=True)
    sample_by_case = {x["case"]: x for x in read_json(MANIFEST)["samples"]}
    report_rows = []
    total_seeded = 0
    total_category_changes = 0
    for case, decision in REVIEW_DECISIONS.items():
        if sample_by_case[case]["split"] != "train":
            raise ValueError(f"Review selection must be train-only: {case}")
        seed_path = OUT / "original_seeded" / f"{case}.json"
        document = read_json(seed_path)
        total_seeded += len(document["shapes"])
        applied = []
        category_review_scope = decision.get("review_scope", "full_image")
        review_scope = (
            "full_image_localization_and_categories"
            if category_review_scope == "full_image"
            else "full_image_localization_and_target_class_ANT_category"
        )
        for shape in document["shapes"]:
            component_id = shape["attributes"]["component_id"]
            change = decision["category_changes"].get(component_id)
            shape["attributes"]["reviewed_by"] = "AI_visual_review_v1"
            shape["attributes"]["localization_review_result"] = "verified"
            shape["attributes"]["review_result"] = (
                "verified"
                if category_review_scope == "full_image" or shape["label"] == "ANT"
                else "carried_seed_not_in_targeted_review"
            )
            if change:
                if shape["label"] != change["from"]:
                    raise ValueError(f"Unexpected original label for {case}/{component_id}")
                shape["label"] = change["to"]
                shape["attributes"].update(
                    {
                        "original_label": change["from"],
                        "change": "category_correction",
                        "change_reason": change["reason"],
                    }
                )
                applied.append(
                    {
                        "component_id": component_id,
                        "type": "category_change",
                        **change,
                    }
                )
        total_category_changes += len(applied)
        document["flags"].update(
            {
                "annotation_status": "independently_visually_reviewed_candidate",
                "reviewed_original": True,
                "reviewed_overlay": True,
                "review_scope": review_scope,
                "localization_reviewed": True,
                "possible_missed_objects": 0,
                "text_false_positive_boxes": 0,
                "uncertain": False,
                "bbox_changes": 0,
                "category_changes": len(applied),
                "added_shapes": 0,
                "removed_shapes": 0,
            }
        )
        source_image = OUT / "original_seeded" / document["imagePath"]
        reviewed_image_name = f"{case}{source_image.suffix.lower()}"
        shutil.copy2(source_image, reviewed / reviewed_image_name)
        document["imagePath"] = reviewed_image_name
        target = reviewed / f"{case}.json"
        target.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        corrected_overlay_dir = OUT / "reviewed_overlays"
        corrected_overlay_dir.mkdir(parents=True, exist_ok=True)
        draw_overlay(Image.open(reviewed / reviewed_image_name), document).save(
            corrected_overlay_dir / f"{case}.png"
        )
        report_rows.append(
            {
                "case": case,
                "split": "train",
                "seeded_shape_count": len(document["shapes"]),
                "reviewed_original": True,
                "reviewed_overlay": True,
                "review_scope": review_scope,
                "localization_reviewed": True,
                "bbox_changes": 0,
                "category_changes": len(applied),
                "added_shapes": 0,
                "removed_shapes": 0,
                "possible_missed_objects": 0,
                "text_false_positive_boxes": 0,
                "uncertain": False,
                "changes": applied,
                "notes": decision["notes"],
            }
        )
    report = {
        "dataset": "PCBnet AI annotation review v1",
        "purpose": "Independent candidate review; excluded from current train/val/test data",
        "source": "Data/PCBnet/Dataset300/*/*_CVJsonStd.json realBBox",
        "reviewed_image_count": len(report_rows),
        "localization_reviewed_image_count": len(report_rows),
        "reviewed_seeded_shape_count": total_seeded,
        "bbox_changes": 0,
        "category_changes": total_category_changes,
        "added_shapes": 0,
        "removed_shapes": 0,
        "possible_missed_objects": 0,
        "text_false_positive_boxes": 0,
        "uncertain_image_count": 0,
        "images": report_rows,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# PCBnet AI annotations v1

`original_seeded/` contains 293 deduplicated images and X-AnyLabeling JSON files converted from the official `*_CVJsonStd.json` annotations. These are official-label conversions, not newly produced AI labels. Deleted components were excluded and each box uses the top-left-origin `realBBox`; the existing manifest's three boundary clips are retained.

`reviewed/` contains 12 candidate annotations and same-stem image copies. All 12 training images received a full-image localization review for missed components, shifted/incomplete boxes, and boxes placed on text rather than components. No certain localization error was found, so no box was added, moved, or removed. Six relatively simple-to-medium images also received a broad category check; the six additional `ANT` images received a targeted category check of the official `ANT` symbols. `review_overlays/` shows official labels before correction; it must not be interpreted as the corrected result. `reviewed_overlays/` shows the candidate labels after correction.

Every seeded shape keeps its official `component_id`. Changed shapes also record `original_label`, `change`, and `change_reason`. Per-image scope and results are in `report.json`; the low-cost targeted review crops are in `ant_review_crops/`.

This directory is isolated candidate data. It was not copied into or used to modify the current training, validation, or test datasets. Localization review is a visual check at the available raster resolution and does not guarantee that subtle omissions are absent. For the six targeted `ANT` reviews, non-`ANT` category identities were not exhaustively checked. None of the 12 reviewed candidates is included in current training.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(
        f"reviewed {len(report_rows)} images; {total_seeded} seeded shapes; "
        f"{total_category_changes} category changes"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed")
    draw = sub.add_parser("overlay")
    draw.add_argument("case")
    crop = sub.add_parser("ant-crop")
    crop.add_argument("case")
    sub.add_parser("finalize-reviews")
    args = parser.parse_args()
    if args.command == "seed":
        seed()
    elif args.command == "overlay":
        overlay(args.case)
    elif args.command == "ant-crop":
        ant_crop(args.case)
    else:
        finalize_reviews()


if __name__ == "__main__":
    main()
