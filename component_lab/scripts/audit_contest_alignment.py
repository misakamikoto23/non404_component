"""Read-only audit of contest labels and overlap with existing detector training."""
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import zipfile
from PIL import Image

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "outputs/contest_alignment_audit_20261017"


def pixel_sha(image):
    image = image.convert("RGB")
    digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def summarize(records):
    components = [(record, key, item) for record in records for key, item in record["annotation"]["components"].items()]
    anonymous = [(record, key, item) for record, key, item in components if key.startswith("∅")]
    keyed = [(record, key, item) for record, key, item in components if not key.startswith("∅")]
    counts = defaultdict(Counter)
    for _, key, item in keyed:
        prefix = re.match(r"[A-Za-z]+", key)
        counts[prefix[0].upper() if prefix else "<other>"][item["type"]] += 1
    majority = {prefix: counter.most_common(1)[0][0] for prefix, counter in counts.items()}
    fit_correct = sum(counter[majority[prefix]] for prefix, counter in counts.items())
    values = [item.get("value") for _, _, item in components if item["type"].lower() in {"r", "c", "resistor", "capacitor"}]
    ic = [(key, item) for _, key, item in components if item["type"].lower() in {"box", "ic"}]
    pin_count = high_pin_count = 0
    high_examples = []
    generated_names = 0
    high_pin_types = Counter()
    passive_orientation = Counter()
    net_members = anonymous_members = net_count = singleton = 0
    for record in records:
        annotation = record["annotation"]
        for key, pins in annotation["pins"].items():
            if key.startswith("∅"):
                continue
            component_type = annotation["components"].get(key, {}).get("type")
            if component_type in {"r", "c", "l"} and "pin_1" in pins and "pin_2" in pins:
                first, second = pins["pin_1"]["point"], pins["pin_2"]["point"]
                dx, dy = second["x"]-first["x"], second["y"]-first["y"]
                direction = ("left" if dx > 0 else "right") if abs(dx) > abs(dy) else ("top" if dy < 0 else "bottom")
                passive_orientation[f"{component_type}_pin1_{direction}"] += 1
            for number, pin in pins.items():
                pin_count += 1
                if re.fullmatch(r"P_\d+", str(pin.get("pinname"))):
                    generated_names += 1
                suffix = number.removeprefix("pin_")
                if suffix.isdigit() and int(suffix) > 40:
                    high_pin_count += 1
                    high_pin_types[str(component_type)] += 1
                    if len(high_examples) < 16:
                        high_examples.append({"case": record["case"], "component": key,
                            "type": annotation["components"].get(key, {}).get("type"), "pin": number,
                            "pinname": pin.get("pinname"), "pins_on_component": len(pins)})
        for net in annotation["nets"].values():
            members = net.get("hyperGraph", "").strip("()").split(",")
            members = [value for value in members if value]
            net_count += 1
            singleton += len(members) == 1
            net_members += len(members)
            anonymous_members += sum(member.startswith("∅") for member in members)
    return {"cases": len(records), "components": len(components), "anonymous_keys": len(anonymous),
            "named_keys": len(keyed), "anonymous_fraction": len(anonymous)/len(components),
            "types": dict(Counter(item["type"] for _, _, item in components)),
            "anonymous_types": dict(Counter(item["type"] for _, _, item in anonymous)),
            "prefix_type_counts": {prefix: dict(counter) for prefix, counter in sorted(counts.items())},
            "prefix_majority_resubstitution_accuracy": fit_correct / len(keyed),
            "prefix_accuracy_warning": "Mapping fitted and measured on same labels; not held-out accuracy or OCR pipeline accuracy",
            "bbox_short_side_median": statistics.median(min(item["bbox"][2]-item["bbox"][0],item["bbox"][3]-item["bbox"][1]) for _,_,item in components),
            "rc_values": {"count": len(values), "null": sum(value is None for value in values),
                          "null_fraction": sum(value is None for value in values)/len(values)},
            "ic_name_different_from_key": {"count": len(ic), "different": sum(item.get("Name") != key for key,item in ic)},
            "nonanonymous_pins": pin_count, "pin_suffix_above40": high_pin_count,
            "high_pin_number_types_not_proof_of_internal_ids": dict(high_pin_types),
            "passive_pin1_orientation_bottom_left_source_coordinates": dict(passive_orientation),
            "pinname_P_number": generated_names, "above40_examples_not_proof_of_internal_ids": high_examples,
            "nets": net_count, "singleton_nets": singleton,
            "net_members": net_members, "anonymous_net_members": anonymous_members}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = LAB.parent / "赛题六公开数据集.zip"
    records = []
    library = []
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        library_object = json.loads(z.read(next(name for name in names if name.endswith("/cpntLibrary"))))
        for name in names:
            if "cpntLibrary" in name:
                data = z.read(name) if not name.endswith("/") else b""
                library.append({"entry": name, "bytes": len(data), "leading_text": data[:500].decode("utf-8-sig",errors="replace")})
            if name.endswith(".json") and ("/200_train_cases/" in name or "/10_GTcase/" in name):
                parent = name.rsplit("/",1)[0]
                images = [p for p in names if p.rsplit("/",1)[0] == parent and p.lower().endswith((".png",".jpg",".jpeg"))]
                if len(images) != 1:
                    raise ValueError(parent)
                annotation = json.loads(z.read(name))
                with Image.open(io.BytesIO(z.read(images[0]))) as image:
                    digest = pixel_sha(image)
                records.append({"case": parent.rsplit("/",1)[-1], "split": "train" if "/200_train_cases/" in name else "gt10",
                                "annotation_file": name, "image_file": images[0], "pixel_sha256": digest, "annotation": annotation})
    existing = defaultdict(list)
    data = LAB / "data/reviewed_versions/manual_round_004/yolo"
    for split in ("train","val","test"):
        for path in (data / "images" / split).iterdir():
            if path.is_file() and path.suffix.lower() in {".png",".jpg",".jpeg"}:
                with Image.open(path) as image:
                    digest = pixel_sha(image)
                existing[digest].append({"case": path.stem, "split": split, "path": str(path)})
    official_train = [r for r in records if r["split"] == "train"]
    official_gt = [r for r in records if r["split"] == "gt10"]
    library_stats = {"entries": len(library_object), "entries_with_duplicate_pin_numbers": 0,
                     "entries_with_exact_duplicate_pin_rows": 0, "ic_Name_exact_key_hits": 0, "ic_count": 0}
    for entry in library_object.values():
        pins = entry.get("pins", [])
        numbers = [str(pin.get("number")) for pin in pins]
        serialized = [json.dumps(pin, sort_keys=True, ensure_ascii=True) for pin in pins]
        library_stats["entries_with_duplicate_pin_numbers"] += len(set(numbers)) < len(numbers)
        library_stats["entries_with_exact_duplicate_pin_rows"] += len(set(serialized)) < len(serialized)
    for record in official_train:
        for item in record["annotation"]["components"].values():
            if item["type"] in {"box", "ic"}:
                library_stats["ic_count"] += 1
                library_stats["ic_Name_exact_key_hits"] += item.get("Name") in library_object
    overlap = []
    for record in records:
        overlap.append({"case": record["case"], "official_split": record["split"], "official_image": record["image_file"],
                        "pixel_sha256": record["pixel_sha256"], "existing_matches": existing.get(record["pixel_sha256"],[])})
    summary = {"source": str(archive), "train200": summarize(official_train), "gt10": summarize(official_gt),
        "official_unique_pixels": len({r["pixel_sha256"] for r in official_train}),
        "gt10_exact_overlap_train200": sum(any(g["pixel_sha256"] == t["pixel_sha256"] for t in official_train) for g in official_gt),
        "official_train_cases_exactly_in_detector_training": sum(any(m["split"]=="train" for m in r["existing_matches"]) for r in overlap if r["official_split"]=="train"),
        "official_train_cases_exactly_in_detector_val": sum(any(m["split"]=="val" for m in r["existing_matches"]) for r in overlap if r["official_split"]=="train"),
        "official_train_cases_exactly_in_detector_test": sum(any(m["split"]=="test" for m in r["existing_matches"]) for r in overlap if r["official_split"]=="train"),
        "overlap_warning": "Decoded RGB exact overlap only; absence of exact match does not exclude rerender/crop/near-duplicate leakage",
        "library_entries": library, "library_quality_inventory": library_stats, "overlap": overlap, "read_only_audit": True,
        "scoring_warning": "Existing non404 accuracy_proxy is not an official organizer evaluator"}
    (OUT / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    console = {k:v for k,v in summary.items() if k not in {"overlap","library_entries","train200","gt10"}}
    for split in ("train200","gt10"):
        console[split] = {k:v for k,v in summary[split].items() if k not in {"types","anonymous_types","prefix_type_counts","above40_examples_not_proof_of_internal_ids"}}
    print(json.dumps(console, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
