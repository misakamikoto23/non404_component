"""Prepare both tasks directly from the official zip; never extract arbitrary paths."""
import hashlib
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path
from PIL import Image
from .geometry import convert_bbox, crop_box


def prepare(archive, output, origin, seed=42):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; use a new directory to preserve previous splits.")
    rows, seen, duplicates = [], {}, []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        annotations = sorted(n for n in names if "/200_train_cases/" in n and n.endswith(".json"))
        if not annotations:
            raise ValueError("No official 200_train_cases annotations found in archive.")
        for annotation in annotations:
            parent = annotation.rsplit("/", 1)[0]
            images = [n for n in names if n.rsplit("/", 1)[0] == parent and n.lower().endswith((".png", ".jpg", ".jpeg"))]
            if len(images) != 1:
                raise ValueError(f"Expected one image: {parent}")
            raw = z.read(images[0])
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGB")
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            case = parent.rsplit("/", 1)[-1]
            if digest in seen:
                duplicates.append({"case": case, "same_as": seen[digest]})
                continue
            seen[digest] = case
            components = json.loads(z.read(annotation))["components"]
            labels = []
            for key, item in components.items():
                labels.append({"source_id": key, "type": str(item["type"]),
                               "Name": item.get("Name"), "value": item.get("value"),
                               "bbox": convert_bbox(item["bbox"], *image.size, origin)})
            rows.append({"case": case, "source": images[0], "sha256": digest, "labels": labels, "image": image})
    random.Random(seed).shuffle(rows)
    n = len(rows)
    if n < 10:
        raise ValueError("Need at least ten distinct diagrams for train/val/test.")
    val_n = test_n = max(1, round(n * 0.1))
    classes = sorted({label["type"] for row in rows for label in row["labels"]})
    mapping = {name: i for i, name in enumerate(classes)}
    counts, cases, manifest = {}, {}, []
    for index, row in enumerate(rows):
        split = "test" if index < test_n else "val" if index < test_n+val_n else "train"
        cases.setdefault(split, []).append(row["case"])
        counts.setdefault(split, Counter()).update(label["type"] for label in row["labels"])
        for name in classes:
            (output / "classification" / split / name).mkdir(parents=True, exist_ok=True)
        image_path = output / "detection/images" / split / (row["case"] + ".png")
        label_path = output / "detection/labels" / split / (row["case"] + ".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        row["image"].save(image_path)
        lines = []
        w, h = row["image"].size
        for j, label in enumerate(row["labels"]):
            x1, y1, x2, y2 = label["bbox"]
            lines.append(f"{mapping[label['type']]} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
            crop_box(row["image"], label["bbox"]).save(output / "classification" / split / label["type"] / f"{row['case']}_{j:04d}.png")
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.append({k: v for k, v in row.items() if k != "image"} | {"split": split})
    yaml = f"path: {json.dumps(str((output/'detection').resolve()), ensure_ascii=False)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {json.dumps(classes, ensure_ascii=False)}\n"
    (output / "detection/dataset.yaml").write_text(yaml, encoding="utf-8")
    report = {"seed": seed, "source_origin": origin, "output_origin": "top-left", "classes": classes,
              "splits": cases, "counts": counts, "duplicates_removed": duplicates,
              "warnings": ["Split is random by distinct diagram, not stratified; review rare classes before training.",
                           "10_GTcase is intentionally excluded because it overlaps the training archive."],
              "samples": manifest}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"diagrams": n, "splits": {k: len(v) for k, v in cases.items()}, "classes": len(classes),
            "components": sum(sum(c.values()) for c in counts.values()), "duplicates": len(duplicates)}
