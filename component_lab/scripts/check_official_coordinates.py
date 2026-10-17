"""Read-only check of the official GT coordinate frame against the images.

For every official case: count coordinates outside [0,W]x[0,H], and compare how
much ink (dark pixels) the GT boxes cover when read as bottom-left (spec) versus
top-left. Writes overlays for a few cases drawing GT boxes under both readings.

Usage: python -I scripts/check_official_coordinates.py <component_lab dir> <output dir>
"""
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

LAB = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
sys.path.insert(0, str(LAB / "src"))
from component_lab.eval.coords import flip_bbox, out_of_bounds  # noqa: E402

# Three ordinary cases plus two where the ink heuristic preferred top-left (text-heavy datasheet pages).
OVERLAY_CASES = ("0001-KiCad", "0003-Altium Designer", "0002-Datasheet", "0078-Datasheet", "0059-Datasheet")


def images_from_zip():
    images = {}
    with zipfile.ZipFile(LAB.parent / "赛题六公开数据集.zip") as z:
        for name in z.namelist():
            if name.lower().endswith((".png", ".jpg", ".jpeg")) and ("/200_train_cases/" in name or "/10_GTcase/" in name):
                split = "train200" if "/200_train_cases/" in name else "gt10"
                stem = name.rsplit("/", 1)[1].rsplit(".", 1)[0]
                images[(split, stem)] = z.read(name)
    return images


def ink_fraction(dark, box_tl):
    x1, y1, x2, y2 = (int(round(v)) for v in box_tl)
    x1, x2 = max(0, x1), min(dark.shape[1], x2)
    y1, y2 = max(0, y1), min(dark.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0
    region = dark[y1:y2, x1:x2]
    return float(region.mean()), region.size


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = images_from_zip()
    rows = []
    for split in ("train200", "gt10"):
        for path in sorted((LAB / "data/official_contest6" / split).glob("*.json")):
            gt = json.loads(path.read_text(encoding="utf-8"))
            with Image.open(io.BytesIO(images[(split, path.stem)])) as image:
                image = image.convert("RGB")
            width, height = image.size
            dark = (np.asarray(image.convert("L")) < 128)
            ink = {"bottom_left": [0.0, 0], "top_left": [0.0, 0]}
            for item in gt["components"].values():
                for frame, box in (("bottom_left", flip_bbox(item["bbox"], height)), ("top_left", item["bbox"])):
                    fraction, area = ink_fraction(dark, box)
                    ink[frame][0] += fraction * area
                    ink[frame][1] += area
            ratio = {frame: (value / area if area else 0.0) for frame, (value, area) in ink.items()}
            bad = out_of_bounds(gt, width, height)
            rows.append({"split": split, "case": path.stem, "width": width, "height": height,
                         "out_of_bounds": len(bad), "out_of_bounds_examples": bad[:5],
                         "ink_in_boxes_bottom_left": round(ratio["bottom_left"], 4),
                         "ink_in_boxes_top_left": round(ratio["top_left"], 4),
                         "page_ink": round(float(dark.mean()), 4)})
            if split == "train200" and path.stem in OVERLAY_CASES:
                canvas = Image.new("RGB", (width * 2 + 20, height + 40), "white")
                for offset, frame, colour in ((0, "bottom_left", (0, 160, 0)), (width + 20, "top_left", (220, 0, 0))):
                    canvas.paste(image, (offset, 40))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((offset + 5, 5), f"{path.stem}: GT bbox read as {frame.replace('_', '-')} origin", fill=colour)
                    for item in gt["components"].values():
                        box = flip_bbox(item["bbox"], height) if frame == "bottom_left" else item["bbox"]
                        draw.rectangle([box[0] + offset, box[1] + 40, box[2] + offset, box[3] + 40], outline=colour, width=max(2, width // 600))
                canvas.save(OUT / f"overlay_{path.stem.replace(' ', '_')}.png")
    better = sum(r["ink_in_boxes_bottom_left"] > r["ink_in_boxes_top_left"] for r in rows)
    summary = {"cases": len(rows), "cases_with_out_of_bounds": sum(r["out_of_bounds"] > 0 for r in rows),
               "cases_where_bottom_left_boxes_cover_more_ink": better,
               "median_ink_bottom_left": float(np.median([r["ink_in_boxes_bottom_left"] for r in rows])),
               "median_ink_top_left": float(np.median([r["ink_in_boxes_top_left"] for r in rows])),
               "median_page_ink": float(np.median([r["page_ink"] for r in rows])),
               "overlays": [f"overlay_{c.replace(' ', '_')}.png" for c in OVERLAY_CASES]}
    (OUT / "coordinate_check.json").write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
