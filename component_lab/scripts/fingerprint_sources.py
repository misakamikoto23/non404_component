"""Read-only: fingerprint every schematic image the detector used or could use.

Sources: official zip (200_train_cases + 10_GTcase), PCBnet Dataset300,
HF open_schematics samples, and full-page images under component_lab/data
(YOLO datasets, reviewed versions, review sources, experiments, prepared
detection, review/annotation folders). Crop datasets are skipped. A file-hash
cache makes reruns cheap.

Usage: python -I scripts/fingerprint_sources.py <component_lab dir> <output json>
"""
import json
from pathlib import Path
import sys
import zipfile

LAB = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
sys.path.insert(0, str(LAB / "src"))
from component_lab.eval.dedup import fingerprint  # noqa: E402

ROOT = LAB.parent
SUFFIXES = (".png", ".jpg", ".jpeg")
SPLITS = ("train", "val", "test", "pending")
SKIP_LAB_DIRS = {"component_crops", "resnet", "official_contest6"}


def lab_role(parts):
    """Detector role from a path like .../images/train/x.png."""
    for i, part in enumerate(parts[:-1]):
        if part == "images" and parts[i + 1] in SPLITS:
            return parts[i + 1]
    return "none"


def entries():
    with zipfile.ZipFile(ROOT / "赛题六公开数据集.zip") as z:
        for name in sorted(z.namelist()):
            if name.lower().endswith(SUFFIXES) and ("/200_train_cases/" in name or "/10_GTcase/" in name):
                split = "train200" if "/200_train_cases/" in name else "gt10"
                stem = name.rsplit("/", 1)[1].rsplit(".", 1)[0]
                yield {"id": f"official:{split}:{stem}", "source": "official", "dataset": split, "role": "official",
                       "path": f"zip:{split}/{stem}"}, (lambda n=name: z.read(n))
    for path in sorted((ROOT / "Data/PCBnet/Dataset300").glob("*/*")):
        if path.suffix.lower() in SUFFIXES and "__MACOSX" not in path.parts:
            yield {"id": f"pcbnet300:{path.stem}", "source": "pcbnet300", "dataset": "Dataset300", "role": "none",
                   "path": str(path.relative_to(ROOT)).replace("\\", "/")}, path.read_bytes
    for path in sorted((ROOT / "Data/open_schematics/samples").glob("*/schematic.png")):
        yield {"id": f"hf_os:{path.parent.name}", "source": "hf_open_schematics", "dataset": "samples", "role": "none",
               "path": str(path.relative_to(ROOT)).replace("\\", "/")}, path.read_bytes
    for path in sorted((LAB / "data").rglob("*")):
        rel = path.relative_to(LAB / "data")
        if path.suffix.lower() not in SUFFIXES or rel.parts[0] in SKIP_LAB_DIRS or "classification" in rel.parts:
            continue
        dataset = "/".join(rel.parts[: rel.parts.index("images")]) if "images" in rel.parts else str(rel.parent).replace("\\", "/")
        yield {"id": f"lab:{rel.as_posix()}", "source": "component_lab", "dataset": dataset,
               "role": lab_role(rel.parts), "path": f"component_lab/data/{rel.as_posix()}"}, path.read_bytes


def main():
    cache = {}
    if OUT.exists():
        for row in json.loads(OUT.read_text(encoding="utf-8"))["images"]:
            cache[row["file_sha256"]] = {k: row[k] for k in ("file_sha256", "pixel_sha256", "width", "height", "dhash", "phash")}
    import hashlib
    rows, skipped = [], 0
    for meta, read in entries():
        raw = read()
        key = hashlib.sha256(raw).hexdigest()
        if key not in cache:
            try:
                cache[key] = fingerprint(raw)
            except Exception as exc:  # unreadable image: record and move on
                rows.append({**meta, "error": str(exc)})
                continue
        fp = cache[key]
        if fp["width"] < 200 and fp["height"] < 200:
            skipped += 1
            continue
        rows.append({**meta, **fp})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"images": rows, "skipped_small": skipped}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps({"images": len(rows), "skipped_small": skipped,
                      "unique_pixels": len({r.get("pixel_sha256") for r in rows})}))


if __name__ == "__main__":
    main()
