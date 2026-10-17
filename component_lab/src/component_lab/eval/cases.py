"""Case discovery, official GT extraction and the fixed held-out split."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

MANIFEST = "_manifest.json"
_SUFFIXES = ("_target_named", "_target", "_prediction", "_pred")
_NUMERIC = re.compile(r"^(?:case[_-]?)?(\d+)(?:$|[-_ ])", re.IGNORECASE)


def case_id(stem):
    """Canonical case id: ``0001-KiCad_target_named``, ``0001`` and ``case_001`` all map to ``1``."""
    for suffix in _SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    match = _NUMERIC.match(stem)
    return str(int(match.group(1))) if match else stem


def source_tag(stem):
    """Source tag from an official file stem: ``0003-Altium Designer`` -> ``altium designer``."""
    for suffix in _SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.split("-", 1)[1].strip().lower() if "-" in stem else "unknown"


def discover(directory):
    """Map canonical case id -> JSON path for every case file below ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    found = {}
    for path in sorted(directory.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        key = case_id(path.stem)
        if key in found:
            raise ValueError(f"two files map to case {key!r}: {found[key]} and {path}")
        found[key] = path
    return found


def load_split(path, partition=None):
    """Case ids of a split file: a flat ``cases`` list, or ``partitions[partition]`` of a group split."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "partitions" in data:
        if partition not in data["partitions"]:
            raise ValueError(f"{path} is a group split; pick --partition from {sorted(data['partitions'])}")
        items = data["partitions"][partition]
    else:
        if partition:
            raise ValueError(f"{path} has no partitions; drop --partition")
        items = data["cases"]
    return [case_id(str(item["case"] if isinstance(item, dict) else item)) for item in items]


def _pixel_sha256(raw):
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode())
        digest.update(image.tobytes())
        return digest.hexdigest(), image.width, image.height


def extract_official(archive, output, with_images=False):
    """Copy official GT JSONs (and optionally images) out of the contest zip.

    Writes ``train200/<stem>.json`` and ``gt10/<stem>.json`` plus a manifest
    with source tags and decoded-pixel hashes. Only the two known folders are
    read and every output name is rebuilt from the case stem, so no archive
    path is ever trusted for writing.
    """
    output = Path(output)
    rows = []
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for name in sorted(names):
            if not name.endswith(".json"):
                continue
            split = "train200" if "/200_train_cases/" in name else "gt10" if "/10_GTcase/" in name else None
            if split is None:
                continue
            parent, file_name = name.rsplit("/", 1)
            stem = file_name[: -len(".json")]
            for suffix in _SUFFIXES:
                stem = stem.removesuffix(suffix)
            if not re.fullmatch(r"\d{4}-[A-Za-z ]+", stem):
                raise ValueError(f"unexpected case file name in archive: {file_name!r}")
            images = [n for n in names if n.rsplit("/", 1)[0] == parent and n.lower().endswith((".png", ".jpg", ".jpeg"))]
            if len(images) != 1:
                raise ValueError(f"expected one image next to {name}")
            payload = json.loads(z.read(name))
            target = output / split / f"{stem}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            raw = z.read(images[0])
            digest, width, height = _pixel_sha256(raw)
            if with_images:
                (output / split / (stem + Path(images[0]).suffix.lower())).write_bytes(raw)
            rows.append({"case": case_id(stem), "stem": stem, "split": split, "tag": source_tag(stem),
                         "pixel_sha256": digest, "width": width, "height": height,
                         "components": len(payload["components"]),
                         "anonymous_components": sum(k.startswith("∅") for k in payload["components"])})
    manifest = {"archive": str(Path(archive).name), "coordinate_origin": "bottom-left", "cases": rows}
    (output / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"output": str(output), "cases": Counter(r["split"] for r in rows)}


def _rank(seed, case):
    return hashlib.sha256(f"{seed}:{case}".encode()).hexdigest()


def make_split(manifest_path, size=30, seed=20261017, detector_audit=None):
    """Deterministic held-out split of official training cases, stratified by source tag.

    Cases whose decoded pixels duplicate another training case are never
    held out. With ``detector_audit`` (outputs/contest_alignment_audit_*/audit.json),
    cases already used to train the current detector are taken last.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    train = [r for r in manifest["cases"] if r["split"] == "train200"]
    pixel_counts = Counter(r["pixel_sha256"] for r in train)
    detector = {}
    if detector_audit:
        audit = json.loads(Path(detector_audit).read_text(encoding="utf-8"))
        for row in audit["overlap"]:
            if row["official_split"] == "train":
                splits = sorted({m["split"] for m in row["existing_matches"]})
                detector[case_id(row["case"])] = "/".join(splits) or "none"
    strata = defaultdict(list)
    for row in train:
        if pixel_counts[row["pixel_sha256"]] == 1:
            strata[row["tag"]].append(row)
    total = sum(len(rows) for rows in strata.values())
    # Largest-remainder allocation, at least one case per stratum.
    quota = {tag: size * len(rows) / total for tag, rows in strata.items()}
    allocation = {tag: max(1, int(q)) for tag, q in quota.items()}
    for tag in sorted(quota, key=lambda t: (-(quota[t] - allocation[t]), t)):
        if sum(allocation.values()) >= size:
            break
        allocation[tag] += 1
    chosen, summary = [], {}
    for tag in sorted(strata):
        rows = sorted(strata[tag], key=lambda r: ("train" in detector.get(r["case"], "").split("/"), _rank(seed, r["case"])))
        picked = rows[: allocation[tag]]
        summary[tag] = {"pool": len(rows), "selected": len(picked),
                        "selected_seen_by_detector_train": sum("train" in detector.get(r["case"], "").split("/") for r in picked)}
        chosen.extend({"case": r["stem"].split("-", 1)[0], "stem": r["stem"], "tag": tag,
                       "detector_split": detector.get(r["case"], "unknown")} for r in picked)
    chosen.sort(key=lambda r: r["case"])
    return {
        "name": "dev_diagnostic_contaminated",
        "description": ("LEGACY stratified split (superseded by scripts/build_group_split.py, which groups duplicates). "
                        "Fixed DEVELOPMENT split drawn from official 200_train_cases, for comparing pipeline versions. "
                        "Not an independent test: no official case is unseen (see detector_split), so a final "
                        "independent test needs separate data and must never be tuned against. Keep these cases out "
                        "of training. 10_GTcase overlaps 200_train_cases and is not independent either."),
        "detector_split_meaning": {
            "train": "used for gradient training of the detector",
            "val": "used for model selection (checkpoint, thresholds, pipeline settings)",
            "test": "used for reported evaluations; documented as not tuned against, but results were seen",
            "none": "not found in the detector datasets by exact decoded-pixel match",
        },
        "seed": seed, "size": len(chosen), "stratified_by": "source tag (file-name suffix, lower-cased)",
        "excluded": "train cases whose decoded pixels duplicate another train case",
        "detector_audit": str(detector_audit) if detector_audit else None,
        "strata": summary, "cases": chosen,
    }
