"""Duplicate groups over every schematic image source, and a group-level official split.

Reads fingerprints.json (scripts/fingerprint_sources.py) and the extracted
official GT; writes groups, threshold calibration, a contact sheet of
threshold-edge pairs, the forbidden-image manifest for a clean retrain, and the
split file. Read-only with respect to every dataset.

Usage: python -I scripts/build_group_split.py <component_lab dir> <audit dir> <split json>
"""
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

LAB, AUDIT, SPLIT = (Path(a).resolve() for a in sys.argv[1:4])
sys.path.insert(0, str(LAB / "src"))
from component_lab.eval.cases import case_id, source_tag  # noqa: E402
from component_lab.eval.dedup import (UnionFind, assign_partitions, hamming_matrix, label_multiset,  # noqa: E402
                                      multiset_jaccard)

SEED = 20261017
PERCEPTUAL_MAX = 12         # link if pHash <= T AND dHash <= T (64-bit each); see calibration
PERCEPTUAL_MAX_HF = 8       # tighter T when either image is an HF KiCad sheet (frame-dominated hashes)
LABEL_JACCARD_MIN = 0.55    # official cases: multiset Jaccard of non-∅ (key, value) pairs
LABEL_MIN_ITEMS = 25        # both cases need at least this many non-∅ components for rule (c)
# Visual check of every official pair with Jaccard >= 0.5 among larger designs (label_pairs_checked_*.png).
# Small designs share generic keys (R1 10k, C1 100nF), so size matters more than the Jaccard value.
LABEL_PAIR_VERDICTS = {
    ("0166-jlc", "0170-jlc"): (0.645, "same design family, revised board: link"),
    ("0017-Datasheet", "0018-Altium Designer"): (0.606, "same STM32 board redrawn: link"),
    ("0166-jlc", "0191-jlc"): (0.578, "same design family: link"),
    ("0170-jlc", "0191-jlc"): (0.554, "same design family: link"),
    ("0078-Datasheet", "0088-Datasheet"): (0.75, "different circuits, small generic designators: no link"),
    ("0097-other", "0099-other"): (0.652, "different motor drivers from one book: no link"),
    ("0048-Datasheet", "0097-other"): (0.576, "different: no link"),
    ("0008-Altium Designer", "0097-other"): (0.556, "different: no link"),
    ("0075-Datasheet", "0173-jlc"): (0.537, "different: no link"),
    ("0048-Datasheet", "0061-Datasheet"): (0.5, "different: no link"),
}
# HF KiCad sheets: the sheet frame dominates the hash, so sparse different sheets reach distance 2-4
# (hf_perceptual_pairs_checked.png). Over-merging HF images is harmless here: HF is train-only, and no
# HF image comes within the threshold of an official image.
TARGETS = {"test": 40, "dev": 20}
ROUND004 = "reviewed_versions/manual_round_004/yolo"
VARIANT = re.compile(r"^(?P<base>.+)_v\d+$")


def load():
    rows = [r for r in json.loads((AUDIT / "fingerprints.json").read_text(encoding="utf-8"))["images"] if "error" not in r]
    labels = {}
    for r in rows:
        if r["source"] == "official":
            split, stem = r["path"].split(":", 1)[1].split("/", 1)
            labels[r["id"]] = label_multiset(json.loads((LAB / "data/official_contest6" / split / f"{stem}.json")
                                                    .read_text(encoding="utf-8")))
    return rows, labels


def calibrate(rows, labels, reps, P, D):
    """Distance histograms for known-distinct official pairs and for all pairs."""
    index = {sha: i for i, sha in enumerate(reps)}
    by_id = {r["id"]: r for r in rows}
    official = [i for i in labels if i.startswith("official:train200")]
    negatives = []
    for a, b in itertools.combinations(official, 2):
        pa, pb = by_id[a]["pixel_sha256"], by_id[b]["pixel_sha256"]
        if pa != pb and multiset_jaccard(labels[a], labels[b]) < 0.1:
            negatives.append((int(P[index[pa], index[pb]]), int(D[index[pa], index[pb]])))
    negatives = np.array(negatives)
    upper = np.triu_indices(len(reps), 1)
    all_p, all_d = P[upper], D[upper]
    table = []
    for t in range(0, 21, 2):
        table.append({"threshold": t,
                      "official_distinct_pairs_linked": int(((negatives[:, 0] <= t) & (negatives[:, 1] <= t)).sum()),
                      "all_pairs_linked": int(((all_p <= t) & (all_d <= t)).sum())})
    return {"official_distinct_pairs": len(negatives),
            "official_distinct_min_max_distance": int(np.max(negatives, axis=1).min()),
            "table": table}


def main():
    rows, labels = load()
    ids = [r["id"] for r in rows]
    by_id = {r["id"]: r for r in rows}
    uf = UnionFind(ids)
    rule_merges = Counter()
    links = Counter()

    # (a) exact decoded pixels
    by_pixel = defaultdict(list)
    for r in rows:
        by_pixel[r["pixel_sha256"]].append(r["id"])
    for members in by_pixel.values():
        for other in members[1:]:
            links["exact_pixels"] += 1
            rule_merges["exact_pixels"] += uf.union(members[0], other)

    # (b) perceptual hashes on one representative per pixel hash
    reps = sorted(by_pixel)
    P = hamming_matrix([by_id[by_pixel[s][0]]["phash"] for s in reps])
    D = hamming_matrix([by_id[by_pixel[s][0]]["dhash"] for s in reps])
    calibration = calibrate(rows, labels, reps, P, D)
    upper = np.triu_indices(len(reps), 1)
    is_hf = np.array([any(by_id[m]["source"] == "hf_open_schematics" for m in by_pixel[s]) for s in reps])
    threshold = np.where(is_hf[upper[0]] | is_hf[upper[1]], PERCEPTUAL_MAX_HF, PERCEPTUAL_MAX)
    distance = np.maximum(P[upper], D[upper])
    close = np.where(distance <= threshold)[0]
    perceptual_pairs = []
    for s in close:
        a, b = reps[upper[0][s]], reps[upper[1][s]]
        links["perceptual"] += 1
        rule_merges["perceptual"] += uf.union(by_pixel[a][0], by_pixel[b][0])
        perceptual_pairs.append((int(P[upper[0][s], upper[1][s]]), int(D[upper[0][s], upper[1][s]]), a, b,
                                 int(threshold[s])))

    # (c) official label similarity (same design, possibly another renderer)
    label_pairs = []
    for a, b in itertools.combinations(sorted(labels), 2):
        la, lb = labels[a], labels[b]
        if sum(la.values()) >= LABEL_MIN_ITEMS and sum(lb.values()) >= LABEL_MIN_ITEMS:
            j = multiset_jaccard(la, lb)
            if j >= LABEL_JACCARD_MIN and by_id[a]["pixel_sha256"] != by_id[b]["pixel_sha256"]:
                links["label_jaccard"] += 1
                rule_merges["label_jaccard"] += uf.union(a, b)
                label_pairs.append((round(j, 3), a, b))

    # (d) declared variants inside one lab folder: 0001_v0 / 0001_v1
    variants = defaultdict(list)
    for r in rows:
        if r["source"] == "component_lab":
            stem = Path(r["path"]).stem
            match = VARIANT.match(stem)
            if match:
                variants[(str(Path(r["path"]).parent), match["base"])].append(r["id"])
    for members in variants.values():
        for other in members[1:]:
            links["lab_variant_name"] += 1
            rule_merges["lab_variant_name"] += uf.union(members[0], other)

    # ---------------------------------------------------------- groups
    raw_groups = uf.groups()
    ordered = sorted(raw_groups.values(), key=lambda m: m[0])
    groups = {}
    for n, members in enumerate(ordered, 1):
        gid = f"G{n:04d}"
        info = {"members": members, "pixel_sha256": sorted({by_id[m]["pixel_sha256"] for m in members}),
                "sources": dict(Counter(by_id[m]["source"] for m in members)),
                "official_train200": sorted({case_id(by_id[m]["path"].split("/", 1)[1]) for m in members
                                             if m.startswith("official:train200")}, key=int),
                "gt10": any(m.startswith("official:gt10") for m in members),
                "lab_roles": dict(Counter(f"{by_id[m]['dataset']}:{by_id[m]['role']}" for m in members
                                          if by_id[m]["source"] == "component_lab" and by_id[m]["role"] != "none"))}
        info["detector_train_any"] = any(k.endswith(":train") for k in info["lab_roles"])
        info["detector_val_any"] = any(k.endswith(":val") for k in info["lab_roles"])
        info["detector_test_any"] = any(k.endswith(":test") for k in info["lab_roles"])
        info["round004_train"] = f"{ROUND004}:train" in info["lab_roles"]
        groups[gid] = info
    size_hist = Counter(len(g["members"]) for g in groups.values())
    unique_hist = Counter(len(g["pixel_sha256"]) for g in groups.values())

    # ---------------------------------------------------------- eligibility
    stems = {case_id(r["path"].split("/", 1)[1]): r["path"].split("/", 1)[1]
             for r in rows if r["id"].startswith("official:train200")}
    official_groups = {gid: g for gid, g in groups.items() if g["official_train200"]}
    count = lambda pred: sum(len(g["official_train200"]) for g in official_groups.values() if pred(g))
    eligibility = {
        "official_train200_cases": count(lambda g: True),
        "in_groups_with_gt10": count(lambda g: g["gt10"]),
        "not_in_any_detector_train_folder": count(lambda g: not g["gt10"] and not g["detector_train_any"]),
        "not_in_round004_train": count(lambda g: not g["gt10"] and not g["round004_train"]),
        "never_in_any_detector_train_or_val": count(lambda g: not g["gt10"] and not g["detector_train_any"]
                                                    and not g["detector_val_any"]),
        "groups_with_multiple_official_cases": sum(len(g["official_train200"]) > 1 for g in official_groups.values()),
    }

    # ---------------------------------------------------------- split
    candidates = {}
    for gid, g in official_groups.items():
        if g["gt10"]:
            continue  # GT10 groups are always train (format smoke test only)
        priority = 2 if g["detector_train_any"] else 1 if g["detector_val_any"] else 0
        candidates[gid] = {"cases": g["official_train200"], "priority": priority,
                           "tags": Counter(source_tag(stems[c]) for c in g["official_train200"])}
    chosen = assign_partitions(candidates, TARGETS, SEED)
    partition_of = {gid: p for p, gids in chosen.items() for gid in gids}

    def case_row(case, gid):
        g = groups[gid]
        return {"case": stems[case].split("-", 1)[0], "stem": stems[case], "tag": source_tag(stems[case]), "group": gid,
                "pixel_sha256": next(by_id[m]["pixel_sha256"] for m in g["members"]
                                     if m == f"official:train200:{stems[case]}"),
                "round004_trained_on_it": g["round004_train"], "any_detector_trained_on_it": g["detector_train_any"],
                "any_detector_validated_on_it": g["detector_val_any"]}

    partitions = {"test": [], "dev": [], "train": []}
    for gid, g in official_groups.items():
        for case in g["official_train200"]:
            partitions[partition_of.get(gid, "train")].append(case_row(case, gid))
    for rows_ in partitions.values():
        rows_.sort(key=lambda r: r["case"])
    held_groups = {gid for gid, p in partition_of.items()}

    # ---------------------------------------------------------- forbidden manifest
    forbidden = {"held_out_groups": sorted(held_groups), "images": [], "resnet_crops": {}}
    file_to_group = {}
    for gid in held_groups:
        for m in groups[gid]["members"]:
            r = by_id[m]
            file_to_group[r["file_sha256"]] = gid
            if r["source"] != "official":
                forbidden["images"].append({"id": r["id"], "path": r["path"], "source": r["source"], "dataset": r["dataset"],
                                            "role": r["role"], "group": gid, "partition": partition_of[gid]})
    for manifest in sorted((LAB / "data/resnet").glob("*/manifest.jsonl")):
        hits = Counter()
        for line in manifest.read_text(encoding="utf-8").splitlines():
            crop = json.loads(line)
            gid = file_to_group.get(crop.get("source_image_sha256"))
            if gid:
                hits[f"{crop.get('split')}:{partition_of[gid]}"] += 1
        forbidden["resnet_crops"][str(manifest.relative_to(LAB)).replace("\\", "/")] = dict(hits)
    forbidden["summary"] = {
        "images": len(forbidden["images"]),
        "by_dataset_role": dict(Counter(f"{i['source']}|{i['dataset']}:{i['role']}" for i in forbidden["images"])),
        "detector_train_images_to_drop": sum(i["role"] == "train" for i in forbidden["images"]),
    }

    # ---------------------------------------------------------- contact sheet near the threshold
    edge = [p for p in perceptual_pairs if max(p[0], p[1]) >= p[4] - 2]
    unlinked_edge = []
    for s in np.where((distance > threshold) & (distance <= threshold + 3))[0]:
        a, b = reps[upper[0][s]], reps[upper[1][s]]
        unlinked_edge.append((int(P[upper[0][s], upper[1][s]]), int(D[upper[0][s], upper[1][s]]), a, b, int(threshold[s])))
    def has_hf(pair):
        return bool(is_hf[reps.index(pair[2])] or is_hf[reps.index(pair[3])])
    by_distance = lambda pairs: sorted(pairs, key=lambda p: (-max(p[0], p[1]), p[2], p[3]))
    sample = (by_distance(p for p in edge if not has_hf(p))[:4] + by_distance(p for p in edge if has_hf(p))[:1]
              + sorted((p for p in unlinked_edge if not has_hf(p)), key=lambda p: (max(p[0], p[1]), p[2]))[:4]
              + sorted((p for p in unlinked_edge if has_hf(p)), key=lambda p: (max(p[0], p[1]), p[2]))[:1])
    tiles = []
    for pd, dd, a, b, limit in sample:
        images = []
        for sha in (a, b):
            r = by_id[by_pixel[sha][0]]
            if r["source"] == "official":
                import io, zipfile
                with zipfile.ZipFile(LAB.parent / "赛题六公开数据集.zip") as z:
                    split, stem = r["path"].split(":", 1)[1].split("/", 1)
                    folder = "/200_train_cases/" if split == "train200" else "/10_GTcase/"
                    name = next(n for n in z.namelist() if folder in n and n.rsplit("/", 1)[1].rsplit(".", 1)[0] == stem
                                and n.lower().endswith((".png", ".jpg", ".jpeg")))
                    image = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            else:
                image = Image.open(LAB.parent / r["path"]).convert("RGB")
            image.thumbnail((700, 480))
            images.append((image, r["id"]))
        tile = Image.new("RGB", (1420, 520), "white")
        draw = ImageDraw.Draw(tile)
        for x, (image, label) in zip((0, 720), images):
            tile.paste(image, (x, 36))
            draw.text((x + 4, 20), label[:110], fill=(0, 0, 160))
        verdict = "LINKED" if max(pd, dd) <= limit else "not linked"
        draw.text((4, 4), f"pHash {pd}  dHash {dd}  -> {verdict} (threshold {limit})", fill=(200, 0, 0))
        tiles.append(tile)
    if tiles:
        sheet = Image.new("RGB", (1420, 520 * len(tiles)), "white")
        for n, tile in enumerate(tiles):
            sheet.paste(tile, (0, 520 * n))
        sheet.save(AUDIT / "contact_sheet_threshold_edge.png")

    # ---------------------------------------------------------- write
    rules = {"exact_pixels": "identical decoded RGB (sha256 over WxH:RGB: + pixels)",
             "perceptual": (f"pHash <= T AND dHash <= T (64-bit, grayscale cropped to ink bbox); T = {PERCEPTUAL_MAX}, "
                            f"or {PERCEPTUAL_MAX_HF} when either image is an HF open_schematics sheet"),
             "label_jaccard": f"official cases only: multiset Jaccard of non-∅ (key, value) >= {LABEL_JACCARD_MIN}, "
                              f"both with >= {LABEL_MIN_ITEMS} non-∅ components",
             "lab_variant_name": "component_lab files <base>_v<k> in the same folder"}
    stats = {"images": len(rows), "unique_pixels": len(by_pixel), "groups": len(groups),
             "group_size_histogram_images": dict(sorted(size_hist.items())),
             "group_size_histogram_unique_pixels": dict(sorted(unique_hist.items())),
             "largest_groups": sorted(((len(g["pixel_sha256"]), gid, g["sources"]) for gid, g in groups.items()),
                                      reverse=True)[:5],
             "links": dict(links), "merges_added_by_rule": dict(rule_merges),
             "label_jaccard_pairs": label_pairs, "eligibility": eligibility, "rules": rules,
             "label_pair_visual_verdicts": [{"pair": list(k), "jaccard": v[0], "verdict": v[1]}
                                            for k, v in LABEL_PAIR_VERDICTS.items()],
             "perceptual_note": ("13,905 visually distinct official pairs (label Jaccard < 0.1) never come within "
                                 "pHash&dHash 14, so T=12 keeps a margin; annotation overlays of the same image sit "
                                 "at 9-12. HF KiCad sheets are frame-dominated and over-merge from distance 2-4, so "
                                 "HF pairs use T=8 (HF-only groups are train-only, so over-merging is conservative)"),
             "calibration": calibration}
    (AUDIT / "groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=0), encoding="utf-8")
    (AUDIT / "group_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    (AUDIT / "forbidden_for_clean_retrain.json").write_text(json.dumps(forbidden, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")
    split = {
        "name": "contest6_group_split_v1",
        "description": ("Duplicate-group split of the official 200_train_cases for a CLEAN RETRAIN. test is sealed "
                        "(score once per frozen system); dev is for model selection; train is everything else. "
                        "Every image in a test/dev group, from any source, must be removed from training "
                        "(see forbidden manifest). Models trained on the current detector datasets are "
                        "contaminated on this split; flags per case say how."),
        "seed": SEED, "targets": TARGETS, "rules": rules,
        "gt10_policy": "every group containing a 10_GTcase image is train (GT10 duplicates train200; format smoke test only)",
        "test_priority": "within each tag, groups never in any detector train/val folder first, then val-only, then train",
        "forbidden_manifest": str((AUDIT / "forbidden_for_clean_retrain.json").relative_to(LAB)).replace("\\", "/"),
        "groups_file": str((AUDIT / "groups.json").relative_to(LAB)).replace("\\", "/"),
        "sizes": {p: len(v) for p, v in partitions.items()},
        "partitions": partitions,
    }
    SPLIT.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("images", "unique_pixels", "groups", "links", "merges_added_by_rule",
                                            "group_size_histogram_unique_pixels", "eligibility")}, indent=1))
    print(json.dumps({"sizes": split["sizes"], "forbidden": forbidden["summary"],
                      "resnet": forbidden["resnet_crops"], "calibration": calibration}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
