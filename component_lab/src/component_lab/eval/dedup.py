"""Image fingerprints, duplicate grouping and group-level splitting.

Grouping links two images when any rule fires (exact decoded pixels,
perceptual hash within a calibrated threshold, official label similarity,
or a declared derivation such as ``0001_v1`` from ``0001``); groups are the
connected components (union-find). Splits are made per group, so no group
can sit on both sides of a train/test boundary.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import math

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ANON_PREFIX = "∅"


# --------------------------------------------------------------- fingerprints

def pixel_sha256(image):
    """Decoded-RGB hash, same convention as outputs/contest_alignment_audit_20261017."""
    image = image.convert("RGB")
    digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _content_gray(image, ink_threshold=200, margin=2):
    """Grayscale image cropped to its ink bounding box, so white padding does not move the hash."""
    gray = image.convert("L")
    small = gray.copy()
    small.thumbnail((1024, 1024))
    array = np.asarray(small)
    rows, cols = np.where(array < ink_threshold)
    if rows.size == 0:
        return gray
    sx, sy = gray.width / small.width, gray.height / small.height
    box = (max(0, int((cols.min() - margin) * sx)), max(0, int((rows.min() - margin) * sy)),
           min(gray.width, int(math.ceil((cols.max() + 1 + margin) * sx))),
           min(gray.height, int(math.ceil((rows.max() + 1 + margin) * sy))))
    return gray.crop(box)


def _bits_to_int(bits):
    value = 0
    for bit in np.asarray(bits, dtype=bool).ravel():
        value = (value << 1) | int(bit)
    return value


def dhash(gray, size=8):
    pixels = np.asarray(gray.resize((size + 1, size), Image.Resampling.BOX), dtype=np.float64)
    return _bits_to_int(pixels[:, 1:] > pixels[:, :-1])


_DCT = np.array([[math.cos(math.pi * (2 * n + 1) * k / 64) for n in range(32)] for k in range(32)])


def phash(gray):
    pixels = np.asarray(gray.resize((32, 32), Image.Resampling.BOX), dtype=np.float64)
    low = (_DCT @ pixels @ _DCT.T)[:8, :8]
    median = np.median(low.ravel()[1:])
    return _bits_to_int(low > median)


def fingerprint(raw):
    """Fingerprint of encoded image bytes: file hash, pixel hash, size and two 64-bit perceptual hashes."""
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        gray = _content_gray(image)
        return {"file_sha256": hashlib.sha256(raw).hexdigest(), "pixel_sha256": pixel_sha256(image),
                "width": image.width, "height": image.height,
                "dhash": f"{dhash(gray):016x}", "phash": f"{phash(gray):016x}"}


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def hamming_matrix(hex_hashes):
    """All-pairs Hamming distances of 64-bit hex hashes (numpy, uint8 result)."""
    values = np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)
    bytes_ = values.view(np.uint8).reshape(len(values), 8)
    bits = np.unpackbits(bytes_, axis=1).astype(bool)
    out = np.zeros((len(values), len(values)), dtype=np.uint8)
    for i in range(len(values)):
        out[i] = (bits[i] ^ bits).sum(axis=1)
    return out


# ------------------------------------------------------------------ grouping

class UnionFind:
    def __init__(self, items=()):
        self.parent = {item: item for item in items}

    def add(self, item):
        self.parent.setdefault(item, item)

    def find(self, item):
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra
        return ra != rb

    def groups(self):
        out = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return {root: sorted(members) for root, members in out.items()}


def label_multiset(case):
    """Multiset of non-∅ (key, value) pairs of an official case JSON."""
    return Counter((key, item.get("value")) for key, item in case["components"].items()
                   if not key.startswith(ANON_PREFIX))


def multiset_jaccard(a, b):
    union = sum((a | b).values())
    return sum((a & b).values()) / union if union else 0.0


# ------------------------------------------------------------------ splitting

def _rank(seed, text):
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def assign_partitions(candidates, targets, seed):
    """Pick whole groups for each partition, stratified by tag.

    ``candidates``: {group_id: {"cases": [...], "tags": Counter, "priority": int}}
    of groups eligible for held-out partitions. ``targets``: ordered
    {partition: n_cases}. Each partition takes groups in tag-stratified round
    robin until it reaches its case target; within a tag, lower ``priority``
    goes first, then a seeded hash order. A group is never split.
    Returns {partition: [group_id]}.
    """
    remaining = dict(candidates)
    out = {}
    for partition, target in targets.items():
        chosen, count = [], 0
        by_tag = {}
        for gid, info in remaining.items():
            tag = info["tags"].most_common(1)[0][0]
            by_tag.setdefault(tag, []).append(gid)
        total = sum(len(remaining[g]["cases"]) for g in remaining)
        quota = {tag: target * sum(len(remaining[g]["cases"]) for g in gids) / total if total else 0
                 for tag, gids in by_tag.items()}
        order = {tag: sorted(gids, key=lambda g: (remaining[g].get("priority", 0), _rank(f"{seed}:{partition}", g)))
                 for tag, gids in by_tag.items()}
        taken = Counter()
        while count < target and any(order.values()):
            # Next group from the tag furthest below its proportional quota.
            tag = min((t for t in order if order[t]), key=lambda t: (taken[t] - quota[t], t))
            gid = order[tag].pop(0)
            size = len(remaining[gid]["cases"])
            if count + size > target and count > 0:
                continue
            chosen.append(gid)
            taken[tag] += size
            count += size
        for gid in chosen:
            remaining.pop(gid)
        out[partition] = chosen
    return out
