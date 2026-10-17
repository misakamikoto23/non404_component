"""Audit raw PCBnet categories and plan review using training data only.

The detector has one ``component`` class. Its original symbol categories do not
need equal counts for localization training. These statistics are useful for
review coverage and a future crop classifier; they do not measure model errors.
No function performs inference, training, file access, or sampling in PyTorch.
"""

from collections import Counter
from copy import deepcopy
import math
import random


SPLITS = ("train", "val", "test")


def build_category_report(manifest: dict) -> dict:
    """Count boxes and distinct source images per raw category and split.

    Each manifest sample represents one source image. A category contributes
    only once to that image's category-image count, regardless of its box count.
    Declared ``original_category_names`` are kept even when they have no boxes.
    Rarity always uses training source-image counts, including for held-out-only
    categories. A zero-training category is therefore ``very_rare``.
    """
    samples = manifest.get("samples", [])
    names = set(manifest.get("original_category_names", []))
    boxes = {split: Counter() for split in SPLITS}
    images = {split: Counter() for split in SPLITS}
    by_split = {split: {"images": 0, "boxes": 0} for split in SPLITS}
    for sample in samples:
        split = sample["split"]
        if split not in by_split:
            by_split[split] = {"images": 0, "boxes": 0}
            boxes[split], images[split] = Counter(), Counter()
        labels = sample.get("labels", [])
        categories = [label["category"] for label in labels]
        names.update(categories)
        by_split[split]["images"] += 1
        by_split[split]["boxes"] += len(labels)
        boxes[split].update(categories)
        images[split].update(set(categories))

    rows = []
    for name in sorted(names):
        row = {
            "category": name,
            "total_boxes": sum(counts[name] for counts in boxes.values()),
            "total_images": sum(counts[name] for counts in images.values()),
        }
        for split in SPLITS:
            row[f"{split}_boxes"] = boxes[split][name]
            row[f"{split}_images"] = images[split][name]
        train_images = row["train_images"]
        row["rarity_tier"] = (
            "very_rare" if train_images < 5 else "rare" if train_images < 15 else "common"
        )
        rows.append(row)

    return {
        "totals": {
            "images": len(samples),
            "boxes": sum(value["boxes"] for value in by_split.values()),
            "by_split": by_split,
        },
        "categories": rows,
        "rarity_basis": "train source images only: <5 very_rare; <15 rare; otherwise common",
        "detector_balance_note": (
            "Single-class component YOLO localization does not require all original "
            "categories to be equally frequent. Improve source diversity and review "
            "coverage without changing validation or test distributions."
        ),
    }


def _identity(sample: dict, index: int) -> tuple:
    """Prefer a source case identifier, with image and row fallbacks."""
    for field in ("case", "image", "sha256"):
        if sample.get(field) is not None:
            return field, str(sample[field])
    return "row", str(index)


def _quotas(count: int) -> list[int]:
    """Largest-remainder allocation in the default 8:6:6 proportions."""
    raw = [divmod(count * weight, 20) for weight in (8, 6, 6)]
    result = [whole for whole, _ in raw]
    order = sorted(range(3), key=lambda i: (-raw[i][1], i))
    for i in order[: count - sum(result)]:
        result[i] += 1
    return result


def _geometry(sample: dict) -> dict:
    """Ground-truth geometry, with smallness after a 1280px longest-side resize."""
    width, height = float(sample.get("width", 0)), float(sample.get("height", 0))
    longest = max(width, height)
    scale = min(1.0, 1280 / longest) if longest > 0 else 1.0
    valid_boxes, small = 0, 0
    for label in sample.get("labels", []):
        bbox = label.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0 or box_height <= 0:
            continue
        valid_boxes += 1
        if box_width * box_height * scale * scale <= 32 * 32:
            small += 1
    return {
        "small_target_ratio": small / valid_boxes if valid_boxes else 0.0,
        "box_count": len(sample.get("labels", [])),
        "image_area": max(0, width) * max(0, height),
    }


def select_review_batch(manifest: dict, count: int = 20, seed: int = 20261015) -> list[dict]:
    """Return copied, unique training samples for annotation review.

    Default groups are 8 random, 6 greedy rare-category coverage, and 6 geometry
    heuristic examples. Other counts use proportional largest-remainder quotas.
    Rarity and heuristic scaling use only eligible training samples; held-out
    labels, dimensions, metadata, and declared category totals cannot affect the
    result. Geometry is a review hint, never an observed false-negative rate.

    Output ``priority`` is the one-based presentation order, not a probability or
    estimated error rate. Duplicate case identifiers are collapsed (image/sha256
    identifiers are used only if no case is available).
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    unique = {}
    for index, sample in enumerate(manifest.get("samples", [])):
        if sample.get("split") == "train":
            unique.setdefault(_identity(sample, index), sample)
    candidates = [unique[key] for key in sorted(unique)]
    target = min(count, len(candidates))
    if not target:
        return []

    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Candidate indices are stable after seeded shuffling and provide tie breaks.
    remaining = list(range(len(candidates)))
    category_sets = [set(label["category"] for label in row.get("labels", [])) for row in candidates]
    train_images = Counter(category for names in category_sets for category in names)
    rare = {name for name, n in train_images.items() if n < 15}
    geometries = [_geometry(row) for row in candidates]
    max_boxes = max((g["box_count"] for g in geometries), default=1) or 1
    max_area = max((g["image_area"] for g in geometries), default=1) or 1
    quotas = _quotas(target)
    selected, covered = [], set()

    def add(index, group, reasons):
        row = deepcopy(candidates[index])
        row["selection_group"] = group
        row["selection_reasons"] = reasons
        row["priority"] = len(selected) + 1
        selected.append(row)
        remaining.remove(index)
        covered.update(category_sets[index] & rare)

    for index in remaining[: quotas[0]]:
        add(index, "random", [f"固定 seed={seed} 的训练集随机样本，用于观察常规情况。"])

    for _ in range(quotas[1]):
        eligible = [index for index in remaining if category_sets[index] & rare]
        if not eligible:
            break

        def coverage_score(index):
            categories = category_sets[index] & rare
            unseen = categories - covered
            return (
                len(unseen),
                sum(1 / train_images[name] for name in sorted(unseen)),
                len(categories),
                sum(1 / train_images[name] for name in sorted(categories)),
                -index,
            )

        index = max(eligible, key=coverage_score)
        categories = sorted(category_sets[index] & rare)
        unseen = sorted(set(categories) - covered)
        add(index, "rare_coverage", [
            "优先覆盖更多训练来源少于 15 张的原始类别。",
            "本图稀缺类别：" + ", ".join(f"{name}（{train_images[name]} 张训练图）" for name in categories),
            "新增稀缺覆盖：" + (", ".join(unseen) or "已有类别的其他训练来源"),
        ])

    def difficulty_score(index):
        values = geometries[index]
        return (
            0.5 * values["small_target_ratio"]
            + 0.3 * values["box_count"] / max_boxes
            + 0.2 * values["image_area"] / max_area,
            -index,
        )

    for index in sorted(remaining, key=difficulty_score, reverse=True)[: quotas[2]]:
        values, sample = geometries[index], candidates[index]
        add(index, "hard_geometry", [
            "训练 GT 几何启发，未运行模型，不能解释为真实漏检率。",
            f"小目标占比 {values['small_target_ratio']:.1%}（长边最多缩至 1280 后，面积 ≤32×32 像素）。",
            f"目标数 {values['box_count']}；原图 {sample.get('width', 0)}×{sample.get('height', 0)}。",
        ])

    # If no rare-category candidate remains, fill the unspent quota randomly.
    for index in remaining[: target - len(selected)]:
        add(index, "random", [f"稀缺覆盖候选不足，使用 seed={seed} 的训练集随机补足。"])
    return selected


def classification_balance_plan(report: dict) -> dict:
    """Suggest capped sqrt inverse-frequency weights for a future classifier.

    Only train instance counts determine weights: ``min(5, sqrt(n_max/n_c))``.
    The largest training class has weight 1. A class absent from training gets
    ``None`` rather than an invented training weight. This is a planning artifact,
    not a configured loss, PyTorch sampler, or executable training recipe.
    """
    categories = report.get("categories", [])
    reference = max((row["train_boxes"] for row in categories), default=0)
    rows = []
    for category in categories:
        count, images = category["train_boxes"], category["train_images"]
        trainable = count > 0
        rows.append({
            "category": category["category"],
            "train_boxes": count,
            "train_images": images,
            "trainable": trainable,
            "class_weight": min(5.0, math.sqrt(reference / count)) if trainable else None,
            "needs_source_diversity": images < 15,
            "recommended_action": (
                "补充多样的训练来源图像后再训练该类。" if not trainable else
                "优先补充不同原理图的训练来源，重复裁剪不能替代来源多样性。" if images < 15 else
                "可作为常规分类训练类别，继续检查裁剪质量。"
            ),
        })
    return {
        "method": "capped_sqrt_inverse_train_instance_frequency",
        "formula": "min(5, sqrt(max_train_class_boxes / train_class_boxes))",
        "weight_cap": 5.0,
        "normalization": "most frequent training class has weight 1",
        "reference_train_boxes": reference,
        "basis": "train instances only; validation and test counts are excluded",
        "is_ready_sampler": False,
        "usage_note": (
            "ResNet classification planning only. These are suggested class weights, "
            "not an existing sampler. Choose and validate a training implementation "
            "separately; weights cannot create missing source diversity."
        ),
        "classes": rows,
        "class_weights": {row["category"]: row["class_weight"] for row in rows},
        "untrainable_categories": [row["category"] for row in rows if not row["trainable"]],
        "diversity_priority_categories": [row["category"] for row in rows if row["needs_source_diversity"]],
    }
