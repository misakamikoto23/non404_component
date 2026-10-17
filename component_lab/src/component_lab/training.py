import json
import math
from numbers import Integral, Real
from pathlib import Path


def _validate_balance_cap(cap):
    if isinstance(cap, bool) or not isinstance(cap, Real) or not math.isfinite(cap) or cap < 1:
        raise ValueError("balance cap must be a finite number at least 1")
    return float(cap)


def build_training_sample_weights(targets, num_classes, cap=5):
    """Build capped square-root weights using training targets alone.

    Returns class counts, class weights, and per-sample weights as plain lists.
    The most frequent class has weight 1; absent training classes are rejected
    rather than assigned fabricated weights. No model or torch import is needed.
    """
    cap = _validate_balance_cap(cap)
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    targets = list(targets)
    counts = [0] * num_classes
    for target in targets:
        if isinstance(target, bool) or not isinstance(target, Integral) or not 0 <= target < num_classes:
            raise ValueError("training target must be an integer class index in [0, num_classes)")
        counts[target] += 1
    missing = [index for index, count in enumerate(counts) if count == 0]
    if missing:
        raise ValueError(f"Training classes have zero samples: {missing}. Add training images for every class.")
    maximum = max(counts)
    weights = [min(cap, math.sqrt(maximum / count)) for count in counts]
    return {"class_counts": counts, "class_weights": weights,
            "sample_weights": [weights[target] for target in targets]}


def train_yolo(args):
    from .models import configure_yolo
    configure_yolo()
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=args.output, name="yolo", seed=42)


def train_resnet(args):
    balance_mode = getattr(args, "balance", "none")
    if balance_mode not in ("none", "sqrt_sampler"):
        raise ValueError("balance must be 'none' or 'sqrt_sampler'")
    balance_cap = _validate_balance_cap(getattr(args, "balance_cap", 5))
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision.datasets import ImageFolder
    from torchvision.models import resnet18, ResNet18_Weights
    from .models import transform
    torch.manual_seed(42)
    root = Path(args.data)
    prepared = (root / "manifest.json").exists()
    image_root = root / "classification" if prepared else root
    if not (image_root / "train").is_dir():
        raise ValueError("Training split directory is missing; add your ResNet training dataset first.")
    if not (image_root / "val").is_dir():
        raise ValueError("Validation split directory is missing; add held-out validation images before training.")
    classes = (json.loads((root / "manifest.json").read_text(encoding="utf-8"))["classes"] if prepared
               else sorted(p.name for p in (image_root / "train").iterdir() if p.is_dir()))
    if not classes:
        raise ValueError("No class directories under train; add your ResNet dataset first.")
    datasets = {split: ImageFolder(image_root / split, transform=transform(), allow_empty=True)
                for split in ("train", "val")}
    for dataset in datasets.values():
        if dataset.classes != classes:
            raise ValueError("Class ordering differs from manifest.")
    counts = [datasets["train"].targets.count(index) for index in range(len(classes))]
    missing = [classes[index] for index, count in enumerate(counts) if count == 0]
    if missing:
        raise ValueError(f"Training classes have zero samples: {missing}. Add diverse training images before training.")
    if not len(datasets["val"]):
        raise ValueError("Validation split has no images; add held-out validation images before training.")
    balance = build_training_sample_weights(datasets["train"].targets, len(classes), balance_cap)
    sampler = None
    if balance_mode == "sqrt_sampler":
        generator = torch.Generator().manual_seed(42)
        sampler = WeightedRandomSampler(balance["sample_weights"], num_samples=len(datasets["train"]),
                                        replacement=True, generator=generator)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch, shuffle=sampler is None,
                            sampler=sampler, num_workers=0),
        "val": DataLoader(datasets["val"], batch_size=args.batch, shuffle=False, num_workers=0),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    applied_weights = balance["class_weights"] if sampler is not None else [1.0] * len(classes)
    balance_record = {
        "mode": balance_mode, "seed": 42, "balance_cap": balance_cap,
        "num_classes": len(classes), "num_training_samples": len(datasets["train"]),
        "class_counts": dict(zip(classes, balance["class_counts"])),
        "class_weights": dict(zip(classes, applied_weights)),
        "sampler": "WeightedRandomSampler" if sampler is not None else None,
        "num_samples_per_epoch": len(datasets["train"]), "replacement": sampler is not None,
        "weight_basis": "train.targets only; validation and test data are excluded",
        "formula": "min(cap, sqrt(max_train_class_count / train_class_count))" if sampler is not None else None,
        "loss_class_weights": None,
        "source_diversity_note": "Sampling cannot replace diverse source diagrams; repeated crops do not add source diversity.",
    }
    (output / "train_balance.json").write_text(json.dumps(balance_record, ensure_ascii=False, indent=2), encoding="utf-8")
    model = resnet18(weights=None if args.no_pretrained else ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    best, history = -1, []
    for epoch in range(args.epochs):
        model.train()
        for images, labels in loaders["train"]:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = torch.zeros(len(classes), dtype=torch.long)
        total = torch.zeros(len(classes), dtype=torch.long)
        with torch.inference_mode():
            for images, labels in loaders["val"]:
                predictions = model(images.to(args.device)).argmax(1).cpu()
                for label, prediction in zip(labels, predictions):
                    total[label] += 1
                    correct[label] += int(label == prediction)
        present = total > 0
        macro = float((correct[present] / total[present]).mean())
        metrics = {"epoch": epoch+1, "accuracy": float(correct.sum()/total.sum()),
                   "macro_recall_present_classes": macro, "missing_val_classes": [classes[i] for i in range(len(classes)) if not present[i]]}
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        if macro > best:
            best = macro
            torch.save({"architecture": "resnet18", "classes": classes,
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "preprocess": "letterbox224_imagenet", "epoch": epoch+1}, output / "best.pt")
        (output / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
