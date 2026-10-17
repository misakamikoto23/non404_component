import json
from pathlib import Path
from PIL import Image, ImageDraw
from .association import associate
from .geometry import convert_bbox, crop_box
from .models import YoloDetector, ResnetClassifier, paddle_texts
from .tiling import validate_settings


def infer(args):
    if args.qwen_max_calls < 0 or args.context_margin < 0:
        raise ValueError("Qwen call limit and crop margin must be non-negative")
    tile_size = getattr(args, "tile_size", None)
    tile_overlap = getattr(args, "tile_overlap", .25)
    tile_max_tiles = getattr(args, "tile_max_tiles", 100)
    validate_settings(tile_size, tile_overlap, tile_max_tiles)
    if tile_size is not None and args.detections:
        raise ValueError("Tile inference requires YOLO, not cached detections")
    image = Image.open(args.image).convert("RGB")
    if args.detections:
        components = json.loads(Path(args.detections).read_text(encoding="utf-8"))
        if isinstance(components, dict):
            components = components["components"]
        for row in components:
            row["bbox"] = convert_bbox(row["bbox"], *image.size, "top-left")
            row.setdefault("detection", {"type": row["type"], "score": row.get("score")})
            row.setdefault("classification", None)
    else:
        detector = YoloDetector(args.yolo, args.device, args.imgsz, args.conf,
                                tile_size=tile_size, tile_overlap=tile_overlap, tile_max_tiles=tile_max_tiles)
        components = detector.predict(image)
    components.sort(key=lambda row: (row["bbox"][1], row["bbox"][0]))
    classifier = ResnetClassifier(args.resnet, args.device) if args.resnet else None
    for i, component in enumerate(components, 1):
        component["id"] = f"component_{i:04d}"
        if classifier:
            prediction = classifier.predict(crop_box(image, component["bbox"]))
            component["classification"] = prediction
            if prediction["score"] >= args.classifier_threshold:
                component["type"] = prediction["type"]
    if args.ocr_json:
        texts = json.loads(Path(args.ocr_json).read_text(encoding="utf-8"))
        if isinstance(texts, dict):
            texts = texts.get("texts", texts.get("ocr_texts", []))
    elif args.ocr == "paddle":
        texts = paddle_texts(args.image)
    else:
        texts = []
    for i, text in enumerate(texts, 1):
        text["id"] = f"text_{i:04d}"
        text["bbox"] = convert_bbox(text["bbox"], *image.size, "top-left")
    span_audit = None
    if getattr(args, "ocr_spans", False):
        from .ocr_spans import split_ocr_spans
        texts, span_audit = split_ocr_spans(texts)
    identity = None
    if getattr(args, "association", "legacy") == "identity-v1":
        from .contest_components import associate_variants
        variants, identity_changes = associate_variants(components, texts)
        identity = variants["identity_v1"]
        if getattr(args, "display_names", False):
            from .display_names import assign_display_names
            identity = assign_display_names(identity)
        components, texts, unassigned = identity["components"], identity["texts"], identity["unassigned_texts"]
    else:
        unassigned = associate(components, texts, args.text_margin)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    local_v6 = {"status": "disabled", "api_calls": 0}
    if getattr(args, "local_v6", False):
        if identity is None:
            local_v6["status"] = "skipped_requires_identity_v1"
        elif not args.ocr_json and getattr(args, "ocr", "none") == "none":
            local_v6["status"] = "skipped_requires_primary_ocr"
        else:
            from .local_v6 import recover_local, save, validate_options
            settings = {"mode": getattr(args, "local_v6_mode", "empty-fields"),
                        "max_rois": getattr(args, "local_v6_max_rois", 2),
                        "area_fraction": getattr(args, "local_v6_area_fraction", .2),
                        "timeout": getattr(args, "local_v6_timeout", 120)}
            validate_options(**settings)
            try:
                identity, local_v6 = recover_local(identity, image, output / "local_v6", **settings,
                    python=getattr(args, "local_v6_python", None),
                    display_names=getattr(args, "display_names", False))
            except Exception as exc:
                # Recovery works on copies. Keep the exact primary result on
                # dependency, worker, malformed-cache or synchronization failure.
                local_v6 = {"status": "failed_baseline_preserved", "error_type": type(exc).__name__, "api_calls": 0}
                save(output / "local_v6" / "summary.json", local_v6)
            components, texts, unassigned = identity["components"], identity["texts"], identity["unassigned_texts"]
    crops = output / "crops"
    crops.mkdir(exist_ok=True)
    if args.qwen:
        from .qwen_api import QwenClient
        qwen = QwenClient(args.qwen_model, args.qwen_timeout, args.qwen_max_tokens, args.qwen_temperature)
    else:
        qwen = None
    for index, component in enumerate(components):
        symbol = crop_box(image, component["bbox"], margin=4)
        context = crop_box(image, component["bbox"], margin=args.context_margin)
        symbol_path = crops / (component["id"] + "_symbol.png")
        context_path = crops / (component["id"] + "_context.png")
        symbol.save(symbol_path)
        context.save(context_path)
        component["crops"] = {"symbol": str(symbol_path.resolve()), "context": str(context_path.resolve())}
        component["qwen"] = {"status": "disabled" if not qwen else "skipped_call_limit"}
        if qwen and index < args.qwen_max_calls:
            try:
                component["qwen"] = qwen.predict(symbol, context, component)
            except Exception as exc:
                component["qwen"] = {"status": "api_error", "error_type": type(exc).__name__}
    report = {"schema_version": "0.1", "image": str(Path(args.image).resolve()), "size": list(image.size),
              "coordinate_system": "top-left-xyxy", "mode": "cached_detections" if args.detections else "yolo",
              "components": components, "ocr_texts": texts, "unassigned_texts": unassigned,
              "local_v6": local_v6,
              "warnings": ["Qwen results are independent candidates, not automatically merged into verified fields.",
                           "Competition-format export is not implemented."]}
    if identity is not None:
        report["identity"] = {"method": "identity-v1", "changes": identity_changes,
                              "diagnostics": identity["diagnostics"]}
        if span_audit is not None:
            report["identity"]["ocr_span_audit"] = span_audit
        if "display_name_diagnostics" in identity:
            report["identity"]["display_name_diagnostics"] = identity["display_name_diagnostics"]
        report["warnings"][-1] = "Pin and net extraction are not implemented in this component pipeline."
        if getattr(args, "contest_export", False):
            from .contest_components import export_contest
            contest, trace = export_contest(identity, image.height, ic_value_as_name=True,
                                             display_names=getattr(args, "display_names", False))
            (output / "contest_components.json").write_text(json.dumps(contest, ensure_ascii=False, indent=2), encoding="utf-8")
            report["contest_export"] = {"path": str((output / "contest_components.json").resolve()),
                                        "coordinate_system": "bottom-left-xyxy", "component_only": True,
                                        "trace": trace}
    if not args.detections:
        report["inference"] = detector.inference_metadata
    (output / "components.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    draw = ImageDraw.Draw(image)
    for component in components:
        draw.rectangle(component["bbox"], outline="red", width=2)
        draw.text(tuple(component["bbox"][:2]), component["id"]+" "+component["type"], fill="red")
    for text in texts:
        draw.rectangle(text["bbox"], outline="blue", width=1)
    image.save(output / "overlay.png")
    return {"components": len(components), "texts": len(texts), "unassigned": len(unassigned),
            "local_v6": local_v6["status"], "local_v6_changed_objects": local_v6.get("changed_objects", 0)}
