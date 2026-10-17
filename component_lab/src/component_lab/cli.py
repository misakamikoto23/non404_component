import argparse
import json
import math
import os


def main():
    parser = argparse.ArgumentParser(description="Schematic component lab")
    from pathlib import Path
    defaults_path = Path(__file__).resolve().parents[2] / "configs/defaults.json"
    defaults = json.loads(defaults_path.read_text(encoding="utf-8")) if defaults_path.exists() else {}
    yolo, resnet, qwen = (defaults.get(k, {}) for k in ("yolo", "resnet", "qwen"))
    local_v6 = defaults.get("local_v6", {})
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Create detection and classification datasets")
    p.add_argument("--archive", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--origin", required=True, choices=["top-left", "bottom-left"])
    p.add_argument("--seed", type=int, default=42)
    for name in ("train-yolo", "train-resnet"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--output", required=True)
        preset = yolo if name == "train-yolo" else resnet
        p.add_argument("--epochs", type=int, default=preset.get("epochs", 50))
        p.add_argument("--batch", type=int, default=preset.get("batch", 8))
        p.add_argument("--device", default=preset.get("device", "cpu"), help="YOLO GPU: 0; ResNet GPU: cuda:0")
        if name == "train-yolo":
            p.add_argument("--model", default=yolo.get("model", "yolo26n.pt"))
            p.add_argument("--imgsz", type=int, default=yolo.get("imgsz", 1280))
        else:
            p.add_argument("--lr", type=float, default=resnet.get("lr", 1e-4))
            p.add_argument("--no-pretrained", action="store_true")
            p.add_argument("--balance", choices=["none", "sqrt_sampler"], default="none",
                           help="Optional capped square-root sampling from training class counts only")
            p.add_argument("--balance-cap", type=float, default=5,
                           help="Maximum sampling weight relative to the most frequent training class (>=1)")
    p = sub.add_parser("infer")
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--yolo", help="Fine-tuned schematic detector checkpoint")
    group.add_argument("--detections", help="Cached detections JSON list, for debugging")
    p.add_argument("--resnet")
    p.add_argument("--device", default=defaults.get("device", "cpu"))
    p.add_argument("--imgsz", type=int, default=yolo.get("imgsz", 1280))
    p.add_argument("--conf", type=float, default=yolo.get("confidence", .25))
    p.add_argument("--tile-size", type=int, help="Enable whole-image + tiled YOLO; tile size in original pixels")
    p.add_argument("--tile-overlap", type=float, default=.25)
    p.add_argument("--tile-max-tiles", type=int, default=100, help="Skip all tiles above this count; preserve whole-image predictions")
    p.add_argument("--classifier-threshold", type=float, default=resnet.get("threshold", .8))
    p.add_argument("--qwen", action="store_true", help="Send symbol and context crops to Qwen API")
    p.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", qwen.get("model", "qwen3.8-flash")))
    p.add_argument("--qwen-max-calls", type=int, default=qwen.get("max_calls", 10))
    p.add_argument("--context-margin", type=int, default=qwen.get("context_margin", 48))
    p.add_argument("--qwen-timeout", type=float, default=qwen.get("timeout", 60))
    p.add_argument("--qwen-max-tokens", type=int, default=qwen.get("max_tokens", 800))
    p.add_argument("--qwen-temperature", type=float, default=qwen.get("temperature", 0))
    p.add_argument("--text-margin", type=float, default=60)
    p.add_argument("--association", choices=["legacy", "identity-v1"], default="legacy",
                   help="Opt-in OCR identity refinement; default preserves the existing pipeline")
    p.add_argument("--contest-export", action="store_true",
                   help="With identity-v1, export component-only contest JSON in bottom-left coordinates")
    p.add_argument("--ocr-spans", action="store_true", help="Experimental exact-substring OCR candidate view (identity-v1 only)")
    p.add_argument("--display-names", action="store_true", help="Experimental box-local literal display Name slot (identity-v1 only)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--ocr", choices=["paddle", "none"], default="none")
    group.add_argument("--ocr-json", help="Normalized OCR JSON list")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--local-v6", dest="local_v6", action="store_true",
                       help="Enable bounded local PP-OCRv6 fallback after identity-v1 (default when configured)")
    group.add_argument("--no-local-v6", dest="local_v6", action="store_false",
                       help="Disable local v6; preserve primary OCR association")
    p.set_defaults(local_v6=local_v6.get("enabled", True))
    p.add_argument("--local-v6-mode", choices=["empty-fields", "unresolved-only"], default=local_v6.get("mode", "empty-fields"))
    p.add_argument("--local-v6-python", help="Python executable with RapidOCR 3.10 and ONNX Runtime; default .venv-ocr")
    p.add_argument("--local-v6-max-rois", type=int, default=local_v6.get("max_rois", 2))
    p.add_argument("--local-v6-area-fraction", type=float, default=local_v6.get("area_fraction", .2))
    p.add_argument("--local-v6-timeout", type=float, default=local_v6.get("timeout", 120), help="Total OCR worker seconds per image")
    p = sub.add_parser("recover-identities", help="Offline recovery of unresolved keys; existing named records stay frozen")
    p.add_argument("--baseline", required=True, help="Frozen case-to-prediction JSON bundle")
    p.add_argument("--manifest", required=True, help="Hash-bound targets and baseline fingerprint")
    p.add_argument("--readings", required=True, help="Machine JSON items; successful subsets are allowed")
    p.add_argument("--output", required=True, help="New output directory; never overwrite inputs")
    p.add_argument("--with-fields", action="store_true", help="Also write Name/value on successfully recovered records; default only recovers keys/types")
    p = sub.add_parser("score", help="Local Contest 6 score (public rules; not the official evaluator)")
    p.add_argument("--pred", required=True, help="Directory of prediction JSONs (any nesting; case id from file name)")
    p.add_argument("--gt", required=True, help="Directory of GT JSONs, e.g. data/official_contest6/train200")
    p.add_argument("--split", help="Split JSON; restricts scoring to its cases (e.g. configs/contest6_dev_split.json)")
    p.add_argument("--partition", help="Partition of a group split file (test, dev, train)")
    p.add_argument("--anon-policy", choices=["strict", "observable_diagnostic", "both"], default="both",
                   help="strict (headline) scores ∅75-style keys like any key; observable_diagnostic drops them "
                        "from GT and prediction (diagnostic only)")
    p.add_argument("--type-policy", choices=["strict", "mapped", "both"], default="strict",
                   help="strict (headline) compares raw type strings; mapped canonicalizes with eval/type_vocab.json")
    p.add_argument("--bbox-metric", choices=["max_corner", "center"], default="max_corner",
                   help="Bbox error definition (assumption until the official evaluator ships)")
    p.add_argument("--bbox-tol", type=float, default=20.0)
    p.add_argument("--point-tol", type=float, default=5.0)
    p.add_argument("--net-tiebreak", choices=["segments", "none"], default="segments",
                   help="segments (assumption): break equal pin-overlap ties and pair pin-less nets by matched segments")
    p.add_argument("--oracle", default="none",
                   help="Output-substitution oracle: comma list of JSON sections replaced by GT after the pipeline "
                        "(components,pins,nets); downstream stages are not rerun")
    p.add_argument("--ablate", action="store_true", help="Also run each single-section substitution oracle and report deltas")
    p.add_argument("--out", help="Write per_case.csv, summary.md and summary.json here")
    p = sub.add_parser("extract-gt", help="Copy official GT JSONs + manifest out of the contest zip")
    p.add_argument("--archive", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--with-images", action="store_true")
    p = sub.add_parser("make-split", help="Deterministic stratified held-out split of official train cases")
    p.add_argument("--manifest", required=True, help="_manifest.json written by extract-gt")
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=30)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--detector-audit", help="Contest alignment audit.json; detector-training cases are taken last")
    args = parser.parse_args()
    if args.command == "train-resnet" and (not math.isfinite(args.balance_cap) or args.balance_cap < 1):
        parser.error("--balance-cap must be a finite number at least 1")
    if args.command == "infer":
        from .local_v6 import validate_options
        try:
            validate_options(mode=args.local_v6_mode, max_rois=args.local_v6_max_rois,
                             area_fraction=args.local_v6_area_fraction, timeout=args.local_v6_timeout)
        except ValueError as exc:
            parser.error(str(exc))
        if (args.ocr_spans or args.display_names) and args.association != "identity-v1":
            parser.error("--ocr-spans/--display-names require --association identity-v1")
        if args.contest_export and args.association != "identity-v1":
            parser.error("--contest-export requires --association identity-v1")
        from .tiling import validate_settings
        try:
            validate_settings(args.tile_size, args.tile_overlap, args.tile_max_tiles)
        except ValueError as exc:
            parser.error(str(exc))
        if args.tile_size is not None and args.detections:
            parser.error("--tile-size requires --yolo; cached detections cannot run tile inference")
    if args.command == "recover-identities":
        from .recovery_io import recover_files
        try:
            result = recover_files(args.baseline, args.manifest, args.readings, args.output,
                                   with_fields=args.with_fields)
        except (ValueError, KeyError, FileExistsError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=True))
        return
    if args.command == "score":
        from .eval.metric import ORACLE_STAGES
        from .eval.report import build_configs, run
        stages = [] if args.oracle == "none" else [s.strip() for s in args.oracle.split(",") if s.strip()]
        if any(stage not in ORACLE_STAGES for stage in stages):
            parser.error(f"--oracle takes 'none' or a comma list of {', '.join(ORACLE_STAGES)}")
        if args.ablate and stages:
            parser.error("--ablate already runs every single-stage oracle; drop --oracle")
        policies = ["strict", "observable_diagnostic"] if args.anon_policy == "both" else [args.anon_policy]
        type_policies = ["strict", "mapped"] if args.type_policy == "both" else [args.type_policy]
        configs = build_configs(policies, args.bbox_metric, args.bbox_tol, args.point_tol, stages, args.ablate,
                                args.net_tiebreak, type_policies)
        _, report = run(args.pred, args.gt, configs, args.split, args.out, args.partition)
        import sys
        sys.stdout.reconfigure(errors="replace")  # GBK consoles cannot print '∅'
        print(report)
        return
    if args.command == "extract-gt":
        from .eval.cases import extract_official
        print(json.dumps(extract_official(args.archive, args.output, args.with_images), ensure_ascii=False))
        return
    if args.command == "make-split":
        from .eval.cases import make_split
        split = make_split(args.manifest, args.size, args.seed, args.detector_audit)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(split["strata"], ensure_ascii=False))
        return
    if args.command == "prepare":
        from .dataset import prepare
        print(json.dumps(prepare(args.archive, args.output, args.origin, args.seed), ensure_ascii=False))
    elif args.command == "infer":
        from .pipeline import infer
        print(json.dumps(infer(args), ensure_ascii=False))
    else:
        from .training import train_yolo, train_resnet
        (train_yolo if args.command == "train-yolo" else train_resnet)(args)


if __name__ == "__main__":
    main()
