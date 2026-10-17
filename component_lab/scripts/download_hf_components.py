"""Download a public YOLO-format dataset to a separate staging directory."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="LibreYOLO/circuit-elements")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/external/circuit-elements")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Install download dependency first: python -m pip install huggingface_hub")
    patterns = ["*.md", "*.yaml", "*.yml", "*.txt"] if args.metadata_only else None
    snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(args.output), allow_patterns=patterns)
    print(f"Downloaded to {args.output.resolve()}. Review images, classes and source license before merging.")


if __name__ == "__main__":
    main()
