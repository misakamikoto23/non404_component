"""Validate downloaded Open Schematics pairs and build a local preview."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics

from PIL import Image


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def inspect(root: Path):
    root = root.resolve()
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = []
    resolutions = Counter()
    projects = Counter()
    libraries = Counter()
    image_hashes = set()
    source_hashes = set()
    sizes = []
    restored_source_bytes = 0
    for row in records:
        try:
            source = (root / row["source_file"]).resolve()
            image = (root / row["image_file"]).resolve()
            if not source.is_relative_to(root) or not image.is_relative_to(root):
                raise ValueError("Path outside dataset directory")
            source_bytes = source.read_bytes()
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            # Early downloader versions used write_text on Windows. Reverse only
            # that newline translation when the original checksum proves it exact.
            if source_hash != row["source_sha256"]:
                original = source_bytes.replace(b"\r\n", b"\n")
                if hashlib.sha256(original).hexdigest() == row["source_sha256"]:
                    source.write_bytes(original)
                    source_hash = row["source_sha256"]
                    restored_source_bytes += 1
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            if source_hash != row["source_sha256"] or image_hash != row["image_sha256"]:
                raise ValueError("Source or image checksum differs from manifest")
            with Image.open(image) as opened:
                opened.load()
                if opened.size != (row["width"], row["height"]):
                    raise ValueError("Image dimensions differ from manifest")
            if source_hash in source_hashes or row["rgb_sha256"] in image_hashes:
                raise ValueError("Duplicate source or RGB image in manifest")
            source_hashes.add(source_hash)
            image_hashes.add(row["rgb_sha256"])
            provenance = json.loads((source.parent / "source.json").read_text(encoding="utf-8"))
            if provenance["id"] != row["id"]:
                raise ValueError("Provenance ID differs from manifest")
            if row["structured_json_valid"]:
                json.loads((source.parent / "schematic_metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as exc:
            errors.append({"id": row.get("id"), "error": str(exc)})
        resolutions[(row["width"], row["height"])] += 1
        projects[row["source_repository"]] += 1
        libraries.update(set(row["component_library_names"]))
        sizes.append(row["placed_symbol_count"])
    disk_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    summary = json.loads((root / "download_summary.json").read_text(encoding="utf-8"))
    profile = {
        "records": len(records), "validation_passed": not errors, "errors": errors,
        "source_files_restored_to_original_bytes": restored_source_bytes,
        "unique_source_projects": len(projects), "disk_bytes_before_gallery": disk_bytes,
        "placed_symbols_including_power_and_other_symbols": sum(sizes),
        "symbols_per_schematic": {"min": min(sizes, default=0), "median": statistics.median(sizes) if sizes else 0, "max": max(sizes, default=0)},
        "resolution_counts": [{"width": w, "height": h, "count": n} for (w, h), n in resolutions.most_common()],
        "library_names_by_schematic_count": dict(libraries.most_common()),
        "projects_by_schematic_count": dict(projects.most_common()),
        "sampling": "First eligible distinct modern-KiCad rows in pinned shard order; not random or class-balanced.",
        "split_policy": "Unassigned. Group by source project before any train/validation/test split.",
        "yolo_labels_generated": False,
    }
    write_json(root / "validation_report.json", profile)
    if errors:
        raise RuntimeError(f"Validation failed for {len(errors)} records; see validation_report.json")
    # Put a varied deterministic selection first, and let the user browse all pairs.
    cards = []
    step = max(1, len(records) // 40)
    preview = records[::step][:40]
    selected = {row["id"] for row in preview}
    ordered = preview + [row for row in records if row["id"] not in selected]
    for row in ordered:
        title = html.escape(row["source_repository"] or row["id"])
        image = html.escape(row["image_file"], quote=True)
        source = html.escape(row["source_file"], quote=True)
        provenance = html.escape(str(Path(row["source_file"]).with_name("source.json")).replace("\\", "/"), quote=True)
        cards.append(f'<article><a href="{image}"><img loading="lazy" src="{image}" alt="{title}"></a><h3>{title}</h3><p>{row["width"]} × {row["height"]} · {row["placed_symbol_count"]} 个符号实例（含电源等符号）</p><p><a href="{source}">KiCad 源文件</a> · <a href="{provenance}">来源信息</a></p></article>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Open Schematics 本地数据</title><style>body{margin:0;background:#f3f5f7;color:#233040;font:16px system-ui}header{padding:28px 4vw;background:white;line-height:1.7}h1{margin:0;font-size:26px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:18px;padding:24px 4vw}article{background:white;border-radius:10px;padding:14px;overflow:hidden}img{width:100%;height:250px;object-fit:contain;background:#fff}h3{font-size:15px;overflow-wrap:anywhere}p{font-size:14px}a{color:#1768bb}</style><header>'''
    page += f'<h1>Open Schematics · {len(records):,} 份原理图</h1><p>来自 {len(projects):,} 个源项目，保留 KiCad 源文件、原图、结构化信息和来源。下载版本：<code>{html.escape(summary["revision"])}</code></p>'
    page += '<p>当前是原始数据，尚未生成 YOLO 框。图片为数据集提供的原图，其分辨率可能不足以直接训练；后续可从源文件重新渲染。先展示间隔抽取的 40 例，下方可浏览全部样本；点击图片放大。</p></header><main>'
    page += "".join(cards) + "</main></html>"
    (root / "gallery.html").write_text(page, encoding="utf-8")
    readme = f'''# Open Schematics 本地子集

- 数据集：[bshada/open-schematics](https://huggingface.co/datasets/bshada/open-schematics)
- 固定版本：`{summary["revision"]}`
- 已下载：{len(records):,} 份不同的现代 KiCad 原理图，来自 {len(projects):,} 个源项目。
- 附带可解析的结构化 JSON：{sum(row['structured_json_valid'] for row in records):,} 份；其他样本仍保留源文件与图片。
- 原理图内的符号实例总数：{sum(sizes):,}（包含电源、地等符号，不等同于题目定义的元件数）。
- 源文件、图片校验和、图像解码、来源文件、结构化 JSON：全部验证通过。
- 只读取原理图相关列，未下载 PCB 二进制，未启动训练。

## 目录

`samples/<样本ID>/` 内保存 `schematic.kicad_sch`、`schematic.png`（或原图格式）、`schematic_metadata.json`（若存在）、`source.json`。

`manifest.jsonl` 是逐条索引；`download_summary.json` 是下载统计；`validation_report.json` 是校验和分布统计；`gallery.html` 可在浏览器中查看图片和配套源文件。`DATASET_CARD.md` 保留该版本的数据说明。

## 使用边界

这是按分片顺序选取的前 {len(records):,} 条满足条件且去重的数据，**不是随机或类别均衡样本**。当前未与 PCBnet 或人工审核训练集混合，也未划分训练/验证/测试集。

**这些不是现成的 YOLO 标签。** 下一步需解析符号几何和变换、统一元件/电源/接口定义，并把渲染坐标转换为像素框。数据集原图可能分辨率较低，宜结合 KiCad 源文件重新渲染；还需检查多页文件、库继承和图像对齐。

`components_used` 是不同库符号名的集合，不能当作实例数量或题目的类别标签。目录内的符号实例计数也包含电源等符号。后续必须按源项目分组划分数据，防止同一项目的相似电路泄漏到验证集；还应与已有训练/验证/测试集去重。

数据卡标注 CC-BY-4.0，同时保留并遵循各原始项目的许可要求；逐条来源见 `source.json`。

## 继续下载

在工作区根目录使用现有 Python 环境运行 `component_lab/scripts/download_open_schematics.py --count 2000 --max-download-gb 4` 可续抓到总计 2000 条。下载器固定原版本、去重、保留检查点，并在剩余空间低于 5 GiB 时停止。完成后运行 `component_lab/scripts/inspect_open_schematics.py` 更新校验和预览。
'''
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({k: profile[k] for k in ["records", "validation_passed", "unique_source_projects", "disk_bytes_before_gallery", "placed_symbols_including_power_and_other_symbols", "symbols_per_schematic"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "Data/open_schematics")
    inspect(parser.parse_args().root)
