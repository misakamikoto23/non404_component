"""Offline OCR ownership comparison; no training, cloud calls, or GT inference.

Example:
  python scripts/evaluate_ocr_association.py --image input.png \
    --detections components.json --ocr ocr.json --output outputs/ocr_association

JSON inputs: detections are a list or {components: [...]}; OCR is a list or
{texts|ocr_texts|results: [...]}, each row {id?, text, bbox:[x1,y1,x2,y2], score?}.
An optional --manifest JSON is a list of {id?,image,detections,ocr,reference?}.
Optional evaluator reference is {components:[{bbox,reference?,value?,part_number?}]}.
Only explicitly present reference fields are scored, after geometric IoU >=.5
matching; no reference information enters parsing, grouping or assignment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from component_lab.association import associate as legacy_associate
from component_lab.text_matching import associate_structured


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows_from(payload, keys):
    if isinstance(payload, list):
        return payload
    for key in keys:
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Expected a list or object containing one of {keys}")


def normalize_texts(payload):
    rows = rows_from(payload, ("texts", "ocr_texts", "results"))
    normalized = []
    for i, source in enumerate(rows, 1):
        row = deepcopy(source)
        row.setdefault("id", f"text_{i:04d}")
        row.setdefault("text", row.get("raw", row.get("rec_text", "")))
        if "score" not in row and "confidence" in row:
            row["score"] = row["confidence"]
        if "bbox" not in row:
            poly = row.get("polygon", row.get("poly"))
            if poly:
                xs, ys = zip(*poly)
                row["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
            else:
                raise ValueError("Text rows require bbox or polygon")
        normalized.append(row)
    return normalized


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def evaluate_reference(predictions, reference):
    """Evaluator only. Compare readable target strings without case correction."""
    truths = rows_from(reference, ("components",))
    overlaps = np.array([[iou(p["bbox"], t["bbox"]) for t in truths] for p in predictions], dtype=float)
    pairs = []
    if predictions and truths:
        pi, ti = linear_sum_assignment(-overlaps)
        pairs = [(int(p), int(t), float(overlaps[p, t])) for p, t in zip(pi, ti) if overlaps[p, t] >= .5]
    metrics = {"evaluation_kind": "optional_reference_not_used_for_predictions",
               "gt_component_count": len(truths), "geometrically_matched_components": len(pairs),
               "component_match_iou_threshold": .5, "fields": {}}
    for field in ("reference", "value", "part_number"):
        eligible = [(p, t) for p, t, _ in pairs if field in truths[t] and truths[t][field] is not None]
        correct = sum(str(predictions[p].get(field, "")).strip() == str(truths[t][field]).strip() for p, t in eligible)
        metrics["fields"][field] = {"explicit_targets_on_matched_components": len(eligible), "exact_matches": correct,
                                    "exact_match_rate": correct/len(eligible) if eligible else None}
    metrics["limitations"] = ["Only matched detection boxes are scored; this is not end-to-end component recall.",
                              "Reference labels have not been independently audited by this script.",
                              "Absent/null GT fields are not treated as negative labels."]
    metrics["source_warnings"] = reference.get("warnings", []) if isinstance(reference, dict) else []
    metrics["reference_label_kind"] = reference.get("label_kind", "unspecified_unverified") if isinstance(reference, dict) else "unspecified_unverified"
    return metrics


def proxy_metrics(components, unassigned, texts):
    return {"evaluation_kind": "coverage_and_consistency_proxies_not_accuracy",
            "components": len(components), "ocr_regions": len(texts),
            "references_assigned": sum(bool(row.get("reference")) for row in components),
            "values_assigned": sum(bool(row.get("value")) for row in components),
            "part_numbers_assigned": sum(bool(row.get("part_number")) for row in components),
            "texts_with_component_owner": sum(len(row.get("texts", [])) for row in components),
            "unassigned_texts": len(unassigned),
            "review_flags": dict(Counter(flag for row in components for flag in row.get("review", [])))}


def _color(index):
    palette = ["#cc3300", "#00886c", "#6a40bd", "#b45b00", "#0065bd", "#a80078", "#537600"]
    return palette[index % len(palette)]


def svg_for(components, texts, size):
    width, height = size
    pieces = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
              f'<image href="source.png" width="{width}" height="{height}"/>']
    for i, component in enumerate(components):
        x1, y1, x2, y2 = component["bbox"]
        text = f'{component["id"]} {component.get("type", "?")} {component.get("reference") or ""}'
        pieces.append(f'<g><title>{html.escape(text)}</title><rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{_color(i)}" stroke-width="1.5"/></g>')
        cx, cy = (x1+x2)/2, (y1+y2)/2
        for text in component.get("texts", []):
            a, b, c, d = text["bbox"]
            pieces.append(f'<line x1="{cx}" y1="{cy}" x2="{(a+c)/2}" y2="{(b+d)/2}" stroke="{_color(i)}" stroke-width="1.3"/>')
    for text in texts:
        a, b, c, d = text["bbox"]
        title = f'{text["id"]}: {text.get("text", "")} | {text.get("primary_role", "legacy unsplit")}'
        pieces.append(f'<g><title>{html.escape(title)}</title><rect x="{a}" y="{b}" width="{c-a}" height="{d-b}" fill="#0055ff" fill-opacity=".06" stroke="#0055ff" stroke-width=".8"/></g>')
    pieces.append('</svg>')
    return "".join(pieces)


def write_gallery(output, source, old, new, texts, metrics):
    image = Image.open(source).convert("RGB")
    image.save(output / "source.png")
    draw = ImageDraw.Draw(image)
    for i, component in enumerate(new["components"]):
        box = component["bbox"]
        draw.rectangle(box, outline=_color(i), width=2)
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        for text in component["texts"]:
            a, b, c, d = text["bbox"]
            draw.line((cx, cy, (a+c)/2, (b+d)/2), fill=_color(i), width=1)
    for text in texts:
        draw.rectangle(text["bbox"], outline="#0055ff", width=1)
    image.save(output / "overlay.png")
    rows = []
    for component in new["components"]:
        evidence = component["association_evidence"]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value or ''))}</td>" for value in
                    [component["id"], component.get("type"), component["reference"], component["value"],
                     component["part_number"], json.dumps(evidence, ensure_ascii=False)]) + "</tr>")
    textrows = []
    owners = {entry["text_id"]: entry["component_id"] for entry in new["assignments"]}
    for text in new["texts"]:
        textrows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in
                        [text["id"], text["raw"], text["primary_role"], text.get("score", ""), owners.get(text["id"], "unmatched"),
                         json.dumps(text["role_candidates"], ensure_ascii=False)]) + "</tr>")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>OCR归属对照</title>
<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;color:#222;background:#f5f6f8}button{padding:9px 20px;margin-right:12px}section{background:white;padding:12px;margin:14px 0}svg{width:100%;height:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top;overflow-wrap:anywhere}td:last-child{max-width:500px}pre{white-space:pre-wrap}#new{display:none}</style>
<h1>OCR文字角色与元件归属对照</h1><p>蓝框：OCR文字。彩框：检测元件。连线：文字所有权。悬停可看框信息。未关联不等于错误；网络名和引脚号保留给后续阶段。</p>
<p><strong>指标为覆盖和一致性代理，不代表准确率。OCR分数、规则分数均未校准。</strong></p>
<button onclick="document.getElementById('old').style.display='block';document.getElementById('new').style.display='none'">原距离基线</button><button onclick="document.getElementById('new').style.display='block';document.getElementById('old').style.display='none'">新角色＋整体匹配</button>
"""
    page += '<section id="old">' + svg_for(old, texts, image.size) + '</section><section id="new">' + svg_for(new["components"], new["texts"], image.size) + '</section>'
    page += '<details><summary>指标与限制</summary><pre>' + html.escape(json.dumps(metrics, ensure_ascii=False, indent=2)) + '</pre></details>'
    page += '<h2>新方法元件字段</h2><table><tr><th>ID</th><th>类型</th><th>位号</th><th>参数</th><th>型号</th><th>依据</th></tr>' + ''.join(rows) + '</table>'
    page += '<h2>OCR原文、候选与归属</h2><table><tr><th>ID</th><th>原文</th><th>主要角色</th><th>OCR分数</th><th>归属</th><th>角色候选</th></tr>' + ''.join(textrows) + '</table></html>'
    (output / "gallery.html").write_text(page, encoding="utf-8")


def run_job(job, output):
    output.mkdir(parents=True, exist_ok=True)
    comps = rows_from(read_json(job["detections"]), ("components", "detections"))
    comps = deepcopy(comps)
    for i, component in enumerate(comps, 1):
        component.setdefault("id", f"component_{i:04d}")
    texts = normalize_texts(read_json(job["ocr"]))
    new = associate_structured(comps, texts)
    old = deepcopy(comps)
    old_unassigned = legacy_associate(old, deepcopy(texts))
    metrics = {"old": proxy_metrics(old, old_unassigned, texts),
               "new": proxy_metrics(new["components"], new["unassigned_texts"], texts),
               "role_distribution": dict(Counter(text["primary_role"] for text in new["texts"])),
               "heldout_reference_available": bool(job.get("reference")),
               "limitations": ["No gold OCR text-owner annotations means ownership accuracy cannot be calculated.",
                               "More assignments are not necessarily better; intentional abstention can reduce false associations.",
                               "Source detection and classifier errors are retained; no oracle boxes or classes used."]}
    # Ground truth is loaded only AFTER prediction construction.
    if job.get("reference"):
        reference = read_json(job["reference"])
        metrics["old_reference_eval"] = evaluate_reference(old, reference)
        metrics["new_reference_eval"] = evaluate_reference(new["components"], reference)
    provenance = {"image": str(Path(job["image"]).resolve()),
                  "detections": str(Path(job["detections"]).resolve()), "ocr": str(Path(job["ocr"]).resolve()),
                  "sha256": {key: hashlib.sha256(Path(job[key]).read_bytes()).hexdigest() for key in ("image", "detections", "ocr")},
                  "prediction_uses_reference": False, "paid_api_calls": 0}
    new["provenance"] = provenance
    for name, data in [("association.json", new), ("legacy_association.json", {"components": old, "unassigned_texts": old_unassigned}),
                       ("metrics.json", metrics), ("provenance.json", provenance)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_gallery(output, job["image"], old, new, texts, metrics)
    report = ["# OCR角色与关联离线对照", "", "该结果是影子实验，未修改生产推理或审核数据。原基线为固定60像素最近距离；新方法为文字高度归一化距离、角色解析、文字组与全局字段匹配。", "",
              "|指标（代理，非准确率）|原基线|新方法|", "|---|---:|---:|"]
    for key in ("components", "ocr_regions", "references_assigned", "values_assigned", "part_numbers_assigned", "texts_with_component_owner", "unassigned_texts"):
        report.append(f'|{key}|{metrics["old"][key]}|{metrics["new"][key]}|')
    report += ["", "没有人工确认的OCR文字归属金标准时，无法判断新增归属是否正确；未关联也可能是正确保留网络名、引脚号或说明。不得将覆盖率写成准确率。", "",
               "原始OCR文字、规则候选、OCR分数和匹配依据保存在association.json；识别分数与规则分数均未校准。型号只保留可见候选，不从电路功能推测。", "",
               "[交互对照](gallery.html) · [结构化结果](association.json) · [指标](metrics.json)"]
    if job.get("reference"):
        report += ["", "可选参考标注仅由评估器读取；字段得分限于IoU≥0.5匹配的检测框，并非端到端准确率。详见metrics.json。"]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return {"id": job.get("id", Path(job["image"]).stem), "output": str(output.resolve()), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image")
    parser.add_argument("--detections")
    parser.add_argument("--ocr")
    parser.add_argument("--reference", help="Evaluator-only optional field labels; not used for predictions")
    parser.add_argument("--manifest")
    parser.add_argument("--output", default=str(ROOT / "outputs/unattended_20261016/ocr_association"))
    args = parser.parse_args()
    output = Path(args.output)
    if args.manifest:
        manifest = Path(args.manifest).resolve()
        jobs = rows_from(read_json(manifest), ("jobs", "images"))
        for job in jobs:
            for key in ("image", "detections", "ocr", "reference"):
                if job.get(key):
                    path = Path(job[key])
                    job[key] = str(path if path.is_absolute() else manifest.parent / path)
        summary = []
        for index, job in enumerate(jobs, 1):
            identifier = str(job.get("id", Path(job["image"]).stem))
            if Path(identifier).name != identifier or identifier in {".", ".."}:
                raise ValueError("Job id must be a safe single path segment")
            summary.append(run_job(job, output / f"{index:02d}_{identifier}"))
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        count_keys = ("components", "ocr_regions", "references_assigned", "values_assigned", "part_numbers_assigned", "texts_with_component_owner", "unassigned_texts")
        aggregate = {"jobs_completed": len(summary), "evaluation_kind": "coverage_and_consistency_proxies_not_accuracy",
                     "old": {key: sum(item["metrics"]["old"][key] for item in summary) for key in count_keys},
                     "new": {key: sum(item["metrics"]["new"][key] for item in summary) for key in count_keys},
                     "paid_api_calls": 0, "prediction_uses_reference": False}
        (output / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        links = ''.join(f'<li><a href="{Path(item["output"]).name}/gallery.html">{html.escape(item["id"])}</a></li>' for item in summary)
        (output / "gallery.html").write_text('<!doctype html><meta charset="utf-8"><h1>OCR归属批量影子实验</h1><p>覆盖率不是准确率；逐图检查原文与归属连线。</p><pre>' + html.escape(json.dumps(aggregate, ensure_ascii=False, indent=2)) + '</pre><ul>' + links + '</ul>', encoding="utf-8")
        report = ["# OCR角色与归属批量影子实验", "", f"完成{len(summary)}张原图。固定实际检测与分类输出，只比较文字解析与归属模块；没有付费API调用，未改生产流程或人工标注。", "",
                  "|指标（代理，非准确率）|原基线|新方法|", "|---|---:|---:|"]
        report += [f'|{key}|{aggregate["old"][key]}|{aggregate["new"][key]}|' for key in count_keys]
        report += ["", "代理计数不能证明新方法更准确：网络名、引脚号、说明文字本来就应留给后续阶段；参考字段即使存在，也须注意原始标签未审计和只评估匹配检测框的限制。", "",
                   "[交互对照图库](gallery.html) · [逐图指标](summary.json) · [汇总](aggregate.json)"]
        (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
        print(json.dumps({"jobs_completed": len(summary), "output": str(output.resolve())}))
    else:
        if not all((args.image, args.detections, args.ocr)):
            parser.error("Supply --manifest or all of --image --detections --ocr")
        result = run_job({"image": args.image, "detections": args.detections, "ocr": args.ocr, "reference": args.reference}, output)
        print(json.dumps({"output": result["output"], "old": result["metrics"]["old"], "new": result["metrics"]["new"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
