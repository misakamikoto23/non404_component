"""Report crop classification agreement with frozen original-label proxies.

No API, model, training operation, or matching recomputation is performed.
Repeated --predictions paths are merged; conflicting categories fail closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


THRESHOLDS = (0.5, 0.65, 0.7)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(row)
    return rows


def load_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    claimed = set()
    for row in read_jsonl(path):
        crop_id = row["crop_id"]
        if crop_id in rows:
            raise ValueError(f"Duplicate crop_id in manifest: {crop_id}")
        proxy = row.get("evaluation_proxy", {})
        component_id = proxy.get("matched_original_component_id")
        if component_id is not None:
            key = (row["image_id"], component_id)
            if key in claimed:
                raise ValueError(f"Primary proxy is not one-to-one: {key}")
            claimed.add(key)
        rows[crop_id] = row
    return rows


def load_predictions(paths: list[Path], allowed: set[str], manifest_ids: set[str]) -> tuple[dict, int]:
    predictions, duplicate_count = {}, 0
    for path in paths:
        for row in read_jsonl(path):
            crop_id = row.get("id")
            category = row.get("category", row.get("semantic_family"))
            score, ambiguous = row.get("confidence"), row.get("ambiguous")
            if not isinstance(crop_id, str) or crop_id not in manifest_ids:
                raise ValueError(f"Unknown or invalid prediction id in {path}: {crop_id!r}")
            if category not in allowed:
                raise ValueError(f"Unsupported category for {crop_id}: {category!r}")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid confidence for {crop_id}: {score!r}")
            if not isinstance(ambiguous, bool):
                raise ValueError(f"Expected boolean ambiguous for {crop_id}")
            if crop_id in predictions:
                previous = predictions[crop_id]
                if category != previous["category"]:
                    raise ValueError(f"Conflicting duplicate prediction id {crop_id}: {previous['category']!r} versus {category!r}")
                # Conservative merge prevents choosing the most confident repeat.
                previous["confidence"] = min(previous["confidence"], float(score))
                previous["ambiguous"] |= ambiguous
                previous["sources"].append(str(path.resolve()))
                duplicate_count += 1
            else:
                predictions[crop_id] = {"id": crop_id, "category": category, "confidence": float(score),
                                        "ambiguous": ambiguous, "sources": [str(path.resolve())]}
    return predictions, duplicate_count


def proxy_for(row: dict, mapping: dict, threshold: float) -> tuple[str | None, str]:
    """Use only the manifest's primary frozen match, never model output."""
    proxy = row.get("evaluation_proxy", {})
    if proxy.get("ambiguous") is not False:
        return None, "ambiguous_or_unspecified_original_geometry_match"
    overlap = proxy.get("matched_iou")
    if not isinstance(overlap, (int, float)) or not math.isfinite(overlap) or overlap < threshold:
        return None, "below_iou_threshold_or_unmatched"
    raw = proxy.get("matched_original_category")
    if raw not in mapping:
        return None, "mixed_or_unmapped_original_category"
    return mapping[raw], "eligible_original_label_proxy"


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(manifest: dict, predictions: dict, mapping: dict, labels: list[str], threshold: float,
             high_confidence: float) -> dict:
    eligible_all = {key for key, row in manifest.items() if proxy_for(row, mapping, threshold)[0] is not None}
    class_counts, class_correct, confusion = Counter(), Counter(), Counter()
    exclusions, by_image, wrong_ids, high_wrong = Counter(), {}, [], []
    model_unknown = model_ambiguous = correct = high_total = high_correct = selective_total = selective_correct = 0
    for crop_id, prediction in predictions.items():
        row = manifest[crop_id]
        expected, reason = proxy_for(row, mapping, threshold)
        if expected is None:
            exclusions[reason] += 1
            continue
        predicted = prediction["category"]
        agrees = predicted == expected
        correct += agrees
        class_counts[expected] += 1
        class_correct[expected] += agrees
        confusion[(expected, predicted)] += 1
        model_unknown += predicted == "unknown"
        model_ambiguous += prediction["ambiguous"]
        source = by_image.setdefault(row["image_id"], {"n": 0, "correct": 0})
        source["n"] += 1
        source["correct"] += agrees
        if not agrees:
            wrong_ids.append(crop_id)
        if prediction["confidence"] >= high_confidence:
            high_total += 1
            high_correct += agrees
            if not agrees:
                high_wrong.append(crop_id)
            if predicted != "unknown" and not prediction["ambiguous"]:
                selective_total += 1
                selective_correct += agrees
    n = sum(class_counts.values())
    per_class = [{"family": name, "n": class_counts[name], "correct": class_correct[name],
                  "agreement": fraction(class_correct[name], class_counts[name])} for name in labels]
    recalls = [row["agreement"] for row in per_class if row["n"]]
    for source in by_image.values():
        source["agreement"] = fraction(source["correct"], source["n"])
    return {"iou_threshold": threshold, "manifest_eligible": len(eligible_all), "predicted_eligible": n,
            "coverage_of_all_manifest_crops": fraction(n, len(manifest)),
            "prediction_coverage_of_eligible_manifest": fraction(n, len(eligible_all)),
            "correct": correct, "incorrect": n - correct, "agreement": fraction(correct, n),
            "macro_recall_over_supported_families": sum(recalls) / len(recalls) if recalls else None,
            "supported_families": len(recalls), "unknown_count": model_unknown,
            "unknown_rate": fraction(model_unknown, n), "model_ambiguous_count": model_ambiguous,
            "model_ambiguous_rate": fraction(model_ambiguous, n),
            "high_confidence_threshold": high_confidence, "high_confidence_n": high_total,
            "high_confidence_correct": high_correct, "high_confidence_error_count": len(high_wrong),
            "high_confidence_error_rate": fraction(len(high_wrong), high_total),
            "high_confidence_wrong_ids": high_wrong, "wrong_ids": wrong_ids,
            "selective_nonunknown_nonambiguous_high_confidence_n": selective_total,
            "selective_nonunknown_nonambiguous_high_confidence_agreement": fraction(selective_correct, selective_total),
            "exclusion_counts_within_predictions": dict(exclusions), "per_class": per_class,
            "confusion_pairs": [{"proxy_family": actual, "predicted_family": predicted, "n": count,
                                  "agrees": actual == predicted}
                                 for (actual, predicted), count in sorted(confusion.items(), key=lambda pair: (-pair[1], pair[0]))],
            "confusion_matrix_proxy_rows_prediction_columns": [[confusion[(a, b)] for b in labels] for a in labels],
            "by_source_image": by_image}


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def image_url(value: str, manifest_dir: Path, output: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Crop image is missing: {path}")
    return quote(os.path.relpath(path, output).replace("\\", "/"), safe="/:")


def write_gallery(output: Path, manifest_path: Path, manifest: dict, predictions: dict, mapping: dict,
                  labels: list[str], report: dict) -> None:
    cards = []
    for crop_id, prediction in predictions.items():
        row = manifest[crop_id]
        expected, reason = proxy_for(row, mapping, 0.65)
        proxy = row.get("evaluation_proxy", {})
        status = ("unknown" if prediction["category"] == "unknown" else "disagreement" if expected is not None and prediction["category"] != expected
                  else "agreement" if expected is not None else "no_proxy")
        cards.append({"id": crop_id, "source": row["image_id"], "category": prediction["category"],
                      "confidence": prediction["confidence"], "ambiguous": prediction["ambiguous"],
                      "expected": expected, "raw": proxy.get("matched_original_category"),
                      "iou": proxy.get("matched_iou"), "reason": reason, "status": status,
                      "tight": image_url(row["tight_image_path"], manifest_path.parent, output),
                      "context": image_url(row["context_image_path"], manifest_path.parent, output)})
    order = {"disagreement": 0, "unknown": 1, "no_proxy": 3, "agreement": 4}
    cards.sort(key=lambda row: (order[row["status"]] if not row["ambiguous"] else min(order[row["status"]], 2),
                               -row["confidence"], row["id"]))
    data = json.dumps(cards, ensure_ascii=False).replace("<", "\\u003c")
    options = "".join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in labels)
    main = report["thresholds"]["0.65"]
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen 单元件分类：代理标签对比</title><style>
*{box-sizing:border-box}body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f4f6fa;color:#14213a}header{padding:24px 28px;background:#fff;border-bottom:1px solid #d9dfeb}h1{margin:0 0 12px;font-size:24px}p{line-height:1.6;margin:7px 0}.notice{color:#805000}.controls{position:sticky;top:0;background:#edf1f8;padding:14px 28px;display:flex;gap:12px;flex-wrap:wrap;z-index:1}select,input{padding:9px;border:1px solid #bdc9dd;border-radius:7px;background:white}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(315px,1fr));gap:16px;padding:22px}.card{background:#fff;border:1px solid #d4dce8;border-top:5px solid #8c9caf;border-radius:10px;padding:13px;overflow:hidden}.disagreement{border-top-color:#d74343}.unknown{border-top-color:#d89420}.agreement{border-top-color:#369f76}.images{display:flex;gap:8px;background:#eef1f5;padding:6px}.image{width:50%;text-align:center;color:#667;font-size:12px}.image img{display:block;object-fit:contain;background:#fff;width:100%;height:165px;image-rendering:auto}.id{font-weight:700;margin-bottom:8px}.predict{font-size:19px;font-weight:700;margin-top:10px}.detail{font-size:13px;line-height:1.7;overflow-wrap:anywhere}.muted{color:#68788f}#shown{align-self:center}button{padding:9px;cursor:pointer}
</style><header><h1>Qwen 单元件分类：代理标签对比</h1><p>__SUMMARY__</p><p class="notice">原类别仅是 PCBnet 的历史标注代理，人工审核确认的是框，未确认类别。红色代表与代理不一致，并不能直接认定模型错误。本批来自 20 张训练图，不是人工分类测试集。</p><p>图片左侧为人工框裁剪，右侧为带少量边缘的裁剪。比较不包含类别混杂或几何不确定的原标签。模型 unknown 计入错误；模型 ambiguous 不从主指标分母删除。</p></header>
<div class="controls"><select id="status"><option value="all">全部预测</option><option value="disagreement">与代理不一致</option><option value="unknown">模型 unknown</option><option value="ambiguous">模型 ambiguous</option><option value="highwrong">高自信且不一致</option><option value="agreement">与代理一致</option><option value="no_proxy">无可靠代理</option></select><select id="category"><option value="all">全部预测类别</option>__OPTIONS__</select><input id="search" placeholder="搜索裁剪 ID / 原类别 / 类别"><span id="shown"></span><button id="reset">重置筛选</button></div><main id="cards"></main>
<script>const records=__DATA__;const highThreshold=__HIGH__;const labels={disagreement:'与代理不一致',unknown:'模型 unknown',agreement:'与代理一致',no_proxy:'无可靠代理'};const root=document.querySelector('#cards');const status=document.querySelector('#status'),category=document.querySelector('#category'),search=document.querySelector('#search');
function text(tag,value,cls){const n=document.createElement(tag);n.textContent=value;if(cls)n.className=cls;return n}
function render(){root.replaceChildren();const wanted=search.value.trim().toLowerCase();let shown=0;for(const r of records){if(status.value!=='all'){if(status.value==='ambiguous'){if(!r.ambiguous)continue}else if(status.value==='highwrong'){if(r.expected===null||r.category===r.expected||r.confidence<highThreshold)continue}else if(r.status!==status.value)continue}if(category.value!=='all'&&r.category!==category.value)continue;if(wanted&&!JSON.stringify([r.id,r.source,r.category,r.raw,r.expected]).toLowerCase().includes(wanted))continue;shown++;const card=text('article','','card '+r.status);card.append(text('div',r.id+' · 来源图 '+r.source,'id'));const imgs=text('div','','images');for(const [label,url] of [['人工框',r.tight],['少量上下文',r.context]]){const wrap=text('div','','image');const a=document.createElement('a');a.href=url;a.target='_blank';a.rel='noopener';const im=document.createElement('img');im.src=url;im.loading='lazy';im.alt=r.id+' '+label;a.append(im);wrap.append(a,text('div',label));imgs.append(wrap)}card.append(imgs,text('div','预测：'+r.category,'predict'),text('div','自报置信度 '+(100*r.confidence).toFixed(0)+'%'+(r.ambiguous?' · ambiguous=true':''),'detail'),text('div',labels[r.status],'detail'),text('div','原类别代理：'+(r.raw??'无')+(r.expected?' → '+r.expected:'（不进入主评价）'),'detail'),text('div','匹配 IoU：'+(r.iou===null||r.iou===undefined?'无':r.iou.toFixed(3)),'detail muted'));root.append(card)}document.querySelector('#shown').textContent=shown+' / '+records.length+' 个裁剪'}for(const el of [status,category,search])el.addEventListener('input',render);document.querySelector('#reset').onclick=()=>{status.value='all';category.value='all';search.value='';render()};render();</script></html>'''
    summary = (f"已分类 {len(predictions)} / {len(manifest)} 个裁剪；主口径 IoU ≥ 0.65，可比较 {main['predicted_eligible']} 个，"
               f"语义一致率 {percent(main['agreement'])}，宏平均召回 {percent(main['macro_recall_over_supported_families'])}。")
    page = (page.replace("__SUMMARY__", html.escape(summary)).replace("__OPTIONS__", options)
            .replace("__DATA__", data).replace("__HIGH__", str(main["high_confidence_threshold"])))
    (output / "gallery.html").write_text(page, encoding="utf-8")


def save_report(output: Path, manifest_path: Path, manifest: dict, predictions: dict, ontology: dict,
                report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iou_threshold", "family", "n", "correct", "agreement"])
        writer.writeheader()
        for threshold, result in report["thresholds"].items():
            for row in result["per_class"]:
                writer.writerow({"iou_threshold": threshold, **row})
    with (output / "confusion.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iou_threshold", "proxy_family", "predicted_family", "n", "agrees"])
        writer.writeheader()
        for threshold, result in report["thresholds"].items():
            for row in result["confusion_pairs"]:
                writer.writerow({"iou_threshold": threshold, **row})
    write_gallery(output, manifest_path, manifest, predictions, ontology["raw_to_family_for_unambiguous"], ontology["allowed_labels"], report)
    lines = ["# Qwen 单元件分类代理评价", "", "本报告是与原 PCBnet 标签的语义一致性研究，**不是人工确认的分类 accuracy**。",
             "20 张来源图均为训练集审核样本；人工确认了框，没有确认类别。本报告没有训练模型或修改训练数据。", "",
             f"清单包含 {len(manifest)} 个裁剪、{report['source_image_count']} 张来源图；本次合并 {len(predictions)} 个预测，缺少 {report['unclassified_count']} 个。",
             f"相同类别的重复 ID 共 {report['same_category_duplicate_count']} 条，按最小 confidence、ambiguous 的逻辑或保守合并；类别冲突会直接报错。", "",
             "| IoU | 全部清单可比较 | 已预测且可比较 n | 一致数 | 语义一致率 | 宏平均召回 |", "|---|---:|---:|---:|---:|---:|"]
    for threshold, result in report["thresholds"].items():
        lines.append(f"| {threshold} | {result['manifest_eligible']} | {result['predicted_eligible']} | {result['correct']} | {percent(result['agreement'])} | {percent(result['macro_recall_over_supported_families'])} |")
    main = report["thresholds"]["0.65"]
    lines += ["", "主口径为 IoU ≥ 0.65。宏平均召回只平均本次有代理样本的类别；所有允许类别（含 n=0）均列在 per_class.csv。",
              f"主口径：unknown {main['unknown_count']} 个（{percent(main['unknown_rate'])}）；模型 ambiguous {main['model_ambiguous_count']} 个（{percent(main['model_ambiguous_rate'])}）。",
              f"confidence ≥ {main['high_confidence_threshold']} 的 {main['high_confidence_n']} 个可比较预测中，{main['high_confidence_error_count']} 个与代理不一致。模型 confidence 是自报分数，并未校准为正确概率。", "",
              "匹配完全采用裁剪清单里预先冻结的主一对一几何匹配。各阈值仅筛选 matched_iou，不重新匹配，不读取模型类别挑选对应关系。原几何匹配 ambiguous 必须为 false，原类别必须存在于 ontology 的保守映射中。",
              "**unknown 算错误；模型自身 ambiguous 不从主评价分母排除。** 缺失预测不参与已预测样本一致率，单独报告覆盖率。",
              "Box、Block、CIRCLE_HEADER、M、Other、PIN、Opt、SEG 等混杂原类别不进入保守代理主指标。它们的预测仍保留在图集。",
              "NET_input、Net_output、Net_bidirection、Net_short 合并为 net_port；方向识别不由该指标衡量。",
              "不一致可能来自模型错误、原标注错误、裁剪上下文不足或类别定义差异。稀有类别样本极少，不能据此声称泛化准确率。",
              "同一原理图的裁剪必须归同一数据划分；本次 train-only 结果不能兼作 ResNet 的最终验证或测试。",
              "未确定类别、unknown 和 ambiguous 结果不可自动作为人工确认的监督标签。", "",
              "文件：gallery.html 提供筛选和裁剪预览；metrics.json 含完整阈值指标、逐图结果、混淆矩阵与高自信不一致 ID；per_class.csv 与 confusion.csv 可直接分析。", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    lab = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=lab / "data/component_crops/review_round_001/manifest.jsonl")
    parser.add_argument("--ontology", type=Path, default=lab / "outputs/qwen_crop_classification_round_001/ontology.json")
    parser.add_argument("--predictions", type=Path, action="append", required=True, help="Repeat to merge nonconflicting JSONL results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--high-confidence", type=float, default=0.9)
    args = parser.parse_args(argv)
    if not 0 <= args.high_confidence <= 1:
        parser.error("--high-confidence must be in [0,1]")
    ontology = json.loads(args.ontology.read_text(encoding="utf-8-sig"))
    manifest = load_manifest(args.manifest)
    predictions, duplicate_count = load_predictions(args.predictions, set(ontology["allowed_labels"]), set(manifest))
    report = {"metric": "agreement_with_original_label_semantic_proxy_not_human_classification_accuracy",
              "primary_iou_threshold": 0.65, "source_split": "train_only_review_batch_not_held_out",
              "manifest": str(args.manifest.resolve()), "ontology": str(args.ontology.resolve()),
              "prediction_files": [str(path.resolve()) for path in args.predictions],
              "manifest_crop_count": len(manifest), "source_image_count": len({row['image_id'] for row in manifest.values()}),
              "classified_count": len(predictions), "unclassified_count": len(manifest) - len(predictions),
              "same_category_duplicate_count": duplicate_count, "classified_coverage": fraction(len(predictions), len(manifest)),
              "predicted_class_counts": dict(Counter(row['category'] for row in predictions.values())),
              "all_prediction_unknown_count": sum(row['category'] == 'unknown' for row in predictions.values()),
              "all_prediction_ambiguous_count": sum(row['ambiguous'] for row in predictions.values()),
              "labels": ontology["allowed_labels"],
              "thresholds": {str(t): evaluate(manifest, predictions, ontology["raw_to_family_for_unambiguous"], ontology["allowed_labels"], t, args.high_confidence) for t in THRESHOLDS}}
    save_report(args.output.resolve(), args.manifest.resolve(), manifest, predictions, ontology, report)
    main_result = report["thresholds"]["0.65"]
    print(json.dumps({"classified": len(predictions), "main_n": main_result["predicted_eligible"],
                      "main_agreement": main_result["agreement"], "macro_recall": main_result["macro_recall_over_supported_families"],
                      "output": str(args.output.resolve())}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
