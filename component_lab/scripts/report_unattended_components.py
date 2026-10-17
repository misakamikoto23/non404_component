"""Assemble local autonomous-experiment evidence; never promotes model outputs."""
import argparse
import hashlib
import html
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "outputs/unattended_20261016"


def read(relative):
    path = OUT / relative
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    candidates = read("candidates_val/evaluation_summary.json")
    structure = read("structure_checks/summary.json")
    local = read("local_recheck/summary.json")
    native = read("native_labels/SUMMARY.json")
    calibration = read("native_labels/calibration_v2/SUMMARY.json")
    ocr = {folder.name: json.loads((folder / "aggregate.json").read_text(encoding="utf-8"))
           for folder in (OUT / "ocr_association").iterdir() if (folder / "aggregate.json").exists()}
    if args.complete and (not all((candidates, structure, local, native)) or not {"tesseract", "rapidocr"} <= set(ocr)):
        raise ValueError("Required real experiment results missing")
    integrity = {}
    source = LAB / "data/yolo/pcbnet_component_v1"
    current = LAB / "data/reviewed_versions/manual_round_004/yolo"
    for split in ("val", "test"):
        for kind in ("images", "labels"):
            for path in (source / kind / split).iterdir():
                if path.is_file() and path.suffix != ".cache":
                    relative = path.relative_to(source)
                    integrity[relative.as_posix()] = sha(path) == sha(current / relative)
    if not all(integrity.values()):
        raise ValueError("Holdout changed")
    save(OUT / "input_integrity.json", {"checked_files": len(integrity), "all_unchanged": True, "files": integrity,
             "resnet_sha256": sha(LAB / "weights/resnet18_component_balanced_v1.pt"),
             "yolo_sha256": sha(LAB / "weights/pcbnet_component_yolo26n_review_round_004.pt")})
    environment = {"rapidocr_model_sha256": {p.name: sha(p) for p in sorted((LAB / "weights/ocr_rapid").glob("*.onnx"))},
                   "local_script_sha256": {p.name: sha(p) for p in (LAB / "scripts").glob("*ocr*.py")},
                   "requirements_file": "ocr_requirements.txt"}
    save(OUT / "environment_provenance.json", environment)
    lines = ["# 无人实验阶段报告", "", "2026-10-16，日本时间。结果均来自本地真实运行；新策略保持旁路，没有替换人工标注或现有模型。", "",
             "## 检测与结构", "", "以下为原30张val、1044个参考框、IoU≥0.5的固定阈值计数，属于开发对照，不是mAP，也不是独立最终测试。", "",
             "| 策略 | TP | FP | FN | P | R |", "|---|---:|---:|---:|---:|---:|"]
    if candidates:
        rows = [("整图+切图，conf0.25", candidates["metrics"]["baseline"]),
                ("保留原始候选，conf0.03", candidates["metrics"]["raw"])]
        if structure:
            rows.append(("保守页框过滤", structure["metrics"]["frame_guard"]))
        if local:
            rows += [("基线+局部0.25", local["metrics"]["base25_plus_local25"]),
                     ("基线+局部0.10", local["metrics"]["base25_plus_local10"])]
        for title, row in rows:
            lines.append(f"| {title} | {row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.2%} | {row['recall']:.2%} |")
    lines += ["", "低阈值原始候选的召回达到约95%，但误框极多，不能称为双95%系统。保守页框策略没有误删，也没有带来提升；24区域局部重检没有增加TP，增加了FP，不启用。", "",
              "结构冲突标记覆盖192个基线候选，其中126个原真匹配、66个假阳性，说明标记只能用于复核，不能直接删除。", "",
              "## OCR与文字归属", "", "固定相同YOLO+ResNet候选，比较最近距离关联与角色/文字组匹配。以下仅为字段填充计数，不代表字段正确率。", "",
              "| OCR引擎/视图 | 图片 | 文字区域 | 旧/新位号数 | 旧/新参数数 | 新型号候选 |", "|---|---:|---:|---:|---:|---:|"]
    for name, result in sorted(ocr.items()):
        a, b = result["old"], result["new"]
        lines.append(f"| {name} | {result['jobs_completed']} | {b['ocr_regions']} | {a['references_assigned']}/{b['references_assigned']} | {a['values_assigned']}/{b['values_assigned']} | {b['part_numbers_assigned']} |")
    lines += ["", "原文、坐标、多边形、分数、候选角色、归属证据和未归属原因均保存。引脚号/网络名不强行填入元件参数；程序内部ID与图上位号分开。缺独立人工文字归属真值，因此不能宣称OCR或关联达到95%。", "",
              "RapidOCR使用PP-OCRv5移动版检测与识别、PP-OCRv4方向分类，CPU限制线程；Tesseract使用eng+chi_sim、psm11。初版ONNX1.31原生崩溃，已在独立.venv-ocr中固定ONNX1.20.1/numpy1.26.4/OpenCV4.11，不影响YOLO环境。", "",
              "超大0046图的12切片上限使其回退全图，另以24切片上限保留单独dense24对照，不覆盖10图基线。文字数量变多不等于准确率提高。", "",
              "## 原生工程自动标签", ""]
    if native:
        count = native["counts"]
        lines += [f"实际解析{native['input_records']}页，成功{count['parsed_files']}页，提取{count['symbols']}个符号、"
                  f"{count['pins']}个引脚和{count['symbol_bbox_candidates']}个几何框候选；源事实包括位号、值、库ID、unit、实例路径。",
                  "", "源项目及精确重复临时分组共998组；尚未完成fork、近重复和与现有数据重叠的完整隔离。无KiCadCLI/SVG独立重渲染，原生框仍为待校准候选；新增正式训练标签为0。更强坐标诊断见原生子目录。", ""]
    if calibration:
        c = calibration["counts"]
        lines += [f"增强诊断{c['images']}页：{c['registration_supported']}页通过导线对齐门槛，"
                  f"{c['pin_alignment_supported']}页通过可见引脚线对齐门槛，{c['both_supported']}页同时通过。"
                  "heldout指同一源图内独立导线线段，不是独立渲染或识别准确率。显式配色、镜像、旋转及DNP状态分别处理。", ""]
    lines += ["## 结果入口", "", "- [OCR RapidOCR图库](ocr_association/rapidocr/gallery.html)",
              "- [OCR Tesseract图库](ocr_association/tesseract/gallery.html)",
              "- [边框及拆合诊断](structure_checks/gallery.html)",
              "- [局部重检对照](local_recheck/REPORT.md)",
              "- [原生工程几何图库](native_labels/gallery.html)",
              "- [原生几何增强诊断](native_labels/calibration_v2/REPORT.md)",
              "- [最大图片密集OCR对照](ocr_association/dense24/gallery.html)",
              "- [计划与边界](PLAN.md)", "",
              "## 下一步优先级", "", "先校准源文件到渲染图的几何，再严格处理库类别与原项目划分；有望同时提供检测、分类、文字归属监督。"
              "对无源图片，优先改善实例完整性与文字归属，暂停没有收益证据的重复训练和重检。", "",
              "本轮新增付费API请求为0，没有恢复Small训练，没有改动现有ResNet及其27类均衡训练数据。原验证/测试120个文件逐一校验未变。", ""]
    tests = OUT / "tests.log"
    if tests.exists():
        lines += ["## 验证", "", "OCR文字归属、结构诊断、原生解析及几何校准的组合测试日志见[tests.log](tests.log)。"
                  "已检查真实OCR与原生框/引脚叠图。新模块为独立影子入口，尚未替换生产推理关联器。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    cards = []
    for title, link in [("OCR：RapidOCR与元件归属", "ocr_association/rapidocr/gallery.html"),
                        ("OCR：Tesseract对照", "ocr_association/tesseract/gallery.html"),
                        ("边框与拆合冲突", "structure_checks/gallery.html"),
                        ("原生KiCad几何与引脚", "native_labels/gallery.html"),
                        ("局部重检指标", "local_recheck/REPORT.md")]:
        cards.append(f'<article><h2><a href="{link}">{html.escape(title)}</a></h2></article>')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>PCB无人实验</title><style>body{font:16px system-ui;margin:32px;max-width:1100px;background:#f4f6fa;color:#182333}article{background:white;padding:16px;margin:16px 0;border-radius:12px}a{color:#245ca3}pre{white-space:pre-wrap}</style><h1>PCB无人实验 · 2026-10-16</h1><p>本地真实运行；新策略保持旁路。字段填充和候选覆盖不是准确率。</p><p><a href="REPORT.md">完整报告</a> · <a href="PLAN.md">运行计划</a> · <a href="STATUS.json">状态</a></p>' + ''.join(cards) + '<details><summary>当前报告内容</summary><pre>' + html.escape('\n'.join(lines)) + '</pre></details></html>'
    (OUT / "gallery.html").write_text(page, encoding="utf-8")
    status = read("STATUS.json")
    status.update(status="complete" if args.complete else "running", jobs={
        "candidate_cache": "complete" if candidates else "pending", "ocr_engine": "two_local_engines_complete" if len(ocr)>=2 else "running",
        "ocr_matching": "real_comparison_complete" if len(ocr)>=2 else "running", "structure_checks": "complete_no_improvement" if local else "running",
        "native_labels": "1000_sources_extracted_calibration_diagnostic" if native else "running", "integration_report": "complete" if args.complete else "checkpoint"})
    save(OUT / "STATUS.json", status)
    save(OUT / "SUMMARY.json", {"complete": args.complete, "detection_metrics": candidates["metrics"] if candidates else None,
        "local_recheck": local["metrics"] if local else None, "ocr": ocr, "native": native, "native_calibration": calibration,
        "holdout_files_unchanged": len(integrity), "paid_api_calls": 0})
    print(json.dumps({"status": status["status"], "ocr_arms": list(ocr), "report": str(OUT / "REPORT.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
