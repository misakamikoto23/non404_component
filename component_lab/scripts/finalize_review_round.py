"""Publish a completed review-round checkpoint and its comparable val results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

LAB = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--date", required=True, help="User-facing report date, YYYY-MM-DD")
    args = parser.parse_args()
    run = args.run.resolve()
    current = read(run / "baseline_summary.json")
    previous = read(args.previous_run.resolve() / "baseline_summary.json")
    if current["status"] != "training_complete" or "test" in current:
        raise ValueError("Expected a completed validation-only training run")
    report_dir = LAB / "outputs/active_learning" / f"round_{args.round:03d}"
    audit = read(report_dir / "review_audit.json")
    if not audit["holdouts_unchanged"]:
        raise ValueError("Holdout integrity must pass before publishing")
    if Path(current["data"]).resolve() != (Path(audit["data_output"]) / "dataset.yaml").resolve():
        raise ValueError("Training data does not match the frozen review export")
    source = Path(current["best"])
    target = LAB / "weights" / f"pcbnet_component_yolo26n_review_round_{args.round:03d}.pt"
    if target.exists() and sha(target) != sha(source):
        raise ValueError("Refusing to replace an existing different versioned checkpoint")
    if not target.exists():
        shutil.copy2(source, target)
    if sha(source) != sha(target):
        raise ValueError("Checkpoint copy integrity failed")
    keys = [("Precision", "metrics/precision(B)"), ("Recall", "metrics/recall(B)"),
            ("mAP50", "metrics/mAP50(B)"), ("mAP50–95", "metrics/mAP50-95(B)")]
    comparison = [{"metric": title, "previous": previous["validation"][key],
                   "current": current["validation"][key],
                   "delta_percentage_points": 100 * (current["validation"][key] - previous["validation"][key])}
                  for title, key in keys]
    with (run / "results.csv").open(encoding="utf-8", newline="") as stream:
        epochs = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]
    best_row = max(epochs, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    log_text = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                         for path in report_dir.glob("training*.log"))
    result = {"round": args.round, "weights": str(target), "weights_sha256": sha(target),
              "run": str(run), "review_audit": audit,
              "training": current, "previous_run": str(args.previous_run.resolve()),
              "max_val_map50_95_epoch": int(best_row["epoch"]), "comparison": comparison,
              "evaluation_scope": "Original PCBnet val only; no independent HF val, no test evaluation",
              "supervision_scope": "Human-reviewed geometry is not human-confirmed component classification"}
    (report_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    totals = audit["totals"]
    lines = [f"# 第{args.round}轮人工审核后 YOLO 微调", "", f"日期：{args.date}。", "",
             f"本批{audit['completed_images']}张已冻结，累计{audit['cumulative_reviewed_images']}张审核图。"
             f"本批最终{totals['accepted']}个框，删除{totals['rejected']}个、新增{totals['added']}个、"
             f"调整{totals['moved_accepted']}个保留框。",
             f"训练集{audit['split_images']['train']}张、{audit['train_counts']['component']}个框；"
             f"先前{len(audit['earlier_labels_retained'])}张修正已保留，原验证/测试的"
             f"{audit['holdout_files_checked_against_original']}个文件哈希核对一致。", "",
             f"本轮从`{Path(current['initial_model']).name}`继续，4060 Ti / cuda:0，输入{current['imgsz']}，batch{current['batch']}，"
             f"有效batch {current['nbs']}，AdamW lr={current['lr0']}。"
             f"最多{current['epochs_requested']}轮，实际{current['epochs_completed']}轮；"
             f"验证mAP50–95最高为第{int(best_row['epoch'])}轮。", "",
             "| 指标 | 上一轮 | 本轮 | 变化（百分点） |", "|---|---:|---:|---:|"]
    for row in comparison:
        lines.append(f"| {row['metric']} | {row['previous']:.2%} | {row['current']:.2%} | {row['delta_percentage_points']:+.2f} |")
    lines += ["", "P/R为Ultralytics按验证集选择的工作点，不是固定conf=0.25的整数计数。"
              "本轮验证集仍为原PCBnet 30图，不能据此证明HF来源效果；本次没有评估测试集。"
              "判断是否达到双95%应使用独立、口径一致的数据。", "",
              f"权重：[{target.name}]({target.as_posix()})", f"SHA256：`{sha(target)}`", "",
              "## 下一步", "",
              "1. 先保留同一验证口径，观察检测提升和召回/精确率取舍，不将新权重自动等同于更好。",
              "2. 接入文字检测、局部OCR和文字归属可视化，利用位号组诊断框的拆分/合并。",
              "3. 本批确认的是检测几何；分类提示不自动视为ResNet真值，后续分类数据仍需独立质量筛选和类别均衡。",
              f"4. 下一批若继续审核，沿用PCBnet/HF各10张，并排除此前全部{audit['cumulative_reviewed_images']}张及验证/测试来源。", ""]
    if "WinError 5" in log_text:
        lines += ["首次启动被Windows沙盒本地通信管道权限阻止，未进入训练；获得执行权限后在独立运行目录重试。", ""]
    if "AMP will be disabled" in log_text:
        lines += ["本次AMP预检查失败后由Ultralytics自动关闭混合精度，训练日志记录实际行为。", ""]
    source_stats = report_dir / "source_review_statistics.json"
    if source_stats.exists():
        lines += ["## 本批来源统计", "", "| 来源 | 图片 | 最终框 | 新增框 | 删除框 |", "|---|---:|---:|---:|---:|"]
        for row in read(source_stats):
            lines.append(f"| {row['source']} | {row['images']} | {int(row['accepted'])} | {int(row['added'])} | {int(row['rejected'])} |")
        lines += ["", "新增框包含人工补漏及框范围修正等操作，不直接等同于独立测试的漏检数。", ""]
    (report_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"weights": str(target), "comparison": comparison,
                      "epochs_completed": current["epochs_completed"],
                      "max_val_map50_95_epoch": int(best_row["epoch"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
