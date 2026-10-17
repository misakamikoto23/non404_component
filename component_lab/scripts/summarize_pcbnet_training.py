"""Finalize the baseline artifact and a short reproducible training report."""
import csv
import hashlib
import json
import shutil
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    run=root/"runs/pcbnet34_yolo26n_v1"
    summary=json.loads((run/"baseline_summary.json").read_text(encoding="utf-8"))
    if summary["status"]!="training_complete":
        raise ValueError("Training is not complete.")
    meta=json.loads((root/"configs/pcbnet34_baseline_metadata.json").read_text(encoding="utf-8"))
    destination=root/"weights/pcbnet34_yolo26n_v1.pt"
    shutil.copy2(run/"weights/best.pt",destination)
    rows=list(csv.DictReader((run/"results.csv").open(encoding="utf-8")))
    best=max(rows,key=lambda r:float(r["metrics/mAP50-95(B)"]))
    peaks=summary["gpu_memory"]
    def format_metrics(metrics):
        return "\n".join(f"- {k}: {float(v):.4f}" for k,v in metrics.items())
    report=f"""# PCBnet YOLO26 Nano 初步训练

本地训练完成，日期 2026-10-15。此模型是34类原始PCBnet标签的检测基线，未训练ResNet，也未调用Qwen。

## 模型与复现

- 推荐推理权重：weights/pcbnet34_yolo26n_v1.pt。
- 原始最佳权重：runs/pcbnet34_yolo26n_v1/weights/best.pt。
- 权重 SHA256：{hashlib.sha256(destination.read_bytes()).hexdigest()}。
- 完成 {len(rows)} 轮，最佳验证 mAP50-95 出现在第 {best['epoch']} 轮。
- 初始权重为官方 yolo26n.pt（成功迁移696/708参数项）。
- 实际参数和曲线：runs/pcbnet34_yolo26n_v1/args.yaml、results.csv、results.png。
- 独立环境 .venv-yolo 复用本机已装 CUDA PyTorch；新增依赖版本见 configs/yolo_environment_local.txt。

复现训练（默认会产生新运行目录）：

```powershell
.\\.venv-yolo\\Scripts\\python.exe scripts/train_pcbnet_baseline.py --imgsz 1280 --batch 2 --epochs 20 --name pcbnet34_yolo26n_repeat
```

## 数据

data/yolo/pcbnet34_v1/dataset.yaml 是训练入口；manifest.json 记录来源、34类映射、逐图框、去重和数据修复。
293张独立图，train233/val30/test30；框数量10126/1044/1366。原验证与测试图片保持原归属，哈希未跨分区。
删除1条deleted记录，裁剪3个越界框。仅使用标准CVJsonStd，不混入四份手工标注。

## 硬件与参数

- RTX3050Ti Laptop，4GB；1280输入，batch2，workers0，AMP。
- PyTorch训练峰值已分配显存：{max(x['allocated_peak_mib'] for x in peaks):.1f} MiB。
- PyTorch峰值预留显存：{max(x['reserved_peak_mib'] for x in peaks):.1f} MiB；与整个GPU占用不是同一统计口径。
- AdamW，lr0=0.001，20轮；禁止镜像/旋转/mosaic，少量亮度、平移和尺度增强。

## 最佳权重验证集指标

{format_metrics(summary['validation'])}

## 测试集指标

{format_metrics(summary['test'])}

## 当前限制

验证集缺少：{', '.join(meta['missing_validation_classes'])}。
测试集缺少：{', '.join(meta['missing_test_classes'])}。
指标仅覆盖评估集合出现的类别；不能解释为34类全部达到相同效果。
9类训练样本少于10个；整图缩小仍容易漏掉极小符号。去重检查只覆盖完全相同像素，未充分排除同一电路的不同渲染版本。
默认nbs64意味着batch2预热后累计32批再更新，20轮是初版，可能尚未充分收敛。
YOLO的框/类别指标不等于比赛ComponentF1，后者还需要编号与必要文字正确。
"""
    (root/"TRAINING_PCBNET.md").write_text(report,encoding="utf-8")
    print(f"Published model: {destination}")


if __name__=="__main__":main()
