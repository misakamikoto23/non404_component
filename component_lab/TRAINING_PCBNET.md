# PCBnet YOLO26 Nano 初步训练

本地训练完成，日期 2026-10-15。此模型是34类原始PCBnet标签的检测基线，未训练ResNet，也未调用Qwen。

## 模型与复现

- 推荐推理权重：weights/pcbnet34_yolo26n_v1.pt。
- 原始最佳权重：runs/pcbnet34_yolo26n_v1/weights/best.pt。
- 权重 SHA256：c6de481f62ecd1fd8e3376827de936997f4ba687006adb92ac9a4e4e9fbf4c69。
- 完成 20 轮，最佳验证 mAP50-95 出现在第 20 轮。
- 初始权重为官方 yolo26n.pt（成功迁移696/708参数项）。
- 实际参数和曲线：runs/pcbnet34_yolo26n_v1/args.yaml、results.csv、results.png。
- 独立环境 .venv-yolo 复用本机已装 CUDA PyTorch；新增依赖版本见 configs/yolo_environment_local.txt。

复现训练（默认会产生新运行目录）：

```powershell
.\.venv-yolo\Scripts\python.exe scripts/train_pcbnet_baseline.py --imgsz 1280 --batch 2 --epochs 20 --name pcbnet34_yolo26n_repeat
```

## 数据

data/yolo/pcbnet34_v1/dataset.yaml 是训练入口；manifest.json 记录来源、34类映射、逐图框、去重和数据修复。
293张独立图，train233/val30/test30；框数量10126/1044/1366。原验证与测试图片保持原归属，哈希未跨分区。
删除1条deleted记录，裁剪3个越界框。仅使用标准CVJsonStd，不混入四份手工标注。

## 硬件与参数

- RTX3050Ti Laptop，4GB；1280输入，batch2，workers0，AMP。
- PyTorch训练峰值已分配显存：1573.8 MiB。
- PyTorch峰值预留显存：2076.0 MiB；与整个GPU占用不是同一统计口径。
- AdamW，lr0=0.001，20轮；禁止镜像/旋转/mosaic，少量亮度、平移和尺度增强。

## 最佳权重验证集指标

- metrics/precision(B): 0.4888
- metrics/recall(B): 0.3413
- metrics/mAP50(B): 0.3566
- metrics/mAP50-95(B): 0.2293
- fitness: 0.2293

## 测试集指标

- metrics/precision(B): 0.5325
- metrics/recall(B): 0.3221
- metrics/mAP50(B): 0.3100
- metrics/mAP50-95(B): 0.1981
- fitness: 0.1981

## 当前限制

验证集缺少：CIRCLE_HEADER, Dc, Jumper, M, Net_short, Opt, Transfomer。
测试集缺少：Block, CIRCLE_HEADER, ESD, Jumper, M, Motor, Net_short, Opt, SEG, Transfomer。
指标仅覆盖评估集合出现的类别；不能解释为34类全部达到相同效果。
9类训练样本少于10个；整图缩小仍容易漏掉极小符号。去重检查只覆盖完全相同像素，未充分排除同一电路的不同渲染版本。
默认nbs64意味着batch2预热后累计32批再更新，20轮是初版，可能尚未充分收敛。
YOLO的框/类别指标不等于比赛ComponentF1，后者还需要编号与必要文字正确。

## 已验证的预测入口

在 component_lab 目录运行：

```powershell
.\.venv-yolo\Scripts\component-lab.exe infer --image "你的原理图.png" --yolo weights/pcbnet34_yolo26n_v1.pt --device 0 --imgsz 1280 --output outputs/my_prediction
```

三张测试图已通过现有 component pipeline：outputs/pcbnet34_predictions/0263、0135、0111。每张均包含 components.json、overlay.png 和 crops/。没有启用OCR/ResNet/Qwen，因此只输出框与34类预测，真实位号和文字字段保留空值。

验证范围：独立数据配对/标签/哈希划分核验通过，真实GPU训练与测试评估完成，三张真实测试图片推理和裁剪成功，7项基础测试通过。
