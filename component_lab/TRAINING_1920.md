# 1920 输入训练实验

日期：2026-10-15。结论：本机 RTX 3050 Ti Laptop GPU 4GB 可以完成
YOLO26 Nano 的 1920、batch 1 训练和验证；当前实验未超过原 1280+切图方案，
默认仍使用原权重 `weights/pcbnet_component_yolo26n_v1.pt`。

## 数据与方法

- 只用 `data/yolo/pcbnet_component_v1/dataset.yaml` 的原始单类数据，
  train 233 张、10,126 框；val 30 张、1,044 框。没有加入手工或 AI 修订标签。
- 从原 1280 Nano 最佳权重微调，迁移 708/708 参数项，重新初始化优化器，
  不属于断点 resume，也不是从零开始训练。
- 输入 1920、batch 1、nbs 16、AdamW、lr0=0.0002、lrf=0.1、
  warmup_epochs=2、warmup_bias_lr=0.0002、AMP、workers=0。
- 其余增强与原实验一致：无旋转/翻转/mosaic，少量亮度、平移、尺度增强。
- 最多 30 轮，patience=10；实际第 16 轮早停，最佳第 6 轮，训练约 20.2 分钟。
- 验证集选模型与比较配置，本轮未在 test 上评估或调参。

训练运行目录：`runs/pcbnet_component_yolo26n_1920_v2/`。
`args.yaml`、`results.csv`、`baseline_summary.json`、`gpu_memory.json`
保存实际配置和结果。前置的 `1920_probe2` 完成了一轮抽样训练及完整验证。
首个正式 `1920_v1` 发现默认 bias 预热学习率 0.1 不适合本次微调后提前停止；
最终 v2 从原始最佳权重重新开始，使用上述较低预热学习率。

## 显存

- 正式训练至最终验证累计 PyTorch 峰值已分配：1868.7 MiB。
- 累计 PyTorch 峰值预留：2116.0 MiB，约 2.07 GiB。
- 运行中 nvidia-smi 抽样观察总占用约 2089 MiB；这是抽样值，非连续监测峰值。
- 没有 CUDA OOM，训练与验证均成功完成。PyTorch 统计和显卡总占用口径不同。
- 本实验只证明 batch 1 可用，没有测试 1920、batch 2 训练。

## 同口径验证结果

固定 confidence=0.25，原图框 IoU>=0.5 贪心一对一匹配。下面所有数值均来自
同一 val 30 张、1,044 个 GT；不是比赛完整 ComponentF1，也不是 mAP。

| 模型与推理配置 | TP | FP | FN | 精确率 | 召回率 |
|---|---:|---:|---:|---:|---:|
| 原权重，1280整图 | 888 | 184 | 156 | 82.84% | 85.06% |
| 原权重，1920整图 | 880 | 227 | 164 | 79.49% | 84.29% |
| 1920微调权重，1920整图 | 888 | 176 | 156 | 83.46% | 85.06% |
| 1920微调权重，1280整图 | 836 | 188 | 208 | 81.64% | 80.08% |
| 原权重，1280整图+切图 | 947 | 198 | 97 | 82.71% | 90.71% |
| 1920微调权重，1920整图+切图 | 915 | 193 | 129 | 82.58% | 87.64% |

切图均使用原图像素 1024 窗口、25% 重叠、内部边缘过滤 2px、合并 NMS IoU 0.5。
切图模型输入分别为表格中的 1280 或 1920，属于各自部署配置的比较。

在相同 1920 整图推理条件下，微调后精确率提高 3.96 个百分点、召回提高
0.77 个百分点，误检减少 51 个。但它主要恢复了旧权重直接放大输入时的退化，
没有超越原 1280+切图的定位召回。新权重直接改回 1280 推理还有明显回退。

训练结束对最佳模型的标准整图验证：mAP@0.5=0.8544，
mAP@0.5:0.95=0.5065。原权重在 1280 的对应结果约为 0.8444、0.5109；
因输入尺度不同，这组数字仅用于各自配置参考，同条件提升以上表的1920行为准。
本实验改变了输入分辨率，并进行了额外训练、降低物理 batch，因此不能把全部
变化严格归因于分辨率本身。

验证明细：

- 原权重 1280/1920：`outputs/pcbnet_component_resolution_probe/summary.json`。
- 微调权重 1280/1920：`outputs/pcbnet_component_1920_finetune_probe/summary.json`。
- 微调权重切图：`outputs/pcbnet_component_1920_finetune_tiled_val/README.md`。
- 原 1280 切图：`outputs/pcbnet_component_tiled_val/README.md`。

## 保存的新权重与复现

候选权重：`weights/pcbnet_component_yolo26n_1920_v2.pt`，与运行目录 best.pt 相同。
SHA256：`44ea25f2fe1abb2cada7953fcde7e65601b6b9db5141535d6408120aa8da4019`。
旧权重和默认推理配置保留。

在 `component_lab` 目录复现，指定新运行名：

```powershell
.\.venv-yolo\Scripts\python.exe scripts/train_pcbnet_baseline.py `
  --data data/yolo/pcbnet_component_v1/dataset.yaml `
  --model pcbnet_component_yolo26n_v1.pt --imgsz 1920 --batch 1 --nbs 16 `
  --lr0 0.0002 --warmup-bias-lr 0.0002 --epochs 30 --patience 10 `
  --skip-test --name pcbnet_component_yolo26n_1920_repeat
```

对新图使用候选权重时选择 `--imgsz 1920`。当前结果不支持仅靠增加输入尺寸
替换原方案，下一步应在独立验证数据上继续检查小符号框、误检负样本和切图合并。
