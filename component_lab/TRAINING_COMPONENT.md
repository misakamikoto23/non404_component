# PCBnet 单类元件定位

1920 输入微调已完成，显存可用，但验证集未超过当前默认方案；见
[TRAINING_1920.md](TRAINING_1920.md)。

本实验按当前两阶段方案训练 YOLO26 Nano：每个官方有效元件框统一标成
`component`。具体的 R、C、MOSFET 等类别由后续 ResNet18 负责。检测结果
是元件位置与截图，尚不包含可验证的位号或参数文字。

## 数据与隔离

- 数据入口：`data/yolo/pcbnet_component_v1/dataset.yaml`。
- 源头：`Data/PCBnet/Dataset300` 中的原始 `*_CVJsonStd.json`；仅取有效
  `realBBox`，剔除 1 个 deleted 条目并裁剪 3 个越界框。
- 293 张像素去重的原理图，train / val / test = 233 / 30 / 30；框数分别为
  10,126 / 1,044 / 1,366。
- 已逐框核对源 JSON、归一化坐标、图片与标签配对、分区间图像哈希；没有
  混入用户手工标注或智能体候选修订。
- `manifest.json` 逐框保留原始 `category`、`component_id` 与源文件路径，
  供后续 ResNet 裁剪数据生成。

## 训练设置

- 训练脚本：`scripts/train_pcbnet_baseline.py`；运行目录：
  `runs/pcbnet_component_yolo26n_v1/`。
- 初始权重：只使用原始 PCBnet 标签训练过的 34 类 YOLO26 Nano 权重
  `runs/pcbnet34_yolo26n_v2/weights/best.pt`。从该权重迁移 696/708 参数项，
  分类头重建为一个类别。
- 1280 输入、batch 2、有效 batch 16、AdamW、初始学习率 0.001、AMP，
  最多 80 轮；验证指标 15 轮不提升时早停。
- 无旋转、翻转、mosaic，仅少量亮度、平移、尺度增强，以免破坏元件方向与极性。
- 本机 RTX 3050 Ti Laptop GPU 4 GB 上运行。详细超参数见运行目录
  `args.yaml`，每轮指标见 `results.csv`。
- 实际完成 32 轮后早停；最佳权重出现在第 17 轮。PyTorch 峰值分配
  1541.3 MiB、预留 2036.0 MiB，均低于显卡 4 GB 容量。

## 验证与交付

Nano 权重：`weights/pcbnet_component_yolo26n_v1.pt`，SHA256
`e447da98749c171c1bf12589ade8a40c2b605e2ed670840ae7b4634ff8e954e2`。
它与运行目录的 `weights/best.pt` 内容相同。最佳权重在验证集上的整图
mAP@0.5 约 0.844、mAP@0.5:0.95 约 0.511。这些是**单类定位**指标，
不应与旧 34 类检测 mAP 直接比较，也不代表 ResNet 分类准确率。

为补回小符号，先在验证集上比较整图与“整图 + 1024 原图像素切图、25% 重叠”。
模型输入均为 1280；置信度固定 0.25；以原图框 IoU ≥ 0.5 逐一匹配官方框：

| 验证集定位 | 整图 | 整图 + 切图 |
|---|---:|---:|
| 真检出 / 漏检 | 888 / 156 | 947 / 97 |
| 误检 | 184 | 198 |
| 精确率 / 召回率 | 82.8% / 85.1% | 82.7% / 90.7% |

切图主要改善小符号：按整图输入时估计为 4–8 像素的 89 个框，召回率
由 45% 提到 66%。代价是多 14 个误检和额外推理时间。因此选定这套
切图参数用于最终测试。完整阈值固定诊断见
`outputs/pcbnet_component_tiled_val/README.md`。

独立测试集 30 张、1,366 个原始框，在相同阈值及合并设置下：

| 测试集定位 | 整图 | 整图 + 切图 |
|---|---:|---:|
| 真检出 / 漏检 | 1,025 / 341 | 1,133 / 233 |
| 误检 | 406 | 453 |
| 精确率 / 召回率 | 71.6% / 75.0% | 71.4% / 82.9% |

测试集整图标准 mAP@0.5 为 0.706、mAP@0.5:0.95 为 0.395；这是按整条
精确率—召回率曲线计算，与上表固定 0.25 阈值的数值定义不同。完整结果见
`outputs/pcbnet_component_final_test/metrics.json` 和
`outputs/pcbnet_component_tiled_test/README.md`。测试集比验证集更难，
尤其是小符号：整图估计为 4–8 像素的 234 个框，在切图后召回由
53% 提到 70%；验证集对应尺寸只有 89 个框。测试集小框更多可能是
指标下降的原因之一，不能仅凭这 30 张图断定模型容量不足。这些数值仅以
当前 PCBnet 标注为准。

另有不参与调参的事后漏检报告：
`outputs/pcbnet_component_category_diagnostics/README.md`。按原始标签回看，
最终 233 个漏检中，电容 C 60 个、电压/电源符号 V 59 个、Gnd 40 个、
电阻 R 21 个；估计整图缩放后短边不足 16px 的目标占 196/233。
这些多是常见类别，所以不能简单归结为“某些类样本太少”。

本单类检测目标覆盖官方标注中的 `Gnd`、`PIN`、网络端口、`Box` 等图形。
它们在原理图识别中有用，后续分类/PCB Layout 阶段还需判断哪些对应实际
可放置的物理器件。

正式单张图推理入口支持 `--tile-size 1024 --tile-overlap 0.25
--tile-max-tiles 100`；超过窗口上限时只用整图结果，不静默丢弃部分区域。
当前 Nano 权重仍是初步定位模型：测试集切图精确率约 71%，需要继续排查
误检与训练/测试图分布差异。

## Nano 与 Small 的验证集对照

另训练了 `weights/pcbnet_component_yolo26s_v1.pt`（SHA256
`e636e906e47c4224b4a7313bffeb69ecfd0b079e6bb8bb037889864902cdeba9`）。
Small 同用单类原始数据、1280 输入、batch 2、有效 batch 16，完成 40 轮，
最佳第 37 轮；PyTorch 峰值预留显存 2996 MiB。两者训练起点不同：
Nano 从已学习 PCBnet 原始 34 类的 Nano 权重继续，Small 从通用预训练
YOLO26s 开始，因此这一对照不能单独证明参数规模的因果效果。

| 验证集指标 | Nano | Small |
|---|---:|---:|
| 整图 mAP@0.5 | 0.844 | 0.837 |
| 整图 mAP@0.5:0.95 | 0.511 | 0.497 |
| 固定阈值切图后真检出 / 误检 | 947 / 198 | 913 / 220 |
| 固定阈值切图后精确率 / 召回率 | 82.7% / 90.7% | 80.6% / 87.5% |
| 峰值预留显存 | 2036 MiB | 2996 MiB |

Small 的固定阈值验证报告在
`outputs/pcbnet_component_yolo26s_tiled_val/README.md`。在此验证条件下
Nano 召回、精确率均较高且显存占用更低，所以选 Nano 作为当前默认模型。
Small 没有参与测试集选择，也没有额外在 test 上运行。更大模型以后仍可在
相同初始化和更充足样本下做严格对照；当前证据更直接支持改善小目标尺度
和标注框质量。

在 `component_lab` 目录复现 Nano 训练（新建运行目录，不覆盖现有结果）：

```powershell
$init = (Resolve-Path runs/pcbnet34_yolo26n_v2/weights/best.pt).Path
.\.venv-yolo\Scripts\python.exe scripts/train_pcbnet_baseline.py `
  --data data/yolo/pcbnet_component_v1/dataset.yaml --model $init `
  --imgsz 1280 --batch 2 --nbs 16 --epochs 80 --patience 15 `
  --skip-test --name pcbnet_component_yolo26n_repeat
```

对单张新原理图定位并导出框图、每个符号和周边截图：

```powershell
.\.venv-yolo\Scripts\component-lab.exe infer `
  --image "你的原理图.png" --yolo weights/pcbnet_component_yolo26n_v1.pt `
  --imgsz 1280 --device 0 --tile-size 1024 --tile-overlap 0.25 `
  --tile-max-tiles 100 --output outputs/your_diagram
```

输出中的 `type=component` 是定位标签，尚未判定 R/C 等类别；内部
`component_0001` 编号也不是图上的真实位号。

实际用选定 Nano 权重和切图模式跑通了 3 张保留测试图：
`outputs/pcbnet_component_predictions/{0263,0135,0111}`。每张都有
`components.json`、`overlay.png`、一框一份的符号与周边截图。
这三张分别导出 21、48、189 个候选框，官方框数是 12、43、162；
候选数量差异不能单独当作该图的精确率或召回率，完整匹配结果见上述
测试集报告。`0263` 可见额外大框，说明当前误检问题仍需处理。
