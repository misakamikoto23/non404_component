# Component 模块开发工作区

2026-10-18：局部 PP-OCRv6 已接入 `infer --association identity-v1`，提供主 OCR 时默认只补空字段，可用 `--no-local-v6` 关闭。每图最多2区/20%面积/120秒，CPU运行、无API。见 [使用与限制](LOCAL_V6_WORKFLOW.md)。旧 legacy/无OCR及检测复核拖拽工具保持原用途。

2026-10-17：新增[位号与文字归属实验说明](COMPONENT_IDENTITY.md)。固定 DEV20 上，严格 Component F1 从 12.95% 提升到 16.69%；现有权重有历史数据暴露，这只是开发结果。可用 `infer --association identity-v1 --contest-export` 显式启用，输出仅覆盖 component。原始标签、权重与旧推理默认行为保留。

**当前入口：先阅读 [START_HERE.md](START_HERE.md)。YOLO 单类定位使用 data/yolo/pcbnet_component_v1，ResNet 分类数据入口在 data/resnet；Qwen API 模块见 src/component_lab/qwen_api.py。下面保留数据准备与字段约定的详细说明。**

最新ResNet18均衡模型已训练：`weights/resnet18_component_balanced_v1.pt`，27类、每类256份训练文件（含训练侧增广），使用4060 Ti。数据与类别质量说明见[分类训练报告](outputs/resnet_bootstrap_20261016/REPORT.md)，图集见[分类数据图库](outputs/resnet_bootstrap_20261016/gallery.html)。

负责原理图中的元件定位、类别识别、位号与文字信息。当前检测目标是统一的 `component` 框，类别留给后续 ResNet18。推荐的单类 Nano 权重是 `weights/pcbnet_component_yolo26n_v1.pt`，结果见 [TRAINING_COMPONENT.md](TRAINING_COMPONENT.md)；PCBnet 34 类旧基线保留在 [TRAINING_PCBNET.md](TRAINING_PCBNET.md)。ResNet 与文字识别仍需后续完善，本模块尚不是完整赛题提交程序。

## 结构

```text
component_lab/
├── src/component_lab/
│   ├── cli.py           命令入口
│   ├── dataset.py       官方 ZIP → YOLO 标签、ResNet 裁剪图、共同划分
│   ├── geometry.py      坐标转换、裁剪
│   ├── models.py        YOLO、ResNet18、PaddleOCR 适配
│   ├── training.py      YOLO 和 ResNet18 训练
│   ├── association.py   OCR 与元件关联、位号和参数候选解析
│   └── pipeline.py      推理、JSON 输出、框图预览
├── tests/               坐标与匹配规则测试
├── data/prepared/       自动生成的数据和 manifest.json
├── weights/             保留模型权重
├── runs/                训练结果
└── outputs/             推理结果
```

## 数据与字段约定

内部坐标统一为左上角原点、像素单位、xyxy。准备官方旧数据包时显式选择 bottom-left；新来源必须先人工核对。不要按文件扩展名推断坐标系。

内部 `id=component_0001` 是本次推理按位置生成的编号，跨图和跨模型版本不保证一致。`reference=R1/U1` 是 OCR 识别到的图上位号。官方 JSON 的 components 字典键与 Name 不总是一样：实际 U1 的 Name 可以是 RP2350B。因此 `Name`、`reference`、型号不能合并。

输出 `components.json` 包含：id、bbox、type、detection、classification、reference、Name、value、texts、review。当前 YOLO 的 `detection.type` 是 `component`；训练好 ResNet 后，其分类结果写在 `classification`，超过阈值才替换最终 `type`。阈值是待验证的初始参数。Name/型号解析尚未实现，保留 null；不能用虚构的 R1 等填补 OCR 漏字。

全图 OCR 与元件检测并行，随后按几何距离关联。元件旁的文字往往在检测框外，不应只对符号裁剪图做 OCR。附近两元件距离接近时文字保留在 unassigned_texts；网络名、引脚名、标题不一定属于元件参数。当前几何匹配与正则只是基线，需要后续分类一致性、文字角色、引脚上下文约束。

## 环境

在本目录执行，建议 Python 3.11，创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"
```

GPU 版 PyTorch 按机器 CUDA 环境从官方渠道安装。PaddleOCR 可放在另一个环境，通过 OCR JSON 交换结果；在同一环境运行则另外安装 `pip install -e ".[ocr]"`。本项目未自动安装依赖或下载权重。首次使用预训练模型可能联网下载。

## 已准备的数据

数据准备命令（输出目录须为空，避免旧标签混入）：

```powershell
component-lab prepare --archive "../赛题六公开数据集.zip" --output data/prepared --origin bottom-left
```

只使用 200_train_cases。按解码后的图像像素去重，按整张图分成约 80%/10%/10%，然后同时导出检测与分类数据。10_GTcase 与训练图重叠，不用于独立测试。

`manifest.json` 保存类别表、来源、SHA256、每图标注、划分和各类别计数。当前划分是固定种子随机划分，未做类别分层；极少样本类别可能在训练或验证中缺失，必须检查 counts。测试集不能参与调参。同一电路不同版本仍可能存在，仅像素去重不足以识别，需要人工分组审计。

符号裁剪四周保留 4 像素，缩放时保持宽高比并补白到 224×224。当前 ResNet 训练未加入随机旋转/镜像，避免破坏带方向和极性的符号语义。后续补充检测框偏移、合理旋转、噪声与困难样本。

## 训练入口

```powershell
component-lab train-yolo --data data/yolo/pcbnet_component_v1/dataset.yaml --output runs --model yolo26n.pt --imgsz 1280 --epochs 50 --device 0
component-lab train-resnet --data data/resnet --output runs/resnet --epochs 50 --device cuda:0
```

无 GPU 时两条命令使用 `--device cpu`。YOLO 输出 best.pt 路径以训练日志为准；ResNet 输出 runs/resnet/best.pt，保存类别映射、预处理约定和参数。现有超参数是实验起点，没有经过性能验证。

当前 YOLO 只定位；ResNet 负责具体类别。后续须先评估 YOLO 框召回，再评估 ResNet 在真实检测裁剪上的分类结果。ResNet 验证会输出整体准确率、出现类别的宏平均召回率、缺失验证类别；这只是标准裁剪图分类指标，不等于完整 component 指标。

## 推理入口

```powershell
component-lab infer --image "你的原理图.png" --yolo runs/pcbnet_component_yolo26n_v1/weights/best.pt --output outputs/example
```

上述图片仅示例，实际选择 manifest 中的 test 图片。去掉 --resnet 可只导出定位框和裁剪。默认 --ocr none，仅检测；启用 --ocr paddle 或提供 --ocr-json 才会识别/载入文字。输出 components.json 和 overlay.png。密集小符号可通过 `--tile-size 1024 --tile-overlap 0.25` 启用重叠切图，需同时关注切图带来的误报与耗时。

缓存格式均为 JSON 列表、左上角像素框：

```json
[{"bbox": [30, 30, 50, 60], "type": "r", "score": 0.95}]
```

```json
[{"bbox": [55, 32, 75, 42], "text": "R12", "score": 0.98}]
```

使用 `--detections detections.json --ocr-json ocr.json` 可以在没有模型的情况下调试后处理。缓存只是调试输入，不能称为模型推理效果。

## 开发顺序与验收

1. 检查 data/prepared/manifest.json，各类别计数和框坐标；人工查看不同来源的样本及裁剪。
2. 训练 YOLO；记录验证集 mAP、每类召回和小目标漏检，保存失败案例。
3. 训练 ResNet；检查混淆矩阵，再用实际检测裁剪验证。测试集不用于困难样本挖掘。
4. 启用 OCR；检查位号、参数、型号原文，以及相邻元件错配。
5. 比较最终框+类别、位号准确率、参数准确率和整条流程耗时。
6. 完善 Name/型号解析与赛题导出适配，核对官方坐标和字段规则，再交给 pin/net 模块。

当前未实现：专用 OCR 训练、全局文字分配、引脚语义、赛题完整导出。不要将本项目的 JSON 直接当作官方完整提交文件。

赛题指标本地评测（key 匹配的 Component/Pin/NetHypergraph/NetLine F1、类型/∅ 策略、输出替换 oracle、按重复组划分的 test40/dev20 拆分，供干净重训使用）：见 [EVAL_CONTEST6.md](EVAL_CONTEST6.md)，命令 `component-lab score`。

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 参考

- [YOLO26 官方结构与用法](https://docs.ultralytics.com/models/yolo26/)
- [ResNet18 官方接口](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [PaddleOCR 3.x OCR 接口](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
