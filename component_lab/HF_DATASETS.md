# Hugging Face 原理图数据源调研

检索日期：2026-10-15。以下为在线数据卡核验结果；本机下载受网络 DNS/SSL 连接错误影响，尚未完成文件和图像验证。不要把数据卡描述当作已运行训练结果。

| 数据源 | 标注 | 适用性和限制 |
| --- | --- | --- |
| [LibreYOLO/circuit-elements](https://huggingface.co/datasets/LibreYOLO/circuit-elements) | 数据卡声明 YOLO 检测标签，772 张图、45 类，train 672/val 64/test 36，约 71.9 MB | 优先小规模下载核验；类别有 Resistor/Resistance/Resistort/Ressistor 等重复或拼写差异。页面预览报错，不等于原始文件不可用。CC-BY-4.0，需保留来源。 |
| [hanky2397/schematic_images](https://huggingface.co/datasets/hanky2397/schematic_images) | images.zip、components.zip（类别、框和方向）、component_id、HSPICE 网表等 | 任务很接近，约 1.74 GB；需要登录并接受共享联系信息的访问条件。方向字段可能需要适配标准五列 YOLO 标签。未验证完整包，许可和类别表待查看。 |
| [lowercaseonly/cghd](https://huggingface.co/datasets/lowercaseonly/cghd) | 手绘电路图、VOC XML 框/类别/文字，部分分割标注 | 适合手绘补充，需要转 YOLO。按同一绘图和电路分组划分，不要把同图多次拍摄拆开。HF 卡标明 CC-BY-3.0；原始出处为 CGHD。 |
| [bshada/open-schematics](https://huggingface.co/datasets/bshada/open-schematics) | 原理图源文件、渲染图、元件列表和部分结构信息 | 适合获取更多公开电路图再人工标注；数据卡未提供现成逐元件 YOLO 框。部分图片缺失。数据卡标明 CC-BY-4.0，并要求遵守原工程许可。 |

## 下载

在 component_lab 目录，联网环境执行：

```powershell
python -m pip install huggingface_hub
python scripts/download_hf_components.py
```

文件保存在 data/external/circuit-elements，和正式 data/yolo、data/resnet 分离。脚本使用当前实际仓库 ID `LibreYOLO/circuit-elements`；数据卡示例中的 `Libre-YOLO/circuit-elements` 与页面地址不一致，应以当前页面地址为准。

先查看几张原图，确认是你希望的符号风格；再检查 data.yaml 类别表和 txt 框是否正确；建立外部类别到本项目类型的映射；最后按框裁剪得到 ResNet 样本。外部无法映射的目标不能简单删掉标签后混入正式检测训练，否则会把它们当背景。

## 对本项目的建议

现有官方数据作为基线，外部数据按来源分开评测。优先检查 circuit-elements；需要更多真实工程图时使用 open-schematics，避免立即投入大范围社区抓图。公开工程可用于人工标注，但页面上的 PCB 3D 图、实物照片、低清缩略图不是原理图训练样本。

当前没有抓取 JLC 页面，也没有自动接受 gated 数据集的条款。已确认存在合适数据入口，因此先使用现成数据源。
