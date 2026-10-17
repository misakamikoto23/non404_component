# 元件预标注与人工复核

第四轮已完成（2026-10-16）：第四批20张全部审核，已冻结到 `data/review_snapshots/batch_004_round_004`，累计80张人工审核。新训练版本 `data/reviewed_versions/manual_round_004/yolo` 含243张训练图、11025框，原验证/测试120个文件哈希不变。4060 Ti完成12轮，最佳第7轮；新权重 `weights/pcbnet_component_yolo26n_review_round_004.pt` 的验证P88.83%、R84.00%、mAP50 86.20%、mAP50–95 53.24%。召回略升、精确率下降，不是双95%达标。详见[第四轮报告](outputs/active_learning/round_004/REPORT.md)。当前页面仍是已完成的第四批，下一批尚未生成；下文保留各批历史说明。

第四批已生成（2026-10-16）：`batch_004`，页面 http://127.0.0.1:8768 。共20张，PCBnet与HF各10张、交替呈现，第三轮Nano预标注1503框。已排除前三批图片，HF按10个独立原项目抽样并做文件、解码像素和近似布局重复筛查。数据源为 `data/review_sources/batch_004/yolo`，包含此前60张累计人工修正；新的HF原图保存在images/pending，没有训练标签，完成审核后才会由导出器移入训练集。未完成的HF图片在导出时省略，不当负样本。120个验证/测试文件已核对保持一致。继续双击“开始检查元件.cmd”会打开本批。混合来源入口为 `scripts/prepare_mixed_review.py`。

最新状态（2026-10-16）：第三批20张已全部审核，三批累计60张修正已导出到 `data/reviewed_versions/manual_round_003/yolo`，233张训练图共10279个框。第三轮Nano已完成，最佳权重为 `weights/pcbnet_component_yolo26n_review_round_003.pt`，验证P89.89%、R83.42%、mAP50 85.91%、mAP50–95 52.81%；双95%尚未达到。[训练与改进方案](D:/大四上/EDA/EIT&BTD/component_lab/outputs/active_learning/round_003/REPORT.md)同时记录低阈值候选验证与Qwen二次复核设计。

后续抽样约定（2026-10-16）：从第四批开始，每20张默认按 **PCBnet 10张 + Hugging Face Open Schematics 10张** 混合；比例未由用户指定，先采用1:1。规则记录于 `data/review_batches/next_batch_policy.json`。排除所有先前批次出现的图片，并对已有train/val/test做重复筛查；保留来源和HF原项目分组。新HF图只有预标注，完成整图审核后才能加入训练，缺标签不能视为空背景。混合来源入口已为第四批接入独立样本ID、累计修正及待审核数据隔离，冻结原验证/测试文件。

前两轮历史：第一批新增22个框、删除572个框，最终保留1687个框；第二批 `batch_002` 新增12个、删除166个，最终保留1266个框。删除数量相对于模型候选，并非相对于原始PCBnet标签。第二轮训练版本为 `data/reviewed_versions/manual_round_002/yolo`：233张训练图、10223个框，其中40张采用人工修正，其余193张沿用原标签；验证/测试保持不变。

第二轮从第一轮最佳权重继续微调，最多15轮、实际第10轮早停，最佳权重来自第5轮，保存在 `weights/pcbnet_component_yolo26n_review_round_002.pt`。最终验证mAP50为85.59%，mAP50–95为52.78%；[训练报告](D:/大四上/EDA/EIT&BTD/component_lab/outputs/active_learning/round_002/REPORT.md)保留对照。该权重为第三批20图生成的1749个候选现已审核：删除359个、另加30个，最终保留1420个框。候选使用1280输入、0.1置信度、原图1024像素切图与整图合并。复核页面及导出操作本身不自动训练、不调用云端。

## 开始检查

在component_lab目录双击“开始检查元件.cmd”。启动器读取 `data/review_batches/active_batch.json` 中的 `batch_id` 和 `port`，打开当前批次；没有配置文件时默认第一批8765端口。服务器只监听本机，后台运行，关闭浏览器不会丢失已经保存的进度，下次双击会复用当前批次。

当前批次为待审核的 `batch_004`，端口8768，页面为 http://127.0.0.1:8768 。前三批的文件和人工审核记录保持完整，第一至第三批原端口分别为8765、8766、8767。继续使用同一个“开始检查元件.cmd”即可打开第四批。

页面用于选择框、移动和缩放框、删除误检、拖画漏检、填写类别和位号，并保存进度。请根据页面说明使用快捷键。先处理1～2张熟悉流程，不需要一次完成20张。

整图必须另扫一遍：仅逐个查看模型已框出的元件，无法发现完全漏检。原始PCBnet框可作为独立参考层打开，不能把它们误当模型预测或已经复核的结果。

## 每张图的操作顺序

1. 先整体观察图像与元件数量，定位主要电路区域和大器件。
2. 逐个确认候选：正确则保留，文字/图框/重复候选则删除；偏框拖动边角修正。
3. 扫描所有电路区域，拖画遗漏元件。小地符号、电源端口、IC边缘和密集电阻电容尤其需要检查。
4. 定位确认后即能用于单类YOLO。分类可以稍后完成：选择具体类别并显式勾选分类确认，才会输出ResNet裁剪。类别提示来自与原标签的几何匹配，不是模型已正确分类的证据。
5. 可逐框处理后勾选整图检查并完成保存；也可点击“本图直接通过”，一键保留所有未删除的框，确认整图并完成保存。已经删除的框保持删除，分类确认状态不变。中途可保存草稿，草稿不进入审核标签。
6. 点击导出审核数据，生成独立新版本；至少有一张完成后才能导出。未完成图片仍使用原训练标签。

删除仅标记rejected，保存修改历史；不会从原始资料中删除元件或图片。分类或位号不确定时保留空值，不推测。不要为了提高IoU分数把全部框机械缩放。

## 框与类别约定

- 坐标始终是原图像素xyxy。页面缩放和平移不改变实际坐标。
- 检测范围继承当前PCBnet数据的原始类别，包括元件、地、电源/网络端口和模块符号；将来如缩减范围，应另立数据版本。
- 框覆盖符号本体及自身短引脚，排除元件编号/参数文字与延伸的外部连线。IC、连接器以整个符号为单位；不把各引脚另标成元件。遇到原框口径不清，记录备注，先保留草稿。
- 引脚末端独立的小叉号（No Connect）不作为component；覆盖在真实元件上的DNP/禁装叉线不单独标框，仍保留真实元件主体。原始Block类别语义混杂，不能仅凭类别名批量删标，应结合图形审核。未审核图片仍可能留有旧口径标签。
- 所有最终检测标签仍为class0 component。ResNet具体类别保留为原始类别字符串；类别确认与定位确认分开。
- 原始类别并非清洗后的最终分类体系，例如Box或不同端口类别，需要在分类训练前确认含义，不能自动合并近似名称。

## 类别均衡如何进入流程

原始数据共293张、12536框；原训练集233张、10126框，验证/测试各30张。34类中14类训练来源不足15张。20张批次按8随机、6稀缺覆盖、6基于小目标/密度/图片大小的几何启发选择；后续批次先排除指定旧批次，再对剩余训练图片抽样。不是用测试集错误挑训练图，也不声称几何困难等于模型真实漏检。

类别统计与分类权重方案位于 data/review_batches/batch_001/reports/ 。稀缺按训练来源图片数定义；不要求每类数量相等，也不会复制图片凑平衡。第一批覆盖14个稀缺类别中的13个，Dc可在下一批优先补充。

ResNet训练入口增加可选温和抽样：`--balance sqrt_sampler --balance-cap 5`。只根据train计数，用平方根逆频率、最高5倍的样本权重抽样；每轮总样本数不变，不同时叠加类别损失权重。默认仍为none，不影响旧训练。统计中的建议权重不是已启动的训练。

## 保存与导出结构

```text
data/review_batches/active_batch.json
                            当前批次与页面端口，供启动器读取
data/review_batches/batch_001/
  batch.json                固定批次、模型配置、初始候选与原标注参照
  images/                   本批图像副本
  items/                    每张图已保存的人工修改
  review_history.jsonl      历次保存记录
  reports/                  类别数量、来源图片数、ResNet均衡计划

data/review_batches/batch_002/
                            已完成第二批审核，独立保存记录

data/review_batches/batch_003/
                            第三批已完成审核，快照及第三轮训练数据已导出

data/reviewed_versions/<新版本>/
  yolo/dataset.yaml          可供后续YOLO训练的数据配置
  yolo/images, labels/       完成图使用审核标签，其余沿用原标签
  yolo/manifest.json         新标签与类别确认状态
  resnet_reviewed/train/     只含明确确认类别的已接受元件裁剪
  resnet_reviewed/crop_manifest.json
  summary.json              修改数量与holdout哈希核对
```

验证/测试图片和标签按字节复制并核对SHA256，不因人工复核修改。ResNet导出目前只是审核过的train裁剪，不是假造的完整分类数据集；需要另配按原理图分组的独立val，训练入口会拒绝空val或缺训练类别。导出不会启动训练或替换现有权重。

每一批的 `batch.json` 会记录其 `data_root`。第四批以 `data/review_sources/batch_004/yolo` 为数据源，由第三轮累计版本扩展而来，保留此前60张修正，并隔离10张尚未审核的HF图片。页面导出按钮读取当前批次的数据源，命令行导出需显式传入相同的 `--data`。未完成的新HF图会被省略；原PCBnet未完成图片仍保留旧标签。

## 命令行入口（可选）

```powershell
.\.venv-yolo\Scripts\python.exe scripts\review_workflow.py stats --data data/reviewed_versions/manual_round_002/yolo

# 当前第三批已生成，以下保留复现命令；非空批次目录会拒绝覆盖。
.\.venv-yolo\Scripts\python.exe scripts\review_workflow.py prepare `
  --data data/reviewed_versions/manual_round_002/yolo `
  --batch data/review_batches/batch_003 `
  --weights weights/pcbnet_component_yolo26n_review_round_002.pt `
  --exclude-batch data/review_batches/batch_001 `
  --exclude-batch data/review_batches/batch_002 `
  --count 20 --seed 20261017 --device 0

.\.venv-yolo\Scripts\python.exe scripts\review_workflow.py serve --batch data/review_batches/batch_003 --port 8767

# 完成第三批审核后，累计导出下一轮数据；输出目录须为新目录。
.\.venv-yolo\Scripts\python.exe scripts\review_workflow.py export `
  --batch data/review_batches/batch_003 `
  --data data/reviewed_versions/manual_round_002/yolo `
  --output data/reviewed_versions/manual_round_003
```

`--exclude-batch` 可重复指定，后续批次应列出所有需要排除的旧批次。排除依据是旧批次内的全部图片ID，与其是否完成审核无关；仅更换seed不能保证不重复。新批次记录排除来源及ID，剩余可选训练图片不足指定数量时会明确报错，不会悄悄少给图片。原始manifest和旧批次不会被修改，统计仍覆盖完整数据源。相同图片不要同时在多个浏览器窗口编辑，版本冲突会阻止覆盖。
