# PCB 原理图自动解析：给 Gemini 的项目交接与技术评审请求

整理日期：2026-10-17。本文基于本地代码、模型记录与已完成实验，不是理想方案说明。请严格区分已实现能力、开发指标、数据限制与后续设想。

## 1. 希望你帮助解决什么

我们参加“AI驱动的PCB原理图自动解析”赛题。我主要负责 component 环节，希望从原理图图片输出元件位置、类型、位号（如R1/U3）、型号/名称与参数，并为下游引脚、网络解析提供可靠输入。

初始路线是 YOLO26 Nano 单类定位 → ResNet18 分类 → OCR/多模态模型读取文字。检测和分类已有真实训练，但对照赛题后发现：检测框指标不能代表最终元件分数，位号与文字关联是当前关键瓶颈。

请结合下文证据评审方案，优先给能工程实现、能在固定开发集验证、减少人工标注的建议，不要仅建议“增加数据/增大模型/多训练几个epoch”。如果认为某项低分主要来自标注协议、输出适配或评测假设，而非模型，请明确区分。

## 2. 工作区、硬件与当前状态

```text
D:\大四上\EDA\EIT&BTD\
├── component_lab/         当前开发程序、训练数据与实验
├── Data/                  PCBnet、HF等原始数据
├── non404/                既有赛题代码，只读参考，本轮未修改
├── research/              规范提取文本、历史调研与数据审计
└── 赛题六公开数据集.zip     官方公开数据
```

- Windows + PowerShell，本地 Python/PyTorch。
- RTX4060 Ti 16GB 与 RTX3050 Ti Laptop 4GB。PyTorch cuda:0 实测是4060 Ti，与 nvidia-smi 显示索引不同。
- `.venv-yolo` 用于检测与分类；`.venv-ocr` 用于本地 RapidOCR。不能假定所有包都在默认 Python 中。
- 当前第一轮身份关联改进已完成，无训练作业继续运行；Small训练维持暂停。
- 本轮没有付费API、没有新训练、没有修改人工标签、原始GT或模型权重。
- 已有人工审核共80张；人工主要审核定位框，不代表这些框已有人工确认的类别。
- 用户希望尽量减少后续人工标注，但不允许把机器建议冒充人工或官方真值。

## 3. 赛题评分：项目应优化什么

每个案例的准确度：

```text
SingleCase = 100 × (
  0.30 × ComponentF1
  + 0.25 × PinF1
  + 0.35 × NetHypergraphF1
  + 0.10 × NetLineF1
)
```

- Component：以 key 严格匹配；还须 type正确、bbox误差≤20px，并通过适用文字字段。
- IC比较Name；电阻/电容比较value；其他类型按规范不比较该文字字段。
- 字符串严格区分大小写、空格、单位写法，例如10k、10K、10 k不相同。
- Pin：按component key与pin number识别；pin name及坐标也要正确，坐标容差5px。
- NetHypergraph：网络看作引脚集合，网络之间匹配后比较集合成员；net自身ID不要求相同。
- NetLine：仅在配对网络内比较线段，端点容差5px。
- component key错误会影响元件、引脚与网络匹配；因此身份识别具有跨阶段价值。

最终准确度部分由95%案例准确度与5%Token项组成，Token按队伍消耗排名赋分，并受完成案例比例影响；另有最高10分数据贡献附加分。所有案例须在2小时内完成，未完成案例不得分。规范预计正式案例30～50个，公开/黑盒约1:2，难度中等40%/难60%。

规范指定测试环境提供qwen3.7-plus。项目此前Qwen开发实验使用配置中的qwen3.8-flash/max，不代表最终指定环境兼容已验证。

我们的评分器是公共规则的本地代理，不是主办方官方评测器。主报告采用 raw type严格相等、匿名对象保留、四个bbox坐标最大绝对误差≤20px。中心距离、类型映射、排除匿名对象均只作显式诊断。

## 4. 当前实现的流水线

```text
原理图
 ├─ YOLO26n：整图＋可选重叠切片 → component框
 │    └─ 框周围裁剪 → ResNet18 → 图像类别候选
 └─ 本地全页/切片OCR → 原文token＋位置
          ↓
  文本角色候选：reference / value / part_number / pin / net / description
          ↓
  全局文字－元件关联，允许弃权
          ↓
  分类弃权时使用明确位号前缀＋保留关联代价差异
          ↓
  component字段、证据、未解决项、原图可视化
          ↓
  可选赛题格式组件导出：左下角bbox，pins/nets暂为空

Qwen为独立可选旁路；本轮未使用，不默认覆盖已解析字段。
```

重要区别：内部 component_0001 是一次推理的临时ID，reference=R1才是图中位号。Name不等同于reference，IC的Name也不总是典型料号。

## 5. 程序结构与责任

```text
component_lab/
├── src/component_lab/
│   ├── cli.py                   prepare/train/infer/score等命令
│   ├── models.py                YOLO与ResNet加载、推理，PaddleOCR适配
│   ├── tiling.py                切片计划、坐标还原、边界过滤与NMS
│   ├── pipeline.py              模型/OCR/关联/裁剪/JSON/预览的集成入口
│   ├── geometry.py              bbox转换、裁剪
│   ├── association.py           旧文字关联基线
│   ├── text_roles.py            位号/参数/型号/网络/引脚等文字候选
│   ├── text_matching.py         全局slot分配、文字分组、几何/类型代价与弃权
│   ├── contest_components.py    新身份关联版本、明确前缀先验、组件格式导出
│   ├── training.py              YOLO/ResNet训练与可选采样平衡
│   ├── dataset.py               早期官方数据准备入口
│   ├── qwen_api.py              可选图像API适配
│   ├── candidate_verifier.py    候选复核、缓存、预算账本等可选API实验工具
│   ├── candidate_postprocess.py 候选来源/几何/保守后处理
│   ├── structure_checks.py      边框、拆合冲突、连线/引脚残段证据
│   ├── review_workflow.py       预标注、网页审核、审核结果隔离导出
│   ├── review_web/              人工审核前端
│   └── eval/
│       ├── metric.py            元件/引脚/网络/线段本地评分
│       ├── matching.py          一对一分配/最大匹配
│       ├── coords.py            左上/左下坐标互换
│       ├── cases.py             案例发现、官方数据提取、划分读取
│       ├── dedup.py             去重辅助
│       ├── report.py            逐图诊断、汇总与置信区间
│       └── type_vocab.json      显式类型词表，映射评分只作诊断
├── scripts/
│   ├── prepare_contest_dev_inputs.py        仅加载固定dev图片跑检测/分类
│   ├── run_local_ocr.py                     本地OCR与带hash缓存
│   ├── run_component_identity_experiment.py 预声明版本对照，预测冻结后才读GT
│   ├── refine_local_reference_ocr.py        一轮有上限的局部补读
│   ├── audit_component_identity.py          历史10图身份失败审计
│   ├── prepare_official_gold_crops.py       按train/dev分组导出官方源标签裁剪
│   ├── probe_pin_geometry_oracle.py         GT元件框输入的引脚点位实验
│   ├── audit_contest_alignment.py           官方规则/数据/既有训练重叠审计
│   ├── fingerprint_sources.py              跨来源图像指纹
│   ├── build_group_split.py                设计分组与新划分
│   └── 其他历史训练、检测、复核与可视化脚本
├── configs/              默认配置、固定分组划分、历史实验配置
├── data/yolo/            原理图检测数据
├── data/resnet/          分类裁剪与版本化manifest
├── data/reviewed_versions/人工定位审核后的数据版本
├── data/official_contest6/官方GT与来源manifest
├── weights/              当前及历史模型、OCR ONNX模型
├── runs/                 训练参数、曲线、评估与best权重
├── outputs/              推理、实验、诊断、图库
├── tests/                数据隔离、匹配、坐标、OCR、导出等测试
├── COMPONENT_IDENTITY.md 新流程使用说明
├── EVAL_CONTEST6.md      本地评分器规则、假设和分组约定
└── GEMINI_HANDOFF_20261017.md 本文
```

现有 `infer` 默认仍为legacy，需显式加 `--association identity-v1 --contest-export` 使用新版本。不能假定全部旧入口已自动切换。

## 6. 检测模型与既有训练

当前检测权重：`weights/pcbnet_component_yolo26n_review_round_004.pt`。

- 单类component，分类交给ResNet。
- 第四轮使用243张训练图、11025个框，累计80张人工审核定位图。
- 本轮微调记录：输入1280、batch2、有效batch16、学习率1e-4；请求15epoch，记录实际完成12epoch。
- 原30张验证图上的Ultralytics指标：P88.83%、R84.00%、mAP50 86.20%、mAP50–95 53.24%。
- 这些是检测指标，不是赛题ComponentF1；不能与后文16.69%直接比较。
- 1920输入Nano实验做过，但当前身份实验固定使用1280模型输入；切片为原图1024px、25%重叠，再合并。
- 整图也保留，以减少被切断的大元件遗漏；内部切片边缘候选过滤、NMS有实现，仍可能出现拆框/合框。
- Small训练已按用户要求暂停，不能自行恢复。

## 7. ResNet18现状与标签可信度

当前权重：`weights/resnet18_component_balanced_v1.pt`。

- 27个语义家族，输入保持宽高比补白到224×224。
- 训练文件每类256份，共6912份：2251份原始裁剪＋4661份训练增强。
- 验证951份，不做训练侧增强。
- 弱标签验证一致率96.74%，有验证支持类别的macro F1约73.94%。
- 该结果是相对于弱标注/AI银标的一致性，不是人工真值分类准确率；一些稀有类真实样本不足。
- 均衡文件数不等于均衡的独立元件/设计数，复制和增强不能补足真实多样性。
- 人工定位审核框不能自动升级为人工分类标签。
- 使用图像分数0.8作为初始弃权阈值，分数未校准。

此外已导出官方源标签候选分类集 `data/resnet/official_gold_v1`：train6417/dev931，38个原始官方类型，训练来自140条记录、138种像素内容、134个设计组。

但标记为 training_ready=false，尚未启动训练：

1. 部分官方bbox只覆盖符号内部或局部，直接裁剪可能是白块、引脚或文字，而不是完整元件。
2. 保守QA发现15个近空白/低对比度裁剪，仍不能覆盖大量不完整符号。
3. 原始类型标签来自官方，不代表经过独立人工验证；例如机械/电源/端口等语义需核查。
4. dev中的原始crystal类在train没有样本；9个train类别在dev没有样本。
5. 27家族与38原始类型不是同一套分类任务，不能无说明地混训或比较分数。

已生成原框与每轴max(12px,50%bbox尺寸)外扩对照。外扩可能带入邻近器件和文字，不能直接视为解决方案，也未修改原始GT。

## 8. 数据划分与污染问题

官方公开200条记录包含9171个元件，图像解码后198种独立像素内容。GT10图像全部与公开200有重叠，不是独立测试。

官方200中有160条图片精确匹配当前第四轮检测训练图，另有20条匹配旧val、19条匹配旧test（按记录计数，含重复可能）。不能现在随机抽30张官方图就宣称是未见测试。

当前固定划分 `configs/contest6_group_split_v1.json`：train140/dev20/test40。先按跨来源设计组划分，再导出训练图/裁剪。

分组线索包括解码RGB hash、感知hash、官方非匿名(key,value)标签相似度和声明的文件变体。现有参数含phash/dhash阈值12（HF相关更严为8），标签多重集合Jaccard≥0.55且两侧至少25个非匿名元件。这些是启发式分组，存在误合并、漏重复和并查集传递风险，欢迎评审。

新划分服务于未来从通用预训练权重做干净重训练。现有权重在它上面仍有污染；封存test40不代表这些图片在项目历史上从未出现。此次新实验没有使用test做推理、调参或裁剪；不能据此抹掉历史观察。

跨来源禁止清单保存在 `outputs/eval_split_audit_20261017/forbidden_for_clean_retrain.json`。以后重训练必须按整个设计组排除dev/test对应的所有来源版本及裁剪，而非只排除官方文件名。

## 9. 最新DEV20组件实验：已经验证的改进

输入：现有YOLO+ResNet，固定20张dev，原始全页/切片RapidOCR。所有版本共用相同框和OCR；先完成全部预测导出，再加载GT评分。

主指标为严格匿名策略、原始type字符串、bbox四坐标最大误差≤20px、逐案例平均ComponentF1：

| 版本 | Macro ComponentF1 | TP | FP | FN | 正确key命中 |
|---|---:|---:|---:|---:|---:|
| baseline：原全局关联与导出 | 12.95% | 118 | 922 | 813 | 225 |
| prefix：分类弃权时使用明确位号先验 | 13.77% | 128 | 912 | 803 | 223 |
| unclamped：再保留关联负代价 | 16.69% | 164 | 876 | 767 | 277 |
| identity_v1：再加IC Name原文候选回退 | 16.69% | 164 | 876 | 767 | 277 |

改动解释：

- 明确位号先验只补分类弃权：R→电阻、C→电容、L→电感等。D/Q等歧义不强行定极性/类型，不无条件覆盖强图像预测。
- 原关联代价包含类型相容的负奖励，但被max(0,cost)截成零。多个本来不同的好候选都变成0，之后被当成歧义而弃权。新版本保留代价差异，仍保留真正歧义的弃权。
- IC Name可使用自己的已归属原文value作为缺少part_number时的候选回退；此项在当前dev20没有额外收益，不能声称规则普遍有效。
- 原始OCR大小写/空格/单位不归一化为最终字段；归一化只供内部解析候选。
- 无key/重复key的检测保留为明确unresolved临时key，不凭空分配∅ID，也不通过删除困难框提高分数。

主改进+3.74个百分点，按案例配对bootstrap 95%区间约[+1.86,+5.81]个百分点；这是开发证据，不是独立统计确认，未计评分假设不确定性。

中心距离替代bbox口径下identity_v1为19.35%，只作敏感性诊断。不能用替代口径当主结果。

535个非匿名GT对象的主版本状态：

| 状态 | 数量 |
|---|---:|
| 完整正确 | 164 |
| 命中key但类型/框/文本至少一项错 | 113 |
| 缓存里没有完全一致的key原文 | 206 |
| 原文存在但没成功导出该key | 32 |
| 没有符合官方bbox容差的预测候选 | 20 |

这些状态按顺序分类；“OCR中没有”不等于图上没有，“官方bbox不匹配”不等于YOLO完全漏检。类型、框和文字错误可重叠，不可直接相加归因。

## 10. 没有成功的局部OCR补读实验

只跑一轮，不用GT选ROI：每图最多4个未解决位号的物理/通用候选，排除ground/power/net port，按检测分数和source ID排序。

- 实际71个ROI，3×放大，每轴边距max(24px,50%元件轴尺寸)，上下文边长上限384原图像素。
- 保留原3862条文字，新增309条原始OCR，精确原文＋重叠去重后增加161条。
- CPU总耗时约11.70秒。
- 合并后key命中277→281，但完整TP164→163，macro ComponentF1 16.69%→16.38%。

因此没有采用该合并策略，原缓存不变；不把文字更多当成准确率更高，也未继续搜索参数。

## 11. 引脚与网络：已做什么、没做什么

未实现完整、可信的pin identity和net系统，不应报告虚假的端到端成绩。

已做首5个dev的GT-components-input几何oracle：仅把GT元件bbox和图片输入候选生成，不把GT pin位置/编号/网络交给推理。候选生成后才与GT pin评价。

- 采用既有纯CV边界扫描＋框外固定3px候选点。
- 共853个候选，538个GT pin，≤5px同owner内一对一匹配64个。
- Micro precision7.50%、recall11.90%、F1 9.20%；这不是官方PinF1，因为没有编号/名称识别。
- 474个未命中GT pin全部位于元件框外超过8px；bbox至pin偏移最高55px。
- 说明固定框边偏移即使用GT框也不成立，需要沿实际引脚短线识别端点/长度，处理旋转、断线、NC和密集IC。

`non404`可参考颜色/暗线mask、骨架、graph、crossing candidates、pin短线扫描等纯CV函数，但完整旧流程存在缺失权重路径、付费API依赖和默认规则风险：

- 有的旧pin函数会将未知编号顺序填充，这不是识别结果。
- 未知/null crossing可能被默认合并，不可把它当已确认连通。
- 某些网络构建函数不输出无pin网络。
- 原生KiCad源解析可用于构造数据/独立验证，不能把原生源当任意PNG推理时偷偷使用的答案。

后续应区分真正的pipeline oracle（将GT输入上游并重新跑下游）与仅在最终JSON中替换组件/pin/net的output-substitution oracle。

## 12. 评分与数据协议尚未解决的问题

### 匿名ID

- 官方9171个元件中4034个key是∅形式，约43.99%。不仅有gnd/v/net label，也含电池、电机、蜂鸣器等真实部件。
- 规范倾向用可见名称＋重复序号，公开数据却常用不可见内部ID；pin自动编号也存在规范/数据不一致。
- 尚无主办方官方Q&A说明如何匹配这些ID，GT10与train200版本的匿名ID也不稳定。
- 不能断言全部不可得分，也不能凭空猜内部编号。严格完整评分为主，排除匿名的结果只是诊断。

### 类型与文字

- box/ic、r/resistor、mosfet与极性细分等存在词表版本差异。
- 推理输出有显式classifier→公开词表适配，评分主口径仍为原字符串严格相等；映射评分单列。
- 官方很多R/C value为null。未提取到输出null不等于确认图上缺失；不要为了匹配数据随意删除真实文字。
- IC Name不一定是标准料号，不能只用“字母开头＋数字”的正则读取。
- cpntLibrary含337个顶层条目，按官方IC Name精确lookup仅覆盖248/960个实例；别名/模糊匹配尚未作为已验证生产能力。
- 库中部分pin number重复，可能是叠脚/表示约定，不能直接转成唯一number→name字典。

### 坐标与框协议

- 内部：左上角原点、xyxy、原图像素。
- 导出：左下角原点，bbox变换为[x1,H-y2,x2,H-y1]。
- 连续像素坐标的H-y约定已实现并测试；新来源仍应做叠加图核查。
- 官方bbox可能是局部/内部符号范围，与检测器习惯的完整符号框不同。
- 不应通过修改GT迎合检测框；可以在train/dev上研究可复现的输出框协议适配。

### 本地评分器

- null对null算文字相同、空GT/空预测指标取1、bbox误差范数、网络匹配平局策略等是显式假设，不代表主办方已确认。
- 修复过一个真实问题：全局匈牙利目标中的线段奖励可能牺牲主要pin overlap。现在次级奖励权重上界覆盖全部可匹配线段总数，回归例验证不能牺牲主要目标。
- segment tie-break本身仍只是本地解释。

## 13. 已有验证、复现与产物

- 92项相关单元/回归测试通过，另3项局部OCR坐标/裁剪/去重自测通过，不代表所有历史测试均已完整重跑。
- 新组件推理真实缓存CLI烟测通过：命名字段与实验一致；未命名临时ID因排序可能改变，元件内容集合一致。
- 当前20图模型前向累计约8.14秒、原始OCR约141.50秒；初始化与完整系统开销未全计入，也不是正式容器CPU/GPU运行时间保证。

本地关键文件：

```text
COMPONENT_IDENTITY.md
EVAL_CONTEST6.md
outputs/component_identity_round_001_20261017/REPORT.md
outputs/component_identity_round_001_20261017/STATUS.json
outputs/component_identity_round_001_20261017/gallery.html
outputs/component_identity_round_001_20261017/experiment_final_v1/report.json
outputs/component_identity_round_001_20261017/experiment_final_v1/gallery.html
outputs/component_identity_round_001_20261017/experiment_local_ocr_v1/report.json
outputs/official_gold_crops_20261017/REPORT.md
outputs/official_gold_crops_20261017/gallery.html
outputs/pin_geometry_oracle_20261017/REPORT.md
outputs/pin_geometry_oracle_20261017/gallery.html
outputs/pin_net_interface_20261017/README.md
```

主要命令（在component_lab目录执行，必要时设置PYTHONPATH=src；输出必须用新目录）：

```powershell
.\.venv-yolo\Scripts\python.exe scripts/prepare_contest_dev_inputs.py --output outputs/component_identity_round_001_20261017/inputs
.\.venv-ocr\Scripts\python.exe scripts/run_local_ocr.py --manifest outputs/component_identity_round_001_20261017/inputs/manifest.json --engine rapidocr --limit 20
python scripts/run_component_identity_experiment.py --output outputs/new_identity_experiment

python -m component_lab.cli infer --image your.png --yolo weights/pcbnet_component_yolo26n_review_round_004.pt --resnet weights/resnet18_component_balanced_v1.pt --device 0 --tile-size 1024 --tile-overlap 0.25 --ocr-json your_ocr.json --association identity-v1 --contest-export --output outputs/new_infer

python -m component_lab.cli score --pred outputs/new_identity_experiment/identity_v1/predictions --gt data/official_contest6/train200 --split configs/contest6_group_split_v1.json --partition dev --anon-policy strict --type-policy strict --out outputs/new_score
```

最后一条score会看到空pin/net，这不能被解释为完整系统能力；组件阶段分析以ComponentF1与字段诊断为主。

## 14. 请Gemini重点回答的问题

1. 依据这些证据，下一轮最该改OCR检测、OCR识别、合并文本拆分、文字－元件关联，还是官方框协议适配？请排序并说明预期影响与成本。
2. 对R1/10k、U3/料号、引脚编号、网络名混排，有没有比当前全局slot分配更稳的工程模型？如何利用方向、文字组、元件结构，同时保留无法判断的候选？
3. 局部OCR增加token却降低完整F1，下一轮应如何做token合并、冲突消解和候选选择？请提出一轮有上限、无GT参与推理的实验，而非大量参数搜索。
4. 如何把不完整的官方bbox裁剪转换成可靠分类训练输入，同时保持评测GT不变？上下文扩展、符号分割、连通结构、双视图分类各有什么风险？
5. 27语义家族和38原始类型应怎样分层建模？是否让ResNet只处理D/Q等歧义组更划算？如何报告每类真实设计支持、缺失类别和弱标签误差？
6. 已证实固定框外3px定位不行。对KiCad/AD/JLC彩色与datasheet黑白图，怎样识别实际pin短线端点、方向、编号和名称？哪些应学习，哪些适合确定性图像/图算法？
7. 有哪些可靠的无需人工标注的数据构建方式，例如原生EDA自动渲染配对、字体/布局/噪声变化？怎样保证位号、bbox、pin和net标签仍然一致？
8. 对当前分组去重与污染控制，是否存在明显的误合并、隐性泄漏或评价偏差？已观察过的公开数据应如何诚实报告？
9. 如果只能新增一周工作，如何分配component identity、pin identity、net connectivity、runtime验证的优先级？请给每个阶段可以验收的指标与停止条件。
10. 如何设计局部VLM裁决接口，让它选择有ID的OCR/类别候选而非自由改写？在指定模型、Token得分和2小时限制下，怎样验证值得调用？

请不要假设我们已做到95%、已拥有人工分类真值、官方评测器已发布、所有官方框可直接训练、现有checkpoint在新test上无污染，或单纯换更大模型就能解决身份问题。若需要额外信息，请明确指出应补充哪种图、JSON字段或失败案例。
