# 从这里开始

2026-10-18 已集成局部v6五图真实CLI对照完成：[逐图结果与裁图](outputs/integrated_v6_five_case_test_20261018/gallery.html)。固定既有5张dev、37类短训1024的76框/v5缓存，两组仅切换local-v6；严格macro ComponentF1 14.10%/16TP→15.40%/17TP，新增1、损失0。4个元件共6字段改变，仅R3补10kΩ成为完整TP；K6/K7与公开GT编号不一致，R4330本身位号错误，补文字未提分。3图触发5ROI，2图不触发仍计分；2张新worker共74.82秒，0131精确复用旧生产缓存；两臂CLI合计10.66/86.35秒，不含已缓存YOLO/ResNet/主OCR推理。10份预测先冻结再读GT，独立复核哈希与指标一致。原权重/标签不变、API0、test40未读；固定dev小样，不能外推全量或混称旧13.29%独立流水线基线。

2026-10-18 局部v6已按用户授权接入主 `infer --association identity-v1`，提供主OCR输入后默认补必要空字段；legacy/无OCR明确跳过，原检测复核工具不增加OCR。每图最多2ROI、20%面积、320px最长边、2x原RGB、CPU双线程、120秒超时；无API，不改模型/人工标签。已有key/非空字段/bbox/type锁定，内部evidence与比赛JSON统一导出，缓存重放不重复识别。关闭用 `--no-local-v6`，更保守用 `--local-v6-mode unresolved-only`。见[使用说明](LOCAL_V6_WORKFLOW.md)与[接入验证](outputs/local_v6_integration_20261018/REPORT.md)。下文“不自动启用主infer”为此前试验结论，授权接入以本条为准。

2026-10-18 局部v6补读实际试验完成：[ROI与结果](outputs/targeted_v6_recovery_20261018/gallery.html)。沿用五图/37类短训1024/固定76框，3图触发5个ROI，另2图baseline回退仍评分。v6 medium CPU额外worker73.13秒、识别71.32秒，原图2x裁剪；ROI/参数在推理和GT前冻结。基线严格macro13.29%/15TP；旧named全锁safe版仍15TP；允许补必要空字段的shadow为R3补10kΩ，14.58%/16TP，新增1、旧TP损失0。修复同位置重复reference anchor及anchor可迁移的合并缺陷，初次零增益产物保存在initial_assembly，修复仅离线重放、无新OCR。41项回归通过，旧named/原crop像素/坐标/15份预测哈希核验通过。0024图面K*与GT U*编号冲突另存证据。API0、无训练、旧模型/标签/默认不变，test40未读；收益有限，不自动扩展或启用主infer默认。

2026-10-18 局部频域OCR选区探针完成：[频域热图与候选区域](outputs/frequency_ocr_regions_20261018/gallery.html)。固定前五图，先用图片/v5证据生成并冻结ROI，再评价v6机器缓存覆盖；128px局部FFT，20%裁剪面积预算、最多8区。纯高频平均面积14.24%，覆盖22/52个v6新增位号样式候选；加入类字符连通域/长线抑制等的简单组合为21/52，未显示优势。它们不是GT召回率；本轮没有执行ROI v6、没有实际提分/加速结论，原模型与默认不变。核实上轮回退：C8→=C8使位号过滤，22uF仍读到；R6→Rô后错配D22，3.3kΩ仍读到。五图153非匿名GT中54无key原文、67有字面但未导出（其中65无同时满足type/20px的框候选）、32命中key但仅20完整正确；不能全部归因OCR。10套选区预算/坐标/哈希检查通过，API0、test40未读。

2026-10-18 non404旧规则component适配运行＋OCR五图对照完成：[图库与报告](outputs/non404_component_ocr_20261018/gallery.html)。原41类YOLO11n权重/类别表缺失，不能复现历史44.31%；本次实际运行现有37类YOLO26n（1024）及原non404 GT-free规则函数，非旧Q123+Qwen27B。固定5图/76预测/229GT，用当前关联strict macro：旧v5缓存13.29%/15TP，v5整图9.62%/11TP，v6 medium整图15.65%/20TP；v6相对旧缓存新增7、损失2、净增5。位号原文字面覆盖91→99/153，仅为字符串存在性诊断。v6 ONNX CPU五图819秒，v5缓存计时38秒，非隔离benchmark；可选v6多视图扩展因耗时在读GT前停止，仅2/5不计主分。原non404规则mapped诊断9.11→11.22%，不可混称历史44分。11项原规则测试通过，无API、无训练、未改默认/标签/原权重，test40未读。v6值得进一步试轻量配置或限定补读区域，未全量替换v5。

2026-10-18 整图Qwen文字归属对照完成：[新图库与报告](outputs/fullpage_component_identity_20261017/gallery.html)。同dev20、同632个短训整图两阶段框/type/OCR；20/20请求成功、189965 token、约430秒HTTP时间，无付费重试。规则基线严格macro18.3238%/163TP；全字段替换18.1773%/171TP（新增19、损失11；micro20.8573→21.8810%）；只补未命名key18.4228%/164TP（+1、旧TP保留）。未出现可稳定采用的宏分增益，不启用生产、不追加API。确认0038的C21/C22图面有12pF/50V而GT value=null，正确归属该文字反而丢2TP，见[证据页](outputs/fullpage_component_identity_20261017/value_conflict.html)和主办方问题草稿；GT和评分规则未改。11项检查通过，全部原weights哈希未变，test40未读。该试验不支持仅用“缺全图Qwen”解释non404历史44.31%，也不是旧模型/用例的复现。

2026-10-17 三臂架构短训对照完成：[同图三臂图库与指标](outputs/multiclass_vs_two_stage_20261017/gallery.html)。仅3050 Ti、官方train140/dev20；两组YOLO从通用权重各16轮，共用新ImageNet ResNet18训练12轮，37个train raw types，按类每轮64次曝光。整图严格ComponentF1：多类17.50%、两阶段18.32%、混合17.70%；1024切片：15.65%/17.13%/15.94%。切片类型macroF1：14.73%/26.36%/17.35%，多类直出更快。新ResNet在931份GT上下文裁剪上的accuracy94.84%、macroF1 73.85%，不等于检测后全流程分数。差值bootstrap区间跨0、仅短预算单seed；暂不支持删除ResNet，默认业务模型保留。切片两阶段新增178个预测但严格TP仍163，后续应优先文字身份/归属而非盲目加框。原权重和数据哈希核验通过，6项针对性检查通过，API0、test40未读；不是与旧86%mAP的横向对比。

2026-10-17 第5批20图已冻结、训练和严格对照完成：[任务清单与本轮汇总](outputs/component_improvement_round005_20261017/gallery.html)。历史100图审核，设计组排除后本次保留92图；训练228图/10596框。3050 Ti，1280/batch1，实际8轮best3（CSV为准，原始日志含重复最终验证回调）。同设置重评Round004→005：P88.95→89.33、R84.39→85.44、mAP50 86.63→87.25、mAP50–95 53.31→52.48；固定OCR/ResNet/关联的严格dev20则17.42%/170TP→17.32%/169TP，默认保留Round004。双模型缓存仅恢复旧未命名key/type的shadow为17.49%/172TP、旧328命名不变，未接主infer。另完成[原生源文件×人工框对照](outputs/native_reviewed_alignment_20261017/gallery.html)：19图794对IoU50匹配，14条源语义银标候选（5R/9C），未入训练；缺纸张尺寸1图隔离。API0、test40未评、原权重未覆盖；7+4项新回归通过。

2026-10-17 YOLO超参数回顾与单变量短训已完成：[报告与曲线](outputs/yolo_hparam_review_20261017/gallery.html)。3050 Ti、官方train140/dev20、相同通用预训练/seed42/1280/batch1/nbs16，各8轮；复用RGB基线，统一重评mAP50/mAP50–95为42.80%/23.44%，仅lr0降至.0003为32.05%/17.21%，仅box权重升至15为42.78%/23.37%。本次没有支持替换业务配方；仅为短预算单seed，不是充分收敛后最优参数，也不能与旧PCBnet30图86.20%直接比。固定阈值P/R与仅几何20px诊断见报告。全部原weights与train/dev图片/标签哈希未变，未训练Small、未调用API或评测test40。必要文字57项审计的解释也整理在新报告。

2026-10-17 二值化YOLO小样已独立完成：[报告与曲线](outputs/binary_yolo_3050_20261017/gallery.html)。按用户要求仅3050 Ti（本次cuda:0），RGB与灰度+Otsu两组各从通用yolo26n.pt训练8轮，1280/batch1；train140/dev20、6417/931框。统一重评RGB mAP50=42.80%、mAP50–95=23.44%；二值源40.87%/22.25%，本次未显示优势。新数据/短训不能与旧PCBnet验证86.20%直接比。全部13份原权重SHA256未变，默认模型未替换；新权重仅在该实验runs目录，不追加训练。当前硬件索引与旧说明不同，实验脚本按3050 Ti名称验证。

2026-10-17 未命名恢复路线完成：[新报告与图库](outputs/unresolved_identity_recovery_20261017/gallery.html)。24个纯预测新目标、6/6 Qwen调用、11,832 tokens；仅恢复key/type并保留原字段得到17.7680%/177TP，新增7、旧170全部保留。328个旧named记录哈希一致，检测总数1040不变。新CLI `recover-identities` 默认只恢复key/type，离线不调用API，见[恢复工作流](RECOVERY_WORKFLOW.md)；主infer默认不变。134项相关回归及真实缓存CLI通过。缓存＋新读取全部字段另得18.0826%/177TP，属于不同shadow，不能混称CLI结果。误差中发现复合符号拆分/归属及公开位号位置冲突，见新图库的eval-only审计。本轮已停止追加API。

2026-10-17 Gemini双向文本组建议已做完受控验证：[新配对图库](outputs/qwen_same_roi_safety_20261017/gallery.html)、[本地消融](outputs/group_owner_coupling_20261017/REPORT.md)。保守反向锚仅+1TP（17.45%），三锁强制清空损失37旧TP，拒绝采用。相同32目标/像素/字典的Qwen测试16/16成功、34,090 token；直接读图替换的安全压力shadow新增9、损失5、净增4TP（17.82%），五个损失均为正确null被补非空，非生产成绩。仅2个实际触发目标无TP净增。现有默认与17.42%本地基线保留，未新增训练或启用Qwen。113项相关回归通过。完整解释见 `research/gemini_group_coupling_review_20261017.md`。

2026-10-17 Qwen局部文字旁路已实测：[对照图库与报告](outputs/qwen_component_trial_20261017/gallery.html)。固定DEV20基线17.42%/170TP；只选OCR候选ID无净增；直接读局部图仅补空达到17.85%/174TP，允许替换文字为18.01%/175TP。后者净增5个完整正确项，用14,876 token；两轮合计26,565 token。现有命名key/bbox锁定，未训练、未改标签或启用生产；30个所选目标基线均未完整得分，零丢失不能作为安全性保证。20项相关测试通过。

2026-10-17 已继续推进至当前规则路线的明确瓶颈：[新报告与图库](outputs/advance_until_bottleneck_20261017/gallery.html)。修复原文 `4.7K` 的参数语法后，固定DEV20严格Component F1为17.42%/170TP（此前17.39%/169TP）。局部空值填充退化，不采用；颜色pin-tip的micro精确率32.58%、召回5.39%，覆盖不足且不是官方PinF1。121项相关测试通过，无新增训练/API/test40评测。少数公开位号与图片不一致，证据和协议问题已整理；Pin身份与Net仍未完成。

2026-10-17 午餐无人实验已完成：[汇总与图库](outputs/unattended_20261017_lunch/gallery.html)。保守字符span＋box显示Name候选在固定DEV20上为17.39%（旧identity_v1为16.69%），通过 `--ocr-spans --display-names` 显式启用。19.43%另属r/c公开bbox模板协议诊断，未接入生产；部分GT位号与图片不一致，须先看报告限制。无新训练或付费API。

2026-10-17 最新进展：优先阅读 [COMPONENT_IDENTITY.md](COMPONENT_IDENTITY.md) 和 [Gemini交接材料](GEMINI_HANDOFF_20261017.md)。第四轮检测权重仍为当前模型；新身份关联是显式 opt-in，固定DEV20严格Component F1为16.69%，不是独立测试成绩。下方保留的第三轮权重、网页批次和默认参数说明包含历史信息，不能替代最新实验记录。

第四轮YOLO已完成（2026-10-16）：累计80张人工审核，训练243图/11025框，第四批PCBnet与HF各10图已冻结。新权重 `weights/pcbnet_component_yolo26n_review_round_004.pt` 在原30图验证集上P88.83%、R84.00%、mAP50 86.20%、mAP50–95 53.24%；相对第三轮召回略升、精确率下降。原验证/测试哈希不变，HF泛化尚无独立验证，下一批未生成。[第四轮报告](outputs/active_learning/round_004/REPORT.md)。ResNet仍使用下述均衡版本，本批定位审核未被当作人工分类真值。

ResNet18均衡版已完成训练（2026-10-16），使用4060 Ti（PyTorch `cuda:0`）。权重为 `weights/resnet18_component_balanced_v1.pt`，数据为 `data/resnet/balanced_20261016`：27类各256份，共6912份训练文件，原图裁剪2251份、训练增广4661份，验证集951份不做增广。弱标签验证一致率96.74%、宏F1 73.94%，部分稀有类真实来源仍不足。完整来源、类别数量、训练图表与HF数据核验见[分类训练报告](outputs/resnet_bootstrap_20261016/REPORT.md)和[分类图库](outputs/resnet_bootstrap_20261016/gallery.html)。类别标签为原始弱标注/AI银标，人工审核只确认了部分检测框几何。

已新增[无需新增人工标注的自动流程](D:/大四上/EDA/EIT&BTD/component_lab/AUTO_WORKFLOW.md)：把图片拖到[自动处理原理图.cmd](D:/大四上/EDA/EIT&BTD/component_lab/自动处理原理图.cmd)，本地生成候选、截图与交互预览；Qwen为可选有预算旁路。完整裁块训练对照和Qwen真实试跑已完成，见[自动化首轮报告](D:/大四上/EDA/EIT&BTD/component_lab/outputs/automation_round_001/REPORT.md)。本轮未达到双95%，未用实验模型或Qwen建议替换现有结果。

已建立[元件预标注与人工复核工作流](D:/大四上/EDA/EIT&BTD/component_lab/REVIEW_WORKFLOW.md)。三批共60张已审核并完成第三轮Nano微调，233张训练图累计10279个框。当前第四批20图页面为 http://127.0.0.1:8768/ ，PCBnet10张＋HF10张。最新检测训练结果及双95%方案见[第三轮报告](D:/大四上/EDA/EIT&BTD/component_lab/outputs/active_learning/round_003/REPORT.md)：P89.89%、R83.42%、mAP50 85.91%，尚未达到双95%。

第一批1687个人工保留框的Qwen分类缓存已复用，历史[实测报告与图集](outputs/qwen_crop_classification_round_001/README.md)和 `data/resnet/qwen_silver_round_001` 保留。本轮新增Qwen上下文/紧框复核及独立智能体抽查；分类训练与验证按原PCBnet原图划分，排除跨集合完全相同的裁剪，原测试集未使用。

1920 训练对照已经完成（Nano、batch 1 显存足够），结果见 `TRAINING_1920.md`。
最新人工修正轮次权重为 `weights/pcbnet_component_yolo26n_review_round_003.pt`，输入1280；切图对照与候选复核方案见第三轮报告。旧基线及旧权重保留。

工作目录是 `component_lab`。YOLO26 Nano定位元件，统一输出component；ResNet18识别27个语义家族。最初单类基线权重及验证/测试记录见 `TRAINING_COMPONENT.md`；34类旧基线保留在 `weights/pcbnet34_yolo26n_v1.pt`。复核工作流默认不调用API；Qwen元件分类实验不能直接代表误框过滤能力。

## 你放数据的位置

1. `data/yolo/`：完整原理图图片、YOLO 框标签、dataset.yaml；目前包含从 PCBnet 原始数据导出的 `pcbnet_component_v1/` 单类数据。你自己的新数据可另建子目录，格式见目录内 README.md。
2. `data/resnet/`：单个元件裁剪图片，按 train/val/test 和类别文件夹组织；格式见目录内 README.md。

`data/resnet/balanced_20261016`为当前分类数据，`data/resnet/active_dataset.json`记录模型与数据路径；预留的train/val/test目录仍可放入以后自己的数据。此前由比赛数据生成的 `data/prepared/` 保留参考，不会自动混入新版本。

## 程序流

```text
原理图 → YOLO26 Nano → bbox + component
                         ├→ 符号截图 → ResNet18 → 具体类别
                         └→ 符号截图 + 周边截图 → Qwen3.8-Flash API → 文字/位号/型号候选
```

`models.py` 包含 YOLO26 Nano 与 ResNet18 模块；`qwen_api.py` 是独立的图像 API 模块；`pipeline.py` 把它们串起来；`training.py` 提供训练入口。截图始终保存，即使没有开启 Qwen。另有可选 PaddleOCR 和已有 OCR JSON 入口，可以后使用。

## 先建环境

在 component_lab 目录执行，建议 Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[vision,api]"
```

本机默认优先4060 Ti：YOLO使用`--device 0`，ResNet训练使用`--device cuda:0`。3050 Ti在PyTorch中对应cuda:1，与nvidia-smi的显示序号不同。Qwen模型ID默认qwen3.8-flash，服务商地址由实际配置提供。

## 超参数

修改 `configs/defaults.json`，或用命令行参数覆盖。通用默认：YOLO 输入 1280、置信度 0.25、50 epoch、batch 8；本机 4 GB 显卡上的 PCBnet 单类实验另用 batch 2、有效 batch 16，实际参数保存在运行目录 `args.yaml`。ResNet18 输入补白到 224、学习率 0.0001、50 epoch、batch 8、替换类别阈值 0.8；Qwen 上下文边距 48 像素、最多 10 次调用、timeout 60 秒、输出上限 800 token、temperature 0。

## 放完数据后训练

```powershell
component-lab train-yolo --data data/yolo/pcbnet_component_v1/dataset.yaml --output runs --device 0
component-lab train-resnet --data data/resnet --output runs/resnet --device cuda:0
```

没有 GPU 改用 `--device cpu`。当前 YOLO 只有 `component` 一类；ResNet 训练目录应按原始具体类别建文件夹，两者类别名无需相同。ResNet 的 val 必须具有与 train 同名的类别目录，少样本类别允许目录为空，但要关注缺失类别警告。ResNet 保存自己的类别映射。训练与验证先按整张原理图划分，再裁剪，避免同图泄漏。

## 推理与截图

```powershell
component-lab infer --image "your_diagram.png" --yolo "runs/pcbnet_component_yolo26n_v1/weights/best.pt" --output outputs/example
```

使用最新检测权重 `weights/pcbnet_component_yolo26n_review_round_003.pt`，并加 `--resnet weights/resnet18_component_balanced_v1.pt --device 0`即可检测并分类。默认不调用Qwen。大图可加`--tile-size 1024 --tile-overlap 0.25 --tile-max-tiles 100`；本轮YOLO→ResNet整图推理示例在 `outputs/resnet_bootstrap_20261016/demo_components`。

配置环境变量后，在上面的命令后加 `--qwen`：

```powershell
$env:QWEN_API_KEY = "你的API密钥"
$env:QWEN_BASE_URL = "你的服务商OpenAI兼容地址"
$env:QWEN_MODEL = "qwen3.8-flash"
```

`.env.example` 只是变量模板，程序不会自动读取 .env。可用 `--qwen-model` 覆盖模型，`--qwen-max-calls 3` 限制调用次数；截取边距可用 `--context-margin 48` 调整。仅调用支持图像输入的模型。接口请求无自动重试，失败会记到各元件的 qwen 状态中。

输出：

- `components.json`：元件框、内部编号、YOLO/ResNet 结果、OCR 字段、Qwen 独立候选和原始响应。
- `overlay.png`：检测框预览。
- `crops/component_0001_symbol.png`：元件符号截图。
- `crops/component_0001_context.png`：周边截图，包含位号/参数附近区域。

注意内部编号 component_0001 与真实位号 R1 不同。Qwen 结果保留在 qwen.result，暂不自动覆盖 reference/Name/value；这是为了让你能评估 API 输出。周边截图包含邻近元件时仍可能误归属，需要人工核查。

## 当前验证范围

已验证坐标转换、OCR关联规则、单类GPU整图与切图推理、Qwen真实API、本地人工复核与隔离导出，以及ResNet18真实训练、严格类别配额、数据分组/哈希检查、公开模型加载器与YOLO→ResNet整图推理。检测指标见 `TRAINING_COMPONENT.md`及后续轮次报告；分类指标和监督质量限制见本页开头的分类训练报告。

详细字段约定和数据准备参考 README.md。官方图像 API 参考：[阿里云文档](https://docs.modelstudio.console.alibabacloud.com/zh/model-studio/vision)。
