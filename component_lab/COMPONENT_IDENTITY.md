# Component identity implementation — 2026-10-17

2026-10-18 更新：[局部v6补读已接入主 infer](LOCAL_V6_WORKFLOW.md)。`identity-v1` 且有主OCR输入时默认启用 `empty-fields`；只填空字段并锁定已有位号/非空文字。`--local-v6-mode unresolved-only` 锁定原named完整记录，`--no-local-v6` 恢复原主OCR路径。下文“默认未变”均为历史实验记录。

最新[未命名恢复实验](outputs/unresolved_identity_recovery_20261017/REPORT.md)：冻结328个旧named记录，只替换未命名key。24个新目标、6/6成功、11,832 token；key/type-only为17.7680%/177TP（新增7、旧170全部保留），同步改字段为17.6601%/174TP。25个历史缓存单独重放fields为17.8432%/173TP；缓存＋新读取fields为18.0826%/177TP，属于不同位置的TP，不是key-only命令成绩。所有预测数仍1040、bbox与其他部分不变，134项回归通过。新增[RECOVERY_WORKFLOW.md](RECOVERY_WORKFLOW.md)及离线CLI `recover-identities`，默认不改已有字段或发起API。主infer默认未变。

最新[相同ROI配对与安全测试](outputs/qwen_same_roi_safety_20261017/REPORT.md)已完成16次Qwen调用（全部成功、34,090 token）。固定30个哈希安全探针+2个预测触发目标，实际含6正确null/9错误null/7正确非null/8错误非null。全目标强制写回：ID补空16.83%（+2/-5TP），ID替换16.86%（+3/-5），直接补空17.53%（+6/-5），直接替换17.82%（+9/-5）。五个丢失均为原本正确null；7个正确非null全部保留，不能外推普遍安全。只写入2个实际触发目标时四种方案均无净增，因此不能将17.82%宣布为生产提升。与旧18.01%试验目标不同，不能直接作优劣结论。默认不启用。

[零API文本组消融](outputs/group_owner_coupling_20261017/REPORT.md)：保守反向参数锚在原紧凑group内仅修复0200/R31，17.42067%→17.45079%/171TP；未达到预声明净增≥2采纳线。点号U子单元语法未新增完整TP。强制三锁清空清除90字段，新增3而损失37旧TP，退至14.38%，不采用；这不否定所有双向关联方法。本轮仅增加显式实验接口，未改默认、权重、标签、封存测试。113项相关回归通过。详见工作区 `research/gemini_group_coupling_review_20261017.md`。

最新[57项纯文本审计](outputs/component_text_only_audit_20261017/REPORT.md)：预测空/GT非空18项、预测非空/GT空12项、较小字面差异17项、较大差异10项。18项漏填中12项全页已有精确OCR原文、6项没有；不能只归因于OCR漏读。170个完整正确元件里有11个必要文本为正确null，故补空也可能损失旧TP。既有Qwen只修复原57项中的2项，原cohort剩55；总增益4/5还包含其他位号恢复。此审计零API/模型前向/训练，不改变既有预测与分数。

Qwen实验：[统一报告](outputs/qwen_component_trial_20261017/REPORT.md)。在同一批prediction-selected的8页30目标上，仅选择OCR候选ID无完整TP净增；直接读局部文字并扩大context后，仅补空得到17.85263%/174TP（+4），允许替换得到18.01136%/175TP（+5），原基线17.42067%/170TP。直接版14,876 token，两轮共26,565；15次成功、1次DNS失败，无自动付费重试。共同成功7页诊断仍净增3/4TP；本轮目标基线均未完整正确，不能据0丢失推断改写安全性。属于机器建议/开发shadow，未接生产默认流程。相关20项测试通过，原标签/权重/封存测试不动。

当前按用户职责只推进Component。新增[冻结阻塞审计](outputs/component_blocker_audit_20261017_v2/REPORT.md)：36个有原文未导出项为22歧义、6竞争、6角色过滤、2重复导出；112个key命中失败项为57仅text、24仅bbox、10仅type、21多项失败。20图reference重放一致，无新模型前向/API/训练或分数变化。key命中不证明owner正确；仅大小写命中也可能来自另一位置。完整评审见工作区 `research/gemini_component_review_20261017.md`。

后续一轮已完成：[推进至瓶颈报告](outputs/advance_until_bottleneck_20261017/REPORT.md)。decimal 大写K原文语法修复令 span+display Name 的严格DEV20宏F1从17.39055%/169TP升至17.42067%/170TP，恢复0200/R34的 `4.7K`；无字符串归一化。reference-local空value补全降至17.38176%/169TP，仅实验保留，未接生产。颜色pin-tip只作5图GT bbox点位oracle（micro P32.58/R5.39/F1 9.25%），不是官方PinF1，覆盖不足。121项相关测试与真实缓存CLI通过，无新训练、付费API或封存测试评测。公开身份协议的具体矛盾见新报告，不按GT重编号预测。

午餐后新增显式实验开关 `--ocr-spans --display-names`，均要求 `--association identity-v1`。两项组合在同一DEV20为17.39%/169TP（相对16.69%/164TP新增5、丢失0）。默认关闭，原版本保留。详见 `outputs/unattended_20261017_lunch/REPORT.md` 与图库；公开r/c框模板诊断19.43%不能当真实框定位提升，尚未生产启用。

当前主线是把 YOLO 框、ResNet 候选与 OCR 证据连接成可评分元件。此轮没有训练新模型、没有调用付费 API，没有打开封存测试图运行新评测。

## 固定开发实验

- 划分：`configs/contest6_group_split_v1.json` 的 dev20。
- 检测器：`weights/pcbnet_component_yolo26n_review_round_004.pt`，whole image + 1024px tiles，1280 input，conf=0.25。
- 分类器：现有 balanced-v1，8%/至少2px context，0.8 以下保留 component，不将置信度当作校准后的正确率。
- 硬件：PyTorch cuda:0 已核验是 RTX 4060 Ti；没有恢复 Small 训练。
- OCR：本地 RapidOCR，原图坐标，原文保留。
- 旧权重对新划分存在历史暴露，全部结论仅用于开发，不宣称独立泛化。

总入口为 `outputs/component_identity_round_001_20261017/gallery.html`。`experiment_final_v1/REPORT.md` 和 `gallery.html` 保留所有20图及四个预声明版本，首次结果在 experiment_v1。

| 版本 | 严格 max-coordinate Component F1（逐图平均） | 完整正确元件 | key命中 |
|---|---:|---:|---:|
| baseline | 12.95% | 118 | 225 |
| prefix：分类弃权时使用明确位号前缀 | 13.77% | 128 | 223 |
| unclamped：再保留关联负代价 | 16.69% | 164 | 277 |
| identity_v1：再增加 IC Name 原文候选回退 | 16.69% | 164 | 277 |

关键修复：原关联器将负代价截断为零，会令若干不同候选都变成0，触发人为平局弃权。新版本保留差异，但仍保留真实歧义弃权；没有直接取消所有审核条件。IC Name 的回退在这20图上没有额外收益，不能将其当作已经验证的普遍规则。

baseline→identity_v1：+3.74个百分点，按案例配对 bootstrap 区间约[+1.86,+5.81]个百分点。该区间不覆盖评分规则不确定性，也不是独立确认性检验。

identity_v1 的535个非匿名GT状态：164完整正确，113 key命中但其他字段错误，206在缓存中没有完全相同的key原文，32有完全相同的OCR key但未成功导出，20没有符合官方20px框容差的候选。分类有顺序，不能把“无官方框匹配”当作纯粹漏检；没有相同OCR也不等于图上没有文字。

第一轮本地模型前向累计约8.14秒、OCR约141.50秒（20图，初始化等开销不全计入）。这不是完整系统或官方容器运行时间保证。

额外受限局部OCR只跑了一轮：71个ROI、CPU11.70秒、新增161条原文。key命中277→281，但完整TP164→163，Component F1降至16.38%，因此不采用。负结果保存在 experiment_local_ocr_v1，不继续盲目重试。

## 推理入口

新流程为显式选项，原有默认推理不变。安装项目时可通过 `.[identity]` 引入 SciPy；本机已有依赖，没有自动安装或下载。

```powershell
python -m component_lab.cli infer `
  --image outputs/component_identity_round_001_20261017/inputs/0055/source.png `
  --detections outputs/component_identity_round_001_20261017/inputs/0055/components.json `
  --ocr-json outputs/component_identity_round_001_20261017/inputs/0055/ocr_rapidocr.json `
  --association identity-v1 --contest-export --qwen-max-calls 0 `
  --output outputs/identity_demo_new
```

也支持原来的 `--yolo <weights> --resnet <weights>` 真实模型入口；OCR JSON 可以是列表或含 texts 的对象。内部仍是左上角坐标，`contest_components.json` 转为左下角坐标。pins/nets 为空，明确不是完整提交系统。

- key、Name、value 从关联证据的 raw 字符串取得，不自动改变大小写、空格或单位。
- R/C/L 等明确前缀仅用于分类弃权候选；不强行判定 D/Q 极性或细分类。
- 无位号/重复位号检测保留为 `__unresolved_detection_*`，不猜测官方隐藏 ∅ 编号，不通过删除困难检测提高分数。
- `component_0001` 与 unresolved 编号是局部追踪ID，不是跨版本稳定身份。
- 缺少文本证据输出 null，不代表已经确认图上没有该字段。
- Qwen没有参与此轮，未来仅对候选冲突做局部实验，不能把建议当人工或官方真值。

实际缓存CLI例子已执行：命名元件字段与实验导出一致；未命名局部ID可因排序改变，元件内容集合一致。

## 复现开发对照

```powershell
.\.venv-yolo\Scripts\python.exe scripts/prepare_contest_dev_inputs.py --output outputs/component_identity_round_001_20261017/inputs
.\.venv-ocr\Scripts\python.exe scripts/run_local_ocr.py --manifest outputs/component_identity_round_001_20261017/inputs/manifest.json --engine rapidocr --limit 20
python scripts/run_component_identity_experiment.py --output outputs/identity_experiment_fresh
```

输入缓存以源图、划分、模型hash校验后复用。实验输出必须选择新目录。所有预测版本先完成导出，再加载开发答案评分；预测函数没有GT参数。

## 官方分类数据与引脚实验

`data/resnet/official_gold_v1` 已从 train140/dev20 导出6417/931份候选裁剪，38个原始类型，附设计组与来源证据。测试标签/图像未读取或裁剪。但部分官方框是符号局部或内部，原文标签也未经独立验证；不是可以直接灌入训练的已审定数据。报告 `outputs/official_gold_crops_20261017/REPORT.md` 设置 training_ready=false，原框与外扩对照用于诊断，不改变GT。尚未替换现有分类器。

`outputs/pin_geometry_oracle_20261017/REPORT.md`：首5个dev，用GT元件框作为上游oracle，仅用图像提取候选点。固定框外3px定位的micro点位F1只有9.20%，不是官方PinF1。474/538个GT引脚在框外超过8px；需要实际引脚短线/端点识别，不能用框边加固定偏移代替。

评分器已修正 segment 次级目标可能覆盖全局 pin overlap 主目标的问题，回归测试保留。segment tie-break 本身仍为明确评分假设。

## 后续自动实验边界

当前最大的已知剩余瓶颈是位号OCR与文字归属，随后是型号/参数转写、分类词表和符号框协议。新实验只能在dev选择规则；测试保持封存。少数类需要更多独立设计或可靠原生源生成，不能靠复制文件声称样本均衡。
