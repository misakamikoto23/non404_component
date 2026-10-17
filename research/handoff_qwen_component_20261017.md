# Qwen Component局部旁路交接 — 2026-10-17

用户明确只负责Component，并授权调用qwen3.8-flash试验，要求考虑比赛token评分。未使用子代理，没有新训练、标签/权重修改、封存test使用或外发沟通。

统一入口：`component_lab/outputs/qwen_component_trial_20261017/{REPORT.md,gallery.html,combined_report.json,STATUS.json}`。

两轮：
- `qwen_identity_pilot_20261017`：prediction-only选择8页30目标，候选ID裁决。0168第一次因sandbox DNS失败，无HTTP响应，保留5448 token预算占位，不重试；联网允许后继续其余7页，7次正常、26目标、11689实际token。补入10位号，但完整TP仍170，F1仍17.42067%。
- `qwen_direct_text_pilot_20261017`：同一批30目标/8页，不按GT错误重新筛；扩大context且直接联合读取reference/Name/value。8正常请求，14876token、22.93秒。仅补空17.85263%/174TP；允许替换18.01136%/175TP。新增0011/R29、0011/R8、0024/R3、0168/U118，另0132/U9只在替换版完整得分。
- 两轮共有15次API成功、1次DNS失败，提供方报告26565token（24471输入/2094输出）。共享账本`outputs/unattended_20261016/qwen_budget.json`保留累计16次尝试/32013预算token，其中5448是失败占位，不是实际模型消耗。没有清零或自动付费重试。请求模型及实际返回均qwen3.8-flash，用户指定端点原样沿用，密钥不打印或持久化到新文件。

限制：
- 第一轮0168没有响应，不能称8页严格配对；共同成功7页诊断仍净增3/4TP。
- 第二轮同时改变context和允许直接转写，不能归因于某一个变化。
- 30目标基线完整正确数为0，所以0TP丢失不能证明对已解决目标安全。新字段是VLM机器建议，绝非人工或训练gold。
- 保持命名key与bbox冻结，新ref仅可填unresolved；类型仅对原先component弃权使用已有prefix先验。主流程默认未接新模块，现有纯本地基线仍17.42%。
- 现有视觉权重历史暴露、固定开发小样、未封存测试，不能宣称独立泛化或95%。比赛模型/token计量和排名可能不同，不能由token直接推算最终净分。

代码：`src/component_lab/qwen_identity.py`；`scripts/pilot_qwen_identity.py` 的 prepare/run/report；`scripts/pilot_qwen_direct_text.py` 同名动作。report只读缓存、免费。run是付费操作；所有已尝试请求不再自动重试。每轮最多8请求，继续遵守共享账本上限。对应20项接口/选择/导出回归通过，日志在汇总目录tests.log。

推荐下一步先增加已正确目标控制与更广prediction-gated开发对照，再考虑仅补空旁路正式接入。不按本轮GT获益case写死触发器；不自动扩大调用或恢复训练。
