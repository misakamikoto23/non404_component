# 2026-10-17 推进至瓶颈交接

工作目录 `D:\大四上\EDA\EIT&BTD\component_lab`。最新报告为 `outputs/advance_until_bottleneck_20261017/REPORT.md` 和 `gallery.html`，接手先读；旧午餐报告保留。

已做：
- `text_roles.py` 支持 decimal 大写 K，精确保留 `4.7K`。固定 dev20 + span + display Name 严格 Component 宏 F1 17.39055%→17.42067%，169→170TP，新增0200/R34。
- `pin_transitions.py`、`probe_pin_color_transitions.py` 完成合成测试和既有首5 dev GT bbox oracle：89候选/29TP，micro P32.58/R5.39/F1 9.25%。不是官方 PinF1；与旧固定3px相比精确率升、召回降、F1几乎不变，不替代生产定位。
- `local_values.py` 实验只填空且锁定 token/span，一次冻结 dev20 对照17.38176%/169TP，新增0丢1，不接生产，实验脚本 `--local-values` 保留复现。
- 只读身份审计：43 key命中bbox失败项里7疑似编号差异、4关联歧义、32未知；不改GT或预测编号，不外推错误率。见 `IDENTITY_AUDIT.md`。
- 121项相关测试通过，真实缓存 CLI 0200运行成功，R34.value 为原文4.7K。

未做：Pin编号/名称、Net与正式端到端系统；没有新训练、付费API、封存test40评测、Small恢复；没有修改non404/人工标签/GT/权重。原模型历史暴露限制不变。原生源自动标注尚不能直接当黄金训练集，需要核对渲染坐标与图像协议。

瓶颈：当前颜色规则对同色/黑白连续线无可见端点语义，bbox seed 也不稳；几何参数补全没有净收益；图片与公开标签的编号协议需澄清。不要无限扫阈值、按GT重编号或为涨分硬填null。

下一步若继续：先用少量可核验原生源验证 body/stub/tip 与文本坐标，再评估独立 pin-tip 检测/结构模板；文字侧需要可靠监督或识别器对照。主办方问题草稿在 `ORGANIZER_QUESTIONS.md`，尚未发送。当前没有继续运行的本轮后台作业；旧pcb-3自动化保持PAUSED。

唯一写入者为root；只读审计子代理未修改文件。报告保留可复现命令、逐图输出与源码哈希。
