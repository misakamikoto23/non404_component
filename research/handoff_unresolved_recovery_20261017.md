# Component未命名恢复交接 — 2026-10-17

本轮按用户授权自主开发。主代理唯一写入，两名协作代理只读审计；`non404/`未修改。没有新训练、标签改动、Small恢复、封存test40使用或新的后台任务。

新增纯预测gate与安全写回模块：`component_lab/src/component_lab/identity_recovery.py`、`recovery_gate.py`、`recovery_io.py`；`cli.py`增加离线`recover-identities`。默认只恢复key/type，原named记录及机器Name/value保留，主infer默认不变。

实验目录：`component_lab/outputs/unresolved_identity_recovery_20261017/`。固定dev20、原YOLO round004与ResNet balanced-v1预测缓存，基线17.42067%/170TP。所有新目标、规则、提示、payload哈希先冻结再读GT。

- 712个unresolved经保守物理类别/分数与文字证据gate得到108个候选（本实现T1=34、T2=74）；密集引脚行另排除1。25个历史direct源项不再付费读取。
- 新选24目标：T1/T2各12、共16图、跨case上限3；公开zip逐像素核验。Qwen6/6成功、11,832 token、请求时间25.051秒，返回model为qwen3.8-flash。
- 新key/type-only：177TP，新增7、损失0，严格逐图宏F1=17.768035%。13条成功rename；T1贡献6TP、T2贡献1TP。
- 新fields：174TP、17.660094%；历史缓存fields：173TP、17.843184%；缓存＋新fields：177TP、18.082609%。这是不同shadow，不按答案择优合并。
- 328个旧named元件全部逐字段/哈希不变，预测数量1040不变、pins/nets不变。134项相关回归通过，真实离线CLI20图与new_keys预测一致。

new_keys13个命名结果：7完整TP；1无同名GT（0130/LED7）；2仅bbox失败（0058/D1约113px、0144/R5约544px）；1仅text（0200/C51缺2.2nF）；1仅type（0130/U12 connector vs box）；1bbox+text（0108/C14约122px、缺10uF）。

主代理已实际查看eval-only对照图：0144返回R5，预测处可见R5，GT R5位置却可见R2；0108返回C14，预测处可见C14，GT C14位置却可见C9。不能按这些GT改写预测编号。0058图是D1/BAV99双二极管结构，模型框住左侧子符号、同名GT小框在右侧子符号；复合符号身份/拆分约定需核实，不能归结为统一内缩即可解决。

相关证据见`eval_only_failures/gallery.html`与audit.json，只供评价，不能用于本轮API输入或训练。U12只有单例类型冲突，未据此改全局前缀/分类映射。

可复现：`python scripts/pilot_unresolved_recovery.py report`免费重评分；`python scripts/audit_unresolved_recovery.py`免费生成审计；安全CLI详见`component_lab/RECOVERY_WORKFLOW.md`。API `run`已完成，不为重放重复付费。

共享账本延续既有累计，只追加6请求容量，token总容量未增加；本轮不再追加API。Qwen调用一般授权及用户指定Host沿用此前会话，凭据只在内存读取，不记录到文档/输出。

剩余瓶颈：文字归属与真实符号实例边界，以及公开位号/字段协议冲突。接下来应先就上述可视证据向主办方确认；纯本地可研究复合符号与重复位号处理，但需固定规则与验证，不能按dev单例重编号。模型权重与所有真值保持原状。
