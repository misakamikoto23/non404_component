# Gemini Component 方案复核与只读审计 — 2026-10-17

范围仅 Component。固定 dev20 冻结 `lexical_experiment/identity_v1`，没有新模型前向、付费API、训练、标签/预测修改或封存test使用。

## 已执行的两张归因表

完整数据：`component_lab/outputs/component_blocker_audit_20261017_v2/audit.json`；图文报告：同目录 `REPORT.md`。脚本：`component_lab/scripts/audit_component_blockers.py`。

### 36个有原文未导出

|阻塞阶段|GT项数|
|---|---:|
|owner歧义弃权|22|
|全局reference槽位竞争落败|6|
|未被接受为reference角色|6|
|重复reference不能唯一导出|2|
|纯距离截断|0|

角色过滤的6项为0107/USB1、0107/SG90、0108/U2.2、0108/U2.3、0182/USB1、0200/USB1。不能因此把任意型号字符串都提升为位号；多单元编号和USB前缀需要通用且带几何上下文的角色规则。

距离按照现有算法的bbox间距/文字窄边≤8归因，而不是固定100px中心距离。20图reference阶段重放与冻结结果完全一致，重建代价与已分配日志一致。逐token代价、候选和后续assignment保存；同key多个原文候选按最深可行路径归类。这是当前算法阻塞阶段，不必然等于根本视觉原因。

### 112个key命中但未得分

|错误组合|数量|
|---|---:|
|仅text|57|
|仅bbox|24|
|仅type|10|
|bbox+text|12|
|type+bbox|6|
|type+text|2|
|type+bbox+text|1|
|合计|112|

text涉及72项，bbox涉及43项，type涉及19项；这些交叉计数不能相加当独立样本数。key字符串匹配并不保证owner正确，112项仍可能包含绑错邻居或公开编号差异。

## 对建议的修正

1. 36+4只是诊断池，不是40个保证可恢复样本。大小写pool已经发现一个反例：0168/u12的casefold命中U12位于另一处，OCR框到GT主体最短间距约414px。字符串近似不能证明读到了目标位号。
2. `parse_text_role`确实为语义候选生成大写reference，但raw保留；contest exporter读取`association_evidence.raw`作为key，因此不能从发现`.upper()`推断最终字符串被改写。
3. 已看原图：0001/u2处显示大写U2，OCR也为U2，而GT是u2，属于疑似标签大小写差异；0001/C20处显示C20，OCR为c20，属于识别大小写问题。0168/u125图小且模糊，不强行判定大小写；0168/u12属于远处字符串命中，不能当单纯大小写恢复。
4. GT bbox外扩60px可筛选待查文字，但附近任意未占用token不是目标位号真值。附近的value/型号/其他元件位号都可误导；已被认领的token也必须纳入诊断，才能发现抢占。只有图像级位置核验才可确认检测漏框或识别错字，缺证据要保留unknown。这是GT辅助诊断，不是无监督方法或部署时的触发规则。
5. fallback只填unresolved有助于限制改动，但不能保证F1不降：新增错误key增加FP；重复key可能使旧正确key在导出时失效；如果重跑全局value/Name匹配，也可能破坏旧字段。应锁定primary导出的key和字段，拒绝冲突key、保留token/span ownership，并分别报告新增TP/FP、旧TP丢失及最终F1。
6. 主办方清单可用，但未经源文件核对的字段归因仍需写成疑问；bbox还应询问四坐标最大误差/中心距离的具体公式，不能只问body还是完整符号。

## 当前最大的Component问题

最大的已确认工程短板是文字身份链路：将正确位号及必要的value/Name可靠地绑定到正确检测框并精确导出。197个无精确原文项的OCR/协议成分尚未完全分清；已存在原文的36项有28项卡在归属/竞争，已命中key的112项有57项仅文字失败。现有证据支持先投入文字识别与联合归属，不支持先靠加YOLO epoch或扩大ResNet解决主要问题。

下一步先研究28项归属/竞争与57项纯文字失败的可共用结构证据；角色规则扩展作为有上下文的小对照；针对确认的旋转/小字漏读做独立fallback实验。此文不承诺潜在样本全部恢复或某个目标分数。
