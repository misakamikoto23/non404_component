# 低消耗智能体辅助标注

本次实际使用 GPT-5.6 Sol 辅助智能体，不需要配置外部 API 密钥。结果位于 data/ai_annotations_v1，和训练目录分开。

## 已生成的内容

- original_seeded/：293 张去重原图及对应 X-AnyLabeling JSON，来自 PCBnet 标准标签的转换；不是293张全部重新由智能体标注。
- reviewed/：12 张图片及候选 JSON。前6张做全图复核，另外6张重点检查 ANT 类。
- reviewed_overlays/：修改后的叠框，review_overlays/ 是修改前的叠框。
- report.json：逐图复核范围、来源、改动理由和不确定标记。

累计11处类别候选修正：9个清晰的运放/放大器三角形由 ANT 改为 AMP，2个开关由 Other 改为 Switch。没有新增、删除或移动框。
本轮训练仍只使用标准原始标签；上述修正没有加入训练、验证、测试。

## 如何查看和继续标注

用 X-AnyLabeling 打开 reviewed/，每张图片都有同名JSON。框的 attributes 保留 component_id、原类别、修改原因、复核范围。
全图复核只覆盖前6张，后6张没有穷尽核查其余类别和漏标；其余281张没有智能体视觉复核。
候选标签用于下一轮数据质量清理，不应当成无误的自动真值。补新数据时应同时检查漏标、类别与框范围，再按原理图分组划分数据。

生成原始种子标签的工具是 scripts/ai_annotation_review.py；实际视觉复核由智能体完成，不能将格式转换脚本的重复执行称为新的视觉复核。
