# 午餐无人实验交接

已完成：dev20文字字段与bbox审计，保守OCR span候选、box显示Name候选及固定对照，TRAIN140模式的r/c公开框协议诊断，pin-tip下一轮契约，相关107项测试。

主要结果：原identity_v1严格ComponentF1=16.69%；span+display Name=17.39%，TP164→169，无原完整TP丢失。r/c公开模板导出诊断=19.43%，仅协议敏感性结果，未接入生产。

重要限制：若干GT普通位号与图片可见位号不一致，见0011/0132裁剪；公开r/c/d/l尺寸模板跨来源固定，不能当实际body几何。未改GT，不让模型迎合不可见编号。

未做：新训练、付费API、新pin-tip算法实测、完整net系统、封存test评测。Small仍暂停；原权重不变。原本16.69%身份流程默认不变，新候选需显式 `--ocr-spans --display-names`。

下一步：读component_lab/outputs/unattended_20261017_lunch/REPORT.md、STATUS.json、PLAN.md。所有命令和证据在对应子目录；根AGENTS部分状态截至10-16已过时，字段语义和评分仍以实际规范与新报告为准。

模型：component_lab/weights/pcbnet_component_yolo26n_review_round_004.pt，resnet18_component_balanced_v1.pt，未重训。当前不是Git仓库，子代理只读分析，主流程单一写入者。
