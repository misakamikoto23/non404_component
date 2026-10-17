# Component：多类YOLO与两阶段对照

用户授权尝试多类YOLO26n直出、单类YOLO26n＋ResNet18、多类YOLO＋低置信度ResNet复核。实验已结束，未替换业务模型，未修改non404。

证据：`component_lab/outputs/multiclass_vs_two_stage_20261017/REPORT.md`、`evaluation.json`、`gallery.html`。复现入口为 `component_lab/scripts/compare_component_architectures.py` 与 `evaluate_component_architectures.py`。

官方train140/6417框，dev20/931框；按设计组排除dev/test训练来源。两组YOLO同通用初始化、1280/batch1、seed42各16轮，固定last；新ResNet从ImageNet训练12轮，37个train类型各轮64次曝光。dev-only crystal不删，计入FN；无新增API或test40评测。额外分类器训练意味着总训练算力并不相同。

|方法|整图严格ComponentF1|1024切片严格ComponentF1|切片类型microF1@IoU50|切片类型macroF1(29类)|
|---|---:|---:|---:|---:|
|多类YOLO|17.50%|15.65%|62.58%|14.73%|
|单类YOLO＋ResNet|18.32%|17.13%|60.77%|26.36%|
|多类YOLO＋ResNet复核|17.70%|15.94%|62.95%|17.35%|

暂不支持删除ResNet。两阶段严格F1点估计略高、稀有类别覆盖较好，多类直出较快、总体类型检测microF1较高。严格F1的配对bootstrap差值区间跨0，短预算单seed不足以确证哪种架构在充分训练后更优。固定conf=.25并不代表两头置信度已经校准。

分类器在GT上下文裁块上accuracy94.84%、macroF1 73.85%；稀有类只有1～2个原始实例，重复均衡不增加独立设计样式。检测类型macro很低同时包含漏检与框IoU失败，不能误读为ResNet的纯分类macro只有26.36%。

更值得优先处理的是身份：切片两阶段有252个GT key命中，命中内type错17、bbox错21、必要文字错73（重叠统计）。整图632预测变成切片810预测，但严格TP仍163。新增框未补齐文字、匿名隐藏ID不可观测和全局关联竞争，会阻止检测召回收益转为ComponentF1。

三臂共用冻结OCR、raw→semantic关联适配和导出规则；禁用prefix覆盖，导出恢复原始细类。YOLO26 end2end会忽略库NMS参数，因此显式对三臂使用相同class-agnostic NMS=.5。所有预测冻结后才进入GT评分；预测只使用source case/partition metadata，不用目标crop标签。

业务Round004/005及旧27-family ResNet保留；本次新37类ResNet只在实验脚本配套外扩和类型适配下测试，未直接接入主infer。新旧数据来源、预算和历史曝光不同，不能把新表与旧PCBnet30图86%mAP直接排名。

后续优先级：继续优化位号OCR/归属及必要文字；多类YOLO保留作速度优化候选，不据本轮额外长训或搜索阈值。若今后接入新分类器，需同时保持上下文裁剪和raw-type适配一致。6项针对性模型无关检查通过，原业务权重与新数据hash未变。
