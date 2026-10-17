# 二值化YOLO26n独立小样 — 2026-10-17

用户要求仅3050 Ti、不影响现有YOLO。当前仅一块RTX3050Ti(4GB)，本次逻辑cuda:0；按GPU名称验证，未使用4060Ti。两组从通用yolo26n.pt各自初始化，所有新权重位于component_lab/outputs/binary_yolo_3050_20261017/runs/，现有13份weights/*.pt的SHA256完全不变，默认权重未替换。

官方train140/dev20设计分组，6417/931个单类框，两种视图标签逐字节一致；不打开test40图片或标签。1280、batch1、nbs16、AdamW lr0=.001、AMP关闭、seed42、各8epoch，二值源为RGB转灰度再Otsu、复制三通道。OCR/ResNet仍应使用原图。

统一batch1、同20dev重评：RGB/RGB的P60.59%、R40.13%、mAP50 42.80%、mAP50–95 23.44%；binary/binary的P59.18%、R38.24%、mAP50 40.87%、mAP50–95 22.25%。二值源mAP50低1.93个百分点。RGB模型直接吃灰度mAP50 26.64%、直接吃二值12.64%；binary模型吃RGB为22.11%。说明预处理分布必须保持一致，本次未显示纯Otsu优势。

从通用权重短训且数据/框协议不同，不能与原PCBnet30图86.20%mAP50比较，也不能称严格ComponentF1。仅一个seed、8epoch，不能证明所有二值化或更长训练都无效；未测试自适应阈值、灰度训练或混合增强。本轮停止追加训练，主模型继续保留。

训练耗时RGB约280秒、binary约220秒，峰值CUDA分配两组约1287MiB、预留约1586MiB；没有OOM。耗时差只是单次运行记录，不能单独归因于二值化计算优势。显存probe实际训练1轮，历史额外最终验证回调不计为第2轮，正式8轮回调已修正。

脚本component_lab/scripts/pilot_binary_yolo.py；完整配置、源标注清单、权重保护哈希、日志、曲线、跨视图指标、报告图库均在本实验目录。report子命令免费重放报告；不要重跑prepare或train覆盖原实验。无API、人工标签修订、Small恢复、封存测试评估。
