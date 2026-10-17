# 局部 PP-OCRv6 补读

已接入 `component-lab infer` 的 `identity-v1` 分支。原有 YOLO、ResNet 和主 OCR 继续工作，完成关联后，v6 仅尝试补未解决的内容。

```text
YOLO → ResNet → 主 OCR（v5 缓存 / Paddle）→ identity-v1 关联
  → 挑选未命名元件、R/C 缺 value、IC 缺 Name
  → 原图 RGB 局部裁剪 → v6 medium（独立 CPU 环境）
  → 空字段及原 ROI 所有权校验 → 同步内部 evidence → 比赛 JSON
```

## 使用

在 `component_lab` 目录执行，替换图片和主 OCR 缓存路径：

```powershell
.\.venv-yolo\Scripts\python.exe -m component_lab.cli infer `
  --image "你的原理图.png" `
  --yolo weights/pcbnet_component_yolo26n_review_round_004.pt `
  --resnet weights/resnet18_component_balanced_v1.pt --device 0 `
  --ocr-json "主OCR缓存.json" `
  --association identity-v1 --ocr-spans --display-names --contest-export `
  --output outputs/my_component_result
```

配置 `configs/defaults.json` 中 `local_v6.enabled=true`，上面命令默认补读。`--ocr-json` 接受包含 `texts` / `ocr_texts` 的 JSON 或文本列表，bbox 必须为原图左上角坐标。也可使用原来的 `--ocr paddle`，需要当前主 Python 已安装 Paddle OCR。没有主 OCR 输入，或仍使用 legacy 关联时，会明确记录跳过原因，不启动 v6。检测/人工复核的旧拖拽工具未增加整页 OCR。

- `--no-local-v6`：完全关闭补读。
- `--local-v6-mode unresolved-only`：锁定所有原已命名完整记录，只尝试未命名元件。
- 默认 `--local-v6-mode empty-fields`：还允许原已命名元件补必要的空参数/型号。
- `--local-v6-python 路径`：指定 OCR Python；默认项目 `.venv-ocr/Scripts/python.exe`。
- `--local-v6-timeout 120`：每图 worker 总超时秒数。
- `--local-v6-max-rois 2`、`--local-v6-area-fraction 0.2`：每图裁剪数量/面积上限。

默认每图最多 2 个 ROI、累计不超过原图面积 20%，单区最长边 320px；优先分别给未命名和已命名缺字段目标一次机会。使用局部频域与类字符连通域辅助挑区，没有适合特征时回退到几何邻域。识别输入是原图 RGB 裁剪后放大 2 倍，没有去线、二值化或旋转文字补读。预算可能使本轮无区可选，不等于所有字段已解决。

## 保护与限制

已有非空 `Name/value`、已命名 key、bbox/type 不覆盖。新增 token 带 `source_tier=fallback_v6`、原 ROI `owner_ids`；不能靠新 OCR ID 抢走邻居已绑定的文字。主位号 anchor 不可迁移，同位置重复 anchor 先去重，新增 key 冲突同时拒绝。原来就重复的 primary 位号保留待定，不自动恢复其中一项并间接重命名另一项。

**默认允许补 null 并非绝对安全。** 公开 GT 有图上印了数值但标签为 null 的例子，补上看得见的数值也可能降低严格分数。需要更保守时使用 `unresolved-only` 或关闭开关。上一轮固定五图只增加 1 个完整正确项，不能承诺全量收益。

结果仍只包含 component；pins/nets 不由本模块生成。所有 OCR 是机器建议，未写回人工标注或任何训练/验证真值。

## 本机运行与产物

补读使用 `.venv-ocr` 中 RapidOCR 3.10 / ONNX Runtime，CPU、2 线程，单进程顺序处理所选 ROI，不占 GPU。模型必须已在 `weights/ocr_rapid/PP-OCRv6_det_medium.onnx` 和 `PP-OCRv6_rec_medium.onnx`；缺模型、依赖失败或超时保留主结果。不调用 API，不自动下载，不自动重试失败作业。

```text
components.json                 # 内部框、字段、OCR 原文和来源
contest_components.json         # 可选；与同一份内部 evidence 统一导出
overlay.png
local_v6/
  baseline_contest.json          # 补读前固定结果
  summary.json                  # 模式、区域、耗时、状态和改动数
  evidence.json / audit.json    # 候选、归属、阻止原因、每个改动字段
  cache/<request_hash>/         # 冻结请求、原裁图、响应、日志和 attempt
```

同输出目录、相同输入/ROI/模型/worker/runtime 元数据复用 OCR 缓存；修改规则后重新合并，不改旧响应。默认 Windows venv 的 RapidOCR/ONNX Runtime METADATA 被纳入指纹，自定义解释器找不到这些元数据时记录 `runtime_metadata_bound=false`，更换依赖后应使用新输出目录。已开始但未完整返回的请求不自动重跑；完成部分可用，失败部分保留基线，并报告 `partial`。

验证与命令见 [接入报告](outputs/local_v6_integration_20261018/REPORT.md)。不重新训练 YOLO/ResNet，不恢复 Small，不读取 test40。
