# 项目档案：Component 段（2026-10-18）

> 用途：交接给外部顾问（argon4）。**只描述 Component 这一段**，pins/nets 由同组其他成员负责。
> 数据来源：本仓库 `component_lab/` 的冻结实验产物与 `research/` 下的笔记；所有数字都可按文中路径复现。
> 硬约束：10gt / Golden 系列（`10gt.zip`、`10GT-0930.zip`、`open_GT.zip`、公开包里的 `10_GTcase`）**只作测试，不得用于训练**。

---

## 1. 任务与边界

- 赛题：单张原理图 PNG → 官方 JSON（`components` / `pins` / `nets`）。
- 权重：Component 0.30、Pin 0.25、Connectivity 0.35、Wire 0.10。
- **本项目只做 `components` 这一段**：检测框、type、位号（key）、Name、value、bbox；`pins`/`nets` 输出为空，由队友拼接。
- 评测形态（官方须知）：Docker 镜像（.tar 或 Dockerfile），Ubuntu 24.04，A800-80GB ×8，网络可用；总时限 **20 分钟**，SIGTERM + 30s 宽限，**已产出的部分结果仍计分**；输出要求 `/output/final_results/<case>_Pred.json` + `/output/decision_log.json`。

---

## 2. 当前流水线

```
PNG
 ├─ YOLO26-Nano（单类 component 框；imgsz 1280；conf 0.25；大图 1024 切块 / overlap 0.25 / ≤100 块）
 │     权重 weights/pcbnet_component_yolo26n_review_round_004.pt
 ├─ RapidOCR（PP-OCRv5 mobile det+rec+cls，ONNX CPU；全页 + 1200px 切片；2× 放大；text_score 0.35）
 ├─ 符号裁剪 → ResNet18（27 类，224 letterbox，阈值 0.8 以下弃权保留 component）
 │     权重 weights/resnet18_component_balanced_v1.pt
 └─ identity-v1 关联（距离 ≤8 字高；未匹配代价 1.75；歧义间隙 0.10；匈牙利全局分配）
        → 位号 / Name / value（全部取 OCR 原文，不归一化）
        → 未命名对象命名 __unresolved_detection_NNNN
 → contest_components.json（左下角坐标，官方格式）
```

可选旁路：`--local-v6`（局部 OCR 复读）、`--ocr-spans`、`--display-names`、`recover-identities`（离线补身份）、Qwen 视觉裁决（实验性，未采用）。

---

## 3. 性能数字（**必须带口径**）

仓库里有三套不同框集的"严格 ComponentF1"，**不可横向比较**。

| 数字 | 框集 | 权重 | 口径 / 来源 |
|---|---:|---|---|
| **17.42% / 170TP** | 1040 | round_004 | 本地评分器；dev20 931 GT；逐图 macro；`outputs/advance_until_bottleneck_20261017` |
| 17.77% / 177TP | 1040 | round_004 | 同上 + `recover-identities` 补回 7 个 |
| 18.32% / 163TP | **632** | 37 类短训 | 整图检测器基线；`outputs/fullpage_component_identity_20261017` |
| **20.36% / 148TP** | 1040 | round_004 | **按官方字面口径**（所有类型都比 Name）；`research/e12_official_text_rule_20261018.md` |
| **46.79% macro / 55.44% micro**（假设） | 1040 | round_004 | 若 `gnd`/`v` **按坐标匹配**（官方 Q&A 方向）；433TP；`research/official_materials_review_20261018.md` |

> 46.79% 是**带假设的诊断**，不是成绩：官方评分代码未取得，bbox 与文本门控未确定。

其他参考：

- Oracle 上界（632 框、931 GT）：文字完美 **25.70%**；非 ∅ key 完美 **44.16%**；连匿名 ID 完美 **62.59%**。
- 提交视图（1040 框）：全提交 38.31% macro（假设 gnd/v 计分）；保留 gnd/v + 只提交已命名已分类 **46.79%**。

### 检测与分类

| 模型 | 指标 | 口径 |
|---|---|---|
| YOLO round_004 | P 88.83 / R 84.00 / mAP50 86.20 / mAP50-95 53.24 | **旧 PCBnet 30 图验证集**（有历史暴露） |
| YOLO round_005 | P 89.33 / R 85.44 / mAP50 87.25 / 52.48 | 同上；但严格 dev20 只有 17.32% → **未替换默认** |
| YOLO 干净划分短训（8 epoch，通用初始化） | mAP50 **42.80** | 官方 train140/dev20，**说明旧 86.20 不可与官方口径混用** |
| ResNet18 | 弱标签一致率 96.74% / 宏 F1 73.94% | 27 类，**弱标签/AI 银标**，非人工真值 |

### OCR 与数据覆盖（dev20）

- 525 个"有可观测位号"的 GT 元件中，OCR 原文命中 **305（58.1%）**；最终导出 key 命中 282（只再丢 7.5%）。
- 按来源：kicad 66.0% / jlc 64.3% / altium 58.5% / **datasheet 45.3%** / other 7.7%（编号协议冲突）。
- 220 个漏读位号：ROI 完美选区 + 4× 重读只能救回 **34（15.2%）**；易混字符修复只多 **6** 个；更细切片/更大放大**反而更差**。

---

## 4. 已验证的负结果（请勿重复踩）

| 尝试 | 结果 |
|---|---|
| ROI 放大重读（local v6 路线） | 上限 **15.2%**；20 图实测 +0.60pt |
| 易混字符修复（l→1、O→0…） | 只多救 6 个 |
| 更细切片 + 3× 放大 | 覆盖率下降（0200：77.1%→65.7%），耗时翻倍 |
| Qwen 替代匈牙利分配 | **−6.70pt**（同池配对），20 次请求 / 185,547 token |
| 整图 Qwen 文字归属 | 无稳定宏分增益（19.0 万 token） |
| "删掉未解析预测提分" | 只在本地宽松评分器成立；**官方 gnd/v 按坐标匹配，会误删 256 个** |
| OCR v6 全量替换 | 仅净增 **1 个**完整正确元件（codex 复核），暂不替换 |
| null 弃权语法规则 | 改动 0 个字段（图上的值都是正常容值） |

---

## 5. 当前瓶颈（按影响排序）

1. **评分口径不确定（最重要，非模型问题）**
   - bbox 容差：Q&A 说 **20px**，平台说明出现 **5px**；还有"mIoU"字样未被证实。
   - 我们的 gnd/v 匹配对阈值极敏感：256 个匹配里 **只有 41 个（16%）落在 5px 内**，误差中位数 **11.6px**（GT 符号短边中位 19px），且偏移只有 1–3px，**不能靠常数补偿**。
   - `Name` 语义：GT 实测非 box 类型 `Name==key` 99–100%（8435/9171），box 只有 24.2%；Q&A 第 10 条"name 是芯片名字"针对 IC。导出已按此修好。
2. **OCR 覆盖率 58.1%**：检测侧已是瓶颈，不是识别侧（220 漏读里 90% 附近有文本，但完美 ROI 也只能救 15%）。
3. **检测召回**：632 框 vs 931 GT；缩放后短边 4–8px 的元件召回仅 **44.9%**（加切图 66.3%）。
4. **IC 型号（box 的 Name）**：严格口径下仍有约 22 个 TP 因型号缺失/错误丢失。
5. **type 词表缺口**：`power_port` 应映射 `v`；`connector` 不在官方词表（Q&A：排针/USB 座属 `box`）；`net_port` 不能硬猜四种 net 类别。官方词表存在 41 类（0917 版）与 34 类（0930 版）两版，以官方最新为准。

---

## 6. 工程现状

- **代码**：`component_lab/src` 30 模块、`scripts` 70+ 脚本、`tests` 33 个测试模块。
- **测试**：`python -m unittest discover -s tests` → **281 项，276 通过，5 项因缺 cv2/ultralytics 失败**（环境问题，非代码缺陷）。
- **仓库**：已 git 化，`main` 跟踪 GitHub；入库 238 文件（源码/文档/测试），`outputs/`、`weights/`、`data/` 不入库。
- **环境风险**：项目内三个 venv **都没有 torch**；训练日志显示用的是外部解释器（`torch 2.5.1+cu118`、Python 3.12.4、RTX 3050 Ti）。**干净 clone 无法复现训练**。
- **容器**：尚未产出 Docker 镜像；输出契约（`final_results/<case>_Pred.json`、`decision_log.json`、token 统计）未实现。

---

## 7. 需要顾问判断的问题

1. 在**无法拿到官方评分代码**的前提下，如何设计一个既能覆盖"gnd/v 坐标匹配 + 20px"又能兼容"5px/mIoU"两种规则的稳健方案？
2. 46.79% 的假设是否值得作为优化目标？还是应该先把**提交适配层**（`power_port→v`、保留 gnd/v、type 词表对齐）做扎实，等官方脚本再验证？
3. OCR 覆盖率（58.1%）与检测召回（4–8px 桶 44.9%）哪一个是更高杠杆？建议的投入顺序？
4. 20 分钟 / 10~N 个 case 的预算下，本地 v6 补读（单 ROI ~14.6s）是否应该进评测版？

---

## 8. 关键文件索引

| 内容 | 路径 |
|---|---|
| 现状一页纸 | `research/status_20261018.md` |
| 官方资料核对（codex） | `research/official_materials_review_20261018.md` |
| GND/VCC 阈值敏感性 | `research/gndv_match_sensitivity_20261018.md` |
| Name 口径（E12） | `research/e12_official_text_rule_20261018.md` |
| OCR 调研与实验 | `research/ocr_improvement_20261018.md` |
| Qwen vs 匈牙利（E0） | `research/e0_qwen_vs_hungarian_20261018.md` |
| 夜间交接 | `research/handoff_20261018.md` |
| 评测规则复刻 | `component_lab/EVAL_CONTEST6.md` |
| 默认超参 | `component_lab/configs/defaults.json` |
