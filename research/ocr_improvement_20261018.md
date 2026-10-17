# OCR 提升：本地证据 + 前人工作调研

日期：2026-10-18。全程 0 次付费 API、0 训练、未改权重/标签/GT/test40。
配套实验目录：`component_lab/outputs/unattended_20261018_e3|e6|e7|e8|e9|e10_*`。

## 一、结论速览

1. **全页 OCR 只能读出 58% 的可见位号**（305/525），且这部分**归属几乎不是瓶颈**（命中 key 只再丢 7.5%）。
2. **完美 ROI 放大重读只能再救 16%**（E8：34/224；E9/E10 口径下 31~37）。**ROI 补读不是解药。**
3. 易混字符修复（l→1、O→0…）只多救 **6 个**（E10）。**字符级后处理也不是解药。**
4. 所以剩下 ~83% 的漏读，**不是"没放大"或"读错一个字符"**，而是这几类字的识别本身失败（太小、抗锯齿、被线条压住）或图上根本没有对应文字。
5. 下一步应该投在**整页文字检测的分辨率/模型档位**上，而不是 ROI 补读。

## 二、证据链

| 实验 | 结论 | 数字 |
|---|---|---|
| E3 | 覆盖是主瓶颈，归属不是 | 525 可见 key → 305 原文命中（58.1%）→ 282 最终命中（只再丢 7.5%） |
| E3 | 来源差异大 | kicad 66.0% / jlc 64.3% / altium 58.5% / **datasheet 45.3%** / other 7.7%（编号协议冲突） |
| E6 | 漏读附近大多"有文本" | 220 个漏读里 199 个附近 2 字高内有 OCR 文本 |
| E7 | 但"像"的只有一半 | 半径 2 字高：contained 11 / near_miss 69 / absent 140；放到 5 字高变成 8/100/112 |
| **E8** | **完美 ROI + 4× 只救回 15.2%** | 224 个漏读，2× 救 22、4× 救 31、任一 34；`<20px` 救回 28/131，`≥40px` **0/15** |
| E9 | ROI 内几乎总有字 | no_text 仅 4；但多数是与目标无关的字（引脚名、网络名、邻居位号） |
| **E10** | **易混字符修复几乎无效** | 精确 31、修复后 +6、仍漏 187 |

### 必须纠正的一个测量假象

E9 里"155 个 near_text（编辑距离 ≤2）"看起来像"只差一两个字符"，但看样例就会发现这是**短位的编辑距离假象**：

- `0001/u2` 的 ROI 文本是 `['5','Raspberry','PiF','RM2']` —— `RM2` 与 `U2` 距离 2，于是被判"近似"，其实是**完全不同的网络标签**。
- `0001/C8` 的 ROI 文本是 `['3.3V','□','CB','C11']` —— `CB` 与 `C8` 距离 1。

**结论：对 2–3 个字符的位号，编辑距离 ≤2 几乎必然误判。** 所以"69% 只差一两个字符"这个说法不成立；E10 用真正可操作的修复规则验证后只多救 6 个，佐证了这一点。

### 一个反直觉的细节

E8 里 `≥40px` 的 15 个漏读，**4× 重读一个都没救回**。大符号的位号本该更好读，救不回来说明：这些 GT key 与图面文字**根本不是一个字符串**（编号协议问题），或位号印在离符号很远的地方。这与 E3 里 `other` 来源只有 7.7% 覆盖率互相印证。

## 三、前人工作（按可借鉴度排序）

> 以下为文献/官方资料的**方法线索**，除标注"已本地验证"外，均**未在本地跑过**，不要直接当成收益承诺。

### 1. 引擎与模型档位（最直接）

- **PP-OCRv6** 官方提供 tiny / small / medium 三档，官方基准显示更大档位在多项中文场景上优于 v5 系列（[PP-OCRv6 文档](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)、[PaddleOCR releases](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0)）。
- 我们已本地验证过一条：**v6 medium 的 OCR 池让严格 ComponentF1 提高 1.38pt**（E0 的 `hungarian_all_ocr` vs `primary`），这是"更大模型档位"的实证。
- **RapidOCR 与 PaddleOCR 跑的是同一批 PP-OCR 权重**（ONNX 移植），因此"换框架"本身不提升精度；差别在**能选到多大的模型档**，以及运行时速度。

### 2. 小目标检测：切片与尺度

- 工程图小目标检测里常用 **scale-aware slicing**（按目标尺度决定切片大小），与我们的 `--tile-size` 思路一致（[IEEE 10482360](https://ieeexplore.ieee.org/document/10482360)）。
- 对我们的启示：**当前整页 1200px 切片 + 2× 放大可能仍不够**。位号普遍 8–20px，缩放到 1200 窗口内不变，2× 后 16–40px 对 det 应该够——但 E8 显示放大也没救回，说明问题可能不在尺度而在**线条干扰/对比度**。

### 3. 文本超分（Text Super-Resolution）

- 有工作用 Swin Transformer 做文本图像超分以提升真实场景 OCR（[ACM DocEng 2025](https://dl.acm.org/doi/abs/10.1145/3704268.3742690)），也有更早的 SR+OCR 联合方案（[arXiv 1506.02211](https://ar5iv.labs.arxiv.org/html/1506.02211)）。
- 但要注意：**超分对"本来就 8px、信息已丢失"的字，收益有物理上限**。E8 的 4× 双三次插值只有 15% 救回率，专用超分模型可能更好，但不太可能翻盘。

### 4. 原理图专用流程

- **SINA**（schematic image → netlist）与 **PCBnet** 都采用「YOLO 检测文本区域 + OCR 识别」的两段式，而不是整图 OCR（[SINA](https://ar5iv.labs.arxiv.org/html/2607.01609)、[PCBnet](https://github.com/ECNUACRush/PCBnet)）。
- 这提示一个**结构性差异**：他们**先检测文字框**（专为原理图训练），再做识别；我们是通用 OCR 的 det 直接上，**det 阶段就可能漏掉小字**。
- 这与 E10 的结论一致：**问题在"这些字没有被检出"，不在"检出后读错"。**

### 5. 专用识别器微调 + 合成数据

- 原理图的字是等宽、单调色、无抗锯齿，**合成数据微调的小 rec 模型**通常优于通用大模型；PCBnet 论文也用了文本区域检测 + PaddleOCR 的组合。
- 成本最高（需训练），且**当前环境没有 torch**，短期不可行。

## 四、按性价比排序的建议

| 优先级 | 方案 | 依据 | 成本 | 预期 |
|---|---|---|---|---|
| **P0** | **整页文字检测升级**：切片更小（600–800px）+ 3× 放大 + `text_score` 降到 0.2，整页重跑 | E10 证明不是"读错"而是"没检出" | 纯 CPU，几十分钟/20 页 | 未知，**建议先做 3 页探针** |
| **P1** | **换 det 模型档位**（v5 server det / v6 det） | 更大 det 对小字更敏感 | 需转 ONNX，CPU 变慢 | 同上 |
| **P2** | **rec 升级**（det 保持 v5，rec 换 v6 medium） | E0 已证 v6 池有 +1.38pt | 一行配置 | 已验证方向 |
| **P3** | 文本超分预处理 | 文献支持，但物理上限低 | 需额外模型 | 低 |
| **P4** | 专用 rec 微调 | 文献与数据都支持 | 需 torch + 训练 | 高，但当前不可行 |
| ❌ | ROI 补读（现有 local v6 路线） | E8/E10 上限 16% | 14.6s/ROI | 不划算，建议降级为可选 |
| ❌ | 易混字符修复 | E10 只多 6 个 | 零成本 | 可作为兜底，但别指望 |

## 五、下一步的验证设计（尚未执行）

**P0 探针**：取 3 张图（1 张 datasheet、1 张 jlc、1 张 kicad），用三组配置整页重跑 OCR：

| 配置 | 切片 | 放大 | text_score |
|---|---|---:|---:|
| 现状 | 1200 | 2× | 0.35 |
| A | 800 | 3× | 0.20 |
| B | 600 | 3× | 0.20 |

度量：**220 个漏读里能被读出的数量**、每页耗时。若 A/B 能多读出 30% 以上，才值得全量重跑。

## 六、复现

```powershell
cd component_lab
python scripts/unattended_e3_ocr_by_source.py
python scripts/unattended_e6_ocr_miss_taxonomy.py
python scripts/unattended_e7_ocr_cause_split.py
.\.venv-ocr\Scripts\python.exe scripts/unattended_e8_roi_reread_probe.py
.\.venv-ocr\Scripts\python.exe scripts/unattended_e9_roi_text_presence.py
.\.venv-ocr\Scripts\python.exe scripts/unattended_e10_confusable_repair.py
```
