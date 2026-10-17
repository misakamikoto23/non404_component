# E0 结案：Qwen 替代匈牙利分配 **失败 −6.70pt**

日期：2026-10-18 06:20 JST。用户授权调用 Qwen3.8-flash 后，补跑了 codex 未执行的三臂。
固定 dev20、同一批检测框与同一 OCR 池，**只切换分配方式**；预测先冻结再读 GT。

## 结果（20 图，严格 ComponentF1，逐图宏平均）

| arm | 说明 | 全量提交 | **不提交未解析** | TP | 提交条目 |
|---|---|---:|---:|---:|---:|
| primary | 现网规则（只补空） | 17.50% | 20.49% | 143 | 265 |
| local_v6 | + 局部 v6 补读 | 18.09% | 21.09% | 147 | 271 |
| qwen_fill | Qwen 选择只补空 | 18.64% | 21.08% | 151 | 315 |
| **hungarian_all_ocr** | **全 v6 OCR 池 + 匈牙利** | 18.91% | **21.87%** | **150** | 280 |
| qwen_replace | 全池交给 Qwen 分配（替代匈牙利） | 13.53% | 15.17% | 76 | 181 |

## 三个关键差值（同图配对）

| 对比 | 差值 | 结论 |
|---|---:|---|
| **Qwen 替代匈牙利**（qwen_replace vs hungarian_all_ocr） | **−6.70pt** | ❌ **明确失败，结案** |
| v6 OCR 全池 + 匈牙利（vs primary） | **+1.38pt** | ✅ 增益来自**OCR 池变大** |
| Qwen 保守补空（qwen_fill vs primary） | +0.59pt | ⚖️ 太小，不值 API 成本 |
| 局部 v6 补读（vs primary） | +0.60pt | ⚖️ 小额真实 |

**核心结论：这套流水线的瓶颈是"OCR 池里有没有那段字"，不是"谁来分配"。** 把池子扩大（v6 全池）能涨 1.38pt；把分配器换成 27B 视觉模型反而掉 6.70pt——因为 Qwen 会把已有正确绑定改错，且 5/20 页返回格式不合规直接回退。

## 用量

- 20 次请求：**15 ok / 5 invalid_response**（格式不合规，按预声明回退）
- 185,547 token，HTTP 合计 104.2 秒
- 共享账本：78/78 请求名额用尽；承诺 453,447 / 667,900 token
- 无付费重试

## 对 experiment 脚本的两处修复（否则根本跑不完）

`scripts/dev20_qwen_assignment.py` 此前**从未成功执行过**，有两个 bug：

1. `accounted_usage()` 返回三元组，脚本直接当成整数用 → `TypeError: tuple - int`；
2. 崩溃后写进 `qwen_usage.json` 的 `accounted_tokens` 是列表，导致 `sum()` 再次崩溃。

修复：解包三元组并记录 `accounted_basis`；把该条目归一化为整数。账本按实际 usage（14,955）而非预留（19,310）修正。

## 限制

- 本实验框集 537 条（20 图），与 E1/E5a 的 1040 框**不是同一套**，分数不可横比。
- 仍是 dev20 开发集；权重对划分有历史暴露。
- `qwen_replace` 的"不提交未解析"会丢弃 `__unresolved_qwen_*` 占位键，已按同一导出策略计算。

## 复现

```powershell
cd component_lab
python scripts/dev20_qwen_assignment.py prepare
python scripts/dev20_qwen_assignment.py run      # 20 次 Qwen 调用
python scripts/dev20_qwen_assignment.py freeze
python scripts/unattended_e0_interpret.py        # 五臂判读
```
