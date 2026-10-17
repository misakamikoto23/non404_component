# 外部 DeepSeek 协作

入口 `scripts/deepseek_worker.py`，2026-10-18 已实测接通，默认官方 `deepseek-flash`（文档当前版本 V4.1-Flash-1017），关闭思考。密钥用环境变量 `DEEPSEEK_API_KEY`，或本机被 gitignore 忽略的 `.env`。不要打印或外发配置。

```powershell
.\.venv-yolo\Scripts\python.exe scripts/deepseek_worker.py --task "只读审查指定代码，最多3项有证据的问题，附最小反例" --source src/component_lab/fullpage_identity.py --max-tokens 2500
```

普通整理/初审交给 DeepSeek；Codex 负责取舍、本地修改与测试。只发相关源码/摘要；不批量发送仓库、凭据或整个聊天。不让模型自动执行命令；所有返回建议都需验证。可用 `--thinking` 显式启用 low 思考，但思考同样消耗输出预算。

用户未设金额上限，仍按任务终止、不自动重试、不后台自我对话。相同请求自动用缓存；每次调用记 usage、耗时、价格估算，账本不覆盖。初版单进程使用。输出在 `outputs/deepseek_collaboration_20261018/`；详情见其中 REPORT.md。

接入实验：两次成功调用 8659 token，按用户截图空闲费率估算约0.020653元（非扣款凭证）。一次网络失败无usage。首轮低思考无最终答案，关闭思考后正常返回；审查发现未经验证不能直接采用。原业务模型/标签/评分器未改变。
