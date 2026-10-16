# 测试与评测

本目录包含两类内容：**离线单元测试**（不花 token）和**在线场景评测**（真实调用云端 LLM）。

## 文件清单

| 文件 | 说明 |
|------|------|
| test_retrieval.py | RAG 检索离线单元测试：索引可用性、同主题命中、噪声过滤、阈值过滤逻辑 |
| test_cases.json | 15 组场景输入（品类 × 场景类型：知识命中/价格疑问/合规风险/知识降级/观众边界/无关提问） |
| evaluate_cases.py | 在线评测脚本：跑完 15 组用例，统计对话连贯性与复盘要点覆盖度 |
| results.json | 评测原始数据（逐轮弹幕与指标，可复核），由评测脚本生成 |
| evaluation_results.md | 评测汇总报告，由评测脚本生成 |

## 一、离线单元测试（快速，不调 LLM）

```bash
pytest tests/test_retrieval.py -v
```

前置条件：已执行 `python main.py build`（FAISS 索引存在）。

## 二、在线场景评测（真实调用云端 LLM）

```bash
python tests/evaluate_cases.py
```

前置条件：
1. `.env` 已配置 `DEEPSEEK_API_KEY`（**会产生真实 token 消耗**，15 组约 45 次 LLM 调用）
2. 已执行 `python main.py build`

预计运行时间：几分钟到十几分钟（取决于模型响应速度）。
结果写入 `tests/evaluation_results.md`（汇总报告）和 `tests/results.json`（逐轮原始数据）。

## 指标口径

**对话连贯性**（每轮三项取平均，每组多轮再取平均，满分 1）：
- `format_ok`：弹幕通过安全校验（2-4 条、格式"观众名：内容"、不越界代写话术）
- `continuation`：主题承接度——主播发言与观众弹幕的词面重合度（0~1）
- `no_repeat`：本轮弹幕未在前几轮出现过（不重复提问）

**复盘要点覆盖度**（每组三项取平均，满分 1）：
- `dims_present`：勾选的评分维度全部出现在复盘报告
- `findings_ok`：各维度 finding 非空且非占位（占位=模型未返回，系统补的假分数）
- `topic_related`：复盘内容命中用例主题关键词（如价格类用例的复盘应提到"价格/工时"）
