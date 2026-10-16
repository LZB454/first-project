"""RAG 运行日志工具。

log() 同时做两件事：
1. 打印到后端控制台（格式如 [RAG] xxxx），方便命令行调试
2. 追加到本次请求的日志列表，由 API 返回给前端右侧日志面板展示

stage 用于前端分组展示：
- "RAG"    检索过程（向量库加载、相似度检索、命中片段）
- "Prompt" Prompt 组装过程
- "LLM"    大模型调用过程
"""


def log(logs: list, stage: str, message: str):
    """打印一条日志，并保存到本次请求的日志列表。"""
    item = {"stage": stage, "message": message}
    logs.append(item)
    print(f"[{stage}] {message}", flush=True)
