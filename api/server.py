"""
FastAPI 服务层 - 提供 RESTful API 接口

本模块是项目的对外接口层，提供四种服务模式：
1. /chat：        直播对话接口（主播发言 → 观众弹幕，支持流式输出）
2. /review：      复盘评分接口（按使用者勾选的评分维度生成复盘报告）
3. /rag/search：  知识检索接口（仅检索知识片段，不生成回答）
4. /health：      健康检查接口（查看 LLM / Embedding / 向量库实际状态）

流式协议说明（/chat 传 stream=true 时）：
    返回 NDJSON，每行一个事件：
    - {"type": "logs",   "items": [...]}     RAG/Prompt/LLM 操作日志（右侧面板）
    - {"type": "answer", "content": "..."}    回答文本片段
    - {"type": "error",  "message": "..."}    错误提示
    日志事件在回答之前和之后各发一次，保证检索过程和 LLM 完成状态都能看到。

启动方式：
    python main.py server
    或直接运行本文件
"""
import json

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional

from config.settings import (
    API_HOST,
    API_PORT,
    DEFAULT_HISTORY_LEN,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    LLM_CONFIG,
)
from core.vector_store import vector_store_exists
from rag.retriever import (
    generate_audience_reply,
    stream_audience_reply,
    generate_review,
    search_relevant_knowledge,
)

# 创建 FastAPI 应用实例
app = FastAPI(title="非遗直播话术生成器API", version="1.0.0")

# 允许跨域访问（Streamlit 前端与后端端口不同）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 健康检查接口 ====================
@app.get("/health")
async def health():
    """
    健康检查：把实际生效的后端配置透出来

    排查"检索怎么不准""为什么报错"时先看这里：
    - LLM 是否配置了 API Key
    - Embedding 模型是否加载成功
    - FAISS 知识库是否已构建
    """
    return {
        "status": "ok",
        "llm_model": LLM_CONFIG["model"],
        "llm_configured": bool(LLM_CONFIG["api_key"]),
        "vector_store_ready": vector_store_exists(),
    }


# ==================== 直播对话接口 ====================
@app.post("/chat")
async def chat(
    query: str = Body(..., description="主播本轮发言"),
    category: str = Body("传统手工技艺", description="非遗品类"),
    history: List = Body([], description="历史对话 [{role, content}, ...]"),
    history_len: int = Body(DEFAULT_HISTORY_LEN, description="保留历史对话的轮数（1-10）"),
    temperature: float = Body(DEFAULT_TEMPERATURE, description="LLM采样温度（0.01-2.00）"),
    top_p: float = Body(DEFAULT_TOP_P, description="LLM核采样概率（0.01-1.00）"),
    max_tokens: int = Body(DEFAULT_MAX_TOKENS, description="LLM最大token数量（2556-4096）"),
    stream: bool = False,  # 查询参数（?stream=true），前端用 params 传递
):
    """
    直播对话接口（RAG + 云端 LLM）

    完整流程：
    主播发言 → FAISS 检索知识 → 组装观众角色 Prompt → LLM 生成观众弹幕

    支持：
    - 多轮对话（通过 history 参数传递历史）
    - 历史长度控制（history_len 限制窗口大小）
    - 流式输出（NDJSON 事件流，日志与回答分开发送）
    - 参数动态调整（temperature/top_p/max_tokens）

    返回：
        stream=true 时返回 NDJSON 事件流；
        stream=false 时返回 JSON：{"answer": "...", "logs": [...]}
    """
    # 每轮请求一个独立的日志列表，检索/Prompt/LLM 全过程都记录在内
    logs = []

    if stream:
        # 流式模式：逐事件返回（日志 → 回答片段 → 日志/错误）
        def generate_stream():
            for event in stream_audience_reply(
                query,
                category=category,
                history=history,
                history_len=history_len,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logs=logs,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson; charset=utf-8",
        )

    # 非流式模式：一次性生成并返回
    try:
        answer = generate_audience_reply(
            query,
            category=category,
            history=history,
            history_len=history_len,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logs=logs,
        )
    except Exception as exc:
        # LLM 未配置密钥 / 调用失败 / 安全校验未通过：返回明确错误提示
        raise HTTPException(status_code=400, detail=str(exc))
    return {"answer": answer, "logs": logs}


# ==================== 复盘评分接口 ====================
@app.post("/review")
async def review(
    category: str = Body("传统手工技艺", description="非遗品类"),
    history: List = Body([], description="完整聊天历史 [{role, content}, ...]"),
    dimensions: Optional[List[str]] = Body(None, description="勾选的评分维度key列表"),
    temperature: float = Body(DEFAULT_TEMPERATURE, description="LLM采样温度"),
    top_p: float = Body(DEFAULT_TOP_P, description="LLM核采样概率"),
    max_tokens: int = Body(DEFAULT_MAX_TOKENS, description="LLM最大token数量"),
):
    """
    复盘评分接口（RAG + 云端 LLM）

    流程：
    检索品类知识 → 组装复盘 Prompt（只评勾选维度）→ LLM 生成 JSON 报告

    评分维度（侧边栏勾选，影响复盘侧重）：
    亮点讲述 / 互动回应 / 疑问回应 / 合规风险 / 节奏把控 / 来源依据
    未勾选任何维度时使用默认三项（亮点讲述、互动回应、节奏把控）。
    LLM 不可用时返回占位报告（review_source=fallback）。

    返回：
        {"review": {...}, "logs": [...]}
    """
    logs = []
    report = generate_review(
        category,
        history=history,
        dimensions=dimensions,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logs=logs,
    )
    return {"review": report, "logs": logs}


# ==================== 知识检索接口 ====================
@app.post("/rag/search")
async def rag_search(
    query: str = Body(..., description="检索关键词"),
    top_k: int = Body(4, description="返回片段数量"),
):
    """
    知识检索接口（仅检索，不生成回答）

    适用场景：
    - 用户只想查看知识库里的原始资料
    - 调试检索效果

    流程：
    query → Embedding 向量化 → FAISS 相似度检索 → 返回 Top-K 片段

    返回：
        JSON 格式的检索结果，包含原文和元数据
    """
    try:
        docs = search_relevant_knowledge(query, k=top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,  # 知识片段原文
            "metadata": doc.metadata,     # 来源文件、标题、分类、段号等
        })
    return {"query": query, "results": results, "count": len(results)}


# 直接运行本文件时启动 uvicorn 服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
