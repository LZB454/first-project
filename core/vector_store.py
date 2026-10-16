"""
FAISS 向量数据库管理模块

使用 Facebook AI Similarity Search (FAISS) 作为向量存储后端：
- build_vector_store():  将文档块向量化并构建索引，持久化到磁盘
- load_vector_store():   从磁盘加载已有索引（避免重复 embedding 计算）
- vector_store_exists(): 检查索引是否已构建，用于启动前的校验

FAISS 优势：纯 CPU 运行，无需 GPU，适合中小规模（<100万条）检索。
索引以 index.faiss + index.pkl 两个文件保存在 data/faiss_index/ 下。
"""
import os

from langchain_community.vectorstores import FAISS

from config.settings import VECTOR_STORE_PATH
from core.embeddings import get_embeddings
from rag.trace import log


def _log(logs: list, message: str):
    """打印日志；logs 不为空时同时保存，供前端右侧日志面板展示。"""
    if logs is None:
        print(f"[RAG] {message}", flush=True)
    else:
        log(logs, "RAG", message)


def build_vector_store(chunks, save: bool = True) -> FAISS:
    """
    构建 FAISS 向量库并持久化

    流程：
    1. 使用本地 embedding 模型将所有文本块转为 512 维向量
    2. 构建 FAISS 索引（默认 L2 欧氏距离）
    3. 保存到本地磁盘（index.faiss + index.pkl）

    参数:
        chunks: LangChain Document 列表（已分割的文本块）
        save:   是否持久化到磁盘，默认 True

    返回:
        构建好的 FAISS 向量库实例
    """
    print(f"[KB] 文本块开始向量化 | 切片数={len(chunks)}", flush=True)
    embedding = get_embeddings()
    vs = FAISS.from_documents(chunks, embedding)  # 向量化 + 建索引
    if save:
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        vs.save_local(VECTOR_STORE_PATH)  # 保存为二进制文件
        print(f"[KB] 向量索引保存完成 | path={VECTOR_STORE_PATH}", flush=True)
    return vs


def load_vector_store(logs: list = None) -> FAISS:
    """
    从磁盘加载已有的 FAISS 向量库

    注意：allow_dangerous_deserialization=True 允许反序列化 pickle 文件，
    仅用于可信的本地文件，不要加载来源不明的向量库文件。

    异常:
        FileNotFoundError: 向量库不存在时抛出，需先执行 python main.py build

    返回:
        FAISS 向量库实例
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"向量库不存在: {VECTOR_STORE_PATH}\n"
            "请先执行 python main.py build 构建知识库"
        )
    # 第一步：加载本地 Embedding 模型并读取 FAISS 索引
    _log(logs, "加载本地Embedding模型并读取FAISS索引")
    embedding = get_embeddings()
    vs = FAISS.load_local(
        VECTOR_STORE_PATH,
        embedding,
        allow_dangerous_deserialization=True,  # 允许加载本地 pickle
    )
    _log(logs, "FAISS索引加载完成")
    return vs


def vector_store_exists() -> bool:
    """
    检查向量库是否已构建

    判断标准：目录存在 且 index.faiss 文件存在

    返回:
        True 如果向量库可用，否则 False
    """
    return os.path.exists(VECTOR_STORE_PATH) and os.path.exists(
        os.path.join(VECTOR_STORE_PATH, "index.faiss")
    )
