"""RAG 检索离线单元测试（不调用 LLM、不产生 token 消耗）。

运行：pytest tests/test_retrieval.py -v
前置：已执行 python main.py build（存在 FAISS 索引）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from langchain_core.documents import Document

from rag.retriever import _filter_relevant, _retrieve_context


def test_vector_store_ready():
    """FAISS 索引已构建，检索可用。"""
    from core.vector_store import vector_store_exists
    assert vector_store_exists(), "向量库不存在，请先执行 python main.py build"


def test_relevant_query_hits_only_same_topic():
    """剪纸查询只命中剪纸相关片段，不混入制瓷/刺绣。"""
    logs = []
    _retrieve_context("剪纸 这幅作品的蝙蝠纹样有什么寓意", logs=logs)
    # 只取"命中N|来源:..."开头的逐条命中日志（"检索完成|命中片段=3个"是汇总行，不算）
    hit_lines = [item["message"] for item in logs if item["message"].startswith("命中")]
    assert hit_lines, "剪纸查询应至少命中一个片段"
    for line in hit_lines:
        assert "剪纸" in line, f"命中了无关片段: {line}"


def test_noise_query_returns_no_hits():
    """完全无关的提问不应命中任何片段。"""
    logs = []
    context = _retrieve_context("今天天气不错，出去玩吧", logs=logs)
    assert context == "", f"无关提问不应有参考资料，实际={context!r}"
    messages = [item["message"] for item in logs]
    assert any("命中片段=0个" in message for message in messages)


def test_filter_relevant_drops_below_threshold():
    """相关性过滤：与最高分差距过大的候选被丢弃。

    输入为 (doc, 距离)，余弦相似度 = 1 - 距离²/2：
    距离 0.30 → cos≈0.955（第一名），距离 0.80 → cos≈0.68。
    阈值 = max(0.955×0.85, 0.5) ≈ 0.81，第二名的 0.68 应被过滤。
    """
    doc_a = Document(page_content="剪纸纹样寓意", metadata={"source": "a.md"})
    doc_b = Document(page_content="制瓷工序", metadata={"source": "b.md"})
    kept = _filter_relevant([(doc_a, 0.30), (doc_b, 0.80)])
    assert len(kept) == 1 and kept[0][0] is doc_a


def test_filter_relevant_keeps_same_topic_chunks():
    """同主题的多个切片（分数接近）应一起保留。

    距离 0.775 → cos≈0.70，距离 0.837 → cos≈0.65。
    阈值 = max(0.70×0.85, 0.5) = 0.595，第二名的 0.65 ≥ 0.595，两者都保留。
    """
    doc_a = Document(page_content="四大名绣", metadata={"source": "a.md"})
    doc_b = Document(page_content="工艺流程", metadata={"source": "a.md"})
    kept = _filter_relevant([(doc_a, 0.775), (doc_b, 0.837)])
    assert len(kept) == 2
