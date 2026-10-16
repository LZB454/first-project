"""
文本分割模块 - 将长文档切分为适合检索的文本块

使用 LangChain 的 RecursiveCharacterTextSplitter（递归字符分割器）：
- 优先按段落（\\n\\n）分割，再按句子（\\n、。），最后按字符
- 相邻块之间有重叠，避免关键信息被切断
- 分割参数可通过 config/settings.py 调整

分割后会给每个文本块补充 chunk_index（在源文档中的序号），
用于检索日志中展示"第 x 段"。
"""
from typing import List

# langchain 1.x 把分割器拆到了独立的 langchain-text-splitters 包，
# 旧版（0.3.x）则还在 langchain.text_splitter 里，这里做兼容导入。
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    将文档列表递归分割为固定大小的文本块

    分割策略：
    1. 先尝试按双换行（段落）分割
    2. 段落过长则按单换行 / 句号（句子）分割
    3. 句子仍过长则按字符硬切分
    4. 相邻块保留 chunk_overlap 字符的重叠，确保语义不会在边界断裂

    参数:
        docs:          原始 Document 列表
        chunk_size:    每个文本块的最大字符数（默认 500）
        chunk_overlap: 相邻块的重叠字符数（默认 50）

    返回:
        分割后的 Document 列表，metadata 保留源文档信息并补充 chunk_index
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 分割符优先级：段落 > 句子 > 空格 > 字符
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    # 每个源文档切出的块，按出现顺序编号（1 起），写入 metadata
    chunks = []
    for doc in docs:
        for index, piece in enumerate(text_splitter.split_text(doc.page_content), start=1):
            # 复制 metadata，追加块序号；PDF 的页码（page）原样保留
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = index
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks
