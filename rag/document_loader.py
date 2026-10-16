"""
文档加载模块 - 支持 Markdown / TXT / PDF 多种格式

通过文件扩展名自动选择对应的加载器：
- .md / .txt → 文本加载（自动检测编码）
- .pdf       → PyPDFLoader（PDF 全文提取，逐页生成 Document）

非遗知识库以 Markdown 为主（沿用 project2 的知识文档），
Markdown 文件支持 YAML front matter 元信息（title/category/source/source_url），
这些元信息会写入 metadata，供检索日志展示来源。

核心设计模式：LOADERS 字典映射 + load_file() 工厂函数，
新增文件格式只需在 LOADERS 中注册即可。
"""
import os
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document


def parse_front_matter(text: str, path: str) -> tuple:
    """
    解析 Markdown 文件开头的 YAML front matter 元信息

    知识文档格式示例：
        ---
        title: 手工制瓷公开知识摘要
        category: 非遗知识
        source: 公开传统工艺资料整理-手工制瓷.md
        source_url: https://www.ihchina.cn/
        ---
        正文内容……

    参数:
        text: 文件完整文本
        path: 文件路径（用于生成默认 source 文件名）

    返回:
        (metadata 字典, 去掉 front matter 后的正文)
    """
    # 默认元信息：标题取文件名，其余给兜底值
    metadata = {
        "title": os.path.splitext(os.path.basename(path))[0],
        "category": "未分类",
        "source": os.path.basename(path),
        "source_url": "",
    }
    if not text.startswith("---"):
        return metadata, text
    # front matter 以两行 "---" 包裹，取中间部分
    parts = text.split("---", 2)
    if len(parts) < 3:
        return metadata, text
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key in metadata:
                metadata[key] = value.strip()
    return metadata, parts[2].strip()


def load_markdown(file_path: str) -> List[Document]:
    """
    加载 Markdown 知识文档

    与普通文本加载的区别：会解析 front matter，
    把 title/category/source 写入 Document 的 metadata。

    参数:
        file_path: 文件绝对路径

    返回:
        包含全文内容的 Document 列表（1 个元素，含元信息）
    """
    with open(file_path, encoding="utf-8") as f:
        raw = f.read()
    metadata, body = parse_front_matter(raw, file_path)
    return [Document(page_content=body, metadata=metadata)]


def load_txt(file_path: str) -> List[Document]:
    """
    加载 TXT 纯文本文件

    参数:
        file_path: 文件绝对路径

    返回:
        包含全文内容的 Document 列表（通常只有 1 个元素）
    """
    loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
    return loader.load()


def load_pdf(file_path: str) -> List[Document]:
    """
    加载 PDF 文件

    使用 PyPDFLoader 逐页提取文本，每页生成一个 Document 对象，
    metadata 中包含 page_number 和 source 信息。

    参数:
        file_path: 文件绝对路径

    返回:
        Document 列表（每个元素对应一页 PDF）
    """
    loader = PyPDFLoader(file_path)
    return loader.load()


# 扩展名 → 加载器函数的映射表
# 新增格式时只需在此添加映射即可，无需修改其他代码
LOADERS = {
    ".md": load_markdown,
    ".txt": load_txt,
    ".pdf": load_pdf,
}


def load_file(file_path: str) -> List[Document]:
    """
    根据文件扩展名自动选择加载器（单文件版）

    参数:
        file_path: 文件绝对路径

    返回:
        Document 列表，不支持的格式返回空列表
    """
    ext = os.path.splitext(file_path)[1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        print(f"不支持的文件格式: {ext} ({file_path})")
        return []
    return loader(file_path)


def load_directory(directory_path: str) -> List[Document]:
    """
    递归加载目录下所有支持的知识文档

    遍历目录树，对每个支持格式的文件自动选择合适的加载器，
    汇总所有文档到统一列表中供后续分割和向量化。

    参数:
        directory_path: 文档根目录路径

    返回:
        所有文档的完整列表
    """
    all_docs = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            # README.md 是目录说明，列出所有文件名后会给检索引入噪声，跳过
            if file.lower() == "readme.md":
                continue
            ext = os.path.splitext(file)[1].lower()
            if ext in LOADERS:  # 只处理支持的文件格式
                file_path = os.path.join(root, file)
                try:
                    docs = load_file(file_path)
                    all_docs.extend(docs)
                    print(f"已加载: {file_path}")
                except Exception as e:
                    print(f"加载失败 {file_path}: {e}")
    return all_docs
