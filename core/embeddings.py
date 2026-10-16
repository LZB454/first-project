"""
Embedding 嵌入模型管理模块（本地运行，不调用云端）

加载 ONNX 格式的 bge-small-zh-v1.5 中文嵌入模型，直接用
onnxruntime 推理（不依赖 fastembed，离线可用）：
- 模型文件缓存在 data/models/（已随项目附带，约 91MB）
- 向量维度 512，适合中文非遗知识文本的语义检索
- 无需 API Key，推理在本地 CPU 完成

推理流程（与 fastembed 对 bge 系列的处理一致）：
1. tokenizers 分词（查询自动加 BGE 检索指令前缀）
2. ONNX 前向计算得到 last_hidden_state
3. 按 attention_mask 做平均池化（mean pooling）
4. L2 归一化，得到可用于余弦相似度的向量

本模块对外暴露 LangChain 兼容的接口（embed_documents / embed_query），
可直接作为 langchain FAISS 向量库的 embedding 参数使用。

性能说明：
- get_embeddings() 使用进程级单例 + 双检锁，模型只加载一次，
  后续请求直接复用实例，避免每次检索都重新加载模型。
"""
import threading
import glob
import os

from langchain_core.embeddings import Embeddings

from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_CACHE_DIR

# BGE 系列官方推荐的检索指令前缀：查询时加在问题前面可提升检索效果
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 进程级单例 + 初始化锁（双检锁，保证并发下只初始化一次）
_embedding_instance = None
_embedding_lock = threading.Lock()


class OnnxEmbedding(Embeddings):
    """直接用 onnxruntime 运行 bge-small-zh-v1.5 的本地嵌入器。

    继承 langchain_core.embeddings.Embeddings：
    FAISS.from_documents / similarity_search 需要标准的
    embed_documents / embed_query 接口。

    依赖：onnxruntime + tokenizers + numpy（均为纯 CPU 库）。
    模型文件：在 EMBEDDING_CACHE_DIR 下查找 model_optimized.onnx
    和 tokenizer.json（fastembed 的 HuggingFace 缓存目录结构）。
    """

    def __init__(self, cache_dir: str = EMBEDDING_CACHE_DIR):
        import numpy as np  # 延迟导入，加快服务启动
        import onnxruntime as ort
        from tokenizers import Tokenizer

        # 1. 在缓存目录下查找模型文件（refs 子目录路径可能变化，用通配符兜底）
        model_files = sorted(glob.glob(os.path.join(cache_dir, "**", "model_optimized.onnx"), recursive=True))
        tokenizer_files = sorted(glob.glob(os.path.join(cache_dir, "**", "tokenizer.json"), recursive=True))
        if not model_files or not tokenizer_files:
            raise FileNotFoundError(
                f"未找到本地 Embedding 模型文件（{cache_dir} 下缺少 model_optimized.onnx / tokenizer.json）。\n"
                "请把 bge-small-zh-v1.5 的模型缓存复制到 data/models/ 目录。"
            )
        self.model_path = model_files[0]
        self.tokenizer_path = tokenizer_files[0]

        # 2. 创建 ONNX 推理会话（CPU）
        self.session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        # 3. 加载分词器并启用批内填充（同一批文本补齐到最长长度）
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        # 4. 记录输入输出名（bge 模型：3 个输入，输出 last_hidden_state）
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        self.np = np

    def _encode_batch(self, texts) -> list:
        """分词一批文本，返回 (input_ids, attention_mask, token_type_ids) 三个 int64 数组。"""
        encodings = self.tokenizer.encode_batch(list(texts))
        return self.np.array([e.ids for e in encodings], dtype="int64"), \
            self.np.array([e.attention_mask for e in encodings], dtype="int64"), \
            self.np.array([e.type_ids for e in encodings], dtype="int64")

    def _run(self, texts) -> list:
        """分词 → ONNX 推理 → 平均池化 → L2 归一化，返回向量列表。"""
        input_ids, attention_mask, token_type_ids = self._encode_batch(texts)
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        # 只传模型实际需要的输入（不同导出版本输入名可能略有差异）
        feed = {name: feed[name] for name in self.input_names if name in feed}
        hidden = self.session.run([self.output_name], feed)[0]  # (B, L, 512)

        # 平均池化：按 attention_mask 对 token 向量求平均，得到句向量
        mask = attention_mask[:, :, None]  # (B, L, 1)
        summed = (hidden * mask).sum(axis=1)  # (B, 512)
        counts = mask.sum(axis=1).clip(min=1e-9)  # (B, 1)，避免除零
        vectors = summed / counts

        # L2 归一化：FAISS 默认按 L2 距离检索，归一化后等价于余弦相似度
        norms = self.np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms.clip(min=1e-9)
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts):
        """批量向量化知识片段（构建知识库时使用，不加检索前缀）"""
        return self._run(texts)

    def embed_query(self, text):
        """向量化查询问题（检索时使用，自动加 BGE 检索指令前缀）"""
        return self._run([BGE_QUERY_PREFIX + text])[0]


def _load_fastembed_fallback():
    """备选方案：fastembed 加载模型（当 onnxruntime 未安装时）。

    正常情况不会走到这里：OnnxEmbedding 已经覆盖了本地推理。
    """
    from fastembed import TextEmbedding  # 延迟导入

    class _FastEmbedAdapter:
        def __init__(self):
            self.model = TextEmbedding(
                model_name=EMBEDDING_MODEL_NAME,
                cache_dir=EMBEDDING_CACHE_DIR,
            )

        def embed_documents(self, texts):
            return [
                [float(value) for value in vector]
                for vector in self.model.embed(list(texts))
            ]

        def embed_query(self, text):
            return [float(value) for value in next(iter(self.model.query_embed([text])))]

    return _FastEmbedAdapter()


def get_embeddings() -> OnnxEmbedding:
    """
    获取 Embedding 模型实例（进程级单例，线程安全）

    加载顺序：
    1. OnnxEmbedding：onnxruntime 直接推理本地缓存模型（首选，离线可用）
    2. fastembed：onnxruntime 不可用时的备选（需要联网或已有缓存）

    双检锁设计：
    - 并发首请求时只有第一个线程执行模型加载，
      其余线程等待锁释放后直接复用已加载实例
    - 避免多线程同时初始化 ONNX 会话导致的冲突

    返回:
        嵌入器实例，可直接用于 FAISS.from_documents()
    """
    global _embedding_instance
    if _embedding_instance is None:
        with _embedding_lock:
            if _embedding_instance is None:
                try:
                    _embedding_instance = OnnxEmbedding()
                except FileNotFoundError:
                    raise  # 模型文件缺失：直接抛出，给出指引
                except ImportError:
                    # onnxruntime/tokenizers 未安装 → 退回 fastembed
                    _embedding_instance = _load_fastembed_fallback()
    return _embedding_instance
