"""
项目配置：DeepSeek 云端 LLM、本地 Embedding 模型和 RAG 资源路径。

所有路径都基于项目根目录解析，无论从哪个工作目录启动都不会跑偏。
敏感信息（API Key）从 .env 文件读取，.env 不提交到版本库。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录：本文件的上一级目录的上一级（config/settings.py → project3/）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 加载项目根目录下的 .env 环境变量文件
load_dotenv(PROJECT_ROOT / ".env")


# ==================== LLM 大语言模型配置（云端调用） ====================
# 使用 DeepSeek 云端 API（OpenAI 兼容协议，国内可直连）。
# DEEPSEEK_API_KEY 留空即可，请在自己申请的密钥基础上填写到 .env 文件中。
LLM_DEEPSEEK_CONFIG = {
    # API 密钥：在 https://platform.deepseek.com 申请后填入 .env
    "api_key": os.getenv("DEEPSEEK_API_KEY", "").strip(),
    # API 地址（OpenAI 兼容格式，不需要再拼 /v1）
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
    # 模型名称：deepseek-chat 指向 DeepSeek-V3 对话模型
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
}

# 当前使用的 LLM 配置（后续如需切换其他云端服务，只需改这一行）
LLM_CONFIG = LLM_DEEPSEEK_CONFIG


# ==================== Embedding 嵌入模型配置（本地运行） ====================
# Embedding 不调用云端：用 onnxruntime 直接推理 ONNX 格式的
# bge-small-zh-v1.5 中文嵌入模型（512 维，约 91MB，无需 API Key）。
# 模型缓存目录 data/models/ 已内置模型文件，离线即可使用。
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "Qdrant/bge-small-zh-v1.5",
).strip()
EMBEDDING_CACHE_DIR = str(PROJECT_ROOT / "data" / "models")


# ==================== RAG 知识库路径 ====================
KNOWLEDGE_DIR = str(PROJECT_ROOT / "data" / "knowledge")   # 非遗知识文档存放目录（md/txt/pdf）
VECTOR_STORE_PATH = str(PROJECT_ROOT / "data" / "faiss_index")  # FAISS 向量索引持久化路径


# ==================== 文本切片和检索参数 ====================
CHUNK_SIZE = 500       # 每个文本块的最大字符数（中文字符）
CHUNK_OVERLAP = 50     # 相邻文本块之间的重叠字符数，保证语义连续性
RETRIEVE_TOP_K = 4     # 检索时返回的最相关文档片段数量

# 检索相关性过滤（向量已 L2 归一化，余弦相似度 = 1 - 距离²/2）
# 同时满足两个条件才保留：相似度 ≥ 最高分 × 相对比值，且 ≥ 绝对下限。
# 目的：只保留与问题真正相关的片段，无关片段不写入日志也不进入 Prompt。
# 相对比值按知识库规模校准：切片较细时同文档多个切片都可能真相关，
# 0.85（允许比最高分低 15%）能保留同主题切片、过滤跨主题噪声。
RETRIEVE_MIN_SCORE = 0.5       # 绝对下限：低于该相似度的片段一律视为无关（真实相关命中一般在 0.59 以上）
RETRIEVE_RELATIVE_RATIO = 0.85  # 相对比值：低于最高分 × 该比值的片段视为无关


# ==================== FastAPI 服务配置 ====================
API_HOST = "0.0.0.0"   # 监听所有网络接口，允许局域网访问
API_PORT = 6077        # 服务端口号（避开 rag_law 的 6066 和 project2 的 8000）


# ==================== LLM 推理默认参数 ====================
DEFAULT_TEMPERATURE = 0.5    # 采样温度：0.01~2.00，越低输出越确定
DEFAULT_TOP_P = 0.5          # 核采样概率阈值：0.01~1.00
DEFAULT_MAX_TOKENS = 3072    # 单次生成最大 token 数（页面可选 2556~4096）
DEFAULT_HISTORY_LEN = 3      # 默认保留的历史对话轮数（页面可选 1~10）


# ==================== 复盘评分维度 ====================
# 维度 key → 中文名 的映射。侧边栏多选后随请求传给后端，
# 复盘时只对勾选的维度评分；未勾选任何维度时使用 DEFAULT_DIMENSIONS。
DIMENSION_LABELS = {
    "craft_clarity": "亮点讲述",
    "interaction_effect": "互动回应",
    "question_response": "疑问回应",
    "compliance_risk": "合规风险",
    "pacing_control": "节奏把控",
    "source_basis": "来源依据",
}

# 页面默认勾选的复盘维度（用户示例：亮点讲述、互动回应、节奏把控）
DEFAULT_DIMENSIONS = ["craft_clarity", "interaction_effect", "pacing_control"]
