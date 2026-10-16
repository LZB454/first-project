# 非遗直播话术生成器 (ICH Livestream Script Generator)

基于 RAG + 云端大模型构建的非遗直播话术演练系统。你扮演主播输入台词，AI 以直播间观众身份回应（提问、质疑、追问），并可按勾选的评分维度生成结构化复盘报告。

## 项目简介

本项目面向非遗直播场景，提供**话术演练**与**复盘评分**两大能力：

- **话术演练**：系统先通过 RAG 检索非遗知识库（剪纸、刺绣、制瓷、直播合规等公开资料），再把检索片段作为背景知识交给云端 LLM，由 LLM 扮演直播间观众生成弹幕。AI 只做观众反应，严禁代写主播话术。
- **复盘评分**：使用者可在左侧勾选评分维度（如亮点讲述、互动回应、节奏把控），复盘报告只对勾选的维度评分，实现"按需侧重"的复盘。
- **过程透明**：每次问答的 RAG 全过程（检索 → Prompt → LLM）以操作日志形式展示在右侧独立面板，与答案正文分开展示。

系统采用前后端分离架构：后端基于 FastAPI 提供 RESTful API 和 NDJSON 流式响应，前端使用 Streamlit。

## 项目架构

```
┌──────────────────────────────────────────────────────────────┐
│                        前端界面层                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │               Streamlit App                          │     │
│  │   左侧参数(评分维度/LLM参数) | 中间对话+复盘 | 右侧RAG日志  │     │
│  └────────┬────────────────────────────────────────────┘     │
│           │         HTTP / NDJSON流式                         │
├───────────┼──────────────────────────────────────────────────┤
│           │         API服务层                                 │
│  ┌────────▼────────────────────────────────────────────┐     │
│  │         FastAPI Server (:6077)                       │     │
│  │  /chat    /review    /rag/search    /health          │     │
│  └────────┬────────────────────────────────────────────┘     │
│           │                                                   │
├───────────┼───────────────────────────────────────────────────┤
│           │         业务逻辑层（RAG 管线）                      │
│  ┌────────▼────────────────────────────────────────────┐     │
│  │  文档加载 → 文本分割 → 向量化 → FAISS 检索            │     │
│  │        → Prompt 组装 → LLM 生成 → 安全校验            │     │
│  └────────┬────────────────────────────────────────────┘     │
│           │                                                   │
├───────────┼───────────────────────────────────────────────────┤
│           │              基础设施层                            │
│  ┌────────▼────────────────────────────────────────────┐     │
│  │  云端LLM(DeepSeek)   本地Embedding(bge-small-zh)    │     │
│  │                      FAISS Vector DB               │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## 目录结构

```
project3/
├── config/                     # 配置模块
│   └── settings.py            # 全局配置（LLM云端、Embedding、检索与推理参数）
├── core/                       # 核心模块
│   ├── llm.py                 # 云端LLM统一管理（OpenAI兼容协议，默认DeepSeek）
│   ├── embeddings.py          # 本地Embedding模型管理（fastembed + bge-small-zh-v1.5）
│   ├── vector_store.py        # FAISS向量库的构建/保存/加载
│   ├── prompts.py             # 业务Prompt集中管理（观众弹幕/复盘评分）
│   └── safety.py              # 输出安全校验（防止模型越权代写主播话术）
├── rag/                        # RAG模块
│   ├── document_loader.py     # 文档加载（MD/TXT/PDF，MD支持front matter元信息）
│   ├── text_splitter.py       # 递归文本分割
│   ├── retriever.py           # 检索引擎 + 观众弹幕生成链 + 复盘生成
│   └── trace.py               # RAG运行日志工具（[RAG]/[Prompt]/[LLM]）
├── api/                        # API服务层
│   └── server.py              # FastAPI服务
├── ui/                         # 前端界面
│   └── streamlit_app.py       # Streamlit界面
├── data/
│   ├── knowledge/             # 非遗知识文档（MD，含front matter元信息）
│   ├── models/                # Embedding模型缓存（已内置，离线可用）
│   └── faiss_index/           # FAISS向量索引（由 build 命令生成）
├── main.py                     # 主入口
└── requirements.txt            # 依赖清单
```

## 快速开始

### 环境要求

- Python 3.10+
- 一个 DeepSeek API Key（https://platform.deepseek.com 申请，云端调用）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 第一步：配置 API Key

复制 `.env.example` 为 `.env`（项目已附带 `.env`，其中 API Key 留空），填写自己的密钥：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### 第二步：构建知识库

```bash
python main.py build
```

该命令会完成：文档加载 → 文本分割 → 向量化 → FAISS 索引构建。
Embedding 模型已缓存在 `data/models/`，离线即可完成构建。

### 第三步：启动后端服务

```bash
python main.py server
```

服务默认启动在 `http://0.0.0.0:6077`，API 文档可访问 `http://127.0.0.1:6077/docs`。

### 第四步：启动前端界面（另开终端）

```bash
python main.py ui
```

浏览器访问 `http://localhost:8501`。

## 界面说明

| 区域 | 内容 |
|------|------|
| 左侧边栏 | 非遗品类、**复盘评分维度勾选**（影响复盘侧重）、历史对话轮数（1-10）、temperature（0.01-2.00）、top_p（0.01-1.00）、max_tokens（2556-4096）、流式输出开关、清空历史、生成复盘按钮 |
| 中间对话区 | 主播输入台词，AI 返回观众弹幕；复盘报告展示在对话下方 |
| 右侧日志面板 | **独立展示** RAG 全过程：🔍 检索 → 📝 Prompt → 🤖 LLM，与答案正文分开，不混入回答 |

### 评分维度说明

左侧"复盘评分维度"可多选，勾选结果会影响复盘侧重——复盘报告只对勾选的维度评分：

| 维度 | 说明 |
|------|------|
| 亮点讲述 | 技艺亮点讲述清晰度 |
| 互动回应 | 与观众的互动效果 |
| 疑问回应 | 观众疑问是否得到回应 |
| 合规风险 | 合规风险检查（虚假夸大、绝对化承诺等） |
| 节奏把控 | 直播讲解节奏是否得当 |
| 来源依据 | 说法是否有可核验的公开来源 |

默认勾选：亮点讲述、互动回应、节奏把控；全部取消勾选时使用默认三项。

### RAG 运行日志格式

每次提问，右侧面板展示本次 RAG 全过程的操作记录，例如：

```
[RAG] 加载本地Embedding模型并读取FAISS索引
[RAG] FAISS索引加载完成
[RAG] 对用户问题向量化，进行相似度检索
[RAG] 检索完成 | 命中片段=4个 |参考资料总字数=1234字
[RAG] 命中1|来源:heritage_ceramics.md 第2段|片段内容节选:手工制瓷通常包括制坯……
[RAG] 命中2|来源:live_compliance.md 第1段|片段内容节选:直播介绍商品或服务时……
[Prompt] 组装观众弹幕Prompt | 历史消息=6条 | 参考资料=1234字符
[LLM] 开始调用 | streaming=True
[LLM] 调用完成 | 输出=156字符 | 安全校验通过
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | 直播对话（主播发言 → 观众弹幕，支持 NDJSON 流式输出） |
| `/review` | POST | 复盘评分（按勾选维度生成结构化报告） |
| `/rag/search` | POST | 仅检索知识片段，不生成回答 |
| `/health` | GET | 健康检查（LLM 配置 / 向量库状态） |

### /chat 流式协议

`stream=true` 时返回 NDJSON，每行一个事件：

```json
{"type": "logs", "items": [{"stage": "RAG", "message": "..."}]}
{"type": "answer", "content": "好奇观众：……"}
{"type": "logs", "items": [{"stage": "LLM", "message": "..."}]}
```

日志事件在回答前后各发一次，右侧面板先看到检索过程，再看到 LLM 完成状态。

## 测试与评测

`tests/` 目录提供两类验证（详见 [tests/README.md](tests/README.md)）：

```bash
# 离线单元测试（不调 LLM）：检索过滤逻辑、噪声过滤等
pytest tests/test_retrieval.py -v

# 在线场景评测（真实调用云端 LLM，15 组场景输入）
# 统计对话连贯性与复盘要点覆盖度，结果写入 tests/evaluation_results.md
python tests/evaluate_cases.py
```

## 配置说明

在 [config/settings.py](config/settings.py) 中可配置：

- **LLM 云端服务**：修改 `LLM_CONFIG` 切换其他 OpenAI 兼容服务
- **Embedding 模型**：修改 `EMBEDDING_MODEL_NAME`（模型缓存目录 `data/models/`）
- **文本分割参数**：`CHUNK_SIZE`、`CHUNK_OVERLAP`
- **检索数量**：`RETRIEVE_TOP_K`
- **API 服务端口**：`API_HOST`、`API_PORT`（默认 6077）
- **推理默认参数**：`DEFAULT_TEMPERATURE`、`DEFAULT_TOP_P`、`DEFAULT_MAX_TOKENS`、`DEFAULT_HISTORY_LEN`

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 大语言模型 | DeepSeek（云端 OpenAI 兼容 API） |
| Embedding | bge-small-zh-v1.5（fastembed 本地 ONNX） |
| 向量数据库 | FAISS |
| RAG 框架 | LangChain |
| 后端框架 | FastAPI + Uvicorn |
| 前端框架 | Streamlit |

## 项目来源

- 架构参考 `rag_law`（config/core/rag/api/ui 分层 + main.py 命令行 + 日志体系）
- 业务逻辑参考 `project2`（非遗直播话术生成器的 Prompt、评分维度、安全校验）
- 知识库沿用 `project2/data/knowledge/` 的非遗公开资料（课程演示用途，不含真实个人隐私）
