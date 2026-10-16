"""
检索引擎模块 - RAG 检索引擎 + 非遗直播话术生成链

核心功能：
1. build_heritage_knowledge_base(): 构建非遗知识库（加载→分割→向量化→持久化）
2. get_retriever():                 获取 FAISS 检索引擎
3. generate_audience_reply():       主播发言 → 检索知识 → 观众弹幕（一次性生成）
4. stream_audience_reply():         同上，流式逐字生成
5. generate_review():               复盘评分（只评使用者勾选的维度）
6. search_relevant_knowledge():     快速检索知识片段（仅检索，不生成）

整体架构：
  用户输入（主播发言）→ FAISS 向量检索 → Top-K 知识片段
      → Prompt 模板（观众角色）→ 云端 LLM 生成 → 观众弹幕

每一步都会通过 rag.trace.log() 输出 [RAG]/[Prompt]/[LLM] 操作记录，
既打印到后端控制台，也随 API 返回给前端右侧日志面板展示。
"""
import json
import os

from config.settings import (
    KNOWLEDGE_DIR,
    RETRIEVE_TOP_K,
    RETRIEVE_MIN_SCORE,
    RETRIEVE_RELATIVE_RATIO,
    DIMENSION_LABELS,
    DEFAULT_DIMENSIONS,
)
from core.llm import LLMError, get_llm
from core.prompts import AUDIENCE_PROMPT, REVIEW_PROMPT
from core.safety import validated_audience
from core.vector_store import build_vector_store, load_vector_store
from rag.document_loader import load_directory, load_file
from rag.text_splitter import split_documents
from rag.trace import log


def _log(stage: str, message: str, logs: list = None):
    """输出日志；传入 logs 列表时同时保存，供前端右侧面板展示。"""
    if logs is None:
        print(f"[{stage}] {message}", flush=True)
    else:
        log(logs, stage, message)


# ==================== 知识库构建 ====================

def build_heritage_knowledge_base(source_path: str = None):
    """
    构建非遗知识库 - RAG 的最关键预处理步骤

    流程：
    1. 加载文档：自动识别 MD/TXT/PDF 格式（MD 会解析 front matter 元信息）
    2. 分割文档：递归切分为 500 字符块（50 字符重叠）
    3. 向量化：用本地 bge-small-zh-v1.5 生成 512 维向量
    4. 持久化：保存 FAISS 索引到 data/faiss_index/

    参数:
        source_path: 可选，指定单个文件或目录路径；
                     默认使用 KNOWLEDGE_DIR 配置的目录

    返回:
        构建好的 FAISS 向量库实例
    """
    _log("KB", "1/4 开始加载非遗知识文档")
    # 第一步：加载文档
    if source_path and os.path.isfile(source_path):
        # 单个文件：自动识别格式加载
        docs = load_file(source_path)
    elif source_path and os.path.isdir(source_path):
        # 目录：递归加载所有支持文件
        docs = load_directory(source_path)
    else:
        # 默认：加载配置的数据目录
        docs = load_directory(KNOWLEDGE_DIR)

    _log("KB", f"1/4 文档加载完成 | 文档数={len(docs)}")

    _log("KB", "2/4 开始文本切片")
    # 第二步：分割为文本块
    chunks = split_documents(docs)
    _log("KB", f"2/4 文本切片完成 | 切片数={len(chunks)}")

    _log("KB", "3/4 开始向量化并构建 FAISS 索引")
    # 第三步 + 第四步：向量化并持久化
    vs = build_vector_store(chunks)
    _log("KB", "4/4 FAISS 索引已保存")
    return vs


# ==================== 检索引擎 ====================

def get_retriever(k: int = RETRIEVE_TOP_K, logs: list = None):
    """
    获取 FAISS 检索引擎

    内部调用 load_vector_store，会自动输出两条日志：
    [RAG] 加载本地Embedding模型并读取FAISS索引
    [RAG] FAISS索引加载完成

    参数:
        k: 返回的文档片段数量，默认 4（从 settings 读取）

    返回:
        LangChain Retriever 对象，可调用 invoke(query) 执行检索
    """
    vs = load_vector_store(logs=logs)
    return vs.as_retriever(search_kwargs={"k": k})


def _filter_relevant(scored_docs: list) -> list:
    """
    相关性过滤：只保留与问题真正相关的片段

    输入：[(Document, 距离), ...]，按距离升序（相似度降序）排列。
    向量已做过 L2 归一化，余弦相似度 = 1 - 距离²/2。

    过滤规则（同时满足才保留）：
    1. 相似度 ≥ 最高分 × RETRIEVE_RELATIVE_RATIO：与第一名差距过大的视为无关
    2. 相似度 ≥ RETRIEVE_MIN_SCORE：整体都低分时（如无关提问）一个也不保留

    返回:
        [(Document, 相似度), ...] 过滤后的列表，顺序不变
    """
    if not scored_docs:
        return []
    scored = [(doc, 1.0 - distance * distance / 2.0) for doc, distance in scored_docs]
    top_score = scored[0][1]
    # 阈值取两者中的较大值，两个条件必须同时满足
    threshold = max(top_score * RETRIEVE_RELATIVE_RATIO, RETRIEVE_MIN_SCORE)
    return [(doc, score) for doc, score in scored if score >= threshold]


def _retrieve_context(question: str, top_k: int = RETRIEVE_TOP_K, logs: list = None) -> str:
    """
    检索与问题相关的知识片段并拼接为上下文

    输出日志（符合项目要求的 [RAG] 格式）：
    [RAG] 加载本地Embedding模型并读取FAISS索引     ← 由 load_vector_store 输出
    [RAG] FAISS索引加载完成                        ← 由 load_vector_store 输出
    [RAG] 对用户问题向量化，进行相似度检索
    [RAG] 检索完成 | 命中片段=4个 |参考资料总字数=xxxx字
    [RAG] 命中1|来源:xxx.md 第2段|片段内容节选

    检索结果先经过相关性过滤：只保留与问题真正相关的片段，
    无关片段既不写日志，也不进入 Prompt 参考资料。

    参数:
        question: 检索问题（主播发言 / 非遗品类）
        top_k:    候选片段数量
        logs:     日志列表（供前端展示）

    返回:
        拼接好的参考资料文本（多个片段以空行分隔；无相关片段时为空字符串）
    """
    question_preview = question.replace("\n", " ")[:80]
    # 加载向量库（内部输出"加载Embedding/索引加载完成"两条日志）
    vs = load_vector_store(logs=logs)
    # 对用户问题向量化，进行相似度检索
    _log("RAG", f"对用户问题向量化，进行相似度检索 | 问题={question_preview}", logs)
    scored_docs = vs.similarity_search_with_score(question, k=top_k)
    # 相关性过滤：只保留真正相关的片段
    kept = _filter_relevant(scored_docs)
    docs = [doc for doc, _ in kept]
    context = "\n\n".join(doc.page_content for doc in docs)
    # 检索完成统计：命中片段数 + 参考资料总字数
    _log(
        "RAG",
        f"检索完成 | 命中片段={len(docs)}个 |参考资料总字数={len(context)}字",
        logs,
    )
    # 逐条输出命中片段（只输出过滤后保留的相关片段）：
    # 来源文件 + 页码/段号 + 内容节选
    for index, doc in enumerate(docs, start=1):
        source_path = str(doc.metadata.get("source", "未知来源")).replace("\\", "/")
        source = os.path.basename(source_path)
        # PDF 显示页码（PyPDFLoader 的 page 从 0 开始）；MD/TXT 显示段号
        page = doc.metadata.get("page")
        if isinstance(page, int):
            position = f"第{page + 1}页"
        else:
            chunk_index = doc.metadata.get("chunk_index")
            position = f"第{chunk_index}段" if chunk_index else "未知位置"
        preview = doc.page_content.replace("\n", " ")[:60]
        _log(
            "RAG",
            f"命中{index}|来源:{source} {position}|片段内容节选:{preview}",
            logs,
        )
    if not docs:
        _log("RAG", "无相关片段命中 | 本次不提供参考资料", logs)
    return context


# ==================== 观众弹幕生成 ====================

def _format_history(history: list, history_len: int) -> str:
    """
    把前端传来的聊天历史格式化为"主播/观众"对话文本

    前端历史格式：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    这里按 history_len 截取最近几轮（每轮含主播+观众两条），
    并转换成 Prompt 需要的角色格式。
    """
    # 控制历史记录长度：只保留最近的 history_len 轮对话
    if history_len > 0:
        history = history[-2 * history_len:]
    lines = []
    for msg in history:
        role = "主播" if msg.get("role") == "user" else "观众"
        lines.append(f"{role}：{msg.get('content', '')}")
    return "\n".join(lines)


def _build_audience_messages(
    host_text: str,
    category: str,
    history: list = None,
    history_len: int = 3,
    logs: list = None,
):
    """
    组装观众弹幕生成的完整消息列表

    结构：
    - system：观众角色规则（AUDIENCE_PROMPT）
    - user：JSON 格式的演练数据（品类、历史对话、本轮主播发言、RAG 参考资料）

    演练数据放在用户消息里，模型无法用演练数据修改 system 的角色规则。
    """
    # 1. 检索相关知识片段作为参考资料
    knowledge = _retrieve_context(f"{category} {host_text}", logs=logs)
    # 2. 格式化历史对话
    history_text = _format_history(history or [], history_len)
    _log(
        "Prompt",
        f"组装观众弹幕Prompt | 历史消息={len(history_text.splitlines())}条 "
        f"| 参考资料={len(knowledge)}字符",
        logs,
    )
    # 3. 拼接用户消息（JSON 字符串，与 system 规则隔离）
    user_data = json.dumps(
        {
            "category": category,
            "history": history_text,
            "host_text": host_text,
            "knowledge": knowledge,
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": AUDIENCE_PROMPT},
        {"role": "user", "content": user_data},
    ]


def generate_audience_reply(
    host_text: str,
    category: str = "传统手工技艺",
    history: list = None,
    history_len: int = 3,
    temperature: float = 0.5,
    top_p: float = 0.5,
    max_tokens: int = 3072,
    logs: list = None,
) -> str:
    """
    一次性生成观众弹幕（非流式）

    完整流程：
    检索知识 → 组装 Prompt → 云端 LLM 生成 → 安全校验（失败重试一次）

    安全校验：防止模型越权代写主播话术；
    若两次生成都未通过校验，抛出 LLMError 由 API 层转成错误提示。
    """
    messages = _build_audience_messages(
        host_text, category, history=history, history_len=history_len, logs=logs
    )
    llm = get_llm()
    last_error = None
    # 最多尝试两次：第一次正常生成，第二次带上"重新生成"的补充提示
    for attempt in range(2):
        if attempt == 1:
            messages[1]["content"] += "\n\n上次候选内容未通过角色格式或重复性检查。" \
                "请重新生成 2 至 4 条观众弹幕，紧扣本轮主播内容，严格保持观众身份。"
        _log("LLM", f"开始调用 | streaming=False | 第{attempt + 1}次尝试", logs)
        try:
            answer = llm.chat_once(
                messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens
            )
        except LLMError as exc:
            _log("LLM", f"调用失败 | {exc}", logs)
            raise
        except Exception as exc:
            _log("LLM", f"调用失败 | {type(exc).__name__}", logs)
            raise
        checked = validated_audience(answer)
        if checked:
            _log("LLM", f"调用完成 | 输出={len(checked)}字符 | 安全校验通过", logs)
            return checked
        last_error = "输出未通过安全校验（疑似越界代写主播话术）"
        _log("LLM", f"{last_error} | 重新生成", logs)
    raise LLMError(f"{last_error}：两次生成均未通过，请换一种问法重试")


def stream_audience_reply(
    host_text: str,
    category: str = "传统手工技艺",
    history: list = None,
    history_len: int = 3,
    temperature: float = 0.5,
    top_p: float = 0.5,
    max_tokens: int = 3072,
    logs: list = None,
):
    """
    流式生成观众弹幕，事件以字典形式逐条产出

    产出的事件类型：
    - {"type": "logs",  "items": [...]}   检索/Prompt 日志（生成开始前先发）
    - {"type": "answer", "content": "..."} 回答文本片段（逐 token）
    - {"type": "error",  "message": "..."} 调用失败或安全校验未通过
    - {"type": "logs",  "items": [...]}   LLM 完成日志（生成结束后补发）

    与一次性生成的区别：流式模式下内容边生成边返回，安全校验只能在
    生成完成后进行；校验失败时前端会丢弃已流出的文本并显示错误提示。
    """
    # 检索 + 组装（发生在第一次迭代前，这两步是同步的）
    messages = _build_audience_messages(
        host_text, category, history=history, history_len=history_len, logs=logs
    )
    # 先把检索和 Prompt 阶段的日志发给前端，右侧面板立刻能看到过程
    yield {"type": "logs", "items": list(logs or [])}

    llm = get_llm()
    chunks = []
    _log("LLM", "开始调用 | streaming=True", logs)
    try:
        for chunk in llm.chat_stream(
            messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        ):
            chunks.append(chunk)
            yield {"type": "answer", "content": chunk}
    except Exception as exc:
        # LLMError 携带可读原因（如未配置 API Key），优先展示原文
        _log("LLM", f"调用失败 | {type(exc).__name__}: {exc}", logs)
        yield {"type": "logs", "items": list(logs or [])}
        yield {"type": "error", "message": f"LLM 调用失败: {exc}"}
        return

    # 生成结束后做安全校验（流式无法提前拦截，校验失败由前端丢弃已显示内容）
    checked = validated_audience("".join(chunks))
    if checked:
        _log("LLM", f"调用完成 | 输出={len(checked)}字符 | 安全校验通过", logs)
        yield {"type": "logs", "items": list(logs or [])}
    else:
        _log("LLM", "输出未通过安全校验（疑似越界代写主播话术）", logs)
        yield {"type": "logs", "items": list(logs or [])}
        yield {
            "type": "error",
            "message": "输出未通过安全校验（疑似越界代写主播话术），本次内容已拦截，请重新提问。",
        }


# ==================== 复盘评分 ====================

def _normalize_dimensions(payload: dict, requested: list) -> list:
    """
    过滤模型返回的维度评分，并按勾选名单补齐占位，保证覆盖恰好一致。

    模型可能少评或多评某个维度，这里只保留勾选过的维度 key，
    缺失的补上 score=3 的占位条目并注明"模型未返回"。
    """
    items = []
    seen = set()
    for item in payload.get("dimensions") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("dimension", ""))
        if key not in requested or key in seen:
            continue
        seen.add(key)
        items.append({
            "dimension": key,
            "name": str(item.get("name") or DIMENSION_LABELS.get(key, key)),
            "score": int(item.get("score", 3)),
            "finding": str(item.get("finding", "")),
            "improvement_direction": str(item.get("improvement_direction", "")),
        })
    for key in requested:
        if key in seen:
            continue
        items.append({
            "dimension": key,
            "name": DIMENSION_LABELS.get(key, key),
            "score": 3,
            "finding": "模型未返回该维度评分，以下为系统内置占位值，不代表对本次对话的评价。",
            "improvement_direction": "重试或检查模型输出格式后重新生成复盘。",
        })
    return items


def _fallback_review(requested: list) -> dict:
    """云端 LLM 没调通时的占位复盘报告。

    下面的分数和结论是写死的常量，跟本次对话内容无关，所以
    review_source 标成 fallback（前端会渲染成醒目提示）。
    命中这里的典型原因：API Key 无效、网络不通、未配置密钥。
    """
    return {
        "dimensions": [
            {
                "dimension": key,
                "name": DIMENSION_LABELS.get(key, key),
                "score": 3,
                "finding": "需要进一步观察本轮对话的具体表现。",
                "improvement_direction": "配置 API Key 后重新生成该维度的真实评分。",
            }
            for key in requested
        ],
        "issues": ["本次复盘未取得模型评价，无法列出具体问题"],
        "improvement_directions": ["配置 API Key 后重新生成复盘"],
        "review_source": "fallback",
        "note": "本次未取得模型生成的复盘：以下分数是系统内置占位值，不代表对本次对话的评价。",
    }


def generate_review(
    category: str,
    history: list,
    dimensions: list = None,
    temperature: float = 0.5,
    top_p: float = 0.5,
    max_tokens: int = 3072,
    logs: list = None,
) -> dict:
    """
    生成直播演练复盘报告（只评使用者勾选的维度）

    流程：
    1. 检索该品类的知识片段作为专业知识判断依据
    2. 组装复盘 Prompt（评分维度 = 勾选名单，空则用默认三项）
    3. 云端 LLM 生成 JSON 报告 → 解析并补齐缺失维度
    4. LLM 不可用时返回占位报告（review_source=fallback）

    参数:
        category:   非遗品类
        history:    完整聊天历史（[{"role","content"}, ...]）
        dimensions: 勾选的维度 key 列表；为空时使用 DEFAULT_DIMENSIONS
        logs:       日志列表（供前端展示）

    返回:
        复盘报告字典：dimensions / issues / improvement_directions /
        review_source（model=真实评价，fallback=占位报告）
    """
    requested = [key for key in (dimensions or []) if key in DIMENSION_LABELS]
    if not requested:
        requested = list(DEFAULT_DIMENSIONS)

    # 1. 检索知识依据（复盘的专业判断只依据检索资料）
    knowledge = _retrieve_context(category, logs=logs)
    # 2. 组装完整对话文本
    transcript = "\n".join(
        f"第{index // 2 + 1}轮 {'主播' if msg.get('role') == 'user' else '观众'}：{msg.get('content', '')}"
        for index, msg in enumerate(history)
    )
    dimension_text = "、".join(
        f"{key}（{DIMENSION_LABELS[key]}）" for key in requested
    )
    _log(
        "Prompt",
        f"组装复盘Prompt | 评分维度={len(requested)}个 | 对话轮数={len(history) // 2}轮 "
        f"| 参考资料={len(knowledge)}字符",
        logs,
    )
    prompt = REVIEW_PROMPT.format(
        dimensions=dimension_text,
        category=category,
        transcript=transcript,
        knowledge=knowledge,
    )

    # 3. 调用 LLM 生成 JSON 报告
    llm = get_llm()
    _log("LLM", f"开始调用 | streaming=False | 任务=复盘评分", logs)
    try:
        raw = llm.chat_once(
            [{"role": "system", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        _log("LLM", f"调用失败 | {type(exc).__name__} | 使用占位复盘", logs)
        return _fallback_review(requested)

    # 4. 解析 JSON：模型输出可能夹带说明文字，先提取花括号内的内容
    try:
        start, end = raw.find("{"), raw.rfind("}")
        payload = json.loads(raw[start:end + 1]) if start >= 0 else {}
    except (ValueError, TypeError):
        payload = {}
    if not payload.get("dimensions"):
        _log("LLM", "输出格式无效 | 使用占位复盘", logs)
        return _fallback_review(requested)

    _log("LLM", f"调用完成 | 输出={len(raw)}字符 | 复盘解析成功", logs)
    return {
        "dimensions": _normalize_dimensions(payload, requested),
        "issues": [str(item) for item in payload.get("issues", [])],
        "improvement_directions": [
            str(item) for item in payload.get("improvement_directions", [])
        ],
        "review_source": "model",
        "note": "",
    }


# ==================== 仅检索（供调试/扩展） ====================

def search_relevant_knowledge(query: str, k: int = RETRIEVE_TOP_K):
    """
    快速检索相关知识片段（仅检索，不生成回答）

    适用场景：只想查看知识库原文，不需要 LLM 参与。

    参数:
        query: 检索关键词或问题描述
        k:     返回的文档数量

    返回:
        LangChain Document 列表，每个包含 page_content（文本）和 metadata
    """
    retriever = get_retriever(k=k)
    return retriever.invoke(query)
