"""
Streamlit 前端界面 - 非遗直播话术生成器的 Web 交互界面

功能：
- 左侧边栏：非遗品类、复盘评分维度勾选、LLM 参数（历史轮数/温度/top_p/max_tokens/流式开关）
- 中间对话区：主播输入台词，AI 扮演直播间观众返回弹幕；下方展示结构化复盘
- 右侧日志面板：独立展示 RAG 全过程（检索 / Prompt / LLM），与答案正文分开

架构：
    浏览器 Streamlit ←→ Backend API (:6077) ←→ RAG/FAISS ←→ 云端 LLM

启动方式：
    streamlit run ui/streamlit_app.py
    或 python main.py ui

注意：必须先启动 API 服务（python main.py server），否则界面无响应。
"""
import json

import requests
import streamlit as st

# 后端 API 地址（必须与 API 服务端口一致）
BACKEND_URL = "http://127.0.0.1:6077"

# ==================== 评分维度配置 ====================
# 中文名 → 后端维度 key 的映射（与 config/settings.py 的 DIMENSION_LABELS 保持同步）
DIMENSION_OPTIONS = {
    "亮点讲述": "craft_clarity",
    "互动回应": "interaction_effect",
    "疑问回应": "question_response",
    "合规风险": "compliance_risk",
    "节奏把控": "pacing_control",
    "来源依据": "source_basis",
}
# 默认勾选的复盘维度（用户示例：亮点讲述、互动回应、节奏把控）
DEFAULT_DIMENSION_LABELS = ["亮点讲述", "互动回应", "节奏把控"]

# 日志 stage → 右侧面板的展示图标/名称
STAGE_DISPLAY = {
    "RAG": "🔍 检索",
    "Prompt": "📝 Prompt",
    "LLM": "🤖 LLM",
}


def render_assistant_content(content: str) -> str:
    """观众弹幕按行显示：Markdown 中单个换行会被合并成一段，
    每行末尾补两个空格即可实现强制换行（标准 Markdown 语法）。"""
    return content.replace("\n", "  \n")


# ==================== 页面基本设置 ====================
st.set_page_config(
    page_title="非遗直播话术生成器",
    page_icon="🎙️",
    layout="wide",  # 为右侧运行日志留出空间
)


# ==================== 侧边栏配置面板 ====================
with st.sidebar:
    st.title("🎙️ 非遗直播话术生成器")
    st.caption("你扮演主播，AI 只模拟观众反应，不提供主播标准话术。")

    # ---- 场景配置 ----
    category = st.text_input(
        "非遗品类：",
        value="传统手工技艺",
        help="用于知识库检索和复盘场景描述，如：剪纸、刺绣、制瓷",
    )
    # 评分维度勾选：影响复盘侧重（后端只评勾选的维度）
    selected_dimensions = st.multiselect(
        "复盘评分维度（勾选=复盘侧重）：",
        options=list(DIMENSION_OPTIONS),
        default=DEFAULT_DIMENSION_LABELS,
        help="勾选后，复盘报告只对这些维度评分；"
             "例如勾选“节奏把控”，复盘会重点分析主播的节奏问题。",
    )

    st.divider()

    # ---- LLM 推理参数配置 ----
    history_len = st.slider(
        "保留历史对话：", 1, 10, 3, 1,
        help="控制传递给 LLM 的对话轮数，越大上下文越多但 token 消耗也越多",
    )
    temperature = st.slider(
        "temperature：", 0.01, 2.0, 0.5, 0.01,
        help="采样温度，越低回答越确定，直播演练建议 0.5~0.8",
    )
    top_p = st.slider(
        "top_p：", 0.01, 1.0, 0.5, 0.01,
        help="核采样概率阈值",
    )
    max_tokens = st.slider(
        "max_tokens：", 2556, 4096, 3072, 32,
        help="单次生成的最大 token 数",
    )
    stream = st.checkbox(
        "流式输出", value=True,
        help="启用后可以实时看到 LLM 逐字生成的回答",
    )

    st.divider()

    # ---- 操作按钮 ----
    if st.button("清空聊天历史", use_container_width=True):
        # 清空对话、日志和复盘，回到初始状态
        st.session_state.history = []
        st.session_state.logs = []
        st.session_state.review = None

    if st.button("生成复盘报告", type="primary", use_container_width=True):
        # 复盘：把完整聊天历史 + 勾选的评分维度发给后端
        with st.spinner("正在生成复盘报告..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/review",
                    json={
                        "category": category,
                        "history": st.session_state.history,
                        "dimensions": [
                            DIMENSION_OPTIONS[label]
                            for label in selected_dimensions
                        ],
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                    },
                    timeout=180,
                )
                if response.status_code == 200:
                    payload = response.json()
                    st.session_state.review = payload.get("review")
                    st.session_state.logs = payload.get("logs", [])
                    st.rerun()  # 刷新页面展示复盘结果
                else:
                    st.error(f"复盘生成失败: {response.status_code}")
            except requests.RequestException as exc:
                st.error(f"无法连接后端: {exc}")


# ==================== 聊天历史管理 ====================
# 使用 session_state 保持页面刷新后的对话状态
if "history" not in st.session_state:
    st.session_state.history = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "review" not in st.session_state:
    st.session_state.review = None
if "show_logs" not in st.session_state:
    st.session_state.show_logs = True

# 右栏可收起：收起时只显示主对话区 + 一个"展开日志"按钮
if st.session_state.show_logs:
    main_col, log_col = st.columns([2, 1])
else:
    _, toggle_col = st.columns([8, 1])
    with toggle_col:
        if st.button("展开日志", use_container_width=True):
            st.session_state.show_logs = True
            st.rerun()
    main_col = st.container()
    log_col = None


# ==================== 对话区（中间主列） ====================
with main_col:
    st.title("🎙️ 非遗直播话术生成器")
    st.caption("在下方输入主播台词，AI 以直播间观众身份回应；右侧面板展示 RAG 全过程。")

    # 渲染历史消息（观众弹幕按行显示，不在一个段落里）
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(render_assistant_content(msg["content"]))
            else:
                st.markdown(msg["content"])

    # ==================== 复盘报告展示 ====================
    # 输入框已用 st.bottom 钉在视口底部，复盘内容正常渲染在其上方
    if st.session_state.review:
        review = st.session_state.review
        st.divider()
        st.subheader("🎯 结构化复盘报告")
        if review.get("review_source") == "fallback":
            # 占位报告不是模型评价，必须醒目提示，避免被当成真实结论
            st.error(
                "这份复盘不是模型生成的：以下分数是系统内置占位值，"
                "跟本次对话内容无关，不能当作评价结果。通常意味着 API Key 无效或网络不通，"
                "访问后端 /health 可以查看实际生效的模型配置。"
            )
        for item in review.get("dimensions", []):
            name = item.get("name") or item.get("dimension")
            st.markdown(f"**{name}：{item['score']}/5** — {item['finding']}")
            st.caption(f"改进方向：{item['improvement_direction']}")
        if review.get("issues"):
            st.markdown("**发现的问题：** " + "；".join(review["issues"]))
        if review.get("improvement_directions"):
            st.markdown("**改进方向汇总：** " + "；".join(review["improvement_directions"]))

    # ==================== 用户输入处理 ====================
    # 用 st.bottom 把输入框固定在视口底部（中间列下部，宽度跟随主列），
    # 消息渲染在退出 bottom 容器之后进行，保证弹幕回到主对话流而不是被钉在底部
    with st.bottom:
        prompt = st.chat_input("请输入主播本轮台词...")

    if prompt:
        # 每次请求只保留本轮日志
        st.session_state.logs = []

        # 1. 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 调用后端 /chat 接口
        data = {
            "query": prompt,
            "category": category,
            "history": st.session_state.history,
            "history_len": history_len,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        try:
            response = requests.post(
                f"{BACKEND_URL}/chat",
                params={"stream": stream},
                json=data,
                stream=stream,
                timeout=120,  # RAG 检索 + LLM 生成可能较慢，设置较长超时
            )
        except requests.RequestException as exc:
            st.error(f"无法连接后端: {exc}")
            response = None

        if response is not None and response.status_code == 200:
            if stream:
                # 流式模式：逐行解析 NDJSON 事件
                chunks = ""
                assistant_placeholder = st.chat_message("assistant")
                assistant_text = assistant_placeholder.empty()
                error_text = None
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # 非 JSON 行（不应出现），按普通文本追加显示
                        chunks += line
                        assistant_text.markdown(render_assistant_content(chunks))
                        continue
                    if event["type"] == "logs":
                        # RAG 日志：更新右侧面板数据
                        st.session_state.logs = event["items"]
                    elif event["type"] == "answer":
                        chunks += event["content"]
                        assistant_text.markdown(render_assistant_content(chunks))
                    elif event["type"] == "error":
                        # 出错（含安全校验拦截）：丢弃已流出的内容Markdown Preview Mermaid
                        error_text = event["message"]
                        assistant_text.empty()
                if error_text:
                    st.error(error_text)
                elif chunks:
                    # 保存对话到历史记录
                    st.session_state.history.append({"role": "user", "content": prompt})
                    st.session_state.history.append({"role": "assistant", "content": chunks})
            else:
                # 非流式模式：一次性显示完整回答（弹幕按行显示）
                payload = response.json()
                st.session_state.logs = payload.get("logs", [])
                answer = payload.get("answer", "未能获取回答")
                with st.chat_message("assistant"):
                    st.markdown(render_assistant_content(answer))
                st.session_state.history.append({"role": "user", "content": prompt})
                st.session_state.history.append({"role": "assistant", "content": answer})
        elif response is not None:
            # 后端返回错误（如未配置 API Key）：提取 detail 显示
            try:
                detail = response.json().get("detail", "未知错误")
            except ValueError:
                detail = "未知错误"
            st.error(f"请求失败: {detail}")

# ==================== 右侧运行日志面板 ====================
if log_col is not None:
    with log_col:
        title_col, close_col = st.columns([4, 1])
        with title_col:
            st.subheader("RAG 运行过程")
        with close_col:
            if st.button("×", help="收起运行日志", use_container_width=True):
                st.session_state.show_logs = False
                st.rerun()
        st.caption("仅显示最近一次提问的内部步骤，与左侧答案分开展示")
        if st.session_state.logs:
            # 按 stage 分组渲染：检索 / Prompt / LLM 三段过程
            for item in st.session_state.logs:
                stage = item.get("stage", "?")
                icon = STAGE_DISPLAY.get(stage, f"[{stage}]")
                st.markdown(f"**[{icon}]** {item['message']}")
        else:
            st.info("输入问题后，这里会显示检索、Prompt 和 LLM 日志。")