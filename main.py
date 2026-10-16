"""
非遗直播话术生成器 - 主入口

命令行接口，支持四种操作模式：
    python main.py build       # 构建非遗知识库（FAISS 索引）
    python main.py server      # 启动 FastAPI 服务
    python main.py ui          # 启动 Streamlit 前端界面
    python main.py test        # 测试各模块功能

项目结构：
    config/   - 全局配置（LLM 云端、Embedding、向量库参数）
    core/     - 核心组件（LLM、Embedding 模型、FAISS 向量库、业务 Prompt）
    rag/      - RAG 管线（文档加载、文本分割、检索引擎）
    api/      - FastAPI 服务端
    ui/       - 前端界面（Streamlit）
    data/     - 数据目录（非遗知识文档 + FAISS 索引 + Embedding 模型缓存）
"""
import sys
import os

# 将项目根目录加入 Python 路径，确保所有模块可以正确导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 控制台默认 GBK 编码，强制 stdout 输出 UTF-8，避免 [RAG] 日志乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_knowledge_base():
    """
    构建非遗知识库

    流程：
    1. 扫描 data/knowledge/ 目录下的所有支持文档（MD/TXT/PDF）
    2. 逐文件加载内容（MD 自动解析 front matter 元信息）
    3. 递归分割为 500 字符的文本块（50 字符重叠）
    4. 用本地 bge-small-zh-v1.5 生成 512 维向量
    5. 保存 FAISS 索引到 data/faiss_index/

    文档要求：
    - 放置在 data/knowledge/ 目录下
    - 支持 .md / .txt / .pdf 格式
    - 建议使用 UTF-8 编码
    """
    print("=" * 50)
    print("开始构建非遗知识库...")
    print("=" * 50)

    from rag.retriever import build_heritage_knowledge_base
    from config.settings import KNOWLEDGE_DIR

    source = KNOWLEDGE_DIR
    if not os.path.exists(source):
        print(f"知识库目录不存在: {source}")
        print("请先在 data/knowledge/ 目录下放置非遗知识文档（支持 MD/TXT/PDF）")
        return

    build_heritage_knowledge_base(source)
    print("知识库构建完成！")


def start_server():
    """
    启动 FastAPI 服务

    启动后可通过以下地址访问：
    - API 文档: http://127.0.0.1:6077/docs（Swagger UI）
    - API 接口: http://127.0.0.1:6077
    """
    from api.server import app
    from config.settings import API_HOST, API_PORT
    import uvicorn

    print(f"启动非遗直播话术生成器API服务: http://{API_HOST}:{API_PORT}")
    print("API文档: http://127.0.0.1:6077/docs")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")


def start_ui():
    """
    启动 Streamlit 前端界面

    注意：需要先启动 API 服务（python main.py server），
    Streamlit 前端依赖后端 API 提供 RAG 检索和 LLM 生成功能。

    用 sys.executable -m streamlit 启动，确保始终使用当前 Python 环境里
    安装的 Streamlit（而不是 PATH 上其它环境的 streamlit 命令）。
    """
    import subprocess

    ui_path = os.path.join(os.path.dirname(__file__), "ui", "streamlit_app.py")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path])
    except KeyboardInterrupt:
        # Ctrl+C 停止界面：正常退出，不打 traceback
        print("\n已停止 Streamlit 界面")


def test():
    """
    测试各模块功能

    验证：
    1. 向量库是否存在
    2. RAG 检索是否正常工作
    3. LLM 是否配置了 API Key
    """
    print("=" * 50)
    print("测试非遗直播话术生成器各模块...")
    print("=" * 50)

    # 1. 检查向量库
    from core.vector_store import vector_store_exists
    if not vector_store_exists():
        print("[警告] 向量库不存在，请先执行 python main.py build 构建知识库")
        return

    # 2. 测试 RAG 检索
    print("\n--- 测试RAG检索 ---")
    from rag.retriever import search_relevant_knowledge
    docs = search_relevant_knowledge("制瓷工艺 烧制 施釉")
    print(f"检索到 {len(docs)} 个相关片段")
    if docs:
        print(f"第一条: {docs[0].page_content[:200]}...")

    # 3. 检查 LLM 配置
    print("\n--- 检查LLM配置 ---")
    from config.settings import LLM_CONFIG
    if LLM_CONFIG["api_key"]:
        print(f"LLM 已配置 | model={LLM_CONFIG['model']}")
    else:
        print("[警告] 未配置 DEEPSEEK_API_KEY，请在 .env 文件中填写后重启服务")


if __name__ == "__main__":
    # 可用的命令映射
    commands = {
        "build": build_knowledge_base,
        "server": start_server,
        "ui": start_ui,
        "test": test,
    }

    if len(sys.argv) < 2:
        print("用法: python main.py [build|server|ui|test]")
        print("  build  - 构建非遗知识库")
        print("  server - 启动API服务")
        print("  ui     - 启动Streamlit界面")
        print("  test   - 测试各模块功能")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd in commands:
        try:
            commands[cmd]()
        except KeyboardInterrupt:
            # 用户按 Ctrl+C 主动停止：正常退出，不打 traceback
            print("\n已退出")
    else:
        print(f"未知命令: {cmd}")
        print("可用命令: build, server, ui, test")
