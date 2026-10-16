"""在线评测 15 组场景输入，统计对话连贯性与复盘要点覆盖度。

运行：
    python tests/evaluate_cases.py

前置条件：
    1. .env 已配置 DEEPSEEK_API_KEY（真实调用云端 LLM，会产生 token 消耗）
    2. 已执行 python main.py build（构建 FAISS 索引）

流程：
    每组用例：逐轮输入主播台词 → 真实调用 RAG+LLM 生成观众弹幕 → 复盘评分，
    逐轮/逐用例计算量化指标，结果打印到终端并写入：
    - tests/results.json            原始逐轮数据（可复核）
    - tests/evaluation_results.md   汇总评测报告

指标说明：
    对话连贯性（每轮，0~1）：
        format_ok    弹幕格式合规（通过 validated_audience 三层校验）
        continuation 主题承接度（主播发言与观众弹幕的词面重合度，0~1）
        no_repeat    不重复历史弹幕（本轮弹幕未在前几轮出现过）
    复盘要点覆盖度（每组，0~1）：
        dims_present 勾选的评分维度全部出现在复盘报告中
        findings_ok  各维度 finding 非空且非占位（占位=模型未返回）
        topic_related 复盘内容命中用例主题关键词
"""
import json
import re
import sys
from pathlib import Path

# 把项目根目录加入 Python 路径，直接复用项目内部函数（不走 HTTP）
sys.path.insert(0, str(Path(__file__).parents[1]))

# Windows 控制台默认 GBK，报告里的 ✓ 等符号打印会报错，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag.retriever import generate_audience_reply, generate_review
from core.safety import validated_audience

TESTS_DIR = Path(__file__).parent
CHAT_MAX_TOKENS = 512    # 观众弹幕很短，限制长度可加快评测
REVIEW_MAX_TOKENS = 2048


# ==================== 词面重合度（主题承接度） ====================
def _tokens(text: str) -> list:
    """分词口径与知识库一致：英文/数字词 + 中文单字 + 中文二元组。"""
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+|[一-鿿]", lowered)
    chinese = [token for token in words if re.fullmatch(r"[一-鿿]", token)]
    bigrams = ["".join(chinese[i:i + 2]) for i in range(max(0, len(chinese) - 1))]
    return words + bigrams


def lexical_overlap(query: str, text: str) -> float:
    """查询 token 在文本中的多集命中比例，区间 [0,1]，值越大承接越紧。"""
    query_tokens = _tokens(query)
    text_tokens = _tokens(text)
    if not query_tokens:
        return 0.0
    remaining = list(text_tokens)
    hits = 0
    for token in query_tokens:
        if token in remaining:
            remaining.remove(token)
            hits += 1
    return hits / len(query_tokens)


def _parse_lines(text: str) -> list:
    """按行解析"观众名：内容"，解析失败的行直接忽略。"""
    lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
    parsed = []
    for line in lines:
        match = re.fullmatch(r"([^：:]{1,20})[：:](.+)", line)
        if match:
            parsed.append(line)
    return parsed


# ==================== 单组用例评测 ====================
def _run_case(case: dict) -> dict:
    """跑完一组用例：逐轮对话 + 复盘，返回全部量化数据。"""
    history = []          # 传给 LLM 的对话历史 [{role, content}]
    round_rows = []       # 每轮指标
    prev_lines = set()    # 已出现过的弹幕（查重）

    for host_text in case["rounds"]:
        row = {"round": len(round_rows) + 1, "host": host_text}
        try:
            reply = generate_audience_reply(
                host_text,
                category=case["category"],
                history=history,
                history_len=5,
                max_tokens=CHAT_MAX_TOKENS,
                logs=None,  # 评测不需要逐条日志，只打印到控制台
            )
            lines = _parse_lines(reply)
            format_ok = validated_audience(reply) is not None
            continuation = lexical_overlap(host_text, reply)
            no_repeat = bool(lines) and all(line not in prev_lines for line in lines)
            prev_lines.update(lines)
            row.update({
                "reply": reply,
                "format_ok": int(format_ok),
                "continuation": round(continuation, 4),
                "no_repeat": int(no_repeat),
            })
        except Exception as exc:
            # LLM 调用失败（如未配 Key）：该轮全部记 0 并保留错误信息
            row.update({
                "reply": "",
                "error": f"{type(exc).__name__}: {exc}",
                "format_ok": 0, "continuation": 0.0, "no_repeat": 0,
            })
        history.append({"role": "user", "content": host_text})
        if row.get("reply"):
            history.append({"role": "assistant", "content": row["reply"]})
        round_rows.append(row)

    # 复盘评分 + 覆盖度
    review_row = {"error": None, "review": None}
    try:
        review = generate_review(
            case["category"],
            history=history,
            dimensions=case["dimensions"],
            max_tokens=REVIEW_MAX_TOKENS,
            logs=None,
        )
        review_row["review"] = review
    except Exception as exc:
        review_row["error"] = f"{type(exc).__name__}: {exc}"

    coverage = _review_coverage(review_row["review"], case)
    return {"case": case, "rounds": round_rows, "review": review_row, "coverage": coverage}


def _review_coverage(review: dict, case: dict) -> dict:
    """复盘要点覆盖度：维度完整性 / 要点质量 / 主题相关性，各 0~1。"""
    requested = case["dimensions"]
    keywords = case.get("topic_keywords", [])
    if review is None:
        return {"dims_present": 0, "findings_ok": 0.0, "topic_related": 0,
                "model_source": 0, "score": 0.0}

    dims = review.get("dimensions", [])
    # 1. 勾选的维度是否全部出现在报告里
    present = [d.get("dimension") for d in dims]
    dims_present = int(all(key in present for key in requested))
    # 2. 每个维度 finding 非空且非占位（"占位/未返回"字样说明是系统补的假分数）
    findings = [str(d.get("finding", "")) for d in dims]
    findings_ok = sum(
        1 for text in findings if text and "占位" not in text and "未返回" not in text
    ) / len(findings) if findings else 0.0
    # 3. 复盘内容是否命中用例主题关键词（边界用例无关键词则不考核）
    review_text = " ".join(findings + [str(item) for item in review.get("issues", [])])
    topic_related = (
        int(any(keyword in review_text for keyword in keywords)) if keywords else 1
    )
    model_source = int(review.get("review_source") == "model")
    return {
        "dims_present": dims_present,
        "findings_ok": round(findings_ok, 4),
        "topic_related": topic_related,
        "model_source": model_source,
        "score": round((dims_present + findings_ok + topic_related) / 3, 4),
    }


# ==================== 汇总统计 ====================
def _case_continuity(rows: list) -> float:
    """单组用例的对话连贯性：各轮 (format + 承接度 + 不重复) / 3 的平均值。"""
    if not rows:
        return 0.0
    return round(sum(
        (row["format_ok"] + row["continuation"] + row["no_repeat"]) / 3
        for row in rows
    ) / len(rows), 4)


def _summarize(case_rows: list) -> dict:
    """把若干组用例的指标汇总成通过率。"""
    total = len(case_rows)
    if not total:
        return {}
    continuity_scores = [row["case_continuity"] for row in case_rows]
    coverage_scores = [row["coverage"]["score"] for row in case_rows]
    return {
        "cases": total,
        "continuity_mean": round(sum(continuity_scores) / total, 4),
        "coverage_mean": round(sum(coverage_scores) / total, 4),
        "format_rate": round(sum(
            all(r["format_ok"] for r in row["rounds"]) for row in case_rows
        ) / total, 4),
        "review_model_rate": round(sum(
            row["coverage"]["model_source"] for row in case_rows
        ) / total, 4),
    }


def run() -> dict:
    cases = json.loads((TESTS_DIR / "test_cases.json").read_text(encoding="utf-8"))
    print(f"开始评测 {len(cases)} 组场景输入（真实调用云端 LLM，预计需要几分钟）...\n")
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} {case['category']}·{case['expected']} ...", end=" ", flush=True)
        row = _run_case(case)
        row["case_continuity"] = _case_continuity(row["rounds"])
        rows.append(row)
        errors = sum(1 for r in row["rounds"] if r.get("error"))
        print(f"连贯性={row['case_continuity']:.2f} 复盘覆盖={row['coverage']['score']:.2f}"
              f" 轮次错误={errors}")

    summary = _summarize(rows)
    grouped_expected, grouped_category = {}, {}
    for row in rows:
        grouped_expected.setdefault(row["case"]["expected"], []).append(row)
        grouped_category.setdefault(row["case"]["category"], []).append(row)

    return {
        "total_cases": len(cases),
        "summary": summary,
        "grouped_by_expected": {
            key: _summarize(group) for key, group in sorted(grouped_expected.items())
        },
        "grouped_by_category": {
            key: _summarize(group) for key, group in sorted(grouped_category.items())
        },
        "cases": rows,
    }


# ==================== 生成 Markdown 报告 ====================
def _pct(value: float) -> str:
    return f"{value:.0%}"


def _markdown(result: dict) -> str:
    s = result["summary"]
    lines = [
        "# 15 组场景在线评测结果",
        "",
        "运行命令：`python tests/evaluate_cases.py`（真实调用 DeepSeek 云端 LLM）",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 对话连贯性（平均分，满分1） | {s['continuity_mean']:.2f} |",
        f"| 弹幕格式合规率（全部轮次合规的用例占比） | {_pct(s['format_rate'])} |",
        f"| 复盘要点覆盖度（平均分，满分1） | {s['coverage_mean']:.2f} |",
        f"| 复盘为模型真实评价的占比 | {_pct(s['review_model_rate'])} |",
        "",
        "## 按用例类型分组",
        "",
        "| 类型 | 用例数 | 连贯性 | 格式合规率 | 复盘覆盖度 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, group in result["grouped_by_expected"].items():
        lines.append(
            f"| {key} | {group['cases']} | {group['continuity_mean']:.2f} "
            f"| {_pct(group['format_rate'])} | {group['coverage_mean']:.2f} |"
        )
    lines += ["", "## 按品类分组", "",
              "| 品类 | 用例数 | 连贯性 | 格式合规率 | 复盘覆盖度 |",
              "|---|---:|---:|---:|---:|"]
    for key, group in result["grouped_by_category"].items():
        lines.append(
            f"| {key} | {group['cases']} | {group['continuity_mean']:.2f} "
            f"| {_pct(group['format_rate'])} | {group['coverage_mean']:.2f} |"
        )
    lines += ["", "## 逐用例明细", "",
              "| 用例 | 品类 | 类型 | 连贯性 | 复盘覆盖 | 维度完整 | 要点非占位 | 主题相关 |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in result["cases"]:
        c = row["coverage"]
        lines.append(
            f"| {row['case']['id']} | {row['case']['category']} | {row['case']['expected']} "
            f"| {row['case_continuity']:.2f} | {c['score']:.2f} | {'✓' if c['dims_present'] else '✗'} "
            f"| {c['findings_ok']:.0%} | {'✓' if c['topic_related'] else '✗'} |"
        )
    lines += [
        "",
        "## 指标口径",
        "",
        "- 对话连贯性（每轮取三项平均）：弹幕格式合规 / 主题承接度（主播发言与弹幕的词面重合度）/ 不重复历史弹幕",
        "- 复盘要点覆盖度（每组取三项平均）：勾选维度全部出现 / 各维度 finding 非占位 / 复盘内容命中主题关键词",
        "- 逐轮原始数据见 `tests/results.json`",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    result = run()
    (TESTS_DIR / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = _markdown(result)
    (TESTS_DIR / "evaluation_results.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"原始数据已写入: {TESTS_DIR / 'results.json'}")
