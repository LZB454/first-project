"""输出边界校验，防止模型越权代写主播话术（移植自 project2）。

观众弹幕生成后必须通过 validated_audience() 校验：
- 不包含"教主播怎么说"的越界内容
- 每一行都是"观众名：弹幕内容"格式，且为 2~4 条
校验不通过时后端直接返回错误提示，要求模型重新生成，而不是把越界内容放给用户。
"""
import re

# 越界模式：出现这些表述说明模型在代写主播话术，必须拦截
AUDIENCE_FORBIDDEN_PATTERNS = [
    r"主播[可以可应最好需要].{0,8}(说|回答|回复|介绍)",
    r"你(可以|应该|最好|需要).{0,8}(说|回答|回复)",
    r"建议.{0,6}(说|回答|回复)",
    r"(?:标准|推荐|参考)(?:主播)?话术",
    r"(?:回复|回答)(?:模板|示例)",
    r"可以这样说",
    r"照着.{0,4}说",
    r"^主播[：:]",
]

# 不允许出现的"观众名"（这些是系统角色，不是直播间观众）
FORBIDDEN_ROLE_NAMES = {"建议", "提示", "系统", "助手"}


def contains_forbidden_host_script(text: str) -> bool:
    """检查文本中是否出现代写主播话术的越界内容"""
    return any(
        re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        for pattern in AUDIENCE_FORBIDDEN_PATTERNS
    )


def validated_audience(text: str) -> str | None:
    """
    校验模型生成的观众弹幕

    校验规则：
    1. 不能包含越界内容（代写主播话术）
    2. 每一条非空行都必须是"观众名：弹幕内容"格式
    3. 弹幕条数必须在 2~4 条之间
    4. 观众名不能是系统角色名

    返回:
        通过校验时返回规范化后的弹幕文本（多行，行首去掉" -"前缀），
        任何一条不合格返回 None（整体降级，避免局部泄露）。
    """
    cleaned = text.strip()
    if not cleaned or contains_forbidden_host_script(cleaned):
        return None

    lines = [line.strip(" -") for line in cleaned.splitlines() if line.strip()]
    if not 2 <= len(lines) <= 4:
        return None
    for line in lines:
        match = re.fullmatch(r"([^：:]{1,20})[：:](.+)", line)
        if not match or "主播" in match[1] or match[1] in FORBIDDEN_ROLE_NAMES:
            return None
    return "\n".join(lines)
