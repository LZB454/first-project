"""
LLM 大语言模型统一管理模块（云端调用）

封装 OpenAI 兼容协议的对话接口（DeepSeek 等云端服务），
提供两种调用方式：
- chat_once():   一次性生成，返回完整回答
- chat_stream(): 流式生成，逐块返回回答内容

对上层调用方（RAG 检索链、复盘生成）完全透明：
上层只需要组装好 messages 列表并传入采样参数即可。

注意：api_key 为空时不会发起网络请求，而是抛出带指引的异常，
由 API 层把错误提示透传给前端显示。
"""
from typing import Iterator, List

from config.settings import LLM_CONFIG


class LLMError(Exception):
    """LLM 调用相关错误（未配置密钥、网络失败、服务端报错等）。"""


class LLMClient:
    """云端 LLM 客户端（OpenAI 兼容协议，默认 DeepSeek）。

    openai SDK 会在请求时带上 base_url 和 api_key，
    这里用懒加载方式创建客户端，避免服务启动时因缺 key 直接崩溃。
    """

    def __init__(self, config: dict = None):
        # 允许注入自定义配置（便于测试），默认使用 settings 的云端配置
        self.config = config or LLM_CONFIG

    def _client(self):
        """懒加载 OpenAI 客户端，并在 key 为空时给出清晰指引。"""
        if not self.config.get("api_key"):
            raise LLMError(
                "未配置 DEEPSEEK_API_KEY：请在项目根目录的 .env 文件中填写 "
                "自己的 API Key 后重启后端服务（参考 .env.example）"
            )
        from openai import OpenAI  # 延迟导入，未安装 openai 时其余功能不受影响

        return OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"],
        )

    def chat_once(
        self,
        messages: List[dict],
        temperature: float = 0.5,
        top_p: float = 0.5,
        max_tokens: int = 3072,
    ) -> str:
        """
        一次性生成回答（非流式）

        参数:
            messages:    标准对话消息列表 [{"role": "system"/"user"/"assistant", "content": "..."}]
            temperature: 采样温度，越低越确定
            top_p:       核采样概率阈值
            max_tokens:  单次生成最大 token 数

        返回:
            LLM 生成的完整回答文本
        """
        response = self._client().chat.completions.create(
            model=self.config["model"],
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: List[dict],
        temperature: float = 0.5,
        top_p: float = 0.5,
        max_tokens: int = 3072,
    ) -> Iterator[str]:
        """
        流式生成回答

        与 chat_once 的区别：stream=True，逐 token 返回文本片段，
        前端可以实时看到 AI 逐字生成的回答，首字响应更快。

        返回:
            生成器，每次 yield 一段回答文本（可能为多个 token）
        """
        response = self._client().chat.completions.create(
            model=self.config["model"],
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content


# 模块级单例：整个服务进程共用一个客户端
_client = LLMClient()


def get_llm() -> LLMClient:
    """获取 LLM 客户端单例"""
    return _client
