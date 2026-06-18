"""
Provider Layer: LLM呼び出しを抽象化する。
直接APIを呼ばず、このモジュール経由で統一する。
"""
from abc import ABC, abstractmethod
import anthropic
import os


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """システムプロンプトとユーザーメッセージを渡して応答テキストを返す。"""
        ...


class ClaudeProvider(LLMProvider):
    """Anthropic Claude (claude-sonnet-4-6) — 回答生成・Reflection用"""

    MODEL = "claude-sonnet-4-6"

    def __init__(self):
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self.MODEL,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class GeminiProvider(LLMProvider):
    """Google Gemini 2.5 Flash — Query Rewrite・Context Compression用（低コスト）"""

    MODEL = "gemini-2.5-flash"

    def __init__(self):
        from google import genai
        from google.genai import types as genai_types
        api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
        self._client = genai.Client(api_key=api_key)
        self._types = genai_types

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self._client.models.generate_content(
            model=self.MODEL,
            contents=user,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                thinking_config=self._types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text


# プロバイダーキャッシュ（名前→インスタンス）
_providers: dict[str, LLMProvider] = {}


def get_provider(name: str | None = None) -> LLMProvider:
    """プロバイダーを返す。name省略時は環境変数 LLM_PROVIDER（デフォルト: claude）を使用。"""
    provider_name = (name or os.getenv("LLM_PROVIDER", "claude")).lower()
    if provider_name not in _providers:
        if provider_name == "claude":
            _providers[provider_name] = ClaudeProvider()
        elif provider_name == "gemini":
            _providers[provider_name] = GeminiProvider()
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider_name!r}")
    return _providers[provider_name]
