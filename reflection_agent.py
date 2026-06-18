"""
Reflection Agent: 回答生成後に自己評価を行い、再検索の要否を判断する。
"""
import json
from dataclasses import dataclass, field
from llm_provider import get_provider

_SYSTEM = """あなたは回答品質の評価専門家です。
ユーザーの質問・参照した資料・生成された回答を受け取り、回答品質を評価してください。

評価基準:
1. 根拠があるか（資料に記載された情報のみを使っているか）
2. 情報不足がないか（質問に対して回答が不完全でないか）
3. 幻覚の可能性がないか（資料にない情報を回答に含めていないか）
4. 再検索が必要か（追加情報を取得すれば回答が改善できるか）

必ず以下のJSON形式のみで返してください。説明文は不要です。

{
  "enough": true または false,
  "reason": "評価の理由（1〜2文）",
  "need_more_search": true または false,
  "additional_queries": ["追加検索クエリ1", "追加検索クエリ2"]
}

- enough=false かつ need_more_search=true の場合、additional_queries に具体的な検索クエリを1〜3件入れる
- enough=true の場合、additional_queries は空配列にする
"""


@dataclass
class ReflectionResult:
    enough: bool
    reason: str
    need_more_search: bool
    additional_queries: list[str] = field(default_factory=list)


class ReflectionAgent:
    def __init__(self):
        self._provider = get_provider()

    def evaluate(self, query: str, answer: str, context: str) -> ReflectionResult:
        """
        回答を評価して ReflectionResult を返す。
        LLM呼び出し失敗時は「十分・再検索不要」として処理を続行する。
        """
        user_msg = (
            f"# ユーザーの質問\n{query}\n\n"
            f"# 参照した資料\n{context}\n\n"
            f"# 生成された回答\n{answer}\n\n"
            "上記を評価してください。"
        )

        try:
            raw = self._provider.complete(
                system=_SYSTEM,
                user=user_msg,
                max_tokens=512,
            )
            # Claudeがmarkdownコードブロックで返す場合を除去
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.rsplit("```", 1)[0].strip()
            data = json.loads(cleaned)
            return ReflectionResult(
                enough=bool(data.get("enough", True)),
                reason=str(data.get("reason", "")),
                need_more_search=bool(data.get("need_more_search", False)),
                additional_queries=list(data.get("additional_queries", [])),
            )
        except Exception:
            # 失敗時はそのまま回答を採用
            return ReflectionResult(
                enough=True,
                reason="評価スキップ（LLMエラー）",
                need_more_search=False,
            )
