"""
Query Rewrite Layer: 曖昧なユーザークエリを複数の検索クエリに展開する。
"""
import json
import re
from llm_provider import get_provider

_SYSTEM = """あなたは全文検索エンジン向けのクエリ最適化AIです。
ユーザーの質問を、ドキュメント検索に適したキーワード群に変換します。

【重要】あなたの仕事は「情報を検索するためのキーワード文字列のリスト」を返すことです。
実際の情報や回答を生成することは禁止です。

出力形式: JSON文字列配列のみ（マークダウン・説明文・コードブロック禁止）

例1:
入力: 「休みっていつ？」
出力: ["定休日", "営業時間", "祝日営業", "年末年始休業"]

例2:
入力: 「最新の求人のおすすめって何ですか？」
出力: ["求人情報", "募集要項", "採用情報", "仕事紹介"]

例3:
入力: 「料金を教えて」
出力: ["料金", "価格", "費用", "プラン"]
"""


def _extract_string_array(raw: str) -> list[str] | None:
    """生テキストからJSON文字列配列を抽出する。"""
    text = raw.strip()
    # マークダウンコードブロックを除去
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    try:
        parsed = json.loads(text)
        # 文字列配列かどうか確認
        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
            return parsed
    except Exception:
        pass
    return None


class QueryRewriter:
    def __init__(self):
        self._provider = get_provider("gemini")  # 低コスト処理はGeminiを使用

    def rewrite(self, query: str) -> list[str]:
        """
        クエリを展開して検索クエリリストを返す。
        LLM呼び出しに失敗した場合は元クエリのみを返す（フォールバック）。
        """
        try:
            raw = self._provider.complete(
                system=_SYSTEM,
                user=query,
                max_tokens=256,
            )
            queries = _extract_string_array(raw)
            if queries:
                if query not in queries:
                    queries = [query] + queries
                return queries[:5]
        except Exception:
            pass
        return [query]
