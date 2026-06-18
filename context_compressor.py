"""
Context Compression Layer: 検索結果から質問に必要な部分だけを抽出・圧縮する。
大量チャンクをそのまま回答生成に渡さず、関連箇所のみに絞る。
"""
from llm_provider import get_provider

_SYSTEM = """あなたはテキスト圧縮の専門家です。
与えられた複数の資料から、ユーザーの質問に答えるために必要な情報だけを抽出してください。

ルール:
- 質問に直接関係する文章・数値・固有名詞のみ残す
- 不要な前置き・繰り返し・無関係な内容は削除する
- 資料に書かれていないことは絶対に追加しない
- 抽出した情報をそのまま出力する（要約・解釈は不要）
- source情報は保持する（"source: URL" の行はそのまま残す）
"""


class ContextCompressor:
    def __init__(self):
        self._provider = get_provider("gemini")  # 低コスト処理はGeminiを使用

    def compress(self, query: str, chunks: list[dict]) -> str:
        """
        質問に関連する部分だけを抽出して返す。

        chunks: [{"text": str, "source": str}, ...]
        戻り値: 圧縮済みコンテキスト文字列
        """
        if not chunks:
            return "(資料なし)"

        context = "\n\n---\n\n".join(
            f"source: {c.get('source', '')}\n{c['text']}" for c in chunks
        )

        user_msg = f"# 質問\n{query}\n\n# 資料\n{context}\n\n# 指示\n上記の資料から、質問に答えるために必要な情報だけを抽出してください。"

        try:
            return self._provider.complete(
                system=_SYSTEM,
                user=user_msg,
                max_tokens=2048,
            )
        except Exception:
            # 失敗時はそのまま返す（フォールバック）
            return context
