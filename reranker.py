"""
Reranker: Claude Haikuによる検索結果の関連度再評価
"""
import json
import re
import anthropic

_client = anthropic.Anthropic()

RERANKER_MODEL = "claude-haiku-4-5-20251001"


def rerank(query: str, docs: list, top_n: int = 5) -> tuple[list, list[float]]:
    """
    docs: [{"text": "...", "source": "..."}, ...]  または
          [({"text": ..., "source": ...}, similarity), ...]  (vector_search.pyの形式)

    戻り値: (再ランク済みの上位top_n件のdocリスト, Claudeスコアリスト)
    Reranker失敗時は元の順序でsliceして返す。
    """
    # タプル形式 (doc, similarity) を正規化
    normalized = []
    is_tuple_format = docs and isinstance(docs[0], tuple)
    if is_tuple_format:
        normalized = [d for d, _ in docs]
    else:
        normalized = list(docs)

    if len(normalized) <= 1:
        return normalized[:top_n], [10.0] * len(normalized[:top_n])

    doc_list = "\n\n---\n\n".join(
        f"[{i + 1}] {d['text'][:400]}" for i, d in enumerate(normalized)
    )

    prompt = f"""ユーザーの質問に対して、各ドキュメントの関連度を0〜10で評価してください。

質問: {query}

ドキュメント一覧:
{doc_list}

JSONのみを返してください（説明不要）:
{{"scores": [{{"index": 1, "score": 8}}, {{"index": 2, "score": 3}}, ...]}}"""

    try:
        res = _client.messages.create(
            model=RERANKER_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        text = res.content[0].text
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return normalized[:top_n], []

        parsed = json.loads(m.group())
        score_map = {s["index"] - 1: float(s["score"]) for s in parsed["scores"]}

        ranked = sorted(
            enumerate(normalized),
            key=lambda x: score_map.get(x[0], 0.0),
            reverse=True,
        )[:top_n]

        return (
            [d for _, d in ranked],
            [score_map.get(i, 0.0) for i, _ in ranked],
        )

    except Exception:
        return normalized[:top_n], []
