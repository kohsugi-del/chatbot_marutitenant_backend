"""
Multi-LLM Agentic RAG Pipeline
================================
Query Rewrite → Hybrid Search → Reranker → Context Compression
→ Answer → Reflection → (再検索 → Final Answer)
"""
from vector_search import search
from reranker import rerank
from llm_provider import get_provider
from query_rewriter import QueryRewriter
from context_compressor import ContextCompressor
from reflection_agent import ReflectionAgent

_DEFAULT_SYSTEM = (
    "あなたは専用の案内チャットボットです。"
    "提供された資料だけを根拠に回答してください。"
    "資料に根拠がない場合は「お電話でご確認ください」と案内してください。"
)

_rewriter = QueryRewriter()
_compressor = ContextCompressor()
_reflector = ReflectionAgent()


def _multi_search(queries: list[str], top_k: int, tenant_id: str | None) -> list:
    """複数クエリで検索し、重複排除して返す。
    source が None/空の場合はテキスト先頭100文字をキーにする。
    """
    seen: set[str] = set()
    merged: list = []
    for q in queries:
        for doc, score in search(q, top_k=top_k, tenant_id=tenant_id):
            src = doc.get("source") or ""
            key = src if src else doc.get("text", "")[:100]
            if key not in seen:
                seen.add(key)
                merged.append((doc, score))
    return merged


def advanced_agentic_answer(
    question: str,
    tenant_id: str | None = None,
    system_prompt: str | None = None,
) -> dict:
    """
    Multi-LLM Agentic RAG で回答を生成する。

    Returns:
        {
            "answer": str,
            "queries": list[str],          # Query Rewriteで展開されたクエリ群
            "reflection": {
                "enough": bool,
                "reason": str,
                "need_more_search": bool,
            },
            "researched": bool,            # 再検索が実行されたか
            "references": [{"source": str}],
        }
    """
    provider = get_provider()
    system = system_prompt or _DEFAULT_SYSTEM

    # ── Step 1: Query Rewrite ────────────────────────────────
    queries = _rewriter.rewrite(question)

    # ── Step 2: Hybrid Search（複数クエリ・重複排除）────────
    raw = _multi_search(queries, top_k=10, tenant_id=tenant_id)

    # ── Step 3: Reranker ─────────────────────────────────────
    reranked, _ = rerank(question, raw, top_n=5)

    # ── Step 4: Context Compression ─────────────────────────
    compressed = _compressor.compress(question, reranked)

    # ── Step 5: Answer Generation ───────────────────────────
    draft = provider.complete(
        system=system,
        user=f"# 資料\n{compressed}\n\n# 質問\n{question}\n\n# 回答（日本語・簡潔）",
        max_tokens=2048,
    )

    # ── Step 6: Reflection ───────────────────────────────────
    reflection = _reflector.evaluate(question, draft, compressed)

    # ── Step 7: 再検索（必要な場合のみ）────────────────────
    researched = False
    final_answer = draft
    all_sources: list[str] = [doc.get("source", "") for doc, _ in raw]

    if reflection.need_more_search and reflection.additional_queries:
        researched = True
        extra_raw = _multi_search(
            reflection.additional_queries, top_k=5, tenant_id=tenant_id
        )
        combined = raw + [
            (doc, score) for doc, score in extra_raw
            if doc.get("source", "") not in set(all_sources)
        ]
        all_sources += [doc.get("source", "") for doc, _ in extra_raw]

        reranked2, _ = rerank(question, combined, top_n=5)
        compressed2 = _compressor.compress(question, reranked2)

        final_answer = provider.complete(
            system=system,
            user=f"# 資料\n{compressed2}\n\n# 質問\n{question}\n\n# 回答（日本語・簡潔）",
            max_tokens=2048,
        )

    return {
        "answer": final_answer,
        "queries": queries,
        "reflection": {
            "enough": reflection.enough,
            "reason": reflection.reason,
            "need_more_search": reflection.need_more_search,
        },
        "researched": researched,
        "references": [{"source": s} for s in dict.fromkeys(all_sources) if s],
    }
