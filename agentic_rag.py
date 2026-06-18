"""
Agentic RAG: Claude の tool_use による自律再検索
================================================
通常RAG: 検索1回 → 回答
Agentic: 検索 → Claudeが「まだ情報が足りない」と判断したら自分でクエリを作って再検索 → 十分な情報が集まったら回答
最大 AGENTIC_MAX_ITER 回まで追加検索を許可する。
"""
import anthropic
from vector_search import search
from reranker import rerank

_client = anthropic.Anthropic()

AGENTIC_MODEL = "claude-sonnet-4-6"
AGENTIC_MAX_ITER = 3
RERANKER_TOP_N = 5

# Claude に渡すツール定義（search_knowledge_base）
_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "社内知識ベースを検索して関連情報を取得します。"
        "情報が不足している場合や、複数の異なるトピックを調べる必要がある場合に使ってください。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "検索クエリ（具体的なキーワードや質問文）"},
            "reason": {"type": "string", "description": "この検索を行う理由（例：「電話番号を調べるため」）"},
        },
        "required": ["query", "reason"],
    },
}

_DEFAULT_PROMPT = (
    "あなたは専用の案内チャットボットです。"
    "提供された資料だけを根拠に回答してください。"
    "資料に根拠がない場合は「お電話でご確認ください」と案内してください。"
)


def _build_ctx(docs: list[dict]) -> str:
    if not docs:
        return "(資料なし)"
    return "\n\n".join(
        f"source: {d.get('source', '')}\n{d['text']}" for d in docs
    )


def agentic_answer(
    question: str,
    tenant_id: str | None = None,
    system_prompt: str | None = None,
) -> dict:
    """
    Agentic RAG で回答を生成する。

    Returns:
        {
            "answer": str,
            "search_log": [
                {"iteration": 0, "query": str, "reason": "初回検索", "hits": int},
                {"iteration": 1, "query": str, "reason": str, "hits": int},
                ...
            ],
            "total_iterations": int,  # 追加検索回数（初回を除く）
            "references": [{"source": str}],
        }
    """
    prompt = system_prompt or _DEFAULT_PROMPT

    # ── 初回検索（常に実行）─────────────────────────────────
    raw = search(question, top_k=20, tenant_id=tenant_id)
    reranked, _ = rerank(question, raw, top_n=RERANKER_TOP_N)

    # source をキーに重複排除しながら全ドキュメントを管理
    all_docs: dict[str, dict] = {d.get("source", f"__{i}"): d for i, d in enumerate(reranked)}
    search_log = [{"iteration": 0, "query": question, "reason": "初回検索", "hits": len(reranked)}]

    # ── 最初のユーザーメッセージ ────────────────────────────
    messages = [
        {
            "role": "user",
            "content": (
                f"# 取得済み資料\n{_build_ctx(reranked)}\n\n"
                f"# 質問\n{question}\n\n"
                "資料で回答できない情報がある場合は search_knowledge_base ツールを使ってください。"
                "すべての情報があれば直接回答してください。"
            ),
        }
    ]

    iterations = 0
    final_answer = ""

    # ── Agentic ループ ───────────────────────────────────────
    while True:
        response = _client.messages.create(
            model=AGENTIC_MODEL,
            max_tokens=2048,
            system=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
            tools=[_TOOL],
            messages=messages,
        )

        # ツール呼び出しなし → 最終回答として取り出して終了
        if response.stop_reason != "tool_use":
            for block in response.content:
                if block.type == "text":
                    final_answer = block.text
            break

        # ── ツール呼び出しを処理 ────────────────────────────
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            iterations += 1
            q = block.input.get("query", "")
            reason = block.input.get("reason", "")

            # 追加検索（Hybrid Search + Reranker）
            add_raw = search(q, top_k=10, tenant_id=tenant_id)
            add_reranked, _ = rerank(q, add_raw, top_n=3)

            for d in add_reranked:
                all_docs[d.get("source", f"__{len(all_docs)}")] = d

            search_log.append({"iteration": iterations, "query": q, "reason": reason, "hits": len(add_reranked)})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _build_ctx(add_reranked),
            })

        messages.append({"role": "user", "content": tool_results})

        # 上限到達 → ツール指定なしで強制最終回答
        if iterations >= AGENTIC_MAX_ITER:
            final_resp = _client.messages.create(
                model=AGENTIC_MODEL,
                max_tokens=2048,
                system=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
            )
            for block in final_resp.content:
                if block.type == "text":
                    final_answer = block.text
            break

    return {
        "answer": final_answer,
        "search_log": search_log,
        "total_iterations": iterations,
        "references": [{"source": src} for src in all_docs],
    }
