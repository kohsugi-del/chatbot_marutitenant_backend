"""
クウェスト合同会社 Reranker 比較テスト
=======================================
同じクエリを「Hybrid Search の順序のまま」と「Reranker適用後」で比較し、
どのドキュメントが何点を付けられたか・順位がどう変わったかを表示します。

実行方法:
  cd D:\\qwst_git\\qwst-plus\\chatbot_marutitenant\\chatbot_backend
  python test_reranker_qwest.py

結果は _reranker_result_qwest.txt にも保存されます。
"""
import os
import sys
import io
from dotenv import load_dotenv

load_dotenv()

from vector_search import search
from reranker import rerank

TENANT_ID = os.getenv("TEST_TENANT_ID", "b7a5fa4c-6dc1-40e2-aa48-d67db2c1e9bb")
TOP_K = 10        # Hybrid Searchで取得する件数
RERANKER_TOP_N = 5  # Rerankerで絞り込む件数

QUERIES = [
    "代表者を教えてください",
    "QWEST",
    "クウェスト",
    "電話番号を教えてください",
    "何時まで営業していますか",
    "旭川ガス燃料",
]


def run() -> str:
    lines = []

    def p(s: str = ""):
        lines.append(s)
        print(s)

    p("=" * 60)
    p("  Reranker 比較テスト（クウェスト合同会社）")
    p(f"  Hybrid Search TOP_K={TOP_K}  →  Reranker TOP_N={RERANKER_TOP_N}")
    p("=" * 60)

    for query in QUERIES:
        p()
        p("■ クエリ: " + query)
        p("-" * 50)

        # Hybrid Search 結果取得
        raw = search(query, top_k=TOP_K, tenant_id=TENANT_ID)
        if not raw:
            p("  (検索結果なし)")
            continue

        # タプル形式 → dicts + scores
        hs_docs = [doc for doc, _ in raw]
        hs_scores = [sim for _, sim in raw]

        p(f"  [Hybrid Search順位] ({len(hs_docs)}件)")
        for i, (doc, sim) in enumerate(zip(hs_docs, hs_scores)):
            snippet = doc["text"][:80].replace("\n", " ")
            p(f"    {i + 1}位 HSSocre={sim:.4f}  {snippet}")

        p()

        # Reranker 適用
        reranked_docs, rerank_scores = rerank(query, raw, top_n=RERANKER_TOP_N)

        p(f"  [Reranker適用後] (上位{RERANKER_TOP_N}件 / Claude採点)")
        for i, (doc, score) in enumerate(zip(reranked_docs, rerank_scores)):
            # Hybrid Search での元の順位を探す
            try:
                orig_rank = hs_docs.index(doc) + 1
            except ValueError:
                orig_rank = "?"
            arrow = ""
            if isinstance(orig_rank, int):
                diff = orig_rank - (i + 1)
                if diff > 0:
                    arrow = f"↑{diff}"
                elif diff < 0:
                    arrow = f"↓{abs(diff)}"
                else:
                    arrow = "→"

            snippet = doc["text"][:80].replace("\n", " ")
            p(f"    {i + 1}位 Rerank={score:.0f}/10 (元{orig_rank}位{arrow})  {snippet}")

        # Reranker で外れたドキュメントを表示
        dropped = [
            (i + 1, d)
            for i, d in enumerate(hs_docs)
            if d not in reranked_docs
        ]
        if dropped:
            p()
            p(f"  [Rerankerで除外されたドキュメント] ({len(dropped)}件)")
            for orig_rank, doc in dropped:
                snippet = doc["text"][:60].replace("\n", " ")
                p(f"    元{orig_rank}位  {snippet}")

        p()

    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    result = run()
    out_path = os.path.join(os.path.dirname(__file__), "_reranker_result_qwest.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n→ 結果を保存しました: {out_path}")
