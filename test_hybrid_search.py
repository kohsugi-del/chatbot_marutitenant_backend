"""
Hybrid Search 比較テスト
=========================
同じクエリを「旧: ベクター検索のみ」と「新: Hybrid Search」で実行し、
結果の違いを並べて表示します。

使い方:
  cd chatbot_backend
  python test_hybrid_search.py

Supabase に 20260615000001_hybrid_search.sql を適用済みであること。
"""

import os, sys, textwrap
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
from supabase_client import supabase

openai_client = OpenAI()

# ============================================================
# 検索関数
# ============================================================

def embed(text: str) -> list:
    res = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return res.data[0].embedding


def search_vector_only(query: str, top_k: int = 5, tenant_id: str | None = None) -> list:
    """旧: ベクター検索のみ (match_documents)"""
    args = {"query_embedding": embed(query), "match_count": top_k}
    if tenant_id:
        args["filter_tenant_id"] = tenant_id
    res = supabase.rpc("match_documents", args).execute()
    return res.data or []


def search_hybrid(query: str, top_k: int = 5, tenant_id: str | None = None) -> list:
    """新: Hybrid Search (hybrid_search_documents)"""
    args = {
        "query_embedding": embed(query),
        "query_text": query,
        "match_count": top_k,
    }
    if tenant_id:
        args["filter_tenant_id"] = tenant_id
    res = supabase.rpc("hybrid_search_documents", args).execute()
    return res.data or []


# ============================================================
# 表示ヘルパー
# ============================================================

def show_results(label: str, results: list, score_key: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if not results:
        print("  (結果なし)")
        return
    for i, row in enumerate(results, 1):
        score = row.get(score_key, row.get("similarity", 0))
        source = row.get("source", "")
        content = row.get("content", "")
        snippet = textwrap.shorten(content, width=80, placeholder="...")
        print(f"  [{i}] score={score:.4f}  source={source}")
        print(f"       {snippet}")


def run_test(query: str, tenant_id: str | None = None):
    print(f"\n{'#'*60}")
    print(f"  クエリ: 「{query}」")
    print(f"{'#'*60}")

    old = search_vector_only(query, top_k=5, tenant_id=tenant_id)
    new = search_hybrid(query, top_k=5, tenant_id=tenant_id)

    show_results("旧: ベクター検索のみ (match_documents)", old, "similarity")
    show_results("新: Hybrid Search (hybrid_search_documents)", new, "rrf_score")

    # 順位の変化を表示
    old_ids = [r.get("id") for r in old]
    new_ids = [r.get("id") for r in new]
    new_only = [i for i in new_ids if i not in old_ids]
    promoted = [
        i for i in new_ids
        if i in old_ids and new_ids.index(i) < old_ids.index(i)
    ]
    if new_only:
        print(f"\n  NEW で初登場 (ベクター検索には出なかった): {len(new_only)}件")
    if promoted:
        print(f"  NEW で順位上昇: {len(promoted)}件")


# ============================================================
# テストケース
# ============================================================

# テナントIDを環境変数から取得（なければNone = 全テナント対象）
TENANT_ID = os.getenv("TEST_TENANT_ID") or None

# ────────────────────────────────────────────────────────────
# ケース1: 固有名詞（社名・品番） → Hybrid が得意
#   ベクター検索は「旭川ガス」という文字列の完全一致が苦手
# ────────────────────────────────────────────────────────────
run_test("旭川ガス", tenant_id=TENANT_ID)

# ────────────────────────────────────────────────────────────
# ケース2: 自然な言い換え → ベクター・Hybrid ともに強い
#   "何時まで営業" ↔ "営業時間 9:00〜18:00"
# ────────────────────────────────────────────────────────────
run_test("何時まで営業していますか", tenant_id=TENANT_ID)

# ────────────────────────────────────────────────────────────
# ケース3: 具体的な施設名 → Hybrid が得意
#   "駐車場" という単語がドキュメントにあるかどうか
# ────────────────────────────────────────────────────────────
run_test("駐車場はありますか", tenant_id=TENANT_ID)

# ────────────────────────────────────────────────────────────
# ケース4: ガス漏れ（緊急ワード） → 両方で検証
# ────────────────────────────────────────────────────────────
run_test("ガス漏れ 対処方法", tenant_id=TENANT_ID)

print("\n" + "="*60)
print("  テスト完了")
print("="*60)
