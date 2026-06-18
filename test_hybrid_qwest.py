"""
クウェスト合同会社 Hybrid Search 比較テスト
=============================================
旧: match_documents（ベクター検索のみ）
新: hybrid_search_documents（ベクター + キーワード RRF統合）

実行前に Supabase SQL Editor で
  supabase/migrations/20260615000001_hybrid_search.sql
を実行してください。

使い方:
  cd chatbot_backend
  python test_hybrid_qwest.py
"""

from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI
from supabase_client import supabase
import textwrap

openai_client = OpenAI()
TENANT_ID = "b7a5fa4c-6dc1-40e2-aa48-d67db2c1e9bb"


def embed(text: str) -> list:
    return openai_client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding


def search_old(query: str, top_k: int = 5) -> list:
    """旧: ベクター検索のみ"""
    res = supabase.rpc("match_documents", {
        "query_embedding": embed(query),
        "match_count": top_k,
        "filter_tenant_id": TENANT_ID,
    }).execute()
    return res.data or []


def search_new(query: str, top_k: int = 5) -> list:
    """新: Hybrid Search"""
    res = supabase.rpc("hybrid_search_documents", {
        "query_embedding": embed(query),
        "query_text": query,
        "match_count": top_k,
        "filter_tenant_id": TENANT_ID,
    }).execute()
    return res.data or []


def show(label: str, results: list, score_key: str, lines: list):
    lines.append(f"\n{'─'*56}")
    lines.append(f"  {label}")
    lines.append(f"{'─'*56}")
    if not results:
        lines.append("  (結果なし)")
        return
    for i, row in enumerate(results, 1):
        score = float(row.get(score_key) or row.get("similarity") or 0)
        snippet = textwrap.shorten(
            (row.get("content") or "").replace("\n", " "), width=90, placeholder="…"
        )
        lines.append(f"  [{i}] score={score:.5f}  {snippet}")


def compare(query: str, note: str = "") -> list:
    lines = []
    lines.append(f"\n{'═'*56}")
    lines.append(f"  クエリ: 「{query}」")
    if note:
        lines.append(f"  ※ {note}")
    lines.append(f"{'═'*56}")

    old = search_old(query)
    new = search_new(query)

    show("旧: ベクター検索のみ (match_documents)", old, "similarity", lines)
    show("新: Hybrid Search  (hybrid_search_documents)", new, "rrf_score", lines)

    old_ids = [r.get("id") for r in old]
    new_ids = [r.get("id") for r in new]
    new_only = [x for x in new_ids if x not in old_ids]
    moved_up = [x for x in new_ids if x in old_ids and new_ids.index(x) < old_ids.index(x)]
    if new_only:
        lines.append(f"\n  >> 新たにヒット（ベクター検索には出なかった）: {len(new_only)} 件")
    if moved_up:
        lines.append(f"  >> 順位が上昇したドキュメント: {len(moved_up)} 件")
    if not new_only and not moved_up:
        lines.append(f"\n  >> 結果に差異なし（このクエリは両手法で同様）")
    return lines


# ============================================================
# クウェスト合同会社向けテストケース
# ============================================================
all_lines = ["=" * 56, "  クウェスト合同会社 Hybrid Search 比較レポート", "=" * 56]

# ① 固有名詞（社名）→ キーワード検索が効く典型例
all_lines += compare(
    "QWEST",
    note="社名の英字表記。ベクター検索は意味的距離で探すため英字固有名詞が苦手"
)

# ② 固有名詞（社名・日本語）
all_lines += compare(
    "クウェスト",
    note="日本語社名。文字列として存在するかどうかがポイント"
)

# ③ 実在するクライアント名（制作実績ページに記載）
all_lines += compare(
    "旭川ガス燃料",
    note="制作実績ページに掲載されている取引先名。完全一致系の典型"
)

# ④ 自然な言い換え → ベクターが得意、Hybridも同等
all_lines += compare(
    "何時まで営業していますか",
    note="「受付時間 9:00〜18:00」という表現と意味的に同じ。ベクター検索の得意分野"
)

# ⑤ 連絡先（電話番号ページから）
all_lines += compare(
    "電話番号を教えてください",
    note="問い合わせページに記載あり。意味検索・キーワード両方が効く"
)

# ⑥ 代表者名（会社概要ページから）
all_lines += compare(
    "代表者",
    note="会社概要に「西 一哉」「北田 宏幸」が記載。固有名詞ではなく役職名なのでどちらが強い？"
)

all_lines.append(f"\n{'='*56}")
all_lines.append("  テスト完了")
all_lines.append("=" * 56)

# ファイルに書き出し
output = "\n".join(all_lines)
with open("_hybrid_result_qwest.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("完了: _hybrid_result_qwest.txt に結果を書き出しました")
