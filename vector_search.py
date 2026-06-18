from supabase_client import supabase
from openai import OpenAI
from typing import Optional

client = OpenAI()


def embed_query(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return res.data[0].embedding


def search(query: str, top_k: int = 3, tenant_id: Optional[str] = None):
    """
    Hybrid Search: ベクター検索 + キーワード検索 (RRF統合)
    tenant_id を指定するとそのテナントのドキュメントのみ返す。
    """
    q_emb = embed_query(query)

    args: dict = {
        "query_embedding": q_emb,
        "query_text": query,
        "match_count": top_k,
    }
    if tenant_id:
        args["filter_tenant_id"] = tenant_id

    res = supabase.rpc(
        "hybrid_search_documents",
        args,
    ).execute()

    results = []
    for row in res.data:
        results.append((
            {
                "text": row["content"],
                "source": row.get("source", ""),
            },
            row["similarity"],  # ベクター類似度（信頼度スコアとして使用）
        ))

    return results
