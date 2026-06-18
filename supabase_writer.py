import os
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or ""
).strip()

_client = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_to_supabase(
    content: str,
    embedding,
    source: Optional[str] = None,
    title: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> dict:
    """documentsテーブルに1チャンクをINSERT"""
    emb = embedding if isinstance(embedding, list) else list(embedding)

    row: dict = {"content": content, "embedding": emb}
    if source:
        row["source"] = source
        row["source_url"] = source
    if title:
        row["title"] = title
    if tenant_id:
        row["tenant_id"] = tenant_id

    result = _client.table("documents").insert(row).execute()
    return result.data[0] if result.data else {}
