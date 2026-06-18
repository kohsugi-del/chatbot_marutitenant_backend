# rag_core.py
# 埋め込み・インデクシング: OpenAI text-embedding-3-small
# 回答生成: Anthropic Claude claude-sonnet-4-6

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import anthropic
from typing import List, Dict
from supabase_writer import save_to_supabase

# ============ OpenAI（埋め込みのみ）============
openai_client = OpenAI()

# ============ Anthropic（回答生成）============
anthropic_client = anthropic.Anthropic()

# デフォルトシステムプロンプト（テナント未設定の場合に使用）
DEFAULT_SYSTEM_PROMPT = """あなたは専用の案内チャットボットです。
以下の資料だけを根拠に回答してください。
推測や一般論は書かないでください。
緊急事態は即座に安全な場所に移動し、担当窓口へ電話するよう案内してください。"""


# ============ 1. Web ============
def load_web_urls(urls: List[str]) -> List[Dict]:
    docs = []
    for url in urls:
        print(f"📘 Web取得中: {url}")
        html = requests.get(url, timeout=20).text
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        docs.append({"source": url, "text": text})
    return docs


# ============ 2. PDF ============
def load_pdfs(paths: List[str]) -> List[Dict]:
    docs = []
    for pdf in paths:
        print(f"📕 PDF読み込み中: {pdf}")
        reader = PdfReader(pdf)
        txt = ""
        for page in reader.pages:
            txt += (page.extract_text() or "") + "\n"
        docs.append({"source": pdf, "text": txt})
    return docs


# ============ 3. Chunk ============
def chunk_docs(
    documents: List[Dict],
    max_chunks: int | None = None,
) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )

    chunks = []
    for d in documents:
        for chunk in splitter.split_text(d["text"]):
            chunks.append({
                "source": d["source"],
                "text": chunk,
            })
            if max_chunks and len(chunks) >= max_chunks:
                return chunks

    return chunks


# ============ 4. Embedding（OpenAI・変更なし）============
def embed_batch(texts: List[str]):
    res = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in res.data]


# ============ 5. インデクシング ============
def build_index(
    web_urls: List[str] | None = None,
    pdf_paths: List[str] | None = None,
    max_chunks: int = 50,
) -> int:
    web_urls = web_urls or []
    pdf_paths = pdf_paths or []

    docs = []
    if web_urls:
        docs.extend(load_web_urls(web_urls))
    if pdf_paths:
        docs.extend(load_pdfs(pdf_paths))

    if not docs:
        return 0

    chunks = chunk_docs(docs, max_chunks=max_chunks)
    texts = [c["text"] for c in chunks]

    print(f"🧩 chunks: {len(texts)}")

    embeddings = embed_batch(texts)

    for chunk, emb in zip(chunks, embeddings):
        save_to_supabase(
            content=chunk["text"],
            embedding=emb,
            source=chunk["source"],
        )

    return len(chunks)


# ============ 6. 回答生成（Claude・プロンプトキャッシュ対応）============
def answer(query: str, retrieved_docs: list, system_prompt: str | None = None) -> str:
    """
    retrieved_docs: [
      ({"text": "...", "source": "..."}, similarity),
      ...
    ]
    system_prompt: テナント固有のプロンプト。Noneの場合はデフォルトを使用。
    """
    if not retrieved_docs:
        context = "該当する情報は見つかりませんでした。"
    else:
        context = "\n\n".join(
            d["text"] for d, _ in retrieved_docs
        )

    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"# 資料\n{context}\n\n# 質問\n{query}\n\n# 回答（日本語・簡潔）",
            }
        ],
    )

    return response.content[0].text
