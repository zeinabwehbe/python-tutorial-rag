"""
Phase 4 – Embedding & Vector Store

Supports multiple embedding providers: HuggingFace, OpenAI, and Deepseek.
Embeds document chunks and persists them in a ChromaDB collection.
"""

from __future__ import annotations

import json
from typing import List
import requests

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument

from src.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION,
    VECTORSTORE_DIR,
    DEEPSEEK_API_URL,
    DEEPSEEK_API_KEY,
)
from src.chunker import Chunk


def _chunks_to_lc_documents(chunks: list[Chunk]) -> list[LCDocument]:
    """Convert our Chunk dataclass into LangChain Documents."""
    return [
        LCDocument(page_content=c.text, metadata=c.metadata)
        for c in chunks
    ]


class DeepseekEmbeddings:
    """Minimal wrapper to call Deepseek embeddings API.

    Expects DEEPSEEK_API_URL to point at the API base (e.g. https://api.deepseek.com)
    and that the embeddings endpoint accepts JSON {"input": [...], "model": "..."}
    and returns a structure similar to OpenAI: {"data":[{"embedding": [...]}, ...]}.
    """

    def __init__(self, api_url: str, api_key: str, model: str | None = None):
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set; cannot use Deepseek embeddings")
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _call_api(self, inputs: List[str]) -> List[List[float]]:
        # Try a few common embeddings endpoint paths in case the API root differs.
        candidate_paths = ["/v1/embeddings", "/embeddings", "/api/embeddings", "/v1/embedding"]
        payload = {"input": inputs}
        if self.model:
            payload["model"] = self.model

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err = None
        for path in candidate_paths:
            url = f"{self.api_url.rstrip('/')}{path}"
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            except Exception as e:
                last_err = e
                continue

            if resp.status_code != 200:
                last_err = RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text}")
                continue

            try:
                data = resp.json()
            except Exception as e:
                last_err = e
                continue

            # Expect OpenAI-like shape: {"data": [{"embedding": [...]}, ...]}
            if "data" not in data:
                last_err = RuntimeError(f"Unexpected Deepseek response shape from {url}: {data}")
                continue

            embeddings = [item.get("embedding") for item in data["data"]]
            return embeddings

        # If we reach here, all candidates failed
        raise RuntimeError(
            "Deepseek embedding request failed for all tried endpoints. "
            f"Last error: {last_err}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0]


def get_embedding_function():
    """Return the configured embedding model object for LangChain/Chroma.

    Supported providers: 'huggingface', 'openai', 'deepseek'.
    """
    provider = EMBEDDING_PROVIDER.lower()
    if provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if provider == "openai":
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if provider == "deepseek":
        return DeepseekEmbeddings(api_url=DEEPSEEK_API_URL, api_key=DEEPSEEK_API_KEY, model=EMBEDDING_MODEL)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def build_vectorstore(chunks: list[Chunk]) -> Chroma:
    """
    Embed all chunks and persist them in a ChromaDB collection.
    Returns the Chroma vectorstore instance.
    """
    lc_docs = _chunks_to_lc_documents(chunks)
    embeddings = get_embedding_function()

    vectorstore = Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(VECTORSTORE_DIR),
    )
    print(f"✅ Vector store created at {VECTORSTORE_DIR}  "
          f"({len(lc_docs)} chunks embedded)")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing persisted ChromaDB vector store."""
    embeddings = get_embedding_function()
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich import print as rprint
    from src.loader import load_documents
    from src.chunker import chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    vs = build_vectorstore(chunks)

    # Test similarity search
    query = "How do list comprehensions work?"
    results = vs.similarity_search(query, k=3)
    rprint(f"\n[bold]Query:[/bold] {query}\n")
    for i, r in enumerate(results, 1):
        rprint(f"[cyan]Result {i}[/cyan]  (source={r.metadata['source']})")
        rprint(r.page_content[:300], "…\n")
