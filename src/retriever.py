"""
Phase 5 – Retrieval

Wraps the ChromaDB vector store in a LangChain retriever
with configurable top-k and search type (similarity / MMR).
"""

from __future__ import annotations

from langchain_core.vectorstores import VectorStoreRetriever

from src.config import RETRIEVER_K, SEARCH_TYPE
from src.embedder import load_vectorstore


def get_retriever(
    k: int = RETRIEVER_K,
    search_type: str = SEARCH_TYPE,
) -> VectorStoreRetriever:
    """
    Return a LangChain retriever backed by the persisted vector store.

    Args:
        k:           Number of chunks to retrieve.
        search_type: "similarity" or "mmr" (Maximal Marginal Relevance).
    """
    vs = load_vectorstore()
    return vs.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich import print as rprint

    retriever = get_retriever()
    query = "How do I handle exceptions in Python?"
    results = retriever.invoke(query)

    rprint(f"[bold]Query:[/bold] {query}\n")
    for i, doc in enumerate(results, 1):
        rprint(f"[cyan]Chunk {i}[/cyan]  source={doc.metadata['source']}  "
               f"heading={doc.metadata.get('heading', '')}")
        rprint(doc.page_content[:300], "…\n")
