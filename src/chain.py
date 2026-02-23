"""
Phase 6 – RAG Chain (Retrieval-Augmented Generation)

Builds a LangChain chain that:
  1. Retrieves relevant chunks for a user question.
  2. Formats them into a prompt.
  3. Sends the prompt to the LLM.
  4. Returns a grounded answer with source citations.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from src.config import LLM_MODEL, LLM_TEMPERATURE, LLM_PROVIDER
from src.retriever import get_retriever


# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about the Python programming
language based ONLY on the provided tutorial documentation.

Context (retrieved from the Python Tutorial):
{context}

Instructions:
- Answer using ONLY the information in the context above.
- If the context does not contain enough information, say:
  "I don't have enough information in the tutorial to answer that."
- Include relevant Python code examples from the context when appropriate.
- At the end of your answer, list the source document(s) you used in the format:
  📄 Sources: <filename1>, <filename2>
"""

USER_PROMPT = "{question}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into a single context string."""
    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        heading = doc.metadata.get("heading", "")
        header = f"[{source}]"
        if heading:
            header += f" {heading}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Chain builder ────────────────────────────────────────────────────────────

def build_rag_chain():
    """
    Build and return the complete RAG chain (LCEL style).

    Returns a runnable that accepts {"question": str} and returns str.
    """
    retriever = get_retriever()

    provider = LLM_PROVIDER.lower() if isinstance(LLM_PROVIDER, str) else "local"

    # If OpenAI is explicitly selected, use the ChatOpenAI model.
    if provider == "openai":
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ])

        chain = (
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    # Local HuggingFace LLM that generates answers using transformers
    class LocalHFChain:
        def __init__(self, retriever, model_name: str):
            self.retriever = retriever
            self.model_name = model_name
            self._pipeline = None

        def _get_pipeline(self):
            """Lazy-load the HF pipeline to avoid loading model on import."""
            if self._pipeline is None:
                from transformers import pipeline
                print(f"Loading local LLM: {self.model_name} (this may take a moment)...")
                self._pipeline = pipeline(
                    "text2text-generation",
                    model=self.model_name,
                    device=-1,  # CPU; use device=0 for GPU
                    max_length=512,
                )
            return self._pipeline

        def invoke(self, question: str) -> str:
            docs = self.retriever.invoke(question)
            context = _format_docs(docs) if docs else "(no retrieved context)"
            sources = ", ".join(sorted({d.metadata.get("source", "unknown") for d in docs})) if docs else "none"

            # Build a prompt for the local model
            prompt_text = (
                "Answer the question using ONLY the context below. "
                "If the context does not contain enough information, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            pipe = self._get_pipeline()
            result = pipe(prompt_text, max_new_tokens=256, do_sample=False)[0]["generated_text"]

            return f"{result.strip()}\n\n📄 Sources: {sources}"

    return LocalHFChain(retriever, LLM_MODEL)


# ── Convenience wrapper ──────────────────────────────────────────────────────

def ask(question: str) -> str:
    """One-shot helper: build chain and answer a question."""
    chain = build_rag_chain()
    try:
        return chain.invoke(question)
    except Exception as e:
        # Fall back to returning retrieved context so the app remains usable
        from src.retriever import get_retriever
        import traceback

        retriever = get_retriever()
        try:
            docs = retriever.invoke(question)
        except Exception:
            docs = []

        context = _format_docs(docs) if docs else "(no retrieved context)"
        sources = ", ".join(sorted({d.metadata.get("source", "unknown") for d in docs})) if docs else "none"

        tb = traceback.format_exception_only(type(e), e)
        return (
            "⚠️ LLM request failed (falling back to retrieved context):\n"
            f"Error: {''.join(tb).strip()}\n\n"
            "--- Retrieved context (for debugging) ---\n\n"
            f"{context}\n\n📄 Sources: {sources}"
        )


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich import print as rprint
    from rich.markdown import Markdown

    q = "How do I use list comprehensions in Python?"
    rprint(f"[bold]Q:[/bold] {q}\n")
    answer = ask(q)
    rprint(Markdown(answer))
