# RAG System – Project Plan

## 📌 Project Overview

Build a **Retrieval-Augmented Generation (RAG)** system that allows users to ask natural-language questions about the **Python 3.14 Tutorial** and receive accurate, context-grounded answers sourced from the local documentation.
### note about installing on the virtual environment
cd "c:\Users\Zayna\OneDrive\Desktop\RAG System"; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

### Source Documents

18 HTML pages mirrored from `docs.python.org/3/tutorial/`, covering:

| File | Topic |
|---|---|
| `appetite.html` | Whetting Your Appetite |
| `interpreter.html` | Using the Python Interpreter |
| `introduction.html` | An Informal Introduction to Python |
| `controlflow.html` | More Control Flow Tools |
| `datastructures.html` | Data Structures |
| `modules.html` | Modules |
| `inputoutput.html` | Input and Output |
| `errors.html` | Errors and Exceptions |
| `classes.html` | Classes |
| `stdlib.html` | Brief Tour of the Standard Library |
| `stdlib2.html` | Brief Tour of the Standard Library – Part II |
| `venv.html` | Virtual Environments and Packages |
| `floatingpoint.html` | Floating-Point Arithmetic |
| `interactive.html` | Interactive Input Editing and History |
| `appendix.html` | Appendix |
| `whatnow.html` | What Now? |
| `index.html` | Tutorial Index |
| `index-2.html` | Tutorial Index (alt) |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────┐
│  HTML Docs   │────▶│  Preprocessing│────▶│  Vector Store   │────▶│  RAG Chat │
│  (py_tutorial)│     │  & Chunking   │     │  (ChromaDB /    │     │  Interface│
└─────────────┘     └──────────────┘     │   FAISS)        │     └───────────┘
                                          └────────────────┘
```

---

## 🗂️ Steps

### Phase 1 – Environment Setup

| # | Task | Details |
|---|------|---------|
| 1.1 | **Create a Python virtual environment** | `python -m venv .venv` inside the `RAG System` folder |
| 1.2 | **Install core dependencies** | `pip install langchain langchain-community langchain-openai openai chromadb tiktoken beautifulsoup4 lxml unstructured rich nltk spacy` and run `python -m spacy download en_core_web_sm` |
| 1.3 | **Configure API keys** | Create a `.env` file with `OPENAI_API_KEY=sk-...` (or whichever LLM provider is chosen). Use `python-dotenv` to load it. |
| 1.4 | **Set up project structure** | See folder layout below |

#### Target Folder Layout

```
RAG System/
├── .env                   # API keys (git-ignored)
├── .gitignore
├── requirements.txt
├── PROJECT_PLAN.md        # ← this file
├── py_tutorial/           # source HTML documents (already present)
│   └── docs.python.org/...
├── src/
│   ├── __init__.py
│   ├── config.py          # central settings (paths, model names, chunk params)
│   ├── loader.py          # Phase 2 – document loading
│   ├── chunker.py         # Phase 3 – text splitting
│   ├── embedder.py        # Phase 4 – embedding + vector store
│   ├── retriever.py       # Phase 5 – retrieval logic
│   ├── chain.py           # Phase 6 – RAG chain / prompt template
│   └── app.py             # Phase 7 – user-facing interface (CLI / Gradio)
├── vectorstore/           # persisted ChromaDB / FAISS index
└── tests/
    ├── test_loader.py
    ├── test_chunker.py
    └── test_retriever.py
```

---

### Phase 2 – Document Loading & Cleaning

| # | Task | Details |
|---|------|---------|
| 2.1 | **Parse HTML files** | Use `BeautifulSoup` (or LangChain's `BSHTMLLoader` / `UnstructuredHTMLLoader`) to read each `.html` file from `py_tutorial/docs.python.org/3/tutorial/`. |
| 2.2 | **Extract text content** | Strip navigation bars, headers, footers, script/style tags. Keep only the main tutorial body. |
| 2.3 | **Attach metadata** | For every document, store: `source` (file path), `title` (page title), `section` (chapter number/name). |
| 2.4 | **Validate output** | Print sample documents, check text is clean and metadata is correct. |

**Key decisions:**
- Skip non-tutorial files (`index.html`, `index-2.html`) or keep them as lightweight context.
- Decide whether to preserve code blocks as-is (recommended for a programming tutorial).

---

### Phase 3 – Sentence-Aware Text Chunking

The chunking strategy must be **sentence-aware**: chunks should never break in the middle of a sentence. This preserves semantic coherence and improves both embedding quality and retrieval accuracy.

| # | Task | Details |
|---|------|---------|
| 3.1 | **Install NLP tokenizer** | Install `spacy` (with the `en_core_web_sm` model) or `nltk` (with `punkt_tab` tokenizer) for reliable sentence boundary detection. |
| 3.2 | **Sentence segmentation** | Split each document's text into individual sentences first, using `spacy` or `nltk.sent_tokenize()`. This is the foundation of sentence-aware chunking. |
| 3.3 | **Group sentences into chunks** | Accumulate consecutive sentences until the chunk approaches `chunk_size=1000` characters. Never split a sentence across two chunks. When adding the next sentence would exceed the limit, finalize the current chunk and start a new one. |
| 3.4 | **Sentence-level overlap** | Instead of character-based overlap, overlap by **the last 2–3 sentences** of the previous chunk. This ensures context continuity at chunk boundaries while keeping sentences intact. |
| 3.5 | **Preserve code blocks** | Before sentence segmentation, detect fenced code blocks and `>>>` REPL examples using regex. Treat each code block as an atomic, unsplittable unit so it is never broken across chunks. |
| 3.6 | **Respect section headings** | When a new heading (e.g., `## Section Title`) is encountered, force a chunk boundary so that chunks don't mix content from unrelated sections. |
| 3.7 | **Propagate metadata** | Each chunk inherits the parent document's metadata (`source`, `title`, `section`) plus a `chunk_index`. |
| 3.8 | **Inspect chunks** | Print statistics: total chunks, avg length, min/max length, and verify no sentence is split across chunks. Manually review a sample. |

**Implementation approach (recommended):**

```python
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

def sentence_aware_chunk(text: str, chunk_size: int = 1000, overlap_sentences: int = 2):
    """Split text into chunks that respect sentence boundaries."""
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        # If adding this sentence would exceed the limit, finalize the chunk
        if current_chunk and current_len + len(sentence) > chunk_size:
            chunks.append(" ".join(current_chunk))
            # Overlap: keep the last N sentences for context continuity
            current_chunk = current_chunk[-overlap_sentences:]
            current_len = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_len += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

**Alternative:** Use LangChain's `SpacyTextSplitter` or `NLTKTextSplitter` which provide built-in sentence-aware splitting, but the custom approach above gives finer control over code-block preservation and heading boundaries.

---

### Phase 4 – Embedding & Vector Store

| # | Task | Details |
|---|------|---------|
| 4.1 | **Select an embedding model** | Option A: `OpenAIEmbeddings` (`text-embedding-3-small`). Option B: Free local model via `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`). |
| 4.2 | **Create embeddings** | Embed all chunks 304 using the chosen model. |
| 4.3 | **Store in a vector database** | Use **ChromaDB** (persistent, file-based) or **FAISS** (fast, in-memory). Persist to `vectorstore/` directory. |
| 4.4 | **Test similarity search** | Run a few sample queries (e.g., *"How do I use list comprehensions?"*) and verify the top-k results are relevant. |

---

### Phase 5 – Retrieval

| # | Task | Details |
|---|------|---------|
| 5.1 | **Configure the retriever** | Wrap the vector store in a LangChain `Retriever` with `k=4` (top 4 chunks). |
| 5.2 | **Add optional re-ranking** | Consider a cross-encoder re-ranker (`CrossEncoderReranker`) or MMR (Maximal Marginal Relevance) to improve diversity. |
| 5.3 | **Metadata filtering (optional)** | Allow filtering by chapter/section so users can scope queries. |

---

### Phase 6 – Generation (RAG Chain)

| # | Task | Details |
|---|------|---------|
| 6.1 | **Choose an LLM** | Option A: `ChatOpenAI` (`gpt-4o-mini` for cost efficiency, `gpt-4o` for quality). Option B: Local model via `Ollama` (e.g., `llama3`). |
| 6.2 | **Design the prompt template** | A system prompt that instructs the model to answer **only** from the provided context, cite sources, and admit when it doesn't know. |
| 6.3 | **Build the RAG chain** | Use LangChain's `RetrievalQA` or the newer LCEL (`create_retrieval_chain`) pattern: `retriever → format_docs → prompt → llm → output_parser`. |
| 6.4 | **Return source references** | Include the source file and section in the response so the user can verify. |

**Example Prompt Template:**

```
You are a helpful assistant that answers questions about the Python programming language
based ONLY on the provided tutorial documentation.

Context:
{context}

Question: {question}

Instructions:
- Answer using ONLY the information in the context above.
- If the context does not contain enough information, say "I don't have enough information to answer that."
- Include relevant Python code examples from the context when appropriate.
- Cite the source document(s) at the end of your answer.
```

---

### Phase 7 – User Interface

| # | Task | Details |
|---|------|---------|
| 7.1 | **CLI interface** | Simple `input()` loop for quick testing. |
| 7.2 | **Gradio / Streamlit app (optional)** | Build a lightweight web UI with a chat interface, source display panel, and optional chapter filter. |
| 7.3 | **Conversation memory (optional)** | Add `ConversationBufferMemory` so follow-up questions work naturally. |

---

### Phase 8 – Evaluation & Tuning

| # | Task | Details |
|---|------|---------|
| 8.1 | **Create a test question set** | Write 10–20 question/answer pairs covering different chapters. |
| 8.2 | **Measure retrieval quality** | Check if the correct chunks appear in top-k for each question. |
| 8.3 | **Measure answer quality** | Evaluate faithfulness (no hallucination), relevance, and completeness. Optionally use `ragas` library. |
| 8.4 | **Tune parameters** | Adjust `chunk_size`, `chunk_overlap`, `k`, prompt wording, and embedding model based on results. |

---

## 🧰 Tech Stack Summary

| Component | Primary Choice | Alternative |
|---|---|---|
| Language | Python 3.11+ | — |
| Framework | LangChain | LlamaIndex |
| HTML Parsing | BeautifulSoup4 | Unstructured |
| Embeddings | OpenAI `text-embedding-3-small` | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB | FAISS |
| LLM | OpenAI `gpt-4o-mini` | Ollama (Llama 3) |
| UI | Gradio / Streamlit | CLI |
| Evaluation | RAGAS | Manual |

---

## ✅ Definition of Done

- [ ] All 18 tutorial HTML pages are parsed and cleaned
- [ ] Text is chunked with metadata preserved
- [ ] Embeddings are generated and stored in a persistent vector store
- [ ] A RAG chain answers questions grounded in the tutorial content
- [ ] Sources are cited in every answer
- [ ] A usable interface (at minimum CLI) is available
- [ ] At least 10 test questions return satisfactory answers

---

## 🚀 Next Step

**→ Proceed to Phase 1: set up the virtual environment, install dependencies, and scaffold the project files.**
