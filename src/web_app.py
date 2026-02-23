"""
Python Tutorial RAG System – Streamlit Web UI

A modern chat interface for asking questions about the Python Tutorial.
Run with:  streamlit run src/web_app.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Python Tutorial RAG",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a modern look ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Source card styling */
    .source-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 10px;
        margin: 4px 0;
        font-size: 0.9em;
        border-left: 4px solid #4dabf7;
    }
    .source-card .source-file {
        font-weight: 600;
        color: #74c0fc;
    }
    .source-card .source-heading {
        color: #a5d8ff;
        font-size: 0.85em;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1929 0%, #1a2f4a 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Title banner */
    .title-banner {
        background: linear-gradient(135deg, #306998 0%, #FFD43B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2em;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 1em;
        margin-top: -8px;
    }

    /* Stats badges */
    .stat-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #74c0fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        margin: 2px 4px 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Lazy-loaded resources (cached) ──────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG chain …")
def load_chain():
    """Build the RAG chain once and cache it across reruns."""
    from src.chain import build_rag_chain
    return build_rag_chain()


@st.cache_resource(show_spinner="Loading retriever …")
def load_retriever():
    """Load the retriever for source inspection."""
    from src.retriever import get_retriever
    return get_retriever()


@st.cache_resource(show_spinner="Loading vector store …")
def load_vectorstore_info():
    """Get vectorstore stats."""
    from src.embedder import load_vectorstore
    vs = load_vectorstore()
    count = vs._collection.count()
    return count


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🐍 Python Tutorial RAG")
    st.markdown("---")

    # Stats
    try:
        chunk_count = load_vectorstore_info()
        st.markdown(f'<span class="stat-badge">📦 {chunk_count} chunks indexed</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="stat-badge">📄 16 tutorial pages</span>', unsafe_allow_html=True)
    except Exception:
        st.warning("Vector store not loaded yet.")

    st.markdown("---")

    st.markdown("### 💡 Example questions")
    example_questions = [
        "What are list comprehensions?",
        "How do I handle exceptions in Python?",
        "Explain Python classes and inheritance",
        "How do virtual environments work?",
        "What are Python decorators?",
        "How do I read and write files?",
        "What is the difference between tuples and lists?",
        "How do I use lambda functions?",
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")
    show_sources = st.toggle("Show source chunks", value=True)
    show_context_length = st.toggle("Show context stats", value=False)

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; font-size:0.75em; color:#666;">'
        "Built with LangChain + ChromaDB + OpenAI<br>"
        "Python 3.14 Tutorial Documentation"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main area ────────────────────────────────────────────────────────────────

st.markdown('<p class="title-banner">🐍 Python Tutorial Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask anything about the official Python Tutorial — answers are grounded in the docs.</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🐍"):
        st.markdown(msg["content"])

        # Show sources if available
        if msg["role"] == "assistant" and "sources" in msg and show_sources:
            with st.expander(f"📄 Sources ({len(msg['sources'])} chunks)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="source-file">📄 {src["source"]}</span><br>'
                        f'<span class="source-heading">{src.get("heading", "")}</span><br>'
                        f'<span style="font-size:0.8em; color:#ccc;">{src["preview"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── Handle input ─────────────────────────────────────────────────────────────

# Check for pending question from sidebar button
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Ask a question about Python …") or pending

if user_input:
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate answer
    with st.chat_message("assistant", avatar="🐍"):
        with st.spinner("Thinking …"):
            chain = load_chain()
            retriever = load_retriever()

            # Get retrieved docs for source display
            try:
                docs = retriever.invoke(user_input)
            except Exception:
                docs = []

            # Get LLM answer
            try:
                answer = chain.invoke(user_input)
            except Exception as e:
                answer = f"⚠️ Error generating answer: {e}"

        st.markdown(answer)

        # Build source metadata
        sources_meta = []
        for doc in docs:
            sources_meta.append({
                "source": doc.metadata.get("source", "unknown"),
                "heading": doc.metadata.get("heading", ""),
                "preview": doc.page_content[:200] + "…",
            })

        # Show sources
        if show_sources and sources_meta:
            with st.expander(f"📄 Sources ({len(sources_meta)} chunks)", expanded=False):
                for src in sources_meta:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="source-file">📄 {src["source"]}</span><br>'
                        f'<span class="source-heading">{src.get("heading", "")}</span><br>'
                        f'<span style="font-size:0.8em; color:#ccc;">{src["preview"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Show context stats
        if show_context_length and docs:
            total_chars = sum(len(d.page_content) for d in docs)
            st.caption(f"📊 Context: {len(docs)} chunks, {total_chars:,} chars total")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_meta,
    })
