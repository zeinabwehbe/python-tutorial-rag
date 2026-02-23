"""
Central configuration for the RAG system.
All paths, model names, and tuning parameters live here.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TUTORIAL_DIR = PROJECT_ROOT / "py_tutorial" / "docs.python.org" / "3" / "tutorial"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000          # max characters per chunk
OVERLAP_SENTENCES = 2      # number of trailing sentences carried into next chunk

# ── Embedding ────────────────────────────────────────────────────────────────
# Embedding provider: "openai", "huggingface", or "deepseek"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_COLLECTION = "python_tutorial"

# Deepseek configuration (when using Deepseek as embedding provider)
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVER_K = 5            # top-k chunks to retrieve
SEARCH_TYPE = "mmr"        # "similarity" or "mmr"

# ── LLM ──────────────────────────────────────────────────────────────────────
# LLM provider: "openai" or "local" (HuggingFace fallback)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
