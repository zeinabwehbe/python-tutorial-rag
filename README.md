# Python Tutorial RAG System 🐍

A **Retrieval-Augmented Generation (RAG)** system built with LangChain that answers questions about Python programming using the official Python Tutorial documentation. Features sentence-aware chunking, multiple embedding providers, and both CLI and modern web interfaces.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.0+-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-API-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red.svg)

## ✨ Features

- **🧠 Smart Document Processing**: Sentence-aware text chunking preserves code blocks and context
- **🔍 Advanced Retrieval**: ChromaDB vector store with MMR (Maximal Marginal Relevance) search
- **🌐 Multiple Providers**: Support for OpenAI, HuggingFace, and custom embedding models
- **💬 Dual Interfaces**: Modern Streamlit web UI + CLI chat interface
- **📚 Rich Context**: Processes 16+ Python tutorial HTML files with metadata preservation
- **⚡ Caching & Performance**: Streamlit caching for instant responses
- **🛠️ Flexible Configuration**: Environment-based config for easy provider switching

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HTML Docs     │───▶│     Loader      │───▶│    Chunker      │
│  (py_tutorial)  │    │  (BeautifulSoup)│    │ (Sentence-aware)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web/CLI UI    │◀───│   RAG Chain     │◀───│   Embeddings    │
│   (Streamlit)   │    │  (LangChain)    │    │   (OpenAI)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Retriever     │◀───│  Vector Store   │
                       │     (MMR)       │    │   (ChromaDB)    │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- OpenAI API key (or HuggingFace/custom provider)
- Git (for cloning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zeinabwehbe/python-tutorial-rag.git
   cd python-tutorial-rag
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Copy and edit .env file
   cp .env.example .env
   ```
   
   Add your API keys to `.env`:
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   EMBEDDING_PROVIDER=openai
   LLM_PROVIDER=openai
   ```

5. **Download NLTK data** (for sentence chunking)
   ```bash
   python -c "import nltk; nltk.download('punkt_tab')"
   ```

### Usage

#### 🌐 Web Interface (Recommended)
```bash
streamlit run src/web_app.py
```

Then open http://localhost:8501 in your browser.

#### 💻 CLI Interface
```bash
python -m src.app
```

#### 🔧 Manual Ingestion (if needed)
```bash
python -c "from src.app import ingest; ingest()"
```

## 📖 Example Questions

Try asking these questions to test the system:

- *"What are list comprehensions in Python?"*
- *"How do I handle exceptions and errors?"*
- *"Explain Python's for loops with examples"*
- *"What is the difference between lists and tuples?"*
- *"How do I create and use functions in Python?"*
- *"What are Python modules and how do I import them?"*

## ⚙️ Configuration

Key settings in `.env`:

| Variable | Description | Options |
|----------|-------------|---------|
| `EMBEDDING_PROVIDER` | Embedding model provider | `openai`, `huggingface`, `deepseek` |
| `EMBEDDING_MODEL` | Specific model name | `text-embedding-3-small`, etc. |
| `LLM_PROVIDER` | Language model provider | `openai`, `local` |
| `LLM_MODEL` | Specific LLM model | `gpt-4o-mini`, `gpt-3.5-turbo`, etc. |
| `CHUNK_SIZE` | Maximum characters per chunk | `1000` (default) |
| `OVERLAP_SENTENCES` | Sentence overlap between chunks | `2` (default) |
| `RETRIEVER_K` | Number of chunks to retrieve | `5` (default) |

## 📁 Project Structure

```
python-tutorial-rag/
├── src/
│   ├── config.py          # Configuration management
│   ├── loader.py          # HTML document parsing
│   ├── chunker.py         # Sentence-aware text chunking
│   ├── embedder.py        # Multi-provider embeddings
│   ├── retriever.py       # ChromaDB retrieval setup
│   ├── chain.py           # RAG chain with LangChain
│   ├── app.py             # CLI interface
│   └── web_app.py         # Streamlit web interface
├── py_tutorial/           # Python tutorial HTML files
├── vectorstore/           # ChromaDB persistence (auto-created)
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
├── PROJECT_PLAN.md        # Development roadmap
└── README.md             # This file
```

## 🧪 Development

### Running Tests
```bash
# Test individual components
python -c "from src.loader import load_documents; print(f'Loaded {len(load_documents())} documents')"
python -c "from src.chunker import chunk_documents; from src.loader import load_documents; chunks = chunk_documents(load_documents()); print(f'Created {len(chunks)} chunks')"
```

### Adding New Document Sources
1. Place HTML files in `py_tutorial/` directory
2. Modify `src/loader.py` if different parsing logic needed
3. Run ingestion: `python -c "from src.app import ingest; ingest()"`

### Switching Embedding Providers

**OpenAI** (recommended):
```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your-key
```

**HuggingFace** (free, local):
```env
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Custom providers**: Extend `src/embedder.py`

## 🐛 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'src'**
- Ensure you're running from the project root directory
- For Streamlit: Fixed automatically with `sys.path` setup in `web_app.py`

**OpenAI quota exceeded**
- Switch to HuggingFace provider temporarily
- Check your OpenAI billing and usage limits

**No vectorstore found**
- Run ingestion manually: `python -c "from src.app import ingest; ingest()"`
- Check that `py_tutorial/` contains HTML files

**Slow embedding performance**
- Use OpenAI for faster cloud embeddings
- For local: ensure CUDA GPU support for sentence-transformers

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Python Software Foundation** for the excellent Python Tutorial documentation
- **LangChain** for the RAG framework
- **ChromaDB** for vector storage
- **OpenAI** for embeddings and language models
- **Streamlit** for the web interface framework

## 📊 Performance Stats

- **Documents**: 16 Python tutorial HTML files
- **Chunks**: 304 sentence-aware chunks (avg ~991 characters)
- **Embedding Model**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Response Time**: <2 seconds for most queries
- **Retrieval**: Top-5 MMR search for optimal relevance/diversity

---

**Built with ❤️ for Python learners everywhere!**