# 🧠 RAG Pipeline with LlamaIndex & NVIDIA NIM

A RAG (Retrieval-Augmented Generation) pipeline built with LlamaIndex and NVIDIA NIM for document querying and summarization.

## 📋 Overview

This project implements a document querying system using LlamaIndex with NVIDIA's LLM and embedding models. It supports two execution modes:

- **Router Mode**: Routes queries between summary and vector indices
- **Agent Mode**: Conversational AI with tool calling (experimental)

## 🚀 Current Features

- Document loading and chunking (PDF support)
- SummaryIndex for document overviews
- VectorStoreIndex for detailed retrieval
- RouterQueryEngine for intelligent routing
- Agent workflow with tool calling (WIP)
- Async support
- Configurable via .env

## 📦 Prerequisites

- Python 3.10+
- NVIDIA API Key (from [NVIDIA NIM](https://build.nvidia.com/))
- Virtual environment (recommended)

## 🔧 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-index-python
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install llama-index llama-index-core llama-index-llms-nvidia
pip install llama-index-embeddings-nvidia llama-index-readers-file
pip install pypdf python-dotenv
```

Or install all at once:

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### 1. Set up environment variables

Create a `.env` file in the project root:

```env
# Required
NVIDIA_API_KEY=your_nvidia_api_key_here

# Optional
DEBUG=True
EXECUTION_MODE=router  # router or agent
AGENT_VERBOSE=True
AGENT_MAX_STEPS=5
```

### 2. Add your documents

Place your PDF or text files in the `data/` directory:

```bash
data/
├── document1.pdf
├── document2.pdf
└── your_cv.pdf
```

## 🎯 Usage

### Run the application

```bash
python app.py
```

### Example queries

```python
# Router mode
response = await orchestrator.query("Summarize my professional profile")
response = await orchestrator.query("What technical skills do I have?")

# Agent mode (if enabled)
response = await orchestrator.query("Tell me about my latest experience")
```

## 📁 Project Structure

```
rag-index-python/
├── app.py                      # Main entry point
├── .env                        # Environment variables
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── config/
│   └── settings.py             # Configuration management
│
├── core/
│   ├── orchestrator.py         # Main orchestration
│   ├── data_loader.py          # Data loading & chunking
│   ├── index_manager.py        # Index creation
│   ├── query_engine.py         # Router query engine
│   └── agent_manager.py        # Agent workflow (WIP)
│
├── tools/
│   └── vector_tools.py         # Custom vector tools
│
├── utils/
│   ├── debug.py                # Debug utilities
│   └── retry.py                # Retry with backoff
│
└── data/                       # Document storage
    └── your_documents.pdf
```

## 🔄 Execution Modes

### Router Mode (default)

Routes queries between SummaryIndex and VectorStoreIndex:

- `"Summarize..."` → Summary Index
- `"What/How/Which..."` → Vector Index

### Agent Mode (experimental)

Uses AgentWorkflow with tool calling. Currently has known issues with NVIDIA models that don't support tool calling natively.

**Known limitations:**
- Not all NVIDIA models support tool calling
- May experience timeout issues
- Streaming response handling is work in progress

## 🐛 Current Issues & Limitations

| Issue | Status | Workaround |
|-------|--------|------------|
| Agent mode may fail with `meta/llama-3.3-70b-instruct` | Investigating | Use router mode or switch to `meta/llama-3.1-70b-instruct` |
| API retry messages | High demand on NVIDIA API | Increase `REQUEST_TIMEOUT` and `MAX_RETRIES` |
| PDF parsing issues | Need pypdf installed | `pip install pypdf` |

## 🧪 Testing

### Test API connection

```python
# test.py
from config.settings import settings
from llama_index.llms.nvidia import NVIDIA

llm = NVIDIA(
    model=settings.LLM_MODEL,
    api_key=settings.NVIDIA_API_KEY
)
response = llm.complete("Hello")
print(response)
```

## 🤔 FAQ

**Q: Why am I getting "Retrying request" messages?**
A: The NVIDIA API is under high demand. Try increasing `REQUEST_TIMEOUT` or using a different model.

**Q: Agent mode doesn't work with my model?**
A: Not all models support tool calling. Use router mode or switch to another model.

**Q: My PDF isn't loading correctly?**
A: Make sure pypdf is installed: `pip install pypdf`

**Q: How do I switch between router and agent mode?**
A: Change `EXECUTION_MODE` in .env to `router` or `agent`.

## 📌 Notes

- This is a **development/learning project**, not production-ready
- The agent mode is experimental and may have issues
- API costs depend on usage
- Models and libraries are subject to change

## 📚 Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [NVIDIA NIM Documentation](https://build.nvidia.com/docs)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)

---

**Work in progress** 🚧