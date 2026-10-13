# 🌊 ARGO-AI Backend Service

The ARGO-AI backend is a lightweight, modular Python service built with **Flask 3.x**, **LangChain**, and **ChromaDB**.

---

## 🏗️ Architecture

```
backend/
├── app/
│   ├── nlp/             # Pydantic Intent Schema & Entity Extractor (parser.py)
│   ├── rag/             # ChromaDB Vector Store & NOAA ERDDAP Ingestion
│   ├── tools/           # Matplotlib Chart Engine & Folium Map Engine
│   └── agent.py         # LangChain Multi-Turn Orchestration Agent
├── tests/               # Regression and pipeline verification tests
├── main.py              # 160-line Production REST API Router
└── requirements.txt     # Pinned Python dependencies
```

---

## ⚙️ Environment Variables

Create a `.env` file in this directory:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
PORT=5000
```

---

## 🚀 Running the Server

```bash
# 1. Activate Virtual Environment
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/macOS

# 2. Start Flask Server
python main.py
```

Server runs on: **`http://localhost:5000`**

---

## 🧪 Running Automated Tests

```bash
# Run End-to-End RAG & NLP Test
.\venv\Scripts\python.exe tests/test_rag_pipeline.py

# Run Multi-Turn Chat & Reasoning Test
.\venv\Scripts\python.exe tests/test_chat_endpoint.py
```

For full system documentation and architectural diagrams, see the root [README.md](../README.md).
