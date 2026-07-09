# Egyptian Legal RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for Egyptian legal documents.

## Tech Stack

- **LLM**: Qwen2.5-7B(via OpenAI-compatible API)
- **Vector DB**: ChromaDB
- **Embeddings**: BAAI/bge-m3
- **Reranker**: BAAI/bge-reranker-v2-m3
- **Framework**: FastAPI
- **Search**: Hybrid (BM25 + Dense + RRF + Cross-Encoder Reranking)

## Architecture

```
Query → Route → Retrieve → Rerank → Generate → Answer
```

### Pipeline Components

| Component | Description |
|-----------|-------------|
| **Ingestion** | Auto-discovers law files, generates metadata, chunks articles, embeds, stores |
| **Retrieval** | Hybrid search (BM25 + Dense), Reciprocal Rank Fusion, Cross-Encoder reranking |
| **Generation** | Qwen2.5-7B-Instruct with context-only prompting, citations, fallback |
| **Routing** | Keyword-based law classification, metadata filtering |

## Project Structure

```
├── config.py              # Configuration
├── main.py                # FastAPI entry point
├── core/                  # Constants, exceptions, logging
├── data/
│   └── raw/               # Law .txt files (auto-discovered)
├── metadata/              # Generated metadata JSON
├── ingestion/             # Load → Clean → Chunk → Normalize → Embed → Store
├── retrieval/             # BM25 + Dense + Hybrid + RRF + Reranker
├── generation/            # LLM → Prompt → Citation → Format
├── routing/               # Query routing & law classification
├── schemas/               # Pydantic models
├── services/              # Business logic layer
├── api/routers/           # FastAPI endpoints
├── prompts/               # Prompt templates
├── scripts/               # Utility scripts
└── tests/                 # Unit tests
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama with Qwen

```bash
ollama pull qwen2.5:7b-instruct
ollama serve
```

### 3. Add Law Files

Place `.txt` law files in `data/raw/`. The system auto-discovers all files.

### 4. Run Ingestion

```bash
# Generate metadata
python scripts/generate_metadata.py

# Or use the API
curl -X POST http://localhost:8000/api/ingest
```

### 5. Start the API

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Ask Questions

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي عقوبة السرقة؟"}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/` | Service info |
| POST | `/api/chat` | Ask a legal question |
| POST | `/api/search` | Search legal documents |
| POST | `/api/ingest` | Trigger ingestion pipeline |

### Chat Request

```json
{
  "question": "ما هي عقوبة السرقة؟",
  "law_filter": "criminal_code",
  "top_k": 5
}
```

### Search Request

```json
{
  "query": "عقوبة السرقة",
  "mode": "hybrid",
  "top_k": 10,
  "law_filter": null
}
```

## Docker

```bash
docker-compose up -d
```

## Adding New Laws

Simply add a `.txt` file to `data/raw/` and run:

```bash
python scripts/generate_metadata.py
curl -X POST http://localhost:8000/api/ingest
```

No code changes required.

## Tests

```bash
python -m pytest tests/ -v
```

## Environment Variables

See `.env.example` for all available settings.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_URL` | `http://localhost:11434/v1` | LLM API endpoint |
| `LLM_API_KEY` | `ollama` | API key |
| `LLM_MODEL` | `qwen2.5:7b` | Model name |
| `LLM_MAX_TOKENS` | `2048` | Max tokens |
| `LLM_TEMPERATURE` | `0.1` | Temperature |
| `APP_HOST` | `0.0.0.0` | Server host |
| `APP_PORT` | `8000` | Server port |
