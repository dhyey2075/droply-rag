# Droply RAG

Internal FastAPI **chat** service for Droply. Document indexing does **not** run in this process — a separate BullMQ worker (`python -m app.ingest.worker`) consumes Redis jobs and runs the ingest LangGraph. This HTTP server only answers questions over already-indexed chunks (Groq + pgvector), streamed via SSE.

## Stack

| Layer | Technology |
| --- | --- |
| API | FastAPI + Uvicorn |
| Orchestration | **LangGraph** StateGraphs (ingest + CRAG chat) |
| Embeddings | Google Gemini (`gemini-embedding-001`, 768 dims) |
| Chat | Groq (configurable; default `openai/gpt-oss-20b`) |
| Vector store | PostgreSQL + `pgvector` |
| Document loaders | LangChain (PDF, DOCX, TXT, and related text types) |
| Web fallback | DDGS (Corrective RAG) |

## Architecture

Pipelines are modular LangGraph nodes. **FastAPI only serves chat.** Indexing runs in a separate worker process.

### Chat (CRAG)

```text
embed_query → retrieve → grade_documents
                              ├─ prepare_documents → build_prompt → (stream tokens)
                              └─ refine_query → web_search → prepare_web → build_prompt → (stream tokens)
```

`refine_query` rewrites the user question (using recent chat history) into a self-contained, search-engine-friendly query before DDGS web search.

![Chat CRAG LangGraph](diagrams/chat_crag_graph.png)

### Ingest (worker process, not this HTTP server)

```text
validate → download → parse → chunk → embed_docs → upsert
```

Run: `python -m app.ingest.worker` (same Redis queues Droply enqueues to).

```text
validate → download → parse → chunk → embed_docs → upsert
```

![Ingest LangGraph](diagrams/ingest_graph.png)

### Regenerating diagrams

```bash
python export_graphs.py
```

Outputs land in `diagrams/`:
- `chat_crag_graph.png` / `.mmd`
- `ingest_graph.png` / `.mmd`

## Features

- Download and chunk documents from a signed/public URL (**worker**, not the API)
- Upsert embeddings into `document_chunks` (scoped by `user_id` + `file_id`)
- Similarity retrieval across a user's full indexed library (optional `file_ids` scope)
- **Corrective RAG (CRAG):** LLM relevance grading of retrieved chunks; if all score below the threshold, fall back to web search
- Streaming chat responses (`meta` → `token` → `done` / `error`)
- Internal API auth via shared bearer key
- Node-wise LangGraph structure for readability and testing

## Requirements

- Python 3.12+
- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension
- Redis (for the indexing worker)
- API keys for Gemini and Groq

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` before starting the service.

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `DATABASE_URL` | Yes | Postgres connection string (must support `pgvector`) |
| `GEMINI_API_KEY` | Yes | Gemini API key for embeddings (`GOOGLE_API_KEY` also accepted) |
| `GROQ_API_KEY` | Yes | Groq API key for chat completions |
| `RAG_INTERNAL_KEY` | Yes | Shared secret; callers must send `Authorization: Bearer <key>` |
| `GROQ_CHAT_MODEL` | No | Chat model (default: `openai/gpt-oss-20b`) |
| `RAG_TOP_K` | No | Number of chunks retrieved per query (default: `8`) |
| `CRAG_ENABLED` | No | Enable Corrective RAG web fallback (default: `1`) |
| `CRAG_RELEVANCE_THRESHOLD` | No | Min chunk relevance % to keep docs (default: `50`) |
| `WEB_SEARCH_RESULTS` | No | Web results to fetch on fallback (default: `5`) |
| `REDIS_URL` | Yes (worker) | Redis URL for BullMQ (`indexing` / `indexing-dlq`) |
| `APP_URL` | No | Droply app origin for indexing SSE (default: `http://localhost:3000`) |
| `INDEX_CONCURRENCY` | No | Max parallel ingests (default: `3`) |
| `INDEX_CONCURRENCY_PER_USER` | No | Max parallel ingests per user (default: `2`) |
| `PORT` | No | Listen port for Docker / Procfile (default: `8001`) |

## Run locally

```bash
uvicorn main:app --reload --port 8001
```

Indexing worker (separate terminal, same venv):

```bash
python -m app.ingest.worker
```

Health check:

```bash
curl http://localhost:8001/health
```

## Docker

```bash
docker build -t droply-rag .
docker run --env-file .env -p 8001:8001 droply-rag
```

## API

All mutating/chat endpoints require:

```http
Authorization: Bearer <RAG_INTERNAL_KEY>
```

### `GET /health`

Public health probe.

```json
{ "status": "ok" }
```

There is **no** `/ingest` HTTP endpoint. Indexing is performed only by `python -m app.ingest.worker`.

### `POST /chat`

Library-wide Q&A for a user. Returns an SSE stream.

**Request**

```json
{
  "user_id": "clerk_user_id",
  "question": "What are the key findings?",
  "history": [
    { "role": "user", "content": "Summarize the intro" },
    { "role": "assistant", "content": "..." }
  ],
  "file_ids": ["uuid-optional", "uuid-optional"]
}
```

`file_ids` is optional. Omit or pass `[]` to search the user’s full indexed library; pass specific IDs to scope retrieval to those files only.

**SSE events**

| Event | Payload | When |
| --- | --- | --- |
| `status` | `{ "step", "message" }` | Live progress as each chat-graph node finishes (and once before Groq tokens) |
| `meta` | `{ "mode", "sources", "relevance?" }` | After the graph finishes (sources / mode ready) |
| `token` | `{ "text": "..." }` | Each streamed completion delta |
| `done` | `{ "mode", "sources", "relevance?" }` | Stream finished |
| `error` | `{ "message": "..." }` | Failure during chat |

`status.step` is a graph node id (`embed_query`, `retrieve`, `grade_documents`, `prepare_documents`, `refine_query`, `web_search`, `prepare_web`, `build_prompt`) plus `start` (before the graph) and `generate` (Groq stream about to begin). `message` is the user-facing phrase for the Ask UI.

`mode` is `"documents"` when answering from indexed files, or `"web"` when CRAG fell back to search because every chunk scored below `CRAG_RELEVANCE_THRESHOLD`.

Example:

```bash
curl -N http://localhost:8001/chat \
  -H "Authorization: Bearer $RAG_INTERNAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"...","question":"What is this document about?"}'
```

## Project layout

```text
droply-rag/
├── main.py                 # FastAPI shell (/health, /chat)
├── app/
│   ├── config.py
│   ├── auth.py
│   ├── models.py
│   ├── clients.py
│   ├── sse.py
│   ├── chat/               # CRAG LangGraph
│   ├── ingest/             # Ingest LangGraph + BullMQ worker
│   │   ├── worker.py       # python -m app.ingest.worker
│   │   ├── jobs.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── loaders.py
│   ├── retrieval/          # embeddings, pgvector, sources
│   ├── grading/            # CRAG relevance grader
│   └── search/             # DDGS web fallback
├── requirements.txt
├── Dockerfile
├── Procfile
├── .env.example
└── README.md
```

## Notes

- Intended as an **internal** service called by the Droply backend — do not expose publicly without additional auth.
- In **documents** mode, answers are grounded only in retrieved context. In **web** mode (CRAG fallback), answers use search results instead.
- Supported ingest types are inferred from filename extension / MIME (PDF, DOCX, TXT, MD, CSV, and similar text formats).
