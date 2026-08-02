# Droply RAG service
#
# Install:
#   python -m venv .venv
#   .venv\Scripts\activate
#   pip install -r requirements.txt
#
# Env (.env):
#   DATABASE_URL=...
#   GEMINI_API_KEY=...        # embeddings (gemini-embedding-001)
#   GROQ_API_KEY=...          # chat LLM
#   RAG_INTERNAL_KEY=...
#   GROQ_CHAT_MODEL=llama-3.1-8b-instant   # optional
#   RAG_TOP_K=8                             # optional
#
# Endpoints:
#   GET  /health
#   POST /ingest   (internal) document indexing
#   POST /chat     (internal) SSE QnA — embed=Gemini, answer=Groq
#
# Run:
#   uvicorn main:app --reload --port 8001
