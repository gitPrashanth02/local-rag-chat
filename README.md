# RAG Pipeline

A local Retrieval-Augmented Generation (RAG) pipeline that answers questions about a PDF document.
It parses the PDF, splits it into chunks, stores embeddings in ChromaDB, retrieves the most relevant
chunks with hybrid search and reranking, and generates an answer with a local language model.
No API keys are needed: every model runs on your machine.

## How it works

```
PDF ──► Parse ──► Chunk ──► Embed ──► ChromaDB
                                          │
Question ──► Hybrid search (vector + TF-IDF keywords) ──► Rerank ──► LLM ──► Answer + sources
```

| Step | Script | What it does |
|---|---|---|
| 1. Parse | `src/document_parser.py` | Converts `data/chatgpt.pdf` to Markdown and JSON with MinerU |
| 2. Chunk | `src/chunking.py` | Splits the text into ~1000-character chunks (150 overlap) → `document_chunks.jsonl` |
| 3. Embed | `src/embedding.py` | Embeds the chunks and stores them in `chroma_db/` |
| 4. Retrieve | `src/retrieval.py` | Hybrid search (70% vector, 30% TF-IDF keyword), then cross-encoder reranking |
| 5. Generate | `src/generation.py` | Answers the question using only the retrieved context, and lists the source pages |

## Models

| Purpose | Model |
|---|---|
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` |
| Reranker | `BAAI/bge-reranker-base` |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` |

The models download from Hugging Face the first time they are used.

## Project structure

```
.
├── src/
│   ├── document_parser.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── pipeline.py          # build_index() and answer(): the whole flow in order
│   ├── api.py               # FastAPI backend (builds the index, then serves)
│   └── streamlit_app.py     # chat page (frontend) that calls the API
├── data/
│   └── chatgpt.pdf          # source document
├── requirements.txt
└── README.md
```

These are generated when you run the pipeline: `data/chatgpt.md`, `data/chatgpt.json`,
`document_chunks.jsonl` and `chroma_db/`.

## Setup

Requires Python 3.12.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Usage

### Quick start: one command

```bash
python src/api.py
```

This builds the index if needed (parse → chunk → embed), loads the models and starts the API at
http://127.0.0.1:8000. Each build step runs only when its output is missing or older than its input,
so later starts skip straight to loading the models. The first start downloads the models and parses
the PDF, so it can take 10+ minutes.

### Running the steps by hand

```bash
python src/pipeline.py            # all three steps, skipping what is up to date
python src/pipeline.py --force    # rebuild everything
```

Or one at a time:

```bash
python src/document_parser.py
python src/chunking.py
python src/embedding.py
```

Then ask questions in the terminal:

```bash
# Interactive mode (type 'exit' to quit)
python src/generation.py

# Single question
python src/generation.py "What is ChatGPT?"

# Retrieval only: show the matching chunks without generating an answer
python src/retrieval.py "What is ChatGPT?"
```

## REST API

Start the server from the project root with `python src/api.py` (or `uvicorn api:app --app-dir src`).
To use another port: `set PORT=8001` (Windows) or `PORT=8001 python src/api.py` (macOS / Linux).

Open http://127.0.0.1:8000 (it redirects to `/docs`) for interactive documentation where you can try the endpoints.

| Method | Endpoint | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | server status and chunk count |
| POST | `/ask` | `{"question": "What is ChatGPT?", "top_k": 5}` | answer and its source chunks |
| POST | `/documents` | a PDF file (form upload) | indexes it and makes it the current document |

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is ChatGPT?\"}"
```

Models load once at startup (this takes a minute), and questions are answered one at a time.

## Chat page

A Streamlit chat page that talks to the API. Run it in a second terminal while the API is running:

```bash
python src/api.py                       # terminal 1: backend
streamlit run src/streamlit_app.py      # terminal 2: chat page, opens http://localhost:8501
```

To switch documents, attach a PDF with the paperclip in the message box and confirm **Ingest** in the popup.
Any text sent with the file is asked as a question once indexing finishes (this takes a few minutes).
The page shows each answer with its source pages, offers example questions, and lets you choose how
many chunks the model reads. If the API runs at another address, set `RAG_API_URL` before starting it.

To use a different document, attach it in the chat page or send it to `POST /documents`.
The last indexed document is remembered in `data/current_document.txt` and loaded on the next start.

## Docker

The `Dockerfile` and `start.sh` run the API and the chat page together in one container,
with the chat page on port 7860. The image downloads the models and indexes `data/chatgpt.pdf`
at build time, so the container starts quickly.

```bash
docker build -t local-rag-chat .
docker run -p 7860:7860 local-rag-chat
```

Then open http://localhost:7860. Uploads are turned off in the image (`ALLOW_UPLOADS=false`
in the `Dockerfile`), so visitors of a public server can't replace the document for everyone.
