"""REST API for the RAG pipeline.

Run from the project root (builds the index first if it is missing):
    python src/api.py

Then open http://127.0.0.1:8000 to try it in the browser.
"""
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from generation import LLM_MODEL, load_generator
from pipeline import ROOT, answer, build_index, current_document
from retrieval import COLLECTION_NAME, DB_PATH, TOP_K, load_chunk_texts, load_index, load_reranker

DATA_DIR = ROOT / "data"                 # uploads must go here: pipeline.py looks for the JSON next to the PDF
MAX_UPLOAD_MB = 50
# Set ALLOW_UPLOADS=false on a public server so visitors can't replace the document for everyone
ALLOW_UPLOADS = os.environ.get("ALLOW_UPLOADS", "true").lower() not in {"false", "0", "no"}

# Uvicorn already sets up this logger to print INFO messages in the terminal,
# so our messages appear in the same format as its own "INFO: ..." lines.
log = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# 1. Request / response schemas (Pydantic models)
#    FastAPI uses these to check incoming JSON, convert it into Python objects,
#    and write the documentation at /docs. Invalid input gets an automatic 422 error.
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, examples=["What is ChatGPT?"])
    top_k: int = Field(TOP_K, ge=1, le=10, description="How many chunks to give the LLM")


class Source(BaseModel):
    chunk_id: int
    page: int
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class DocumentResponse(BaseModel):
    document: str
    chunks: int
    seconds: float


# ---------------------------------------------------------------------------
# 2. Lifespan: code that runs ONCE when the server starts (before `yield`)
#    and once when it stops (after `yield`). Loading models takes a while,
#    so we do it here instead of inside every request.
# ---------------------------------------------------------------------------
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()                            # parse/chunk/embed only if missing or out of date
    collection, embed_model = load_index()   # ChromaDB + embedding model
    load_reranker()                          # cached by lru_cache
    load_generator()                         # cached by lru_cache
    state["collection"] = collection
    state["embed_model"] = embed_model
    print("Models loaded - API is ready.")
    yield
    state.clear()


app = FastAPI(
    title="RAG API",
    description="Upload a PDF and ask questions about it. Answers come from a local LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

# The models are not safe to run from several threads at the same time,
# so only one question is answered at a time; other requests wait their turn.
model_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 3. Endpoints: a decorator connects a URL + HTTP method to a Python function.
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    """Opening the main address sends the browser to the docs page."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Quick check that the server is up and the index is loaded."""
    return {
        "status": "ok",
        "document": current_document().name,
        "chunks": state["collection"].count(),
        "llm": LLM_MODEL,
        "uploads": ALLOW_UPLOADS,
    }


# A plain `def` (not `async def`) tells FastAPI to run this in a worker thread,
# so the slow model call does not freeze the whole server.
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Run the question pipeline (search -> rerank -> generate) and return the result."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be blank.")

    log.info("Question: %s", question)
    start = time.perf_counter()
    with model_lock:
        result = answer(question, state["collection"], state["embed_model"], top_k=request.top_k)
    pages = sorted({h["page"] for h in result["hits"]})
    log.info("Answer (%.1fs, pages %s): %s", time.perf_counter() - start, pages, result["answer"])

    return AskResponse(
        question=question,
        answer=result["answer"],
        sources=[Source(chunk_id=h["chunk_id"], page=h["page"], score=h["score"], text=h["text"])
                 for h in result["hits"]],
    )


@app.post("/documents", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(..., description="A PDF to index")):
    """Upload a PDF, index it (parse -> chunk -> embed) and make it the document questions are about.

    Indexing takes minutes; questions wait until it's finished.
    """
    if not ALLOW_UPLOADS:
        raise HTTPException(status_code=403, detail="Uploading documents is turned off on this server.")

    # 1. Check the file. Path(...).name drops any folders in the name, e.g. "..\\..\\x.pdf" -> "x.pdf"
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be uploaded.")
    content = file.file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"The file is larger than {MAX_UPLOAD_MB} MB.")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="This file is not a valid PDF.")

    # 2. Save it, rebuild the index and reload. The lock makes questions wait during the rebuild.
    log.info("Indexing uploaded document: %s", filename)
    start = time.perf_counter()
    with model_lock:
        pdf_path = DATA_DIR / filename
        pdf_path.write_bytes(content)
        try:
            build_index(pdf_path, force=True)
        except Exception as error:
            log.exception("Indexing %s failed", filename)
            raise HTTPException(status_code=500, detail=f"Indexing failed: {error}")
        finally:
            # embedding.py deletes and recreates the collection, so the old handle no longer works
            try:
                state["collection"] = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
            except Exception:
                log.exception("Could not reopen the collection; upload a document again to rebuild it")
            load_chunk_texts.cache_clear()  # keyword search must read the new chunks
        chunks = state["collection"].count()

    seconds = time.perf_counter() - start
    log.info("Indexed %s: %d chunks in %.0fs", filename, chunks, seconds)
    return DocumentResponse(document=filename, chunks=chunks, seconds=round(seconds, 1))


# `python src/api.py` starts the server directly (same as `uvicorn api:app --app-dir src`)
if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
