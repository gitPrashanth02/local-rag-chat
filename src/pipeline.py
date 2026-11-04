
import sys
from pathlib import Path

from generation import generate
from retrieval import RETRIEVAL_TOP_K, TOP_K, hybrid_search, rerank

ROOT = Path(__file__).resolve().parent.parent  # project root
PDF_PATH = ROOT / "data" / "chatgpt.pdf"
#JSON_PATH = ROOT / "data" / "chatgpt.json"
CHUNKS_PATH = ROOT / "document_chunks.jsonl"
INDEX_MARKER = ROOT / "chroma_db" / "index_built.txt"   # written after embedding finishes
CURRENT_DOC_FILE = ROOT / "data" / "current_document.txt"  # name of the PDF that is indexed now

NOT_FOUND = "I couldn't find this in the document."


# ---------------------------------------------------------------------------
# 1. Indexing pipeline
# ---------------------------------------------------------------------------
def _needs_update(output: Path, source: Path) -> bool:
    """True if `output` is missing or older than `source`."""
    return not output.exists() or output.stat().st_mtime < source.stat().st_mtime


def current_document() -> Path:
    """The PDF that was indexed last (saved by build_index), or chatgpt.pdf if none was saved."""
    if CURRENT_DOC_FILE.exists():
        pdf = ROOT / "data" / CURRENT_DOC_FILE.read_text(encoding="utf-8").strip()
        if pdf.exists():
            return pdf
    return PDF_PATH


def build_index(pdf_path=None, force=False):
    """Run parse -> chunk -> embed, skipping each step whose output is up to date."""
    pdf = Path(pdf_path) if pdf_path else current_document()
    json_path = pdf.parent / f"{pdf.stem}.json"
    if not pdf.exists():
        raise FileNotFoundError(f"Source document not found: {pdf}")

    # The step modules are imported only when needed: MinerU and the models are slow to load.
    if force or _needs_update(json_path, pdf):
        print("\n[1/3] Parsing PDF...")
        import document_parser
        document_parser.main(pdf)
    else:
        print("[1/3] Parse:  up to date, skipped")

    if force or _needs_update(CHUNKS_PATH, json_path):
        print("\n[2/3] Chunking...")
        import chunking
        chunking.main(json_path)
    else:
        print("[2/3] Chunk:  up to date, skipped")

    if force or _needs_update(INDEX_MARKER, CHUNKS_PATH):
        print("\n[3/3] Embedding...")
        import embedding
        embedding.main()
        INDEX_MARKER.write_text("Index built from document_chunks.jsonl\n", encoding="utf-8")
    else:
        print("[3/3] Embed:  up to date, skipped")

    CURRENT_DOC_FILE.write_text(pdf.name, encoding="utf-8")  # remember it for the next start


# ---------------------------------------------------------------------------
# 2. Question pipeline
# ---------------------------------------------------------------------------
def answer(question: str, collection, model, top_k: int = TOP_K) -> dict:
    """Find the best chunks for a question and generate an answer from them.

    Returns {"answer": str, "hits": list of chunk dicts}.
    """
    hits = hybrid_search(question, collection, model, top_k=RETRIEVAL_TOP_K)  # many candidates
    hits = rerank(question, hits, top_k=top_k)                                # keep the best
    if not hits:
        return {"answer": NOT_FOUND, "hits": []}
    return {"answer": generate(question, hits), "hits": hits}


if __name__ == "__main__":
    build_index(force="--force" in sys.argv)
