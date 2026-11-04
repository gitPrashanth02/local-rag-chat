import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent  # project root
CHUNKS_PATH = ROOT / "document_chunks.jsonl"
DB_PATH = str(ROOT / "chroma_db")                        # folder where ChromaDB saves data
COLLECTION_NAME = "chatgpt"                  # like a "table" inside the database

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"     # 4B / 8B are better but need a strong GPU
BATCH_SIZE = 16


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # 1. Read chunks
    chunks = load_chunks(CHUNKS_PATH)
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks from {CHUNKS_PATH}")

    # 2. Embed chunks (documents need no prompt; only questions use prompt_name="query")
    print(f"Loading model {MODEL_NAME} (first time downloads it)...")
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # 3. Open the database (creates the folder if it doesn't exist)
    client = chromadb.PersistentClient(path=DB_PATH)

    # Delete the old collection so re-running doesn't create duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",   # compare vectors by cosine similarity
            "model": MODEL_NAME,      # retrieval.py must use this same model
        },
    )

    # 4. Store vectors + texts + metadata
    collection.add(
        ids=[str(c["id"]) for c in chunks],
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=[{"chunk_id": c["id"], "page": c["page"],"source": c["source"]} for c in chunks],
    )

    print(f"\nStored:     {collection.count()} chunks")
    print(f"Dimension:  {vectors.shape[1]}")
    print(f"Database:   {DB_PATH}/  (collection '{COLLECTION_NAME}')")


if __name__ == "__main__":
    main()