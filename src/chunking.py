import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent  # project root
JSON_PATH = ROOT / "data" / "chatgpt.json"
OUT_PATH = ROOT / "document_chunks.jsonl"
CHUNK_SIZE = 1000     # max characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks


def _text_from_content(content) -> list[str]:
    if isinstance(content, list):
        return [text for item in content for text in _text_from_content(item)]
    if isinstance(content, dict):
        if content.get("type") == "text" and isinstance(content.get("content"), str):
            return [content["content"]]
        return _text_from_content(content.get("content", []))
    return []


def main(json_path):
    json_path = Path(json_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # paragraph -> line -> sentence -> word -> char
    )

    document = json.loads(json_path.read_text(encoding="utf-8"))
    chunks = []
    for page in document["pages"]:
        page_text = "\n".join(
            text
            for block in page["blocks"]
            for text in _text_from_content(block)
        )
        chunks.extend(
            {"page": page["page_idx"] + 1, "text": chunk, "source": f"{json_path.stem}.pdf"}
            for chunk in splitter.split_text(page_text)
            if chunk.strip()
        )

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(json.dumps({"id": i, **chunk}, ensure_ascii=False) + "\n")

    print(f"Pages:      {len(document['pages'])}")
    print(f"Chunks:     {len(chunks)}")
    print(f"Saved:      {OUT_PATH}")
    if chunks:
        print(f"\n--- chunk 0 (page {chunks[0]['page']}) ---\n{chunks[0]['text'][:300]}")


if __name__ == "__main__":
    main(JSON_PATH)
