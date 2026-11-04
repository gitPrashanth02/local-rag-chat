from pathlib import Path

from mineru.parser import parse

ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = ROOT / "data"


def main(pdf_path):
    pdf_path = Path(pdf_path)
    result = parse(str(pdf_path), tier="flash")
    markdown_text = result.markdown()
    json_result = result.to_json()

    (DATA_DIR / f"{pdf_path.stem}.md").write_text(markdown_text, encoding="utf-8")
    (DATA_DIR / f"{pdf_path.stem}.json").write_text(json_result, encoding="utf-8")

    print(f"Characters: {len(markdown_text)}")
    print(markdown_text[:500])


if __name__ == "__main__":
    main(DATA_DIR / "chatgpt.pdf")