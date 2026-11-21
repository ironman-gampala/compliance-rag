"""Read PDFs from ./docs, chunk each page, embed, and store in Chroma."""

import argparse
import hashlib
import sys
from pathlib import Path

import pymupdf

from rag.catalog import labels_for, load_catalog
from rag.chunking import chunk_text
from rag.config import CHUNK_OVERLAP, CHUNK_TOKENS, DOCS_DIR
from rag.embeddings import embed_passages, get_embedder
from rag.store import add_chunks, delete_source, get_collection, reset_collection
from rag.types import Chunk


def load_pages(path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for pages that have extractable text.

    Page numbers are 1-based, matching the number printed by a PDF viewer.
    """
    pages = []
    with pymupdf.open(path) as document:
        for index, page in enumerate(document):
            text = page.get_text("text").strip()
            if text:
                pages.append((index + 1, text))
    return pages


def build_chunks(path: Path, tokenizer, catalog: dict) -> list[Chunk]:
    meta = labels_for(path.name, catalog)
    chunks = []
    for page_number, page_text in load_pages(path):
        pieces = chunk_text(page_text, tokenizer, CHUNK_TOKENS, CHUNK_OVERLAP)
        for index, piece in enumerate(pieces):
            digest = hashlib.sha1(f"{path.name}\n{page_number}\n{index}\n{piece}".encode()).hexdigest()[:12]
            chunks.append(
                Chunk(
                    id=f"{path.name}::p{page_number}::c{index}::{digest}",
                    text=piece,
                    source=path.name,
                    page=page_number,
                    region=meta["region"],
                    doc_type=meta["doc_type"],
                )
            )
    return chunks


def pdfs_from_args(paths: list[str]) -> list[Path]:
    if not paths:
        targets = [DOCS_DIR]
    else:
        targets = [Path(path) for path in paths]
    found = []
    for target in targets:
        if target.is_dir():
            found.extend(sorted(target.glob("*.pdf")))
        elif target.is_file() and target.suffix.lower() == ".pdf":
            found.append(target)
        else:
            raise SystemExit(f"Not a PDF or directory: {target}")
    # Stable, unique order.
    unique = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def ingest(paths: list[str], reset: bool = False) -> int:
    files = pdfs_from_args(paths)
    if not files:
        raise SystemExit(f"No PDFs found. Put files in {DOCS_DIR} and run this again.")
    if reset:
        reset_collection()
        print("Cleared the existing index.")

    catalog = load_catalog()
    tokenizer = get_embedder().tokenizer
    collection = get_collection()
    total = 0
    for path in files:
        if path.name not in catalog:
            print(f"Note: {path.name} is not in docs/catalog.json, so region and doc_type are 'unspecified'.")
        chunks = build_chunks(path, tokenizer, catalog)
        if not chunks:
            print(f"{path.name}: no extractable text. A scanned PDF needs OCR before ingest.")
            continue
        delete_source(path.name, collection)
        embeddings = embed_passages([chunk.text for chunk in chunks], show_progress=True)
        add_chunks(chunks, embeddings, collection)
        total += len(chunks)
        pages = len({chunk.page for chunk in chunks})
        print(f"{path.name}: {pages} pages with text, {len(chunks)} chunks ({chunks[0].region}, {chunks[0].doc_type})")
    print(f"Index now holds {collection.count()} chunks.")
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest compliance PDFs into the local Chroma index.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="PDF files or directories. Defaults to ./docs.",
    )
    parser.add_argument("--reset", action="store_true", help="Delete the index before ingesting.")
    args = parser.parse_args(argv)
    try:
        ingest(args.paths, reset=args.reset)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
