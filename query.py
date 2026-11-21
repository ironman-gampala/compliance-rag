"""Ask a question and answer it from the indexed PDFs."""

import argparse
import sys

from rag.answer import answer_question, format_answer
from rag.config import NOT_FOUND, TOP_K
from rag.llm import LLMError
from rag.retrieve import search
from rag.store import get_collection


def ask(
    question: str,
    k: int = TOP_K,
    region: str | None = None,
    doc_type: str | None = None,
    hybrid: bool = False,
    rerank: bool = False,
    retrieve_only: bool = False,
):
    if get_collection().count() == 0:
        raise SystemExit("The index is empty. Run: python ingest.py")
    hits = search(
        question,
        k=k,
        region=region,
        doc_type=doc_type,
        hybrid=hybrid,
        rerank=rerank,
    )
    if retrieve_only:
        return hits
    return answer_question(question, hits)


def _print_hits(hits) -> None:
    if not hits:
        print(NOT_FOUND)
        return
    for number, hit in enumerate(hits, start=1):
        score = "" if hit.score is None else f"  score={hit.score:.3f}"
        print(f"[{number}] {hit.label()}  {hit.region}/{hit.doc_type}{score}")
        preview = " ".join(hit.text.split())
        print(preview[:400])
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Answer a compliance question from the local PDFs.")
    parser.add_argument("question", help="Question to answer from the documents.")
    parser.add_argument("--region", help="Metadata filter, for example US or UK.")
    parser.add_argument("--doc-type", help="Metadata filter, for example regulation, statute, handbook, guidance.")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of chunks to give the model.")
    parser.add_argument("--hybrid", action="store_true", help="Fuse BM25 keyword search with vector search.")
    parser.add_argument("--rerank", action="store_true", help="Rescore the shortlist with bge-reranker.")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Print the retrieved chunks and do not call an LLM.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the chunks after the answer.",
    )
    args = parser.parse_args(argv)
    try:
        result = ask(
            args.question,
            k=args.k,
            region=args.region,
            doc_type=args.doc_type,
            hybrid=args.hybrid,
            rerank=args.rerank,
            retrieve_only=args.retrieve_only,
        )
    except LLMError as exc:
        raise SystemExit(str(exc)) from exc
    if args.retrieve_only:
        _print_hits(result)
        return
    print(format_answer(result))
    if args.show_context:
        print("\nRetrieved context:")
        _print_hits(result.hits)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
