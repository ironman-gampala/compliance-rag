"""Score retrieval, and optionally the written answer, against eval/questions.jsonl.

Retrieval passes when a known source page is in the top results (or, for a
filter check, when a forbidden file stays out). Answer scoring calls the LLM
configured in .env and is optional, because it needs a key.

    python eval/score.py
    python eval/score.py --hybrid --rerank
    python eval/score.py --hybrid --rerank --answers
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.answer import NOT_FOUND, answer_question
from rag.config import TOP_K
from rag.llm import LLMError
from rag.retrieve import search
from rag.store import get_collection

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"


def load_questions() -> list[dict]:
    questions = []
    for line in QUESTIONS.read_text().splitlines():
        line = line.strip()
        if line:
            questions.append(json.loads(line))
    return questions


def retrieval_ok(hits, item: dict) -> bool:
    forbidden = set(item.get("forbid_sources") or [])
    if any(hit.source in forbidden for hit in hits):
        return False
    sources = item.get("expect_sources") or []
    pages = item.get("expect_pages") or []
    if not sources and not pages:
        return True
    for hit in hits:
        source_ok = not sources or hit.source in sources
        page_ok = not pages or hit.page in pages
        if source_ok and page_ok:
            return True
    return False


def answer_ok(answer, item: dict) -> bool:
    if item.get("expect_not_found"):
        return answer.text.strip() == NOT_FOUND
    text = answer.text.lower()
    return all(phrase.lower() in text for phrase in item.get("answer_must_include") or [])


def citation_ok(answer, item: dict) -> bool:
    if item.get("expect_not_found"):
        return answer.cited == []
    sources = item.get("expect_sources") or []
    pages = item.get("expect_pages") or []
    if not sources and not pages:
        return True
    for _, hit in answer.sources():
        source_ok = not sources or hit.source in sources
        page_ok = not pages or hit.page in pages
        if source_ok and page_ok:
            return True
    return False


def describe(hits) -> str:
    if not hits:
        return "(no chunks)"
    return ", ".join(f"{hit.source} p.{hit.page}" for hit in hits)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the eval questions.")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--answers", action="store_true", help="Also call the LLM and score the written answer.")
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args(argv)

    if get_collection().count() == 0:
        raise SystemExit("The index is empty. Run: python ingest.py")

    questions = load_questions()
    retrieval_hits = 0
    answer_hits = 0
    citation_hits = 0
    for item in questions:
        hits = search(
            item["question"],
            k=args.k,
            region=item.get("region"),
            doc_type=item.get("doc_type"),
            hybrid=args.hybrid,
            rerank=args.rerank,
        )
        got_retrieval = retrieval_ok(hits, item)
        retrieval_hits += int(got_retrieval)
        mark = "PASS" if got_retrieval else "FAIL"
        print(f"{mark}  retrieval  {item['id']}")
        if not got_retrieval:
            print(f"       got {describe(hits)}")

        if args.answers:
            try:
                answer = answer_question(item["question"], hits)
            except LLMError as exc:
                raise SystemExit(str(exc)) from exc
            got_answer = answer_ok(answer, item)
            got_citation = citation_ok(answer, item)
            answer_hits += int(got_answer)
            citation_hits += int(got_citation)
            print(f"{'PASS' if got_answer else 'FAIL'}  answer     {item['id']}")
            print(f"{'PASS' if got_citation else 'FAIL'}  citation   {item['id']}")
            if not got_answer or not got_citation:
                print(f"       {answer.text[:300].replace(chr(10), ' ')}")

    total = len(questions)
    print(f"\nRetrieval {retrieval_hits}/{total}")
    if args.answers:
        print(f"Answers   {answer_hits}/{total}")
        print(f"Citations {citation_hits}/{total}")
        if answer_hits < total or citation_hits < total or retrieval_hits < total:
            sys.exit(1)
    elif retrieval_hits < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
