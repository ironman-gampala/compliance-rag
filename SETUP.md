# Setup

Python 3.13. PyTorch and Chroma install with the packages below.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Documents

Put PDFs in `docs/`. One row in `docs/catalog.json` per file:

| Field | Meaning |
| --- | --- |
| `region` | `US`, `UK`, or another label the chat filter uses |
| `doc_type` | `regulation`, `statute`, `handbook`, or `guidance` |
| `title` | Name shown only to you. Search uses the file name and page. |

A PDF missing from the catalog is still indexed, with region and doc type `unspecified`.

The PDFs are not in git. After a clone, copy them into `docs/` yourself. The catalog in the repo lists the six files this project was built against:

- `til.pdf` — Regulation Z handbook (US regulation)
- `credit-card-pub-l-111-24_0.pdf` — Credit CARD Act of 2009 (US statute)
- `pub-ch-credit-card.pdf` — OCC credit card lending handbook (US handbook)
- `PCI_Compliance_Guidance_091720.pdf` — PCI guidance (US guidance)
- `CONC.pdf` — FCA CONC consumer credit sourcebook (UK handbook)
- `12-1264-consumer-credit-directive-guidance.pdf` — Consumer Credit Directive guidance (UK guidance)

## Index

```bash
python ingest.py docs/til.pdf    # one file, useful the first time
python ingest.py                 # every PDF in docs/
python ingest.py --reset         # delete the index, then ingest docs/
```

The first ingest downloads `BAAI/bge-small-en-v1.5` (about 130 MB). The index is written to `chroma_db/` and is not committed. A fresh clone has an empty index until you run ingest.

## Model that writes the answer

Set one provider in `.env`. The key stays on your machine. `.env` is gitignored.

| `LLM_PROVIDER` | What you need |
| --- | --- |
| `groq` | `GROQ_API_KEY` from the Groq console. Set `GROQ_MODEL` to a current free-tier model id. |
| `gemini` | `GEMINI_API_KEY` from Google AI Studio. This project was run with `GEMINI_MODEL=gemini-3.1-flash-lite`. |
| `ollama` | [Ollama](https://ollama.com) running locally, and `ollama pull llama3.2`. No key. |

## Ask a question

```bash
python query.py "In the US region, what is the cap on late fees for a credit card?" --region US
python query.py "..." --region US --hybrid --rerank
python query.py "..." --retrieve-only
streamlit run app.py
```

The chat is at http://localhost:8501. In the sidebar, set **Region** to **US** or **Any** for the Regulation Z handbook. **UK** hides that file, so a US rule comes back as "Not found in the documents". Leave **Hybrid search and reranker** on. The first reranked question downloads `BAAI/bge-reranker-base`.

## Check the index

```bash
python -m unittest tests/test_chunking.py tests/test_answer.py
python eval/score.py --hybrid --rerank
python eval/score.py --hybrid --rerank --answers
```

The unit tests do not download a model or call an API. `--answers` calls the LLM once per question in `eval/questions.jsonl`.
