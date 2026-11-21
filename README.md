# compliance-rag

A local chatbot that answers compliance questions from your own PDFs and cites the page each claim came from. The models and the database run on your machine, except the final answer, which can use a free Groq or Gemini key or a local Ollama model.

- [SETUP.md](SETUP.md) — install, PDFs, the index, and how to run the chat
- [ARCHITECTURE.md](ARCHITECTURE.md) — what was built, and how a question becomes a cited answer

This README walks through the pipeline in the order the code runs. Each stage has one job. The next stage only sees the previous stage's output.

Phase 2 is not in this repo: packaging the bot as a plugin, and an agent that searches in several steps or compares regions on its own.

## Setup

Install, configure `.env`, and build the index with the steps in [SETUP.md](SETUP.md).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python ingest.py
streamlit run app.py
```

The PDFs themselves are not committed. The index is also local (`chroma_db/`) and is not committed. The first ingest downloads `BAAI/bge-small-en-v1.5` (about 130 MB). The first rerank downloads `BAAI/bge-reranker-base`.

Set one provider in `.env`:

| `LLM_PROVIDER` | What you need |
| --- | --- |
| `groq` | `GROQ_API_KEY` from the Groq console. Set `GROQ_MODEL` to a current free-tier model id. |
| `gemini` | `GEMINI_API_KEY` from Google AI Studio. |
| `ollama` | [Ollama](https://ollama.com) running locally, and `ollama pull llama3.2`. No key. |

## Stage 1 — Load each page

`ingest.py` opens every PDF with PyMuPDF and reads one page at a time. PyMuPDF is used instead of Docling because it is small, free, and the page index it returns is the same number you see in a PDF viewer. Docling is the better tool later if you need tables or a scanned layout reconstructed.

Each page keeps two facts that citations need: the file name and the page number (starting at 1). Pages with no text are skipped. A scan that is only an image will produce nothing here; this project does not OCR.

```bash
python ingest.py docs/til.pdf          # one file, useful the first time
python ingest.py                        # every PDF in docs/
python ingest.py --reset                # delete the index, then ingest docs/
```

## Stage 2 — Chunk

An embedding model cannot take a whole handbook in one vector, and a vector of an entire page often mixes several topics so the nearest-neighbor search gets muddy. `rag/chunking.py` cuts each page into windows of **500 tokens** with an **80-token overlap**.

500 is the low end of the 500–800 range on purpose. `bge-small` discards anything past 512 tokens, so a 700-token chunk would be stored with its tail missing. The overlap keeps a sentence that lands on a cut inside at least one full window. Chunks do not cross a page boundary, so a chunk has exactly one page number to cite.

The tokenizer is the one that shipped with `bge-small`, so "500 tokens" means the same tokens the embedder will see. The stored text is the original page text for that span, not the tokenizer's decoded text, so capitalization and section numbers stay as they are in the PDF.

## Stage 3 — Embed

`rag/embeddings.py` loads `BAAI/bge-small-en-v1.5` through `sentence-transformers` and turns every chunk into a 384-number vector. Similar meanings land near each other, which is why a question phrased "cap on late fees" can find a page that says "penalty fee … for paying late".

The model asks you to put a prefix on the **question only**:

```text
Represent this sentence for searching relevant passages:
```

Passages are embedded as plain text. Vectors are normalized so Chroma's cosine distance is a fair comparison.

## Stage 4 — Store

`rag/store.py` writes the chunks to a [Chroma](https://www.trychroma.com/) database in `chroma_db/`. Chroma does not embed anything here; the vectors from stage 3 are stored as-is. Each record holds:

- the chunk text
- the vector
- metadata: `source`, `page`, `region`, `doc_type`

Re-running ingest for a file deletes that file's old chunks first, so you do not get duplicates.

## Stage 5 — Retrieve

`query.py` embeds the question and asks Chroma for the nearest chunks. The default, which is step 1, returns the **top 5**.

```bash
python query.py "In the US region, what is the cap on late fees for a credit card?" --region US
python query.py "In the US region, what is the cap on late fees for a credit card?" --region US --retrieve-only
```

`--retrieve-only` prints the chunks and does not call an LLM. Use it when you want to see what the search did, or when you have not set an API key yet.

### Filters

`--region US` and `--doc-type regulation` are passed to Chroma as metadata filters, before similarity is computed. A UK-only search cannot return `til.pdf`, because that file was labeled `US` at ingest. The labels come from `docs/catalog.json`.

### Hybrid search

Vector search is weak when the important part of the question is a literal string: a section number like `1026.52`, or a defined term. BM25 is the opposite: it scores shared words and ignores meaning.

`--hybrid` runs both, takes a shortlist of 20 from each, and merges them with **reciprocal rank fusion**. Fusion uses rank rather than the raw score, because a BM25 score of 12 and a cosine similarity of 0.7 are not on the same scale. A chunk that is near the top of both lists beats a chunk that only one list likes.

### Rerank

The bi-encoder that built the index reads the question and the chunk separately. A **cross-encoder** (`BAAI/bge-reranker-base`) reads them together and scores the pair. That is more accurate and much slower, so it only runs on the shortlist, not on the whole index. `--rerank` keeps the top 5 after that rescore.

```bash
python query.py "..." --region US --hybrid --rerank
```

## Stage 6 — Answer, and refuse to guess

`rag/answer.py` builds a prompt that contains only those chunks, numbered `[1]`, `[2]`, and so on. The instructions tell the model:

- use only the excerpts
- cite the excerpt numbers it used
- if several limits in the excerpts apply, apply all of them and use the strictest one
- if the excerpts do not state the rules needed to answer, reply with exactly `Not found in the documents`

If the filter matches nothing, the code returns that sentence **without calling the model**. There is nothing to ground an answer in, and a model with an empty prompt will improvise.

The model cites numbers, not file names. `query.py` prints the file and page for each number it cited:

```text
The first penalty fee for paying late cannot be more than $25 [1].

Sources:
[1] til.pdf, page 78
```

Resolving the number in code means a citation cannot point at a file the model invented. The page image in the chat UI uses the same mapping.

## Stage 7 — Eval

`eval/questions.jsonl` is a small set of questions with known pages. `eval/score.py` checks whether retrieval put one of those pages in the top 5. With `--answers` it also calls the LLM and checks that the reply contains the expected phrases, or the exact not-found sentence.

```bash
python eval/score.py
python eval/score.py --hybrid --rerank
python eval/score.py --hybrid --rerank --answers
```

One question asks for the US late-fee cap while the region filter is `UK`. Retrieval fails if `til.pdf` still shows up. That is how you know the filter is applied before search, not after.

## Stage 8 — Chat UI

```bash
streamlit run app.py
```

The sidebar sets the country, the kind of document, and whether careful search is on. Careful search is hybrid search plus the reranker. The answer is followed by a citation. Open it to see that PDF page rendered from the file in `docs/`. "Passages the answer was based on" shows every passage the model was given, including ones the answer did not cite.

The UI reads `LLM_PROVIDER` from `.env`. A missing key is shown as an error in the chat rather than a guessed answer, and the pages search found are still listed so you can open them. Hybrid search can hand the model more than five chunks when a fuller subsection has to come along. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Project layout

```text
ingest.py            load, chunk, embed, store
query.py             retrieve and answer from the command line
app.py               Streamlit chat
rag/                 the pipeline the three commands share
eval/questions.jsonl known questions
eval/score.py        retrieval and answer scores
docs/                your PDFs and catalog.json
SETUP.md             install and run
ARCHITECTURE.md      how the pipeline fits together
.env.example         Groq, Gemini, or Ollama
```

## Tests that do not need a model

```bash
python -m unittest tests/test_chunking.py tests/test_answer.py
```

These check the overlap math, the not-found short circuit, and rank fusion. They do not download a model or call an API.

## Limits

- The answer can only be as current as the PDF. Regulation Z's $25 / $35 penalty-fee safe harbor is adjusted over time; the bot will quote the copy in `docs/`, not today's figure.
- Two columns on a page sometimes come out of PyMuPDF in a jumbled order.
- Hybrid search loads every matching chunk to score BM25. That is fine for this folder of handbooks and a poor fit for millions of pages.
- The reranker runs on CPU unless PyTorch sees a GPU. The first question with the checkbox on is slow while the model downloads and scores five pairs.

## Phase 2, not built

- A plugin wrapper so another app can call this index.
- Agentic RAG: the model plans several searches, then compares regions, instead of one retrieve-and-answer step.
