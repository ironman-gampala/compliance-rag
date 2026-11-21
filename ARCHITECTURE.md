# Architecture

compliance-rag answers a compliance question from the PDFs in `docs/` and cites the page. It does not browse the web, and it does not answer from the model's own memory of the law. If the retrieved pages do not state a rule that governs the question, the reply is exactly `Not found in the documents`.

Three commands share one pipeline: `ingest.py` builds the index, `query.py` answers on the command line, and `app.py` is the Streamlit chat. The stages live in `rag/`.

```text
PDF page
  -> chunk (500 tokens, 80 overlap, one page)
  -> bge-small vector
  -> Chroma (source, page, region, doc_type)
  -> vector search, optional BM25 fusion, optional rerank
  -> extra chunks that carry a fuller subsection
  -> LLM, excerpts only
  -> [n] resolved in code to file + page
```

## What was built

Step 1 is a local index and a grounded answer. PyMuPDF reads each page. Chunks stay under the embedding model's 512-token limit and do not cross a page, so every citation has one page number. `BAAI/bge-small-en-v1.5` embeds the chunks. Chroma stores the vectors. The question prefix `Represent this sentence for searching relevant passages:` is applied to the question only.

Step 2 adds accuracy on top of that index. A region or document-type filter is applied inside Chroma before similarity is computed, so a UK search cannot return a US file. Hybrid search runs BM25 beside the vector search and merges the two lists with reciprocal rank fusion. `BAAI/bge-reranker-base` then rescores a shortlist of 20. `eval/score.py` checks that known questions retrieve the expected page, and with `--answers` that the written reply contains the expected phrases.

Step 3 is the chat. Streamlit shows the answer, opens a cited page as an image rendered from the PDF, and lists every chunk the model was given.

Not built, on purpose: a plugin wrapper, and an agent that plans several searches.

## How a question is answered

1. `rag/retrieve.py` embeds the question and asks Chroma for the nearest chunks. With hybrid search it also scores every chunk in the filter with BM25 and fuses the lists. With the reranker on, the cross-encoder reorders that shortlist and the top 5 are kept.
2. Two small follow-ups add a subsection the first pass dropped. A handbook often explains a rule in prose ("Section 1026.5(b)") and writes the exact subsection on a later page ("§1026.5(b)(2)(ii)"). `supplement_with_specific_sections` keeps a more specific label that fusion already ranked just below the cutoff. `expand_related_subsections` looks for a section number sitting in the same paragraph as words from the question (a due date, a late payment, a cut-off) and pulls the fullest subsection of that section. The window stops at the next section number, so a contents list does not drag in every section on the page.
3. `rag/answer.py` sends only those numbered excerpts. The instructions say to apply every rule that governs the facts, including two different sections, and to use the stricter limit when two limits both apply. Dollar figures in the question describe the scenario. They are not a legal cap unless an excerpt says they are.
4. The model cites excerpt numbers like `[1]`. `query.py` and the chat turn each number into `filename, page`. A section number the model quotes is attached to the excerpt that actually contains it, when the model forgot the bracket. The model cannot invent a file name.

An empty retrieval never calls the model. A missing API key is shown as an error, and the pages that were found are still listed.

## Where the facts live

| Piece | Where | Committed? |
| --- | --- | --- |
| PDF text | `docs/*.pdf` | No |
| Region and document type | `docs/catalog.json` | Yes |
| Chunks and vectors | `chroma_db/` | No. Rebuild with `python ingest.py`. |
| Embedding and reranker weights | Hugging Face cache | No. Downloaded on first use. |
| Answer model | Groq, Gemini, or Ollama, from `.env` | The key is not committed. |

The answer is only as current as the PDF. A later inflation adjustment that is not in the file will not appear in the reply.

## Chat filters

The sidebar region and document type become the Chroma `where` clause. Hybrid search and the reranker default to on, which is the path the eval set is scored on. Vector-only search is the original step 1 pipeline and is what you get with the checkbox off, or with `query.py` and no flags.
