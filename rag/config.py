"""Paths and settings shared by ingest, query, and the chat UI."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Avoid a noisy warning when the embedding tokenizer forks.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DOCS_DIR = ROOT / "docs"
CHROMA_PATH = ROOT / os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = "compliance"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
# bge-small tells you to put this in front of the question only, not the passage.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# bge-small truncates at 512 tokens, so stay under that with room for special tokens.
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 80
TOP_K = int(os.getenv("TOP_K", "5"))
# How many candidates hybrid search and the reranker look at before cutting to TOP_K.
CANDIDATE_POOL = 20

NOT_FOUND = "Not found in the documents"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
