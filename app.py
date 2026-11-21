"""Chat window for the compliance documents."""

import pymupdf
import streamlit as st

from rag.answer import answer_question
from rag.catalog import load_catalog
from rag.config import DOCS_DIR, TOP_K
from rag.llm import LLMError
from rag.retrieve import search
from rag.store import get_collection

st.set_page_config(page_title="Compliance RAG", layout="wide")


def render_page(source: str, page: int) -> bytes:
    path = (DOCS_DIR / source).resolve()
    if not path.is_relative_to(DOCS_DIR.resolve()) or not path.exists():
        raise FileNotFoundError(source)
    with pymupdf.open(path) as document:
        pixmap = document[page - 1].get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4), alpha=False)
        return pixmap.tobytes("png")


def show_citations(answer) -> None:
    sources = answer.sources()
    if not sources:
        return
    st.markdown("**Sources**")
    for number, hit in sources:
        with st.expander(f"[{number}] {hit.label()}"):
            try:
                st.image(render_page(hit.source, hit.page), caption=hit.label())
            except FileNotFoundError:
                st.warning(f"Could not open {hit.source} in docs/.")
            st.caption(" ".join(hit.text.split())[:500])


def show_pages(hits, heading: str) -> None:
    if not hits:
        st.write("No chunks matched the filters.")
        return
    st.markdown(f"**{heading}**")
    for number, hit in enumerate(hits, start=1):
        with st.expander(f"[{number}] {hit.label()}"):
            try:
                st.image(render_page(hit.source, hit.page), caption=hit.label())
            except FileNotFoundError:
                st.warning(f"Could not open {hit.source} in docs/.")
            st.caption(" ".join(hit.text.split())[:500])


def show_hits(hits) -> None:
    with st.expander("Chunks the model was given"):
        if not hits:
            st.write("No chunks matched the filters.")
            return
        for number, hit in enumerate(hits, start=1):
            st.markdown(f"**[{number}] {hit.label()}** · {hit.region} / {hit.doc_type}")
            st.write(" ".join(hit.text.split())[:700])


def draw_assistant(answer, hits) -> None:
    st.markdown(answer.text)
    show_citations(answer)
    show_hits(hits)


def main() -> None:
    st.title("Compliance RAG")
    st.caption("Answers come only from the PDFs in docs/. Open a citation to see that page.")

    catalog = load_catalog()
    regions = ["Any"] + sorted({meta["region"] for meta in catalog.values()})
    doc_types = ["Any"] + sorted({meta["doc_type"] for meta in catalog.values()})

    with st.sidebar:
        st.header("Search")
        region = st.selectbox("Region", regions)
        doc_type = st.selectbox("Document type", doc_types)
        accurate = st.checkbox("Hybrid search and reranker", value=True)
        st.caption("Uncheck this to use vector search only, which is the step 1 pipeline.")
        if get_collection().count() == 0:
            st.warning("Index is empty. Run python ingest.py in a terminal.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] != "assistant":
                st.markdown(message["content"])
            elif message.get("error"):
                st.error(message["error"])
                show_pages(message["hits"], "Retrieved pages")
            else:
                draw_assistant(message["answer"], message["hits"])

    question = st.chat_input("Ask a compliance question")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if get_collection().count() == 0:
            st.error("The index is empty. Run: python ingest.py")
            return
        with st.spinner("Searching the documents..."):
            try:
                hits = search(
                    question,
                    k=TOP_K,
                    region=None if region == "Any" else region,
                    doc_type=None if doc_type == "Any" else doc_type,
                    hybrid=accurate,
                    rerank=accurate,
                )
                answer = answer_question(question, hits)
            except LLMError as exc:
                st.error(str(exc))
                show_pages(hits, "Retrieved pages")
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(exc), "hits": hits, "error": str(exc)}
                )
                return
        draw_assistant(answer, hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "answer": answer, "hits": hits}
    )


main()
