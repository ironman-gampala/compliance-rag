"""Chat window for the compliance documents."""

import pymupdf
import streamlit as st

from rag.answer import answer_question
from rag.catalog import load_catalog
from rag.config import DOCS_DIR, TOP_K
from rag.llm import LLMError
from rag.retrieve import search
from rag.store import get_collection

st.set_page_config(page_title="Compliance Agent", layout="centered")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.4rem; max-width: 760px; }
    header[data-testid="stHeader"] { background: transparent; }
    .mark {
      margin: 0 0 1.6rem;
      padding-bottom: 1.15rem;
      border-bottom: 1px solid color-mix(in srgb, currentColor 14%, transparent);
    }
    .mark .title {
      display: flex;
      align-items: center;
      gap: 0.55rem;
      margin: 0;
      font-size: 2.05rem;
      line-height: 1.05;
      letter-spacing: -0.04em;
      font-weight: 600;
    }
    .mark .bot { width: 0.95em; height: 0.95em; flex: none; }
    .mark .sub {
      margin: 0.55rem 0 0;
      font-size: 1rem;
      line-height: 1.45;
      opacity: 0.72;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        st.write("No pages matched those filters.")
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
    with st.expander("Passages the answer was based on"):
        if not hits:
            st.write("No pages matched those filters.")
            return
        for number, hit in enumerate(hits, start=1):
            st.markdown(f"**[{number}] {hit.label()}** · {hit.region} · {_label(hit.doc_type)}")
            st.write(" ".join(hit.text.split())[:700])


def draw_assistant(answer, hits) -> None:
    st.markdown(answer.text)
    show_citations(answer)
    show_hits(hits)


def _label(value: str) -> str:
    names = {
        "Any": "Any",
        "regulation": "Regulation",
        "statute": "Statute",
        "handbook": "Handbook",
        "guidance": "Guidance",
    }
    return names.get(value, value.replace("_", " ").capitalize())


def main() -> None:
    st.markdown(
        """
        <div class="mark">
          <div class="title">
            <svg class="bot" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 2.5v2.2" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
              <circle cx="12" cy="2.4" r="1" fill="currentColor"/>
              <rect x="4.5" y="6.5" width="15" height="12" rx="3.2" fill="none" stroke="currentColor" stroke-width="1.7"/>
              <circle cx="9.2" cy="12" r="1.15" fill="currentColor"/>
              <circle cx="14.8" cy="12" r="1.15" fill="currentColor"/>
              <path d="M9 16.2h6" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
            </svg>
            <span>Compliance Agent</span>
          </div>
          <p class="sub">Answers from the documents, with the page cited.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    catalog = load_catalog()
    regions = ["Any"] + sorted({meta["region"] for meta in catalog.values()})
    doc_types = ["Any"] + sorted({meta["doc_type"] for meta in catalog.values()})

    with st.sidebar:
        st.header("Narrow the search")
        region = st.selectbox(
            "Country or region",
            regions,
            format_func=_label,
            help="Search only the documents for that country. Any searches everything that is loaded.",
        )
        doc_type = st.selectbox(
            "Kind of document",
            doc_types,
            format_func=_label,
            help="Regulation, statute, handbook, or guidance. Any includes all of them.",
        )
        accurate = st.checkbox("Careful search", value=True)
        st.caption(
            "On by default. Matches both the meaning of your question and the exact words, "
            "then re-reads the best pages before answering. Turn it off for a faster search."
        )
        if get_collection().count() == 0:
            st.warning("No documents are indexed yet. From the project folder, run: python ingest.py")
        else:
            st.divider()
            st.markdown("**Documents**")
            for meta in sorted(catalog.values(), key=lambda item: item["title"]):
                st.markdown(f"{meta['title']}")
                st.caption(f"{meta['region']} · {_label(meta['doc_type'])}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] != "assistant":
                st.markdown(message["content"])
            elif message.get("error"):
                st.error(message["error"])
                show_pages(message["hits"], "Pages found")
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
            st.error("No documents are indexed yet. From the project folder, run: python ingest.py")
            return
        with st.spinner("Looking through the documents..."):
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
                show_pages(hits, "Pages found")
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(exc), "hits": hits, "error": str(exc)}
                )
                return
        draw_assistant(answer, hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "answer": answer, "hits": hits}
    )


main()
