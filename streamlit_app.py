import os
import time
from typing import Union
import streamlit as st
from dotenv import load_dotenv

from src.pdf_loader import extract_pdf_pages
from src.chunking import chunk_pages
from src.openai_client import embed_texts, generate_answer
from src.pinecone_store import PineconeVectorStore

load_dotenv()

def set_progress(bar, value: Union[int, float], text: str):
    bar.progress(min(100, max(0, int(value))), text=text)

st.set_page_config(page_title="RAG Demo (PDFs + Pinecone + OpenAI)", layout="wide")
st.title("RAG Demo: PDFs → Embeddings → Pinecone → Answer (with citations)")

with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=int(os.getenv("TOP_K", "6")))
    st.caption("Tip: start with 5–8. More isn't always better.")
    st.divider()
    st.markdown("**Environment**")
    st.code(
        "\n".join([
            f"PINECONE_INDEX_NAME={os.getenv('PINECONE_INDEX_NAME','')}",
            f"OPENAI_EMBED_MODEL={os.getenv('OPENAI_EMBED_MODEL','')}",
            f"OPENAI_CHAT_MODEL={os.getenv('OPENAI_CHAT_MODEL','')}",
        ]),
        language="text",
    )

# Initialize Pinecone store once
@st.cache_resource
def get_store():
    return PineconeVectorStore.from_env()

store = get_store()

tabs = st.tabs(["1) Upload & Index", "2) Ask Questions"])

# -------------------- Tab 1: Index --------------------
with tabs[0]:
    st.subheader("Upload PDFs")
    uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        do_index = st.button("Index uploaded PDFs", type="primary", disabled=not uploaded)
    with col2:
        if st.button("Reset chat"):
            st.session_state.pop("messages", None)
            st.toast("Chat reset.")

    if do_index and uploaded:
        progress = st.progress(0, text="Starting...")
        total_files = len(uploaded)
        upserted_total = 0

        for i, uf in enumerate(uploaded, start=1):
            base = ((i - 1) / total_files) * 100
            set_progress(progress, base + 0,  f"Extracting: {uf.name}")
            pages = extract_pdf_pages(uf)

            set_progress(progress, base + 5,  f"Chunking: {uf.name}")
            chunks = chunk_pages(pages, source_name=uf.name)

            # Embed in batches
            set_progress(progress, base + 10, f"Embedding: {uf.name}")
            vectors = embed_texts([c["text"] for c in chunks])

            set_progress(progress, base + 70, f"Upserting: {uf.name}")
            upserted = store.upsert_chunks(chunks, vectors)
            upserted_total += upserted

            set_progress(progress, (i / total_files) * 100, f"Done: {uf.name} (upserted {upserted})")

        st.success(f"Indexing complete. Upserted {upserted_total} chunks.")
        st.info("Now go to **Ask Questions** tab and try queries like: 'What is the PTO policy?'")

# -------------------- Tab 2: Ask --------------------
with tabs[1]:
    st.subheader("Ask a question")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about the indexed PDFs...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving..."):
                qvec = embed_texts([question])[0]
                matches = store.query(qvec, top_k=top_k)

            if not matches:
                st.markdown("I couldn't find anything relevant in the index. Try indexing PDFs first, or rephrase.")
            else:
                # Build context block for the model
                context_blocks = []
                for m in matches:
                    meta = m["metadata"]
                    context_blocks.append(
                        f"[source={meta.get('source')} page={meta.get('page')} chunk_id={meta.get('chunk_id')}]\n{m['text']}"
                    )
                context = "\n\n---\n\n".join(context_blocks)

                with st.spinner("Answering..."):
                    answer = generate_answer(question=question, context=context)

                st.markdown(answer)

                with st.expander("Sources (retrieved chunks)"):
                    for m in matches:
                        meta = m["metadata"]
                        st.markdown(
                            f"**{meta.get('source')} — page {meta.get('page')} — score {m.get('score'):.4f}**"
                        )
                        st.caption(f"chunk_id: {meta.get('chunk_id')}")
                        st.write(m["text"])

        st.session_state.messages.append({"role": "assistant", "content": answer})
