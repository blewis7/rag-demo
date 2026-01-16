# Streamlit RAG (Synthetic PDFs) + Pinecone + OpenAI

This project is a **Streamlit** RAG app:

- Upload PDFs
- Extract per-page text (no OCR)
- Chunk + embed with OpenAI embeddings
- Store/search vectors in Pinecone
- Answer with citations (doc + page)

## Quickstart

1. Create and activate a venv

```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
.venv\Scripts\activate
```

2. Install deps

```bash
pip install -r requirements.txt
```

3. Create `.env` from the example and fill in keys

```bash
cp .env.example .env
```

4. Run Streamlit

```bash
streamlit run streamlit_app.py
```

## Notes

- Uses `text-embedding-3-small` (dimension 1536).
- Uses Pinecone **serverless** index creation if the index does not exist.
