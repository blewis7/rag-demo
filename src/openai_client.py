from __future__ import annotations

import os
from typing import List
from openai import OpenAI

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Make sure it is set in .env and load_dotenv() is called."
            )
        _client = OpenAI(api_key=api_key)
    return _client

def embed_texts(texts: List[str]) -> List[List[float]]:
    client = _get_client()
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    resp = client.embeddings.create(
        model=model,
        input=texts,
    )
    return [d.embedding for d in resp.data]

def generate_answer(question: str, context: str) -> str:
    client = _get_client()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2-mini")

    prompt = f"""
You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context contains relevant information that allows you to reasonably answer the question, summarize it.
If the context does not contain the answer, say you don't know.

Cite sources inline like: (source=..., page=...)

Question:
{question}

Context:
{context}
"""

    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return resp.output_text.strip()
