from __future__ import annotations

import hashlib
from typing import List, Dict, Any

def _stable_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(str(page).encode("utf-8"))
    h.update(str(chunk_index).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:24]

def chunk_pages(
    pages: List[Dict[str, Any]],
    source_name: str,
    max_chars: int = 1200,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Simple character-based chunking per page.
    Keeps page numbers for citations.
    """
    chunks: List[Dict[str, Any]] = []
    for p in pages:
        page_num = p["page"]
        text = (p["text"] or "").strip()
        if not text:
            continue

        start = 0
        chunk_i = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = _stable_chunk_id(source_name, page_num, chunk_i, chunk_text)
                chunks.append({
                    "id": chunk_id,  # used as Pinecone vector id
                    "text": chunk_text,
                    "metadata": {
                        "source": source_name,
                        "page": page_num,
                        "chunk_id": chunk_id,
                    }
                })
                chunk_i += 1
            if end == len(text):
                break
            start = max(0, end - overlap)

    return chunks
