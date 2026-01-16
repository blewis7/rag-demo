from __future__ import annotations

from typing import List, Dict, Any
from pypdf import PdfReader

def extract_pdf_pages(uploaded_file) -> List[Dict[str, Any]]:
    """
    Extract text per page from a Streamlit UploadedFile.
    Returns: [{ 'page': 1, 'text': '...', 'source': 'file.pdf' }, ...]
    """
    reader = PdfReader(uploaded_file)
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # normalize whitespace a bit
        text = " ".join(text.split())
        pages.append({"page": idx, "text": text})
    return pages

