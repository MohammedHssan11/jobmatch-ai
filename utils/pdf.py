"""
Resume PDF -> plain text, using PyMuPDF (fitz).

Kept intentionally small: the Streamlit app hands us bytes, we hand back text.
"""

import fitz


def pdf_bytes_to_text(data: bytes) -> str:
    """Extract text from an in-memory PDF (what st.file_uploader gives us)."""
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def pdf_file_to_text(path: str) -> str:
    """Extract text from a PDF on disk."""
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)
