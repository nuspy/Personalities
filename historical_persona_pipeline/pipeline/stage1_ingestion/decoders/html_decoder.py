"""Decoder HTML.

La versione precedente apriva sempre in UTF-8 (le pagine storiche sono spesso
ISO-8859-1) e teneva menu, barre laterali e pie' di pagina: boilerplate che
falsa sia il vocabolario sia le collocazioni. Qui gli elementi di navigazione
vengono scartati e il contenuto principale isolato quando riconoscibile.
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseDecoder, DecodeResult, read_text_with_fallback
from ..chunker import chunk_text

_STRIP_TAGS = (
    "script", "style", "nav", "header", "footer", "aside",
    "form", "noscript", "iframe", "svg", "button",
)

# Contenitori che, se presenti, delimitano il testo vero della pagina.
_CONTENT_SELECTORS = ("article", "main", '[role="main"]', "#content", ".content")


class HTMLDecoder(BaseDecoder):
    extensions = (".html", ".htm", ".xhtml")
    display_name = "Documento HTML"

    def __init__(self, target_chunk_chars: int = 4000, min_chunk_chars: int = 200):
        self.target_chunk_chars = target_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def decode(self, file_path: Path) -> DecodeResult:
        from bs4 import BeautifulSoup

        raw, errors = read_text_with_fallback(file_path)
        soup = BeautifulSoup(raw, "html.parser")

        metadata = {}
        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta and author_meta.get("content"):
            metadata["author"] = author_meta["content"].strip()

        for tag in soup(_STRIP_TAGS):
            tag.decompose()

        root = soup
        for selector in _CONTENT_SELECTORS:
            found = soup.select_one(selector)
            if found and len(found.get_text(strip=True)) > 200:
                root = found
                break

        text = root.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)

        chunks = chunk_text(
            cleaned,
            target_chars=self.target_chunk_chars,
            min_chars=self.min_chunk_chars,
        )
        sections = [
            {"type": "chunk", "number": i + 1, "content": chunk}
            for i, chunk in enumerate(chunks)
        ]

        return DecodeResult(
            text=cleaned, metadata=metadata, sections=sections, errors=errors
        )
