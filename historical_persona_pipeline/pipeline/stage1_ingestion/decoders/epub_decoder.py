"""Decoder EPUB — il formato tipico dei libri.

Era dichiarato in `FileFormat`, offerto dal selettore file e la dipendenza
`ebooklib` era gia' installata, ma il decoder non esisteva: caricare un EPUB
produceva "No suitable decoder found".
"""
from __future__ import annotations

import logging
from pathlib import Path

from bs4 import BeautifulSoup

from .base import BaseDecoder, DecodeResult

logger = logging.getLogger(__name__)

# Elementi che non contengono prosa dell'autore.
_STRIP_TAGS = ("script", "style", "nav", "header", "footer")

# Capitoli sotto questa soglia sono quasi sempre frontespizi, colophon o
# indici: inquinano le statistiche di stile.
_MIN_CHAPTER_CHARS = 200


class EPUBDecoder(BaseDecoder):
    extensions = (".epub",)
    display_name = "Libro EPUB"

    def decode(self, file_path: Path) -> DecodeResult:
        import ebooklib
        from ebooklib import epub

        errors: list[str] = []
        sections: list[dict] = []
        full_text: list[str] = []

        book = epub.read_epub(str(file_path))

        metadata = {
            "title": self._first_meta(book, "title"),
            "author": self._first_meta(book, "creator"),
            "language": self._first_meta(book, "language"),
        }

        skipped_short = 0
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            try:
                soup = BeautifulSoup(item.get_content(), "html.parser")
            except Exception as exc:
                errors.append(f"Capitolo '{item.get_name()}' illeggibile: {exc}")
                continue

            for tag in soup(_STRIP_TAGS):
                tag.decompose()

            heading_tag = soup.find(["h1", "h2", "h3"])
            heading = heading_tag.get_text(strip=True) if heading_tag else None

            # separator="\n" preserva i confini di paragrafo, che servono al chunker.
            text = soup.get_text(separator="\n").strip()
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

            if len(text) < _MIN_CHAPTER_CHARS:
                skipped_short += 1
                continue

            sections.append({
                "type": "chapter",
                "heading": heading,
                "content": text,
                "source_item": item.get_name(),
            })
            full_text.append(text)

        if skipped_short:
            logger.debug(f"{file_path.name}: {skipped_short} capitoli brevi ignorati")

        return DecodeResult(
            text="\n\n".join(full_text),
            metadata=metadata,
            sections=sections,
            errors=errors,
        )

    @staticmethod
    def _first_meta(book, name: str) -> str:
        try:
            values = book.get_metadata("DC", name)
            if values:
                return str(values[0][0])
        except Exception:
            pass
        return ""
