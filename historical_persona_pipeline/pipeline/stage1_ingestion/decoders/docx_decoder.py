"""Decoder DOCX (Word moderno).

Rispetto alla versione precedente: gli heading finivano nelle `sections` ma
venivano esclusi da `text` (due viste incoerenti dello stesso file), le
tabelle erano ignorate del tutto e i capitoli lunghi non erano suddivisi.
"""
from __future__ import annotations

from pathlib import Path

from .base import BaseDecoder, DecodeResult
from ..chunker import chunk_text


class DOCXDecoder(BaseDecoder):
    extensions = (".docx",)
    display_name = "Documento Word"

    def __init__(self, target_chunk_chars: int = 4000, min_chunk_chars: int = 200):
        self.target_chunk_chars = target_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def decode(self, file_path: Path) -> DecodeResult:
        from docx import Document

        errors: list[str] = []
        doc = Document(str(file_path))

        metadata = {}
        try:
            props = doc.core_properties
            metadata = {
                "title": props.title or "",
                "author": props.author or "",
                "created": str(props.created) if props.created else "",
            }
        except Exception as exc:
            errors.append(f"Proprieta' del documento illeggibili: {exc}")

        sections: list[dict] = []
        full_text: list[str] = []

        current_heading: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            block = "\n\n".join(buffer)
            for chunk in chunk_text(
                block,
                target_chars=self.target_chunk_chars,
                min_chars=self.min_chunk_chars,
            ):
                sections.append({
                    "type": "section",
                    "heading": current_heading,
                    "content": chunk,
                })
                full_text.append(chunk)
            buffer = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            style_name = ""
            try:
                style_name = para.style.name or ""
            except Exception:
                pass

            if style_name.startswith("Heading") or style_name.startswith("Titolo"):
                flush()
                current_heading = text
                continue

            buffer.append(text)

        flush()

        # Le tabelle spesso contengono dati veri (cronologie, elenchi di opere):
        # ogni riga diventa una frase leggibile invece di essere scartata.
        for table_index, table in enumerate(doc.tables):
            rows: list[str] = []
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                content = "\n".join(rows)
                sections.append({
                    "type": "table",
                    "number": table_index + 1,
                    "content": content,
                })
                full_text.append(content)

        return DecodeResult(
            text="\n\n".join(full_text),
            metadata=metadata,
            sections=sections,
            errors=errors,
        )
