"""Decoder per testo semplice e Markdown.

I file di testo sono il caso piu' frequente e quello con piu' varianti di
codifica: la versione precedente apriva in UTF-8 e ripiegava su latin-1,
perdendo silenziosamente gli accenti dei file Windows-1252 o ISO-8859-1.
La sezionatura non spezza piu' su ogni riga vuota: sarebbe un segmento per
paragrafo, troppo corto per rilevare la lingua e per le statistiche di stile.
"""
from __future__ import annotations

import re
from pathlib import Path

from .base import BaseDecoder, DecodeResult, read_text_with_fallback
from ..chunker import chunk_text

# Intestazione Markdown ATX (# Titolo) o Setext (sottolineata da === / ---).
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# Marcatori Markdown da rimuovere: alterano il conteggio del vocabolario.
_MD_INLINE = [
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),   # immagini -> alt text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),    # link -> testo
    (re.compile(r"`{1,3}([^`]*)`{1,3}"), r"\1"),      # codice inline
    (re.compile(r"[*_]{1,3}([^*_]+)[*_]{1,3}"), r"\1"),  # enfasi
    (re.compile(r"^>\s?", re.MULTILINE), ""),        # citazioni
]


class TXTDecoder(BaseDecoder):
    extensions = (".txt", ".text", ".log")
    display_name = "File di testo"

    def __init__(self, target_chunk_chars: int = 4000, min_chunk_chars: int = 200):
        self.target_chunk_chars = target_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def decode(self, file_path: Path) -> DecodeResult:
        text, errors = read_text_with_fallback(file_path)

        chunks = chunk_text(
            text,
            target_chars=self.target_chunk_chars,
            min_chars=self.min_chunk_chars,
        )
        sections = [
            {"type": "chunk", "number": i + 1, "content": chunk}
            for i, chunk in enumerate(chunks)
        ]
        return DecodeResult(text=text, sections=sections, errors=errors)


class MarkdownDecoder(BaseDecoder):
    extensions = (".md", ".markdown", ".mdown")
    display_name = "Markdown"

    def __init__(self, target_chunk_chars: int = 4000, min_chunk_chars: int = 200):
        self.target_chunk_chars = target_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def decode(self, file_path: Path) -> DecodeResult:
        raw, errors = read_text_with_fallback(file_path)

        metadata, body = self._split_front_matter(raw)
        sections: list[dict] = []
        full_text: list[str] = []

        for heading, block in self._split_by_heading(body):
            clean = self._strip_markup(block)
            if not clean.strip():
                continue
            for chunk in chunk_text(
                clean,
                target_chars=self.target_chunk_chars,
                min_chars=self.min_chunk_chars,
            ):
                sections.append({"type": "section", "heading": heading, "content": chunk})
                full_text.append(chunk)

        return DecodeResult(
            text="\n\n".join(full_text),
            metadata=metadata,
            sections=sections,
            errors=errors,
        )

    @staticmethod
    def _split_front_matter(raw: str) -> tuple[dict, str]:
        """Estrae il front matter YAML se presente."""
        if not raw.startswith("---"):
            return {}, raw
        parts = raw.split("---", 2)
        if len(parts) < 3:
            return {}, raw
        try:
            import yaml
            meta = yaml.safe_load(parts[1]) or {}
            if isinstance(meta, dict):
                return meta, parts[2]
        except Exception:
            pass
        return {}, raw

    @staticmethod
    def _split_by_heading(body: str) -> list[tuple[str | None, str]]:
        matches = list(_MD_HEADING.finditer(body))
        if not matches:
            return [(None, body)]

        blocks: list[tuple[str | None, str]] = []
        preamble = body[: matches[0].start()].strip()
        if preamble:
            blocks.append((None, preamble))

        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            blocks.append((match.group(2).strip(), body[start:end]))
        return blocks

    @staticmethod
    def _strip_markup(text: str) -> str:
        for pattern, replacement in _MD_INLINE:
            text = pattern.sub(replacement, text)
        return text
