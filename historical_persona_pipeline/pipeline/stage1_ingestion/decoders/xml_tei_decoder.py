"""Decoder XML/TEI — il formato delle edizioni critiche digitali.

TEI e' lo standard delle biblioteche digitali (Perseus, MQDQ, DTA): per un
corpus storico e' spesso la fonte migliore disponibile. L'apparato critico,
le note e i titoli editoriali vengono scartati: non sono voce dell'autore.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseDecoder, DecodeResult

logger = logging.getLogger(__name__)

# Elementi TEI che non contengono testo d'autore.
_TEI_STRIP = (
    "teiHeader", "note", "app", "rdg", "witDetail", "figure",
    "figDesc", "bibl", "listBibl", "gap", "del",
)

_GENERIC_STRIP = ("script", "style")


class XMLTEIDecoder(BaseDecoder):
    extensions = (".xml", ".tei")
    display_name = "XML / TEI"

    def decode(self, file_path: Path) -> DecodeResult:
        from bs4 import BeautifulSoup

        errors: list[str] = []
        raw = file_path.read_bytes()

        # lxml-xml preserva i namespace TEI; se manca lxml si degrada a html.parser.
        try:
            soup = BeautifulSoup(raw, "lxml-xml")
        except Exception:
            errors.append("Parser lxml-xml non disponibile: uso html.parser")
            soup = BeautifulSoup(raw, "html.parser")

        metadata = self._extract_metadata(soup)

        for tag_name in _TEI_STRIP + _GENERIC_STRIP:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        sections: list[dict] = []
        full_text: list[str] = []

        # `div` e' l'unita' strutturale TEI (libro, capitolo, sezione).
        divs = soup.find_all("div")
        for div in divs:
            # Solo le foglie: un div che ne contiene altri duplicherebbe il testo.
            if div.find("div"):
                continue
            content = self._clean(div.get_text(separator="\n"))
            if not content:
                continue
            head = div.find("head")
            sections.append({
                "type": div.get("type", "div"),
                "heading": head.get_text(strip=True) if head else None,
                "number": div.get("n"),
                "content": content,
            })
            full_text.append(content)

        if not sections:
            # Nessun div: si ripiega sul body, o sull'intero documento.
            body = soup.find("body") or soup
            content = self._clean(body.get_text(separator="\n"))
            if content:
                sections.append({"type": "body", "content": content})
                full_text.append(content)

        return DecodeResult(
            text="\n\n".join(full_text),
            metadata=metadata,
            sections=sections,
            errors=errors,
        )

    @staticmethod
    def _clean(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    @staticmethod
    def _extract_metadata(soup) -> dict:
        metadata: dict = {}
        title = soup.find("title")
        if title:
            metadata["title"] = title.get_text(strip=True)
        author = soup.find("author")
        if author:
            metadata["author"] = author.get_text(strip=True)
        return metadata
