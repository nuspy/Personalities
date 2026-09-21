"""Registro dei decoder disponibili.

Unica fonte di verita' sui formati supportati: il selettore file della GUI e
l'ingestione leggono da qui. Prima le due liste erano scritte a mano in due
punti diversi e avevano divergato — la GUI offriva .epub, .doc, .xml e .tei
per i quali nessun decoder esisteva.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseDecoder, DecodeResult
from .docx_decoder import DOCXDecoder
from .epub_decoder import EPUBDecoder
from .html_decoder import HTMLDecoder
from .office_legacy_decoder import DOCDecoder, ODTDecoder, RTFDecoder
from .pdf_decoder import PDFDecoder
from .txt_decoder import MarkdownDecoder, TXTDecoder
from .xml_tei_decoder import XMLTEIDecoder

__all__ = [
    "BaseDecoder", "DecodeResult", "build_decoders",
    "supported_extensions", "extension_labels",
]


def build_decoders(config: Optional[Dict[str, Any]] = None) -> List[BaseDecoder]:
    """Istanzia tutti i decoder con i parametri di configurazione."""
    config = config or {}
    ingestion = config.get("ingestion", {})
    target = ingestion.get("target_chunk_chars", 4000)
    minimum = ingestion.get("min_chunk_chars", 200)

    return [
        TXTDecoder(target, minimum),
        MarkdownDecoder(target, minimum),
        PDFDecoder(
            ocr_enabled=ingestion.get("ocr_fallback", True),
            ocr_max_pages=ingestion.get("ocr_max_pages", 50),
        ),
        DOCXDecoder(target, minimum),
        DOCDecoder(),
        RTFDecoder(),
        ODTDecoder(),
        EPUBDecoder(),
        HTMLDecoder(target, minimum),
        XMLTEIDecoder(),
    ]


def extension_labels() -> Dict[str, str]:
    """Mappa estensione -> nome leggibile, per la GUI."""
    labels: Dict[str, str] = {}
    for decoder in build_decoders():
        for ext in decoder.extensions:
            labels[ext] = decoder.display_name
    return labels


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(extension_labels().keys()))


def find_decoder(file_path: Path, decoders: List[BaseDecoder]) -> Optional[BaseDecoder]:
    return next((d for d in decoders if d.can_decode(file_path)), None)
