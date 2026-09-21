"""Decoder PDF con de-sillabazione, pulizia di testatine e fallback OCR.

Tre problemi che la versione precedente ignorava:

1. **PDF scansionati** — `page.get_text()` restituisce stringa vuota senza
   alcun errore: la pipeline proseguiva su un corpus vuoto credendo di aver
   funzionato. Ora scatta l'OCR se disponibile, altrimenti l'errore e' esplicito.
2. **Sillabazione di fine riga** — `con-\\nsiderare` restava spezzato e
   inquinava il conteggio del vocabolario con due non-parole.
3. **Testatine e numeri di pagina** — ripetuti su ogni pagina, falsavano le
   collocazioni piu' frequenti (il titolo del libro risultava la formula
   caratteristica dell'autore).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import List

from .base import BaseDecoder, DecodeResult

logger = logging.getLogger(__name__)

# Sillabazione: parola, trattino, a capo, prosecuzione minuscola.
_HYPHEN_BREAK = re.compile(r"(\w)[-­]\s*\n\s*(\w)")

# Riga composta solo da un numero (di pagina), eventualmente tra ornamenti.
_PAGE_NUMBER_LINE = re.compile(r"^[\s\-–—\[\(]*\d{1,4}[\s\-–—\]\)]*$")

# Una riga ricorrente su almeno questa frazione di pagine e' una testatina.
_HEADER_FREQUENCY_THRESHOLD = 0.5
_MIN_PAGES_FOR_HEADER_DETECTION = 5

# Sotto questa media di caratteri per pagina il PDF e' considerato scansionato.
_SCANNED_CHARS_PER_PAGE = 100


class PDFDecoder(BaseDecoder):
    extensions = (".pdf",)
    display_name = "Documento PDF"

    def __init__(self, ocr_enabled: bool = True, ocr_max_pages: int = 50):
        self.ocr_enabled = ocr_enabled
        self.ocr_max_pages = ocr_max_pages

    def decode(self, file_path: Path) -> DecodeResult:
        import fitz  # PyMuPDF

        errors: List[str] = []
        doc = fitz.open(str(file_path))

        try:
            metadata = {
                "title": (doc.metadata or {}).get("title", ""),
                "author": (doc.metadata or {}).get("author", ""),
                "page_count": len(doc),
            }

            raw_pages = [page.get_text("text") for page in doc]
            total_chars = sum(len(p.strip()) for p in raw_pages)
            page_count = max(len(raw_pages), 1)

            if total_chars / page_count < _SCANNED_CHARS_PER_PAGE:
                ocr_pages, ocr_errors = self._try_ocr(doc, file_path)
                errors.extend(ocr_errors)
                if ocr_pages:
                    raw_pages = ocr_pages
                    metadata["extraction"] = "ocr"

            boilerplate = self._detect_boilerplate(raw_pages)
            if boilerplate:
                logger.debug(f"{file_path.name}: rimosse {len(boilerplate)} righe ricorrenti")

            sections: List[dict] = []
            full_text: List[str] = []

            for page_num, raw in enumerate(raw_pages, start=1):
                cleaned = self._clean_page(raw, boilerplate)
                if not cleaned.strip():
                    continue
                sections.append({"type": "page", "number": page_num, "content": cleaned})
                full_text.append(cleaned)

            return DecodeResult(
                text="\n\n".join(full_text),
                metadata=metadata,
                sections=sections,
                errors=errors,
            )
        finally:
            doc.close()

    # ------------------------------------------------------------------ OCR

    def _try_ocr(self, doc, file_path: Path) -> tuple[List[str], List[str]]:
        """OCR pagina per pagina; richiede Tesseract installato nel sistema."""
        errors: List[str] = []
        if not self.ocr_enabled:
            errors.append(
                f"{file_path.name} sembra scansionato ma l'OCR e' disabilitato in configurazione"
            )
            return [], errors

        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            errors.append(
                f"{file_path.name} sembra un PDF scansionato (quasi nessun testo estraibile). "
                "Per leggerlo servono 'pip install pytesseract pillow' e Tesseract OCR installato."
            )
            return [], errors

        import io

        import pytesseract
        from PIL import Image

        try:
            pytesseract.get_tesseract_version()
        except Exception:
            errors.append(
                f"{file_path.name} e' scansionato ma l'eseguibile Tesseract non e' nel PATH."
            )
            return [], errors

        import fitz

        pages: List[str] = []
        limit = min(len(doc), self.ocr_max_pages)
        if len(doc) > limit:
            errors.append(
                f"OCR limitato alle prime {limit} pagine di {len(doc)} "
                "(parametro ingestion.ocr_max_pages)"
            )

        for index in range(limit):
            try:
                # 300 DPI: soglia sotto la quale l'accuratezza OCR degrada molto.
                pixmap = doc[index].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                pages.append(pytesseract.image_to_string(image))
            except Exception as exc:
                errors.append(f"OCR fallito su pagina {index + 1}: {exc}")
                pages.append("")

        return pages, errors

    # --------------------------------------------------------------- pulizia

    @staticmethod
    def _detect_boilerplate(pages: List[str]) -> set[str]:
        """Righe che si ripetono su meta' delle pagine: testatine e pie' di pagina."""
        if len(pages) < _MIN_PAGES_FOR_HEADER_DETECTION:
            return set()

        counter: Counter[str] = Counter()
        for page in pages:
            lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
            # Solo le prime e le ultime due righe possono essere testatine.
            for line in lines[:2] + lines[-2:]:
                if 3 < len(line) < 120:
                    counter[line] += 1

        threshold = len(pages) * _HEADER_FREQUENCY_THRESHOLD
        return {line for line, count in counter.items() if count >= threshold}

    @staticmethod
    def _clean_page(raw: str, boilerplate: set[str]) -> str:
        text = _HYPHEN_BREAK.sub(r"\1\2", raw)

        kept: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                kept.append("")
                continue
            if stripped in boilerplate:
                continue
            if _PAGE_NUMBER_LINE.match(stripped):
                continue
            kept.append(stripped)

        # Righe consecutive non vuote appartengono allo stesso paragrafo:
        # si ricompongono, mentre la riga vuota resta separatore di paragrafo.
        paragraphs: List[str] = []
        buffer: List[str] = []
        for line in kept:
            if line:
                buffer.append(line)
            elif buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
        if buffer:
            paragraphs.append(" ".join(buffer))

        return "\n\n".join(paragraphs)
