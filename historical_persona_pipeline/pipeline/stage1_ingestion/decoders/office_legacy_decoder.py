"""Decoder per i formati office non-OOXML: .doc, .rtf, .odt.

`.doc` (Word 97-2003) e' un contenitore OLE binario. Nessuna libreria pura
Python lo legge in modo affidabile, quindi la strategia e' a cascata:
1. `antiword` se presente nel PATH (il piu' fedele);
2. LibreOffice in modalita' headless, se installato;
3. estrazione grezza dello stream OLE con `olefile`, che recupera il testo ma
   perde la formattazione.
Se falliscono tutte, l'errore e' esplicito e suggerisce la conversione.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import BaseDecoder, DecodeResult, read_text_with_fallback

logger = logging.getLogger(__name__)

_CONVERSION_TIMEOUT = 120


class DOCDecoder(BaseDecoder):
    extensions = (".doc",)
    display_name = "Word 97-2003 (legacy)"

    def decode(self, file_path: Path) -> DecodeResult:
        errors: list[str] = []

        text = self._try_antiword(file_path, errors)
        if not text:
            text = self._try_libreoffice(file_path, errors)
        if not text:
            text = self._try_olefile(file_path, errors)

        if not text:
            errors.append(
                "Impossibile leggere il .doc: installare antiword o LibreOffice, "
                "oppure convertire il file in .docx"
            )
            return DecodeResult(errors=errors)

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sections = [
            {"type": "paragraph", "number": i + 1, "content": p}
            for i, p in enumerate(paragraphs)
        ]
        return DecodeResult(text=text, sections=sections, errors=errors)

    def _try_antiword(self, file_path: Path, errors: list[str]) -> str:
        exe = shutil.which("antiword")
        if not exe:
            return ""
        try:
            out = subprocess.run(
                [exe, "-m", "UTF-8.txt", str(file_path)],
                capture_output=True, timeout=_CONVERSION_TIMEOUT,
            )
            if out.returncode == 0:
                return out.stdout.decode("utf-8", errors="replace")
            errors.append(f"antiword uscito con codice {out.returncode}")
        except Exception as exc:
            errors.append(f"antiword fallito: {exc}")
        return ""

    def _try_libreoffice(self, file_path: Path, errors: list[str]) -> str:
        exe = shutil.which("soffice") or shutil.which("libreoffice")
        if not exe:
            return ""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(
                    [exe, "--headless", "--convert-to", "txt:Text",
                     "--outdir", tmp, str(file_path)],
                    capture_output=True, timeout=_CONVERSION_TIMEOUT,
                )
                produced = list(Path(tmp).glob("*.txt"))
                if produced:
                    text, enc_errors = read_text_with_fallback(produced[0])
                    errors.extend(enc_errors)
                    return text
        except Exception as exc:
            errors.append(f"LibreOffice fallito: {exc}")
        return ""

    def _try_olefile(self, file_path: Path, errors: list[str]) -> str:
        try:
            import olefile
        except ImportError:
            return ""
        try:
            if not olefile.isOleFile(str(file_path)):
                errors.append("Non e' un file OLE valido")
                return ""
            with olefile.OleFileIO(str(file_path)) as ole:
                if not ole.exists("WordDocument"):
                    return ""
                raw = ole.openstream("WordDocument").read()
            # Lo stream contiene testo UTF-16 frammisto a strutture binarie:
            # si recuperano le sequenze stampabili sufficientemente lunghe.
            decoded = raw.decode("utf-16-le", errors="ignore")
            fragments = re.findall(r"[^\x00-\x08\x0b\x0c\x0e-\x1f]{40,}", decoded)
            if fragments:
                errors.append(
                    "Testo recuperato in modo grezzo dallo stream OLE: "
                    "formattazione e ordine dei paragrafi possono essere imprecisi"
                )
            return "\n\n".join(f.strip() for f in fragments)
        except Exception as exc:
            errors.append(f"Lettura OLE fallita: {exc}")
        return ""


class RTFDecoder(BaseDecoder):
    extensions = (".rtf",)
    display_name = "Rich Text Format"

    def decode(self, file_path: Path) -> DecodeResult:
        errors: list[str] = []
        raw, enc_errors = read_text_with_fallback(file_path)
        errors.extend(enc_errors)

        try:
            from striprtf.striprtf import rtf_to_text
            text = rtf_to_text(raw, errors="ignore")
        except ImportError:
            errors.append("striprtf non installato: eseguire 'pip install striprtf'")
            return DecodeResult(errors=errors)

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        sections = [
            {"type": "paragraph", "number": i + 1, "content": p}
            for i, p in enumerate(paragraphs)
        ]
        return DecodeResult(
            text="\n\n".join(paragraphs), sections=sections, errors=errors
        )


class ODTDecoder(BaseDecoder):
    extensions = (".odt",)
    display_name = "OpenDocument Text"

    def decode(self, file_path: Path) -> DecodeResult:
        errors: list[str] = []
        try:
            from odf import text as odf_text, teletype
            from odf.opendocument import load
        except ImportError:
            errors.append("odfpy non installato: eseguire 'pip install odfpy'")
            return DecodeResult(errors=errors)

        doc = load(str(file_path))
        sections: list[dict] = []
        full_text: list[str] = []
        current_heading = None

        for element in doc.getElementsByType(odf_text.H) + doc.getElementsByType(odf_text.P):
            content = teletype.extractText(element).strip()
            if not content:
                continue
            if element.qname[1] == "h":
                current_heading = content
                continue
            sections.append({
                "type": "paragraph",
                "heading": current_heading,
                "content": content,
            })
            full_text.append(content)

        return DecodeResult(
            text="\n\n".join(full_text), sections=sections, errors=errors
        )
