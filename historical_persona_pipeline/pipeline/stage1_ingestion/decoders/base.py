"""Contratto comune a tutti i decoder di formato.

Ogni decoder restituisce lo stesso dizionario:

    {
        "text":     str,           # testo completo
        "metadata": dict,          # titolo/autore/pagine quando disponibili
        "sections": list[dict],    # blocchi con 'content' e opz. 'heading'/'number'
        "errors":   list[str],     # errori NON fatali
    }

Regola sulle eccezioni: un decoder che non riesce a leggere NULLA deve
segnalarlo in `errors`, mai restituire un risultato vuoto silenzioso.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DecodeResult(dict):
    """Dizionario di risultato con costruttore esplicito."""

    def __init__(
        self,
        text: str = "",
        metadata: Dict[str, Any] | None = None,
        sections: List[Dict[str, Any]] | None = None,
        errors: List[str] | None = None,
    ):
        super().__init__(
            text=text,
            metadata=metadata or {},
            sections=sections or [],
            errors=errors or [],
        )

    @property
    def is_empty(self) -> bool:
        return not self["text"].strip() and not self["sections"]


class BaseDecoder(ABC):
    """Base per i decoder di formato."""

    #: Estensioni gestite, minuscole e col punto.
    extensions: tuple[str, ...] = ()

    #: Nome leggibile mostrato nella GUI.
    display_name: str = ""

    def can_decode(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.extensions

    @abstractmethod
    def decode(self, file_path: Path) -> DecodeResult:
        """Estrae testo e struttura dal file."""

    def safe_decode(self, file_path: Path) -> DecodeResult:
        """Esegue `decode` convertendo un crash in un errore esplicito."""
        try:
            result = self.decode(file_path)
        except Exception as exc:
            logger.exception(f"{type(self).__name__} fallito su {file_path.name}")
            return DecodeResult(
                errors=[f"{type(self).__name__}: {exc}"]
            )

        if result.is_empty and not result["errors"]:
            # Silenzio sospetto: un file leggibile ma senza testo (PDF scansionato,
            # EPUB con solo immagini) deve emergere, non passare per successo.
            result["errors"].append(
                f"{type(self).__name__}: nessun testo estratto da {file_path.name} "
                f"(file scansionato o protetto?)"
            )
        return result


def read_text_with_fallback(file_path: Path) -> tuple[str, List[str]]:
    """Legge un file di testo indovinando la codifica.

    Prova UTF-8, poi il rilevamento di `charset_normalizer`, infine latin-1
    (che non fallisce mai ma puo' produrre mojibake: viene segnalato).
    """
    errors: List[str] = []
    raw = file_path.read_bytes()

    try:
        return raw.decode("utf-8"), errors
    except UnicodeDecodeError:
        pass

    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None:
            errors.append(f"Codifica non UTF-8, rilevata: {best.encoding}")
            return str(best), errors
    except ImportError:
        pass
    except Exception as exc:
        errors.append(f"Rilevamento codifica fallito: {exc}")

    errors.append("Codifica non riconosciuta: letto come latin-1, possibili caratteri errati")
    return raw.decode("latin-1", errors="replace"), errors
