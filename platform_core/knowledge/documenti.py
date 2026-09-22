"""Da un file a un testo da indicizzare.

I decoder dei formati vengono dalla pipeline storica e si usano così come
sono: diciassette formati già provati, con le loro stranezze note — il PDF
scansionato che richiede l'OCR, il `.doc` che vuole LibreOffice, l'EPUB fatto
di sole immagini. Riscriverli per la piattaforma significherebbe riscoprire
quelle stranezze una alla volta, sui file dei clienti.

Audio e video passano invece da Whisper, e il documento porta
`registro: parlato`: come qualcuno parla non è come scrive, e mescolarli senza
distinzione insegna a un modello a scrivere come si parla.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from .transcription import (
    ESTENSIONI_AUDIO, ESTENSIONI_VIDEO, TrascrizioneNonDisponibile, e_multimediale,
)

logger = logging.getLogger(__name__)


class FormatoNonSupportato(ValueError):
    pass


@dataclass
class TestoEstratto:
    titolo: str
    testo: str
    lingua: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    #: Errori non fatali del decoder: pagine illeggibili, sezioni vuote.
    avvisi: List[str] = field(default_factory=list)


def _decoder():
    """I decoder della pipeline storica, importati solo quando servono.

    Portano PyMuPDF, python-docx, ebooklib e affini: dipendenze del worker che
    ingerisce, non dell'API — che dei decoder sa soltanto le estensioni.
    """
    try:
        from historical_persona_pipeline.pipeline.stage1_ingestion.decoders import (
            build_decoders, find_decoder,
        )
    except ImportError as exc:  # pragma: no cover — dipende dall'immagine
        raise FormatoNonSupportato(
            f"i decoder dei documenti non sono installati su questo nodo: {exc}"
        ) from exc
    return build_decoders(), find_decoder


@lru_cache(maxsize=1)
def formati_documento() -> Dict[str, str]:
    """Estensione → nome leggibile, per i documenti scritti."""
    try:
        from historical_persona_pipeline.pipeline.stage1_ingestion.decoders import (
            extension_labels,
        )
    except ImportError:
        logger.warning("Decoder dei documenti non disponibili: solo testo semplice")
        return {".txt": "Testo", ".md": "Markdown"}
    return extension_labels()


def formati_supportati() -> Dict[str, str]:
    """Tutto ciò che la console può caricare, con un nome per ciascuno."""
    formati = dict(formati_documento())
    for estensione in sorted(ESTENSIONI_AUDIO):
        formati.setdefault(estensione, "Audio (trascritto)")
    for estensione in sorted(ESTENSIONI_VIDEO):
        formati.setdefault(estensione, "Video (trascritto)")
    return formati


def e_supportato(nome: str) -> bool:
    return pathlib.Path(nome).suffix.lower() in formati_supportati()


def estrai_documento(percorso: pathlib.Path, *, nome: str) -> TestoEstratto:
    """Il testo di un documento scritto.

    `nome` è quello originale del file: su disco il file ha un nome generato,
    e il titolo del documento deve essere quello che l'amministratore
    riconosce.
    """
    decoders, trova = _decoder()
    # Il decoder si sceglie dall'estensione del nome originale: su disco è la
    # stessa, ma è quello il nome di cui l'utente è responsabile.
    decoder = trova(pathlib.Path(nome), decoders)
    if decoder is None:
        raise FormatoNonSupportato(f"nessun decoder per «{pathlib.Path(nome).suffix}»")

    esito = decoder.safe_decode(percorso)
    testo = esito["text"] or "\n\n".join(
        s.get("content", "") for s in esito["sections"]
    )
    metadati = esito["metadata"] or {}
    if not testo.strip():
        raise FormatoNonSupportato(
            "; ".join(esito["errors"]) or "nessun testo estratto"
        )

    return TestoEstratto(
        titolo=str(metadati.get("title") or pathlib.Path(nome).stem),
        testo=testo,
        meta={
            "registro": "scritto",
            "formato": decoder.display_name,
            **({"autore": metadati["author"]} if metadati.get("author") else {}),
            **({"pagine": metadati["pages"]} if metadati.get("pages") else {}),
        },
        avvisi=list(esito["errors"]),
    )


def estrai_parlato(
    percorso: pathlib.Path,
    *,
    nome: str,
    lingua: Optional[str],
    trascrittore: Any,
) -> TestoEstratto:
    """Il testo di un audio o di un video, trascritto."""
    trascrizione = trascrittore.trascrivi(percorso, lingua=lingua)
    if not trascrizione.testo.strip():
        raise FormatoNonSupportato("la trascrizione è vuota")

    avvisi = []
    if trascrizione.da_riascoltare:
        # Un avviso e non un rifiuto: nessuna difesa automatica distingue un
        # errore plausibile, e l'unica verifica che funziona è un orecchio.
        avvisi.append(
            f"fiducia media {trascrizione.fiducia_media:.2f}: ascolta un "
            f"campione prima di fidarti di questo testo"
        )
    return TestoEstratto(
        titolo=pathlib.Path(nome).stem,
        testo=trascrizione.testo,
        lingua=trascrizione.lingua or lingua,
        meta=trascrizione.meta(),
        avvisi=avvisi,
    )


def estrai(
    percorso: pathlib.Path,
    *,
    nome: str,
    lingua: Optional[str] = None,
    trascrittore: Optional[Callable[[], Any]] = None,
) -> TestoEstratto:
    """Documento o parlato, secondo l'estensione del nome originale."""
    if e_multimediale(pathlib.Path(nome)):
        if trascrittore is None:
            raise TrascrizioneNonDisponibile("nessun trascrittore configurato")
        return estrai_parlato(
            percorso, nome=nome, lingua=lingua, trascrittore=trascrittore(),
        )
    return estrai_documento(percorso, nome=nome)
