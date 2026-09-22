"""Dove aspettano i file caricati dalla console.

L'API riceve il file e il worker lo legge: sono processi diversi, in
produzione su pod diversi, e fra i due serve un posto che vedano entrambi.
Qui è una directory — in Kubernetes un volume `ReadWriteMany` montato su
entrambi; dietro lo stesso contratto può stare un object storage il giorno
che serve, senza toccare chi carica né chi legge.

**Il nome del file non viene dall'utente.** Si salva come `<uuid><estensione>`
sotto la cartella della base: un nome scelto da chi carica può contenere
`../`, un percorso assoluto o un nome riservato di Windows, e ciascuno di
questi è un modo di scrivere dove non si dovrebbe. Il nome originale resta
nei parametri del lavoro, per mostrarlo.
"""
from __future__ import annotations

import logging
import pathlib
import re
import uuid
from typing import AsyncIterator, Optional, Protocol

logger = logging.getLogger(__name__)

#: Un'estensione accettabile nel nome su disco: punto, lettere e cifre.
_ESTENSIONE = re.compile(r"^\.[a-z0-9]{1,8}$")

#: Un riferimento valido: cartella della base e nome generato qui.
_RIFERIMENTO = re.compile(r"^[0-9a-f-]{36}/[0-9a-f]{32}\.[a-z0-9]{1,8}$")


class FileTroppoGrande(ValueError):
    def __init__(self, limite_byte: int) -> None:
        super().__init__(f"oltre {limite_byte // (1024 * 1024)} MB")
        self.limite_byte = limite_byte


class ArchivioCaricamenti(Protocol):
    async def salva(
        self, kb_id: uuid.UUID, estensione: str, pezzi: AsyncIterator[bytes],
    ) -> tuple[str, int]:
        """Scrive il file, restituisce riferimento e dimensione."""

    def percorso(self, riferimento: str) -> pathlib.Path: ...

    def elimina(self, riferimento: str) -> None: ...


class ArchivioSuDisco:
    def __init__(self, radice: pathlib.Path | str, *, limite_byte: int) -> None:
        self._radice = pathlib.Path(radice).resolve()
        self._limite = limite_byte

    async def salva(
        self, kb_id: uuid.UUID, estensione: str, pezzi: AsyncIterator[bytes],
    ) -> tuple[str, int]:
        estensione = estensione.lower()
        if not _ESTENSIONE.match(estensione):
            raise ValueError(f"estensione non valida: {estensione!r}")

        riferimento = f"{kb_id}/{uuid.uuid4().hex}{estensione}"
        destinazione = self._radice / riferimento
        destinazione.parent.mkdir(parents=True, exist_ok=True)

        scritti = 0
        try:
            with destinazione.open("wb") as uscita:
                async for pezzo in pezzi:
                    scritti += len(pezzo)
                    # Il limite si controlla mentre si scrive, non dopo: un
                    # file da dieci gigabyte riempirebbe il volume prima di
                    # essere rifiutato.
                    if scritti > self._limite:
                        raise FileTroppoGrande(self._limite)
                    uscita.write(pezzo)
        except BaseException:
            destinazione.unlink(missing_ok=True)
            raise
        return riferimento, scritti

    def percorso(self, riferimento: str) -> pathlib.Path:
        if not _RIFERIMENTO.match(riferimento):
            # Il riferimento arriva dai parametri di un lavoro, cioè dal
            # database: se non ha la forma che questo archivio produce,
            # qualcuno l'ha scritto a mano, e non si segue.
            raise ValueError(f"riferimento non valido: {riferimento!r}")
        return self._radice / riferimento

    def elimina(self, riferimento: str) -> None:
        try:
            percorso = self.percorso(riferimento)
            percorso.unlink(missing_ok=True)
        except (ValueError, OSError) as exc:
            logger.warning("File caricato non eliminato (%s): %s", riferimento, exc)
            return
        try:
            # La cartella della base, se è rimasta vuota: altrimenti ogni
            # corpus che abbia mai ricevuto un file lascia la sua traccia.
            percorso.parent.rmdir()
        except OSError:
            pass  # non vuota: un altro caricamento è in attesa


def archivio_caricamenti(
    radice: Optional[str] = None, *, limite_mb: Optional[int] = None,
) -> ArchivioSuDisco:
    from ..settings import get_settings

    settings = get_settings()
    return ArchivioSuDisco(
        radice or settings.upload_dir,
        limite_byte=(limite_mb or settings.upload_max_mb) * 1024 * 1024,
    )
