"""I file caricati dalla console, aggiunti a un corpus.

Prima l'unica via era la riga di comando sul nodo: il cliente non poteva
aggiornare le proprie knowledge base senza chiedere a chi ha accesso al
server. Qui lo stesso percorso — estrazione, passaggi, vettori — gira nel
worker come un lavoro della console, con l'avanzamento che si segue da lì.

**Un file che non entra non ferma gli altri.** Un PDF protetto in mezzo a
dieci documenti buoni è la situazione normale, non l'eccezione: si registra
perché non è entrato, e si prosegue. Il lavoro fallisce solo se non è entrato
niente.
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.knowledge_models import KnowledgeBase
from .archivio import ArchivioCaricamenti, archivio_caricamenti
from .documenti import FormatoNonSupportato, estrai
from .embedding import Embedder
from .indexer import Indexer
from .transcription import TrascrizioneNonDisponibile

logger = logging.getLogger(__name__)

Avanzamento = Callable[[int, int, str], Awaitable[None]]


@dataclass
class EsitoIngestione:
    documenti: int = 0
    passaggi: int = 0
    saltati: List[Dict[str, str]] = field(default_factory=list)
    falliti: List[Dict[str, str]] = field(default_factory=list)
    avvisi: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documenti": self.documenti,
            "passaggi": self.passaggi,
            "saltati": self.saltati,
            "falliti": self.falliti,
            "avvisi": self.avvisi,
        }


def _trascrittore_predefinito():
    from ..settings import get_settings
    from .transcription import Trascrittore

    return Trascrittore(get_settings().transcription_model)


class IngestioneCaricamenti:
    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        *,
        archivio: Optional[ArchivioCaricamenti] = None,
        trascrittore: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._session = session
        self._indexer = Indexer(session, embedder)
        self._archivio = archivio or archivio_caricamenti()
        self._fabbrica_trascrittore = trascrittore or _trascrittore_predefinito
        self._trascrittore: Any = None

    def _trascrittore_pronto(self):
        # Whisper si carica al primo audio e si tiene: pesa gigabyte, e
        # ricaricarlo a ogni file trasformerebbe dieci registrazioni in dieci
        # minuti di soli caricamenti.
        if self._trascrittore is None:
            self._trascrittore = self._fabbrica_trascrittore()
        return self._trascrittore

    async def ingerisci(
        self,
        kb: KnowledgeBase,
        file: List[Dict[str, Any]],
        *,
        lingua: Optional[str] = None,
        avanzamento: Optional[Avanzamento] = None,
    ) -> EsitoIngestione:
        esito = EsitoIngestione()
        totale = len(file)

        async def annuncia(fatti: int, messaggio: str) -> None:
            if avanzamento:
                await avanzamento(fatti, totale, messaggio)

        for i, voce in enumerate(file):
            nome = str(voce.get("nome") or "senza nome")
            await annuncia(i, f"{i + 1} di {totale} · leggo «{nome}»")

            try:
                percorso = self._archivio.percorso(str(voce.get("riferimento", "")))
                if not percorso.exists():
                    raise FileNotFoundError("il file caricato non è più nell'archivio")

                # In un thread: un PDF da mille pagine con OCR, o un'ora di
                # audio, occuperebbero il loop per minuti — e con lui la
                # scrittura dell'avanzamento, che è ciò che si sta guardando.
                estratto = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda p=percorso, n=nome: estrai(
                        pathlib.Path(p), nome=n, lingua=lingua,
                        trascrittore=self._trascrittore_pronto,
                    ),
                )
            except (FormatoNonSupportato, TrascrizioneNonDisponibile, FileNotFoundError, ValueError) as exc:
                esito.falliti.append({"nome": nome, "motivo": str(exc)})
                await annuncia(i + 1, f"«{nome}» non è entrato: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("Estrazione fallita per %s", nome)
                esito.falliti.append({"nome": nome, "motivo": f"errore inatteso: {exc}"})
                await annuncia(i + 1, f"«{nome}» non è entrato: errore inatteso")
                continue

            for avviso in estratto.avvisi:
                esito.avvisi.append({"nome": nome, "avviso": avviso})

            await annuncia(i, f"{i + 1} di {totale} · indicizzo «{estratto.titolo}»")
            try:
                risultato = await self._indexer.indicizza(
                    kb,
                    titolo=estratto.titolo,
                    testo=estratto.testo,
                    # Un indirizzo che dice da dove viene, non un percorso: il
                    # file su disco sparisce a lavoro finito.
                    uri=f"caricamento:{nome}",
                    lingua=estratto.lingua or lingua,
                    meta={
                        **estratto.meta,
                        "origine": "console",
                        "file": nome,
                        **({"caricato_da": voce["da"]} if voce.get("da") else {}),
                    },
                )
            except ValueError as exc:
                # L'embedder della base non è quello configurato: vale per
                # tutti i file, e continuare produrrebbe lo stesso rifiuto
                # per ciascuno.
                esito.falliti.append({"nome": nome, "motivo": str(exc)})
                break

            if risultato.saltato:
                esito.saltati.append({"nome": nome, "motivo": risultato.motivo})
            else:
                esito.documenti += 1
                esito.passaggi += risultato.passaggi
            # Un documento alla volta nella transazione: un errore al decimo
            # file non deve portarsi via i nove già indicizzati.
            await self._session.commit()

        await annuncia(
            totale,
            f"{esito.documenti} documenti aggiunti, {esito.passaggi} passaggi"
            + (f", {len(esito.saltati)} già presenti" if esito.saltati else "")
            + (f", {len(esito.falliti)} non entrati" if esito.falliti else ""),
        )
        return esito
