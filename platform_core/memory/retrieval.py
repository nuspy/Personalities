"""Quali memorie portare in una risposta.

Quattro segnali, e la ragione di ciascuno.

**Semantica** — la memoria parla di ciò di cui si sta parlando. È il segnale
principale, ma da solo pesca ogni volta le stesse cose: su cento memorie,
quelle più vicine a una domanda generica sono sempre le medesime.

**Recency** — le cose recenti valgono di più, perché è più probabile che siano
ancora vere. Un decadimento e non una soglia: «negli ultimi trenta giorni»
taglia netto ciò che ha trentuno giorni, e quella scogliera non corrisponde a
nulla nella realtà.

**Importanza** — assegnata all'estrazione: sapere dove vive qualcuno conta più
di sapere che gli è piaciuto un film.

**Frequenza** — ciò che è servito spesso è probabile serva ancora. È l'unico
segnale che si costruisce con l'uso, e serve a far emergere col tempo le cose
che contano davvero per quella persona.

**Perché una somma pesata e non RRF.** Qui i segnali non sono liste ordinate
prodotte da metodi diversi, ma quattro numeri **sulla stessa riga**, e tre dei
quattro sono già normalizzati per costruzione (importanza e confidenza stanno
in [0,1], la somiglianza coseno anche). Con RRF si perderebbe l'intensità:
una memoria molto più pertinente delle altre conterebbe come una appena più
pertinente, e per la memoria quella differenza è precisamente ciò che conta.
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.memory_models import Memory
from ..knowledge.embedding import Embedder
from ..observability.tracing import traccia

logger = logging.getLogger(__name__)

#: Quanto pesa ciascun segnale. Somma a 1 perché il punteggio resti leggibile
#: come «quanto vale questa memoria, da 0 a 1»: un punteggio che non si sa
#: interpretare non si sa nemmeno tarare.
PESI = {
    "semantica": 0.55,
    "recenza": 0.20,
    "importanza": 0.15,
    "frequenza": 0.10,
}

#: Dopo quanti giorni la recenza vale metà.
#:
#: Trenta: abbastanza perché una conversazione del mese scorso conti ancora,
#: abbastanza poco perché una di un anno fa non competa con una di ieri. Non è
#: una misura fisica — è una scelta, e sta scritta qui perché si possa
#: discutere invece di doverla dedurre da una formula.
EMIVITA_GIORNI = 30.0

#: Oltre questo numero di richiami la frequenza smette di crescere. Senza un
#: tetto, una memoria richiamata cento volte schiaccerebbe ogni altro segnale
#: e la risposta parlerebbe sempre della stessa cosa.
RICHIAMI_PER_IL_MASSIMO = 10


@dataclass
class MemoriaRecuperata:
    """Una memoria scelta, con il perché."""

    memoria: Memory
    punteggio: float
    semantica: float = 0.0
    recenza: float = 0.0
    importanza: float = 0.0
    frequenza: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.memoria.id),
            "kind": self.memoria.kind,
            "content": self.memoria.content,
            "punteggio": round(self.punteggio, 4),
            "segnali": {
                "semantica": round(self.semantica, 4),
                "recenza": round(self.recenza, 4),
                "importanza": round(self.importanza, 4),
                "frequenza": round(self.frequenza, 4),
            },
        }


def recenza(quando: Optional[datetime], *, adesso: Optional[datetime] = None) -> float:
    """Decadimento esponenziale: 1 adesso, 0,5 dopo un'emivita.

    Non arriva mai a zero, e la scelta è deliberata: una memoria vecchissima
    ma perfettamente pertinente deve poter emergere comunque. Azzerarla
    significherebbe che dopo un anno il sistema dimentica anche ciò che gli è
    stato detto una volta sola e conta più di tutto.
    """
    if quando is None:
        return 0.0
    adesso = adesso or utcnow()
    giorni = max(0.0, (adesso - quando).total_seconds() / 86400.0)
    return math.pow(0.5, giorni / EMIVITA_GIORNI)


def frequenza(richiami: int) -> float:
    """Da un conteggio a un valore fra 0 e 1, con saturazione.

    Logaritmica: la differenza fra una e due volte conta più di quella fra
    nove e dieci, ed è come funziona la memoria di chiunque.
    """
    if richiami <= 0:
        return 0.0
    return min(1.0, math.log1p(richiami) / math.log1p(RICHIAMI_PER_IL_MASSIMO))


def punteggio_di(
    *, semantica: float, recenza_: float, importanza: float, frequenza_: float,
) -> float:
    return (
        PESI["semantica"] * semantica
        + PESI["recenza"] * recenza_
        + PESI["importanza"] * importanza
        + PESI["frequenza"] * frequenza_
    )


class MemoryRetriever:
    def __init__(self, session: AsyncSession, embedder: Embedder) -> None:
        self._session = session
        self._embedder = embedder

    async def cerca(
        self,
        *,
        user_id: int,
        domanda: str,
        personality_id: Optional[uuid.UUID] = None,
        limite: int = 5,
        candidati: int = 30,
        soglia: float = 0.25,
    ) -> List[MemoriaRecuperata]:
        """Le memorie da portare in una risposta.

        `soglia` esiste perché **nessuna memoria è una risposta legittima**:
        se niente di ciò che si ricorda riguarda la domanda, portare le cinque
        meno peggio significa mettere nel prompt materiale irrilevante, e un
        modello che legge materiale irrilevante ci costruisce sopra.
        """
        with traccia("memoria.recupero", utente=user_id):
            embedding = await self._embedder.query(domanda)

            # Il primo filtro è vettoriale e lo fa il database: prendere tutte
            # le memorie di un utente per ordinarle in Python funziona finché
            # sono cento, e smette quando sono diecimila.
            query = text("""
                SELECT
                    m.*,
                    1 - (m.embedding <=> CAST(:embedding AS vector)) AS somiglianza
                FROM memories m
                WHERE m.user_id = :user_id
                  AND m.valid_to IS NULL
                  AND m.superseded_by IS NULL
                  AND (m.expires_at IS NULL OR m.expires_at > now())
                  AND (m.personality_id IS NULL OR m.personality_id = :personality_id)
                  AND m.embedding IS NOT NULL
                ORDER BY m.embedding <=> CAST(:embedding AS vector)
                LIMIT :candidati
            """)

            righe = (await self._session.execute(query, {
                "embedding": "[" + ",".join(f"{x:.7g}" for x in embedding) + "]",
                "user_id": user_id,
                # `NULL` non è mai uguale a `NULL` in SQL: senza un valore
                # segnaposto, il confronto escluderebbe tutto quando non c'è
                # una personalità, invece di lasciar passare le memorie
                # generali.
                "personality_id": personality_id or uuid.UUID(int=0),
                "candidati": candidati,
            })).mappings().all()

        adesso = utcnow()
        valutate: List[MemoriaRecuperata] = []

        for riga in righe:
            memoria = await self._session.get(Memory, riga["id"])
            if memoria is None:
                continue

            s = float(riga["somiglianza"])
            r = recenza(memoria.last_referenced_at or memoria.first_seen_at, adesso=adesso)
            f = frequenza(memoria.times_referenced)
            # L'importanza si sconta per la confidenza: una cosa importante ma
            # dedotta vale meno della stessa cosa detta esplicitamente.
            i = memoria.importance * memoria.confidence

            valutate.append(MemoriaRecuperata(
                memoria=memoria,
                punteggio=punteggio_di(
                    semantica=s, recenza_=r, importanza=i, frequenza_=f,
                ),
                semantica=s, recenza=r, importanza=i, frequenza=f,
            ))

        valutate.sort(key=lambda m: m.punteggio, reverse=True)
        scelte = [m for m in valutate if m.punteggio >= soglia][:limite]

        logger.debug(
            "Memoria: %d candidate, %d sopra soglia, %d scelte",
            len(valutate), sum(1 for m in valutate if m.punteggio >= soglia),
            len(scelte),
        )
        return scelte

    async def segna_usate(self, recuperate: Sequence[MemoriaRecuperata]) -> None:
        """Registra che queste memorie sono state usate.

        Alimenta il segnale di frequenza, e insieme quello di recenza: una
        memoria richiamata oggi torna «recente» anche se è vecchia, il che è
        esattamente come funziona ricordarsi di qualcosa.
        """
        adesso = utcnow()
        for r in recuperate:
            r.memoria.times_referenced += 1
            r.memoria.last_referenced_at = adesso
        await self._session.flush()
