"""Scrittura e lettura delle memorie, con il versionamento dei fatti.

**Il metodo che dà senso al modulo è `sostituisci`.** Quando arriva un fatto
che ne contraddice uno vecchio, il vecchio non si cancella: gli si mette una
data di fine e lo si collega al nuovo. Restano entrambi, in ordine, e la
domanda «dove vivevo l'anno scorso?» continua ad avere risposta.

La cancellazione vera esiste — un utente ha il diritto di far dimenticare
qualcosa, ed è un diritto, non una preferenza — ma è un'azione esplicita di
chi possiede la memoria, non l'effetto collaterale di un aggiornamento.

**Ogni lettura passa per l'utente.** Non c'è un metodo che restituisca una
memoria senza sapere di chi sia: come per le conversazioni, la difesa non è
ricordarsi la `WHERE`, è non avere scritto l'alternativa.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.memory_models import Memory, MemoryAccess
from ..knowledge.embedding import Embedder

logger = logging.getLogger(__name__)

#: Quanto vive un riassunto di sessione se nessuno lo richiama.
#:
#: Serve a riprendere il filo la volta dopo, non a essere ricordato per
#: sempre: senza una scadenza, dopo un anno di conversazioni quotidiane le
#: memorie di sessione sarebbero il novanta per cento di tutto, e
#: seppellirebbero le poche cose che contano.
GIORNI_SESSIONE = 30

#: Quanto vive un impegno. Più a lungo: «ricordamelo la prossima volta» è una
#: promessa, e dimenticarla è il modo peggiore di gestire la memoria.
GIORNI_IMPEGNO = 180


class MemoriaPiena(Exception):
    """Il piano non consente altre memorie, e la nuova non conta abbastanza."""

    def __init__(self, limite: int) -> None:
        super().__init__(f"raggiunto il limite di {limite} memorie")
        self.limite = limite


class MemoryStore:
    def __init__(self, session: AsyncSession, embedder: Optional[Embedder] = None):
        self._session = session
        self._embedder = embedder

    # -- lettura -----------------------------------------------------------

    async def per_utente(
        self,
        user_id: int,
        *,
        includi_superate: bool = False,
        personality_id: Optional[uuid.UUID] = None,
        limite: int = 200,
    ) -> Sequence[Memory]:
        query = select(Memory).where(Memory.user_id == user_id)

        if not includi_superate:
            query = query.where(
                Memory.valid_to.is_(None), Memory.superseded_by.is_(None),
            )
        if personality_id is not None:
            query = query.where(
                (Memory.personality_id == personality_id)
                | Memory.personality_id.is_(None)
            )

        query = query.order_by(Memory.created_at.desc()).limit(limite)
        return (await self._session.execute(query)).scalars().all()

    async def una(self, user_id: int, memory_id: uuid.UUID) -> Optional[Memory]:
        """Una memoria, **se** è di questo utente.

        Come per le conversazioni: chi non la possiede non deve distinguere
        «non esiste» da «non è tua».
        """
        return (await self._session.execute(
            select(Memory).where(
                Memory.id == memory_id, Memory.user_id == user_id,
            )
        )).scalar_one_or_none()

    async def storia_di(
        self, user_id: int, memory_id: uuid.UUID
    ) -> List[Memory]:
        """La catena di ciò che questa memoria ha sostituito.

        È la risposta a «da quando?» e a «cosa credevi prima?»: senza, una
        memoria corretta sembra sempre essere stata così.
        """
        catena: List[Memory] = []
        corrente = await self.una(user_id, memory_id)

        while corrente is not None:
            catena.append(corrente)
            precedente = (await self._session.execute(
                select(Memory).where(
                    Memory.superseded_by == corrente.id,
                    Memory.user_id == user_id,
                )
            )).scalar_one_or_none()
            if precedente is None or precedente in catena:
                break
            corrente = precedente

        return catena

    async def conta(self, user_id: int) -> Dict[str, int]:
        righe = (await self._session.execute(
            select(Memory.kind, func.count())
            .where(
                Memory.user_id == user_id,
                Memory.valid_to.is_(None),
                Memory.superseded_by.is_(None),
            )
            .group_by(Memory.kind)
        )).all()
        return {kind: int(n) for kind, n in righe}

    # -- scrittura ---------------------------------------------------------

    async def ricorda(
        self,
        *,
        user_id: int,
        content: str,
        kind: str = "fatto",
        personality_id: Optional[uuid.UUID] = None,
        importance: float = 0.5,
        confidence: float = 0.8,
        source_message_id: Optional[uuid.UUID] = None,
        valid_from: Optional[datetime] = None,
        meta: Optional[Dict[str, Any]] = None,
        limite: Optional[int] = None,
    ) -> Memory:
        adesso = utcnow()
        if limite is not None:
            await self._fai_spazio(user_id, limite, importance, adesso)
        memoria = Memory(
            user_id=user_id,
            personality_id=personality_id,
            kind=kind,
            content=content.strip(),
            importance=max(0.0, min(1.0, importance)),
            confidence=max(0.0, min(1.0, confidence)),
            first_seen_at=adesso,
            valid_from=valid_from or adesso,
            expires_at=_scadenza_per(kind, adesso),
            source_message_id=source_message_id,
            meta=meta,
        )

        if self._embedder is not None:
            # Senza vettore la memoria esiste e non si trova mai: è lo stesso
            # difetto dei passaggi senza embedding, e si manifesta allo stesso
            # modo — come un sistema che sembra semplicemente smemorato.
            vettori = await self._embedder.documenti([memoria.content])
            memoria.embedding = vettori[0]

        self._session.add(memoria)
        await self._session.flush()
        return memoria

    async def _fai_spazio(
        self, user_id: int, limite: int, importanza: float, adesso: datetime,
    ) -> None:
        """Al limite del piano, la memoria meno importante lascia il posto.

        Solo se la nuova conta di più: altrimenti si rinuncia alla nuova. Il
        contrario — rifiutare sempre oltre il limite — congelerebbe la memoria
        di chi l'ha riempita il primo mese con dettagli, e le cose importanti
        dette dopo non entrerebbero mai. La memoria che esce non si cancella:
        si chiude con una data di fine e il motivo, come ogni fatto superato.
        """
        vive = (Memory.user_id == user_id) & Memory.valid_to.is_(None) & Memory.superseded_by.is_(None) & (
            Memory.expires_at.is_(None) | (Memory.expires_at > adesso)
        )
        quante = int(await self._session.scalar(
            select(func.count()).select_from(Memory).where(vive)
        ) or 0)
        if quante < limite:
            return

        meno = (await self._session.execute(
            select(Memory).where(vive)
            .order_by(Memory.importance.asc(), Memory.last_referenced_at.asc().nulls_first())
            .limit(1)
        )).scalar_one_or_none()

        if meno is None or importanza <= (meno.importance or 0.0):
            raise MemoriaPiena(limite)

        meno.valid_to = adesso
        meno.meta = {**(meno.meta or {}), "chiusa": "limite del piano"}
        logger.info(
            "Memoria %s chiusa per far posto (limite %d)", str(meno.id)[:8], limite,
        )

    async def sostituisci(
        self, vecchia: Memory, nuova_contenuto: str, **campi: Any
    ) -> Memory:
        """Il fatto è cambiato: si chiude il vecchio e si apre il nuovo.

        Questo è il cuore della memoria temporale. «Vivo a Budapest» non viene
        cancellata quando arriva «mi sono trasferito a Vienna»: viene chiusa
        con una data di fine, e le due restano collegate.
        """
        adesso = utcnow()

        nuova = await self.ricorda(
            user_id=vecchia.user_id,
            content=nuova_contenuto,
            kind=campi.get("kind", vecchia.kind),
            personality_id=campi.get("personality_id", vecchia.personality_id),
            importance=campi.get("importance", vecchia.importance),
            confidence=campi.get("confidence", vecchia.confidence),
            source_message_id=campi.get("source_message_id"),
            valid_from=adesso,
            meta={**(vecchia.meta or {}), "sostituisce": str(vecchia.id)},
        )

        vecchia.valid_to = adesso
        vecchia.superseded_by = nuova.id
        await self._session.flush()

        logger.info(
            "Memoria %s superata da %s", str(vecchia.id)[:8], str(nuova.id)[:8],
        )
        return nuova

    async def dimentica(self, user_id: int, memory_id: uuid.UUID) -> bool:
        """Cancellazione vera, su richiesta di chi possiede la memoria.

        Qui si cancella davvero e non si chiude: chiudere conserverebbe il
        contenuto, e chi chiede di dimenticare qualcosa chiede che sparisca.
        La catena resta coerente perché `superseded_by` è `SET NULL`.
        """
        memoria = await self.una(user_id, memory_id)
        if memoria is None:
            return False

        await self._session.delete(memoria)
        await self._session.flush()
        logger.info("Memoria %s cancellata su richiesta", str(memory_id)[:8])
        return True

    async def dimentica_tutto(
        self, user_id: int, *, personality_id: Optional[uuid.UUID] = None
    ) -> int:
        """Cancella tutte le memorie di un utente, o quelle di una voce sola."""
        from sqlalchemy import delete

        condizioni = [Memory.user_id == user_id]
        if personality_id is not None:
            condizioni.append(Memory.personality_id == personality_id)

        risultato = await self._session.execute(delete(Memory).where(*condizioni))
        quante = risultato.rowcount or 0
        await self._session.flush()

        logger.info("Cancellate %d memorie dell'utente %s", quante, user_id)
        return quante

    # -- accessi -----------------------------------------------------------

    async def registra_accesso(
        self,
        *,
        subject_id: int,
        actor_id: Optional[int],
        action: str,
        count: int = 0,
        reason: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> MemoryAccess:
        """Chi ha guardato le memorie di chi.

        Un accesso non registrato alle memorie di qualcun altro è
        indistinguibile da un abuso — e queste sono cose confidate a un
        servizio, non dati operativi.
        """
        accesso = MemoryAccess(
            subject_id=subject_id,
            actor_id=actor_id,
            action=action,
            count=count,
            reason=reason,
            correlation_id=correlation_id,
            created_at=utcnow(),
        )
        self._session.add(accesso)
        await self._session.flush()
        return accesso

    async def accessi_a(
        self, subject_id: int, *, limite: int = 100
    ) -> Sequence[MemoryAccess]:
        """«Chi ha letto le mie memorie?» — con una risposta."""
        return (await self._session.execute(
            select(MemoryAccess)
            .where(MemoryAccess.subject_id == subject_id)
            .order_by(MemoryAccess.created_at.desc())
            .limit(limite)
        )).scalars().all()


def _scadenza_per(kind: str, adesso: datetime) -> Optional[datetime]:
    """Quando una memoria smette di valere, se ha una scadenza.

    Identità e preferenze non scadono: sono ciò che rende utile ricordare.
    Sessioni e impegni sì, per ragioni opposte — le prime perché sono
    impalcature, i secondi perché un impegno vecchio di un anno non è più un
    impegno.
    """
    if kind == "sessione":
        return adesso + timedelta(days=GIORNI_SESSIONE)
    if kind == "impegno":
        return adesso + timedelta(days=GIORNI_IMPEGNO)
    return None
