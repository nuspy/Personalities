"""Chiusura di una conversazione: cosa resta.

**Quando una conversazione è «finita»** non è una domanda con risposta netta:
nessuno preme un pulsante per dire di aver smesso. Si considera chiusa quando
è passato abbastanza tempo dall'ultimo messaggio — abbastanza da rendere
improbabile che riprenda, non tanto da aver dimenticato di che si parlava.

Il lavoro lo fa il worker CPU, periodicamente. Farlo alla fine di una
richiesta HTTP sarebbe sbagliato due volte: non si saprebbe che è l'ultima, e
costerebbe a chi ha appena scritto l'attesa di un'estrazione che non lo
riguarda.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.memory_models import Memory
from ..domain.models import Conversation, Message
from ..knowledge.embedding import Embedder
from ..llm.base import Message as MessaggioLLM
from .extraction import Estrattore
from .store import MemoriaPiena, MemoryStore

logger = logging.getLogger(__name__)

#: Dopo quanto silenzio una conversazione si considera conclusa.
#:
#: Trenta minuti: sotto, si rischia di estrarre da uno scambio ancora in corso
#: e di ritrovarsi memorie che il seguito contraddice; sopra, chi torna dopo
#: un'ora non trova traccia di ciò che aveva raccontato.
SILENZIO_MINUTI = 30

#: Quanti turni guardare. Le conversazioni lunghe si estraggono dalla coda:
#: è lì che le cose si sono precisate, e l'inizio contiene le versioni che
#: sono state poi corrette.
TURNI_DA_ESAMINARE = 40


@dataclass
class EsitoChiusura:
    conversazione_id: uuid.UUID
    memorie: int = 0
    riassunto: bool = False
    saltata: str = ""


class ChiusuraConversazioni:
    def __init__(
        self,
        session: AsyncSession,
        provider,
        embedder: Optional[Embedder] = None,
        *,
        limite_per_utente: Optional[Callable[[int], Awaitable[Optional[int]]]] = None,
    ) -> None:
        self._session = session
        self._estrattore = Estrattore(provider)
        self._store = MemoryStore(session, embedder)
        #: Quante memorie il piano dell'utente consente. Una funzione e non un
        #: numero perché la passata attraversa conversazioni di utenti diversi;
        #: e passata da fuori perché la memoria non deve sapere che esistono
        #: i piani — le due cose devono poter cambiare separatamente.
        self._limite_per_utente = limite_per_utente

    async def da_chiudere(self, *, limite: int = 20) -> Sequence[Conversation]:
        """Le conversazioni ferme da abbastanza tempo e mai estratte.

        Il filtro sull'assenza di memorie è ciò che rende l'operazione
        ripetibile: senza, ogni passaggio del worker riestrarrebbe le stesse
        conversazioni, moltiplicando le memorie a ogni giro.
        """
        soglia = utcnow() - timedelta(minutes=SILENZIO_MINUTI)

        gia_estratta = (
            select(Memory.id)
            .join(Message, Message.id == Memory.source_message_id)
            .where(Message.conversation_id == Conversation.id)
            .limit(1)
        )

        return (await self._session.execute(
            select(Conversation)
            .where(
                Conversation.last_message_at.is_not(None),
                Conversation.last_message_at < soglia,
                Conversation.status == "active",
                ~exists(gia_estratta),
            )
            .order_by(Conversation.last_message_at)
            .limit(limite)
        )).scalars().all()

    async def chiudi(self, conversazione: Conversation) -> EsitoChiusura:
        esito = EsitoChiusura(conversazione_id=conversazione.id)

        if conversazione.owner_id is None:
            esito.saltata = "conversazione senza proprietario"
            return esito

        messaggi = (await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversazione.id)
            .order_by(Message.created_at.desc())
            .limit(TURNI_DA_ESAMINARE)
        )).scalars().all()

        if len(messaggi) < 2:
            esito.saltata = "troppo breve"
            return esito

        turni = [
            MessaggioLLM(role=m.role, content=m.content)  # type: ignore[arg-type]
            for m in reversed(messaggi)
        ]
        ultimo_utente = next(
            (m for m in messaggi if m.role == "user"), None,
        )

        limite = (
            await self._limite_per_utente(conversazione.owner_id)
            if self._limite_per_utente else None
        )

        estratte = await self._estrattore.estrai(turni)
        salvate = 0
        for memoria in estratte:
            try:
                await self._store.ricorda(
                    user_id=conversazione.owner_id,
                    content=memoria.contenuto,
                    kind=memoria.genere,
                    personality_id=conversazione.personality_id,
                    importance=memoria.importanza,
                    confidence=memoria.confidenza,
                    # Da quale messaggio viene: è la risposta a «perché ti
                    # ricordi questo?», e senza di essa la memoria è
                    # un'affermazione senza provenienza.
                    source_message_id=ultimo_utente.id if ultimo_utente else None,
                    limite=limite,
                )
            except MemoriaPiena:
                # Il piano è pieno e questa conta meno di tutte le presenti:
                # si rinuncia a lei, non alla conversazione.
                continue
            salvate += 1
        esito.memorie = salvate

        riassunto = await self._estrattore.riassumi(turni)
        if riassunto:
            await self._store.ricorda(
                user_id=conversazione.owner_id,
                content=riassunto,
                kind="sessione",
                personality_id=conversazione.personality_id,
                importance=0.4,
                confidence=0.9,
                source_message_id=ultimo_utente.id if ultimo_utente else None,
                meta={"conversazione": str(conversazione.id)},
            )
            esito.riassunto = True

        logger.info(
            "Conversazione %s chiusa: %d memorie%s",
            str(conversazione.id)[:8], esito.memorie,
            " e un riassunto" if esito.riassunto else "",
        )
        return esito

    async def chiudi_le_ferme(self, *, limite: int = 20) -> List[EsitoChiusura]:
        esiti: List[EsitoChiusura] = []
        for conversazione in await self.da_chiudere(limite=limite):
            try:
                esiti.append(await self.chiudi(conversazione))
            except Exception:
                # Una conversazione che fallisce non deve fermare le altre:
                # il worker gira periodicamente, e bloccarsi sulla prima
                # significherebbe non estrarre più nulla da nessuna.
                logger.exception(
                    "Chiusura fallita per %s", conversazione.id,
                )
        return esiti
