"""I limiti per piano, applicati.

Il catalogo li definiva — messaggi al giorno, conversazioni, memorie — e
nessuno li controllava: un piano gratuito da venti messaggi al giorno ne
concedeva quanti se ne volevano. Qui si contano e si fanno rispettare.

**Tre limiti, tre comportamenti**, perché si superano in modi diversi:

| Limite | Si supera… | Cosa succede |
|---|---|---|
| messaggi al giorno | col tempo, e passa da sé | 429 con il tempo di attesa: fra un po' si può di nuovo |
| conversazioni | accumulandole | 403 con l'indicazione: archiviarne una libera il posto |
| memorie | ricordando troppo | la meno importante lascia il posto, se la nuova conta di più |

**Una finestra mobile di 24 ore, non «oggi».** «Al giorno» a mezzanotte UTC
significherebbe che a Budapest il contatore si azzera all'una di notte, e che
chi scrive alle 23:50 e alle 00:10 ha usato due giorni in venti minuti. La
finestra mobile non dipende da nessun fuso orario e si spiega da sola: «hai
mandato venti messaggi nelle ultime 24 ore».

**Solo dove si fa pagare.** Senza un catalogo attivo l'installazione non
addebita, e allora nemmeno limita: sarebbe incoerente regalare le risposte e
contingentarle.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.memory_models import Memory
from ..domain.models import Conversation, Message
from .entitlements import Diritti

logger = logging.getLogger(__name__)

#: La finestra dei messaggi «al giorno».
FINESTRA = timedelta(hours=24)


class QuotaSuperata(Exception):
    """Un limite del piano è raggiunto."""

    def __init__(
        self, limite: str, *, consentiti: int, usati: int,
        riprova_tra: Optional[int] = None, messaggio: str = "",
    ) -> None:
        super().__init__(messaggio or f"limite «{limite}» raggiunto: {usati}/{consentiti}")
        self.limite = limite
        self.consentiti = consentiti
        self.usati = usati
        #: Secondi prima che il limite si liberi da solo; `None` se non si
        #: libera col tempo (le conversazioni si liberano archiviandole).
        self.riprova_tra = riprova_tra
        self.messaggio = messaggio or str(self)


@dataclass
class Uso:
    """Quanto di ciascun limite è stato usato, per mostrarlo all'utente."""

    messaggi_24h: int
    conversazioni: int
    memorie: int

    def to_dict(self, diritti: Diritti) -> Dict[str, Dict[str, Optional[int]]]:
        def voce(nome: str, usati: int) -> Dict[str, Optional[int]]:
            limite = diritti.limite(nome)
            return {
                "usati": usati,
                "limite": limite,
                "restanti": None if limite is None else max(0, limite - usati),
            }

        return {
            "messaggi_al_giorno": voce("messaggi_al_giorno", self.messaggi_24h),
            "conversazioni": voce("conversazioni", self.conversazioni),
            "memorie": voce("memorie", self.memorie),
        }


class ContatoreQuote:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- conteggi ----------------------------------------------------------

    async def messaggi_24h(self, user_id: int) -> int:
        """I messaggi dell'utente nelle ultime 24 ore.

        Solo i suoi — `role = user` — e non le risposte: il limite è su quante
        domande si fanno, non su quanto è lunga la conversazione.
        """
        return int(await self._session.scalar(
            select(func.count())
            .select_from(Message)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(
                Conversation.owner_id == user_id,
                Message.role == "user",
                Message.created_at >= utcnow() - FINESTRA,
            )
        ) or 0)

    async def conversazioni_attive(self, user_id: int) -> int:
        """Le conversazioni non archiviate: archiviarne una libera il posto."""
        return int(await self._session.scalar(
            select(func.count()).select_from(Conversation).where(
                Conversation.owner_id == user_id, Conversation.status == "active",
            )
        ) or 0)

    async def memorie_vive(self, user_id: int) -> int:
        adesso = utcnow()
        return int(await self._session.scalar(
            select(func.count()).select_from(Memory).where(
                Memory.user_id == user_id,
                Memory.valid_to.is_(None),
                Memory.superseded_by.is_(None),
                (Memory.expires_at.is_(None)) | (Memory.expires_at > adesso),
            )
        ) or 0)

    async def uso(self, user_id: int) -> Uso:
        return Uso(
            messaggi_24h=await self.messaggi_24h(user_id),
            conversazioni=await self.conversazioni_attive(user_id),
            memorie=await self.memorie_vive(user_id),
        )

    # -- verifiche ---------------------------------------------------------

    async def verifica_messaggio(
        self, user_id: int, diritti: Diritti, *, nuova_conversazione: bool,
    ) -> None:
        """Solleva `QuotaSuperata` se questo messaggio non si può mandare."""
        limite = diritti.limite("messaggi_al_giorno")
        if limite is not None:
            usati = await self.messaggi_24h(user_id)
            if usati >= limite:
                raise QuotaSuperata(
                    "messaggi_al_giorno", consentiti=limite, usati=usati,
                    riprova_tra=await self._secondi_al_prossimo_posto(user_id),
                    messaggio=(
                        f"Hai mandato {usati} messaggi nelle ultime 24 ore: il "
                        f"piano «{diritti.piano}» ne consente {limite}."
                    ),
                )

        if nuova_conversazione:
            limite = diritti.limite("conversazioni")
            if limite is not None:
                usate = await self.conversazioni_attive(user_id)
                if usate >= limite:
                    raise QuotaSuperata(
                        "conversazioni", consentiti=limite, usati=usate,
                        messaggio=(
                            f"Hai {usate} conversazioni aperte: il piano "
                            f"«{diritti.piano}» ne consente {limite}. "
                            f"Archiviane una per aprirne un'altra."
                        ),
                    )

    async def _secondi_al_prossimo_posto(self, user_id: int) -> int:
        """Fra quanto il messaggio più vecchio della finestra ne esce.

        È il numero da mettere in `Retry-After`: dire «riprova più tardi»
        senza dire quando costringe a riprovare a caso.
        """
        piu_vecchio = await self._session.scalar(
            select(func.min(Message.created_at))
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(
                Conversation.owner_id == user_id,
                Message.role == "user",
                Message.created_at >= utcnow() - FINESTRA,
            )
        )
        if piu_vecchio is None:
            return 0
        restante = (piu_vecchio + FINESTRA) - utcnow()
        return max(1, int(restante.total_seconds()))
