"""Abbonamenti: chi ha cosa, e fino a quando.

**Il pagamento non è qui.** `BillingProvider` è il confine: quando arriverà un
fornitore vero — Stripe o altri — sarà un'implementazione di questo protocollo,
e il resto della piattaforma non cambierà. Per ora c'è quello finto, che
concede tutto: in sviluppo serve provare il servizio, non il pagamento.

**Il rinnovo accredita, non azzera.** A ogni periodo il piano aggiunge i suoi
crediti con una scadenza; quelli comprati a parte restano. Azzerare il saldo
al rinnovo sarebbe più semplice e porterebbe via crediti pagati.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Protocol, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.billing_models import Plan, Subscription
from .credits import RegistroCrediti
from .entitlements import PIANO_PREDEFINITO, Diritti, diritti_da

logger = logging.getLogger(__name__)

#: Quanto dura un periodo mensile. Trenta giorni e non «il mese»: un mese
#: civile dura fra ventotto e trentuno giorni, e far dipendere i crediti dal
#: calendario significa che a febbraio se ne ricevono meno senza che nessuno
#: lo abbia deciso.
GIORNI_PERIODO = 30


class BillingProvider(Protocol):
    """Il confine verso chi incassa davvero.

    Volutamente magro: creare un abbonamento, disdirlo, dire se è pagato.
    Tutto ciò che riguarda carte, ricevute e imposte resta dall'altra parte —
    quella è materia in cui sbagliare costa, e va gestita da chi lo fa di
    mestiere.
    """

    async def crea_abbonamento(
        self, user_id: int, piano: Plan, *, annuale: bool = False,
    ) -> str:
        """Restituisce l'identificativo esterno dell'abbonamento."""
        ...

    async def disdici(self, external_id: str) -> None:
        ...

    async def e_pagato(self, external_id: str) -> bool:
        ...


class ProviderDiSviluppo(BillingProvider):
    """Concede tutto senza incassare nulla.

    Esiste perché in sviluppo serve provare il servizio, non il pagamento. Il
    nome lo dice a chi legge i log: un `external_id` che comincia per
    `sviluppo-` in un database di produzione è un difetto visibile a occhio.
    """

    async def crea_abbonamento(
        self, user_id: int, piano: Plan, *, annuale: bool = False,
    ) -> str:
        return f"sviluppo-{uuid.uuid4().hex[:12]}"

    async def disdici(self, external_id: str) -> None:
        return None

    async def e_pagato(self, external_id: str) -> bool:
        return True


@dataclass
class EsitoRinnovo:
    abbonamento_id: uuid.UUID
    crediti_accreditati: int = 0
    nuovo_periodo_fine: Optional[datetime] = None
    chiuso: bool = False
    motivo: str = ""


class GestoreAbbonamenti:
    def __init__(
        self,
        session: AsyncSession,
        provider: Optional[BillingProvider] = None,
    ) -> None:
        self._session = session
        self._provider = provider or ProviderDiSviluppo()
        self._crediti = RegistroCrediti(session)

    # -- lettura -----------------------------------------------------------

    async def piani(self, *, solo_attivi: bool = True) -> Sequence[Plan]:
        query = select(Plan).order_by(Plan.rank)
        if solo_attivi:
            query = query.where(Plan.active.is_(True))
        return (await self._session.execute(query)).scalars().all()

    async def piano_per_slug(self, slug: str) -> Optional[Plan]:
        return (await self._session.execute(
            select(Plan).where(Plan.slug == slug)
        )).scalar_one_or_none()

    async def abbonamento_di(self, user_id: int) -> Optional[Subscription]:
        """L'abbonamento vivo di un utente, se ne ha uno.

        Il più recente fra quelli vivi: un utente che passa da un piano a un
        altro può averne due per qualche istante, e servire il vecchio
        sarebbe l'errore più evidente possibile.
        """
        return (await self._session.execute(
            select(Subscription)
            .where(
                Subscription.user_id == user_id,
                Subscription.status.in_(("in_prova", "attivo")),
            )
            .order_by(Subscription.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()

    async def tariffe_in_vigore(self) -> bool:
        """C'è un catalogo, cioè: questa installazione fa pagare?

        Serve a non far dipendere il funzionamento del prodotto da un seed.
        Senza nessun piano in tabella, addebitare ogni risposta significa che
        **nessuno** può parlare con nessuna personalità: un'installazione
        appena migrata sarebbe muta, e il motivo — una tabella vuota — non
        somiglia per niente al sintomo.

        La scelta opposta, regalare tutto per una tabella vuota, si paga in
        mancati incassi e si vede nei log; questa si paga in un servizio che
        non parte e sembra rotto. Fra i due si sceglie quello che lascia il
        prodotto usabile e lo dichiara.
        """
        return bool(await self._session.scalar(
            select(Plan.id).where(Plan.active.is_(True)).limit(1)
        ))

    async def diritti_di(self, user_id: int) -> Diritti:
        """Cosa può fare questo utente, adesso."""
        abbonamento = await self.abbonamento_di(user_id)
        base = await self.piano_per_slug(PIANO_PREDEFINITO)
        return diritti_da(abbonamento, piano_base=base)

    # -- scrittura ---------------------------------------------------------

    async def apri_piano_base(self, user_id: int) -> Optional[Subscription]:
        """Apre il piano gratuito a chi non ha nessun abbonamento.

        Serve perché i diritti e i crediti ricadono in modi diversi: senza
        abbonamento `diritti_di` restituisce comunque quelli del piano
        gratuito — sono scritti nel codice come ultima rete — ma i crediti no,
        perché un saldo è una somma di righe e righe non ce ne sono. Il
        risultato era che un account nuovo poteva vedere le voci gratuite e
        riceveva 402 al primo messaggio: il peggior modo di cominciare.

        Non fa nulla se un abbonamento c'è già, né se il piano gratuito non è
        stato ancora scritto in tabella: in quel caso il servizio funziona
        lo stesso sui diritti base, e inventare un piano da qui
        significherebbe che il catalogo si crea da solo.
        """
        if await self.abbonamento_di(user_id) is not None:
            return None

        piano = await self.piano_per_slug(PIANO_PREDEFINITO)
        if piano is None:
            logger.warning(
                "Nessun piano «%s» in catalogo: l'utente %s resta sui diritti "
                "base e senza crediti.", PIANO_PREDEFINITO, user_id,
            )
            return None

        return await self.sottoscrivi(user_id, piano)

    async def sottoscrivi(
        self,
        user_id: int,
        piano: Plan,
        *,
        annuale: bool = False,
        in_prova: bool = False,
    ) -> Subscription:
        """Apre un abbonamento e accredita i crediti del primo periodo."""
        precedente = await self.abbonamento_di(user_id)
        if precedente is not None:
            # Chiuso e non cancellato: il passaggio da un piano all'altro
            # resta leggibile nella storia dell'utente.
            precedente.status = "disdetto"
            precedente.cancel_at = utcnow()

        external_id = await self._provider.crea_abbonamento(
            user_id, piano, annuale=annuale,
        )

        inizio = utcnow()
        durata = timedelta(days=365 if annuale else GIORNI_PERIODO)

        abbonamento = Subscription(
            user_id=user_id,
            plan_id=piano.id,
            status="in_prova" if in_prova else "attivo",
            period_start=inizio,
            period_end=inizio + durata,
            external_id=external_id,
        )
        self._session.add(abbonamento)
        await self._session.flush()

        if piano.credits_per_period > 0:
            await self._crediti.accredita(
                user_id,
                piano.credits_per_period,
                reason="accredito_piano",
                note=f"piano {piano.slug}",
                # Scadono con il periodo: un piano che ne dà cento al mese non
                # deve farli accumulare all'infinito, o il piano più basso
                # diventa col tempo indistinguibile dal più alto.
                expires_at=abbonamento.period_end,
            )

        logger.info(
            "Abbonamento %s aperto per l'utente %s (piano %s)",
            str(abbonamento.id)[:8], user_id, piano.slug,
        )
        return abbonamento

    async def disdici(self, abbonamento: Subscription) -> Subscription:
        """Segna la disdetta. L'abbonamento resta attivo fino a scadenza.

        È già stato pagato: chiuderlo subito toglierebbe qualcosa a cui
        l'utente ha diritto, e la differenza fra «ho disdetto» e «non ho più
        accesso» è esattamente ciò che rende accettabile disdire.
        """
        if abbonamento.external_id:
            await self._provider.disdici(abbonamento.external_id)

        abbonamento.cancel_at = abbonamento.period_end
        await self._session.flush()
        return abbonamento

    async def rinnova(self, abbonamento: Subscription) -> EsitoRinnovo:
        """Apre il periodo successivo, o chiude l'abbonamento se disdetto."""
        esito = EsitoRinnovo(abbonamento_id=abbonamento.id)
        adesso = utcnow()

        if abbonamento.cancel_at is not None and abbonamento.cancel_at <= adesso:
            abbonamento.status = "disdetto"
            esito.chiuso = True
            esito.motivo = "disdetto dall'utente"
            await self._session.flush()
            return esito

        if abbonamento.external_id:
            if not await self._provider.e_pagato(abbonamento.external_id):
                # Sospeso e non disdetto: un pagamento può fallire per una
                # carta scaduta, e chiudere l'abbonamento al primo tentativo
                # non riuscito perderebbe un cliente che voleva restare.
                abbonamento.status = "sospeso"
                esito.chiuso = True
                esito.motivo = "pagamento non riuscito"
                await self._session.flush()
                return esito

        durata = abbonamento.period_end - abbonamento.period_start
        abbonamento.period_start = abbonamento.period_end
        abbonamento.period_end = abbonamento.period_start + durata
        abbonamento.status = "attivo"

        piano = abbonamento.plan
        if piano.credits_per_period > 0:
            await self._crediti.accredita(
                abbonamento.user_id,
                piano.credits_per_period,
                reason="accredito_piano",
                note=f"rinnovo {piano.slug}",
                expires_at=abbonamento.period_end,
            )
            esito.crediti_accreditati = piano.credits_per_period

        esito.nuovo_periodo_fine = abbonamento.period_end
        await self._session.flush()
        return esito

    async def da_rinnovare(self, *, limite: int = 100) -> Sequence[Subscription]:
        """Gli abbonamenti il cui periodo è finito.

        È il lavoro periodico del worker: senza, un abbonamento pagato smette
        di accreditare crediti alla fine del primo mese e nessuno se ne
        accorge finché l'utente non se ne lamenta.
        """
        return (await self._session.execute(
            select(Subscription)
            .where(
                Subscription.status.in_(("in_prova", "attivo")),
                Subscription.period_end <= utcnow(),
            )
            .order_by(Subscription.period_end)
            .limit(limite)
        )).scalars().all()
