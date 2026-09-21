"""Consolidamento: ciò che tiene in piedi la memoria.

Senza, dopo mesi il recupero **peggiora** invece di migliorare. Non è
un'ipotesi: ogni conversazione aggiunge memorie, molte dicono la stessa cosa
con parole diverse, e quando sono mille le dieci che contano non emergono più.
Il consolidamento è la manutenzione che rende utile ricordare, non una
rifinitura da aggiungere quando ci sarà tempo.

Tre operazioni, in ordine di sicurezza crescente.

**Scadenza** — si chiudono le memorie il cui tempo è finito. Deterministica,
nessun giudizio.

**Fusione dei quasi-duplicati** — due memorie molto vicine nello spazio
vettoriale dicono probabilmente la stessa cosa: se ne tiene una, e
l'importanza della sopravvissuta sale, perché essere stata detta due volte è
di per sé un segnale.

**Conflitti** — due memorie che si contraddicono. Qui non si decide da soli:
la più recente vince **solo se il genere la rende sostituibile**. Che qualcuno
viva a Vienna sostituisce che viva a Budapest; che gli piaccia il jazz non
sostituisce che gli piaccia il blues, perché le preferenze si accumulano
invece di escludersi.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.memory_models import Memory
from ..knowledge.embedding import somiglianza_coseno
from ..observability.tracing import traccia
from .store import MemoryStore

logger = logging.getLogger(__name__)

#: Sopra questa somiglianza due memorie dicono la stessa cosa.
#:
#: 0,92 è alto di proposito. Sbagliare fondendo è peggio che sbagliare
#: lasciando due righe: una fusione sbagliata perde informazione e non si
#: recupera, un duplicato costa una riga e lo si fonde la volta dopo.
SOGLIA_DUPLICATO = 0.92

#: I generi in cui un fatto nuovo **sostituisce** il vecchio invece di
#: affiancarlo. Dove vivi è uno solo; cosa ti piace, no.
GENERI_ESCLUSIVI = ("identita",)

#: Quanto sale l'importanza di una memoria confermata da un duplicato.
PREMIO_CONFERMA = 0.1


@dataclass
class EsitoConsolidamento:
    scadute: int = 0
    fuse: int = 0
    superate: int = 0
    esaminate: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "esaminate": self.esaminate,
            "scadute": self.scadute,
            "fuse": self.fuse,
            "superate": self.superate,
        }

    @property
    def qualcosa_e_cambiato(self) -> bool:
        return bool(self.scadute or self.fuse or self.superate)


class Consolidatore:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._store = MemoryStore(session)

    async def consolida(self, user_id: int) -> EsitoConsolidamento:
        esito = EsitoConsolidamento()

        with traccia("memoria.consolidamento", utente=user_id):
            esito.scadute = await self._chiudi_scadute(user_id)

            vive = list(await self._store.per_utente(user_id, limite=1000))
            esito.esaminate = len(vive)

            esito.fuse = await self._fondi_duplicati(vive)
            esito.superate = await self._risolvi_conflitti(vive)

        if esito.qualcosa_e_cambiato:
            logger.info("Consolidamento per %s: %s", user_id, esito.to_dict())
        return esito

    async def _chiudi_scadute(self, user_id: int) -> int:
        """Chiude le memorie il cui tempo è finito.

        Chiude e non cancella: una memoria di sessione scaduta racconta
        comunque che quella conversazione c'è stata, e toglierla del tutto
        renderebbe la storia incompleta senza guadagnare nulla — occupa una
        riga, non un posto fra i risultati.
        """
        adesso = utcnow()

        # Una alla volta e non con un `UPDATE` massivo: quello cambierebbe le
        # righe ma **non** gli oggetti già caricati nella sessione, e chi li ha
        # in mano continuerebbe a leggere `valid_to = None` su memorie appena
        # chiuse — agendo su uno stato che non esiste più. Il difetto è muto,
        # perché il conteggio restituito resta corretto.
        #
        # Le memorie di un utente sono decine, non milioni: il costo di
        # passare dall'ORM è irrilevante, e la coerenza è gratuita.
        scadute = (await self._session.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.valid_to.is_(None),
                Memory.expires_at.is_not(None),
                Memory.expires_at <= adesso,
            )
        )).scalars().all()

        for memoria in scadute:
            memoria.valid_to = adesso

        if scadute:
            await self._session.flush()
        return len(scadute)

    async def _fondi_duplicati(self, memorie: Sequence[Memory]) -> int:
        """Chiude le memorie che ne ripetono una già presente."""
        fuse = 0
        # Dalla più importante: la sopravvissuta dev'essere quella che vale di
        # più, non quella che il database ha restituito per prima.
        ordinate = sorted(
            [m for m in memorie if m.embedding is not None],
            key=lambda m: (m.importance, m.times_referenced),
            reverse=True,
        )
        chiuse: set = set()

        for i, tenuta in enumerate(ordinate):
            if tenuta.id in chiuse:
                continue
            for candidata in ordinate[i + 1:]:
                if candidata.id in chiuse or candidata.kind != tenuta.kind:
                    continue

                s = somiglianza_coseno(tenuta.embedding, candidata.embedding)
                if s < SOGLIA_DUPLICATO:
                    continue

                # Essere stata detta due volte è un segnale: l'importanza
                # della sopravvissuta sale, e con essa la probabilità che
                # riemerga quando serve.
                tenuta.importance = min(1.0, tenuta.importance + PREMIO_CONFERMA)
                tenuta.times_referenced += candidata.times_referenced

                candidata.valid_to = utcnow()
                candidata.superseded_by = tenuta.id
                chiuse.add(candidata.id)
                fuse += 1

        if fuse:
            await self._session.flush()
        return fuse

    async def _risolvi_conflitti(self, memorie: Sequence[Memory]) -> int:
        """Chiude i fatti superati da uno più recente dello stesso genere.

        Solo per i generi esclusivi: dove vivi è uno solo, cosa ti piace no.
        Applicare la regola alle preferenze farebbe dimenticare tutto tranne
        l'ultima cosa detta, che è l'opposto di ricordare.
        """
        superate = 0

        per_genere: Dict[Tuple[str, Optional[uuid.UUID]], List[Memory]] = {}
        for m in memorie:
            if m.kind not in GENERI_ESCLUSIVI or m.embedding is None:
                continue
            per_genere.setdefault((m.kind, m.personality_id), []).append(m)

        for gruppo in per_genere.values():
            if len(gruppo) < 2:
                continue

            # Dalla più recente: è quella che sopravvive.
            gruppo.sort(key=lambda m: m.first_seen_at, reverse=True)
            recente = gruppo[0]

            for vecchia in gruppo[1:]:
                if vecchia.valid_to is not None:
                    continue
                # Si chiudono solo quelle che parlano della stessa cosa: due
                # memorie di identità possono riguardare il lavoro e la città,
                # e chiuderle a vicenda perderebbe metà di ciò che si sa.
                s = somiglianza_coseno(recente.embedding, vecchia.embedding)
                if s < 0.75:
                    continue

                vecchia.valid_to = utcnow()
                vecchia.superseded_by = recente.id
                superate += 1

        if superate:
            await self._session.flush()
        return superate


async def consolida_tutti(
    session: AsyncSession, *, limite_utenti: int = 100
) -> Dict[int, EsitoConsolidamento]:
    """Consolida le memorie di tutti gli utenti che ne hanno.

    È il lavoro periodico del worker CPU. Un utente per volta e non tutto
    insieme: le memorie di una persona si confrontano solo fra loro, e
    tenerle separate significa che un consolidamento fallito ne riguarda uno
    solo.
    """
    utenti = (await session.execute(
        select(Memory.user_id).distinct().limit(limite_utenti)
    )).scalars().all()

    consolidatore = Consolidatore(session)
    esiti: Dict[int, EsitoConsolidamento] = {}

    for user_id in utenti:
        try:
            esiti[user_id] = await consolidatore.consolida(user_id)
        except Exception:
            logger.exception("Consolidamento fallito per l'utente %s", user_id)

    return esiti
