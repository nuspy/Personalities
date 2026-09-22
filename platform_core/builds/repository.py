"""Il ciclo di vita di una realizzazione.

Le transizioni di stato stanno qui e non nei chiamanti perché sono poche e
vincolate: una build va da `in_coda` a `in_corso` a uno dei tre esiti finali,
e non torna indietro. Sparse fra worker e API, quelle regole si contraddicono
alla prima fretta — e una build che risulta `riuscita` senza artefatto, o
`in_corso` da tre giorni, è un dato che nessuno sa più interpretare.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional, Sequence

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.build_models import Build, BuildEvent, RuntimeBinding
from .queue import TIPI_SENZA_ACCELERATORE

logger = logging.getLogger(__name__)

#: Quante righe di avanzamento conservare per build.
#:
#: Un addestramento ne produce centinaia — una per passo — e tenerle tutte
#: gonfierebbe la tabella senza aggiungere nulla: chi guarda dopo vuole sapere
#: come è andata, non rivedere ogni passo. Le più vecchie si perdono, le
#: ultime restano, e quelle contengono sempre l'esito.
EVENTI_MASSIMI = 200


class BuildRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- lettura -----------------------------------------------------------

    async def per_id(self, build_id: uuid.UUID) -> Optional[Build]:
        return await self._session.get(Build, build_id)

    async def per_personalita(
        self, personality_id: uuid.UUID, *, limite: int = 50
    ) -> Sequence[Build]:
        return (await self._session.execute(
            select(Build)
            .where(Build.personality_id == personality_id)
            .order_by(desc(Build.created_at))
            .limit(limite)
        )).scalars().all()

    async def eventi(
        self, build_id: uuid.UUID, *, dopo: int = 0, limite: int = 200
    ) -> Sequence[BuildEvent]:
        """Le righe di avanzamento, da un certo punto in poi.

        `dopo` permette a chi osserva di riprendere da dove era rimasto senza
        rileggere tutto: è ciò che rende praticabile seguire un job lungo
        senza ricaricare la pagina.
        """
        return (await self._session.execute(
            select(BuildEvent)
            .where(BuildEvent.build_id == build_id, BuildEvent.id > dopo)
            .order_by(BuildEvent.id)
            .limit(limite)
        )).scalars().all()

    async def in_coda(self, *, limite: int = 50) -> Sequence[Build]:
        return (await self._session.execute(
            select(Build)
            .where(Build.status == "in_coda")
            .order_by(Build.created_at)
            .limit(limite)
        )).scalars().all()

    # -- scrittura ---------------------------------------------------------

    async def crea(
        self,
        *,
        personality_id: Optional[uuid.UUID],
        kind: str,
        params: Optional[Dict[str, Any]] = None,
        personality_version_id: Optional[uuid.UUID] = None,
        owner_id: Optional[int] = None,
    ) -> Build:
        build = Build(
            personality_id=personality_id,
            personality_version_id=personality_version_id,
            kind=kind,
            params=params,
            owner_id=owner_id,
            status="in_coda",
            message=(
                "In attesa di un worker"
                if kind in TIPI_SENZA_ACCELERATORE
                else "In attesa di un worker con acceleratore"
            ),
        )
        self._session.add(build)
        await self._session.flush()
        return build

    async def prendi_in_carico(
        self, build: Build, worker_id: str
    ) -> bool:
        """Segna che un worker ha cominciato.

        Restituisce `False` se la build non era più in coda: capita quando
        qualcuno l'ha annullata mentre stava per partire, e in quel caso il
        worker deve lasciarla stare invece di lavorare per nulla.
        """
        if build.status != "in_coda":
            logger.info(
                "Build %s non più in coda (%s): il worker la lascia",
                build.id, build.status,
            )
            return False

        build.status = "in_corso"
        build.worker_id = worker_id
        build.started_at = utcnow()
        build.progress = 0
        build.message = "Avvio"
        await self._session.flush()
        return True

    async def avanza(
        self, build: Build, progresso: int, messaggio: str, *, livello: str = "info"
    ) -> None:
        build.progress = max(0, min(100, progresso))
        build.message = messaggio

        self._session.add(BuildEvent(
            build_id=build.id,
            progress=build.progress,
            message=messaggio,
            level=livello,
            created_at=utcnow(),
        ))
        await self._session.flush()

    async def conclusa(
        self,
        build: Build,
        *,
        artifact_path: Optional[str] = None,
        artifact_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        build.status = "riuscita"
        build.progress = 100
        build.message = "Completata"
        build.artifact_path = artifact_path
        build.artifact_meta = artifact_meta
        build.finished_at = utcnow()
        await self.avanza(build, 100, "Completata")

    async def fallita(self, build: Build, errore: str) -> None:
        build.status = "fallita"
        build.error = errore[:2000]
        build.message = "Fallita"
        build.finished_at = utcnow()
        await self.avanza(build, build.progress, f"Fallita: {errore[:200]}", livello="errore")

    async def annulla(self, build: Build) -> bool:
        """Annulla una build che non è ancora partita.

        Non interrompe un job in corso: fermare un addestramento a metà
        richiede che il worker collabori, e fingere di averlo fatto
        lascerebbe un processo vivo e una riga che dice il contrario.
        """
        if build.status != "in_coda":
            return False

        build.status = "annullata"
        build.message = "Annullata prima di partire"
        build.finished_at = utcnow()
        await self._session.flush()
        return True

    async def potatura_eventi(self, build: Build) -> int:
        """Tiene solo gli ultimi eventi di una build."""
        eccedenti = (await self._session.execute(
            select(BuildEvent.id)
            .where(BuildEvent.build_id == build.id)
            .order_by(desc(BuildEvent.id))
            .offset(EVENTI_MASSIMI)
        )).scalars().all()

        if not eccedenti:
            return 0

        from sqlalchemy import delete

        await self._session.execute(
            delete(BuildEvent).where(BuildEvent.id.in_(eccedenti))
        )
        return len(eccedenti)


class RuntimeRepository:
    """Quale realizzazione una personalità sta usando."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def per_personalita(
        self, personality_id: uuid.UUID
    ) -> Optional[RuntimeBinding]:
        return await self._session.get(RuntimeBinding, personality_id)

    async def imposta(
        self,
        personality_id: uuid.UUID,
        *,
        mode: str,
        build_id: Optional[uuid.UUID] = None,
        serving: Optional[str] = None,
    ) -> RuntimeBinding:
        """Sceglie cosa serve una personalità.

        «Realizzazione multipla, uso singolo»: le build restano tutte, ma solo
        una è in uso, e questa riga è il posto dove sta scritto quale.
        """
        legame = await self.per_personalita(personality_id)
        if legame is None:
            legame = RuntimeBinding(personality_id=personality_id)
            self._session.add(legame)

        legame.mode = mode
        legame.build_id = build_id
        legame.serving = serving
        await self._session.flush()
        return legame

    async def modo_di(self, personality_id: uuid.UUID) -> str:
        """Il modo con cui rispondere. `rag` quando non è stato scelto altro."""
        legame = await self.per_personalita(personality_id)
        return legame.mode if legame else "rag"
