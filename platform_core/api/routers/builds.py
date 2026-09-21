"""Realizzazioni: avvio, stato, avanzamento.

**Il 409 è il punto di questo modulo.** Se nessun worker con acceleratore è
registrato, avviare una build significherebbe mettere in coda un lavoro che
nessuno raccoglierà: l'utente vedrebbe «in attesa» per sempre, e nulla nella
schermata direbbe che l'attesa non finirà. Qui si rifiuta subito, **con il
motivo**, e il motivo è quello che il registro delle capacità riporta.

Il controllo nell'interfaccia — la casella disabilitata nella scheda
Realizzazione — è cortesia. Questo è la regola.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...auth.dependencies import CurrentUser, DbSession, require_role
from ...builds.queue import CodaBuild, JobBuild
from ...builds.repository import BuildRepository, RuntimeRepository
from ...capabilities.registry import CapabilityRegistry, Feature
from ...domain.session import SessionFactory
from ..deps import get_capability_registry, get_key_value_store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["realizzazioni"],
    dependencies=[Depends(require_role("admin"))],
)

#: Quale capacità serve a ciascun tipo di realizzazione.
CAPACITA_RICHIESTA = {
    "lora": Feature.BUILD_LORA,
    "finetune": Feature.FULL_FINETUNE,
    "gguf": Feature.GGUF_EXPORT,
    "merge": Feature.BUILD_LORA,
}


class AvviaBuild(BaseModel):
    kind: str = Field(pattern=r"^(lora|finetune|gguf|merge)$")
    params: Dict[str, Any] = Field(default_factory=dict)


class ScegliRuntime(BaseModel):
    mode: str = Field(pattern=r"^(rag|lora|finetune)$")
    build_id: Optional[uuid.UUID] = None
    serving: Optional[str] = None


def _build_json(b) -> Dict[str, Any]:
    return {
        "id": str(b.id),
        "kind": b.kind,
        "status": b.status,
        "progress": b.progress,
        "message": b.message,
        "error": b.error,
        "worker_id": b.worker_id,
        "artifact_path": b.artifact_path,
        "created_at": b.created_at.isoformat(),
        "started_at": b.started_at.isoformat() if b.started_at else None,
        "finished_at": b.finished_at.isoformat() if b.finished_at else None,
    }


@router.get("/personalities/{personality_id}/builds")
async def elenco_build(
    personality_id: uuid.UUID, session: DbSession,
) -> List[Dict[str, Any]]:
    return [
        _build_json(b)
        for b in await BuildRepository(session).per_personalita(personality_id)
    ]


@router.post("/personalities/{personality_id}/builds", status_code=202)
async def avvia_build(
    personality_id: uuid.UUID,
    payload: AvviaBuild,
    session: DbSession,
    user: CurrentUser,
    registry: Annotated[CapabilityRegistry, Depends(get_capability_registry)],
    supporto: Annotated[Any, Depends(get_key_value_store)],
) -> Dict[str, Any]:
    """Mette in coda una realizzazione, se qualcuno può eseguirla."""
    from ...domain.knowledge_models import Personality

    personalita = await session.get(Personality, personality_id)
    if personalita is None:
        raise HTTPException(status_code=404, detail="Personalità non trovata")

    disponibile, motivo = registry.can(CAPACITA_RICHIESTA[payload.kind])
    if not disponibile:
        # 409 e non 503: il servizio funziona, è la richiesta a essere
        # impossibile **adesso**. E non 400: non c'è nulla di sbagliato in ciò
        # che è stato chiesto, manca solo la macchina che lo faccia.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=motivo,
        )

    repo = BuildRepository(session)
    build = await repo.crea(
        personality_id=personality_id,
        kind=payload.kind,
        params=payload.params,
        personality_version_id=personalita.current_version_id,
        owner_id=user.id,
    )
    await session.commit()

    CodaBuild(supporto).accoda(JobBuild(
        build_id=build.id, kind=build.kind, personality_id=personality_id,
    ))

    return {**_build_json(build), "in_coda": True}


@router.get("/builds/{build_id}")
async def stato_build(build_id: uuid.UUID, session: DbSession) -> Dict[str, Any]:
    build = await BuildRepository(session).per_id(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail="Realizzazione non trovata")
    return _build_json(build)


@router.post("/builds/{build_id}/cancel")
async def annulla_build(
    build_id: uuid.UUID, session: DbSession,
) -> Dict[str, Any]:
    """Annulla una realizzazione non ancora partita.

    Una già in corso non si ferma da qui: interrompere un addestramento
    richiede che il worker collabori, e dichiararlo annullato mentre un
    processo continua a girare lascerebbe una riga che contraddice la realtà.
    """
    repo = BuildRepository(session)
    build = await repo.per_id(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail="Realizzazione non trovata")

    if not await repo.annulla(build):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"La realizzazione è {build.status}: si annulla solo ciò che "
                f"non è ancora partito."
            ),
        )

    await session.commit()
    return _build_json(build)


@router.get("/builds/{build_id}/events")
async def avanzamento(
    build_id: uuid.UUID,
    session: DbSession,
    session_factory: SessionFactory,
    dopo: int = 0,
) -> StreamingResponse:
    """L'avanzamento, mentre accade.

    Gli eventi si leggono dal database e non da un canale in memoria: chi apre
    la console a job già iniziato vede anche ciò che è successo prima. Un
    flusso che comincia dall'istante in cui ci si collega mostra metà storia a
    chi arriva tardi, che è quasi sempre.
    """
    build = await BuildRepository(session).per_id(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail="Realizzazione non trovata")

    async def flusso():
        ultimo = dopo
        for _ in range(600):   # circa venti minuti a due secondi per giro
            async with session_factory() as s:
                repo = BuildRepository(s)
                eventi = await repo.eventi(build_id, dopo=ultimo)
                for e in eventi:
                    ultimo = e.id
                    yield _sse("progress", {
                        "id": e.id, "progress": e.progress,
                        "message": e.message, "level": e.level,
                    })

                corrente = await repo.per_id(build_id)
                if corrente is not None and corrente.conclusa:
                    yield _sse("done", _build_json(corrente))
                    return

            await asyncio.sleep(2)

        yield _sse("timeout", {
            "message": "Il flusso si è chiuso; la realizzazione continua.",
        })

    return StreamingResponse(
        flusso(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/personalities/{personality_id}/runtime")
async def runtime(personality_id: uuid.UUID, session: DbSession) -> Dict[str, Any]:
    legame = await RuntimeRepository(session).per_personalita(personality_id)
    return {
        "mode": legame.mode if legame else "rag",
        "build_id": str(legame.build_id) if legame and legame.build_id else None,
        "serving": legame.serving if legame else None,
    }


@router.put("/personalities/{personality_id}/runtime")
async def scegli_runtime(
    personality_id: uuid.UUID,
    payload: ScegliRuntime,
    session: DbSession,
    registry: Annotated[CapabilityRegistry, Depends(get_capability_registry)],
) -> Dict[str, Any]:
    """Sceglie con quale realizzazione servire una personalità.

    «Realizzazione multipla, uso singolo»: le build restano tutte, una sola è
    in uso.
    """
    repo = BuildRepository(session)

    if payload.mode != "rag":
        if payload.build_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Il modo '{payload.mode}' richiede una realizzazione",
            )
        build = await repo.per_id(payload.build_id)
        if build is None or build.personality_id != personality_id:
            raise HTTPException(
                status_code=404, detail="Realizzazione non trovata",
            )
        if build.status != "riuscita":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"La realizzazione è {build.status}: non è servibile.",
            )

        disponibile, motivo = registry.can(Feature.LOCAL_INFERENCE)
        if not disponibile:
            # Si accetta comunque, ma si dice cosa accadrà: la degradazione
            # dichiarata è preferibile sia al rifiuto sia al silenzio.
            logger.info(
                "Modo %s scelto senza motore locale: %s", payload.mode, motivo,
            )

    legame = await RuntimeRepository(session).imposta(
        personality_id,
        mode=payload.mode,
        build_id=payload.build_id,
        serving=payload.serving,
    )
    await session.commit()

    return {
        "mode": legame.mode,
        "build_id": str(legame.build_id) if legame.build_id else None,
        "serving": legame.serving,
    }


def _sse(evento: str, dati: dict) -> str:
    return f"event: {evento}\ndata: {json.dumps(dati, ensure_ascii=False)}\n\n"
