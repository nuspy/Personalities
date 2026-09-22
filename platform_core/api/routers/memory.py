"""Memoria: ispezione, cancellazione, e chi l'ha guardata.

Due gruppi di rotte con due perimetri diversi.

**`/memory/…`** appartiene all'utente: vede le proprie memorie, la storia di
ciò che è cambiato, e può far dimenticare. Non registra accessi — leggere le
proprie cose non è un evento da sorvegliare, e un registro che si riempie
degli accessi legittimi del proprietario nasconde quelli che contano.

**`/admin/users/{id}/memories`** appartiene a chi amministra, e **ogni lettura
lascia una traccia**. Sono cose confidate a un servizio, non dati operativi:
un accesso non registrato alle memorie di qualcun altro è indistinguibile da
un abuso.
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ...auth.dependencies import CurrentUser, DbSession, require_role
from ...domain.memory_models import Memory
from ...knowledge.embedding import Embedder
from ...billing.plans import GestoreAbbonamenti
from ...memory.consolidation import Consolidatore
from ...memory.store import MemoriaPiena, MemoryStore
from ...observability.correlation import current_correlation_id
from ..deps import get_embedder

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memoria"])
admin_router = APIRouter(
    prefix="/admin",
    tags=["memoria"],
    dependencies=[Depends(require_role("admin"))],
)


class AggiungiMemoria(BaseModel):
    content: str = Field(min_length=3, max_length=2000)
    kind: str = Field(default="fatto", pattern=r"^(identita|preferenza|fatto|impegno)$")
    importance: float = Field(default=0.6, ge=0.0, le=1.0)


def _memoria_json(m: Memory) -> Dict[str, Any]:
    return {
        "id": str(m.id),
        "kind": m.kind,
        "content": m.content,
        "importance": round(m.importance, 3),
        "confidence": round(m.confidence, 3),
        "times_referenced": m.times_referenced,
        "first_seen_at": m.first_seen_at.isoformat(),
        "last_referenced_at": (
            m.last_referenced_at.isoformat() if m.last_referenced_at else None
        ),
        "valid_from": m.valid_from.isoformat() if m.valid_from else None,
        "valid_to": m.valid_to.isoformat() if m.valid_to else None,
        "superseded_by": str(m.superseded_by) if m.superseded_by else None,
        "expires_at": m.expires_at.isoformat() if m.expires_at else None,
        "viva": m.viva,
        "personality_id": str(m.personality_id) if m.personality_id else None,
    }


# --- l'utente sulle proprie memorie ---------------------------------------


@router.get("/memory")
async def le_mie_memorie(
    user: CurrentUser,
    session: DbSession,
    includi_superate: bool = Query(
        default=False,
        description="Anche ciò che non vale più, per vedere cosa è cambiato",
    ),
) -> Dict[str, Any]:
    store = MemoryStore(session)
    memorie = await store.per_utente(
        user.id, includi_superate=includi_superate,
    )
    return {
        "memorie": [_memoria_json(m) for m in memorie],
        "conteggi": await store.conta(user.id),
    }


@router.get("/memory/{memory_id}/history")
async def storia(
    memory_id: uuid.UUID, user: CurrentUser, session: DbSession,
) -> List[Dict[str, Any]]:
    """Cosa questa memoria ha sostituito.

    È la risposta a «da quando?»: senza, una memoria corretta sembra sempre
    essere stata così.
    """
    store = MemoryStore(session)
    if await store.una(user.id, memory_id) is None:
        raise HTTPException(status_code=404, detail="Memoria non trovata")

    return [_memoria_json(m) for m in await store.storia_di(user.id, memory_id)]


@router.post("/memory", status_code=status.HTTP_201_CREATED)
async def aggiungi(
    payload: AggiungiMemoria,
    user: CurrentUser,
    session: DbSession,
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> Dict[str, Any]:
    """Aggiunge una memoria a mano.

    Confidenza alta: l'ha scritta la persona a cui appartiene, e non è una
    deduzione di un modello.
    """
    gestore = GestoreAbbonamenti(session)
    limite = (
        (await gestore.diritti_di(user.id)).limite("memorie")
        if await gestore.tariffe_in_vigore() else None
    )
    try:
        memoria = await MemoryStore(session, embedder).ricorda(
            user_id=user.id,
            content=payload.content,
            kind=payload.kind,
            importance=payload.importance,
            confidence=1.0,
            meta={"origine": "utente"},
            limite=limite,
        )
    except MemoriaPiena as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Hai raggiunto il limite di {exc.limite} memorie del tuo "
                f"piano, e le memorie presenti sono tutte più importanti di "
                f"questa. Cancellane una per fare posto."
            ),
        ) from exc
    await session.commit()
    return _memoria_json(memoria)


@router.delete("/memory/{memory_id}")
async def dimentica(
    memory_id: uuid.UUID, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    """Cancella davvero. Non è una preferenza: è un diritto."""
    if not await MemoryStore(session).dimentica(user.id, memory_id):
        raise HTTPException(status_code=404, detail="Memoria non trovata")
    await session.commit()
    return {"dimenticata": True}


@router.delete("/memory")
async def dimentica_tutto(
    user: CurrentUser,
    session: DbSession,
    conferma: bool = Query(
        default=False,
        description="Obbligatorio: cancella tutto e non si torna indietro",
    ),
) -> Dict[str, Any]:
    if not conferma:
        # Un `DELETE` senza corpo è facile da mandare per sbaglio — un link
        # visitato, una richiesta ripetuta — e questa è l'unica operazione
        # della memoria che non si può annullare.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aggiungi ?conferma=true: questa operazione non si annulla.",
        )

    quante = await MemoryStore(session).dimentica_tutto(user.id)
    await session.commit()
    return {"dimenticate": quante}


@router.get("/memory/accesses")
async def chi_ha_guardato(
    user: CurrentUser, session: DbSession,
) -> List[Dict[str, Any]]:
    """«Chi ha letto le mie memorie?» — con una risposta."""
    return [
        {
            "action": a.action,
            "actor_id": a.actor_id,
            "count": a.count,
            "reason": a.reason,
            "created_at": a.created_at.isoformat(),
        }
        for a in await MemoryStore(session).accessi_a(user.id)
    ]


# --- l'amministratore sulle memorie altrui --------------------------------


@admin_router.get("/users/{user_id}/memories")
async def memorie_di(
    user_id: int,
    user: CurrentUser,
    session: DbSession,
    motivo: str = Query(
        default="",
        max_length=300,
        description="Perché stai guardando: finisce nel registro che l'utente legge",
    ),
    includi_superate: bool = False,
) -> Dict[str, Any]:
    """Le memorie di un utente. **Ogni lettura lascia una traccia.**

    Il motivo è facoltativo per non bloccare un'urgenza, ma la sua assenza si
    vede nel registro: l'utente legge «nessun motivo indicato», ed è
    un'informazione.
    """
    store = MemoryStore(session)
    memorie = await store.per_utente(user_id, includi_superate=includi_superate)

    await store.registra_accesso(
        subject_id=user_id,
        actor_id=user.id,
        action="memorie.lette",
        count=len(memorie),
        reason=motivo or None,
        correlation_id=current_correlation_id(),
    )
    await session.commit()

    logger.info(
        "Amministratore %s ha letto %d memorie dell'utente %s (%s)",
        user.id, len(memorie), user_id, motivo or "nessun motivo indicato",
    )

    return {
        "memorie": [_memoria_json(m) for m in memorie],
        "conteggi": await store.conta(user_id),
        "avviso": "Questo accesso è stato registrato e l'utente può vederlo.",
    }


@admin_router.post("/users/{user_id}/memories/consolidate")
async def consolida(
    user_id: int, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    """Esegue il consolidamento adesso, invece di aspettare il worker."""
    store = MemoryStore(session)
    esito = await Consolidatore(session).consolida(user_id)

    await store.registra_accesso(
        subject_id=user_id,
        actor_id=user.id,
        action="memorie.consolidate",
        count=esito.esaminate,
        correlation_id=current_correlation_id(),
    )
    await session.commit()
    return esito.to_dict()


@admin_router.get("/users/{user_id}/memories/accesses")
async def accessi_a(
    user_id: int, session: DbSession,
) -> List[Dict[str, Any]]:
    return [
        {
            "action": a.action, "actor_id": a.actor_id, "count": a.count,
            "reason": a.reason, "created_at": a.created_at.isoformat(),
        }
        for a in await MemoryStore(session).accessi_a(user_id)
    ]
