"""Piani, abbonamenti, crediti.

**Tre «no» diversi, e vanno detti in modo diverso.** Un utente a cui viene
negata una risposta deve sapere se gli manca il *diritto* (serve un altro
piano), la *moneta* (servono crediti) o se il servizio è rotto. Un unico 403
li confonderebbe tutti, e chi lo riceve non saprebbe cosa fare — che è il modo
in cui un limite commerciale diventa un guasto percepito.

| | codice | come si risolve |
|---|---|---|
| il piano non comprende quella voce | 403 | si cambia piano |
| il saldo non basta | 402 | si comprano crediti, o si aspetta il rinnovo |
| il provider non risponde | 503 | non dipende dall'utente |

**Il saldo non si scrive da qui.** Nessun endpoint accredita su richiesta
dell'utente: i crediti entrano col rinnovo del piano o con un acquisto
registrato dal provider, e una rettifica manuale è un'operazione
amministrativa con il suo motivo obbligatorio e il suo record di audit.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...auth.dependencies import CurrentUser, DbSession, require_role
from ...billing.credits import RegistroCrediti
from ...billing.plans import GestoreAbbonamenti
from ...domain.models import User
from ...domain.repositories import AuditRepository

logger = logging.getLogger(__name__)

router = APIRouter(tags=["abbonamenti"])

#: Le rotte amministrative stanno su un router a parte perché il ruolo si
#: dichiara una volta sola: ripeterlo endpoint per endpoint significa
#: dimenticarlo su quello aggiunto di fretta.
router_admin = APIRouter(
    prefix="/admin",
    tags=["abbonamenti"],
    dependencies=[Depends(require_role("admin"))],
)


class Sottoscrizione(BaseModel):
    piano: str = Field(min_length=1, max_length=40)
    annuale: bool = False


class Rettifica(BaseModel):
    delta: int = Field(description="positivo accredita, negativo toglie")
    #: Obbligatorio: una rettifica senza motivo è indistinguibile da un
    #: errore, e sei mesi dopo nessuno sa più perché quel saldo è cambiato.
    motivo: str = Field(min_length=3, max_length=500)


def _piano_json(p) -> Dict[str, Any]:
    return {
        "slug": p.slug,
        "nome": p.name,
        "prezzo_mensile": p.price_monthly,
        "prezzo_annuale": p.price_yearly,
        "crediti_per_periodo": p.credits_per_period,
        "limiti": p.limits or {},
        "diritti": p.entitlements or {},
    }


def _abbonamento_json(a) -> Optional[Dict[str, Any]]:
    if a is None:
        return None
    return {
        "id": str(a.id),
        "piano": a.plan.slug,
        "stato": a.status,
        "periodo_inizio": a.period_start.isoformat(),
        "periodo_fine": a.period_end.isoformat(),
        "disdetto_il": a.cancel_at.isoformat() if a.cancel_at else None,
    }


@router.get("/plans")
async def piani(session: DbSession) -> List[Dict[str, Any]]:
    """Il catalogo. Pubblico: si guarda prima di avere un account."""
    return [_piano_json(p) for p in await GestoreAbbonamenti(session).piani()]


@router.get("/me/billing")
async def il_mio_conto(user: CurrentUser, session: DbSession) -> Dict[str, Any]:
    """Piano, diritti e saldo in una chiamata.

    Insieme e non in tre endpoint perché è una schermata sola: chiederli
    separatamente significa mostrarne uno prima degli altri, e un saldo che
    compare prima del piano a cui appartiene si legge male.
    """
    gestore = GestoreAbbonamenti(session)
    abbonamento = await gestore.abbonamento_di(user.id)
    diritti = await gestore.diritti_di(user.id)

    return {
        "abbonamento": _abbonamento_json(abbonamento),
        "diritti": diritti.to_dict(),
        "saldo": await RegistroCrediti(session).saldo(user.id),
    }


@router.get("/me/credits")
async def i_miei_crediti(
    user: CurrentUser, session: DbSession, limite: int = 50,
) -> Dict[str, Any]:
    """Saldo e movimenti.

    I movimenti e non il solo saldo: il registro è append-only proprio perché
    chi vede sparire dei crediti possa vedere anche *quando* e *per cosa*. Un
    numero da solo non si può contestare.
    """
    registro = RegistroCrediti(session)
    return {
        "saldo": await registro.saldo(user.id),
        "movimenti": [
            m.to_dict() for m in await registro.movimenti(user.id, limite=limite)
        ],
    }


@router.post("/me/subscription", status_code=status.HTTP_201_CREATED)
async def sottoscrivi(
    payload: Sottoscrizione, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    """Apre un abbonamento, chiudendo il precedente."""
    gestore = GestoreAbbonamenti(session)
    piano = await gestore.piano_per_slug(payload.piano)
    if piano is None or not piano.active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Il piano «{payload.piano}» non esiste o non è più offerto.",
        )

    abbonamento = await gestore.sottoscrivi(
        user.id, piano, annuale=payload.annuale,
    )
    await session.commit()
    await session.refresh(abbonamento, ["plan"])

    return {
        "abbonamento": _abbonamento_json(abbonamento),
        "saldo": await RegistroCrediti(session).saldo(user.id),
    }


@router.post("/me/subscription/cancel")
async def disdici(user: CurrentUser, session: DbSession) -> Dict[str, Any]:
    """Disdice: l'abbonamento resta attivo fino a scadenza.

    Già pagato. Chiuderlo subito toglierebbe qualcosa a cui l'utente ha
    diritto, e la differenza fra «ho disdetto» e «non ho più accesso» è
    esattamente ciò che rende accettabile disdire.
    """
    gestore = GestoreAbbonamenti(session)
    abbonamento = await gestore.abbonamento_di(user.id)
    if abbonamento is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Non hai un abbonamento attivo.",
        )

    await gestore.disdici(abbonamento)
    await session.commit()
    await session.refresh(abbonamento, ["plan"])
    return {"abbonamento": _abbonamento_json(abbonamento)}


@router_admin.get("/users/{user_id}/credits")
async def crediti_di(
    user_id: int, session: DbSession, limite: int = 100,
) -> Dict[str, Any]:
    registro = RegistroCrediti(session)
    if await session.get(User, user_id) is None:
        raise HTTPException(status_code=404, detail="Utente non trovato")

    return {
        "saldo": await registro.saldo(user_id),
        "movimenti": [
            m.to_dict() for m in await registro.movimenti(user_id, limite=limite)
        ],
    }


@router_admin.post("/users/{user_id}/credits")
async def rettifica(
    user_id: int,
    payload: Rettifica,
    user: CurrentUser,
    session: DbSession,
) -> Dict[str, Any]:
    """Corregge un saldo, lasciando scritto perché.

    Una riga in più e non una modifica di quelle esistenti: entrambe restano,
    e la storia resta leggibile. Un saldo aggiustato cancellando ciò che non
    tornava è un saldo giusto che non sa spiegarsi.
    """
    if await session.get(User, user_id) is None:
        raise HTTPException(status_code=404, detail="Utente non trovato")

    registro = RegistroCrediti(session)
    prima = await registro.saldo(user_id)
    try:
        await registro.rettifica(user_id, payload.delta, note=payload.motivo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dopo = await registro.saldo(user_id)
    await AuditRepository(session).record(
        actor_id=user.id,
        action="credits.adjust",
        target_type="user",
        target_id=str(user_id),
        before={"saldo": prima},
        after={"saldo": dopo, "delta": payload.delta, "motivo": payload.motivo},
    )
    await session.commit()

    return {"saldo": dopo, "delta": payload.delta}
