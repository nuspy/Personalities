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

import html
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from ...auth.dependencies import CurrentUser, DbSession, require_role
from ...billing.credits import RegistroCrediti
from ...billing.pagamenti import (
    ANNULLATO, PAGATO, FirmaNonValida, GestorePagamenti, PagamentiSimulati,
    provider_pagamenti,
)
from ...billing.plans import GestoreAbbonamenti
from ...billing.quote import ContatoreQuote
from ...domain.billing_models import PaymentCheckout
from ...domain.models import User
from ...settings import get_settings
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
        # Quanto dei limiti è usato: mostrarlo prima evita che il limite si
        # scopra come un rifiuto. `limite` nullo significa illimitato, e senza
        # catalogo attivo l'installazione non limita affatto.
        "uso": (await ContatoreQuote(session).uso(user.id)).to_dict(diritti),
        "limiti_attivi": await gestore.tariffe_in_vigore(),
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

    prezzo = piano.price_yearly if payload.annuale else piano.price_monthly
    if prezzo > 0:
        # Il varco che c'era: questo endpoint attivava qualunque piano, anche
        # il più caro, senza passare da nessun pagamento. Un piano che costa
        # qualcosa nasce solo dall'evento firmato del fornitore.
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Il piano «{piano.slug}» è a pagamento: si attiva passando dal "
                f"pagamento (POST /me/checkout)."
            ),
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


@router.post("/me/checkout", status_code=status.HTTP_201_CREATED)
async def apri_checkout(
    payload: Sottoscrizione, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    """Apre una sessione di pagamento e dice dove andare a pagare.

    L'abbonamento non nasce qui: nasce quando il fornitore conferma il
    pagamento. Chi apre la pagina e poi la chiude non ha comprato niente.
    """
    gestore = GestoreAbbonamenti(session)
    piano = await gestore.piano_per_slug(payload.piano)
    if piano is None or not piano.active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Il piano «{payload.piano}» non esiste o non è più offerto.",
        )
    if (piano.price_yearly if payload.annuale else piano.price_monthly) <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"Il piano «{piano.slug}» è gratuito: non serve pagare.",
        )

    ritorno = f"{get_settings().web_public_url.rstrip('/')}/piano"
    checkout, url = await GestorePagamenti(session).apri(
        user.id, piano, annuale=payload.annuale, ritorno=ritorno,
    )
    await session.commit()
    return {"checkout_id": str(checkout.id), "url": url, "importo": checkout.importo, "valuta": checkout.currency}


@router.get("/me/checkout/{checkout_id}")
async def stato_checkout(
    checkout_id: uuid.UUID, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    """Com'è andata: la pagina di ritorno lo chiede invece di fidarsi dell'URL.

    L'indirizzo di ritorno lo può scrivere chiunque; lo stato della sessione
    no — è cambiato solo dall'evento firmato del fornitore.
    """
    checkout = await session.get(PaymentCheckout, checkout_id)
    if checkout is None or checkout.user_id != user.id:
        raise HTTPException(status_code=404, detail="Pagamento non trovato")
    return {"checkout_id": str(checkout.id), "stato": checkout.status, "piano": checkout.plan.slug}


@router.post("/billing/webhook")
async def webhook(request: Request, session: DbSession) -> Dict[str, Any]:
    """Gli eventi del fornitore di pagamento.

    Senza autenticazione dell'utente — chi chiama è il fornitore — e per
    questo con la firma verificata prima di leggere qualunque cosa. Risponde
    200 anche a un evento già applicato: il fornitore rimanda finché non lo
    riceve, e un errore qui lo farebbe rimandare per sempre.
    """
    corpo = await request.body()
    gestore = GestorePagamenti(session)
    try:
        evento = gestore.provider.verifica(corpo, request.headers)
    except FirmaNonValida as exc:
        logger.warning("Evento di pagamento rifiutato: %s", exc)
        raise HTTPException(status_code=400, detail=f"evento non valido: {exc}") from exc

    esito = await gestore.applica(evento)
    await session.commit()
    logger.info("Evento %s (%s): %s", evento.id, evento.tipo, esito)
    return {"ricevuto": evento.id, "esito": esito}


# ---- la pagina di pagamento del simulatore ---------------------------------


def _simulatore() -> PagamentiSimulati:
    provider = provider_pagamenti()
    if not isinstance(provider, PagamentiSimulati):
        raise HTTPException(status_code=404, detail="Non disponibile")
    return provider


@router.get("/billing/mock/checkout/{checkout_id}", response_class=HTMLResponse)
async def pagina_simulata(
    checkout_id: uuid.UUID, session: DbSession, ritorno: str = "",
) -> HTMLResponse:
    """La pagina che un fornitore vero ospiterebbe. Esiste solo col simulatore."""
    _simulatore()
    checkout = await session.get(PaymentCheckout, checkout_id)
    if checkout is None:
        raise HTTPException(status_code=404, detail="Pagamento non trovato")

    importo = f"{checkout.importo / 100:.2f} {checkout.currency}"
    periodo = "anno" if checkout.annuale else "mese"
    chiuso = checkout.status != "aperto"
    return HTMLResponse(_PAGINA.format(
        piano=html.escape(checkout.plan.name),
        importo=html.escape(importo),
        periodo=periodo,
        azione=f"/billing/mock/checkout/{checkout.id}/esito",
        ritorno=html.escape(ritorno, quote=True),
        stato=html.escape(checkout.status),
        disabilitato="disabled" if chiuso else "",
    ))


@router.post("/billing/mock/checkout/{checkout_id}/esito")
async def esito_simulato(
    checkout_id: uuid.UUID, request: Request, session: DbSession,
) -> RedirectResponse:
    """Il clic su «Paga» o «Annulla».

    Non attiva niente da sé: costruisce l'evento che il fornitore
    manderebbe, lo firma e lo fa passare dalla stessa verifica e dallo stesso
    gestore di un evento vero. È questo che rende il simulatore una prova del
    percorso di produzione invece di una scorciatoia.
    """
    simulatore = _simulatore()
    modulo = await request.form()
    scelta = str(modulo.get("esito", ""))
    ritorno = str(modulo.get("ritorno", "")) or get_settings().web_public_url

    checkout = await session.get(PaymentCheckout, checkout_id)
    if checkout is None:
        raise HTTPException(status_code=404, detail="Pagamento non trovato")

    tipo = PAGATO if scelta == "paga" else ANNULLATO
    corpo = simulatore.evento(tipo, {
        "checkout_id": str(checkout.id),
        "subscription_id": f"mock_sub_{checkout.id.hex[:16]}",
        "importo": checkout.importo,
    })
    gestore = GestorePagamenti(session, simulatore)
    evento = simulatore.verifica(corpo, {"X-Mock-Signature": simulatore.firma(corpo)})
    await gestore.applica(evento)
    await session.commit()

    separatore = "&" if "?" in ritorno else "?"
    return RedirectResponse(
        f"{ritorno}{separatore}checkout={checkout.id}", status_code=status.HTTP_303_SEE_OTHER,
    )


_PAGINA = """<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pagamento simulato</title>
<style>
  body {{ font: 16px/1.5 system-ui, sans-serif; background: #eceef1; color: #16212e;
         margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 16px; }}
  main {{ background: #fff; border-top: 4px solid #a02c16; max-width: 420px; width: 100%;
          padding: 28px; box-sizing: border-box; }}
  .avviso {{ font-size: 13px; color: #8f2617; background: #fbf1ee; padding: 8px 10px; margin: 0 0 20px; }}
  h1 {{ font: 400 28px Georgia, serif; margin: 0 0 4px; }}
  .importo {{ font: 400 36px Georgia, serif; margin: 16px 0 24px; }}
  form {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  button {{ min-height: 48px; padding: 0 18px; font-size: 15px; border: 1px solid #16212e;
            background: #16212e; color: #fff; cursor: pointer; flex: 1; }}
  button.annulla {{ background: transparent; color: #4a5866; border-color: #d3d8de; }}
  button:disabled {{ opacity: .45; cursor: not-allowed; }}
  button:focus-visible {{ outline: 3px solid #c4624d; outline-offset: 2px; }}
</style></head>
<body><main>
  <p class="avviso">Pagamento simulato: nessuna carta viene addebitata. In produzione questa
  pagina è ospitata dal fornitore di pagamento.</p>
  <h1>{piano}</h1>
  <div>per un {periodo}</div>
  <div class="importo">{importo}</div>
  <form method="post" action="{azione}">
    <input type="hidden" name="ritorno" value="{ritorno}">
    <button name="esito" value="paga" {disabilitato}>Paga</button>
    <button name="esito" value="annulla" class="annulla" {disabilitato}>Annulla</button>
  </form>
  <p style="font-size:13px;color:#5f6b76;margin-top:16px">Stato: {stato}</p>
</main></body></html>"""



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
