"""Il pagamento: predisposto per un fornitore vero, oggi simulato.

**Non sappiamo ancora chi incasserà** — Stripe, Barion, SimplePay, altri. Il
flusso però è lo stesso per tutti, ed è quello che si predispone qui:

1. l'utente sceglie un piano e la piattaforma apre una **sessione di pagamento**;
2. il fornitore ospita la pagina dove si paga, e l'utente ci viene mandato;
3. a pagamento avvenuto il fornitore manda un **evento firmato**;
4. solo allora l'abbonamento nasce e i crediti del periodo vengono accreditati.

Un fornitore vero è un'implementazione di `ProviderPagamenti`: dove sta la
sua pagina, come firma i suoi eventi, cosa dice un rinnovo. Il resto — la
sessione, l'idempotenza, l'attivazione — non cambia.

**Il simulatore è un fornitore a tutti gli effetti.** Firma i suoi eventi con
HMAC e un timestamp come fanno quelli veri, e l'evento passa dalla stessa
verifica. Un simulatore che attivasse l'abbonamento direttamente proverebbe
un percorso che in produzione non esiste; questo prova quello vero, meno la
carta. **In produzione si rifiuta di partire**: altrimenti chiunque potrebbe
«pagare» il piano più caro con un clic.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.billing_models import PaymentCheckout, PaymentEvent, Plan, Subscription
from ..settings import Settings, get_settings

logger = logging.getLogger(__name__)

#: Quanto può essere vecchio un evento firmato. Oltre, si rifiuta: è la
#: difesa contro chi intercetta un evento valido e lo rimanda più tardi.
TOLLERANZA_SECONDI = 300

#: Gli eventi che la piattaforma sa gestire.
PAGATO = "checkout.pagato"
ANNULLATO = "checkout.annullato"
RINNOVO_FALLITO = "abbonamento.pagamento_fallito"
ABBONAMENTO_CHIUSO = "abbonamento.annullato"


class FirmaNonValida(Exception):
    """L'evento non viene dal fornitore, o è stato alterato, o è vecchio."""


@dataclass
class EventoPagamento:
    """Un evento del fornitore, già verificato."""

    id: str
    tipo: str
    dati: Dict[str, Any] = field(default_factory=dict)


class ProviderPagamenti(Protocol):
    """Ciò che un fornitore di pagamento deve saper fare."""

    name: str

    def url_di_pagamento(self, checkout: PaymentCheckout, *, ritorno: str) -> str:
        """Dove mandare l'utente a pagare."""
        ...

    def verifica(self, corpo: bytes, intestazioni: Mapping[str, str]) -> EventoPagamento:
        """Verifica la firma e interpreta l'evento. Solleva `FirmaNonValida`."""
        ...

    # Il rinnovo e la disdetta, come li vuole `GestoreAbbonamenti`.
    async def crea_abbonamento(self, user_id: int, piano: Plan, *, annuale: bool = False) -> str: ...
    async def disdici(self, external_id: str) -> None: ...
    async def e_pagato(self, external_id: str) -> bool: ...


class PagamentiSimulati:
    """Un fornitore che non incassa, e che per il resto si comporta come uno vero.

    La pagina di pagamento è servita dall'API stessa (`/billing/mock/...`) e
    manda i suoi eventi firmati allo stesso webhook dei fornitori veri. Per
    provare la sospensione di un abbonamento, `rinnovo_fallisce` fa rispondere
    «non pagato» ai rinnovi.
    """

    name = "mock"

    def __init__(self, settings: Optional[Settings] = None, *, rinnovo_fallisce: bool = False) -> None:
        settings = settings or get_settings()
        self._segreto = settings.billing_webhook_secret.encode("utf-8")
        self._api = settings.api_public_url.rstrip("/")
        self.rinnovo_fallisce = rinnovo_fallisce or settings.billing_mock_renewal_fails

    # -- pagina di pagamento -----------------------------------------------

    def url_di_pagamento(self, checkout: PaymentCheckout, *, ritorno: str) -> str:
        from urllib.parse import quote

        return f"{self._api}/billing/mock/checkout/{checkout.id}?ritorno={quote(ritorno, safe='')}"

    # -- firma -------------------------------------------------------------

    def firma(self, corpo: bytes, *, istante: Optional[int] = None) -> str:
        """L'intestazione di firma, nel formato dei fornitori veri.

        `t=<timestamp>,v1=<hmac>`: il timestamp entra nel messaggio firmato,
        così che un evento non possa essere rimandato ringiovanito.
        """
        t = istante if istante is not None else int(time.time())
        firmato = f"{t}.".encode("utf-8") + corpo
        mac = hmac.new(self._segreto, firmato, hashlib.sha256).hexdigest()
        return f"t={t},v1={mac}"

    def verifica(self, corpo: bytes, intestazioni: Mapping[str, str]) -> EventoPagamento:
        intestazione = _cerca(intestazioni, "x-mock-signature")
        if not intestazione:
            raise FirmaNonValida("firma assente")

        parti = dict(p.split("=", 1) for p in intestazione.split(",") if "=" in p)
        try:
            t = int(parti["t"])
            ricevuta = parti["v1"]
        except (KeyError, ValueError) as exc:
            raise FirmaNonValida("firma malformata") from exc

        if abs(time.time() - t) > TOLLERANZA_SECONDI:
            raise FirmaNonValida("evento troppo vecchio: possibile ripetizione")

        attesa = self.firma(corpo, istante=t).split("v1=", 1)[1]
        # Confronto a tempo costante: un confronto normale si ferma al primo
        # carattere diverso, e il tempo di risposta rivelerebbe quanti ne
        # sono giusti a chi prova firme a tentativi.
        if not hmac.compare_digest(attesa, ricevuta):
            raise FirmaNonValida("firma non valida")

        try:
            evento = json.loads(corpo)
            return EventoPagamento(id=str(evento["id"]), tipo=str(evento["tipo"]), dati=evento.get("dati") or {})
        except (ValueError, KeyError) as exc:
            raise FirmaNonValida("corpo dell'evento illeggibile") from exc

    def evento(self, tipo: str, dati: Dict[str, Any]) -> bytes:
        """Il corpo di un evento, come lo emetterebbe il fornitore."""
        return json.dumps(
            {"id": f"evt_{uuid.uuid4().hex}", "tipo": tipo, "dati": dati},
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")

    # -- rinnovi ------------------------------------------------------------

    async def crea_abbonamento(self, user_id: int, piano: Plan, *, annuale: bool = False) -> str:
        return f"mock_sub_{uuid.uuid4().hex[:16]}"

    async def disdici(self, external_id: str) -> None:
        return None

    async def e_pagato(self, external_id: str) -> bool:
        return not self.rinnovo_fallisce


def _cerca(intestazioni: Mapping[str, str], nome: str) -> str:
    for k, v in intestazioni.items():
        if k.lower() == nome:
            return v
    return ""


class PagamentiNonDisponibili(RuntimeError):
    """Nessun fornitore di pagamento configurato."""


class PagamentiDisattivati:
    """Nessun fornitore ancora: i piani gratuiti funzionano, quelli a pagamento aspettano.

    È il modo di andare in produzione prima di aver scelto con chi incassare.
    Il simulatore lì non si può usare — regalerebbe i piani a chiunque apra la
    sua pagina — e senza un fornitore il servizio non partirebbe affatto.
    Qui un pagamento si rifiuta con un motivo che si può mostrare, e nessun
    evento viene accettato: non c'è nessuno autorizzato a mandarne.
    """

    name = "disattivato"

    def url_di_pagamento(self, checkout: PaymentCheckout, *, ritorno: str) -> str:
        raise PagamentiNonDisponibili(
            "I pagamenti non sono ancora attivi: i piani a pagamento saranno "
            "disponibili a breve."
        )

    def verifica(self, corpo: bytes, intestazioni: Mapping[str, str]) -> EventoPagamento:
        raise FirmaNonValida("nessun fornitore di pagamento configurato")

    async def crea_abbonamento(self, user_id: int, piano: Plan, *, annuale: bool = False) -> str:
        return f"gratuito-{uuid.uuid4().hex[:12]}"

    async def disdici(self, external_id: str) -> None:
        return None

    async def e_pagato(self, external_id: str) -> bool:
        # Solo ciò che non costava nulla si rinnova: un abbonamento pagato
        # rimasto da un fornitore precedente non ha più chi lo incassi.
        return external_id.startswith("gratuito-")


def provider_pagamenti(settings: Optional[Settings] = None) -> ProviderPagamenti:
    """Il fornitore configurato: il simulatore, o nessuno.

    Un nome sconosciuto è un errore all'avvio e non un ripiego silenzioso sul
    simulatore: in produzione il ripiego significherebbe regalare i piani.
    """
    settings = settings or get_settings()
    if settings.billing_provider == "mock":
        return PagamentiSimulati(settings)
    if settings.billing_provider == "disattivato":
        return PagamentiDisattivati()
    raise ValueError(
        f"fornitore di pagamento sconosciuto: «{settings.billing_provider}». "
        f"Oggi sono disponibili `mock` e `disattivato`; un fornitore vero si "
        f"aggiunge implementando ProviderPagamenti."
    )


# ---------------------------------------------------------------------------


class GestorePagamenti:
    """Apre le sessioni e applica gli eventi, una volta sola ciascuno."""

    def __init__(self, session: AsyncSession, provider: Optional[ProviderPagamenti] = None) -> None:
        self._session = session
        self._provider = provider or provider_pagamenti()

    @property
    def provider(self) -> ProviderPagamenti:
        return self._provider

    async def apri(
        self, user_id: int, piano: Plan, *, annuale: bool, ritorno: str,
    ) -> tuple[PaymentCheckout, str]:
        checkout = PaymentCheckout(
            user_id=user_id,
            plan_id=piano.id,
            annuale=annuale,
            status="aperto",
            provider=self._provider.name,
            importo=piano.price_yearly if annuale else piano.price_monthly,
            currency=piano.currency,
        )
        self._session.add(checkout)
        await self._session.flush()
        checkout.external_id = f"{self._provider.name}_cs_{checkout.id.hex[:16]}"
        return checkout, self._provider.url_di_pagamento(checkout, ritorno=ritorno)

    async def applica(self, evento: EventoPagamento) -> str:
        """Applica un evento verificato. Restituisce cosa è successo.

        Idempotente per identificativo: la riga dell'evento si scrive nella
        stessa transazione dell'effetto, e un secondo invio dello stesso
        evento trova la riga già presente e non fa nulla. I fornitori
        rimandano gli eventi finché non ricevono un 200 — senza, un pagamento
        confermato due volte accrediterebbe due periodi.
        """
        if await self._session.get(PaymentEvent, evento.id) is not None:
            return "già applicato"

        self._session.add(PaymentEvent(
            id=evento.id, provider=self._provider.name, tipo=evento.tipo,
            payload=evento.dati, received_at=utcnow(),
        ))
        try:
            await self._session.flush()
        except IntegrityError:
            # Due repliche che ricevono lo stesso evento nello stesso istante:
            # la seconda perde la corsa sulla chiave primaria, e va bene così.
            await self._session.rollback()
            return "già applicato"

        if evento.tipo == PAGATO:
            return await self._pagato(evento)
        if evento.tipo == ANNULLATO:
            return await self._chiudi_checkout(evento, "annullato")
        if evento.tipo == RINNOVO_FALLITO:
            return await self._stato_abbonamento(evento, "sospeso")
        if evento.tipo == ABBONAMENTO_CHIUSO:
            return await self._stato_abbonamento(evento, "disdetto")

        logger.info("Evento di pagamento ignorato: %s", evento.tipo)
        return "ignorato"

    async def _checkout(self, evento: EventoPagamento) -> Optional[PaymentCheckout]:
        try:
            checkout_id = uuid.UUID(str(evento.dati.get("checkout_id")))
        except ValueError:
            return None
        return await self._session.get(PaymentCheckout, checkout_id)

    async def _pagato(self, evento: EventoPagamento) -> str:
        from .plans import GestoreAbbonamenti

        checkout = await self._checkout(evento)
        if checkout is None:
            return "checkout sconosciuto"
        if checkout.status != "aperto":
            # Pagato dopo essere stato annullato o scaduto: è un caso che il
            # fornitore può produrre, e va visto da una persona, non risolto
            # attivando in silenzio.
            logger.warning("Pagamento per un checkout %s: %s", checkout.status, checkout.id)
            return f"checkout {checkout.status}"

        checkout.status = "pagato"
        checkout.completed_at = utcnow()
        checkout.subscription_external_id = str(
            evento.dati.get("subscription_id") or f"{self._provider.name}_sub_{checkout.id.hex[:16]}"
        )

        await GestoreAbbonamenti(self._session, self._provider).sottoscrivi(
            checkout.user_id, checkout.plan, annuale=checkout.annuale,
            external_id=checkout.subscription_external_id,
        )
        return "abbonamento attivato"

    async def _chiudi_checkout(self, evento: EventoPagamento, stato: str) -> str:
        checkout = await self._checkout(evento)
        if checkout is None:
            return "checkout sconosciuto"
        if checkout.status == "aperto":
            checkout.status = stato
            checkout.completed_at = utcnow()
        return f"checkout {checkout.status}"

    async def _stato_abbonamento(self, evento: EventoPagamento, stato: str) -> str:
        external_id = str(evento.dati.get("subscription_id") or "")
        abbonamento = (await self._session.execute(
            select(Subscription).where(Subscription.external_id == external_id)
        )).scalar_one_or_none()
        if abbonamento is None:
            return "abbonamento sconosciuto"
        abbonamento.status = stato
        return f"abbonamento {stato}"
