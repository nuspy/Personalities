"""Il pagamento predisposto, provato col simulatore.

Il simulatore firma i suoi eventi come un fornitore vero, e gli eventi passano
dalla stessa verifica: queste prove esercitano quindi il percorso di
produzione — sessione, pagina, evento firmato, attivazione — meno la carta.
"""
from __future__ import annotations

import time

import pytest
from sqlalchemy import select

from platform_core.billing.credits import RegistroCrediti
from platform_core.billing.pagamenti import (
    ANNULLATO, PAGATO, RINNOVO_FALLITO, FirmaNonValida, GestorePagamenti,
    PagamentiSimulati, TOLLERANZA_SECONDI,
)
from platform_core.billing.plans import GestoreAbbonamenti
from platform_core.domain.billing_models import PaymentCheckout, Subscription
from platform_core.settings import Settings

from .conftest import richiede_database
from .test_billing_api import abbona, piani  # noqa: F401
from .test_chat_personalita import client, embedder  # noqa: F401

pytestmark = richiede_database


def simulatore() -> PagamentiSimulati:
    return PagamentiSimulati(Settings())


class TestFirma:
    def test_una_firma_valida_passa(self):
        s = simulatore()
        corpo = s.evento(PAGATO, {"checkout_id": "x"})
        evento = s.verifica(corpo, {"X-Mock-Signature": s.firma(corpo)})
        assert evento.tipo == PAGATO

    def test_un_corpo_alterato_viene_rifiutato(self):
        s = simulatore()
        corpo = s.evento(PAGATO, {"checkout_id": "x", "importo": 900})
        firma = s.firma(corpo)
        alterato = corpo.replace(b"900", b"1")

        with pytest.raises(FirmaNonValida, match="non valida"):
            s.verifica(alterato, {"X-Mock-Signature": firma})

    def test_un_evento_vecchio_viene_rifiutato(self):
        """La difesa contro chi intercetta un evento valido e lo rimanda."""
        s = simulatore()
        corpo = s.evento(PAGATO, {})
        vecchia = s.firma(corpo, istante=int(time.time()) - TOLLERANZA_SECONDI - 10)

        with pytest.raises(FirmaNonValida, match="vecchio"):
            s.verifica(corpo, {"X-Mock-Signature": vecchia})

    def test_senza_firma_non_si_legge_nulla(self):
        with pytest.raises(FirmaNonValida, match="assente"):
            simulatore().verifica(b"{}", {})

    def test_un_segreto_diverso_non_firma_eventi_validi(self):
        buono = simulatore()
        falso = PagamentiSimulati(Settings(billing_webhook_secret="altro"))
        corpo = falso.evento(PAGATO, {})

        with pytest.raises(FirmaNonValida):
            buono.verifica(corpo, {"X-Mock-Signature": falso.firma(corpo)})


class TestFlusso:
    async def test_aprire_il_pagamento_non_attiva_niente(
        self, client, piani, session, utente,  # noqa: F811
    ):
        """Chi apre la pagina e poi la chiude non ha comprato niente."""
        risposta = await client.post("/me/checkout", json={"piano": "gold"})

        assert risposta.status_code == 201
        assert "/billing/mock/checkout/" in risposta.json()["url"]
        assert risposta.json()["importo"] == 2400
        assert await GestoreAbbonamenti(session).abbonamento_di(utente.id) is None
        assert await RegistroCrediti(session).saldo(utente.id) == 0

    async def test_pagare_attiva_il_piano_e_accredita_il_periodo(
        self, client, piani, session, utente,  # noqa: F811
    ):
        checkout = (await client.post("/me/checkout", json={"piano": "gold"})).json()

        pagina = await client.get(f"/billing/mock/checkout/{checkout['checkout_id']}")
        assert pagina.status_code == 200
        assert "24.00 EUR" in pagina.text

        esito = await client.post(
            f"/billing/mock/checkout/{checkout['checkout_id']}/esito",
            data={"esito": "paga", "ritorno": "http://localhost:3000/piano"},
        )
        assert esito.status_code == 303
        assert esito.headers["location"].startswith("http://localhost:3000/piano?checkout=")

        abbonamento = await GestoreAbbonamenti(session).abbonamento_di(utente.id)
        assert abbonamento.plan.slug == "gold"
        assert abbonamento.external_id.startswith("mock_sub_")
        assert await RegistroCrediti(session).saldo(utente.id) == 2000

        stato = (await client.get(f"/me/checkout/{checkout['checkout_id']}")).json()
        assert stato["stato"] == "pagato"

    async def test_annullare_non_attiva_niente(
        self, client, piani, session, utente,  # noqa: F811
    ):
        checkout = (await client.post("/me/checkout", json={"piano": "gold"})).json()

        await client.post(
            f"/billing/mock/checkout/{checkout['checkout_id']}/esito",
            data={"esito": "annulla", "ritorno": "http://localhost:3000/piano"},
        )

        assert await GestoreAbbonamenti(session).abbonamento_di(utente.id) is None
        stato = (await client.get(f"/me/checkout/{checkout['checkout_id']}")).json()
        assert stato["stato"] == "annullato"

    async def test_un_checkout_annullato_poi_pagato_non_si_attiva_da_solo(
        self, client, piani, session, utente,  # noqa: F811
    ):
        """Il fornitore può produrre questo caso: va visto da una persona."""
        checkout = (await client.post("/me/checkout", json={"piano": "gold"})).json()
        s = simulatore()
        gestore = GestorePagamenti(session, s)

        for tipo in (ANNULLATO, PAGATO):
            corpo = s.evento(tipo, {"checkout_id": checkout["checkout_id"]})
            await gestore.applica(s.verifica(corpo, {"X-Mock-Signature": s.firma(corpo)}))

        assert await GestoreAbbonamenti(session).abbonamento_di(utente.id) is None

    async def test_il_piano_gratuito_non_passa_dal_pagamento(self, client, piani):  # noqa: F811
        risposta = await client.post("/me/checkout", json={"piano": "free"})
        assert risposta.status_code == 400


class TestCambioDiPiano:
    async def test_il_piano_lasciato_viene_disdetto_anche_dal_fornitore(
        self, piani, session, utente,  # noqa: F811
    ):
        """Chiuderlo solo da noi lasciava il fornitore libero di rinnovarlo:
        chi passava da un piano all'altro li avrebbe pagati entrambi."""
        disdetti: list = []

        class Fornitore(PagamentiSimulati):
            async def disdici(self, external_id: str) -> None:
                disdetti.append(external_id)

        fornitore = Fornitore(Settings())
        gestore = GestoreAbbonamenti(session, fornitore)
        vecchio = await gestore.sottoscrivi(
            utente.id, await gestore.piano_per_slug("gold"), external_id="sub_vecchio",
        )
        await gestore.sottoscrivi(
            utente.id, await gestore.piano_per_slug("gold"), annuale=True,
            external_id="sub_nuovo",
        )

        assert disdetti == ["sub_vecchio"]
        assert vecchio.status == "disdetto"

    async def test_un_piano_gia_disdetto_non_si_disdice_due_volte(
        self, piani, session, utente,  # noqa: F811
    ):
        disdetti: list = []

        class Fornitore(PagamentiSimulati):
            async def disdici(self, external_id: str) -> None:
                disdetti.append(external_id)

        gestore = GestoreAbbonamenti(session, Fornitore(Settings()))
        vecchio = await gestore.sottoscrivi(
            utente.id, await gestore.piano_per_slug("gold"), external_id="sub_vecchio",
        )
        await gestore.disdici(vecchio)
        await gestore.sottoscrivi(
            utente.id, await gestore.piano_per_slug("free"), external_id="sub_free",
        )

        assert disdetti == ["sub_vecchio"]


class TestWebhook:
    async def test_lo_stesso_evento_due_volte_accredita_una_volta(
        self, client, piani, session, utente,  # noqa: F811
    ):
        """I fornitori rimandano finché non ricevono un 200."""
        checkout = (await client.post("/me/checkout", json={"piano": "gold"})).json()
        s = simulatore()
        corpo = s.evento(PAGATO, {"checkout_id": checkout["checkout_id"]})
        firma = s.firma(corpo)

        prima = await client.post("/billing/webhook", content=corpo, headers={"X-Mock-Signature": firma})
        seconda = await client.post("/billing/webhook", content=corpo, headers={"X-Mock-Signature": firma})

        assert prima.status_code == seconda.status_code == 200
        assert seconda.json()["esito"] == "già applicato"
        assert await RegistroCrediti(session).saldo(utente.id) == 2000

    async def test_una_firma_falsa_e_un_400(self, client, piani):  # noqa: F811
        s = simulatore()
        corpo = s.evento(PAGATO, {"checkout_id": "x"})

        risposta = await client.post(
            "/billing/webhook", content=corpo,
            headers={"X-Mock-Signature": f"t={int(time.time())},v1=deadbeef"},
        )

        assert risposta.status_code == 400

    async def test_un_rinnovo_non_pagato_sospende(
        self, client, piani, session, utente,  # noqa: F811
    ):
        abbonamento = await abbona(session, utente, "gold")
        s = simulatore()
        corpo = s.evento(RINNOVO_FALLITO, {"subscription_id": abbonamento.external_id})

        await client.post("/billing/webhook", content=corpo, headers={"X-Mock-Signature": s.firma(corpo)})

        await session.refresh(abbonamento)
        assert abbonamento.status == "sospeso", "sospeso e non disdetto: la carta può essere scaduta"

    async def test_il_checkout_di_un_altro_non_si_legge(
        self, client, piani, session, altro_utente,  # noqa: F811
    ):
        gold = await GestoreAbbonamenti(session).piano_per_slug("gold")
        suo, _ = await GestorePagamenti(session, simulatore()).apri(
            altro_utente.id, gold, annuale=False, ritorno="x",
        )
        await session.flush()

        assert (await client.get(f"/me/checkout/{suo.id}")).status_code == 404


class TestProduzione:
    def test_il_simulatore_e_rifiutato_in_produzione(self):
        """Altrimenti chiunque potrebbe «pagare» il piano più caro con un clic."""
        problemi = Settings(environment="prod", database_url="postgresql+psycopg://x@db/p").validate_production()
        assert any("simulato" in p for p in problemi)
        assert any("segreto" in p for p in problemi)

    def test_un_fornitore_sconosciuto_non_ripiega_sul_simulatore(self):
        from platform_core.billing.pagamenti import provider_pagamenti

        with pytest.raises(ValueError, match="sconosciuto"):
            provider_pagamenti(Settings(billing_provider="stripe"))
