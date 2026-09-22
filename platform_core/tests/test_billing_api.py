"""Abbonamenti e crediti, dal lato di chi li usa.

Il cuore è il comportamento di `/chat` quando qualcosa manca, perché è lì che
una scelta commerciale diventa un'esperienza:

- **il diritto manca** → 403, e si risolve cambiando piano;
- **la moneta manca** → 402, e si risolve aspettando il rinnovo o comprando;
- **la risposta non arriva** → i crediti tornano, con una riga propria.

Tenerli distinti non è pedanteria: un unico «non puoi» lascia chi lo riceve
senza sapere cosa fare, ed è così che un limite previsto si legge come un
guasto.
"""
from __future__ import annotations

import uuid
from typing import AsyncIterator, List

import pytest_asyncio
from sqlalchemy import select

from platform_core.api.deps import get_llm_provider
from platform_core.billing.credits import RegistroCrediti
from platform_core.billing.plans import GestoreAbbonamenti
from platform_core.domain.base import utcnow
from platform_core.domain.billing_models import CreditEntry, Plan
from platform_core.domain.knowledge_models import (
    CommercialCategory, Personality, PersonalityVersion,
)
from platform_core.llm.base import StreamChunk, Usage

from .conftest import richiede_database
from .test_chat_personalita import (  # noqa: F401
    client, embedder, eventi_di, installa, personalita,
)

pytestmark = richiede_database


class ModelloMuto:
    """Finisce senza aver prodotto testo: il caso da rimborsare."""

    name = "muto"

    async def stream(self, request) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(done=True, usage=Usage(total_tokens=3))


class ModelloBreve:
    name = "breve"

    async def stream(self, request) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="Rispondo.")
        yield StreamChunk(done=True, usage=Usage(total_tokens=7))


@pytest_asyncio.fixture
async def piani(session) -> List[Plan]:
    """Il catalogo minimo: un gratuito e un gold."""
    righe = [
        Plan(
            slug="free", name="Gratuito", rank=0, credits_per_period=50,
            entitlements={"categorie": ["free"]}, limits={},
        ),
        Plan(
            slug="gold", name="Gold", rank=2, price_monthly=2400,
            credits_per_period=2000,
            entitlements={"categorie": ["free", "gold"], "voce": True},
            limits={"messaggi_al_giorno": 1000},
        ),
    ]
    session.add_all(righe)
    await session.flush()
    return righe


@pytest_asyncio.fixture
async def categoria_gold(session) -> CommercialCategory:
    categoria = CommercialCategory(
        slug=f"gold-{uuid.uuid4().hex[:6]}", name="Gold", rank=2,
    )
    session.add(categoria)
    await session.flush()
    return categoria


async def abbona(session, utente, slug: str):
    """Un abbonamento pagato, preparato dal gestore.

    Non dall'endpoint: un piano a pagamento nasce solo dall'evento firmato
    del fornitore (vedi `test_pagamenti`), e qui serve soltanto averlo.
    """
    gestore = GestoreAbbonamenti(session)
    abbonamento = await gestore.sottoscrivi(utente.id, await gestore.piano_per_slug(slug))
    await session.flush()
    return abbonamento


async def _con_crediti(session, utente, quanti: int) -> None:
    if quanti:
        await RegistroCrediti(session).accredita(
            utente.id, quanti, note="apertura del test",
        )
    await session.flush()


class TestCatalogoEConto:
    async def test_i_piani_si_leggono_senza_abbonamento(self, client, piani):
        elenco = (await client.get("/plans")).json()

        slug = {p["slug"] for p in elenco}
        assert {"free", "gold"} <= slug
        gold = next(p for p in elenco if p["slug"] == "gold")
        assert gold["prezzo_mensile"] == 2400
        assert gold["crediti_per_periodo"] == 2000

    async def test_senza_abbonamento_si_ricade_sul_gratuito(
        self, client, piani, utente,
    ):
        """Non sul nulla: un servizio che non risponde a chi non paga non si
        può nemmeno provare, e la prima esperienza di chiunque comincia da
        qui."""
        conto = (await client.get("/me/billing")).json()

        assert conto["abbonamento"] is None
        assert conto["diritti"]["piano"] == "free"
        assert conto["diritti"]["predefiniti"] is True

    async def test_un_piano_a_pagamento_non_si_attiva_senza_pagare(
        self, client, piani, utente, session,
    ):
        """Il varco che c'era: l'endpoint attivava anche il piano più caro
        senza passare da nessun pagamento."""
        risposta = await client.post("/me/subscription", json={"piano": "gold"})

        assert risposta.status_code == 402
        assert "/me/checkout" in risposta.json()["detail"]
        assert await RegistroCrediti(session).saldo(utente.id) == 0

    async def test_il_piano_gratuito_si_attiva_direttamente(
        self, client, piani, utente,
    ):
        risposta = await client.post("/me/subscription", json={"piano": "free"})

        assert risposta.status_code == 201
        assert risposta.json()["abbonamento"]["piano"] == "free"
        assert risposta.json()["saldo"] == 50

    async def test_un_piano_inesistente_e_un_404(self, client, piani):
        risposta = await client.post("/me/subscription", json={"piano": "platino"})

        assert risposta.status_code == 404
        assert "platino" in risposta.json()["detail"]

    async def test_disdire_lascia_l_abbonamento_vivo_fino_a_scadenza(
        self, client, piani, utente, session,
    ):
        """È già stato pagato: chiuderlo subito toglierebbe qualcosa a cui
        l'utente ha diritto."""
        await abbona(session, utente, "gold")

        risposta = await client.post("/me/subscription/cancel")

        assert risposta.status_code == 200
        abbonamento = risposta.json()["abbonamento"]
        assert abbonamento["disdetto_il"] is not None
        assert abbonamento["stato"] in ("attivo", "in_prova")

    async def test_dal_pagato_al_gratuito_si_passa_a_fine_periodo(
        self, client, piani, utente, session,
    ):
        """Il periodo è già stato pagato: scegliere il gratuito disdice il
        piano pagato a scadenza, invece di toglierlo subito."""
        gold = await abbona(session, utente, "gold")
        await session.commit()

        risposta = await client.post("/me/subscription", json={"piano": "free"})

        assert risposta.status_code == 202
        corpo = risposta.json()
        assert corpo["abbonamento"]["piano"] == "gold"
        assert corpo["abbonamento"]["disdetto_il"] == corpo["passaggio"]["dal"]
        assert corpo["passaggio"]["piano"] == "free"
        await session.refresh(gold)
        assert gold.status == "attivo"
        attuale = await GestoreAbbonamenti(session).abbonamento_di(utente.id)
        assert attuale.id == gold.id

    async def test_i_movimenti_accompagnano_il_saldo(
        self, client, piani, utente, session,
    ):
        """Un numero da solo non si può contestare."""
        await abbona(session, utente, "gold")

        corpo = (await client.get("/me/credits")).json()

        assert corpo["saldo"] == 2000
        assert corpo["movimenti"]
        assert corpo["movimenti"][0]["delta"] == 2000
        assert corpo["movimenti"][0]["reason"] == "accredito_piano"


class TestDirittoEMoneta:
    async def test_senza_il_diritto_e_403_col_motivo(
        self, client, personalita, piani, categoria_gold, session, utente,
    ):
        """Il piano gratuito non comprende una voce Gold."""
        p, _, _ = personalita
        p.commercial_category_id = categoria_gold.id
        await session.flush()
        await _con_crediti(session, utente, 100)

        installa(client, ModelloBreve())
        risposta = await client.post(
            "/chat", json={"message": "Ciao", "personality": p.slug},
        )

        assert risposta.status_code == 403
        assert "gratuito" in risposta.json()["detail"].lower()

    async def test_col_piano_giusto_si_passa(
        self, client, personalita, piani, categoria_gold, session, utente,
    ):
        p, _, _ = personalita
        p.commercial_category_id = categoria_gold.id
        # La categoria commerciale della personalità e lo slug elencato nei
        # diritti del piano devono coincidere: sono la stessa chiave vista da
        # due parti, ed è il punto in cui un refuso si traduce in un utente
        # che paga e non entra.
        categoria_gold.slug = "gold"
        await session.flush()

        await abbona(session, utente, "gold")
        installa(client, ModelloBreve())

        risposta = await client.post(
            "/chat", json={"message": "Ciao", "personality": p.slug},
        )

        assert risposta.status_code == 200

    async def test_senza_crediti_e_402_non_403(
        self, client, personalita, piani, utente,
    ):
        """Codici diversi perché si risolvono in modi diversi: 403 si risolve
        cambiando piano, 402 aspettando il rinnovo o comprando."""
        p, _, _ = personalita

        installa(client, ModelloBreve())
        risposta = await client.post(
            "/chat", json={"message": "Ciao", "personality": p.slug},
        )

        assert risposta.status_code == 402
        # Concordato: «Servono 1 crediti» è il genere di dettaglio che fa
        # sembrare improvvisato tutto il resto.
        assert risposta.json()["detail"] == (
            "Serve 1 credito per questa risposta, ne hai 0."
        )

    async def test_una_risposta_costa_un_credito(
        self, client, personalita, piani, session, utente,
    ):
        p, _, _ = personalita
        await _con_crediti(session, utente, 10)

        installa(client, ModelloBreve())
        await client.post("/chat", json={"message": "Ciao", "personality": p.slug})

        assert await RegistroCrediti(session).saldo(utente.id) == 9

    async def test_il_costo_lo_decide_la_versione(
        self, client, personalita, piani, session, utente,
    ):
        """È la voce a scegliere il modello, quindi è la voce a decidere
        quanto costa: metterlo sul piano direbbe che la stessa personalità
        costa diversamente a due utenti."""
        p, versione, _ = personalita
        versione.llm_config = {"costo_crediti": 5}
        await session.flush()
        await _con_crediti(session, utente, 10)

        installa(client, ModelloBreve())
        await client.post("/chat", json={"message": "Ciao", "personality": p.slug})

        assert await RegistroCrediti(session).saldo(utente.id) == 5

    async def test_senza_personalita_non_si_paga(
        self, client, piani, session, utente,
    ):
        """La chat nuda è lo scheletro di sviluppo, non un prodotto: farla
        pagare significherebbe che una prova senza voce consuma il saldo di
        chi non ha ancora scelto nulla."""
        await _con_crediti(session, utente, 3)

        installa(client, ModelloBreve())
        risposta = await client.post("/chat", json={"message": "Ciao"})

        assert risposta.status_code == 200
        assert await RegistroCrediti(session).saldo(utente.id) == 3


class TestRimborso:
    async def test_una_risposta_non_prodotta_viene_restituita(
        self, client, personalita, piani, session, utente,
    ):
        p, _, _ = personalita
        await _con_crediti(session, utente, 10)

        installa(client, ModelloMuto())
        await client.post("/chat", json={"message": "Ciao", "personality": p.slug})

        assert await RegistroCrediti(session).saldo(utente.id) == 10

    async def test_il_rimborso_e_una_riga_in_piu_non_una_cancellazione(
        self, client, personalita, piani, session, utente,
    ):
        """Chi legge il registro deve vedere che una risposta è stata
        tentata, è fallita ed è stata restituita. Cancellando, il saldo
        tornerebbe giusto senza spiegare nulla."""
        p, _, _ = personalita
        await _con_crediti(session, utente, 10)

        installa(client, ModelloMuto())
        await client.post("/chat", json={"message": "Ciao", "personality": p.slug})

        motivi = [
            r.reason for r in (await session.execute(
                select(CreditEntry)
                .where(CreditEntry.user_id == utente.id)
                .order_by(CreditEntry.id)
            )).scalars()
        ]
        assert motivi == ["accredito_piano", "consumo", "rimborso"], motivi


class TestPianoAllaNascita:
    """Un account nuovo deve poter parlare subito.

    I diritti e i crediti ricadono in modi diversi: senza abbonamento i primi
    tornano comunque quelli del piano gratuito — scritti nel codice come
    ultima rete — mentre un saldo è una somma di righe, e righe non ce ne
    sono. Il risultato era un utente che vedeva le voci gratuite e riceveva
    402 al primo messaggio.
    """

    async def test_il_primo_accesso_apre_il_piano_gratuito(
        self, session, piani, principal_altro,
    ):
        from platform_core.auth.dependencies import current_user

        utente = await current_user(principal_altro, session)

        gestore = GestoreAbbonamenti(session)
        abbonamento = await gestore.abbonamento_di(utente.id)
        assert abbonamento is not None
        assert abbonamento.plan.slug == "free"
        assert await RegistroCrediti(session).saldo(utente.id) == 50

    async def test_gli_accessi_successivi_non_ne_aprono_altri(
        self, session, piani, principal_altro,
    ):
        """Due richieste nello stesso secondo non devono accreditare due
        volte il periodo iniziale."""
        from platform_core.auth.dependencies import current_user

        utente = await current_user(principal_altro, session)
        for _ in range(3):
            await current_user(principal_altro, session)

        assert await RegistroCrediti(session).saldo(utente.id) == 50

    async def test_senza_catalogo_l_accesso_funziona_lo_stesso(
        self, session, principal_altro,
    ):
        """Nessun piano in tabella: si resta sui diritti base e senza crediti,
        invece di rifiutare l'accesso. Inventare un piano da qui vorrebbe dire
        che il catalogo si crea da solo."""
        from platform_core.auth.dependencies import current_user

        utente = await current_user(principal_altro, session)

        assert utente.id is not None
        assert await GestoreAbbonamenti(session).abbonamento_di(utente.id) is None


class TestSenzaCatalogo:
    """Un'installazione senza piani non fa pagare, e lo dice.

    Non è indulgenza: addebitare con la tabella vuota renderebbe muta ogni
    personalità di un'installazione appena migrata, e «tabella `plans`
    vuota» non somiglia per niente a «nessuno riesce a parlare con
    nessuno». Fra un servizio che non incassa e uno che non parte, si sceglie
    quello che resta usabile — e si scrive un avviso nei log.
    """

    async def test_senza_piani_la_risposta_non_costa(
        self, client, personalita, session, utente,
    ):
        p, _, _ = personalita
        assert not await GestoreAbbonamenti(session).tariffe_in_vigore()

        installa(client, ModelloBreve())
        risposta = await client.post(
            "/chat", json={"message": "Ciao", "personality": p.slug},
        )

        assert risposta.status_code == 200
        assert await RegistroCrediti(session).saldo(utente.id) == 0

    async def test_un_catalogo_di_soli_piani_ritirati_non_e_un_catalogo(
        self, client, personalita, session, utente,
    ):
        """`active` dice se un piano si può ancora sottoscrivere: uno ritirato
        resta in tabella per gli abbonati che ce l'hanno, e non basta a dire
        che l'installazione vende qualcosa."""
        session.add(Plan(slug="vecchio", name="Ritirato", rank=9, active=False))
        await session.flush()

        assert not await GestoreAbbonamenti(session).tariffe_in_vigore()

    async def test_con_un_piano_attivo_le_tariffe_valgono(
        self, client, personalita, piani, session, utente,
    ):
        p, _, _ = personalita
        assert await GestoreAbbonamenti(session).tariffe_in_vigore()

        installa(client, ModelloBreve())
        risposta = await client.post(
            "/chat", json={"message": "Ciao", "personality": p.slug},
        )

        assert risposta.status_code == 402
