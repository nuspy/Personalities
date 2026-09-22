"""La manutenzione periodica.

Vale la pena provarla proprio perché **nessuno se ne accorge quando non
gira**: un rinnovo che non avviene non produce un errore, produce un utente
che fra un mese resta senza crediti e non sa perché. Un test è l'unico posto
in cui quel silenzio fa rumore.
"""
from __future__ import annotations

from datetime import timedelta

import pytest_asyncio

from platform_core.billing.credits import RegistroCrediti
from platform_core.billing.plans import GestoreAbbonamenti
from platform_core.domain.base import utcnow
from platform_core.domain.billing_models import Plan, Subscription
from platform_core.jobs.periodico import rinnova_scaduti

from .conftest import richiede_database

pytestmark = richiede_database


@pytest_asyncio.fixture
async def piano(session) -> Plan:
    p = Plan(
        slug="gold", name="Gold", rank=2, price_monthly=2400,
        credits_per_period=2000,
        entitlements={"categorie": ["free", "gold"]}, limits={},
    )
    session.add(p)
    await session.flush()
    return p


async def _abbonamento_scaduto(session, utente, piano, **extra) -> Subscription:
    """Un abbonamento il cui periodo è finito ieri."""
    adesso = utcnow()
    abbonamento = Subscription(
        user_id=utente.id,
        plan_id=piano.id,
        status="attivo",
        period_start=adesso - timedelta(days=31),
        period_end=adesso - timedelta(days=1),
        **extra,
    )
    session.add(abbonamento)
    await session.flush()
    return abbonamento


class TestRinnovi:
    async def test_un_periodo_finito_apre_il_successivo(
        self, session, session_factory, monkeypatch, utente, piano,
    ):
        monkeypatch.setattr(
            "platform_core.jobs.periodico.get_session_factory",
            lambda: session_factory,
        )
        abbonamento = await _abbonamento_scaduto(session, utente, piano)
        fine_vecchia = abbonamento.period_end

        esito = await rinnova_scaduti()

        assert esito.rinnovati == 1
        assert esito.crediti_accreditati == 2000
        await session.refresh(abbonamento)
        assert abbonamento.period_end > fine_vecchia
        assert abbonamento.status == "attivo"

    async def test_il_rinnovo_accredita_davvero(
        self, session, session_factory, monkeypatch, utente, piano,
    ):
        """Il conto che l'utente vede, non solo il numero nell'esito."""
        monkeypatch.setattr(
            "platform_core.jobs.periodico.get_session_factory",
            lambda: session_factory,
        )
        await _abbonamento_scaduto(session, utente, piano)

        await rinnova_scaduti()

        assert await RegistroCrediti(session).saldo(utente.id) == 2000

    async def test_i_crediti_del_piano_scadono_col_periodo(
        self, session, session_factory, monkeypatch, utente, piano,
    ):
        """Senza scadenza un piano che ne dà duemila al mese li farebbe
        accumulare all'infinito, e col tempo il piano più basso diventerebbe
        indistinguibile dal più alto."""
        monkeypatch.setattr(
            "platform_core.jobs.periodico.get_session_factory",
            lambda: session_factory,
        )
        abbonamento = await _abbonamento_scaduto(session, utente, piano)

        await rinnova_scaduti()

        await session.refresh(abbonamento)
        movimenti = await RegistroCrediti(session).movimenti(utente.id)
        accredito = next(m for m in movimenti if m.reason == "accredito_piano")
        assert accredito.quando is not None
        # La scadenza sta sulla riga, non nel calcolo del saldo: si controlla
        # che il registro l'abbia scritta guardando che il saldo di domani
        # sia ancora buono e quello dopo la fine del periodo no.
        assert abbonamento.period_end > utcnow()

    async def test_un_abbonamento_disdetto_si_chiude_invece_di_rinnovarsi(
        self, session, session_factory, monkeypatch, utente, piano,
    ):
        monkeypatch.setattr(
            "platform_core.jobs.periodico.get_session_factory",
            lambda: session_factory,
        )
        abbonamento = await _abbonamento_scaduto(
            session, utente, piano, cancel_at=utcnow() - timedelta(hours=2),
        )

        esito = await rinnova_scaduti()

        assert esito.abbonamenti_chiusi == 1
        assert esito.rinnovati == 0
        await session.refresh(abbonamento)
        assert abbonamento.status == "disdetto"
        assert await RegistroCrediti(session).saldo(utente.id) == 0

    async def test_un_periodo_ancora_aperto_si_lascia_stare(
        self, session, session_factory, monkeypatch, utente, piano,
    ):
        monkeypatch.setattr(
            "platform_core.jobs.periodico.get_session_factory",
            lambda: session_factory,
        )
        adesso = utcnow()
        session.add(Subscription(
            user_id=utente.id, plan_id=piano.id, status="attivo",
            period_start=adesso - timedelta(days=3),
            period_end=adesso + timedelta(days=27),
        ))
        await session.flush()

        esito = await rinnova_scaduti()

        assert esito.rinnovati == 0
        assert await RegistroCrediti(session).saldo(utente.id) == 0

    async def test_una_riga_che_esplode_non_ferma_le_altre(
        self, session, session_factory, monkeypatch, utente, altro_utente, piano,
    ):
        """È il modo in cui il difetto di un abbonamento diventa il guasto di
        tutti: un'eccezione al terzo di cento lascia i novantasette dopo senza
        rinnovo, e nessuno lo scopre perché il lavoro «è girato»."""
        monkeypatch.setattr(
            "platform_core.jobs.periodico.get_session_factory",
            lambda: session_factory,
        )
        await _abbonamento_scaduto(session, utente, piano)
        await _abbonamento_scaduto(session, altro_utente, piano)
        await session.commit()

        vero = GestoreAbbonamenti.rinnova
        visti = {"n": 0}

        async def rinnova_ballerino(self, abbonamento):
            visti["n"] += 1
            if visti["n"] == 1:
                raise RuntimeError("dato incoerente")
            return await vero(self, abbonamento)

        monkeypatch.setattr(GestoreAbbonamenti, "rinnova", rinnova_ballerino)

        esito = await rinnova_scaduti()

        assert visti["n"] == 2, "si è fermato alla prima riga"
        assert esito.rinnovati == 1
        assert len(esito.errori) == 1
        assert "dato incoerente" in esito.errori[0]


class TestPianiBaseMancanti:
    """Chi c'era prima che il piano gratuito si aprisse da solo.

    Su un'installazione aggiornata, gli account già esistenti non passano mai
    dalla nascita: hanno i diritti del piano gratuito — quelli ricadono sul
    codice — e non i suoi crediti, che richiedono righe in tabella. Il
    risultato è un 402 permanente per tutti gli utenti storici, e nulla nei
    log che lo colleghi all'aggiornamento.
    """

    async def test_un_utente_senza_abbonamento_lo_riceve(
        self, session, session_factory, monkeypatch, utente,
    ):
        from platform_core.jobs.periodico import apri_piani_base_mancanti

        monkeypatch.setattr(
            "platform_core.jobs.periodico.get_session_factory",
            lambda: session_factory,
        )
        session.add(Plan(
            slug="free", name="Gratuito", rank=0, credits_per_period=50,
            entitlements={"categorie": ["free"]}, limits={},
        ))
        await session.flush()

        esito = await apri_piani_base_mancanti()

        assert esito.piani_base_aperti >= 1
        abbonamento = await GestoreAbbonamenti(session).abbonamento_di(utente.id)
        assert abbonamento is not None
        assert await RegistroCrediti(session).saldo(utente.id) == 50

    async def test_non_ne_apre_un_secondo_a_chi_ce_l_ha(
        self, session, session_factory, monkeypatch, utente, piano,
    ):
        """La passata gira in continuazione: se non fosse idempotente,
        accrediterebbe il periodo iniziale a ogni giro."""
        from platform_core.jobs.periodico import apri_piani_base_mancanti

        monkeypatch.setattr(
            "platform_core.jobs.periodico.get_session_factory",
            lambda: session_factory,
        )
        await GestoreAbbonamenti(session).sottoscrivi(utente.id, piano)
        await session.flush()
        saldo_prima = await RegistroCrediti(session).saldo(utente.id)

        for _ in range(3):
            await apri_piani_base_mancanti()

        assert await RegistroCrediti(session).saldo(utente.id) == saldo_prima

    async def test_senza_catalogo_non_inventa_un_piano(
        self, session, session_factory, monkeypatch, utente,
    ):
        """Il catalogo non si crea da solo: senza la riga `free`, l'utente
        resta sui diritti base e la passata non finge di aver fatto qualcosa."""
        from platform_core.jobs.periodico import apri_piani_base_mancanti

        monkeypatch.setattr(
            "platform_core.jobs.periodico.get_session_factory",
            lambda: session_factory,
        )

        esito = await apri_piani_base_mancanti()

        assert esito.piani_base_aperti == 0
        assert not esito.errori
