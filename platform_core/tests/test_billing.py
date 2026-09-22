"""Piani, diritti e crediti.

Il test che il piano indica come irrinunciabile è
`test_richieste_concorrenti_non_portano_il_saldo_sotto_zero`, e non è teorico:
due schede aperte bastano a provocarlo. Senza il lock, entrambe le richieste
leggono lo stesso saldo, entrambe lo trovano sufficiente, entrambe sottraggono
— e il servizio ha regalato una risposta.

Il test gira su **connessioni separate**, perché è l'unico modo di provarlo:
due sessioni sulla stessa connessione non competono, e un test che usa quelle
passerebbe anche senza lock.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from platform_core.billing.credits import CreditiInsufficienti, RegistroCrediti
from platform_core.billing.entitlements import (
    LIMITI_BASE, PIANO_PREDEFINITO, Diritti, diritti_da, puo_parlare_con,
)
from platform_core.billing.plans import (
    GestoreAbbonamenti, ProviderDiSviluppo,
)
from platform_core.domain.base import utcnow
from platform_core.domain.billing_models import CreditEntry, Plan, Subscription

from .conftest import URL_TEST, richiede_database

pytestmark = richiede_database


@pytest_asyncio.fixture
async def piano_base(session) -> Plan:
    piano = Plan(
        slug=PIANO_PREDEFINITO, name="Gratuito", rank=0,
        credits_per_period=10,
        entitlements={"categorie": ["free"], "voce": False},
        limits={"messaggi_al_giorno": 20},
    )
    session.add(piano)
    await session.flush()
    return piano


@pytest_asyncio.fixture
async def piano_gold(session) -> Plan:
    piano = Plan(
        slug="gold", name="Gold", rank=20,
        price_monthly=1990, credits_per_period=500,
        entitlements={
            "categorie": ["free", "base", "gold"],
            "voce": True,
            "corpora_propri": True,
            "personalita_proprie": 3,
        },
        limits={"messaggi_al_giorno": 500, "conversazioni": 100},
    )
    session.add(piano)
    await session.flush()
    return piano


class TestRegistro:
    async def test_il_saldo_parte_da_zero(self, session, utente):
        assert await RegistroCrediti(session).saldo(utente.id) == 0

    async def test_il_saldo_e_la_somma_dei_movimenti(self, session, utente):
        registro = RegistroCrediti(session)
        await registro.accredita(utente.id, 100)
        await registro.consuma(utente.id, 30)
        await registro.accredita(utente.id, 5, reason="acquisto")

        assert await registro.saldo(utente.id) == 75

    async def test_consumare_piu_del_saldo_viene_rifiutato(self, session, utente):
        registro = RegistroCrediti(session)
        await registro.accredita(utente.id, 10)

        with pytest.raises(CreditiInsufficienti) as errore:
            await registro.consuma(utente.id, 11)

        assert errore.value.servono == 11
        assert errore.value.disponibili == 10

    async def test_il_saldo_resta_intatto_dopo_un_rifiuto(self, session, utente):
        registro = RegistroCrediti(session)
        await registro.accredita(utente.id, 10)

        with pytest.raises(CreditiInsufficienti):
            await registro.consuma(utente.id, 50)

        assert await registro.saldo(utente.id) == 10

    async def test_gli_accrediti_scaduti_non_contano(self, session, utente):
        """Un piano che dà cento crediti al mese non deve farli accumulare
        all'infinito, o il piano più basso diventa col tempo
        indistinguibile dal più alto."""
        from datetime import timedelta

        registro = RegistroCrediti(session)
        await registro.accredita(
            utente.id, 100, expires_at=utcnow() - timedelta(days=1),
        )
        await registro.accredita(utente.id, 7, reason="acquisto")

        assert await registro.saldo(utente.id) == 7

    async def test_i_crediti_comprati_non_scadono(self, session, utente):
        """Sono stati pagati: farli scadere sarebbe portare via qualcosa."""
        registro = RegistroCrediti(session)
        voce = await registro.accredita(utente.id, 50, reason="acquisto")

        assert voce.expires_at is None

    async def test_il_rimborso_e_una_riga_propria(self, session, utente):
        """Chi legge il registro deve vedere che una risposta è stata
        tentata, è fallita, ed è stata restituita. Cancellando il consumo,
        quella storia sparirebbe."""
        registro = RegistroCrediti(session)
        await registro.accredita(utente.id, 100)
        await registro.consuma(utente.id, 10)
        await registro.rimborsa(utente.id, 10, note="il modello non ha risposto")

        movimenti = await registro.movimenti(utente.id)

        assert [m.reason for m in movimenti] == [
            "rimborso", "consumo", "accredito_piano",
        ]
        assert await registro.saldo(utente.id) == 100

    async def test_una_rettifica_senza_motivo_viene_rifiutata(
        self, session, utente
    ):
        with pytest.raises(ValueError, match="senza motivo"):
            await RegistroCrediti(session).rettifica(utente.id, 10, note="  ")

    async def test_un_movimento_nullo_viene_rifiutato(self, session, utente):
        """Una riga che non muove nulla è rumore in un registro che si legge
        a mano."""
        with pytest.raises(ValueError):
            await RegistroCrediti(session).accredita(utente.id, 0)

    async def test_il_consumo_registra_la_conversazione(self, session, utente):
        """Senza, «dove sono finiti i miei crediti?» non ha risposta."""
        from platform_core.domain.repositories import ConversationRepository

        conversazione = await ConversationRepository(session).create(utente)
        registro = RegistroCrediti(session)
        await registro.accredita(utente.id, 10)
        await registro.consuma(utente.id, 1, conversation_id=conversazione.id)

        movimenti = await registro.movimenti(utente.id)
        assert movimenti[0].conversation_id == conversazione.id

    async def test_i_saldi_sono_separati_per_utente(
        self, session, utente, altro_utente
    ):
        registro = RegistroCrediti(session)
        await registro.accredita(utente.id, 100)

        assert await registro.saldo(altro_utente.id) == 0


class TestConcorrenza:
    """Il controllo che il piano chiede."""

    async def test_richieste_concorrenti_non_portano_il_saldo_sotto_zero(
        self, engine
    ):
        """Dieci consumi simultanei su un saldo che ne copre tre.

        Su **connessioni separate**, ed è l'unico modo di provarlo: due
        sessioni sulla stessa connessione non competono, e un test che usa
        quelle passerebbe anche senza lock.

        Senza `FOR UPDATE` tutte e dieci leggono lo stesso saldo, tutte lo
        trovano sufficiente, e il saldo finisce a −7.

        Il test non usa le fixture `session` e `utente`: quelle vivono in una
        transazione che viene annullata, quindi le loro righe non esistono per
        le altre connessioni. Si crea un utente proprio e lo si toglie alla
        fine.
        """
        from sqlalchemy import delete

        from platform_core.domain.models import User

        factory = async_sessionmaker(engine, expire_on_commit=False)
        sub = f"concorrenza-{uuid.uuid4().hex[:12]}"

        async with factory() as preparazione:
            utente = User(keycloak_sub=sub, email=f"{sub}@example.com")
            preparazione.add(utente)
            await preparazione.flush()
            user_id = utente.id
            await RegistroCrediti(preparazione).accredita(user_id, 3)
            await preparazione.commit()

        async def prova_a_consumare() -> bool:
            async with factory() as s:
                try:
                    await RegistroCrediti(s).consuma(user_id, 1)
                    await s.commit()
                    return True
                except CreditiInsufficienti:
                    await s.rollback()
                    return False

        try:
            esiti = await asyncio.gather(
                *(prova_a_consumare() for _ in range(10))
            )

            async with factory() as verifica:
                saldo = await RegistroCrediti(verifica).saldo(user_id)

            assert sum(esiti) == 3, (
                f"dovevano passarne tre, ne sono passate {sum(esiti)}"
            )
            assert saldo == 0, f"il saldo è finito a {saldo}"
        finally:
            async with factory() as pulizia:
                await pulizia.execute(delete(User).where(User.id == user_id))
                await pulizia.commit()


class TestDiritti:
    def test_senza_abbonamento_si_ricade_sul_piano_base(self):
        """Un servizio che smette di rispondere a chi non paga è una scelta
        legittima; farlo senza dire quale piano servirebbe non lo è."""
        diritti = diritti_da(None)

        assert diritti.piano == PIANO_PREDEFINITO
        assert diritti.predefiniti
        assert diritti.limiti == LIMITI_BASE

    async def test_un_abbonamento_vivo_da_i_suoi_diritti(
        self, session, utente, piano_gold
    ):
        abbonamento = await GestoreAbbonamenti(session).sottoscrivi(
            utente.id, piano_gold,
        )
        diritti = diritti_da(abbonamento)

        assert diritti.piano == "gold"
        assert diritti.voce
        assert not diritti.predefiniti
        assert "gold" in diritti.categorie

    async def test_un_abbonamento_sospeso_non_da_nulla_in_piu(
        self, session, utente, piano_gold, piano_base
    ):
        """«Non ha mai pagato» e «ha smesso di pagare» contano per la
        contabilità, non per cosa può fare oggi."""
        abbonamento = await GestoreAbbonamenti(session).sottoscrivi(
            utente.id, piano_gold,
        )
        abbonamento.status = "sospeso"

        diritti = diritti_da(abbonamento, piano_base=piano_base)

        assert diritti.piano == PIANO_PREDEFINITO
        assert not diritti.voce

    def test_i_diritti_del_piano_sostituiscono_quelli_base(self):
        """Unire le due liste renderebbe impossibile togliere qualcosa da un
        piano."""
        piano = Plan(
            slug="strano", name="Strano",
            entitlements={"categorie": ["gold"]},
        )
        abbonamento = Subscription(
            status="attivo", period_start=utcnow(), period_end=utcnow(),
        )
        abbonamento.plan = piano

        diritti = diritti_da(abbonamento)

        assert diritti.categorie == ["gold"]
        assert "free" not in diritti.categorie

    def test_una_personalita_senza_categoria_e_di_tutti(self):
        """È lo stato in cui nasce: negarla renderebbe invisibile ogni voce
        che nessuno ha ancora collocato."""
        assert Diritti().puo_usare(None)

    def test_un_limite_negativo_significa_illimitato(self):
        """`None` e non un numero grandissimo: «illimitato» e «un milione» si
        comportano allo stesso modo finché qualcuno non arriva a un
        milione."""
        diritti = Diritti(limiti={"messaggi_al_giorno": -1})
        assert diritti.limite("messaggi_al_giorno") is None


class TestVerdetti:
    def test_il_rifiuto_dice_cosa_manca(self):
        """L'utente deve sapere cosa fare, non solo che non può."""
        verdetto = puo_parlare_con(
            Diritti(), "gold", piani_che_la_comprendono=["gold"],
        )

        assert not verdetto
        assert verdetto.motivo
        assert verdetto.serve == "gold"

    def test_il_rifiuto_distingue_i_due_casi(self):
        """«Non hai un piano» e «il tuo piano non lo comprende» richiedono
        azioni diverse."""
        senza = puo_parlare_con(Diritti(), "gold")
        con = puo_parlare_con(
            Diritti(piano="base", categorie=["free", "base"], predefiniti=False),
            "gold",
        )

        assert "gratuito" in senza.motivo
        assert "base" in con.motivo

    def test_il_permesso_non_ha_bisogno_di_motivi(self):
        verdetto = puo_parlare_con(Diritti(), "free")
        assert verdetto and not verdetto.motivo


class TestAbbonamenti:
    async def test_sottoscrivere_accredita_il_primo_periodo(
        self, session, utente, piano_gold
    ):
        gestore = GestoreAbbonamenti(session)
        await gestore.sottoscrivi(utente.id, piano_gold)

        assert await RegistroCrediti(session).saldo(utente.id) == 500

    async def test_i_crediti_del_piano_scadono_col_periodo(
        self, session, utente, piano_gold
    ):
        gestore = GestoreAbbonamenti(session)
        abbonamento = await gestore.sottoscrivi(utente.id, piano_gold)

        movimenti = await RegistroCrediti(session).movimenti(utente.id)
        from sqlalchemy import select

        voce = (await session.execute(
            select(CreditEntry).where(CreditEntry.user_id == utente.id)
        )).scalars().first()

        assert voce.expires_at == abbonamento.period_end

    async def test_cambiare_piano_chiude_il_precedente(
        self, session, utente, piano_base, piano_gold
    ):
        gestore = GestoreAbbonamenti(session)
        primo = await gestore.sottoscrivi(utente.id, piano_base)
        await gestore.sottoscrivi(utente.id, piano_gold)

        assert primo.status == "disdetto"
        corrente = await gestore.abbonamento_di(utente.id)
        assert corrente.plan_id == piano_gold.id

    async def test_disdire_lascia_attivo_fino_a_scadenza(
        self, session, utente, piano_gold
    ):
        """È già stato pagato: chiuderlo subito toglierebbe qualcosa a cui
        l'utente ha diritto."""
        gestore = GestoreAbbonamenti(session)
        abbonamento = await gestore.sottoscrivi(utente.id, piano_gold)

        await gestore.disdici(abbonamento)

        assert abbonamento.vivo
        assert abbonamento.cancel_at == abbonamento.period_end

    async def test_il_rinnovo_accredita_senza_azzerare(
        self, session, utente, piano_gold
    ):
        """Azzerare al rinnovo porterebbe via i crediti comprati a parte."""
        gestore = GestoreAbbonamenti(session)
        registro = RegistroCrediti(session)
        abbonamento = await gestore.sottoscrivi(utente.id, piano_gold)
        await registro.accredita(utente.id, 50, reason="acquisto")
        await registro.consuma(utente.id, 100)

        esito = await gestore.rinnova(abbonamento)

        assert esito.crediti_accreditati == 500
        # 500 del primo periodo − 100 consumati + 50 comprati + 500 nuovi.
        assert await registro.saldo(utente.id) == 950

    async def test_il_rinnovo_di_un_disdetto_lo_chiude(
        self, session, utente, piano_gold
    ):
        gestore = GestoreAbbonamenti(session)
        abbonamento = await gestore.sottoscrivi(utente.id, piano_gold)
        abbonamento.cancel_at = utcnow()

        esito = await gestore.rinnova(abbonamento)

        assert esito.chiuso
        assert abbonamento.status == "disdetto"

    async def test_un_pagamento_fallito_sospende_e_non_disdice(
        self, session, utente, piano_gold
    ):
        """Una carta scaduta non è una disdetta: chiudere al primo tentativo
        non riuscito perderebbe un cliente che voleva restare."""

        class ProviderCheRifiuta(ProviderDiSviluppo):
            async def e_pagato(self, external_id: str) -> bool:
                return False

        gestore = GestoreAbbonamenti(session, ProviderCheRifiuta())
        abbonamento = await gestore.sottoscrivi(utente.id, piano_gold)

        esito = await gestore.rinnova(abbonamento)

        assert abbonamento.status == "sospeso"
        assert "pagamento" in esito.motivo

    async def test_si_trovano_quelli_da_rinnovare(
        self, session, utente, piano_gold
    ):
        from datetime import timedelta

        gestore = GestoreAbbonamenti(session)
        abbonamento = await gestore.sottoscrivi(utente.id, piano_gold)
        # Anche `period_start`, o si viola `period_end > period_start`: il
        # vincolo è giusto, ed è il test a dover simulare un periodo finito
        # invece di uno impossibile.
        abbonamento.period_start = utcnow() - timedelta(days=31)
        abbonamento.period_end = utcnow() - timedelta(hours=1)
        await session.flush()

        da_fare = await gestore.da_rinnovare()
        assert abbonamento.id in [a.id for a in da_fare]
