"""La memoria.

Due proprietà da proteggere sopra le altre.

**I fatti si chiudono, non si cancellano.** «Vivo a Budapest» non sparisce
quando arriva «mi sono trasferito a Vienna»: viene chiusa, e la storia resta.
Se qualcuno sostituisse `sostituisci` con una `UPDATE`, i test qui sotto
cadrebbero — ed è il punto, perché nient'altro se ne accorgerebbe: le
risposte resterebbero corrette, e sparirebbe soltanto la capacità di dire
cosa è cambiato e quando.

**Il consolidamento non è una rifinitura.** Senza, dopo mesi mille ricordi
mediocri seppelliscono i dieci che contano, e il recupero peggiora invece di
migliorare.
"""
from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
import pytest_asyncio

from platform_core.domain.base import utcnow
from platform_core.domain.memory_models import Memory
from platform_core.memory.consolidation import (
    GENERI_ESCLUSIVI, SOGLIA_DUPLICATO, Consolidatore,
)
from platform_core.memory.extraction import MemoriaEstratta, _interpreta
from platform_core.memory.retrieval import (
    EMIVITA_GIORNI, PESI, MemoryRetriever, frequenza, punteggio_di, recenza,
)
from platform_core.memory.store import GIORNI_SESSIONE, MemoryStore

from .conftest import richiede_database
from .test_retrieval import EmbedderFinto

pytestmark = richiede_database


@pytest_asyncio.fixture
async def embedder() -> EmbedderFinto:
    return EmbedderFinto()


@pytest_asyncio.fixture
async def store(session, embedder) -> MemoryStore:
    return MemoryStore(session, embedder)


class TestSegnali:
    """I quattro segnali del recupero, presi uno per volta."""

    def test_la_recenza_decade_a_meta_dopo_un_emivita(self):
        adesso = utcnow()
        assert recenza(adesso, adesso=adesso) == pytest.approx(1.0)
        assert recenza(
            adesso - timedelta(days=EMIVITA_GIORNI), adesso=adesso,
        ) == pytest.approx(0.5, abs=0.01)

    def test_la_recenza_non_arriva_mai_a_zero(self):
        """Una memoria vecchissima ma pertinente deve poter emergere."""
        adesso = utcnow()
        assert recenza(adesso - timedelta(days=3650), adesso=adesso) > 0

    def test_senza_data_la_recenza_e_nulla(self):
        assert recenza(None) == 0.0

    def test_la_frequenza_satura(self):
        """Senza tetto, una memoria richiamata cento volte schiaccerebbe tutto."""
        assert frequenza(0) == 0.0
        assert frequenza(10) == pytest.approx(1.0)
        assert frequenza(1000) == pytest.approx(1.0)

    def test_la_frequenza_cresce_di_meno_col_crescere(self):
        """La differenza fra una e due volte conta più di quella fra nove e
        dieci: è come funziona la memoria di chiunque."""
        primo_salto = frequenza(2) - frequenza(1)
        ultimo_salto = frequenza(10) - frequenza(9)
        assert primo_salto > ultimo_salto

    def test_i_pesi_sommano_a_uno(self):
        """Il punteggio dev'essere leggibile come «quanto vale, da 0 a 1»: uno
        che non si sa interpretare non si sa nemmeno tarare."""
        assert sum(PESI.values()) == pytest.approx(1.0)

    def test_la_semantica_pesa_piu_di_tutto(self):
        assert PESI["semantica"] > max(
            v for k, v in PESI.items() if k != "semantica"
        )

    def test_il_punteggio_massimo_e_uno(self):
        assert punteggio_di(
            semantica=1.0, recenza_=1.0, importanza=1.0, frequenza_=1.0,
        ) == pytest.approx(1.0)


class TestScrittura:
    async def test_una_memoria_nasce_viva(self, store, utente):
        m = await store.ricorda(user_id=utente.id, content="Vive a Budapest")

        assert m.viva
        assert m.valid_to is None
        assert m.embedding is not None, "senza vettore non si troverà mai"

    async def test_le_sessioni_scadono(self, store, utente):
        """Senza scadenza, dopo un anno le memorie di sessione sarebbero il
        novanta per cento di tutto."""
        m = await store.ricorda(
            user_id=utente.id, content="Si è parlato di stoicismo", kind="sessione",
        )

        assert m.expires_at is not None
        giorni = (m.expires_at - m.first_seen_at).days
        assert giorni == GIORNI_SESSIONE

    async def test_l_identita_non_scade(self, store, utente):
        m = await store.ricorda(
            user_id=utente.id, content="Si chiama Anna", kind="identita",
        )
        assert m.expires_at is None

    async def test_i_valori_fuori_scala_si_riportano_dentro(self, store, utente):
        m = await store.ricorda(
            user_id=utente.id, content="Qualcosa", importance=5.0, confidence=-1.0,
        )
        assert m.importance == 1.0
        assert m.confidence == 0.0


class TestVersionamentoDeiFatti:
    """La proprietà che dà senso al modulo."""

    async def test_sostituire_chiude_invece_di_cancellare(self, store, utente):
        vecchia = await store.ricorda(
            user_id=utente.id, content="Vive a Budapest", kind="identita",
        )
        nuova = await store.sostituisci(vecchia, "Vive a Vienna")

        assert vecchia.valid_to is not None, "la vecchia è stata cancellata"
        assert vecchia.superseded_by == nuova.id
        assert not vecchia.viva
        assert nuova.viva

    async def test_la_vecchia_resta_leggibile(self, store, utente):
        """«Dove vivevo l'anno scorso?» deve avere una risposta."""
        vecchia = await store.ricorda(
            user_id=utente.id, content="Vive a Budapest", kind="identita",
        )
        await store.sostituisci(vecchia, "Vive a Vienna")

        tutte = await store.per_utente(utente.id, includi_superate=True)
        contenuti = [m.content for m in tutte]

        assert "Vive a Budapest" in contenuti
        assert "Vive a Vienna" in contenuti

    async def test_la_superata_non_entra_nelle_risposte(self, store, utente):
        vecchia = await store.ricorda(
            user_id=utente.id, content="Vive a Budapest", kind="identita",
        )
        await store.sostituisci(vecchia, "Vive a Vienna")

        vive = await store.per_utente(utente.id)
        assert [m.content for m in vive] == ["Vive a Vienna"]

    async def test_la_catena_si_ripercorre(self, store, utente):
        prima = await store.ricorda(
            user_id=utente.id, content="Vive a Roma", kind="identita",
        )
        seconda = await store.sostituisci(prima, "Vive a Budapest")
        terza = await store.sostituisci(seconda, "Vive a Vienna")

        catena = await store.storia_di(utente.id, terza.id)

        assert [m.content for m in catena] == [
            "Vive a Vienna", "Vive a Budapest", "Vive a Roma",
        ]

    async def test_la_nuova_sa_cosa_ha_sostituito(self, store, utente):
        vecchia = await store.ricorda(user_id=utente.id, content="Lavora a Milano")
        nuova = await store.sostituisci(vecchia, "Lavora a Torino")

        assert nuova.meta["sostituisce"] == str(vecchia.id)


class TestIsolamento:
    async def test_non_si_legge_la_memoria_di_un_altro(
        self, store, utente, altro_utente
    ):
        mia = await store.ricorda(user_id=utente.id, content="Un segreto")

        assert await store.una(altro_utente.id, mia.id) is None
        assert await store.una(utente.id, mia.id) is not None

    async def test_l_elenco_e_solo_il_proprio(self, store, utente, altro_utente):
        await store.ricorda(user_id=utente.id, content="Cosa mia")
        await store.ricorda(user_id=altro_utente.id, content="Cosa sua")

        mie = await store.per_utente(utente.id)
        assert [m.content for m in mie] == ["Cosa mia"]

    async def test_non_si_cancella_la_memoria_di_un_altro(
        self, store, utente, altro_utente
    ):
        mia = await store.ricorda(user_id=utente.id, content="Un segreto")

        assert not await store.dimentica(altro_utente.id, mia.id)
        assert await store.una(utente.id, mia.id) is not None

    async def test_le_memorie_di_una_personalita_non_escono_da_li(
        self, store, utente, session
    ):
        """Un dettaglio confidato a una voce non deve riaffiorare parlando con
        un'altra: è il modo in cui una memoria utile diventa inquietante."""
        from platform_core.domain.knowledge_models import Personality

        una = Personality(slug=f"a-{uuid.uuid4().hex[:6]}", display_name="A")
        altra = Personality(slug=f"b-{uuid.uuid4().hex[:6]}", display_name="B")
        session.add_all([una, altra])
        await session.flush()

        await store.ricorda(
            user_id=utente.id, content="Confidato a una sola",
            personality_id=una.id,
        )
        await store.ricorda(user_id=utente.id, content="Vale con tutte")

        con_altra = await store.per_utente(utente.id, personality_id=altra.id)
        assert [m.content for m in con_altra] == ["Vale con tutte"]


class TestCancellazione:
    async def test_dimenticare_cancella_davvero(self, store, utente):
        """Chi chiede di dimenticare chiede che sparisca, non che si chiuda."""
        m = await store.ricorda(user_id=utente.id, content="Da dimenticare")

        assert await store.dimentica(utente.id, m.id)

        tutte = await store.per_utente(utente.id, includi_superate=True)
        assert tutte == []

    async def test_dimenticare_tutto(self, store, utente, altro_utente):
        for i in range(3):
            await store.ricorda(user_id=utente.id, content=f"Cosa {i}")
        await store.ricorda(user_id=altro_utente.id, content="Cosa sua")

        quante = await store.dimentica_tutto(utente.id)

        assert quante == 3
        assert await store.per_utente(utente.id, includi_superate=True) == []
        assert len(await store.per_utente(altro_utente.id)) == 1


class TestRecupero:
    async def test_trova_la_memoria_pertinente(
        self, session, store, embedder, utente
    ):
        await store.ricorda(user_id=utente.id, content="Vive a Vienna e lavora come architetto")
        await store.ricorda(user_id=utente.id, content="Preferisce il caffè al tè")

        trovate = await MemoryRetriever(session, embedder).cerca(
            user_id=utente.id, domanda="dove vive e che lavoro fa", limite=1,
            soglia=0.0,
        )

        assert trovate
        assert "Vienna" in trovate[0].memoria.content

    async def test_le_superate_non_si_recuperano(
        self, session, store, embedder, utente
    ):
        vecchia = await store.ricorda(
            user_id=utente.id, content="Vive a Budapest", kind="identita",
        )
        await store.sostituisci(vecchia, "Vive a Vienna")

        trovate = await MemoryRetriever(session, embedder).cerca(
            user_id=utente.id, domanda="dove vive", soglia=0.0,
        )
        contenuti = [t.memoria.content for t in trovate]

        assert "Vive a Budapest" not in contenuti

    async def test_niente_di_pertinente_restituisce_niente(
        self, session, store, embedder, utente
    ):
        """Nessuna memoria è una risposta legittima: portare le meno peggio
        significa mettere nel prompt materiale irrilevante, e un modello che
        legge materiale irrilevante ci costruisce sopra."""
        await store.ricorda(user_id=utente.id, content="Preferisce il caffè al tè")

        trovate = await MemoryRetriever(session, embedder).cerca(
            user_id=utente.id,
            domanda="quali sono le tariffe doganali fra Cile e Norvegia",
            soglia=0.4,
        )

        assert trovate == []

    async def test_l_uso_alimenta_frequenza_e_recenza(
        self, session, store, embedder, utente
    ):
        await store.ricorda(user_id=utente.id, content="Vive a Vienna")

        recuperatore = MemoryRetriever(session, embedder)
        trovate = await recuperatore.cerca(
            user_id=utente.id, domanda="dove vive", soglia=0.0,
        )
        await recuperatore.segna_usate(trovate)

        assert trovate[0].memoria.times_referenced == 1
        assert trovate[0].memoria.last_referenced_at is not None

    async def test_non_si_recuperano_memorie_altrui(
        self, session, store, embedder, utente, altro_utente
    ):
        await store.ricorda(user_id=altro_utente.id, content="Vive a Vienna")

        trovate = await MemoryRetriever(session, embedder).cerca(
            user_id=utente.id, domanda="dove vive", soglia=0.0,
        )
        assert trovate == []


class TestConsolidamento:
    async def test_le_scadute_si_chiudono(self, session, store, utente):
        m = await store.ricorda(
            user_id=utente.id, content="Vecchio riassunto", kind="sessione",
        )
        m.expires_at = utcnow() - timedelta(days=1)
        await session.flush()

        esito = await Consolidatore(session).consolida(utente.id)

        assert esito.scadute == 1
        assert m.valid_to is not None

    async def test_i_duplicati_si_fondono(self, session, store, utente):
        await store.ricorda(user_id=utente.id, content="Vive a Vienna da tre anni")
        await store.ricorda(user_id=utente.id, content="Vive a Vienna da tre anni")

        esito = await Consolidatore(session).consolida(utente.id)

        assert esito.fuse == 1
        assert len(await store.per_utente(utente.id)) == 1

    async def test_la_confermata_diventa_piu_importante(
        self, session, store, utente
    ):
        """Essere stata detta due volte è un segnale."""
        prima = await store.ricorda(
            user_id=utente.id, content="Vive a Vienna da tre anni", importance=0.5,
        )
        await store.ricorda(
            user_id=utente.id, content="Vive a Vienna da tre anni", importance=0.5,
        )

        await Consolidatore(session).consolida(utente.id)

        assert prima.importance > 0.5

    async def test_le_cose_diverse_non_si_fondono(self, session, store, utente):
        """Sbagliare fondendo perde informazione e non si recupera."""
        await store.ricorda(user_id=utente.id, content="Vive a Vienna")
        await store.ricorda(user_id=utente.id, content="Preferisce il caffè al tè")

        esito = await Consolidatore(session).consolida(utente.id)

        assert esito.fuse == 0
        assert len(await store.per_utente(utente.id)) == 2

    async def test_le_preferenze_si_accumulano(self, session, store, utente):
        """Cosa ti piace non è esclusivo: applicare la sostituzione farebbe
        dimenticare tutto tranne l'ultima cosa detta."""
        await store.ricorda(
            user_id=utente.id, content="Gli piace il jazz", kind="preferenza",
        )
        await store.ricorda(
            user_id=utente.id, content="Gli piace il blues", kind="preferenza",
        )

        await Consolidatore(session).consolida(utente.id)

        assert len(await store.per_utente(utente.id)) == 2

    def test_solo_l_identita_e_esclusiva(self):
        assert "identita" in GENERI_ESCLUSIVI
        assert "preferenza" not in GENERI_ESCLUSIVI

    def test_la_soglia_dei_duplicati_e_alta(self):
        """Sbagliare fondendo è peggio che lasciare due righe."""
        assert SOGLIA_DUPLICATO >= 0.9


class TestEstrazione:
    def test_interpreta_la_forma_attesa(self):
        estratte = _interpreta({"memorie": [
            {"contenuto": "Vive a Vienna", "genere": "identita",
             "importanza": 0.9, "confidenza": 0.95},
        ]})

        assert len(estratte) == 1
        assert estratte[0].genere == "identita"
        assert estratte[0].importanza == 0.9

    def test_tollera_le_chiavi_inglesi(self):
        estratte = _interpreta({"memories": [
            {"content": "Lives in Vienna", "kind": "identity", "importance": 0.8},
        ]})

        assert len(estratte) == 1
        assert estratte[0].genere == "identita"

    def test_un_genere_ignoto_diventa_fatto(self):
        """`fatto` e non `identita`: una cosa classificata male come fatto
        scade e pesa poco, come identità resta per sempre e pesa molto."""
        estratte = _interpreta({"memorie": [
            {"contenuto": "Qualcosa di lungo abbastanza", "genere": "boh"},
        ]})
        assert estratte[0].genere == "fatto"

    def test_le_frasi_troppo_brevi_si_scartano(self):
        """«sì» e «ok» passano quando il modello ha poco da dire."""
        estratte = _interpreta({"memorie": [
            {"contenuto": "ok"}, {"contenuto": "Vive a Vienna da tre anni"},
        ]})
        assert len(estratte) == 1

    def test_una_lista_vuota_e_un_esito_normale(self):
        assert _interpreta({"memorie": []}) == []

    def test_una_forma_illeggibile_non_esplode(self):
        assert _interpreta("non è un oggetto") == []
        assert _interpreta(None) == []


class TestAccessi:
    async def test_l_accesso_altrui_lascia_traccia(
        self, session, store, utente, altro_utente
    ):
        """Un accesso non registrato alle memorie di qualcun altro è
        indistinguibile da un abuso."""
        await store.registra_accesso(
            subject_id=utente.id,
            actor_id=altro_utente.id,
            action="memorie.lette",
            count=7,
            reason="verifica di una segnalazione",
        )

        accessi = await store.accessi_a(utente.id)

        assert len(accessi) == 1
        assert accessi[0].actor_id == altro_utente.id
        assert accessi[0].count == 7
        assert "segnalazione" in accessi[0].reason

    async def test_l_azione_di_sistema_non_ha_autore(self, session, store, utente):
        """«Non è stato nessuno» va distinto da «non si sa chi»."""
        await store.registra_accesso(
            subject_id=utente.id, actor_id=None, action="memorie.consolidate",
        )

        accessi = await store.accessi_a(utente.id)
        assert accessi[0].actor_id is None
