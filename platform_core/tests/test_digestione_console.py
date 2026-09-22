"""La digestione vista dalla console.

Tre promesse, e ciascuna ha già avuto modo di essere tradita altrove.

1. **Le etichette si leggono per passaggio.** Un'aggregazione per categoria
   dice che il corpus è distribuito in un certo modo e non dice mai se una
   singola classificazione sia giusta: è la domanda che ci si pone quando il
   recupero sbaglia, e serve l'endpoint che la sappia rispondere.
2. **La digestione si accoda come un lavoro.** Su un corpus vero sono ore, e
   una richiesta HTTP tenuta aperta per tutto quel tempo viene chiusa da un
   proxy a metà, lasciando chi guardava senza sapere se stia proseguendo.
3. **Il testo di prima non si perde.** Se la pulizia porta via prosa
   dell'autore insieme all'apparato del curatore, col solo testo finale
   l'errore è invisibile — resta un passaggio plausibile, solo più corto.
"""
from __future__ import annotations

import uuid

import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from httpx import ASGITransport, AsyncClient

from platform_core.api.deps import get_key_value_store
from platform_core.builds.queue import CodaBuild, CodaInMemoria, JobBuild
from platform_core.builds.repository import BuildRepository
from platform_core.builds.worker import WorkerBuild
from platform_core.domain.build_models import Build
from platform_core.domain.knowledge_models import Chunk, Document, KnowledgeBase
from platform_core.knowledge.digestion import EsitoDigestione

from .conftest import richiede_database
from .test_admin import app_con, embedder  # noqa: F401

pytestmark = richiede_database


@pytest_asyncio.fixture
async def coda() -> CodaInMemoria:
    """Una coda che vive e muore col test.

    Senza, il client di prova accoda nel Redis di sviluppo lavori che puntano
    a build create in una transazione annullata: il worker vero li prende, non
    li trova, e ne scrive un avviso — rumore prodotto da un test in un
    processo che non c'entra.
    """
    return CodaInMemoria()


@pytest_asyncio.fixture
async def admin(session, session_factory, principal_admin, embedder, utente, coda):  # noqa: F811
    app = app_con(principal_admin, session, session_factory, embedder)
    app.dependency_overrides[get_key_value_store] = lambda: coda
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


class CodaFinta:
    """La coda, ridotta a ciò che il worker le chiede durante un job."""

    def completa(self, worker_id, job) -> None:
        pass

    def in_carico(self, worker_id):
        return []


class DigestioneFinta:
    """Un classificatore che non chiama nessun modello.

    Serve a verificare il percorso — job preso, avanzamenti scritti, esito
    registrato — senza far dipendere il test da un provider raggiungibile e
    da quattro ore di lavoro vero.
    """

    def __init__(self, *, avvisi=(), esplode: str = "") -> None:
        self._avvisi = list(avvisi)
        self._esplode = esplode
        self.chiamata = None

    async def digerisci(self, kb, *, chi="", rifai=False, avanzamento=None):
        self.chiamata = {"chi": chi, "rifai": rifai}
        for percentuale, messaggio in self._avvisi:
            if avanzamento:
                avanzamento(percentuale, 100, messaggio)
        if self._esplode:
            raise RuntimeError(self._esplode)
        return EsitoDigestione(esaminati=3, classificati=2, scartati_a_vista=1)


@pytest_asyncio.fixture
async def corpus(session, utente) -> KnowledgeBase:
    """Un corpus già digerito: prosa dell'autore, apparato, un ripulito."""
    kb = KnowledgeBase(
        slug=f"digerito-{uuid.uuid4().hex[:8]}",
        name="Corpus digerito",
        kind="corpus",
        embed_model="prova",
        owner_id=utente.id,
    )
    session.add(kb)
    await session.flush()

    documento = Document(kb_id=kb.id, title="Lettere", sha256=uuid.uuid4().hex)
    session.add(documento)
    await session.flush()

    session.add_all([
        Chunk(
            kb_id=kb.id, document_id=documento.id, ordinal=0,
            text="La virtù non si impara a parole ma con l'esercizio.",
            labels={
                "categorie": {"valori": 0.9, "stile": 0.3},
                "sintesi": "sulla virtù",
            },
            label_main="valori", provenance="autore", quality=0.95,
        ),
        Chunk(
            kb_id=kb.id, document_id=documento.id, ordinal=1,
            text="Ut a communibus initium faciam etc. Ep. lxvii",
            labels={
                "categorie": {"apparato": 1.0},
                "sintesi": "rimando del curatore",
            },
            label_main="apparato", provenance="editoriale", quality=0.1,
            discarded=True, discard_reason="rimando bibliografico del curatore",
        ),
        Chunk(
            kb_id=kb.id, document_id=documento.id, ordinal=2,
            text="Dopo tanto tempo ho riveduto i tuoi luoghi.",
            text_original="Ep. xlix . Dopo tanto tempo ho riveduto i tuoi luoghi.",
            labels={"categorie": {"avvenimento": 0.8}, "sintesi": "il ritorno"},
            label_main="avvenimento", provenance="autore", quality=0.9,
        ),
        Chunk(
            kb_id=kb.id, document_id=documento.id, ordinal=3,
            text="Un passaggio che nessuno ha ancora guardato.",
        ),
    ])
    await session.flush()
    return kb


async def _accoda(session, corpus_id, **params) -> Build:
    build = Build(
        personality_id=None, kind="digestione", status="in_coda",
        params={"kb_id": str(corpus_id), **params},
    )
    session.add(build)
    await session.flush()
    await session.commit()
    return build


def _job(build: Build) -> JobBuild:
    return JobBuild(
        build_id=build.id, kind="digestione", personality_id=uuid.UUID(int=0),
    )


class TestStato:
    async def test_conta_cio_che_serve_a_decidere(self, admin, corpus):  # noqa: F811
        r = await admin.get(f"/admin/knowledge-bases/{corpus.id}/digestion")
        assert r.status_code == 200

        n = r.json()["passaggi"]
        assert n["totale"] == 4
        assert n["etichettati"] == 3
        assert n["da_fare"] == 1
        assert n["scartati"] == 1
        assert n["ripuliti"] == 1

    async def test_gli_scartati_non_entrano_nelle_distribuzioni(
        self, admin, corpus,  # noqa: F811
    ):
        """Contarli falserebbe la lettura: la distribuzione serve a sapere di
        cosa è fatto il corpus **che verrà usato**, e l'apparato è proprio ciò
        che è stato tolto."""
        corpo = (await admin.get(
            f"/admin/knowledge-bases/{corpus.id}/digestion"
        )).json()

        assert "apparato" not in corpo["per_categoria"]
        assert corpo["per_categoria"] == {
            "valori": 1, "avvenimento": 1, "(nessuna)": 1,
        }
        assert corpo["per_provenienza"] == {"autore": 2, "(ignota)": 1}

    async def test_una_base_inesistente_e_un_404(self, admin):  # noqa: F811
        r = await admin.get(f"/admin/knowledge-bases/{uuid.uuid4()}/digestion")
        assert r.status_code == 404


class TestPassaggi:
    async def test_ogni_passaggio_porta_le_sue_etichette(
        self, admin, corpus,  # noqa: F811
    ):
        corpo = (await admin.get(
            f"/admin/knowledge-bases/{corpus.id}/chunks"
        )).json()

        assert corpo["totale"] == 4
        primo = corpo["passaggi"][0]
        assert primo["categoria"] == "valori"
        assert primo["categorie"] == {"valori": 0.9, "stile": 0.3}
        assert primo["provenienza"] == "autore"
        assert primo["sintesi"] == "sulla virtù"
        assert primo["documento"] == "Lettere"

    async def test_il_testo_di_prima_arriva_insieme_a_quello_di_dopo(
        self, admin, corpus,  # noqa: F811
    ):
        """Senza l'originale, una pulizia che ha tagliato troppo è
        indistinguibile da una riuscita."""
        corpo = (await admin.get(
            f"/admin/knowledge-bases/{corpus.id}/chunks?solo_ripuliti=true"
        )).json()

        assert corpo["totale"] == 1
        ripulito = corpo["passaggi"][0]
        assert ripulito["testo_originale"].endswith(ripulito["testo"])
        assert "Ep. xlix" in ripulito["testo_originale"]
        assert "Ep. xlix" not in ripulito["testo"]

    async def test_lo_scarto_dice_perche(self, admin, corpus):  # noqa: F811
        corpo = (await admin.get(
            f"/admin/knowledge-bases/{corpus.id}/chunks?solo_scartati=true"
        )).json()

        assert corpo["totale"] == 1
        assert corpo["passaggi"][0]["motivo_scarto"]

    async def test_il_filtro_per_categoria_restringe(
        self, admin, corpus,  # noqa: F811
    ):
        corpo = (await admin.get(
            f"/admin/knowledge-bases/{corpus.id}/chunks?categoria=avvenimento"
        )).json()

        assert corpo["totale"] == 1
        assert corpo["passaggi"][0]["categoria"] == "avvenimento"

    async def test_il_totale_e_quello_del_filtro_non_della_pagina(
        self, admin, corpus,  # noqa: F811
    ):
        """Il totale serve a sapere quanto resta da scorrere: se riportasse la
        pagina, il pulsante «mostra altri» sparirebbe al primo caricamento."""
        corpo = (await admin.get(
            f"/admin/knowledge-bases/{corpus.id}/chunks?limite=1"
        )).json()

        assert len(corpo["passaggi"]) == 1
        assert corpo["totale"] == 4


class TestAvvio:
    async def test_accoda_un_lavoro_senza_personalita(
        self, admin, corpus, session, coda,  # noqa: F811
    ):
        """La digestione riguarda un corpus, non una voce: inventare una
        personalità per riempire la colonna renderebbe il dato bugiardo
        invece che mancante."""
        r = await admin.post(
            f"/admin/knowledge-bases/{corpus.id}/digestion",
            json={"chi": "Seneca", "rifai": False},
        )
        assert r.status_code == 202
        assert r.json()["passaggi_da_fare"] == 1

        build = await session.get(Build, uuid.UUID(r.json()["build_id"]))
        assert build.kind == "digestione"
        assert build.personality_id is None
        assert build.params["kb_id"] == str(corpus.id)
        assert build.params["chi"] == "Seneca"

        # Accodato davvero: una build in stato «in coda» che nessun worker
        # vedrà mai resta in attesa per sempre, e sembra solo lenta.
        assert CodaBuild(coda).in_attesa() == 1

    async def test_senza_nulla_da_fare_e_un_409_non_un_lavoro_vuoto(
        self, admin, corpus, session,  # noqa: F811
    ):
        """Accodare un job che non ha niente da fare lo farebbe risultare
        riuscito senza aver toccato nulla: il modo peggiore di dire di no."""
        for c in (await session.execute(
            select(Chunk).where(Chunk.kb_id == corpus.id)
        )).scalars():
            c.labels = c.labels or {"categorie": {"stile": 0.5}}
        await session.flush()

        r = await admin.post(
            f"/admin/knowledge-bases/{corpus.id}/digestion", json={},
        )
        assert r.status_code == 409
        assert "rianalizza" in r.json()["detail"].lower()

    async def test_rianalizza_riparte_anche_da_completo(
        self, admin, corpus, session,  # noqa: F811
    ):
        for c in (await session.execute(
            select(Chunk).where(Chunk.kb_id == corpus.id)
        )).scalars():
            c.labels = {"categorie": {"stile": 0.5}}
        await session.flush()

        r = await admin.post(
            f"/admin/knowledge-bases/{corpus.id}/digestion",
            json={"rifai": True},
        )
        assert r.status_code == 202
        assert r.json()["passaggi_da_fare"] == 4


@pytest_asyncio.fixture
async def worker_sul_test(session_factory, monkeypatch):
    """Fa aprire al worker le sessioni del test, non quelle vere.

    Il worker non prende una fabbrica di sessioni dall'esterno — gira da solo,
    senza iniezione di dipendenze — e con quella vera aprirebbe una
    connessione a parte, dove il corpus creato dalla transazione del test non
    esiste ancora: la build risulterebbe «sparita dal database».
    """
    monkeypatch.setattr(
        "platform_core.builds.worker.get_session_factory",
        lambda: session_factory,
    )


class TestEsecuzione:
    """Il lavoro attraversa davvero il worker.

    Non basta che l'endpoint accodi: finché il worker non sa cosa sia una
    digestione, il job fallisce con «tipo sconosciuto» e sparisce dalla coda
    sembrando lavorato — il modo peggiore di non funzionare.
    """

    async def test_un_job_di_digestione_arriva_a_conclusione(
        self, session, worker_sul_test, corpus,  # noqa: F811
    ):
        build = await _accoda(session, corpus.id, chi="Seneca")

        digestione = DigestioneFinta()
        worker = WorkerBuild(
            "prova", coda=CodaFinta(), digestione=lambda s: digestione,
        )
        await worker._esegui(_job(build))

        await session.refresh(build)
        assert build.status == "riuscita"
        assert build.artifact_path is None, "una digestione non produce file"
        assert build.artifact_meta["classificati"] == 2
        assert digestione.chiamata == {"chi": "Seneca", "rifai": False}

    async def test_ogni_avviso_arriva_al_database(
        self, session, worker_sul_test, corpus,  # noqa: F811
    ):
        """Il filtro degli addestramenti — uno su cinque — qui farebbe sparire
        gli avvisi che escono una volta sola: «37 capolettera ricongiunti» non
        ha un secondo tentativo."""
        build = await _accoda(session, corpus.id)

        avvisi = [
            (0, "12 capolettera ricongiunti"),
            (3, "1 scartati a vista"),
            (50, "2 di 4 passaggi"),
            (100, "4 di 4 passaggi"),
        ]
        worker = WorkerBuild(
            "prova", coda=CodaFinta(),
            digestione=lambda s: DigestioneFinta(avvisi=avvisi),
        )
        await worker._esegui(_job(build))

        scritti = [
            (e.progress, e.message)
            for e in await BuildRepository(session).eventi(build.id)
        ]
        for avviso in avvisi:
            assert avviso in scritti, f"perso: {avviso}"
        assert [s for s in scritti if s in avvisi] == avvisi, "fuori ordine"

    async def test_una_base_sparita_e_un_fallimento_dichiarato(
        self, session, worker_sul_test,  # noqa: F811
    ):
        build = await _accoda(session, uuid.uuid4())

        worker = WorkerBuild(
            "prova", coda=CodaFinta(), digestione=lambda s: DigestioneFinta(),
        )
        await worker._esegui(_job(build))

        await session.refresh(build)
        assert build.status == "fallita"
        assert "non trovata" in build.error

    async def test_se_esplode_resta_scritto_fin_dove_era_arrivata(
        self, engine, monkeypatch,
    ):
        """Di un lavoro di tre ore morto al secondo non resta altro.

        Su connessioni vere e non sulla transazione del test: qui il worker
        scrive l'avanzamento da una sessione e la digestione fallisce da
        un'altra, e sulla stessa connessione il rollback dell'una invalida i
        punti di ripristino dell'altra — un artefatto dell'ambiente di prova
        che nasconderebbe proprio il comportamento da verificare.
        """
        factory = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(
            "platform_core.builds.worker.get_session_factory", lambda: factory,
        )

        async with factory() as s:
            kb = KnowledgeBase(
                slug=f"esplode-{uuid.uuid4().hex[:8]}", name="Corpus",
                kind="corpus", embed_model="prova",
            )
            s.add(kb)
            await s.flush()
            build = Build(
                personality_id=None, kind="digestione", status="in_coda",
                params={"kb_id": str(kb.id)},
            )
            s.add(build)
            await s.commit()
            build_id, kb_id = build.id, kb.id

        try:
            worker = WorkerBuild(
                "prova", coda=CodaFinta(),
                digestione=lambda _: DigestioneFinta(
                    avvisi=[(30, "30 di 100 passaggi")],
                    esplode="il provider ha smesso di rispondere",
                ),
            )
            await worker._esegui(_job(build))

            async with factory() as s:
                concluso = await s.get(Build, build_id)
                assert concluso.status == "fallita"
                assert "smesso di rispondere" in concluso.error
                messaggi = [
                    e.message
                    for e in await BuildRepository(s).eventi(build_id)
                ]
                assert "30 di 100 passaggi" in messaggi
        finally:
            async with factory() as s:
                await s.execute(delete(Build).where(Build.id == build_id))
                await s.execute(
                    delete(KnowledgeBase).where(KnowledgeBase.id == kb_id)
                )
                await s.commit()


class TestAvanzamento:
    def test_il_lotto_annuncia_solo_quando_la_percentuale_cambia(self):
        """Cinquemila passaggi a quattro per lotto sarebbero milleduecento
        righe conservate per muovere una barra da cento posizioni."""
        annunci = []
        annunciata = -1

        for fatti in range(4, 5004, 4):
            percentuale = int(fatti * 100 / 5000)
            if percentuale != annunciata:
                annunciata = percentuale
                annunci.append(percentuale)

        assert len(annunci) <= 101
        assert annunci[-1] == 100


class TestTipoDiLavoro:
    def test_digestione_e_un_tipo_ammesso(self):
        from platform_core.domain.build_models import TIPI_BUILD

        assert "digestione" in TIPI_BUILD

    def test_un_nodo_senza_acceleratore_puo_prenderla(self):
        """La digestione interroga un modello, non ne addestra uno: pretendere
        una GPU la renderebbe impossibile proprio dove serve, su un server."""
        import inspect

        from platform_core.builds.worker import avvia

        assert "solo_digestione" in inspect.signature(avvia).parameters


class TestMotoreAcceso:
    """La digestione interroga un modello: se non c'è, va acceso.

    Il test che conta è il secondo. Una digestione dura ore, e il timer di
    inattività gira mentre lavora: se il conteggio dei lavori in corso non
    salisse, il motore verrebbe spento sotto un lavoro vivo — e il guasto
    comparirebbe solo sui corpora grandi, cioè in produzione.
    """

    async def test_il_motore_si_accende_prima_di_digerire(
        self, session, worker_sul_test, corpus,  # noqa: F811
    ):
        from platform_core.llm.accensione import MotoreLocale

        lanciati: list[str] = []

        async def esecutore(comando, attesa):
            lanciati.append(comando)
            return 0, ""

        risposte = iter([False, True])

        async def sonda():
            return next(risposte, True)

        build = await _accoda(session, corpus.id)
        motore = MotoreLocale(
            avvio="accendi-bonsai", arresto="spegni-bonsai",
            esecutore=esecutore, sonda=sonda,
        )
        worker = WorkerBuild(
            "prova", coda=CodaFinta(),
            digestione=lambda s: DigestioneFinta(), motore=motore,
        )
        await worker._esegui(_job(build))

        await session.refresh(build)
        assert build.status == "riuscita"
        assert lanciati == ["accendi-bonsai"]

    async def test_resta_acceso_per_tutta_la_durata_del_lavoro(
        self, session, worker_sul_test, corpus,  # noqa: F811
    ):
        from platform_core.llm.accensione import MotoreLocale

        motore = MotoreLocale(
            avvio="accendi", arresto="spegni",
            esecutore=lambda c, a: _subito(),
            sonda=_sempre_pronto,
        )

        class DigestioneCheGuarda(DigestioneFinta):
            def __init__(self) -> None:
                super().__init__()
                self.in_corso_durante = None

            async def digerisci(self, kb, **kwargs):
                self.in_corso_durante = motore.stato().in_corso
                return await super().digerisci(kb, **kwargs)

        digestione = DigestioneCheGuarda()
        build = await _accoda(session, corpus.id)
        worker = WorkerBuild(
            "prova", coda=CodaFinta(),
            digestione=lambda s: digestione, motore=motore,
        )
        await worker._esegui(_job(build))

        assert digestione.in_corso_durante == 1, (
            "il lavoro non era contato: il timer di inattività avrebbe potuto "
            "spegnere il motore a metà digestione"
        )
        assert motore.stato().in_corso == 0, "il conteggio non è tornato a zero"

    async def test_senza_motore_il_lavoro_dichiara_perche(
        self, session, worker_sul_test, corpus,  # noqa: F811
    ):
        """«Errore inatteso» manderebbe a cercare un difetto nel corpus. Il
        corpus sta bene: è la macchina dei modelli che non risponde."""
        from platform_core.llm.accensione import MotoreLocale

        async def mai_pronto():
            return False

        motore = MotoreLocale(
            avvio="accendi", arresto="spegni", attesa_s=0.0,
            esecutore=lambda c, a: _subito(), sonda=mai_pronto,
        )
        build = await _accoda(session, corpus.id)
        worker = WorkerBuild(
            "prova", coda=CodaFinta(),
            digestione=lambda s: DigestioneFinta(), motore=motore,
        )
        await worker._esegui(_job(build))

        await session.refresh(build)
        assert build.status == "fallita"
        assert "motore locale non disponibile" in build.error


async def _subito():
    return 0, ""


async def _sempre_pronto() -> bool:
    return True
