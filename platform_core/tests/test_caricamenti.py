"""Caricare documenti dalla console.

Prima l'unica via era la riga di comando sul nodo, e il cliente non poteva
aggiornare le proprie basi senza chiedere a chi ha accesso al server. Le
promesse da tenere:

1. **si accoda, non si legge nella richiesta** — un PDF con l'OCR o un'ora di
   audio chiuderebbero la connessione a metà;
2. **i formati si controllano prima di scrivere** — un rifiuto a metà
   lascerebbe file orfani;
3. **il nome su disco non viene dall'utente** — `../` in un nome di file è un
   modo di scrivere dove non si dovrebbe;
4. **un file che non entra non ferma gli altri**, e il lavoro fallisce solo
   se non è entrato niente;
5. **un worker senza acceleratore non vede gli addestramenti** — con una coda
   sola li avrebbe presi e fatti fallire.
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from platform_core.api.deps import get_key_value_store
from platform_core.builds.queue import (
    CODA, CODA_SENZA_ACCELERATORE, CodaBuild, CodaInMemoria, JobBuild,
)
from platform_core.builds.worker import WorkerBuild
from platform_core.domain.models import AuditLog
from platform_core.domain.build_models import Build
from platform_core.domain.knowledge_models import Document, KnowledgeBase
from platform_core.knowledge.archivio import ArchivioSuDisco, FileTroppoGrande
from platform_core.knowledge.documenti import FormatoNonSupportato, estrai
from platform_core.knowledge.ingestione import IngestioneCaricamenti
from platform_core.settings import get_settings

from .conftest import richiede_database
from .test_admin import app_con, embedder  # noqa: F401
from .test_digestione_console import CodaFinta, worker_sul_test  # noqa: F401


# ---- senza database ------------------------------------------------------


class TestCode:
    def test_digestione_e_ingestione_vanno_nella_coda_senza_acceleratore(self):
        supporto = CodaInMemoria()
        coda = CodaBuild(supporto)
        for kind in ("digestione", "ingestione", "lora"):
            coda.accoda(JobBuild(build_id=uuid.uuid4(), kind=kind, personality_id=uuid.UUID(int=0)))

        assert supporto.llen(CODA_SENZA_ACCELERATORE) == 2
        assert supporto.llen(CODA) == 1

    def test_un_worker_senza_acceleratore_non_prende_un_addestramento(self):
        """Il difetto che c'era: una coda sola, e il worker CPU prendeva anche
        i LoRA — falliti uno dopo l'altro, la coda vuota, niente addestrato."""
        coda = CodaBuild(CodaInMemoria())
        coda.accoda(JobBuild(build_id=uuid.uuid4(), kind="lora", personality_id=uuid.uuid4()))

        preso = coda.prendi("cpu-1", attesa=0, code=(CODA_SENZA_ACCELERATORE,))

        assert preso is None
        assert coda.in_attesa() == 1

    def test_un_worker_con_acceleratore_prende_entrambe(self):
        coda = CodaBuild(CodaInMemoria())
        coda.accoda(JobBuild(build_id=uuid.uuid4(), kind="ingestione", personality_id=uuid.UUID(int=0)))
        coda.accoda(JobBuild(build_id=uuid.uuid4(), kind="lora", personality_id=uuid.uuid4()))

        tipi = {coda.prendi("gpu-1", attesa=0).kind, coda.prendi("gpu-1", attesa=0).kind}

        assert tipi == {"ingestione", "lora"}


class TestArchivio:
    async def _pezzi(self, *blocchi: bytes):
        for b in blocchi:
            yield b

    async def test_il_nome_su_disco_e_generato(self, tmp_path):
        archivio = ArchivioSuDisco(tmp_path, limite_byte=1024)
        kb = uuid.uuid4()

        riferimento, dimensione = await archivio.salva(kb, ".txt", self._pezzi(b"ciao"))

        assert riferimento.startswith(f"{kb}/")
        assert dimensione == 4
        assert archivio.percorso(riferimento).read_bytes() == b"ciao"

    async def test_un_riferimento_scritto_a_mano_non_si_segue(self, tmp_path):
        archivio = ArchivioSuDisco(tmp_path, limite_byte=1024)

        with pytest.raises(ValueError):
            archivio.percorso("../../etc/passwd")

    async def test_oltre_il_limite_non_resta_niente(self, tmp_path):
        """Il limite si controlla mentre si scrive: un file enorme
        riempirebbe il volume prima di essere rifiutato."""
        archivio = ArchivioSuDisco(tmp_path, limite_byte=10)

        with pytest.raises(FileTroppoGrande):
            await archivio.salva(uuid.uuid4(), ".txt", self._pezzi(b"x" * 8, b"x" * 8))

        assert not any(p.is_file() for p in tmp_path.rglob("*"))


class TestEstrazione:
    def test_un_testo_diventa_un_documento_scritto(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("Non è che abbiamo poco tempo, è che ne perdiamo molto.", encoding="utf-8")

        estratto = estrai(f, nome="Sulla brevità della vita.txt")

        assert estratto.titolo == "Sulla brevità della vita"
        assert "perdiamo molto" in estratto.testo
        assert estratto.meta["registro"] == "scritto"

    def test_il_decoder_si_sceglie_dal_nome_originale(self, tmp_path):
        """Su disco il file ha un nome generato: l'estensione che conta è
        quella di cui risponde chi carica."""
        f = tmp_path / "0f3a.bin"
        f.write_text("# Lettera\n\nLa virtù si esercita.", encoding="utf-8")

        estratto = estrai(f, nome="lettera.md")

        assert "virtù" in estratto.testo

    def test_un_file_vuoto_non_entra_in_silenzio(self, tmp_path):
        f = tmp_path / "vuoto.txt"
        f.write_text("   ", encoding="utf-8")

        with pytest.raises(FormatoNonSupportato):
            estrai(f, nome="vuoto.txt")

    def test_l_audio_senza_trascrittore_dice_perche(self, tmp_path):
        from platform_core.knowledge.transcription import TrascrizioneNonDisponibile

        f = tmp_path / "x.mp3"
        f.write_bytes(b"ID3")

        with pytest.raises(TrascrizioneNonDisponibile):
            estrai(f, nome="intervista.mp3")


# ---- con database --------------------------------------------------------


@pytest.fixture
def cartella(tmp_path, monkeypatch):
    """I caricamenti del test in una cartella del test, non in data/."""
    monkeypatch.setenv("PERSONA_UPLOAD_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    monkeypatch.undo()
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def coda() -> CodaInMemoria:
    return CodaInMemoria()


@pytest_asyncio.fixture
async def admin(session, session_factory, principal_admin, embedder, utente, coda):  # noqa: F811
    app = app_con(principal_admin, session, session_factory, embedder)
    app.dependency_overrides[get_key_value_store] = lambda: coda
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def base(session, utente, embedder) -> KnowledgeBase:  # noqa: F811
    kb = KnowledgeBase(
        slug=f"caricata-{uuid.uuid4().hex[:8]}", name="Caricata", kind="corpus",
        embed_model=embedder.modello, owner_id=utente.id, text_config="italian",
    )
    session.add(kb)
    await session.flush()
    await session.commit()
    return kb


TESTO = (
    "Non è che abbiamo poco tempo, è che ne perdiamo molto. La vita è "
    "abbastanza lunga, e ci è stata data con larghezza per compiere grandi "
    "cose, se fosse tutta ben spesa."
)


@richiede_database
class TestCaricamento:
    async def test_si_accoda_e_non_si_legge(self, admin, base, session, coda, cartella):
        r = await admin.post(
            f"/admin/knowledge-bases/{base.id}/uploads",
            files=[("file", ("Brevità.txt", TESTO.encode(), "text/plain"))],
        )

        assert r.status_code == 202, r.text
        corpo = r.json()
        assert corpo["file"] == [{"nome": "Brevità.txt", "byte": len(TESTO.encode())}]
        assert corpo["lingua"] == "it", "la lingua si propone da quella della base"

        build = await session.get(Build, uuid.UUID(corpo["build_id"]))
        assert build.kind == "ingestione"
        assert build.message == "In attesa di un worker", "non serve una GPU"
        assert CodaBuild(coda).in_attesa(code=(CODA_SENZA_ACCELERATORE,)) == 1
        # Nessun documento ancora: lo aggiunge il worker.
        assert not (await session.execute(
            select(Document).where(Document.kb_id == base.id)
        )).scalars().first()

    async def test_il_nome_non_decide_dove_si_scrive(self, admin, base, session, cartella):
        r = await admin.post(
            f"/admin/knowledge-bases/{base.id}/uploads",
            files=[("file", ("../../fuori.txt", TESTO.encode(), "text/plain"))],
        )

        assert r.status_code == 202
        assert r.json()["file"][0]["nome"] == "fuori.txt"
        assert not (cartella.parent / "fuori.txt").exists()
        scritti = [p for p in cartella.rglob("*") if p.is_file()]
        assert len(scritti) == 1 and scritti[0].parent.name == str(base.id)

    async def test_un_formato_sconosciuto_non_scrive_nulla(self, admin, base, cartella):
        r = await admin.post(
            f"/admin/knowledge-bases/{base.id}/uploads",
            files=[
                ("file", ("buono.txt", TESTO.encode(), "text/plain")),
                ("file", ("eseguibile.exe", b"MZ", "application/octet-stream")),
            ],
        )

        assert r.status_code == 415
        assert "eseguibile.exe" in r.json()["detail"]
        assert not any(p.is_file() for p in cartella.rglob("*"))

    async def test_oltre_il_limite_e_un_413_e_non_resta_niente(
        self, admin, base, cartella, monkeypatch,
    ):
        monkeypatch.setenv("PERSONA_UPLOAD_MAX_MB", "1")
        get_settings.cache_clear()

        r = await admin.post(
            f"/admin/knowledge-bases/{base.id}/uploads",
            files=[
                ("file", ("piccolo.txt", TESTO.encode(), "text/plain")),
                ("file", ("grande.txt", b"x" * (1024 * 1024 + 1), "text/plain")),
            ],
        )

        assert r.status_code == 413
        assert "grande.txt" in r.json()["detail"]
        assert not any(p.is_file() for p in cartella.rglob("*"))

    async def test_il_caricamento_resta_nel_registro(self, admin, base, session, cartella):
        await admin.post(
            f"/admin/knowledge-bases/{base.id}/uploads",
            files=[("file", ("Brevità.txt", TESTO.encode(), "text/plain"))],
        )

        voce = (await session.execute(
            select(AuditLog).where(AuditLog.action == "kb.documenti_caricati")
        )).scalars().first()
        assert voce is not None
        assert voce.after["file"][0]["nome"] == "Brevità.txt"

    async def test_i_formati_si_leggono_dalla_console(self, admin):
        r = await admin.get("/admin/ingestion/formats")

        assert r.status_code == 200
        assert ".pdf" in r.json()["formati"]
        assert ".mp3" in r.json()["formati"]

    async def test_un_utente_qualunque_non_carica(
        self, session, session_factory, principal_utente, embedder, utente, base,  # noqa: F811
    ):
        app = app_con(principal_utente, session, session_factory, embedder)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                f"/admin/knowledge-bases/{base.id}/uploads",
                files=[("file", ("a.txt", b"testo", "text/plain"))],
            )
        assert r.status_code == 403


@richiede_database
class TestEsecuzione:
    async def _carica(self, admin, base, *file):
        r = await admin.post(
            f"/admin/knowledge-bases/{base.id}/uploads",
            files=[("file", f) for f in file],
        )
        assert r.status_code == 202, r.text
        return uuid.UUID(r.json()["build_id"])

    def _worker(self, embedder):  # noqa: F811
        return WorkerBuild(
            "prova", coda=CodaFinta(),
            ingestione=lambda s: IngestioneCaricamenti(s, embedder),
        )

    async def test_il_worker_aggiunge_il_documento_e_toglie_il_file(
        self, admin, base, session, embedder, cartella, worker_sul_test,  # noqa: F811
    ):
        build_id = await self._carica(admin, base, ("Brevità.txt", TESTO.encode(), "text/plain"))

        await self._worker(embedder)._esegui(
            JobBuild(build_id=build_id, kind="ingestione", personality_id=uuid.UUID(int=0)),
        )

        build = await session.get(Build, build_id)
        await session.refresh(build)
        assert build.status == "riuscita", build.error
        assert build.artifact_meta["documenti"] == 1
        documento = (await session.execute(
            select(Document).where(Document.kb_id == base.id)
        )).scalars().one()
        assert documento.title == "Brevità"
        assert documento.uri == "caricamento:Brevità.txt"
        assert documento.meta["origine"] == "console"
        assert not any(p.is_file() for p in cartella.rglob("*")), "il file caricato resta su disco"

    async def test_un_file_che_non_entra_non_ferma_gli_altri(
        self, admin, base, session, embedder, cartella, worker_sul_test,  # noqa: F811
    ):
        build_id = await self._carica(
            admin, base,
            ("vuoto.txt", b"   ", "text/plain"),
            ("Brevità.txt", TESTO.encode(), "text/plain"),
        )

        await self._worker(embedder)._esegui(
            JobBuild(build_id=build_id, kind="ingestione", personality_id=uuid.UUID(int=0)),
        )

        build = await session.get(Build, build_id)
        await session.refresh(build)
        assert build.status == "riuscita"
        assert build.artifact_meta["documenti"] == 1
        assert [f["nome"] for f in build.artifact_meta["falliti"]] == ["vuoto.txt"]

    async def test_se_non_entra_niente_il_lavoro_fallisce(
        self, admin, base, session, embedder, cartella, worker_sul_test,  # noqa: F811
    ):
        """Un successo con zero documenti direbbe «fatto» a chi ha caricato un
        PDF scansionato che non è entrato."""
        build_id = await self._carica(admin, base, ("vuoto.txt", b"   ", "text/plain"))

        await self._worker(embedder)._esegui(
            JobBuild(build_id=build_id, kind="ingestione", personality_id=uuid.UUID(int=0)),
        )

        build = await session.get(Build, build_id)
        await session.refresh(build)
        assert build.status == "fallita"
        assert "vuoto.txt" in build.error

    async def test_lo_stesso_testo_due_volte_non_raddoppia(
        self, admin, base, session, embedder, cartella, worker_sul_test,  # noqa: F811
    ):
        for _ in range(2):
            build_id = await self._carica(admin, base, ("Brevità.txt", TESTO.encode(), "text/plain"))
            await self._worker(embedder)._esegui(
                JobBuild(build_id=build_id, kind="ingestione", personality_id=uuid.UUID(int=0)),
            )

        build = await session.get(Build, build_id)
        await session.refresh(build)
        assert build.artifact_meta["saltati"][0]["nome"] == "Brevità.txt"
        documenti = (await session.execute(
            select(Document).where(Document.kb_id == base.id)
        )).scalars().all()
        assert len(documenti) == 1
