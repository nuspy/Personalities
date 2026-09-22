"""Caricare documenti in un corpus dalla console.

La richiesta salva i file e accoda un lavoro; non legge né indicizza nulla.
Un PDF da mille pagine con l'OCR o un'ora di audio da trascrivere sono
minuti di lavoro, e una richiesta HTTP tenuta aperta tanto viene chiusa da un
proxy a metà — lasciando chi guardava senza sapere se il file sia entrato.
Il lavoro si segue come le digestioni, dagli eventi della build.
"""
from __future__ import annotations

import logging
import pathlib
import re
import uuid
from typing import Annotated, Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ...auth.dependencies import DbSession, require_role
from ...builds.queue import CodaBuild, JobBuild
from ...builds.repository import BuildRepository
from ...domain.admin_repositories import AdminKnowledgeRepository
from ...knowledge.archivio import FileTroppoGrande, archivio_caricamenti
from ...knowledge.documenti import e_supportato, formati_supportati
from ...knowledge.indexer import CONFIGURAZIONI_TESTO
from ...settings import get_settings
from ..deps import get_key_value_store
from .admin import Contesto

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["caricamenti"],
    dependencies=[Depends(require_role("admin"))],
)

#: Quanti file per volta. Oltre, la console diventa uno strumento di
#: migrazione di massa, e per quello c'è la riga di comando sul nodo.
FILE_MASSIMI = 20

#: Da configurazione di ricerca testuale a lingua, per proporre quella della
#: base quando chi carica non la indica.
_LINGUA_DI = {v: k for k, v in CONFIGURAZIONI_TESTO.items() if v != "simple"}

_CONTROLLO = re.compile(r"[\x00-\x1f\x7f]")


def nome_leggibile(grezzo: Optional[str]) -> str:
    """Il nome da mostrare: senza cartelle, senza caratteri di controllo.

    Serve solo a mostrarlo e a scegliere il decoder — su disco il file ha un
    nome generato — ma un nome con `\\n` dentro spezzerebbe i log, e uno con
    le cartelle di chi carica ne rivelerebbe la struttura del disco.
    """
    nome = (grezzo or "").replace("\\", "/").rsplit("/", 1)[-1]
    nome = _CONTROLLO.sub("", nome).strip()
    return nome[:200] or "senza nome"


@router.get("/ingestion/formats")
async def formati() -> Dict[str, Any]:
    """Cosa si può caricare, e quanto grande."""
    return {
        "formati": formati_supportati(),
        "byte_massimi": get_settings().upload_max_mb * 1024 * 1024,
        "file_massimi": FILE_MASSIMI,
    }


async def _pezzi(file: UploadFile) -> AsyncIterator[bytes]:
    while pezzo := await file.read(1024 * 1024):
        yield pezzo


async def accoda_ingestione(
    session,
    supporto,
    kb,
    file: List[UploadFile],
    *,
    lingua: Optional[str],
    attore_id: int,
    registra,
) -> Dict[str, Any]:
    """Salva i file, crea il lavoro, lo accoda. Comune a console e utenti.

    `registra` scrive la riga di audit: la console passa dal suo repository,
    l'utente sui propri corpora da quello generico — la riga è la stessa,
    cambia chi la firma.
    """
    if len(file) > FILE_MASSIMI:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Al massimo {FILE_MASSIMI} file per volta.",
        )

    nomi = [nome_leggibile(f.filename) for f in file]
    # Tutti i formati si controllano prima di scrivere il primo byte: un
    # rifiuto a metà lascerebbe su disco i file già salvati, e chi carica
    # non saprebbe quali sono entrati.
    rifiutati = [n for n in nomi if not e_supportato(n)]
    if rifiutati:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Formato non supportato: {', '.join(rifiutati)}. Si caricano "
                f"{', '.join(sorted(formati_supportati()))}."
            ),
        )

    archivio = archivio_caricamenti()
    salvati: List[Dict[str, Any]] = []
    nome = ""
    try:
        for caricato, nome in zip(file, nomi):
            riferimento, dimensione = await archivio.salva(
                kb.id, pathlib.Path(nome).suffix, _pezzi(caricato),
            )
            salvati.append({
                "riferimento": riferimento, "nome": nome, "byte": dimensione,
                "da": attore_id,
            })
    except FileTroppoGrande as exc:
        for voce in salvati:
            archivio.elimina(voce["riferimento"])
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"«{nome}» supera {exc.limite_byte // (1024 * 1024)} MB. Per "
                f"file così grandi c'è l'ingestione da riga di comando sul nodo."
            ),
        ) from exc
    except BaseException:
        for voce in salvati:
            archivio.elimina(voce["riferimento"])
        raise

    if lingua is None:
        lingua = _LINGUA_DI.get(kb.text_config)

    build = await BuildRepository(session).crea(
        personality_id=None,
        kind="ingestione",
        params={
            "kb_id": str(kb.id),
            "kb_slug": kb.slug,
            "lingua": lingua,
            "file": salvati,
        },
        owner_id=attore_id,
    )
    await registra(build.id, [{"nome": v["nome"], "byte": v["byte"]} for v in salvati])
    await session.commit()

    CodaBuild(supporto).accoda(JobBuild(
        build_id=build.id, kind="ingestione", personality_id=uuid.UUID(int=0),
    ))
    logger.info(
        "%d file caricati in %s da %s (build %s)",
        len(salvati), kb.slug, attore_id, str(build.id)[:8],
    )

    return {
        "build_id": str(build.id),
        "stato": build.status,
        "lingua": lingua,
        "file": [{"nome": v["nome"], "byte": v["byte"]} for v in salvati],
    }


@router.post("/knowledge-bases/{kb_id}/uploads", status_code=status.HTTP_202_ACCEPTED)
async def carica(
    kb_id: uuid.UUID,
    session: DbSession,
    ctx: Contesto,
    supporto: Annotated[Any, Depends(get_key_value_store)],
    file: List[UploadFile] = File(...),
    lingua: Optional[str] = Form(default=None, pattern=r"^[a-z]{2}$"),
) -> Dict[str, Any]:
    repo = AdminKnowledgeRepository(session, ctx)
    kb = await repo.per_id(kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base non trovata")

    async def registra(build_id, voci):
        await repo.registra_caricamento(kb, build_id=build_id, file=voci)

    return await accoda_ingestione(
        session, supporto, kb, file,
        lingua=lingua, attore_id=ctx.attore.id, registra=registra,
    )
