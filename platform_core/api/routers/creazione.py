"""Le personalità e i corpora di chi li crea da sé.

Tutto sotto `/me`: ogni rotta risponde solo delle cose di chi chiede, e ciò che
appartiene ad altri è «non trovato» — mai «vietato», che confermerebbe che
esiste.
"""
from __future__ import annotations

import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ...auth.dependencies import CurrentUser, DbSession
from ...creazione import PROMPT_MASSIMO, Creazioni, NonConsentito
from ...domain.build_models import Build
from ...domain.knowledge_models import Chunk, Document, KnowledgeBase, Personality
from ...domain.repositories import AuditRepository
from ...knowledge.embedding import Embedder
from ...observability.correlation import current_correlation_id
from ..deps import get_embedder, get_key_value_store
from .caricamenti import accoda_ingestione

router = APIRouter(prefix="/me", tags=["creazione"])


class NuovaPersonalita(BaseModel):
    nome: str = Field(min_length=2, max_length=120)
    descrizione: Optional[str] = Field(default=None, max_length=500)
    prompt: str = Field(min_length=20, max_length=PROMPT_MASSIMO)


class ModificaPersonalita(BaseModel):
    nome: Optional[str] = Field(default=None, min_length=2, max_length=120)
    descrizione: Optional[str] = Field(default=None, max_length=500)
    prompt: Optional[str] = Field(default=None, min_length=20, max_length=PROMPT_MASSIMO)


class NuovoCorpus(BaseModel):
    nome: str = Field(min_length=2, max_length=120)


class Collegamenti(BaseModel):
    kb_ids: List[uuid.UUID] = Field(default_factory=list, max_length=20)


def _rifiuto(exc: NonConsentito) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN if exc.serve_piano else status.HTTP_404_NOT_FOUND,
        detail=exc.messaggio,
    )


async def _personalita_json(creazioni: Creazioni, session, p: Personality) -> Dict[str, Any]:
    from ...domain.knowledge_models import PersonalityVersion

    versione = await session.get(PersonalityVersion, p.current_version_id) if p.current_version_id else None
    return {
        "id": str(p.id),
        "slug": p.slug,
        "nome": p.display_name,
        "descrizione": p.description,
        "versione": versione.version if versione else None,
        "prompt": versione.system_prompt if versione else "",
        "corpora": [str(k) for k in await creazioni.collegati(p)],
    }


async def _corpus_json(session, kb: KnowledgeBase) -> Dict[str, Any]:
    documenti = await session.scalar(
        select(func.count()).select_from(Document).where(Document.kb_id == kb.id)
    )
    return {
        "id": str(kb.id),
        "nome": kb.name,
        "documenti": int(documenti or 0),
        "passaggi": int((kb.stats or {}).get("passaggi", 0)),
    }


@router.get("/ingestion/formats")
async def formati_caricabili() -> Dict[str, Any]:
    """Cosa si può caricare nei propri corpora: gli stessi formati della console."""
    from ...knowledge.documenti import formati_supportati
    from ...settings import get_settings
    from .caricamenti import FILE_MASSIMI

    return {
        "formati": formati_supportati(),
        "byte_massimi": get_settings().upload_max_mb * 1024 * 1024,
        "file_massimi": FILE_MASSIMI,
    }


@router.get("/creations")
async def le_mie_creazioni(user: CurrentUser, session: DbSession) -> Dict[str, Any]:
    creazioni = Creazioni(session, user.id)
    return {
        "permessi": (await creazioni.permessi()).to_dict(),
        "personalita": [await _personalita_json(creazioni, session, p) for p in await creazioni.personalita()],
        "corpora": [await _corpus_json(session, kb) for kb in await creazioni.corpora()],
    }


@router.post("/personalities", status_code=status.HTTP_201_CREATED)
async def crea_personalita(
    payload: NuovaPersonalita, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    creazioni = Creazioni(session, user.id)
    try:
        personalita, _ = await creazioni.crea_personalita(
            nome=payload.nome, descrizione=payload.descrizione, prompt=payload.prompt,
        )
    except NonConsentito as exc:
        raise _rifiuto(exc) from exc
    await session.commit()
    return await _personalita_json(creazioni, session, personalita)


async def _sua(creazioni: Creazioni, personality_id: uuid.UUID) -> Personality:
    p = await creazioni.sua_personalita(personality_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Personalità non trovata")
    return p


@router.put("/personalities/{personality_id}")
async def modifica_personalita(
    personality_id: uuid.UUID, payload: ModificaPersonalita, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    creazioni = Creazioni(session, user.id)
    p = await _sua(creazioni, personality_id)
    await creazioni.modifica_personalita(
        p, nome=payload.nome, descrizione=payload.descrizione, prompt=payload.prompt,
    )
    await session.commit()
    return await _personalita_json(creazioni, session, p)


@router.delete("/personalities/{personality_id}")
async def archivia_personalita(
    personality_id: uuid.UUID, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    creazioni = Creazioni(session, user.id)
    await creazioni.archivia(await _sua(creazioni, personality_id))
    await session.commit()
    return {"archiviata": True}


@router.put("/personalities/{personality_id}/corpora")
async def collega_corpora(
    personality_id: uuid.UUID, payload: Collegamenti, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    creazioni = Creazioni(session, user.id)
    p = await _sua(creazioni, personality_id)
    try:
        await creazioni.collega(p, payload.kb_ids)
    except NonConsentito as exc:
        raise _rifiuto(exc) from exc
    await session.commit()
    return await _personalita_json(creazioni, session, p)


@router.post("/corpora", status_code=status.HTTP_201_CREATED)
async def crea_corpus(
    payload: NuovoCorpus,
    user: CurrentUser,
    session: DbSession,
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> Dict[str, Any]:
    try:
        kb = await Creazioni(session, user.id).crea_corpus(nome=payload.nome, embed_model=embedder.modello)
    except NonConsentito as exc:
        raise _rifiuto(exc) from exc
    await session.commit()
    return await _corpus_json(session, kb)


async def _suo_corpus(session, user, kb_id: uuid.UUID) -> KnowledgeBase:
    kb = await Creazioni(session, user.id).suo_corpus(kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Corpus non trovato")
    return kb


@router.get("/corpora/{kb_id}/documents")
async def documenti_del_corpus(kb_id: uuid.UUID, user: CurrentUser, session: DbSession) -> List[Dict[str, Any]]:
    kb = await _suo_corpus(session, user, kb_id)
    documenti = (await session.execute(
        select(Document).where(Document.kb_id == kb.id).order_by(Document.title)
    )).scalars().all()
    passaggi = dict((await session.execute(
        select(Chunk.document_id, func.count()).where(Chunk.kb_id == kb.id).group_by(Chunk.document_id)
    )).all())
    return [
        {
            "id": str(d.id), "titolo": d.title,
            "file": (d.meta or {}).get("file"), "registro": (d.meta or {}).get("registro"),
            "passaggi": int(passaggi.get(d.id, 0)),
        }
        for d in documenti
    ]


@router.delete("/corpora/{kb_id}/documents/{document_id}")
async def elimina_documento(
    kb_id: uuid.UUID, document_id: uuid.UUID, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    kb = await _suo_corpus(session, user, kb_id)
    documento = await session.get(Document, document_id)
    if documento is None or documento.kb_id != kb.id:
        raise HTTPException(status_code=404, detail="Documento non trovato")
    await session.delete(documento)
    await session.commit()
    return {"eliminato": True}


@router.post("/corpora/{kb_id}/uploads", status_code=status.HTTP_202_ACCEPTED)
async def carica_nel_corpus(
    kb_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    supporto: Annotated[Any, Depends(get_key_value_store)],
    file: List[UploadFile] = File(...),
    lingua: Optional[str] = Form(default=None, pattern=r"^[a-z]{2}$"),
) -> Dict[str, Any]:
    kb = await _suo_corpus(session, user, kb_id)
    # Il diritto si ricontrolla qui e non solo alla creazione: chi è sceso a
    # un piano senza corpora propri li conserva, ma non ci aggiunge altro.
    if not (await Creazioni(session, user.id).permessi()).corpora:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Il tuo piano non comprende corpora propri.",
        )

    async def registra(build_id, voci):
        await AuditRepository(session).record(
            action="kb.documenti_caricati", actor_id=user.id,
            target_type="knowledge_base", target_id=str(kb.id),
            after={"build_id": str(build_id), "file": voci},
            correlation_id=current_correlation_id(),
        )

    return await accoda_ingestione(
        session, supporto, kb, file, lingua=lingua, attore_id=user.id, registra=registra,
    )


@router.get("/jobs/{build_id}")
async def stato_lavoro(build_id: uuid.UUID, user: CurrentUser, session: DbSession) -> Dict[str, Any]:
    """Com'è andato un caricamento. Solo i propri lavori."""
    build = await session.get(Build, build_id)
    if build is None or build.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Lavoro non trovato")
    return {
        "id": str(build.id),
        "stato": build.status,
        "avanzamento": build.progress,
        "messaggio": build.message,
        "errore": build.error,
        "esito": build.artifact_meta,
    }
