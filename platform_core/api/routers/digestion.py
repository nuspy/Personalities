"""Digestione dei corpora, dalla console.

**Perché la digestione è un job e non una richiesta.** Su un corpus vero sono
ore: tenere aperta una connessione HTTP per tutto quel tempo significa che un
proxy la chiude a metà, e chi guardava non sa se il lavoro stia proseguendo o
sia morto. Si accoda come una build, e l'avanzamento si segue dagli eventi.

**Le etichette si ispezionano passaggio per passaggio.** È l'unico modo di
rispondere alla domanda che conta quando il recupero sbaglia: *questo
passaggio è classificato bene?* Un'aggregazione per categoria dice che il
corpus è distribuito in un certo modo e non dice mai se una singola
classificazione sia giusta.
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ...auth.dependencies import CurrentUser, DbSession, require_role
from ...builds.queue import CodaBuild, JobBuild
from ...builds.repository import BuildRepository
from ...domain.knowledge_models import Chunk, Document, KnowledgeBase
from ..deps import get_key_value_store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["digestione"],
    dependencies=[Depends(require_role("admin"))],
)


class AvviaDigestione(BaseModel):
    #: Di chi parla il corpus. Cambia il giudizio sulla provenienza: senza,
    #: «autore» e «posteriore» si distinguono solo dal tono, e il tono inganna.
    chi: str = Field(default="", max_length=200)
    #: Rianalizza anche i passaggi già etichettati. Serve quando cambia il
    #: classificatore o la tassonomia: senza, il corpus resta metà vecchio e
    #: metà nuovo, incoerente e impossibile da interpretare.
    rifai: bool = False


def _chunk_json(c: Chunk, titolo_documento: str) -> Dict[str, Any]:
    etichette = c.labels or {}
    return {
        "id": str(c.id),
        "ordinale": c.ordinal,
        "documento": titolo_documento,
        "sezione": c.section,
        "testo": c.text,
        # Quello di prima della pulizia, quando ce n'è stata una: è l'unico
        # modo di controllare che la rimozione non abbia portato via prosa.
        "testo_originale": c.text_original,
        "categoria": c.label_main,
        "categorie": etichette.get("categorie") or {},
        "provenienza": c.provenance,
        "qualita": c.quality,
        "sintesi": etichette.get("sintesi") or "",
        "da_togliere": etichette.get("da_togliere") or "",
        "scartato": c.discarded,
        "motivo_scarto": c.discard_reason,
    }


@router.get("/knowledge-bases/{kb_id}/digestion")
async def stato_digestione(
    kb_id: uuid.UUID, session: DbSession,
) -> Dict[str, Any]:
    """Quanto è stato digerito e come è distribuito."""
    kb = await session.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base non trovata")

    totale = await session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.kb_id == kb_id)
    )
    etichettati = await session.scalar(
        select(func.count()).select_from(Chunk).where(
            Chunk.kb_id == kb_id, Chunk.labels.is_not(None),
        )
    )
    scartati = await session.scalar(
        select(func.count()).select_from(Chunk).where(
            Chunk.kb_id == kb_id, Chunk.discarded.is_(True),
        )
    )
    ripuliti = await session.scalar(
        select(func.count()).select_from(Chunk).where(
            Chunk.kb_id == kb_id, Chunk.text_original.is_not(None),
        )
    )

    per_categoria = {
        categoria or "(nessuna)": int(quanti)
        for categoria, quanti in (await session.execute(
            select(Chunk.label_main, func.count())
            .where(Chunk.kb_id == kb_id, Chunk.discarded.is_(False))
            .group_by(Chunk.label_main)
            .order_by(func.count().desc())
        )).all()
    }
    per_provenienza = {
        provenienza or "(ignota)": int(quanti)
        for provenienza, quanti in (await session.execute(
            select(Chunk.provenance, func.count())
            .where(Chunk.kb_id == kb_id, Chunk.discarded.is_(False))
            .group_by(Chunk.provenance)
            .order_by(func.count().desc())
        )).all()
    }

    return {
        "kb": {"id": str(kb.id), "slug": kb.slug, "name": kb.name},
        "passaggi": {
            "totale": int(totale or 0),
            "etichettati": int(etichettati or 0),
            "da_fare": int((totale or 0) - (etichettati or 0)),
            "scartati": int(scartati or 0),
            "ripuliti": int(ripuliti or 0),
        },
        "per_categoria": per_categoria,
        "per_provenienza": per_provenienza,
    }


@router.get("/knowledge-bases/{kb_id}/chunks")
async def passaggi(
    kb_id: uuid.UUID,
    session: DbSession,
    categoria: Optional[str] = Query(default=None),
    provenienza: Optional[str] = Query(default=None),
    solo_scartati: bool = False,
    solo_ripuliti: bool = False,
    limite: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    """I passaggi con le loro etichette.

    I filtri servono alle due domande che ci si pone davvero guardando un
    corpus digerito: «cosa è stato buttato via?» e «cosa è stato toccato?».
    Scorrere tutto in ordine non risponde a nessuna delle due.
    """
    if await session.get(KnowledgeBase, kb_id) is None:
        raise HTTPException(status_code=404, detail="Knowledge base non trovata")

    query = (
        select(Chunk, Document.title)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.kb_id == kb_id)
    )

    if categoria:
        query = query.where(Chunk.label_main == categoria)
    if provenienza:
        query = query.where(Chunk.provenance == provenienza)
    if solo_scartati:
        query = query.where(Chunk.discarded.is_(True))
    if solo_ripuliti:
        query = query.where(Chunk.text_original.is_not(None))

    totale = await session.scalar(
        select(func.count()).select_from(query.subquery())
    )

    righe = (await session.execute(
        query.order_by(Chunk.ordinal).limit(limite).offset(offset)
    )).all()

    return {
        "totale": int(totale or 0),
        "passaggi": [_chunk_json(c, titolo) for c, titolo in righe],
    }


@router.post("/knowledge-bases/{kb_id}/digestion", status_code=202)
async def avvia_digestione(
    kb_id: uuid.UUID,
    payload: AvviaDigestione,
    session: DbSession,
    user: CurrentUser,
    supporto: Annotated[Any, Depends(get_key_value_store)],
) -> Dict[str, Any]:
    """Accoda la digestione di un corpus.

    Nessun controllo sulle capacità, a differenza delle build: la digestione
    interroga un modello ma non addestra nulla, e un provider remoto va bene
    quanto uno locale. Il worker CPU basta.
    """
    kb = await session.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base non trovata")

    da_fare = await session.scalar(
        select(func.count()).select_from(Chunk).where(
            Chunk.kb_id == kb_id,
            Chunk.labels.is_(None) if not payload.rifai else True,
        )
    )
    if not da_fare:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Non c'è nulla da digerire: tutti i passaggi sono già "
                "etichettati. Usa «rianalizza» per rifarli."
            ),
        )

    repo = BuildRepository(session)
    build = await repo.crea(
        # La digestione riusa la tabella delle realizzazioni: è un lavoro
        # lungo con avanzamento e un esito, esattamente come un addestramento,
        # e duplicare quel meccanismo significherebbe mantenerne due.
        personality_id=None,
        kind="digestione",
        params={
            "kb_id": str(kb_id),
            "kb_slug": kb.slug,
            "chi": payload.chi,
            "rifai": payload.rifai,
        },
        owner_id=user.id,
    )
    await session.commit()

    CodaBuild(supporto).accoda(JobBuild(
        build_id=build.id, kind="digestione", personality_id=uuid.UUID(int=0),
    ))

    return {
        "build_id": str(build.id),
        "passaggi_da_fare": int(da_fare),
        "stato": build.status,
    }
