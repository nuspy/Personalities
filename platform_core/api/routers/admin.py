"""Console di amministrazione.

Ogni rotta qui pretende il ruolo `admin`, dichiarato una volta sul router e
non endpoint per endpoint: un controllo ripetuto è un controllo che prima o
poi si dimentica in uno dei punti, e quello diventa la porta aperta.

**L'endpoint che conta di più è `/admin/retrieval/preview`.** Quando una
risposta è sbagliata, la domanda è sempre la stessa — il recupero non ha
trovato ciò che serviva, o l'ha trovato e scartato? Senza un modo di
interrogare il recupero da solo, quella domanda si risponde per tentativi:
si cambia il prompt, si riprova, e si conclude qualcosa sulla base di una
variabile diversa da quella che si stava studiando. Qui si vedono i punteggi
di entrambe le ricerche, la posizione in ciascuna, e ciò che è rimasto fuori.
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ...auth.dependencies import CurrentUser, DbSession, require_role
from ...avatar.engine import ConfigurazioneAvatarNonValida
from ...avatar.engine import descrivi as descrivi_avatar
from ...domain.admin_repositories import (
    AvatarRepository,
    AdminKnowledgeRepository, AdminPersonalityRepository,
    ContestoAmministrativo, RegistroAuditRepository, TassonomiaRepository,
)
from ...domain.knowledge_models import Personality
from ...knowledge.embedding import Embedder
from ...knowledge.retriever import Retriever
from ...observability.correlation import current_correlation_id
from ..deps import get_capability_registry, get_embedder
from ...capabilities.registry import CapabilityRegistry, Feature

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["amministrazione"],
    # Il ruolo si pretende sul router: elencarlo su ogni rotta significa
    # dimenticarlo su quella aggiunta di fretta.
    dependencies=[Depends(require_role("admin"))],
)


# --- forme delle richieste -------------------------------------------------


class CreaPersonalita(BaseModel):
    slug: str = Field(min_length=2, max_length=80, pattern=r"^[a-z0-9][a-z0-9-]*$")
    display_name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    commercial_category_id: Optional[int] = None


class ModificaPersonalita(BaseModel):
    display_name: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = None
    commercial_category_id: Optional[int] = None


class CreaVersione(BaseModel):
    system_prompt: str = Field(min_length=10)
    behavior_rules: Optional[Dict[str, Any]] = None
    llm_config: Optional[Dict[str, Any]] = None
    rag_config: Optional[Dict[str, Any]] = None
    guard_config: Optional[Dict[str, Any]] = None
    pubblica: bool = False


class CollegaCorpus(BaseModel):
    kb_id: uuid.UUID
    role: str = Field(default="knowledge", pattern=r"^(voice|knowledge)$")
    max_chunks: int = Field(default=6, ge=1, le=30)
    enabled: bool = True


class CreaBase(BaseModel):
    slug: str = Field(min_length=2, max_length=80, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    kind: str = Field(default="corpus", pattern=r"^(corpus|reference|news)$")
    visibility: str = Field(default="private", pattern=r"^(private|shared|public)$")


class ProvaRecupero(BaseModel):
    domanda: str = Field(min_length=1, max_length=2000)
    kb_ids: List[uuid.UUID] = Field(min_length=1)
    limite: int = Field(default=6, ge=1, le=20)


class CreaTipo(BaseModel):
    slug: str = Field(min_length=2, max_length=60, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1, max_length=120)
    description: Optional[str] = None


class CreaAvatar(BaseModel):
    slug: str = Field(min_length=2, max_length=80, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1, max_length=160)
    kind: str = Field(pattern=r"^(immagine|video|modello)$")
    config: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class ModificaAvatar(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=160)
    kind: Optional[str] = Field(default=None, pattern=r"^(immagine|video|modello)$")
    config: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class VolgiAvatar(BaseModel):
    """Quale volto indossa una personalità. `null` lo toglie."""

    avatar_id: Optional[uuid.UUID] = None


class CreaCategoria(BaseModel):
    slug: str = Field(min_length=2, max_length=40, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1, max_length=80)
    rank: int = Field(ge=0, le=100)


# --- contesto --------------------------------------------------------------


def contesto(request: Request, user: CurrentUser) -> ContestoAmministrativo:
    """Chi agisce, da dove, dentro quale richiesta."""
    return ContestoAmministrativo(
        user,
        ip=request.client.host if request.client else None,
        correlation_id=current_correlation_id(),
    )


Contesto = Annotated[ContestoAmministrativo, Depends(contesto)]


def _personalita_json(p: Personality) -> Dict[str, Any]:
    return {
        "id": str(p.id),
        "slug": p.slug,
        "display_name": p.display_name,
        "description": p.description,
        "status": p.status,
        # Le private sono degli utenti che le hanno create: la console le
        # vede — risponde anche di quelle — ma non le mette nel catalogo.
        "visibility": p.visibility,
        "current_version_id": str(p.current_version_id) if p.current_version_id else None,
        "avatar_id": str(p.avatar_id) if p.avatar_id else None,
        "tipi": [{"id": t.id, "slug": t.slug, "name": t.name} for t in p.types],
        "categoria": (
            {"id": p.commercial_category.id, "name": p.commercial_category.name}
            if p.commercial_category else None
        ),
    }


# --- personalità -----------------------------------------------------------


@router.get("/personalities")
async def elenco_personalita(
    session: DbSession, ctx: Contesto, includi_bozze: bool = True,
) -> List[Dict[str, Any]]:
    repo = AdminPersonalityRepository(session, ctx)
    return [
        _personalita_json(p)
        for p in await repo.elenco(includi_bozze=includi_bozze)
    ]


@router.post("/personalities", status_code=status.HTTP_201_CREATED)
async def crea_personalita(
    payload: CreaPersonalita, session: DbSession, ctx: Contesto,
) -> Dict[str, Any]:
    repo = AdminPersonalityRepository(session, ctx)

    if await repo.per_slug(payload.slug) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Esiste già una personalità con lo slug '{payload.slug}'",
        )

    personalita = await repo.crea(**payload.model_dump())
    await session.commit()
    return _personalita_json(personalita)


@router.get("/personalities/{personality_id}")
async def dettaglio_personalita(
    personality_id: uuid.UUID, session: DbSession, ctx: Contesto,
) -> Dict[str, Any]:
    repo = AdminPersonalityRepository(session, ctx)
    personalita = await _o_404(repo.per_id(personality_id))

    versioni = await repo.versioni(personalita)
    corpora = await repo.corpora(personalita)

    return {
        **_personalita_json(personalita),
        "versioni": [
            {
                "id": str(v.id),
                "version": v.version,
                "published_at": v.published_at.isoformat() if v.published_at else None,
                "corrente": v.id == personalita.current_version_id,
                "system_prompt": v.system_prompt,
                "behavior_rules": v.behavior_rules,
                "llm_config": v.llm_config,
                "rag_config": v.rag_config,
                "guard_config": v.guard_config,
            }
            for v in versioni
        ],
        "corpora": [
            {
                "kb_id": str(c.kb_id), "role": c.role,
                "max_chunks": c.max_chunks, "enabled": c.enabled,
            }
            for c in corpora
        ],
    }


@router.patch("/personalities/{personality_id}")
async def modifica_personalita(
    personality_id: uuid.UUID,
    payload: ModificaPersonalita,
    session: DbSession,
    ctx: Contesto,
) -> Dict[str, Any]:
    repo = AdminPersonalityRepository(session, ctx)
    personalita = await _o_404(repo.per_id(personality_id))

    await repo.aggiorna(personalita, **payload.model_dump(exclude_unset=True))
    await session.commit()
    return _personalita_json(personalita)


@router.post("/personalities/{personality_id}/versions", status_code=201)
async def crea_versione(
    personality_id: uuid.UUID,
    payload: CreaVersione,
    session: DbSession,
    ctx: Contesto,
) -> Dict[str, Any]:
    """Crea una versione nuova. Le precedenti restano intatte."""
    repo = AdminPersonalityRepository(session, ctx)
    personalita = await _o_404(repo.per_id(personality_id))

    versione = await repo.nuova_versione(personalita, **payload.model_dump())
    await session.commit()

    return {
        "id": str(versione.id),
        "version": versione.version,
        "pubblicata": versione.id == personalita.current_version_id,
    }


@router.post("/personalities/{personality_id}/versions/{version_id}/publish")
async def pubblica_versione(
    personality_id: uuid.UUID,
    version_id: uuid.UUID,
    session: DbSession,
    ctx: Contesto,
) -> Dict[str, Any]:
    from ...domain.repositories import PersonalityRepository

    repo = AdminPersonalityRepository(session, ctx)
    personalita = await _o_404(repo.per_id(personality_id))
    versione = await PersonalityRepository(session).versione(version_id)

    if versione is None:
        raise HTTPException(status_code=404, detail="Versione non trovata")

    try:
        await repo.pubblica(personalita, versione)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await session.commit()
    return _personalita_json(personalita)


@router.post("/personalities/{personality_id}/archive")
async def archivia_personalita(
    personality_id: uuid.UUID, session: DbSession, ctx: Contesto,
) -> Dict[str, Any]:
    repo = AdminPersonalityRepository(session, ctx)
    personalita = await _o_404(repo.per_id(personality_id))

    await repo.archivia(personalita)
    await session.commit()
    return _personalita_json(personalita)


@router.put("/personalities/{personality_id}/types")
async def imposta_tipi(
    personality_id: uuid.UUID,
    type_ids: List[int],
    session: DbSession,
    ctx: Contesto,
) -> Dict[str, Any]:
    repo = AdminPersonalityRepository(session, ctx)
    personalita = await _o_404(repo.per_id(personality_id))

    await repo.imposta_tipi(personalita, type_ids)
    await session.commit()
    await session.refresh(personalita)
    return _personalita_json(personalita)


@router.post("/personalities/{personality_id}/corpora")
async def collega_corpus(
    personality_id: uuid.UUID,
    payload: CollegaCorpus,
    session: DbSession,
    ctx: Contesto,
) -> Dict[str, Any]:
    repo = AdminPersonalityRepository(session, ctx)
    personalita = await _o_404(repo.per_id(personality_id))

    if await AdminKnowledgeRepository(session, ctx).per_id(payload.kb_id) is None:
        raise HTTPException(status_code=404, detail="Knowledge base non trovata")

    await repo.collega_corpus(
        personalita, payload.kb_id, role=payload.role,
        max_chunks=payload.max_chunks, enabled=payload.enabled,
    )
    await session.commit()
    return {"collegato": True}


@router.delete("/personalities/{personality_id}/corpora/{kb_id}")
async def scollega_corpus(
    personality_id: uuid.UUID,
    kb_id: uuid.UUID,
    session: DbSession,
    ctx: Contesto,
) -> Dict[str, Any]:
    repo = AdminPersonalityRepository(session, ctx)
    personalita = await _o_404(repo.per_id(personality_id))

    await repo.scollega_corpus(personalita, kb_id)
    await session.commit()
    return {"scollegato": True}


# --- knowledge base --------------------------------------------------------


@router.get("/knowledge-bases")
async def elenco_basi(session: DbSession, ctx: Contesto) -> List[Dict[str, Any]]:
    return [
        {
            "id": str(kb.id), "slug": kb.slug, "name": kb.name,
            "description": kb.description, "kind": kb.kind,
            "visibility": kb.visibility, "embed_model": kb.embed_model,
            "text_config": kb.text_config, "stats": kb.stats,
        }
        for kb in await AdminKnowledgeRepository(session, ctx).elenco()
    ]


@router.post("/knowledge-bases", status_code=status.HTTP_201_CREATED)
async def crea_base(
    payload: CreaBase,
    session: DbSession,
    ctx: Contesto,
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> Dict[str, Any]:
    repo = AdminKnowledgeRepository(session, ctx)

    if await repo.per_slug(payload.slug) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Esiste già una base con lo slug '{payload.slug}'",
        )

    # Il modello di embedding non si sceglie qui: è quello configurato, e
    # registrarlo sulla base è ciò che impedisce di interrogare domani con un
    # modello diverso da quello che ha scritto i vettori.
    kb = await repo.crea(**payload.model_dump(), embed_model=embedder.modello)
    await session.commit()
    return {"id": str(kb.id), "slug": kb.slug, "embed_model": kb.embed_model}


@router.get("/knowledge-bases/{kb_id}/documents")
async def elenco_documenti(
    kb_id: uuid.UUID, session: DbSession, ctx: Contesto,
) -> List[Dict[str, Any]]:
    repo = AdminKnowledgeRepository(session, ctx)
    kb = await _o_404(repo.per_id(kb_id), "Knowledge base non trovata")

    return [
        {
            "id": str(d.id), "title": d.title, "uri": d.uri,
            "sha256": d.sha256[:12], "meta": d.meta,
            "fetched_at": d.fetched_at.isoformat() if d.fetched_at else None,
        }
        for d in await repo.documenti(kb)
    ]


@router.delete("/knowledge-bases/{kb_id}/documents/{document_id}")
async def elimina_documento(
    kb_id: uuid.UUID,
    document_id: uuid.UUID,
    session: DbSession,
    ctx: Contesto,
) -> Dict[str, Any]:
    from ...domain.knowledge_models import Document

    repo = AdminKnowledgeRepository(session, ctx)
    kb = await _o_404(repo.per_id(kb_id), "Knowledge base non trovata")

    documento = await session.get(Document, document_id)
    if documento is None or documento.kb_id != kb.id:
        raise HTTPException(status_code=404, detail="Documento non trovato")

    await repo.elimina_documento(kb, documento)
    await session.commit()
    return {"eliminato": True}


# --- osservabilità del recupero -------------------------------------------


@router.post("/retrieval/preview")
async def prova_recupero(
    payload: ProvaRecupero,
    session: DbSession,
    ctx: Contesto,
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> Dict[str, Any]:
    """Interroga il recupero senza generare nulla.

    Restituisce ciò che sarebbe finito nel prompt **e ciò che è rimasto
    fuori**, con i punteggi di entrambe le ricerche. È l'unico modo di
    studiare il recupero come variabile isolata: con la generazione in mezzo,
    ogni conclusione è confusa dal modello.
    """
    esito = await Retriever(session, embedder).cerca(
        payload.domanda, payload.kb_ids, limite=payload.limite,
    )

    def voce(p) -> Dict[str, Any]:
        c = p.corrispondenza
        return {
            "etichetta": p.etichetta or None,
            "documento": c.documento_titolo,
            "sezione": c.sezione,
            "testo": c.testo,
            "rrf": round(p.punteggio_rrf, 5),
            "posizione_vettoriale": p.posizione_vettoriale,
            "posizione_lessicale": p.posizione_lessicale,
            "somiglianza": round(p.somiglianza, 4) if p.somiglianza else None,
            "rilevanza_lessicale": (
                round(p.rilevanza_lessicale, 4) if p.rilevanza_lessicale else None
            ),
            "trovato_da_entrambe": p.trovato_da_entrambe,
            # Cio' che la digestione ha capito: serve a chi studia un recupero
            # sbagliato per sapere se il problema sia nell'ordinamento o nella
            # classificazione del passaggio.
            "categoria": c.categoria,
            "provenienza": c.provenienza,
            "qualita": c.qualita,
        }

    return {
        "domanda": payload.domanda,
        "scelti": [voce(p) for p in esito.scelti],
        # Gli scartati sono la metà utile: «non l'ha trovato» e «l'ha trovato
        # e messo settimo» sono due difetti diversi, con due rimedi diversi.
        "scartati": [voce(p) for p in esito.scartati[:12]],
    }


# --- tassonomia ------------------------------------------------------------


@router.get("/types")
async def elenco_tipi(session: DbSession, ctx: Contesto) -> List[Dict[str, Any]]:
    return [
        {"id": t.id, "slug": t.slug, "name": t.name, "description": t.description}
        for t in await TassonomiaRepository(session, ctx).tipi()
    ]


@router.post("/types", status_code=status.HTTP_201_CREATED)
async def crea_tipo(
    payload: CreaTipo, session: DbSession, ctx: Contesto,
) -> Dict[str, Any]:
    tipo = await TassonomiaRepository(session, ctx).crea_tipo(**payload.model_dump())
    await session.commit()
    return {"id": tipo.id, "slug": tipo.slug, "name": tipo.name}


@router.get("/categories")
async def elenco_categorie(session: DbSession, ctx: Contesto) -> List[Dict[str, Any]]:
    return [
        {"id": c.id, "slug": c.slug, "name": c.name, "rank": c.rank}
        for c in await TassonomiaRepository(session, ctx).categorie()
    ]


@router.post("/categories", status_code=status.HTTP_201_CREATED)
async def crea_categoria(
    payload: CreaCategoria, session: DbSession, ctx: Contesto,
) -> Dict[str, Any]:
    categoria = await TassonomiaRepository(session, ctx).crea_categoria(
        **payload.model_dump()
    )
    await session.commit()
    return {"id": categoria.id, "slug": categoria.slug, "rank": categoria.rank}


# --- avatar ----------------------------------------------------------------


def _avatar_json(a, descrittore=None) -> Dict[str, Any]:
    corpo = {
        "id": str(a.id),
        "slug": a.slug,
        "name": a.name,
        "kind": a.kind,
        "description": a.description,
        "config": a.config or {},
    }
    if descrittore is not None:
        corpo["descrittore"] = descrittore.to_dict()
    return corpo


@router.get("/avatars")
async def elenco_avatar(session: DbSession, ctx: Contesto) -> List[Dict[str, Any]]:
    return [
        _avatar_json(a) for a in await AvatarRepository(session, ctx).elenco()
    ]


@router.post("/avatars", status_code=status.HTTP_201_CREATED)
async def crea_avatar(
    payload: CreaAvatar, session: DbSession, ctx: Contesto,
) -> Dict[str, Any]:
    repo = AvatarRepository(session, ctx)
    if await repo.per_slug(payload.slug) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Esiste già un avatar con lo slug '{payload.slug}'",
        )

    try:
        avatar = await repo.crea(**payload.model_dump())
    except ConfigurazioneAvatarNonValida as exc:
        # 400 e non 422: la forma del corpo è corretta, è il contenuto a non
        # descrivere qualcosa di disegnabile. Il messaggio dice cosa manca,
        # ed è l'unica cosa utile a chi lo sta configurando.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await session.commit()
    return _avatar_json(avatar, descrivi_avatar(avatar.kind, avatar.config))


@router.patch("/avatars/{avatar_id}")
async def modifica_avatar(
    avatar_id: uuid.UUID,
    payload: ModificaAvatar,
    session: DbSession,
    ctx: Contesto,
) -> Dict[str, Any]:
    repo = AvatarRepository(session, ctx)
    avatar = await _o_404(await repo.per_id(avatar_id), "Avatar non trovato")

    try:
        await repo.modifica(avatar, **payload.model_dump(exclude_unset=True))
    except ConfigurazioneAvatarNonValida as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await session.commit()
    return _avatar_json(avatar, descrivi_avatar(avatar.kind, avatar.config))


@router.delete("/avatars/{avatar_id}")
async def elimina_avatar(
    avatar_id: uuid.UUID, session: DbSession, ctx: Contesto,
) -> Dict[str, Any]:
    repo = AvatarRepository(session, ctx)
    avatar = await _o_404(await repo.per_id(avatar_id), "Avatar non trovato")

    quante = await repo.elimina(avatar)
    await session.commit()
    # Quante voci sono rimaste senza volto: cancellare un avatar cambia
    # qualcosa altrove, e scoprirlo guardando le personalità una per una
    # sarebbe il modo peggiore.
    return {"eliminato": True, "personalita_senza_volto": quante}


@router.put("/personalities/{personality_id}/avatar")
async def volgi_avatar(
    personality_id: uuid.UUID,
    payload: VolgiAvatar,
    session: DbSession,
    ctx: Contesto,
) -> Dict[str, Any]:
    """Assegna o toglie il volto a una personalità."""
    personalita = await _o_404(
        await AdminPersonalityRepository(session, ctx).per_id(personality_id)
    )

    if payload.avatar_id is not None:
        avatar = await AvatarRepository(session, ctx).per_id(payload.avatar_id)
        if avatar is None:
            raise HTTPException(status_code=404, detail="Avatar non trovato")

    personalita.avatar_id = payload.avatar_id
    await session.commit()
    return {
        "personality_id": str(personality_id),
        "avatar_id": str(payload.avatar_id) if payload.avatar_id else None,
    }


# --- realizzazione ---------------------------------------------------------


@router.get("/build-options")
async def opzioni_di_realizzazione(
    registry: Annotated[CapabilityRegistry, Depends(get_capability_registry)],
) -> Dict[str, Any]:
    """Cosa si può costruire qui, e perché no.

    È ciò che guida la scheda «Realizzazione»: le caselle non disponibili si
    mostrano disabilitate **con il motivo accanto**, non si nascondono. Un
    comando che sparisce sembra un difetto dell'interfaccia; uno spento con la
    ragione scritta è informazione.
    """
    opzioni = {}
    for feature, etichetta in (
        (Feature.BUILD_LORA, "Adapter LoRA"),
        (Feature.FULL_FINETUNE, "Fine-tuning completo"),
        (Feature.GGUF_EXPORT, "Esportazione GGUF"),
        (Feature.KV_CACHE_CAG, "CAG con KV-cache"),
    ):
        disponibile, motivo = registry.can(feature)
        opzioni[feature.value] = {
            "label": etichetta,
            "available": disponibile,
            "reason": motivo,
        }

    return {
        "modi": {
            # RAG è sempre disponibile: non richiede acceleratori, ed è la
            # ragione per cui la piattaforma resta utile su un server nudo.
            "rag": {"label": "RAG (prompt e recupero)", "available": True, "reason": ""},
            "lora": opzioni[Feature.BUILD_LORA.value],
            "finetune": opzioni[Feature.FULL_FINETUNE.value],
        },
        "esportazioni": {"gguf": opzioni[Feature.GGUF_EXPORT.value]},
        "contesto": {"kv_cache": opzioni[Feature.KV_CACHE_CAG.value]},
    }


# --- registro --------------------------------------------------------------


@router.get("/audit")
async def registro(
    session: DbSession,
    limite: int = 100,
    azione: Optional[str] = None,
    target_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Il registro delle operazioni. Solo lettura, per costruzione."""
    voci = await RegistroAuditRepository(session).recenti(
        limite=min(limite, 500), azione=azione, target_type=target_type,
    )
    return [
        {
            "id": v.id,
            "action": v.action,
            "actor_id": v.actor_id,
            "target_type": v.target_type,
            "target_id": v.target_id,
            "before": v.before,
            "after": v.after,
            "ip": v.ip,
            "correlation_id": v.correlation_id,
            "created_at": v.created_at.isoformat(),
        }
        for v in voci
    ]


async def _o_404(atteso, messaggio: str = "Personalità non trovata"):
    risultato = await atteso
    if risultato is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=messaggio)
    return risultato
