"""Il laboratorio della console: analisi, esperimenti, prove dei prompt.

Tre strumenti per la stessa domanda — *la voce funziona?* — da tre distanze:
l'analisi guarda l'uso di tutti, l'esperimento confronta due versioni su
utenti veri, il playground prova una modifica su una domanda sola prima di
farla vedere a qualcuno.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from ...auth.dependencies import DbSession, require_role
from ...domain.admin_repositories import AdminPersonalityRepository
from ...domain.knowledge_models import Personality, PersonalityVersion
from ...domain.lab_models import Experiment
from ...domain.repositories import AuditRepository, PersonalityRepository
from ...guards.groundcheck import controlla_citazioni
from ...guards.policy import RegistroGuardrail
from ...knowledge.embedding import Embedder
from ...knowledge.retriever import Retriever
from ...lab.analisi import riepilogo
from ...lab.esperimenti import EsperimentoNonValido, GestoreEsperimenti, esperimento_json
from ...llm.base import GenerationError, LLMProvider
from ...runtime.persona_engine import PersonaEngine
from ..deps import get_embedder, get_guardrail, get_llm_provider
from .admin import Contesto

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["laboratorio"],
    dependencies=[Depends(require_role("admin"))],
)


async def _registra(session, ctx, azione: str, *, tipo: str, target: Any, dopo: Dict[str, Any]) -> None:
    await AuditRepository(session).record(
        action=azione, actor_id=ctx.attore.id, target_type=tipo,
        target_id=str(target), after=dopo, ip=ctx.ip,
        correlation_id=ctx.correlation_id,
    )


# --- analisi ---------------------------------------------------------------


@router.get("/analytics")
async def analisi(
    session: DbSession,
    giorni: int = Query(default=30, ge=1, le=365),
) -> Dict[str, Any]:
    return await riepilogo(session, giorni=giorni)


# --- esperimenti -----------------------------------------------------------


class VarianteRichiesta(BaseModel):
    version_id: uuid.UUID
    peso: int = Field(default=50, ge=1, le=100)


class NuovoEsperimento(BaseModel):
    nome: str = Field(min_length=3, max_length=120)
    ipotesi: Optional[str] = Field(default=None, max_length=2000)
    varianti: List[VarianteRichiesta] = Field(min_length=2, max_length=8)


class Conclusione(BaseModel):
    vincitore: Optional[uuid.UUID] = None
    conclusione: Optional[str] = Field(default=None, max_length=2000)
    #: Rende la versione vincitrice quella servita a tutti. Facoltativo: si
    #: può concludere un esperimento che non ha dato un vincitore, o
    #: pubblicare più tardi.
    pubblica: bool = False


@router.get("/experiments")
async def elenco_esperimenti(
    session: DbSession, personality_id: Optional[uuid.UUID] = None,
) -> List[Dict[str, Any]]:
    esperimenti = await GestoreEsperimenti(session).elenco(personality_id=personality_id)
    nomi = {
        p.id: (p.slug, p.display_name)
        for p in (await session.execute(
            select(Personality).where(
                Personality.id.in_({e.personality_id for e in esperimenti})
            )
        )).scalars()
    }
    return [
        {**esperimento_json(e), "personalita": {
            "slug": nomi.get(e.personality_id, ("", ""))[0],
            "nome": nomi.get(e.personality_id, ("", ""))[1],
        }}
        for e in esperimenti
    ]


@router.post("/personalities/{personality_id}/experiments", status_code=status.HTTP_201_CREATED)
async def apri_esperimento(
    personality_id: uuid.UUID, payload: NuovoEsperimento, session: DbSession, ctx: Contesto,
) -> Dict[str, Any]:
    personalita = await session.get(Personality, personality_id)
    if personalita is None:
        raise HTTPException(status_code=404, detail="Personalità non trovata")
    try:
        esperimento = await GestoreEsperimenti(session).crea(
            personalita,
            nome=payload.nome,
            ipotesi=payload.ipotesi,
            varianti=[v.model_dump() for v in payload.varianti],
            autore_id=ctx.attore.id,
        )
    except EsperimentoNonValido as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    await _registra(
        session, ctx, "esperimento.aperto", tipo="experiment", target=esperimento.id,
        dopo={"personalita": personalita.slug, "varianti": esperimento.variants},
    )
    await session.commit()
    return esperimento_json(esperimento)


async def _esperimento(session, esperimento_id: uuid.UUID) -> Experiment:
    esperimento = await session.get(Experiment, esperimento_id)
    if esperimento is None:
        raise HTTPException(status_code=404, detail="Esperimento non trovato")
    return esperimento


@router.get("/experiments/{experiment_id}")
async def dettaglio_esperimento(experiment_id: uuid.UUID, session: DbSession) -> Dict[str, Any]:
    esperimento = await _esperimento(session, experiment_id)
    return {
        **esperimento_json(esperimento),
        "risultati": await GestoreEsperimenti(session).risultati(esperimento),
    }


@router.post("/experiments/{experiment_id}/conclude")
async def concludi_esperimento(
    experiment_id: uuid.UUID, payload: Conclusione, session: DbSession, ctx: Contesto,
) -> Dict[str, Any]:
    esperimento = await _esperimento(session, experiment_id)
    try:
        await GestoreEsperimenti(session).concludi(
            esperimento, vincitore=payload.vincitore, conclusione=payload.conclusione,
        )
    except EsperimentoNonValido as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    pubblicata = False
    if payload.pubblica and payload.vincitore is not None:
        personalita = await session.get(Personality, esperimento.personality_id)
        versione = await session.get(PersonalityVersion, payload.vincitore)
        # Passa dal repository della console: la pubblicazione lascia la sua
        # riga nel registro come quella fatta a mano.
        await AdminPersonalityRepository(session, ctx).pubblica(personalita, versione)
        pubblicata = True

    await _registra(
        session, ctx, "esperimento.concluso", tipo="experiment", target=esperimento.id,
        dopo={
            "vincitore": str(payload.vincitore) if payload.vincitore else None,
            "pubblicata": pubblicata,
        },
    )
    await session.commit()
    return {**esperimento_json(esperimento), "pubblicata": pubblicata}


# --- playground ------------------------------------------------------------


class Prova(BaseModel):
    personality_id: uuid.UUID
    #: La versione di partenza; assente, quella corrente.
    version_id: Optional[uuid.UUID] = None
    #: Ciò che si vuole provare a cambiare. Assente, resta quello della
    #: versione di partenza.
    system_prompt: Optional[str] = Field(default=None, max_length=40_000)
    regole: Optional[List[str]] = Field(default=None, max_length=50)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_chunks: Optional[int] = Field(default=None, ge=0, le=20)
    domanda: str = Field(min_length=1, max_length=4000)


def _versione_di_prova(base: PersonalityVersion, prova: Prova) -> PersonalityVersion:
    """Una versione che non esiste: la base con le modifiche, mai salvata.

    Un oggetto staccato dalla sessione, e senza `id`: se per errore finisse in
    una `flush`, il vincolo di chiave primaria lo fermerebbe invece di
    lasciarlo diventare una versione vera che nessuno ha pubblicato.
    """
    regole = dict(base.behavior_rules or {})
    if prova.regole is not None:
        regole["regole"] = prova.regole
    llm = dict(base.llm_config or {})
    if prova.temperature is not None:
        llm["temperature"] = prova.temperature
    rag = dict(base.rag_config or {})
    if prova.max_chunks is not None:
        rag["max_chunks"] = prova.max_chunks
    return PersonalityVersion(
        personality_id=base.personality_id,
        version=base.version,
        system_prompt=prova.system_prompt if prova.system_prompt is not None else base.system_prompt,
        behavior_rules=regole,
        llm_config=llm,
        rag_config=rag,
        memory_config=base.memory_config,
        guard_config=base.guard_config,
    )


@router.post("/playground")
async def prova_prompt(
    payload: Prova,
    session: DbSession,
    ctx: Contesto,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    guardrail: Annotated[RegistroGuardrail, Depends(get_guardrail)],
) -> Dict[str, Any]:
    """Una risposta di prova: nessuna conversazione, nessun credito, nessuna memoria.

    Restituisce anche il prompt esatto che è stato mandato: il punto del
    playground è vedere *perché* la risposta è venuta così, e la risposta da
    sola non lo dice.
    """
    personalita = await session.get(Personality, payload.personality_id)
    if personalita is None:
        raise HTTPException(status_code=404, detail="Personalità non trovata")

    repo = PersonalityRepository(session)
    base_id = payload.version_id or personalita.current_version_id
    base = await repo.versione(base_id) if base_id else None
    if base is None or base.personality_id != personalita.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Nessuna versione di partenza: crea o scegli una versione della personalità.",
        )

    modificata = any(
        v is not None for v in (payload.system_prompt, payload.regole, payload.temperature, payload.max_chunks)
    )
    versione = _versione_di_prova(base, payload) if modificata else base
    kb_ids = await repo.corpora_di(personalita.id)

    motore = PersonaEngine(provider, retriever=Retriever(session, embedder) if kb_ids else None)
    turno = await motore.prepara(
        versione=versione,
        domanda=payload.domanda,
        kb_ids=kb_ids,
        politiche=guardrail.istruzioni_per(personalita.slug),
    )

    inizio = time.perf_counter()
    try:
        async for _ in motore.rispondi_in_streaming(turno, versione=versione):
            pass
    except GenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Il modello non ha risposto: {exc}",
        ) from exc
    durata = int((time.perf_counter() - inizio) * 1000)

    passaggi = turno.recupero.scelti if turno.recupero else []
    citazioni = controlla_citazioni(turno.testo, [p.etichetta for p in passaggi])

    # Nel registro senza la risposta: chi ha provato cosa, su quale versione,
    # e se l'ha modificata. La domanda sì, accorciata: è ciò che permette di
    # rifare la prova.
    await _registra(
        session, ctx, "playground.eseguito", tipo="personality", target=personalita.id,
        dopo={
            "versione": base.version, "modificata": modificata,
            "domanda": payload.domanda[:300],
        },
    )
    await session.commit()

    return {
        "risposta": turno.testo,
        "versione": {"id": str(base.id), "numero": base.version, "modificata": modificata},
        "prompt": [
            {"role": m.role, "content": m.content} for m in turno.contesto.messaggi
        ],
        "punto_di_cache": turno.contesto.punto_di_cache,
        "passaggi": [
            {
                "etichetta": p.etichetta,
                "documento": p.corrispondenza.documento_titolo,
                "sezione": p.corrispondenza.sezione,
                "estratto": p.corrispondenza.testo[:400].strip(),
            }
            for p in passaggi
        ],
        "riferimenti_inventati": citazioni.riferimenti_inventati,
        "uso": turno.usage.to_dict() if turno.usage else None,
        "strategia": turno.traccia_risposta()["usage"]["strategia"],
        "tempi_ms": {**turno.tempi_ms, "totale": durata},
    }
