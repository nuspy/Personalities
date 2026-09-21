"""Conversazione con streaming, con o senza personalità.

Se la richiesta nomina una personalità, il turno passa dal motore: recupero
dal corpus, strati del prompt, citazioni numerate. Senza, il modello risponde
con la propria voce — è il percorso della fase 0, e resta perché è il modo di
provare il motore senza interporre un carattere.

**Perché le fonti viaggiano come evento SSE.** Il client le riceve quando la
risposta comincia, non alla fine: così il margine dell'apparato si popola
mentre il testo arriva, e chi legge sa da subito su cosa si regge ciò che sta
leggendo. Mandarle in coda vorrebbe dire mostrarle quando non servono più.

**Perché SSE e non WebSocket.** Il flusso è unidirezionale: l'utente manda una
domanda e riceve token. Un WebSocket aggiungerebbe una connessione
bidirezionale da tenere viva, riconnettere e autenticare fuori dal normale
ciclo HTTP, per un canale di ritorno che non serve.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Annotated, Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...auth.dependencies import CurrentUser, DbSession
from ...api.deps import get_embedder, get_guardrail, get_llm_provider
from ...domain.knowledge_models import PersonalityVersion
from ...domain.models import Conversation
from ...domain.repositories import (
    ConversationRepository, PersonalityRepository, TraceRepository,
)
from ...domain.session import SessionFactory
from ...guards.groundcheck import Giudice, controlla_citazioni
from ...guards.policy import RegistroGuardrail
from ...knowledge.embedding import Embedder
from ...knowledge.retriever import Retriever
from ...llm.base import (
    GenerationError, GenerationRequest, LLMProvider, Message as LLMMessage,
    TruncatedResponse,
)
from ...observability.correlation import current_correlation_id
from ...runtime.persona_engine import PersonaEngine, riferimenti_citati

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversazione"])

#: Quanti turni passati rimandare al modello. In questa fase è una finestra
#: fissa; dalla fase 4 il riassunto della sessione e le memorie prendono il
#: posto della coda più vecchia, che è ciò che permette conversazioni lunghe
#: senza far crescere il prompt senza limite.
TURNI_DI_CONTESTO = 20


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32_000)
    conversation_id: Optional[uuid.UUID] = None
    #: Lo slug della personalità. Assente: risponde il modello nudo.
    personality: Optional[str] = Field(default=None, max_length=80)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    last_message_at: Optional[str]
    personality: Optional[str] = None

    @classmethod
    def of(cls, c: Conversation, slug: Optional[str] = None) -> "ConversationSummary":
        return cls(
            id=c.id,
            title=c.title,
            last_message_at=c.last_message_at.isoformat() if c.last_message_at else None,
            personality=slug,
        )


class PersonalitySummary(BaseModel):
    slug: str
    display_name: str
    description: Optional[str]


def sse(evento: str, dati: dict) -> str:
    """Un evento SSE.

    `ensure_ascii=False` perché il testo è in italiano e le sequenze di escape
    raddoppierebbero i byte delle lettere accentate su ogni singolo frammento.
    """
    return f"event: {evento}\ndata: {json.dumps(dati, ensure_ascii=False)}\n\n"


@router.get("/personalities", response_model=List[PersonalitySummary])
async def elenco_personalita(session: DbSession) -> List[PersonalitySummary]:
    """Il catalogo. Non richiede identità: è ciò che si vede prima di entrare."""
    return [
        PersonalitySummary(
            slug=p.slug, display_name=p.display_name, description=p.description,
        )
        for p in await PersonalityRepository(session).pubblicate()
    ]


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    user: CurrentUser,
    session: DbSession,
    request: Request,
    session_factory: SessionFactory,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    guardrail: Annotated[RegistroGuardrail, Depends(get_guardrail)],
) -> StreamingResponse:
    """Manda un messaggio e ricevi la risposta mentre viene generata."""
    repo = ConversationRepository(session)
    personalita_repo = PersonalityRepository(session)

    versione: Optional[PersonalityVersion] = None
    kb_ids: List[uuid.UUID] = []
    slug_personalita: Optional[str] = None

    if payload.personality:
        personalita = await personalita_repo.per_slug(payload.personality)
        if personalita is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Personalità non trovata",
            )
        versione = await personalita_repo.versione_corrente(personalita)
        if versione is None:
            # Pubblicata ma senza versione: è uno stato incoerente, non una
            # richiesta sbagliata. 409 e non 404, perché l'utente non può farci
            # nulla e chi amministra sì.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Questa personalità non ha ancora una versione pubblicata",
            )
        kb_ids = await personalita_repo.corpora(personalita)
        slug_personalita = personalita.slug

    if payload.conversation_id:
        conversazione = await repo.get(user, payload.conversation_id)
        if conversazione is None:
            # 404 e non 403: vedi `ConversationRepository.get`. Distinguere
            # «non esiste» da «non è tua» rivelerebbe quali id esistono.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversazione non trovata",
            )
    else:
        conversazione = await repo.create(
            user, title=payload.message[:80].strip() or None
        )
        if versione is not None:
            conversazione.personality_id = versione.personality_id
            # La versione si fissa all'apertura e non si risolve a ogni turno:
            # è ciò che rende riproducibile una conversazione dopo che la
            # personalità è stata modificata.
            conversazione.personality_version_id = versione.id

    # Una conversazione già aperta porta con sé la sua versione, e quella
    # vince: proseguire con una versione diversa da quella che ha prodotto i
    # turni precedenti renderebbe lo scambio incoerente a metà, e
    # irriproducibile dopo.
    if conversazione.personality_version_id is not None:
        versione = await personalita_repo.versione(
            conversazione.personality_version_id
        )
        if versione is not None:
            kb_ids = await personalita_repo.corpora_di(versione.personality_id)
            slug_personalita = await personalita_repo.slug_di(versione.personality_id)

    correlation_id = current_correlation_id()

    storico = await repo.messages(user, conversazione.id, limit=TURNI_DI_CONTESTO)
    messaggi_storico = [
        LLMMessage(role=m.role, content=m.content) for m in storico  # type: ignore[arg-type]
    ]

    await repo.add_message(
        conversazione,
        role="user",
        content=payload.message,
        correlation_id=correlation_id,
    )
    # Il turno dell'utente si salva **prima** di generare: se il modello
    # fallisce, la sua domanda non deve andare perduta insieme all'errore.
    await session.commit()

    conversazione_id = conversazione.id
    versione_id = versione.id if versione else None

    motore = PersonaEngine(
        provider,
        retriever=Retriever(session, embedder) if kb_ids else None,
    )

    turno = None
    rubriche = []
    if versione is not None:
        turno = await motore.prepara(
            versione=versione,
            domanda=payload.message,
            kb_ids=kb_ids,
            storico=messaggi_storico,
            # La metà preventiva dei guardrail entra nello strato stabile:
            # costa una volta sola perché sta nel prefisso, e agisce prima che
            # il problema esista.
            politiche=guardrail.istruzioni_per(slug_personalita),
        )
        rubriche = guardrail.rubriche_per(slug_personalita)

    #: `citations` verifica soltanto le etichette, `nli` interroga un giudice.
    #: La scelta sta sulla versione della personalita' perche' dipende da cosa
    #: quella voce fa: una che risponde di fatti merita il giudice, una che
    #: consiglia e basta pagherebbe una chiamata per nulla.
    livello_verifica = (
        (versione.guard_config or {}).get("groundcheck", "citations")
        if versione is not None
        else "citations"
    )

    async def flusso() -> AsyncIterator[str]:
        yield sse("start", {
            "conversation_id": str(conversazione_id),
            "correlation_id": correlation_id,
            "personality": slug_personalita,
        })

        if turno is not None and turno.recupero:
            # Le fonti prima del testo: il margine si popola mentre la
            # risposta arriva, non dopo che è finita.
            yield sse("sources", {
                "passaggi": [
                    {
                        "etichetta": p.etichetta,
                        "documento": p.corrispondenza.documento_titolo,
                        "sezione": p.corrispondenza.sezione,
                        "uri": p.corrispondenza.documento_uri,
                        "estratto": p.corrispondenza.testo[:240].strip(),
                    }
                    for p in turno.recupero.scelti
                ]
            })

        if turno is not None and turno.degradato:
            yield sse("degradato", {"motivo": turno.motivo_degrado})

        pezzi: List[str] = []
        uso = None
        try:
            if turno is not None:
                async for chunk in motore.rispondi_in_streaming(turno, versione=versione):
                    async for evento in _eventi_da(chunk, pezzi):
                        yield evento
                    if chunk.done:
                        uso = chunk.usage
            else:
                richiesta = GenerationRequest(
                    messages=[*messaggi_storico,
                              LLMMessage(role="user", content=payload.message)],
                    temperature=payload.temperature,
                )
                async for chunk in provider.stream(richiesta):
                    async for evento in _eventi_da(chunk, pezzi):
                        yield evento
                    if chunk.done:
                        uso = chunk.usage

        except TruncatedResponse as exc:
            logger.warning("Risposta troncata (%s): %s", correlation_id, exc)
            yield sse("error", {
                "message": "Il modello si è fermato prima di rispondere.",
                "recoverable": True,
            })
        except GenerationError as exc:
            logger.error("Generazione fallita (%s): %s", correlation_id, exc)
            yield sse("error", {
                "message": "Il modello non è raggiungibile.",
                "recoverable": True,
            })
        except Exception:
            logger.exception("Errore inatteso nel flusso (%s)", correlation_id)
            yield sse("error", {"message": "Errore interno.", "recoverable": False})

        risposta = "".join(pezzi)
        passaggi = turno.recupero.scelti if (turno and turno.recupero) else []
        forniti = [p.etichetta for p in passaggi]

        # Livello deterministico: sempre, perché costa zero e non può
        # sbagliare. Si esegue prima di `done` — l'utente deve sapere se ciò
        # che ha appena letto cita qualcosa che non esiste.
        esito = controlla_citazioni(risposta, forniti)
        if esito.riferimenti_inventati:
            logger.warning(
                "Riferimenti inventati (%s): %s",
                correlation_id, esito.riferimenti_inventati,
            )

        yield sse("done", {
            "usage": uso.to_dict() if uso else None,
            "characters": len(risposta),
            "riferimenti_inventati": esito.riferimenti_inventati,
        })

        # Il giudice **dopo** `done`: costa una chiamata intera, e farla
        # aspettare a chi ha già finito di leggere significherebbe spegnere la
        # verifica alla prima lamentela sulla lentezza. Il verdetto arriva
        # quando arriva, e il client aggiorna la nota sotto la risposta.
        if livello_verifica == "nli" and risposta.strip() and passaggi:
            esito = await Giudice(provider).valuta(
                risposta, passaggi, rubriche=rubriche,
            )
            yield sse("verifica", {
                "livello": esito.livello,
                "fondata": esito.fondata,
                "non_eseguito": esito.non_eseguito,
                "infondate": [
                    {"testo": a.testo, "nota": a.nota} for a in esito.infondate
                ],
                "conteggi": esito.to_dict()["conteggi"],
            })

        if risposta.strip():
            # Una sessione nuova: quella della richiesta è chiusa dalla
            # dipendenza appena l'endpoint restituisce la `StreamingResponse`,
            # mentre questo generatore continua a girare dopo.
            async with session_factory() as scrittura:
                repo_scrittura = ConversationRepository(scrittura)
                conv = await scrittura.get(Conversation, conversazione_id)
                if conv is not None:
                    messaggio = await repo_scrittura.add_message(
                        conv,
                        role="assistant",
                        content=risposta,
                        correlation_id=correlation_id,
                        tokens=uso.to_dict() if uso else None,
                    )
                    if turno is not None:
                        await TraceRepository(scrittura).registra(
                            message_id=messaggio.id,
                            personality_version_id=versione_id,
                            **{
                                k: v for k, v in turno.traccia_risposta().items()
                                if k in ("retrieved", "usage", "latency_ms")
                            },
                            grounding=esito.to_dict(),
                        )
                    await scrittura.commit()

    return StreamingResponse(
        flusso(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Senza, nginx accumula la risposta e la consegna tutta insieme
            # alla fine: lo streaming smette di esistere e non c'è nulla, nei
            # log dell'applicazione, che lo riveli.
            "X-Accel-Buffering": "no",
        },
    )


async def _eventi_da(chunk, pezzi: List[str]) -> AsyncIterator[str]:
    """Traduce un frammento del modello negli eventi del flusso."""
    if chunk.text:
        pezzi.append(chunk.text)
        yield sse("token", {"text": chunk.text})
    elif chunk.reasoning:
        # Il ragionamento si segnala ma non si trasmette: serve a mostrare che
        # il modello sta lavorando, non a essere letto.
        yield sse("thinking", {})


@router.get("/conversations", response_model=List[ConversationSummary])
async def elenco_conversazioni(
    user: CurrentUser,
    session: DbSession,
    limit: int = 50,
    offset: int = 0,
) -> List[ConversationSummary]:
    conversazioni = await ConversationRepository(session).list_recent(
        user, limit=min(limit, 200), offset=offset,
    )
    return [ConversationSummary.of(c) for c in conversazioni]


@router.get("/conversations/{conversation_id}/messages")
async def messaggi(
    conversation_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> List[Dict[str, Any]]:
    repo = ConversationRepository(session)
    if await repo.get(user, conversation_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversazione non trovata",
        )
    return [
        {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
        }
        for m in await repo.messages(user, conversation_id)
    ]
