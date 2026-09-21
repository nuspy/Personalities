"""Conversazione con streaming.

In questa fase il modello risponde **senza personalità**: nessun prompt di
carattere, nessun recupero, nessuna memoria. È voluto — lo scheletro serve a
dimostrare che il percorso completo regge (token → utente → conversazione →
modello → token in arrivo → persistenza → traccia), e mescolarci subito la
qualità delle risposte confonderebbe due domande diverse. Gli strati del
prompt entrano nella fase 1, in questo stesso punto.

**Perché SSE e non WebSocket.** Il flusso è unidirezionale: l'utente manda una
domanda e riceve token. Un WebSocket aggiungerebbe una connessione bidirezionale
da tenere viva, riconnettere e autenticare fuori dal normale ciclo HTTP, per un
canale di ritorno che non serve. SSE passa dai proxy, si riconnette da solo e
usa lo stesso header `Authorization` di tutto il resto.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Annotated, AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...auth.dependencies import CurrentUser, DbSession
from ...api.deps import get_llm_provider
from ...domain.models import Conversation
from ...domain.repositories import ConversationRepository
from ...domain.session import SessionFactory
from ...llm.base import (
    GenerationError, GenerationRequest, LLMProvider, Message as LLMMessage,
    TruncatedResponse,
)
from ...observability.correlation import current_correlation_id

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
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    last_message_at: Optional[str]

    @classmethod
    def of(cls, c: Conversation) -> "ConversationSummary":
        return cls(
            id=c.id,
            title=c.title,
            last_message_at=c.last_message_at.isoformat() if c.last_message_at else None,
        )


def sse(evento: str, dati: dict) -> str:
    """Un evento SSE.

    `ensure_ascii=False` perché il testo è in italiano e le sequenze di escape
    raddoppierebbero i byte delle lettere accentate su ogni singolo frammento.
    """
    return f"event: {evento}\ndata: {json.dumps(dati, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    user: CurrentUser,
    session: DbSession,
    request: Request,
    session_factory: SessionFactory,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> StreamingResponse:
    """Manda un messaggio e ricevi la risposta mentre viene generata."""
    repo = ConversationRepository(session)

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

    correlation_id = current_correlation_id()

    storico = await repo.messages(user, conversazione.id, limit=TURNI_DI_CONTESTO)
    messaggi: List[LLMMessage] = [
        LLMMessage(role=m.role, content=m.content) for m in storico  # type: ignore[arg-type]
    ]
    messaggi.append(LLMMessage(role="user", content=payload.message))

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

    async def flusso() -> AsyncIterator[str]:
        yield sse("start", {
            "conversation_id": str(conversazione_id),
            "correlation_id": correlation_id,
        })

        pezzi: List[str] = []
        uso = None
        try:
            async for chunk in provider.stream(GenerationRequest(
                messages=messaggi, temperature=payload.temperature,
            )):
                if chunk.text:
                    pezzi.append(chunk.text)
                    yield sse("token", {"text": chunk.text})
                elif chunk.reasoning:
                    # Il ragionamento si segnala ma non si trasmette: serve a
                    # mostrare che il modello sta lavorando, non a essere letto.
                    yield sse("thinking", {})
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
        if risposta.strip():
            # Una sessione nuova: quella della richiesta è chiusa dalla
            # dipendenza appena l'endpoint restituisce la `StreamingResponse`,
            # mentre questo generatore continua a girare dopo. Usarla qui
            # significherebbe scrivere su una sessione chiusa — e l'errore si
            # manifesterebbe solo con risposte lunghe, cioè in modo
            # intermittente.
            async with session_factory() as scrittura:
                repo_scrittura = ConversationRepository(scrittura)
                conv = await scrittura.get(Conversation, conversazione_id)
                if conv is not None:
                    await repo_scrittura.add_message(
                        conv,
                        role="assistant",
                        content=risposta,
                        correlation_id=correlation_id,
                        tokens=uso.to_dict() if uso else None,
                    )
                    await scrittura.commit()

        yield sse("done", {
            "usage": uso.to_dict() if uso else None,
            "characters": len(risposta),
        })

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
) -> List[dict]:
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
