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
from sqlalchemy import select
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...auth.dependencies import CurrentUser, DbSession
from ...billing.credits import CreditiInsufficienti, RegistroCrediti
from ...billing.entitlements import puo_parlare_con
from ...billing.plans import GestoreAbbonamenti
from ...billing.quote import ContatoreQuote, QuotaSuperata
from ...billing.tariffe import costo_risposta
from ...api.deps import (
    aggiorna_assegnazioni, get_embedder, get_guardrail, get_llm_provider,
    provider_per,
)
from ...llm.compiti import Compito
from ...domain.knowledge_models import CommercialCategory, Personality, PersonalityVersion
from ...domain.lab_models import AnswerFeedback
from ...lab.esperimenti import GestoreEsperimenti
from ...domain.models import Conversation
from ...domain.repositories import (
    ConversationRepository, PersonalityRepository, TraceRepository,
)
from ...domain.session import SessionFactory
from ...guards.groundcheck import Giudice, controlla_citazioni
from ...guards.policy import RegistroGuardrail
from ...knowledge.embedding import Embedder
from ...knowledge.retriever import Retriever
from ...memory.retrieval import MemoryRetriever
from ...llm.base import (
    GenerationError, GenerationRequest, LLMProvider, Message as LLMMessage,
    TruncatedResponse,
)
from ...observability.correlation import current_correlation_id
from ...runtime.menzioni import menzioni_json
from ...runtime.menzioni import risolvi as risolvi_menzioni
from ...runtime.persona_engine import PersonaEngine, riferimenti_citati
from ...runtime.recupero_assistito import RecuperoAssistito

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
    # Chi serve quale compito può essere cambiato dalla console, e su
    # un'altra replica: il registro si rilegge da sé con una scadenza breve.
    # Qui e non prima perché serve una sessione; il modello della
    # conversazione è già stato risolto dalle dipendenze, quindi un cambio
    # vale dalla richiesta successiva — il giudice, risolto più sotto, lo
    # segue subito.
    await aggiorna_assegnazioni(session)

    repo = ConversationRepository(session)
    personalita_repo = PersonalityRepository(session)

    versione: Optional[PersonalityVersion] = None
    kb_ids: List[uuid.UUID] = []
    slug_personalita: Optional[str] = None

    if payload.personality:
        personalita = await personalita_repo.per_slug(payload.personality, per_utente=user.id)
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

    # -- limiti del piano ---------------------------------------------------
    #
    # Prima di creare la conversazione: un rifiuto per troppe conversazioni
    # aperte non deve lasciarne una in più. Solo per le personalità e solo
    # dove si fa pagare — senza catalogo l'installazione non addebita, e non
    # avrebbe senso contingentare ciò che regala.
    if versione is not None:
        gestore_quote = GestoreAbbonamenti(session)
        if await gestore_quote.tariffe_in_vigore():
            try:
                await ContatoreQuote(session).verifica_messaggio(
                    user.id,
                    await gestore_quote.diritti_di(user.id),
                    nuova_conversazione=not payload.conversation_id,
                )
            except QuotaSuperata as exc:
                # 429 per ciò che si libera col tempo, 403 per ciò che si
                # libera agendo: sono due indicazioni diverse per chi le legge.
                if exc.riprova_tra is not None:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=exc.messaggio,
                        headers={"Retry-After": str(exc.riprova_tra)},
                    ) from exc
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=exc.messaggio,
                ) from exc

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
        esperimento_id = None
        if versione is not None:
            # Un esperimento attivo sceglie la versione di chi apre la
            # conversazione — sempre la stessa per la stessa persona — e la
            # scelta resta sulla conversazione, come ogni versione.
            assegnata = await GestoreEsperimenti(session).versione_per(
                versione.personality_id, user.id,
            )
            if assegnata is not None:
                esperimento, versione = assegnata
                esperimento_id = esperimento.id
        conversazione = await repo.create(
            user, title=payload.message[:80].strip() or None
        )
        conversazione.experiment_id = esperimento_id
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

    # -- diritto e moneta ---------------------------------------------------
    #
    # Due controlli e non uno, in quest'ordine, perché si risolvono in modi
    # diversi: chi non ha il diritto deve cambiare piano, chi non ha crediti
    # deve aspettare il rinnovo o comprarne. Un solo «non puoi» li
    # confonderebbe, e chi lo riceve non saprebbe cosa fare.
    costo = 0
    if versione is not None:
        gestore = GestoreAbbonamenti(session)
        categoria = await _categoria_di(session, versione)
        verdetto = puo_parlare_con(await gestore.diritti_di(user.id), categoria)
        if not verdetto:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=verdetto.motivo,
                headers={"X-Serve-Piano": verdetto.serve} if verdetto.serve else None,
            )

        # Solo dove un catalogo esiste: senza piani in tabella questa
        # installazione non fa pagare, e addebitare comunque la renderebbe
        # muta per una tabella vuota.
        if await gestore.tariffe_in_vigore():
            costo = costo_risposta(versione.llm_config)
        else:
            logger.warning(
                "Nessun piano attivo in catalogo: la risposta di «%s» non "
                "viene addebitata. Esegui `python -m "
                "platform_core.tools.seed_piani`.",
                slug_personalita,
            )

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

    # I crediti si tolgono **prima** di generare, non dopo.
    #
    # Dopo sembrerebbe più giusto — si paga ciò che si è ricevuto — e
    # lascerebbe scoperta la porta: con il solo controllo in testa, dieci
    # richieste concorrenti dello stesso utente passano tutte, e il saldo
    # finisce sotto zero. Il consumo prende un lock sulla riga dell'utente e
    # verifica e sottrae nello stesso istante; ciò che non riesce si rimborsa,
    # con una riga propria, perché resti scritto che è stato tentato.
    if costo > 0:
        try:
            await RegistroCrediti(session).consuma(
                user.id, costo,
                conversation_id=conversazione.id,
                note=slug_personalita,
            )
            await session.commit()
        except CreditiInsufficienti as exc:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"Serve {exc.servono} credito per questa risposta, "
                    f"ne hai {exc.disponibili}."
                    if exc.servono == 1
                    else f"Servono {exc.servono} crediti per questa risposta, "
                         f"ne hai {exc.disponibili}."
                ),
            ) from exc

    conversazione_id = conversazione.id
    versione_id = versione.id if versione else None

    # Le altre voci chiamate con `@Nome`: con gli stessi diritti con cui si
    # parla a quella principale, o una menzione aggirerebbe il piano.
    menzioni, escluse = [], []
    if versione is not None:
        menzioni, escluse = await risolvi_menzioni(
            session, payload.message,
            user_id=user.id,
            diritti=await GestoreAbbonamenti(session).diritti_di(user.id),
            escludi=versione.personality_id,
        )

    motore = PersonaEngine(
        provider,
        retriever=Retriever(session, embedder) if kb_ids or menzioni else None,
        # Riscrive la domanda e sceglie i passaggi, dove la voce lo chiede:
        # un modello a parte, di norma più economico di quello che risponde.
        assistente=RecuperoAssistito(provider_per(Compito.RECUPERO)),
    )

    # Le memorie entrano nello strato 1, sotto il punto di cache: cambiano a
    # ogni turno, e metterle sopra annullerebbe lo sconto sul prefisso.
    memorie: List[str] = []
    memorie_usate = []
    if versione is not None and (versione.memory_config or {}).get("enabled", True):
        recuperatore = MemoryRetriever(session, embedder)
        memorie_usate = await recuperatore.cerca(
            user_id=user.id,
            domanda=payload.message,
            personality_id=versione.personality_id,
            limite=int((versione.memory_config or {}).get("max_memories", 5)),
        )
        memorie = [m.memoria.content for m in memorie_usate]
        if memorie_usate:
            # Usarle le rende insieme più recenti e più frequenti: è come
            # funziona ricordarsi di qualcosa.
            await recuperatore.segna_usate(memorie_usate)
            await session.commit()

    turno = None
    rubriche = []
    if versione is not None:
        turno = await motore.prepara(
            versione=versione,
            domanda=payload.message,
            kb_ids=kb_ids,
            storico=messaggi_storico,
            memorie=memorie,
            # La metà preventiva dei guardrail entra nello strato stabile:
            # costa una volta sola perché sta nel prefisso, e agisce prima che
            # il problema esista.
            politiche=guardrail.istruzioni_per(slug_personalita),
            menzioni=menzioni,
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

        if menzioni or escluse:
            # Chi è entrato nella risposta e chi no, e perché: una menzione
            # ignorata in silenzio sembra un difetto, una rifiutata col motivo
            # è un'informazione.
            yield sse("menzioni", menzioni_json(
                [m for m in menzioni if turno and m.nome in turno.menzioni], escluse,
            ))

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
                        "voce": p.voce,
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

        # Niente testo, niente addebito. Il rimborso è una riga in più e non
        # la cancellazione del consumo: chi legge il registro deve vedere che
        # una risposta è stata tentata, è fallita ed è stata restituita.
        # Cancellando, il saldo tornerebbe giusto senza spiegare nulla.
        if costo > 0 and not risposta.strip():
            async with session_factory() as rimborso:
                await RegistroCrediti(rimborso).rimborsa(
                    user.id, costo,
                    conversation_id=conversazione_id,
                    note="risposta non prodotta",
                )
                await rimborso.commit()

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
            # Il giudice del compito «giudizio», che di norma è un modello
            # diverso da quello che ha appena risposto: un giudice che è
            # anche l'autore assolve sé stesso.
            esito = await Giudice(provider_per(Compito.GIUDIZIO)).valuta(
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
                    # L'identificativo della risposta, per votarla: arriva per
                    # ultimo perché la riga esiste solo adesso.
                    yield sse("salvato", {"message_id": str(messaggio.id)})

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


async def _categoria_di(session, versione: PersonalityVersion) -> Optional[str]:
    """Lo slug della categoria commerciale della personalità, se ne ha una.

    Caricata qui e non insieme alla versione perché serve solo a questo
    controllo: una `join` in più su ogni turno di chi parla con una voce
    gratuita si pagherebbe per niente.
    """
    from ...domain.knowledge_models import Personality

    return await session.scalar(
        select(CommercialCategory.slug)
        .join(Personality, Personality.commercial_category_id == CommercialCategory.id)
        .where(Personality.id == versione.personality_id)
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
    # Lo slug della personalità in una sola query: serve al client per
    # riprendere la conversazione con la voce giusta, e una query per riga
    # renderebbe lenta proprio la pagina che elenca tutto.
    ids = {c.personality_id for c in conversazioni if c.personality_id}
    slug = dict((await session.execute(
        select(Personality.id, Personality.slug).where(Personality.id.in_(ids))
    )).all()) if ids else {}
    return [ConversationSummary.of(c, slug.get(c.personality_id)) for c in conversazioni]


@router.post("/conversations/{conversation_id}/archive")
async def archivia_conversazione(
    conversation_id: uuid.UUID, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    """Archivia una conversazione: resta leggibile, non conta più fra le aperte.

    Senza questa strada un piano che consente cinque conversazioni sarebbe
    bloccato per sempre alla sesta. Archiviare e non cancellare: le memorie
    estratte da quella conversazione vi rimandano, e cancellarla lascerebbe
    ricordi senza provenienza.
    """
    conversazione = await ConversationRepository(session).get(user, conversation_id)
    if conversazione is None:
        raise HTTPException(status_code=404, detail="Conversazione non trovata")
    conversazione.status = "archived"
    await session.commit()
    return {"id": str(conversazione.id), "status": conversazione.status}


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
    messaggi = await repo.messages(user, conversation_id)
    voti = dict((await session.execute(
        select(AnswerFeedback.message_id, AnswerFeedback.vote).where(
            AnswerFeedback.message_id.in_([m.id for m in messaggi]),
            AnswerFeedback.user_id == user.id,
        )
    )).all()) if messaggi else {}
    return [
        {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
            "voto": voti.get(m.id),
        }
        for m in messaggi
    ]


class Voto(BaseModel):
    voto: int = Field(description="1 approva, -1 disapprova")
    motivo: Optional[str] = Field(default=None, max_length=1000)


async def _risposta_propria(session, user, message_id: uuid.UUID):
    """La risposta, se è di una conversazione di chi chiede.

    404 anche quando esiste ma è di altri: distinguere rivelerebbe quali
    identificativi esistono.
    """
    from ...domain.models import Message

    messaggio = (await session.execute(
        select(Message)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(
            Message.id == message_id,
            Message.role == "assistant",
            Conversation.owner_id == user.id,
        )
    )).scalar_one_or_none()
    if messaggio is None:
        raise HTTPException(status_code=404, detail="Risposta non trovata")
    return messaggio


@router.put("/messages/{message_id}/feedback")
async def vota(
    message_id: uuid.UUID, payload: Voto, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    """Approva o disapprova una risposta. Rivotare cambia il voto."""
    if payload.voto not in (1, -1):
        raise HTTPException(status_code=422, detail="Il voto è 1 oppure -1.")
    await _risposta_propria(session, user, message_id)

    voto = await session.get(AnswerFeedback, message_id)
    if voto is None:
        voto = AnswerFeedback(message_id=message_id, user_id=user.id, vote=payload.voto)
        session.add(voto)
    voto.vote = payload.voto
    # Il motivo vale per il voto a cui si accompagna: passando da contrario a
    # favorevole, «troppo lunga» non descrive più niente.
    voto.reason = (payload.motivo or "").strip() or None
    await session.commit()
    return {"message_id": str(message_id), "voto": voto.vote}


@router.delete("/messages/{message_id}/feedback")
async def togli_voto(
    message_id: uuid.UUID, user: CurrentUser, session: DbSession,
) -> Dict[str, Any]:
    await _risposta_propria(session, user, message_id)
    voto = await session.get(AnswerFeedback, message_id)
    if voto is not None:
        await session.delete(voto)
        await session.commit()
    return {"message_id": str(message_id), "voto": None}
