"""Applicazione FastAPI.

Fase 0 dello scheletro verticale: salute del servizio, capacita' della
piattaforma e conversazione in streaming. Quest'ultima risponde ancora senza
personalita' — gli strati del prompt, il recupero e la memoria entrano nelle
fasi successive, nello stesso punto. Lo scopo qui e' dimostrare che il
percorso regge da capo a fondo: token, utente, conversazione, modello,
persistenza, traccia.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..observability.correlation import correlation_scope
from ..observability.tracing import (
    instrument_engine, setup_tracing, span_corrente_con_correlazione,
)
from ..settings import get_settings
from .routers import admin, builds, capabilities, chat, digestion, memory

logger = logging.getLogger(__name__)

#: Intestazione che lega fra loro tutte le tappe di una richiesta. La sezione
#: 21 della specifica chiede di poter seguire una domanda attraverso memoria,
#: recupero, modello, guardrail e sintesi vocale: senza un identificativo
#: comune, i tempi dei singoli stadi non si ricompongono.
CORRELATION_HEADER = "X-Correlation-ID"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for problem in settings.validate_production():
        logger.error("Configurazione non adatta alla produzione: %s", problem)

    logger.info(
        "Avvio di %s (ruolo=%s, ambiente=%s)",
        settings.service_name, settings.role, settings.environment,
    )

    if settings.auth_disabled:
        logger.warning(
            "Autenticazione disattivata: ogni endpoint risponde senza credenziali. "
            "Accettabile solo in sviluppo."
        )

    # Il motore si costruisce qui e non alla prima richiesta: cosi' un errore
    # nella stringa di connessione si vede all'avvio. La strumentazione di
    # FastAPI invece e' gia' stata agganciata in `create_app()`, prima che
    # Starlette congelasse la catena dei middleware.
    try:
        from ..domain.session import get_engine

        instrument_engine(get_engine())
    except Exception as exc:
        logger.error("Motore di database non inizializzato: %s", exc)

    yield

    from ..domain.session import dispose_engine

    await dispose_engine()
    logger.info("Arresto di %s", settings.service_name)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Personality Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[CORRELATION_HEADER],
    )

    @app.middleware("http")
    async def correlation_id(request: Request, call_next):
        """Assegna o propaga l'identificativo di correlazione.

        Se il chiamante ne porta uno lo si conserva — cosi' una richiesta
        iniziata nel frontend resta riconoscibile attraverso i servizi.
        """
        value = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())
        request.state.correlation_id = value

        # Anche nella variabile di contesto: `request.state` e' raggiungibile
        # solo da chi ha l'oggetto richiesta sottomano, e gli strati in fondo
        # — repository, client dei modelli — non ce l'hanno.
        with correlation_scope(value):
            span_corrente_con_correlazione()
            response = await call_next(request)

        response.headers[CORRELATION_HEADER] = value
        return response

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        """Il processo è vivo. Deliberatamente senza controlli sui supporti.

        Una sonda di liveness che verifica database e Redis fa riavviare un
        servizio sano quando è il database a essere lento, moltiplicando il
        guasto invece di contenerlo. La prontezza a ricevere traffico è
        un'altra domanda, e avrà il suo endpoint.
        """
        return {"status": "ok", "service": get_settings().service_name}

    app.include_router(capabilities.router)
    app.include_router(chat.router)
    app.include_router(admin.router)
    app.include_router(builds.router)
    app.include_router(memory.router)
    app.include_router(memory.admin_router)
    app.include_router(digestion.router)

    # Qui e non nel lifespan: la strumentazione di FastAPI inserisce un
    # middleware ASGI, e Starlette costruisce la catena dei middleware alla
    # prima richiesta — agganciarla dopo significa non agganciarla affatto,
    # senza che nulla lo segnali. Il sintomo e' sottile: le tracce arrivano
    # lo stesso, ma contengono solo le chiamate in uscita e mai la richiesta
    # che le ha causate.
    setup_tracing(app)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        """Nessuna traccia di stack verso il chiamante, ma il riferimento sì.

        L'identificativo restituito è lo stesso che compare nei log: è ciò che
        permette di ritrovare l'errore senza chiedere all'utente di
        descriverlo.
        """
        correlation = getattr(request.state, "correlation_id", "n/d")
        logger.exception("Errore non gestito (correlation_id=%s)", correlation)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "correlation_id": correlation},
        )

    return app


app = create_app()
