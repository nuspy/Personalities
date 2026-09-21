"""Tracciamento distribuito.

La specifica chiede di poter seguire una domanda attraverso memoria, recupero,
modello, guardrail e sintesi vocale, e di sapere quanto è costato ogni stadio.
Con quattro servizi e un worker, i log da soli non ci arrivano: dicono cosa è
successo in ciascun processo, non in quale ordine né dentro quale richiesta.

**Cosa viene strumentato e perché quello.** FastAPI dà lo span di ingresso;
SQLAlchemy mostra le query, che sono il primo sospettato di ogni lentezza
inspiegabile; httpx copre le chiamate ai modelli, che è dove il tempo passa
davvero. Il resto si aggiunge quando serve — la strumentazione non è gratis, e
uno span per ogni funzione rende le tracce illeggibili invece che informative.

**Se il collector non c'è**, il servizio parte lo stesso. L'osservabilità è
importante, ma non al punto di impedire a un'applicazione sana di rispondere:
in sviluppo capita continuamente di avviare l'API da sola.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..settings import Settings, get_settings
from .correlation import current_correlation_id

logger = logging.getLogger(__name__)

#: Attributo con cui l'identificativo di correlazione entra negli span. Il
#: `trace_id` di OpenTelemetry esiste già, ma è generato da OpenTelemetry: il
#: nostro attraversa anche ciò che non è strumentato — i log, le righe di
#: `messages`, il registro di audit — ed è quello che si incolla in una ricerca.
ATTRIBUTO_CORRELAZIONE = "persona.correlation_id"

_configurato = False


def setup_tracing(app=None, settings: Optional[Settings] = None) -> bool:
    """Avvia il tracciamento. Restituisce se è davvero attivo.

    Il valore di ritorno non è cortesia: senza, un errore di configurazione si
    manifesta come assenza di tracce — indistinguibile dall'assenza di
    traffico, e quindi non segnalato da nessuno.
    """
    global _configurato
    settings = settings or get_settings()

    if not settings.tracing_enabled:
        logger.info("Tracciamento disattivato da configurazione")
        return False

    if _configurato:
        return True

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning("OpenTelemetry non installato: nessuna traccia — %s", exc)
        return False

    risorsa = Resource.create({
        "service.name": settings.service_name,
        "service.version": "0.1.0",
        # Distingue API e worker nella stessa traccia: senza, gli span dei due
        # ruoli si mescolano sotto lo stesso nome di servizio.
        "service.role": settings.role,
        "deployment.environment": settings.environment,
    })

    provider = TracerProvider(resource=risorsa)
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{settings.otlp_endpoint.rstrip('/')}/v1/traces")
        )
    )
    trace.set_tracer_provider(provider)

    _configurato = True
    _strumenta(app)

    logger.info("Tracciamento attivo verso %s", settings.otlp_endpoint)
    return True


def instrument_engine(engine) -> None:
    """Aggancia il motore di database, se il tracciamento è attivo.

    Separato da `setup_tracing` per una questione di tempi: la strumentazione
    di FastAPI dev'essere installata **prima** che Starlette congeli la catena
    dei middleware, mentre il motore si costruisce all'avvio. Tenerle insieme
    obbligherebbe a scegliere quale delle due fare tardi.
    """
    if not _configurato or engine is None:
        return
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(
            engine=engine.sync_engine if hasattr(engine, "sync_engine") else engine
        )
        logger.info("Query del database tracciate")
    except Exception as exc:
        logger.warning("Strumentazione di SQLAlchemy non riuscita: %s", exc)


def _strumenta(app) -> None:
    """Aggancia le librerie, una alla volta e senza fermarsi ai guasti.

    Ogni strumentazione è indipendente: se quella di SQLAlchemy fallisce per
    una versione incompatibile, le richieste HTTP devono comunque essere
    tracciate. Un `try` unico attorno a tutte le farebbe cadere insieme.
    """
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(
                app,
                # Le sonde di salute sono interrogate ogni pochi secondi da
                # Kubernetes: tracciarle seppellirebbe il traffico vero.
                excluded_urls="healthz,readyz,metrics",
                # Senza questo, ogni messaggio ASGI diventa uno span: una
                # risposta in streaming ne produce uno per token, e una singola
                # conversazione arriva a centinaia di span vuoti che nascondono
                # i pochi che contano — e vengono tutti esportati.
                exclude_spans=["send", "receive"],
            )
        except Exception as exc:
            logger.warning("Strumentazione di FastAPI non riuscita: %s", exc)

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception as exc:
        logger.warning("Strumentazione di httpx non riuscita: %s", exc)


def span_corrente_con_correlazione() -> None:
    """Marca lo span in corso con l'identificativo di correlazione.

    Da chiamare all'inizio di una richiesta: lega la traccia a tutto ciò che
    non è strumentato ma porta lo stesso identificativo.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute(ATTRIBUTO_CORRELAZIONE, current_correlation_id())
    except Exception:  # pragma: no cover - mai far cadere una richiesta per questo
        pass


def traccia(nome: str, **attributi):
    """Contesto che apre uno span.

        with traccia("recupero.ibrido", kb_id=str(kb.id)):
            passaggi = await retriever.search(domanda)

    Se OpenTelemetry non è configurato non fa nulla, e il codice attorno non
    deve accorgersene: la strumentazione non può essere una condizione per
    l'esecuzione.
    """
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("platform_core")
        return tracer.start_as_current_span(nome, attributes=attributi)
    except Exception:  # pragma: no cover
        from contextlib import nullcontext

        return nullcontext()


def reset_tracing() -> None:
    """Solo per i test."""
    global _configurato
    _configurato = False
