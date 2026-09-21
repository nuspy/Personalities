"""Dipendenze condivise dagli endpoint.

Tutto cio' che un endpoint riceve dall'esterno passa da qui: la connessione a
Redis, il registro delle capacita', piu' avanti la sessione di database e
l'identita' dell'utente. Averle in un punto solo e' cio' che permette a un
test di sostituirle senza avviare l'infrastruttura.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from ..capabilities.registry import CapabilityRegistry, InMemoryStore, KeyValueStore
from ..settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_key_value_store() -> KeyValueStore:
    """Redis, o un supporto in memoria se non e' raggiungibile.

    In sviluppo capita di avviare l'API senza Redis. Fermarsi renderebbe
    impossibile provare gli endpoint che non ne hanno bisogno, quindi si
    prosegue in memoria — ma la cosa **va detta**, perche' in quel modo le
    capacita' annunciate da altri processi non si vedono, e un'indisponibilita'
    inspiegabile e' peggio di un avviso.
    """
    settings = get_settings()
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        logger.info("Registro delle capacita' su Redis (%s)", settings.redis_url)
        return client
    except Exception as exc:
        if settings.environment == "prod":
            # In produzione i worker stanno altrove: un registro locale li
            # renderebbe invisibili e la piattaforma direbbe di non saper fare
            # nulla, senza che nulla sia rotto.
            raise RuntimeError(
                f"Redis non raggiungibile ({settings.redis_url}): in produzione "
                f"il registro delle capacita' non puo' essere locale — {exc}"
            ) from exc

        logger.warning(
            "Redis non raggiungibile (%s): registro in memoria, le capacita' "
            "annunciate da altri processi non saranno visibili — %s",
            settings.redis_url, exc,
        )
        return InMemoryStore()


@lru_cache(maxsize=1)
def get_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry(get_key_value_store())


@lru_cache(maxsize=1)
def get_llm_provider():
    """Il fornitore di generazione, condiviso fra le richieste.

    Condiviso e non creato ogni volta: l'oggetto ricorda quale modello il
    server ha caricato, e costruirne uno nuovo a ogni domanda significa
    chiedere di nuovo l'elenco dei modelli prima di ogni singola risposta —
    una chiamata di rete in piu' sul percorso piu' sensibile alla latenza che
    ci sia. Non tiene connessioni aperte: il client HTTP nasce e muore dentro
    ciascuna generazione.
    """
    from ..llm.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider()


@lru_cache(maxsize=1)
def get_embedder():
    """Il vettorizzatore, condiviso fra le richieste.

    Non tiene stato fra una chiamata e l'altra — solo il profilo di prefissi
    del modello, che si ricava una volta sola. Condividerlo evita di rifare
    quel lavoro, e soprattutto di ripetere l'avviso sul modello non
    riconosciuto a ogni domanda.
    """
    from ..knowledge.embedding import OpenAICompatibleEmbedder

    return OpenAICompatibleEmbedder()


def reset_dependencies() -> None:
    """Dimentica le istanze memorizzate. Solo per i test."""
    get_key_value_store.cache_clear()
    get_capability_registry.cache_clear()
    get_llm_provider.cache_clear()
    get_embedder.cache_clear()
