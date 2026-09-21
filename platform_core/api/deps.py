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


def reset_dependencies() -> None:
    """Dimentica le istanze memorizzate. Solo per i test."""
    get_key_value_store.cache_clear()
    get_capability_registry.cache_clear()
