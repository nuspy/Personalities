"""Dipendenze condivise dagli endpoint.

Tutto cio' che un endpoint riceve dall'esterno passa da qui: la connessione a
Redis, il registro delle capacita', piu' avanti la sessione di database e
l'identita' dell'utente. Averle in un punto solo e' cio' che permette a un
test di sostituirle senza avviare l'infrastruttura.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends

from ..capabilities.registry import CapabilityRegistry, InMemoryStore, KeyValueStore
from ..domain.session import get_db_session
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
def get_registro_modelli():
    """L'elenco dei modelli e chi serve quale compito.

    In cache come i fornitori che contiene: l'oggetto ricorda quale modello
    ciascun server ha caricato, e ricostruirlo a ogni domanda significherebbe
    chiedere di nuovo l'elenco dei modelli prima di ogni singola risposta —
    una chiamata di rete in piu' sul percorso piu' sensibile alla latenza che
    ci sia.

    Le assegnazioni le carica chi ha una sessione di database
    (`aggiorna_assegnazioni`): qui non se ne apre una, perche' questa
    funzione viene chiamata anche dal worker e dagli strumenti da riga di
    comando, dove un accesso al database all'import sarebbe una sorpresa.
    """
    from ..llm.compiti import RegistroModelli, modelli_da_impostazioni

    return RegistroModelli(modelli_da_impostazioni())


async def get_llm_provider(session=Depends(get_db_session)):
    """Il fornitore che risponde agli utenti.

    Resta con questo nome perche' e' quello che gli endpoint di chat
    chiedono, ed e' il compito piu' ovvio: «il modello», senza aggettivi, e'
    quello che parla.

    **Rilegge le assegnazioni prima di scegliere**, e per questo vuole una
    sessione. Senza, un processo appena avviato risolverebbe la prima
    richiesta su un registro che il database non l'ha ancora letto, e quella
    risposta uscirebbe dal modello predefinito invece che da quello
    assegnato: un errore per ogni replica a ogni rilascio, silenzioso perche'
    la risposta arriva comunque — dal modello sbagliato. Visto succedere
    davvero, dopo un ricaricamento a caldo.
    """
    from ..llm.compiti import Compito

    await aggiorna_assegnazioni(session)
    return get_registro_modelli().per(Compito.CONVERSAZIONE)


def provider_per(compito) -> object:
    """Il fornitore di un compito qualunque."""
    return get_registro_modelli().per(compito)


async def aggiorna_assegnazioni(session, *, forza: bool = False) -> None:
    """Rilegge dal database chi serve quale compito, se e' ora.

    Con piu' repliche dell'API una modifica fatta su una non arriva alle
    altre da sola, e nessuna se ne accorgerebbe: il registro ha una scadenza
    breve e si rinfresca da se'. Un errore qui non ferma la richiesta —
    l'assegnazione di prima e' vecchia di mezzo minuto, non sbagliata.
    """
    from ..domain.admin_repositories import assegnazioni_correnti
    from ..llm.compiti import Compito

    registro = get_registro_modelli()
    if not forza and not registro.da_rileggere():
        return

    try:
        righe = await assegnazioni_correnti(session)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Assegnazioni dei modelli non rilette: si continua con quelle "
            "in memoria", exc_info=True,
        )
        return

    valide = {}
    for compito in Compito:
        nome = righe.get(compito.value)
        if nome:
            valide[compito] = nome
    registro.aggiorna(valide)


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


@lru_cache(maxsize=1)
def get_guardrail():
    """I guardrail, letti una volta dal disco.

    In cache perche' sono file che non cambiano mentre il processo gira: in
    sviluppo basta riavviare, e in produzione arrivano con l'immagine. Se un
    giorno dovranno cambiare a caldo, questo e' il punto in cui invalidarli —
    non ogni chiamante.
    """
    from ..guards.policy import RegistroGuardrail

    return RegistroGuardrail()


def reset_dependencies() -> None:
    """Dimentica le istanze memorizzate. Solo per i test."""
    get_key_value_store.cache_clear()
    get_capability_registry.cache_clear()
    get_registro_modelli.cache_clear()
    get_embedder.cache_clear()
    get_guardrail.cache_clear()


@lru_cache(maxsize=1)
def get_tts():
    """Il fornitore di voce, condiviso fra le richieste.

    Con `tts_enabled` spento si usa quello muto. Non e' un ripiego di
    comodita': l'intero percorso — diritto, sintesi, allineamento, visemi,
    consegna — dev'essere provabile su una macchina senza modello di voce, e
    un endpoint che risponde 503 a ogni chiamata non prova niente.
    """
    from ..settings import get_settings
    from ..voice.tts import SintesiMuta, SintesiOpenAICompatibile

    if not get_settings().tts_enabled:
        logger.info(
            "Sintesi disattivata (`PERSONA_TTS_ENABLED`): si usa la voce muta."
        )
        return SintesiMuta()
    return SintesiOpenAICompatibile()


@lru_cache(maxsize=1)
def get_allineatore():
    """Chi misura i tempi delle parole, o `None` se non si puo'.

    `None` e non un oggetto che fallisce: chi chiama deve poter consegnare
    l'audio senza labiale invece di gestire un'eccezione per una funzione che
    e' facoltativa per natura. Il modello pesa centinaia di megabyte e si
    carica una volta sola, quindi la cache qui non e' un'ottimizzazione — e'
    cio' che rende praticabile rispondere a voce piu' di una volta.
    """
    from ..settings import get_settings
    from ..voice.allineamento import Allineatore

    if not get_settings().tts_align_words:
        return None

    allineatore = Allineatore()
    if not allineatore.disponibile():
        logger.warning(
            "faster-whisper non e' installato: la voce andra' senza labiale. "
            "I tempi delle parole non si possono stimare senza misurarli, e "
            "una bocca animata su una stima si vede fuori sincrono."
        )
        return None
    return allineatore


@lru_cache(maxsize=1)
def get_stt():
    """Chi trascrive la dettatura, o `None` se non si puo'.

    `None` e non un oggetto che fallisce: il riconoscimento del browser resta
    la via principale, e questo e' il ripiego per chi non ce l'ha. Un endpoint
    che risponde 503 dice a chi chiama di usare l'altra strada, cosa che
    un'eccezione generica non direbbe.
    """
    from ..voice.stt import AscoltoWhisper

    ascoltatore = AscoltoWhisper()
    if not ascoltatore.disponibile():
        logger.info(
            "faster-whisper non e' installato: la dettatura lato server non "
            "e' offerta. Il riconoscimento del browser continua a funzionare."
        )
        return None
    return ascoltatore
