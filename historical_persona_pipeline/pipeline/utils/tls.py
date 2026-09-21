"""Verifica TLS usando l'archivio certificati del sistema operativo.

Su molte reti aziendali e con diversi antivirus il traffico HTTPS passa da un
proxy che lo ri-firma con una CA propria. Quella CA e' installata nel sistema
ma non nel bundle di `certifi` che `requests` usa per default: il risultato e'
un `CERTIFICATE_VERIFY_FAILED` su qualunque richiesta, anche se il browser
sulla stessa macchina funziona.

`truststore` fa usare a Python l'archivio del sistema, quindi la verifica
continua a essere reale — l'alternativa diffusa (`verify=False`) disattiva
proprio il controllo che protegge la connessione, e non viene usata qui.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_injected = False

# Contesto TLS originale, catturato prima dell'iniezione di truststore.
# Serve a `prepare_model_download`: dopo l'iniezione `ssl.create_default_context`
# restituisce gia' un contesto agganciato ai certificati di sistema, quindi
# interrogarlo direbbe sempre che va tutto bene — anche dove le librerie native,
# che quei certificati non li usano, falliscono.
_original_context_factory = None


def enable_system_certificates() -> bool:
    """Attiva i certificati di sistema. Idempotente.

    Restituisce True se l'iniezione e' attiva, False se `truststore` non e'
    installato (in quel caso si resta sul bundle di certifi).
    """
    global _injected, _original_context_factory
    if _injected:
        return True

    try:
        import ssl

        import truststore
    except ImportError:
        logger.debug(
            "truststore non installato: si usa il bundle certifi. "
            "Se la rete usa un proxy TLS, installare 'truststore'."
        )
        return False

    # Va catturata *prima* dell'iniezione: dopo, e' la versione truststore.
    _original_context_factory = ssl.create_default_context

    try:
        truststore.inject_into_ssl()
    except Exception as exc:
        logger.warning(f"Attivazione dei certificati di sistema fallita: {exc}")
        return False

    _injected = True
    logger.debug("Verifica TLS agganciata all'archivio certificati di sistema")
    return True


_download_prepared = False


def prepare_model_download(probe_host: str = "huggingface.co") -> bool:
    """Assicura che i download di modelli funzionino su reti con proxy TLS.

    `truststore` copre solo il modulo `ssl` di Python. I downloader veloci di
    HuggingFace — `hf_xet` e `hf_transfer` — sono scritti in Rust e usano il
    proprio archivio di certificati: su una rete che rifirma il traffico HTTPS
    falliscono comunque, e l'errore arriva da dentro le librerie, dove non si
    puo' intervenire.

    Qui si verifica una volta sola se la verifica TLS standard passa. Se non
    passa, i downloader nativi vengono disattivati: il download procede in
    Python — piu' lento, ma funzionante. Una variabile gia' impostata
    dall'utente non viene mai sovrascritta.

    Restituisce True se i downloader nativi sono stati disattivati.
    """
    global _download_prepared
    if _download_prepared:
        return False

    _download_prepared = True

    import os
    import socket
    import ssl

    # Contesto con i certificati di default, *non* quelli di sistema: e'
    # quello che usano le librerie native. Dopo l'iniezione di truststore,
    # `ssl.create_default_context` e' gia' agganciato al sistema, quindi si
    # usa la versione catturata prima.
    factory = _original_context_factory or ssl.create_default_context
    try:
        context = factory()
        with socket.create_connection((probe_host, 443), timeout=8) as sock:
            with context.wrap_socket(sock, server_hostname=probe_host):
                return False  # la verifica standard basta: nulla da fare
    except ssl.SSLError:
        pass  # certificato non verificabile: e' il caso che ci interessa
    except OSError as exc:
        # Host irraggiungibile o rete assente: non e' un problema di
        # certificati, e disattivare i downloader non aiuterebbe.
        logger.debug(f"Sonda TLS verso {probe_host} non riuscita: {exc}")
        return False

    disabled = []
    for name, value in (("HF_HUB_DISABLE_XET", "1"), ("HF_HUB_ENABLE_HF_TRANSFER", "0")):
        if name not in os.environ:
            os.environ[name] = value
            disabled.append(name)

    if disabled:
        logger.info(
            "La verifica TLS passa solo con i certificati di sistema: "
            "disattivo i downloader nativi di HuggingFace "
            f"({', '.join(disabled)}), che non li usano. "
            "I download saranno piu' lenti ma funzioneranno."
        )
    return bool(disabled)
