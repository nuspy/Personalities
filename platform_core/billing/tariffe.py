"""Quanto costa una risposta.

**Costo fisso per risposta, non per token.** È la scelta meno accurata e la
più utile: un utente che guarda il proprio saldo deve poter dire quante
risposte gli restano. Con un costo a token la stessa domanda posta due volte
toglie cifre diverse, e il saldo smette di voler dire qualcosa — mentre la
differenza che il conteggio esatto recupererebbe la paga comunque chi gestisce
la piattaforma, che i token li vede nelle tracce.

**Il costo sta sulla versione della personalità**, non sul piano: è la voce a
decidere quanto costa, perché è lei a scegliere il modello. Una personalità
servita da un modello grande costa di più a chiunque la usi, e metterlo sul
piano significherebbe dire che la stessa voce costa diversamente a due utenti.

`llm_config.costo_crediti`, quando c'è; altrimenti uno. Uno e non zero: una
risposta gratuita per omissione è il modo in cui una configurazione
dimenticata diventa un servizio regalato senza che nessuno lo decida.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Quanto costa una risposta quando la versione non dice altro.
COSTO_PREDEFINITO = 1

#: Oltre questo, un costo è quasi certamente un errore di configurazione.
#:
#: Non un limite tecnico: serve a impedire che un refuso in `llm_config` —
#: `1000` invece di `10` — svuoti il saldo di chi fa una domanda. Si registra
#: e si riporta al massimo, invece di applicarlo o di rifiutare la risposta.
COSTO_MASSIMO = 100


def costo_risposta(llm_config: Optional[Dict[str, Any]]) -> int:
    """Quanti crediti toglie una risposta di questa personalità."""
    if not llm_config:
        return COSTO_PREDEFINITO

    grezzo = llm_config.get("costo_crediti", COSTO_PREDEFINITO)
    try:
        costo = int(grezzo)
    except (TypeError, ValueError):
        logger.warning(
            "costo_crediti non numerico (%r): si applica %d",
            grezzo, COSTO_PREDEFINITO,
        )
        return COSTO_PREDEFINITO

    if costo < 0:
        logger.warning("costo_crediti negativo (%d): si applica 0", costo)
        return 0
    if costo > COSTO_MASSIMO:
        logger.error(
            "costo_crediti fuori scala (%d): si applica il massimo %d",
            costo, COSTO_MASSIMO,
        )
        return COSTO_MASSIMO
    return costo
