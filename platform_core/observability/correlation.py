"""L'identificativo che tiene insieme le tappe di una richiesta.

Perché una variabile di contesto e non un parametro passato di funzione in
funzione: l'identificativo serve in fondo alla catena — nel repository che
scrive un messaggio, nel client del modello, nel registro di audit — e
portarcelo a mano significherebbe aggiungerlo alla firma di ogni funzione
attraversata, comprese quelle a cui non interessa. Un parametro del genere
viene dimenticato al primo strato nuovo, e la traccia si spezza senza che
niente lo segnali.

`ContextVar` si propaga da sé attraverso `await` e nei task figli, e resta
distinto fra richieste concorrenti — che è esattamente il comportamento
voluto, e che una variabile globale non avrebbe.
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from typing import Optional

_correlation_id: ContextVar[Optional[str]] = ContextVar(
    "persona_correlation_id", default=None
)


def current_correlation_id() -> str:
    """L'identificativo in corso, o uno nuovo se non c'è.

    Non restituisce mai una stringa vuota: un campo di correlazione vuoto in
    un log è peggio di uno inventato, perché sembra un dato mancante invece
    che un contesto assente — tipicamente un job del worker, che una richiesta
    HTTP non l'ha mai avuta.
    """
    valore = _correlation_id.get()
    if valore is None:
        valore = str(uuid.uuid4())
        _correlation_id.set(valore)
    return valore


def set_correlation_id(valore: str) -> Token:
    """Fissa l'identificativo. Il token restituito serve a ripristinarlo."""
    return _correlation_id.set(valore)


def reset_correlation_id(token: Token) -> None:
    _correlation_id.reset(token)


class correlation_scope:
    """Contesto con un identificativo proprio.

        with correlation_scope():          # il worker ne genera uno
            consolida_memorie(utente)

        with correlation_scope(id_della_richiesta):   # oppure lo eredita
            ...
    """

    def __init__(self, valore: Optional[str] = None) -> None:
        self._valore = valore or str(uuid.uuid4())
        self._token: Optional[Token] = None

    def __enter__(self) -> str:
        self._token = set_correlation_id(self._valore)
        return self._valore

    def __exit__(self, *_exc) -> None:
        if self._token is not None:
            reset_correlation_id(self._token)
