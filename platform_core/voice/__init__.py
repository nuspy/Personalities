"""La voce: sintesi, riconoscimento, e le forme della bocca."""

from .base import (
    Ascolto, Sintesi, SintesiNonDisponibile, STTProvider, TTSProvider,
)
from .visemi import VISEMI, Parola, Viseme, forme_di, visemi_da

__all__ = [
    "Ascolto",
    "Parola",
    "Sintesi",
    "SintesiNonDisponibile",
    "STTProvider",
    "TTSProvider",
    "VISEMI",
    "Viseme",
    "forme_di",
    "visemi_da",
]
