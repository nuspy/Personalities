"""Guardrail e verifica di fondatezza.

Due momenti distinti: la prevenzione entra nel prompt del personaggio, la
rilevazione avviene dopo e non passa mai da lui.
"""
from .claims import Affermazione, EsitoGroundcheck, Tipo
from .groundcheck import Giudice, controlla_citazioni, riferimenti_citati
from .policy import Guardrail, RegistroGuardrail, carica_da, interpreta

__all__ = [
    "Affermazione",
    "EsitoGroundcheck",
    "Giudice",
    "Guardrail",
    "RegistroGuardrail",
    "Tipo",
    "carica_da",
    "controlla_citazioni",
    "interpreta",
    "riferimenti_citati",
]
