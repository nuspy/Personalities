"""Realizzazioni: coda, esecuzione, ciclo di vita.

Il lavoro pesante gira sul nodo con l'acceleratore; l'API si limita ad
accodare, e rifiuta subito se nessuno puo' raccogliere.
"""
from .queue import CodaBuild, CodaInMemoria, JobBuild
from .repository import BuildRepository, RuntimeRepository

__all__ = [
    "BuildRepository",
    "CodaBuild",
    "CodaInMemoria",
    "JobBuild",
    "RuntimeRepository",
]
