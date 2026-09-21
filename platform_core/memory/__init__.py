"""La memoria: cosa la piattaforma ricorda di chi le parla.

Quattro pezzi con quattro tempi diversi: si recupera a ogni turno, si estrae a
fine conversazione, si consolida periodicamente, e si dimentica su richiesta.
"""
from .consolidation import Consolidatore, EsitoConsolidamento, consolida_tutti
from .extraction import Estrattore, MemoriaEstratta
from .retrieval import MemoriaRecuperata, MemoryRetriever, frequenza, recenza
from .store import MemoryStore

__all__ = [
    "Consolidatore",
    "Estrattore",
    "EsitoConsolidamento",
    "MemoriaEstratta",
    "MemoriaRecuperata",
    "MemoryRetriever",
    "MemoryStore",
    "consolida_tutti",
    "frequenza",
    "recenza",
]
