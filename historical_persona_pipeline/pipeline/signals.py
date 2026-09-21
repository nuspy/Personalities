"""Segnali senza Qt.

Gli stadi della pipeline comunicano l'avanzamento emettendo segnali. Finora
quei segnali erano `pyqtSignal`, e bastava importare uno stadio per tirarsi
dietro PyQt6 — in un worker che gira in un container senza interfaccia
grafica, è una dipendenza da centinaia di megabyte per una cosa che qui fa il
lavoro di dieci righe.

**L'interfaccia è deliberatamente identica a quella di Qt** — `connect`,
`disconnect`, `emit` — perché gli stadi non debbano cambiare. La pipeline
funziona già ed è coperta da test: riscriverne le chiamate per una ragione
architetturale significherebbe rischiare di romperla per nulla.

Non è thread-safe per scelta: un segnale si collega quando si costruisce lo
stadio e si emette mentre gira, e proteggere ogni emissione con un lock
costerebbe su ogni riga di avanzamento per un caso che non esiste.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List

logger = logging.getLogger(__name__)


class Segnale:
    """Un punto di emissione a cui si possono agganciare ascoltatori.

        avanzamento = Segnale()
        avanzamento.connect(lambda pct, msg: print(f"{pct}% {msg}"))
        avanzamento.emit(50, "a metà")
    """

    __slots__ = ("_ascoltatori", "_nome")

    def __init__(self, nome: str = "") -> None:
        self._ascoltatori: List[Callable[..., Any]] = []
        self._nome = nome

    def connect(self, ascoltatore: Callable[..., Any]) -> None:
        """Aggancia un ascoltatore.

        Accetta anche un altro `Segnale`, così una catena di inoltri — come
        quella che porta l'avanzamento di uno stadio fino a chi lo osserva —
        si scrive `stadio.avanzamento.connect(mio_segnale)` esattamente come
        con Qt.
        """
        if isinstance(ascoltatore, Segnale):
            altro = ascoltatore
            self._ascoltatori.append(lambda *a, **k: altro.emit(*a, **k))
        else:
            self._ascoltatori.append(ascoltatore)

    def disconnect(self, ascoltatore: Callable[..., Any] | None = None) -> None:
        if ascoltatore is None:
            self._ascoltatori.clear()
        elif ascoltatore in self._ascoltatori:
            self._ascoltatori.remove(ascoltatore)

    def emit(self, *args: Any, **kwargs: Any) -> None:
        """Avvisa gli ascoltatori.

        Un ascoltatore che solleva non ferma gli altri e non ferma lo stadio:
        l'avanzamento è un'informazione accessoria, e far cadere
        un'elaborazione di venti minuti perché una barra di progresso ha
        sbagliato un conto sarebbe sproporzionato.
        """
        for ascoltatore in list(self._ascoltatori):
            try:
                ascoltatore(*args, **kwargs)
            except Exception:
                logger.exception(
                    "Ascoltatore del segnale %s fallito", self._nome or "anonimo",
                )

    def __len__(self) -> int:
        return len(self._ascoltatori)

    def __repr__(self) -> str:  # pragma: no cover - diagnostica
        return f"<Segnale {self._nome or 'anonimo'} ascoltatori={len(self._ascoltatori)}>"
