"""Da una conversazione a ciò che vale la pena ricordare.

**L'estrazione avviene a fine conversazione e non a ogni turno.** Durante uno
scambio le cose si precisano: «penso di trasferirmi» diventa «mi trasferisco a
marzo» diventa «ho rimandato». Estrarre a ogni turno produrrebbe tre memorie
in conflitto fra loro, e il consolidamento passerebbe la vita a ricomporle.
Alla fine, invece, si guarda cosa è rimasto vero.

**Il modello estrae solo ciò che riguarda chi parla.** Non cosa ha detto la
personalità — quello è già nel corpus — e non i fatti del mondo, che non sono
memoria di nessuno. La distinzione è nel prompt, ed è la differenza fra un
sistema che ricorda una persona e uno che archivia trascrizioni.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..domain.memory_models import GENERI
from ..llm.base import GenerationError, GenerationRequest, Message
from ..observability.tracing import traccia

logger = logging.getLogger(__name__)

ISTRUZIONI = """Leggi la conversazione e individua ciò che vale la pena \
ricordare **di chi scrive** per le volte successive.

Estrai solo affermazioni su chi parla: chi è, cosa preferisce, cosa gli \
accade, cosa ha chiesto di ricordare. Non estrarre:

- ciò che ha detto l'altro interlocutore;
- fatti del mondo, storici o generali;
- il contenuto della conversazione in quanto tale;
- cose che valgono solo dentro questo scambio e non avranno senso fra un mese.

Per ciascuna scrivi una frase breve, in terza persona, comprensibile da sola \
fra sei mesi. «Vive a Vienna» e non «si è trasferito», che senza contesto non \
dice dove.

Classifica con uno di questi generi:

- `identita` — chi è: nome, lavoro, dove vive, lingua;
- `preferenza` — cosa vuole, come preferisce essere trattato;
- `fatto` — qualcosa che gli è accaduto o che lo riguarda;
- `impegno` — qualcosa che ha chiesto di ricordare per la volta dopo.

Assegna a ciascuna:

- `importanza` fra 0 e 1: quanto cambierebbe una risposta futura il saperlo;
- `confidenza` fra 0 e 1: 0,9 se l'ha detto esplicitamente, meno se lo hai \
dedotto.

Se non c'è nulla da ricordare, restituisci una lista vuota: è un esito \
normale e frequente, e inventare qualcosa per non tornare a mani vuote \
riempirebbe la memoria di rumore.

Rispondi in JSON:

{"memorie": [{"contenuto": "...", "genere": "fatto", "importanza": 0.6, \
"confidenza": 0.9}]}"""

RIASSUNTO = """Riassumi questa conversazione in tre o quattro frasi, come \
promemoria per riprenderla più avanti. Di cosa si è parlato, a che punto si \
è arrivati, cosa è rimasto in sospeso.

Rispondi in JSON: {"riassunto": "..."}"""


@dataclass
class MemoriaEstratta:
    contenuto: str
    genere: str = "fatto"
    importanza: float = 0.5
    confidenza: float = 0.8

    def valida(self) -> bool:
        """Vera se vale la pena conservarla.

        Le frasi troppo brevi non sono memorie: «sì», «va bene», «ok» passano
        attraverso l'estrazione quando il modello ha poco da dire, e
        riempirebbero la tabella di righe che non risponderanno mai a nulla.
        """
        return len(self.contenuto.strip()) >= 10 and self.genere in GENERI


class Estrattore:
    """Interroga un modello per capire cosa ricordare."""

    def __init__(self, provider, *, max_tokens: int = 1500) -> None:
        self._provider = provider
        self._max_tokens = max_tokens

    async def estrai(
        self, turni: Sequence[Message], *, nome_utente: Optional[str] = None
    ) -> List[MemoriaEstratta]:
        if not turni:
            return []

        trascrizione = _trascrizione(turni, nome_utente=nome_utente)
        if len(trascrizione) < 80:
            # Uno scambio di due battute non contiene memorie, e chiederlo a
            # un modello costa una chiamata per sentirsi dire di no.
            return []

        try:
            with traccia("memoria.estrazione", turni=len(turni)):
                dati = await self._provider.complete_json(GenerationRequest(
                    messages=[
                        Message(role="system", content=ISTRUZIONI),
                        Message(role="user", content=trascrizione),
                    ],
                    temperature=0.2,
                    max_tokens=self._max_tokens,
                ))
        except (GenerationError, ValueError) as exc:
            # Un'estrazione fallita non è un guasto della conversazione: si
            # perde ciò che si sarebbe ricordato, e si prosegue. Far fallire
            # la chiusura di uno scambio riuscito sarebbe sproporzionato.
            logger.warning("Estrazione delle memorie non riuscita: %s", exc)
            return []

        return _interpreta(dati)

    async def riassumi(self, turni: Sequence[Message]) -> str:
        """Il promemoria per riprendere la conversazione."""
        if len(turni) < 4:
            return ""

        try:
            dati = await self._provider.complete_json(GenerationRequest(
                messages=[
                    Message(role="system", content=RIASSUNTO),
                    Message(role="user", content=_trascrizione(turni)),
                ],
                temperature=0.2,
                max_tokens=600,
            ))
        except (GenerationError, ValueError) as exc:
            logger.warning("Riassunto non riuscito: %s", exc)
            return ""

        if isinstance(dati, dict):
            return str(dati.get("riassunto") or dati.get("summary") or "").strip()
        return ""


def _trascrizione(
    turni: Sequence[Message], *, nome_utente: Optional[str] = None
) -> str:
    chi = nome_utente or "L'utente"
    righe = []
    for t in turni:
        etichetta = chi if t.role == "user" else "La personalità"
        righe.append(f"{etichetta}: {t.content.strip()}")
    return "\n\n".join(righe)


def _interpreta(dati: Any) -> List[MemoriaEstratta]:
    """Legge il verdetto, tollerando le forme che i modelli producono."""
    if isinstance(dati, list):
        grezze = dati
    elif isinstance(dati, dict):
        grezze = dati.get("memorie") or dati.get("memories") or []
    else:
        return []

    estratte: List[MemoriaEstratta] = []
    for voce in grezze:
        if not isinstance(voce, dict):
            continue

        contenuto = str(
            voce.get("contenuto") or voce.get("content") or voce.get("text") or ""
        ).strip()
        if not contenuto:
            continue

        memoria = MemoriaEstratta(
            contenuto=contenuto,
            genere=_genere(voce.get("genere") or voce.get("kind")),
            importanza=_numero(voce.get("importanza", voce.get("importance")), 0.5),
            confidenza=_numero(voce.get("confidenza", voce.get("confidence")), 0.8),
        )
        if memoria.valida():
            estratte.append(memoria)

    return estratte


def _genere(valore: Any) -> str:
    """Il genere, ricadendo su `fatto` se non si riconosce.

    `fatto` e non `identita`: una cosa classificata male come fatto scade e
    pesa poco, mentre una classificata male come identità resta per sempre e
    pesa molto. Si sbaglia dalla parte che costa meno.
    """
    if not valore:
        return "fatto"
    normalizzato = str(valore).strip().lower()
    if normalizzato in GENERI:
        return normalizzato
    sinonimi = {
        "identity": "identita", "identità": "identita",
        "preference": "preferenza", "preferenze": "preferenza",
        "fact": "fatto", "fatti": "fatto",
        "commitment": "impegno", "promessa": "impegno", "task": "impegno",
        "session": "sessione", "summary": "sessione",
    }
    return sinonimi.get(normalizzato, "fatto")


def _numero(valore: Any, predefinito: float) -> float:
    try:
        return max(0.0, min(1.0, float(valore)))
    except (TypeError, ValueError):
        return predefinito
