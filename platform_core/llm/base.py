"""Il contratto con cui il resto della piattaforma parla ai modelli.

Nessun modulo di dominio importa mai l'SDK di un fornitore: importa questo.
È la condizione perché il giorno in cui una personalità passa da un modello
locale a uno remoto — o viceversa, quando appare un acceleratore — la cosa
riguardi la configurazione e non il codice.

`stream()` è il metodo primario, non una variante di `complete()`. In una chat
il tempo al primo token è ciò che l'utente percepisce come velocità: costruire
l'interfaccia sulla generazione completa e aggiungere lo streaming dopo
significa riscriverla.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """Un turno, nella forma che i modelli si aspettano."""

    role: Role
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Usage:
    """Quanto è costato il turno.

    `cached_tokens` esiste perché senza di esso il vantaggio del prompt caching
    resta un'ipotesi: è l'unico numero che dice se lo strato stabile del prompt
    sia davvero stato riconosciuto come tale. Vale zero dove il fornitore non
    lo riporta, e zero è un'informazione — non un'assenza di dati.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
        }


@dataclass
class StreamChunk:
    """Un frammento di risposta.

    `reasoning` separato dal testo perché i modelli che ragionano ad alta voce
    emettono entrambi, e mescolarli manderebbe il pensiero in faccia
    all'utente. Il ragionamento si conserva per la traccia e non si mostra.
    """

    text: str = ""
    reasoning: str = ""
    done: bool = False
    usage: Optional[Usage] = None
    finish_reason: str = ""


class GenerationError(Exception):
    """Il fornitore non ha prodotto una risposta utilizzabile."""


class TruncatedResponse(GenerationError):
    """La generazione si è fermata per esaurimento del budget, non per fine.

    Va distinta da un errore di trasporto: la risposta *esiste* ma è tagliata,
    e chi chiama può decidere se riprovare con un budget più ampio. Confonderla
    con un successo è il modo in cui una risposta mozzata arriva all'utente
    come se fosse completa.
    """

    def __init__(self, message: str, partial: str = "", usage: Optional[Usage] = None):
        super().__init__(message)
        self.partial = partial
        self.usage = usage


@dataclass(frozen=True)
class CacheHint:
    """Come far riconoscere al motore la parte stabile del prompt.

    Neutro rispetto al fornitore: lo scrive una `ContextStrategy`, lo traduce
    ciascun fornitore nel proprio meccanismo — `cache_control` per Anthropic,
    `prompt_cache_key` per OpenAI, `cache_prompt` e uno slot fisso per
    llama.cpp — e chi non sa cosa farsene lo ignora. Il testo dei messaggi non
    cambia mai: l'indicazione dice *dove* finisce il prefisso e *come*
    chiamarlo, non cosa contiene.
    """

    #: `kv` (riuso della KV-cache su un motore locale), `prompt` (prompt
    #: caching del fornitore), `none`.
    modo: str = "none"
    #: Identificativo stabile del prefisso: lo stesso per tutte le richieste
    #: che condividono lo strato 0, cioè per la stessa versione della stessa
    #: personalità. Deriva dal testo e non da un id, così che due versioni con
    #: lo stesso prompt condividano la cache e una versione modificata no.
    chiave: str = ""
    #: Per llama.cpp: lo slot a cui inviare il prefisso, perché resti caldo.
    slot: Optional[int] = None
    #: Il nome della strategia che l'ha prodotta, per la traccia.
    strategia: str = ""


@dataclass
class GenerationRequest:
    """Cosa si chiede al modello.

    `cache_breakpoint_after` è l'indice del messaggio dopo il quale finisce la
    parte stabile del prompt. I fornitori che supportano il prompt caching lo
    traducono nel proprio meccanismo; gli altri lo ignorano senza che il
    chiamante debba saperlo.
    """

    messages: List[Message]
    model: str = ""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stop: List[str] = field(default_factory=list)
    cache_breakpoint_after: Optional[int] = None
    cache: Optional[CacheHint] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Ciò che un fornitore deve saper fare."""

    name: str

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Genera, un frammento per volta. L'ultimo ha `done=True`."""
        ...

    async def complete(self, request: GenerationRequest) -> str:
        """Genera e restituisce il testo intero. Comodità sopra `stream`."""
        ...

    async def available_models(self) -> List[str]:
        ...
