"""Come far pagare meno il prefisso stabile del prompt.

Il prompt si costruisce sempre allo stesso modo — strato stabile in testa,
volatile in coda, punto di cache fra i due (vedi `context_builder`). Cambia
**chi ne trae vantaggio e come**, e questa è la decisione presa qui:

| Strategia | Dove | Cosa succede davvero |
|---|---|---|
| `KVCacheStrategy` | motore locale: llama.cpp, vLLM, LM Studio | i blocchi KV del prefisso si riusano: il prefisso non si ricalcola affatto (CAG vero) |
| `PromptCacheStrategy` | fornitore cloud con caching: Anthropic, OpenAI | i token del prefisso si pagano una frazione: esistono, ma costano meno |
| `NoCacheStrategy` | tutti gli altri | prezzo pieno, stessa struttura — pronta per il giorno in cui si cambia fornitore |

**Una strategia non tocca mai il testo.** Scrive un `CacheHint` — dove finisce
il prefisso, come si chiama, a quale slot mandarlo — e il fornitore lo
traduce. Se una strategia potesse riscrivere i messaggi, due strategie
diverse produrrebbero due prompt diversi per la stessa personalità, e il test
che garantisce lo strato 0 byte-identico non basterebbe più.

**La chiave deriva dal testo stabile, non da un identificativo.** Due versioni
con lo stesso prompt condividono la cache; una versione modificata ne apre
una nuova. Legarla all'id della versione farebbe perdere la cache a ogni
pubblicazione anche quando il prompt non è cambiato, e — peggio — la farebbe
conservare a una versione il cui prompt è stato modificato sul posto.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import Optional, Protocol

from ..llm.base import CacheHint, GenerationRequest

logger = logging.getLogger(__name__)

#: I motori locali che riusano davvero la KV-cache del prefisso.
MOTORI_KV = ("llamacpp", "vllm", "lmstudio")

#: I fornitori cloud che scontano un prefisso ripetuto.
FORNITORI_PROMPT_CACHE = ("anthropic", "openai")


def chiave_del_prefisso(testo_stabile: str) -> str:
    """Un identificativo corto e stabile del prefisso.

    SHA-256 e non `hash()`: quello di Python cambia a ogni avvio del processo
    (per difesa contro gli attacchi alle tabelle hash), e due repliche
    dell'API assegnerebbero chiavi diverse allo stesso prompt — cioè slot
    diversi e cache fredde.
    """
    return hashlib.sha256(testo_stabile.encode("utf-8")).hexdigest()[:24]


class ContextStrategy(Protocol):
    """Decide come il prefisso stabile viene riconosciuto dal motore."""

    name: str

    def applica(self, richiesta: GenerationRequest, *, testo_stabile: str) -> GenerationRequest:
        """La stessa richiesta, con l'indicazione di cache per questo motore."""
        ...


class KVCacheStrategy:
    """Riuso vero della KV-cache, su un motore locale.

    Per vLLM e LM Studio il riuso è automatico: il motore riconosce da sé un
    prefisso già calcolato, e l'unico compito della strategia è che il
    prefisso sia identico — cosa che la costruzione del prompt garantisce già.

    Per llama.cpp servono due cose in più: chiedere di conservare lo stato del
    prompt (`cache_prompt`) e mandare lo stesso prefisso **sempre allo stesso
    slot**. Con più slot in parallelo, una richiesta che finisce su uno slot
    diverso da quello che ha già il prefisso lo ricalcola da capo: il riuso
    c'è sulla carta e non nei fatti.
    """

    name = "kv-cache"

    def __init__(self, motore: str, *, slot: int = 1) -> None:
        if motore not in MOTORI_KV:
            raise ValueError(f"motore senza riuso della KV-cache: {motore}")
        self.motore = motore
        self._slot = max(1, int(slot))

    def applica(self, richiesta: GenerationRequest, *, testo_stabile: str) -> GenerationRequest:
        chiave = chiave_del_prefisso(testo_stabile)
        slot = None
        if self.motore == "llamacpp":
            slot = int(chiave[:8], 16) % self._slot
        return replace(richiesta, cache=CacheHint(
            modo="kv", chiave=chiave, slot=slot, strategia=f"{self.name}:{self.motore}",
        ))


class PromptCacheStrategy:
    """Prompt caching di un fornitore cloud.

    Non elimina i token del prefisso: li sconta, e solo se il prefisso è
    byte-identico e le richieste abbastanza ravvicinate da trovarlo ancora in
    cache (per Anthropic, cinque minuti). Il vantaggio si misura: i token
    letti da cache finiscono nella traccia di ogni risposta, e senza quel
    numero il risparmio resterebbe un'ipotesi.
    """

    name = "prompt-cache"

    def __init__(self, fornitore: str) -> None:
        if fornitore not in FORNITORI_PROMPT_CACHE:
            raise ValueError(f"fornitore senza prompt caching: {fornitore}")
        self.fornitore = fornitore

    def applica(self, richiesta: GenerationRequest, *, testo_stabile: str) -> GenerationRequest:
        return replace(richiesta, cache=CacheHint(
            modo="prompt",
            chiave=chiave_del_prefisso(testo_stabile),
            strategia=f"{self.name}:{self.fornitore}",
        ))


class NoCacheStrategy:
    """Nessun riuso: prezzo pieno.

    Esiste perché «nessuna strategia» e «strategia che non fa nulla» non sono
    la stessa cosa nella traccia. La seconda dice esplicitamente che quella
    risposta non ha beneficiato di alcuna cache, e perché — che è l'unica
    informazione utile quando qualcuno chiede come mai costa così tanto.
    """

    name = "no-cache"

    def applica(self, richiesta: GenerationRequest, *, testo_stabile: str) -> GenerationRequest:
        return replace(richiesta, cache=CacheHint(
            modo="none",
            chiave=chiave_del_prefisso(testo_stabile),
            strategia=self.name,
        ))


def scegli_strategia(provider: object, *, slot: int = 1) -> ContextStrategy:
    """La strategia adatta al fornitore in uso.

    Ciascun fornitore dichiara il proprio `motore`: è lui a sapere a cosa sta
    parlando, e dedurlo qui dal nome della classe legherebbe la scelta a un
    dettaglio che cambia. Un fornitore che non lo dichiara ottiene la
    strategia senza cache — mai una che prometta un riuso che non avverrà.
    """
    motore: Optional[str] = getattr(provider, "motore", None)

    if motore in MOTORI_KV:
        return KVCacheStrategy(motore, slot=slot)
    if motore in FORNITORI_PROMPT_CACHE:
        return PromptCacheStrategy(motore)

    if motore:
        logger.info("Motore «%s» senza caching noto: prezzo pieno", motore)
    return NoCacheStrategy()
