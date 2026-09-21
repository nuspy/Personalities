"""Il motore: da una domanda a una risposta in carattere e ancorata.

Orchestra recupero, costruzione del prompt e generazione, e registra come la
risposta è nata.

**Sulla scelta del modo.** Una personalità può essere servita in tre modi —
`rag` (prompt e recupero), `lora` (pesi addestrati) o `finetune` — e la
differenza non riguarda solo dove si prende il carattere: riguarda quanto
prompt serve. Reiniettare il profilo stilistico completo sopra pesi già
addestrati raddoppia le istruzioni e produce una caricatura, ed è il rischio 4
del piano. Per questo `strato_persona` si riduce quando i pesi portano già la
voce.

**Sulla degradazione.** Se una personalità punta a una build che nessun worker
può servire — GPU spenta, adapter rimosso — si ricade su `rag` invece di
fallire, **dichiarandolo nella traccia**. Una risposta un po' meno in carattere
è meglio di nessuna risposta; una degradazione silenziosa, invece, è peggio di
entrambe.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from ..domain.knowledge_models import PersonalityVersion
from ..knowledge.retriever import EsitoRecupero, Retriever
from ..llm.base import GenerationRequest, LLMProvider, Message, StreamChunk, Usage
from ..observability.tracing import traccia
from .context_builder import (
    ContestoCostruito, ContextBuilder, StratoStabile, StratoVolatile,
    strato_volatile_da_recupero,
)

logger = logging.getLogger(__name__)

#: I modi in cui una personalità può essere servita.
MODI = ("rag", "lora", "finetune")


@dataclass
class EsitoTurno:
    """Cosa è successo in un turno, oltre al testo."""

    testo: str = ""
    usage: Optional[Usage] = None
    recupero: Optional[EsitoRecupero] = None
    contesto: Optional[ContestoCostruito] = None
    modo: str = "rag"
    modo_richiesto: str = "rag"
    motivo_degrado: str = ""
    tempi_ms: Dict[str, int] = field(default_factory=dict)

    @property
    def degradato(self) -> bool:
        return self.modo != self.modo_richiesto

    def traccia_risposta(self) -> Dict[str, Any]:
        """Il contenuto di `answer_traces` per questo turno."""
        return {
            "retrieved": self.recupero.to_dict() if self.recupero else None,
            "usage": {
                **(self.usage.to_dict() if self.usage else {}),
                "contesto": self.contesto.to_dict() if self.contesto else None,
            },
            "latency_ms": self.tempi_ms,
            "modo": {
                "usato": self.modo,
                "richiesto": self.modo_richiesto,
                "motivo": self.motivo_degrado,
            },
        }


def strato_stabile_da_versione(
    versione: PersonalityVersion,
    *,
    modo: str = "rag",
    politiche: Sequence[str] = (),
) -> StratoStabile:
    """Costruisce lo strato 0 da una versione pubblicata.

    Con `lora` o `finetune` il prompt si accorcia: i pesi portano già lessico e
    ritmo, e ripeterli a parole li spinge oltre il segno — il modello smette di
    scrivere *come* quella persona e comincia a imitarla.
    """
    regole = list((versione.behavior_rules or {}).get("regole", []))

    if modo in ("lora", "finetune"):
        prompt = (versione.behavior_rules or {}).get("prompt_ridotto") or ""
        if not prompt:
            # Nessuno strato ridotto preparato: si usa la prima frase del
            # prompt completo, che di norma è l'identità, senza le istruzioni
            # di stile che i pesi già contengono.
            prompt = versione.system_prompt.split("\n\n", 1)[0]
    else:
        prompt = versione.system_prompt

    return StratoStabile(
        prompt_personalita=prompt,
        regole=regole,
        politiche=list(politiche),
        documenti_integrali=list((versione.rag_config or {}).get("documenti_integrali", [])),
    )


class PersonaEngine:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        retriever: Optional[Retriever] = None,
        builder: Optional[ContextBuilder] = None,
    ) -> None:
        self._provider = provider
        self._retriever = retriever
        self._builder = builder or ContextBuilder()

    async def prepara(
        self,
        *,
        versione: PersonalityVersion,
        domanda: str,
        kb_ids: Sequence[uuid.UUID] = (),
        storico: Sequence[Message] = (),
        memorie: Sequence[str] = (),
        riassunto: str = "",
        modo: str = "rag",
        modi_disponibili: Sequence[str] = ("rag",),
        politiche: Sequence[str] = (),
    ) -> EsitoTurno:
        """Recupera e costruisce il prompt, senza ancora generare.

        Separato dalla generazione perché è la parte ispezionabile: un test —
        e la console di amministrazione — possono guardare il prompt esatto che
        sarebbe stato mandato, senza spendere una chiamata al modello.
        """
        esito = EsitoTurno(modo_richiesto=modo, modo=modo)

        if modo not in modi_disponibili:
            esito.modo = "rag"
            esito.motivo_degrado = (
                f"modo '{modo}' non servibile ora (disponibili: "
                f"{', '.join(modi_disponibili)}): risposta in RAG"
            )
            logger.info("Degradazione del modo: %s", esito.motivo_degrado)

        rag_config = versione.rag_config or {}
        limite = int(rag_config.get("max_chunks", 6))

        inizio = time.perf_counter()
        if self._retriever is not None and kb_ids and esito.modo == "rag":
            esito.recupero = await self._retriever.cerca(
                domanda, kb_ids, limite=limite,
            )
        esito.tempi_ms["recupero"] = int((time.perf_counter() - inizio) * 1000)

        volatile = (
            strato_volatile_da_recupero(
                esito.recupero, memorie=memorie, riassunto=riassunto,
            )
            if esito.recupero
            else StratoVolatile(memorie=list(memorie), riassunto_sessione=riassunto)
        )

        esito.contesto = self._builder.costruisci(
            stabile=strato_stabile_da_versione(
                versione, modo=esito.modo, politiche=politiche,
            ),
            volatile=volatile,
            storico=storico,
            domanda=domanda,
        )
        return esito

    async def rispondi_in_streaming(
        self, turno: EsitoTurno, *, versione: PersonalityVersion,
    ) -> AsyncIterator[StreamChunk]:
        """Genera la risposta, aggiornando `turno` mentre procede."""
        if turno.contesto is None:
            raise ValueError("chiamare prima `prepara()`")

        llm_config = versione.llm_config or {}
        richiesta = GenerationRequest(
            messages=turno.contesto.messaggi,
            model=llm_config.get("model", ""),
            temperature=float(llm_config.get("temperature", 0.7)),
            max_tokens=llm_config.get("max_tokens"),
            cache_breakpoint_after=turno.contesto.punto_di_cache,
        )

        pezzi: List[str] = []
        inizio = time.perf_counter()
        primo_token_a: Optional[float] = None

        with traccia(
            "generazione",
            modo=turno.modo,
            passaggi=len(turno.contesto.passaggi),
        ):
            async for chunk in self._provider.stream(richiesta):
                if chunk.text and primo_token_a is None:
                    primo_token_a = time.perf_counter()
                if chunk.text:
                    pezzi.append(chunk.text)
                if chunk.done:
                    turno.usage = chunk.usage
                yield chunk

        turno.testo = "".join(pezzi)
        turno.tempi_ms["generazione"] = int((time.perf_counter() - inizio) * 1000)
        if primo_token_a is not None:
            turno.tempi_ms["primo_token"] = int((primo_token_a - inizio) * 1000)


def riferimenti_citati(testo: str) -> set:
    """Le etichette `[Kn]` che compaiono in una risposta.

    Base del groundcheck deterministico: confrontate con i riferimenti
    effettivamente forniti, dicono se il modello ha citato qualcosa che non
    esisteva. Costa zero e non richiede un secondo modello — per questo è il
    primo dei due livelli della fase 2.
    """
    import re

    return set(re.findall(r"\[(K\d+)\]", testo))
