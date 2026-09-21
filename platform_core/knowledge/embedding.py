"""Vettorizzazione dei testi.

**I prefissi non sono un dettaglio.** Le famiglie di modelli di embedding sono
addestrate con istruzioni diverse davanti al testo — `search_document:` per il
documento, `search_query:` per l'interrogazione — e usare quelli sbagliati, o
non usarli affatto, peggiora il recupero senza che nulla lo segnali: i
risultati restano plausibili, semplicemente non sono i migliori. Peggio ancora
è usarli in modo asimmetrico, vettorizzando i documenti con il prefisso e le
domande senza: le due rappresentazioni finiscono in regioni diverse dello
spazio e la somiglianza misura la differenza fra i prefissi invece che fra i
significati.

La tabella dei profili è **copiata** da `bookwriter/config.py:221-243`: sono
dati misurati, e riscriverli a memoria significa sbagliarli.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

import httpx

from ..settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfiloEmbedding:
    """Come parlare a una famiglia di modelli di embedding."""

    max_tokens: int = 512
    prefisso_documento: str = ""
    prefisso_query: str = ""


#: Origine: `E:\\Projects\\bookwriter\\bookwriter\\config.py:221-243`.
#: La chiave è un frammento del nome del modello, confrontato in minuscolo.
PROFILI: List[Tuple[str, ProfiloEmbedding]] = [
    ("nomic-embed-text", ProfiloEmbedding(2048, "search_document: ", "search_query: ")),
    ("bge-m3", ProfiloEmbedding(8192, "", "")),
    ("bge-", ProfiloEmbedding(
        512, "", "Represent this sentence for searching relevant passages: ",
    )),
    ("multilingual-e5", ProfiloEmbedding(512, "passage: ", "query: ")),
    ("e5-", ProfiloEmbedding(512, "passage: ", "query: ")),
    ("qwen3-embedding", ProfiloEmbedding(
        8192, "",
        "Instruct: Given a search query, retrieve relevant passages\nQuery: ",
    )),
    ("gte-", ProfiloEmbedding(8192, "", "")),
    ("jina-embeddings", ProfiloEmbedding(8192, "", "")),
]

#: Usato quando il modello non è riconosciuto. Nessun prefisso, finestra
#: prudente: scegliere i prefissi di un'altra famiglia sarebbe peggio che non
#: usarne affatto.
PROFILO_IGNOTO = ProfiloEmbedding(512, "", "")


def profilo_per(modello: str) -> Optional[ProfiloEmbedding]:
    """Il profilo del modello, o `None` se sconosciuto."""
    nome = (modello or "").lower()
    for marcatore, profilo in PROFILI:
        if marcatore in nome:
            return profilo
    return None


class Embedder(Protocol):
    """Da testo a vettore."""

    modello: str
    dimensioni: int

    async def documenti(self, testi: Sequence[str]) -> List[List[float]]:
        ...

    async def query(self, testo: str) -> List[float]:
        ...


class EmbeddingError(Exception):
    pass


class OpenAICompatibleEmbedder(Embedder):
    """Vettorizzazione via endpoint in dialetto OpenAI.

    Vale per LM Studio, vLLM, TEI e OpenAI stesso.
    """

    def __init__(
        self,
        modello: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dimensioni: int = 1024,
        lotto: int = 32,
        settings: Optional[Settings] = None,
    ) -> None:
        settings = settings or get_settings()
        self.modello = modello or settings.embedding_model
        self._base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self._api_key = api_key or settings.llm_api_key
        self.dimensioni = dimensioni
        self._lotto = lotto

        profilo = profilo_per(self.modello)
        if profilo is None:
            logger.warning(
                "Modello di embedding non riconosciuto (%s): nessun prefisso. "
                "Il recupero funzionerà, ma non al meglio possibile per questa "
                "famiglia di modelli.",
                self.modello,
            )
        self.profilo = profilo or PROFILO_IGNOTO

    async def documenti(self, testi: Sequence[str]) -> List[List[float]]:
        if not testi:
            return []
        vettori: List[List[float]] = []
        for i in range(0, len(testi), self._lotto):
            porzione = testi[i : i + self._lotto]
            vettori.extend(
                await self._chiedi([self.profilo.prefisso_documento + t for t in porzione])
            )
        return vettori

    async def query(self, testo: str) -> List[float]:
        vettori = await self._chiedi([self.profilo.prefisso_query + testo])
        return vettori[0]

    async def _chiedi(self, testi: Sequence[str]) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            risposta = await client.post(
                f"{self._base_url}/embeddings",
                json={"model": self.modello, "input": list(testi)},
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            if risposta.status_code >= 400:
                raise EmbeddingError(
                    f"{risposta.status_code} da {self._base_url}: "
                    f"{risposta.text[:300]}"
                )
            corpo = risposta.json()

        # L'ordine dei risultati non è garantito dalla specifica: si riordina
        # per `index`. Su un lotto sbagliato l'errore sarebbe silenzioso — i
        # vettori finirebbero associati al testo sbagliato, e il recupero
        # restituirebbe passaggi plausibili ma casuali.
        dati = sorted(corpo["data"], key=lambda d: d.get("index", 0))
        vettori = [d["embedding"] for d in dati]

        if len(vettori) != len(testi):
            raise EmbeddingError(
                f"chiesti {len(testi)} vettori, ricevuti {len(vettori)}"
            )
        if vettori and len(vettori[0]) != self.dimensioni:
            raise EmbeddingError(
                f"il modello {self.modello} produce vettori di "
                f"{len(vettori[0])} dimensioni, lo schema ne attende "
                f"{self.dimensioni}: sono incompatibili, e mescolarli darebbe "
                f"risultati privi di senso invece di un errore"
            )
        return vettori


def somiglianza_coseno(a: Sequence[float], b: Sequence[float]) -> float:
    """Utilità per i test e le diagnosi. In produzione la calcola il database."""
    import math

    prodotto = sum(x * y for x, y in zip(a, b))
    norma_a = math.sqrt(sum(x * x for x in a))
    norma_b = math.sqrt(sum(y * y for y in b))
    if not norma_a or not norma_b:
        return 0.0
    return prodotto / (norma_a * norma_b)
