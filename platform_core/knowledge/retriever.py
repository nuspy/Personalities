"""Recupero ibrido.

Due ricerche, una fusione, e la traccia di ciò che è stato scartato.

**Perché due ricerche.** Il vettoriale trova un passaggio che dice la stessa
cosa con parole diverse — ed è il caso normale, perché nessuno interroga un
corpus con le parole esatte che vi sono scritte. Il lessicale trova il nome
proprio, la data, la citazione letterale: proprio i casi in cui la somiglianza
semantica è debole, perché per un modello di embedding due nomi diversi si
somigliano molto. Usato da solo, ciascuno dei due manca sistematicamente una
categoria di domande.

**Perché RRF e non una media dei punteggi.** I due punteggi non sono
commensurabili: una somiglianza coseno sta fra 0 e 1 con la massa attorno a
0,6-0,8, mentre `ts_rank_cd` non ha limite superiore e dipende dalla lunghezza
del documento. Normalizzarli richiede di conoscere la distribuzione, che
cambia da corpus a corpus e da domanda a domanda; una soglia tarata su un
corpus sarebbe sbagliata sul successivo. RRF ignora i punteggi e usa solo le
posizioni — l'unica cosa che i due metodi esprimono nella stessa unità.

    punteggio(d) = Σ  1 / (k + posizione_i(d))

`k` smorza il vantaggio delle primissime posizioni: senza, un risultato al
primo posto in una sola lista batterebbe qualunque risultato presente in
entrambe, e l'ibrido non servirebbe a nulla.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from ..observability.tracing import traccia
from .embedding import Embedder
from .vector_store import Corrispondenza, PgVectorStore, RicercaLessicale, VectorStore

logger = logging.getLogger(__name__)

#: La costante di smorzamento di RRF. 60 è il valore del lavoro originale
#: (Cormack, Clarke, Büttcher 2009) ed è rimasto lo standard perché funziona
#: senza taratura: è abbastanza grande da impedire che il primo posto di una
#: lista domini, abbastanza piccolo da non appiattire tutto.
K_RRF = 60

#: Quanti candidati chiedere a ciascuna ricerca prima di fondere. Più del
#: risultato finale, perché la fusione ha senso solo se ha di che scegliere: un
#: passaggio secondo nel vettoriale e terzo nel lessicale batte uno primo in
#: una sola lista, ma solo se entrambe le liste sono abbastanza lunghe da
#: contenerlo.
CANDIDATI_PER_RICERCA = 24


@dataclass
class PassaggioRecuperato:
    """Un passaggio scelto, con la storia di come è stato scelto."""

    corrispondenza: Corrispondenza
    punteggio_rrf: float
    posizione_vettoriale: Optional[int] = None
    posizione_lessicale: Optional[int] = None
    somiglianza: Optional[float] = None
    rilevanza_lessicale: Optional[float] = None

    #: L'etichetta con cui il passaggio compare nel prompt: `[K1]`, `[K2]`…
    #: È ciò su cui si regge il groundcheck deterministico della fase 2 — un
    #: marcatore citato che non esiste è un'allucinazione, e si rileva senza
    #: chiamare nessun modello.
    etichetta: str = ""

    #: La voce da cui viene il passaggio, quando non è quella che risponde:
    #: un `@Nome` nella domanda porta nel prompt i passaggi di un'altra
    #: personalità, e il modello deve sapere di chi sono.
    voce: Optional[str] = None

    @property
    def trovato_da_entrambe(self) -> bool:
        return (
            self.posizione_vettoriale is not None
            and self.posizione_lessicale is not None
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "etichetta": self.etichetta,
            "chunk_id": str(self.corrispondenza.chunk_id),
            "documento": self.corrispondenza.documento_titolo,
            "sezione": self.corrispondenza.sezione,
            "rrf": round(self.punteggio_rrf, 5),
            "pos_vettoriale": self.posizione_vettoriale,
            "pos_lessicale": self.posizione_lessicale,
            "somiglianza": round(self.somiglianza, 4) if self.somiglianza else None,
            **({"voce": self.voce} if self.voce else {}),
        }


@dataclass
class EsitoRecupero:
    """Ciò che è stato scelto e ciò che è stato lasciato fuori.

    Gli scartati non sono un lusso diagnostico: quando una risposta manca di
    un fatto che il corpus contiene, la domanda è sempre «il recupero non l'ha
    trovato, o l'ha trovato e scartato?» — e senza questa lista non ha
    risposta.
    """

    scelti: List[PassaggioRecuperato] = field(default_factory=list)
    scartati: List[PassaggioRecuperato] = field(default_factory=list)
    domanda: str = ""
    kb_interrogate: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domanda": self.domanda,
            "kb": self.kb_interrogate,
            "scelti": [p.to_dict() for p in self.scelti],
            # Solo i primi scartati: la coda lunga non aggiunge nulla e la
            # traccia finisce in una colonna che viene letta a mano.
            "scartati": [p.to_dict() for p in self.scartati[:8]],
        }


class Retriever:
    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        *,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._vettoriale = vector_store or PgVectorStore(session)
        # La configurazione linguistica non si passa qui: viaggia con ciascuna
        # base, ed e' la query a leggerla riga per riga.
        self._lessicale = RicercaLessicale(session)

    async def cerca(
        self,
        domanda: str,
        kb_ids: Sequence[uuid.UUID],
        *,
        limite: int = 6,
        candidati: int = CANDIDATI_PER_RICERCA,
    ) -> EsitoRecupero:
        if not kb_ids:
            return EsitoRecupero(domanda=domanda)

        with traccia("recupero.ibrido", kb=len(kb_ids), limite=limite):
            embedding = await self._embedder.query(domanda)

            # Sequenziali e non concorrenti: condividono la stessa sessione, e
            # una sessione SQLAlchemy non è sicura per l'uso simultaneo. Il
            # guadagno sarebbe di pochi millisecondi, il rischio è un errore
            # che si manifesta solo sotto carico.
            per_vettore = await self._vettoriale.cerca(
                embedding, kb_ids, limite=candidati,
            )
            per_parole = await self._lessicale.cerca(
                domanda, kb_ids, limite=candidati,
            )

        fusi = self._fondi(per_vettore, per_parole)

        scelti = fusi[:limite]
        for i, passaggio in enumerate(scelti, start=1):
            passaggio.etichetta = f"K{i}"

        logger.debug(
            "Recupero: %d vettoriali, %d lessicali, %d fusi, %d scelti",
            len(per_vettore), len(per_parole), len(fusi), len(scelti),
        )

        return EsitoRecupero(
            scelti=scelti,
            scartati=fusi[limite:],
            domanda=domanda,
            kb_interrogate=[str(k) for k in kb_ids],
        )

    def _fondi(
        self,
        per_vettore: Sequence[Corrispondenza],
        per_parole: Sequence[Corrispondenza],
    ) -> List[PassaggioRecuperato]:
        """Reciprocal Rank Fusion sulle due liste."""
        indice: Dict[uuid.UUID, PassaggioRecuperato] = {}

        for posizione, corrispondenza in enumerate(per_vettore, start=1):
            indice[corrispondenza.chunk_id] = PassaggioRecuperato(
                corrispondenza=corrispondenza,
                punteggio_rrf=1.0 / (K_RRF + posizione),
                posizione_vettoriale=posizione,
                somiglianza=corrispondenza.punteggio,
            )

        for posizione, corrispondenza in enumerate(per_parole, start=1):
            contributo = 1.0 / (K_RRF + posizione)
            esistente = indice.get(corrispondenza.chunk_id)
            if esistente is None:
                indice[corrispondenza.chunk_id] = PassaggioRecuperato(
                    corrispondenza=corrispondenza,
                    punteggio_rrf=contributo,
                    posizione_lessicale=posizione,
                    rilevanza_lessicale=corrispondenza.punteggio,
                )
            else:
                esistente.punteggio_rrf += contributo
                esistente.posizione_lessicale = posizione
                esistente.rilevanza_lessicale = corrispondenza.punteggio

        return sorted(
            indice.values(),
            # A parità di punteggio vince chi è stato trovato da entrambe le
            # ricerche: due metodi indipendenti che concordano sono un segnale
            # più forte di uno solo che insiste.
            key=lambda p: (p.punteggio_rrf, p.trovato_da_entrambe),
            reverse=True,
        )
