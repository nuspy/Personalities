"""Ricerca per somiglianza, dietro un'interfaccia.

Il `Protocol` esiste perché un giorno i vettori potrebbero non stare più in
Postgres — con decine di milioni di passaggi un archivio dedicato ha vantaggi
reali — e quel giorno la cosa deve riguardare questo file e nessun altro.

Oggi però stanno lì, e non per ripiego: i vettori vivono nella stessa
transazione dei dati relazionali, quindi un documento e i suoi passaggi
compaiono insieme o non compaiono affatto. Con due archivi separati esiste
sempre una finestra in cui il testo c'è e il vettore no, e quella finestra si
manifesta come un passaggio che esiste ma non si trova mai.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class Corrispondenza:
    """Un passaggio trovato, con il suo punteggio grezzo."""

    chunk_id: uuid.UUID
    testo: str
    ordinale: int
    sezione: Optional[str]
    documento_id: uuid.UUID
    documento_titolo: str
    documento_uri: Optional[str]
    kb_id: uuid.UUID
    punteggio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "documento": self.documento_titolo,
            "sezione": self.sezione,
            "punteggio": round(self.punteggio, 4),
        }


class VectorStore(Protocol):
    async def cerca(
        self,
        embedding: Sequence[float],
        kb_ids: Sequence[uuid.UUID],
        *,
        limite: int = 20,
    ) -> List[Corrispondenza]:
        ...


class PgVectorStore(VectorStore):
    """Ricerca vettoriale su pgvector."""

    def __init__(self, session: AsyncSession, *, ef_search: int = 80) -> None:
        self._session = session
        #: Quanti candidati l'indice HNSW esamina. Più alto significa richiamo
        #: migliore e ricerca più lenta; il valore predefinito di pgvector (40)
        #: è prudente per corpora grandi, qui si preferisce non perdere
        #: passaggi — su un corpus di poche migliaia la differenza di tempo è
        #: impercettibile.
        self._ef_search = ef_search

    async def cerca(
        self,
        embedding: Sequence[float],
        kb_ids: Sequence[uuid.UUID],
        *,
        limite: int = 20,
    ) -> List[Corrispondenza]:
        if not kb_ids:
            return []

        await self._session.execute(
            text(f"SET LOCAL hnsw.ef_search = {int(self._ef_search)}")
        )

        # `<=>` è la distanza coseno: 0 identico, 2 opposto. Si converte in
        # somiglianza perché tutto il resto del sistema ragiona in «più alto è
        # meglio», e mescolare le due convenzioni è il modo tipico di ordinare
        # i risultati al contrario.
        query = text("""
            SELECT
                c.id, c.text, c.ordinal, c.section,
                d.id AS documento_id, d.title, d.uri,
                c.kb_id,
                1 - (v.embedding <=> CAST(:embedding AS vector)) AS somiglianza
            FROM chunk_vectors v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE v.kb_id = ANY(:kb_ids)
            ORDER BY v.embedding <=> CAST(:embedding AS vector)
            LIMIT :limite
        """)

        risultato = await self._session.execute(
            query,
            {
                "embedding": _come_letterale(embedding),
                "kb_ids": list(kb_ids),
                "limite": limite,
            },
        )

        return [
            Corrispondenza(
                chunk_id=r.id, testo=r.text, ordinale=r.ordinal, sezione=r.section,
                documento_id=r.documento_id, documento_titolo=r.title,
                documento_uri=r.uri, kb_id=r.kb_id, punteggio=float(r.somiglianza),
            )
            for r in risultato
        ]


class RicercaLessicale:
    """Ricerca per parole, sulla colonna `tsv`.

    Il vettoriale trova ciò che è detto con altre parole; questa trova la
    parola esatta. Servono entrambe: un nome proprio, una data, una citazione
    letterale sono proprio i casi in cui la somiglianza semantica è debole —
    per un modello di embedding due nomi diversi si somigliano molto, ed è
    l'opposto di ciò che serve quando la domanda riguarda una persona precisa.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def cerca(
        self,
        domanda: str,
        kb_ids: Sequence[uuid.UUID],
        *,
        limite: int = 20,
    ) -> List[Corrispondenza]:
        if not kb_ids or not domanda.strip():
            return []

        # La domanda viene interpretata **con la configurazione di ciascuna
        # base**, presa dalla riga: le basi interrogate insieme possono essere
        # in lingue diverse, e una configurazione unica ne tradirebbe almeno
        # una. Interpretare la domanda diversamente da come sono indicizzati i
        # documenti significa non trovarli — in silenzio.
        #
        # Il cast a `regconfig` e' necessario: la funzione non accetta una
        # stringa, vuole il tipo che nomina una configurazione di ricerca.
        #
        # **La congiunzione diventa disgiunzione.** `websearch_to_tsquery`
        # unisce i termini in AND, e su una domanda in lingua naturale — sette
        # o otto parole — nessun passaggio le contiene tutte: la ricerca
        # lessicale restituisce sistematicamente zero risultati e l'ibrido si
        # riduce al solo vettoriale, senza che nulla lo segnali. Misurato su
        # questo corpus: 0 corrispondenze in AND, 55 in OR.
        #
        # La sostituzione avviene sulla query gia' costruita invece che sul
        # testo della domanda, cosi' la sanificazione dell'input resta a
        # PostgreSQL: si eredita la tolleranza di `websearch_to_tsquery` alla
        # punteggiatura e alle virgolette, e le frasi fra virgolette — che
        # usano `<->` e non `&` — restano frasi.
        #
        # A scremare pensa `ts_rank_cd`, che premia chi contiene piu' termini e
        # piu' vicini fra loro: e' l'ordinamento a fare la selezione, non il
        # filtro.
        query = text("""
            SELECT
                c.id, c.text, c.ordinal, c.section,
                d.id AS documento_id, d.title, d.uri,
                c.kb_id,
                ts_rank_cd(c.tsv, q.query) AS rilevanza
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN knowledge_bases kb ON kb.id = c.kb_id
            CROSS JOIN LATERAL (
                SELECT replace(
                    websearch_to_tsquery(kb.text_config::regconfig, :domanda)::text,
                    '&', '|'
                )::tsquery AS query
            ) q
            WHERE c.kb_id = ANY(:kb_ids)
              AND q.query IS NOT NULL
              AND c.tsv @@ q.query
            ORDER BY rilevanza DESC
            LIMIT :limite
        """)

        risultato = await self._session.execute(
            query,
            {
                "domanda": domanda,
                "kb_ids": list(kb_ids),
                "limite": limite,
            },
        )

        return [
            Corrispondenza(
                chunk_id=r.id, testo=r.text, ordinale=r.ordinal, sezione=r.section,
                documento_id=r.documento_id, documento_titolo=r.title,
                documento_uri=r.uri, kb_id=r.kb_id, punteggio=float(r.rilevanza),
            )
            for r in risultato
        ]


def _come_letterale(embedding: Sequence[float]) -> str:
    """Il vettore nella forma che pgvector accetta come parametro.

    Passare una lista Python direttamente funziona solo se il tipo è
    registrato sulla connessione; qui la query è SQL testuale e il modo
    portabile è la rappresentazione letterale.
    """
    return "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"
