"""Da testo a passaggi interrogabili.

Il percorso è: testo → passaggi → vettori → righe. Tre punti meritano una
spiegazione.

**L'impronta prima di tutto.** Un documento già presente non si reindicizza: si
riconosce dal `sha256` del testo e si salta. Senza, ogni raccolta ripetuta di
una fonte moltiplicherebbe i passaggi, e il modello leggerebbe tre volte la
stessa frase credendo di avere tre conferme indipendenti.

**Il `tsvector` lo costruisce chi indicizza**, perché solo qui si sa in che
lingua è il testo. Un indice su `to_tsvector('italian', text)` tratterebbe da
italiano anche un corpus latino, e lo stemming sbagliato non dà errore: dà
meno risultati.

**Vettori e testo nella stessa transazione.** Un passaggio senza vettore
esiste e non si trova mai: è il difetto peggiore, perché il corpus sembra
completo e il recupero sembra semplicemente mediocre.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow
from ..domain.knowledge_models import Chunk, ChunkVector, Document, KnowledgeBase
from .chunker import ConfigurazioneChunking, dividi_documento
from .embedding import Embedder

logger = logging.getLogger(__name__)

#: Configurazioni di ricerca testuale che PostgreSQL conosce, per lingua.
#: Quelle mancanti ricadono su `simple`, che non fa stemming — meno efficace
#: ma mai sbagliato, mentre lo stemming della lingua sbagliata lo è.
CONFIGURAZIONI_TESTO = {
    "it": "italian",
    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "pt": "portuguese",
    "nl": "dutch",
    "ru": "russian",
    # Il latino non ha una configurazione in PostgreSQL: `simple` è corretto,
    # e per un corpus classico è anche difendibile — lo stemming italiano su
    # un testo latino produce radici inesistenti.
    "la": "simple",
}


def configurazione_per_lingua(lingua: Optional[str]) -> str:
    if not lingua:
        return "simple"
    return CONFIGURAZIONI_TESTO.get(lingua.lower()[:2], "simple")


@dataclass
class EsitoIndicizzazione:
    documento_id: Optional[uuid.UUID]
    passaggi: int
    saltato: bool = False
    motivo: str = ""


class Indexer:
    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        *,
        config: Optional[ConfigurazioneChunking] = None,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._config = config or ConfigurazioneChunking()

    async def indicizza(
        self,
        kb: KnowledgeBase,
        *,
        titolo: str,
        testo: str,
        uri: Optional[str] = None,
        lingua: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        pubblicato_il=None,
    ) -> EsitoIndicizzazione:
        """Aggiunge un documento a una base, se non c'è già."""
        if kb.embed_model != self._embedder.modello:
            # Mescolare vettori di modelli diversi nello stesso spazio produce
            # somiglianze prive di senso. Non è un avviso: è un rifiuto.
            raise ValueError(
                f"la base '{kb.slug}' è indicizzata con {kb.embed_model}, "
                f"l'embedder corrente è {self._embedder.modello}: "
                f"i vettori non sarebbero confrontabili"
            )

        configurazione = configurazione_per_lingua(lingua)

        # `kb.documents` non si può interrogare qui: è una relazione pigra, e
        # in una sessione asincrona il caricamento implicito solleva errore.
        primo_documento = not await self._session.scalar(
            select(func.count()).select_from(Document).where(Document.kb_id == kb.id)
        )

        if primo_documento:
            # La configurazione della base la fissa il primo documento: da quel
            # momento tutti i passaggi e tutte le domande passano di lì.
            kb.text_config = configurazione
        elif kb.text_config != configurazione:
            # Cambiarla a meta' strada renderebbe irraggiungibile la parte gia'
            # indicizzata: la domanda verrebbe interpretata in un modo solo, e
            # i passaggi scritti con l'altro non la incontrerebbero mai.
            logger.warning(
                "La base '%s' e' indicizzata in '%s' ma questo documento e' in "
                "'%s' (lingua %s): i suoi passaggi resteranno fuori dalla "
                "ricerca lessicale. Conviene una base per lingua.",
                kb.slug, kb.text_config, configurazione, lingua,
            )
            configurazione = kb.text_config

        impronta = hashlib.sha256(testo.encode("utf-8")).hexdigest()

        esistente = await self._session.execute(
            select(Document.id).where(
                Document.kb_id == kb.id, Document.sha256 == impronta,
            )
        )
        if (gia_presente := esistente.scalar_one_or_none()) is not None:
            return EsitoIndicizzazione(
                documento_id=gia_presente, passaggi=0, saltato=True,
                motivo="documento già presente, contenuto identico",
            )

        passaggi = dividi_documento(testo, config=self._config)
        if not passaggi:
            return EsitoIndicizzazione(
                documento_id=None, passaggi=0, saltato=True,
                motivo="nessun testo utilizzabile",
            )

        documento = Document(
            kb_id=kb.id, title=titolo, uri=uri, sha256=impronta,
            published_at=pubblicato_il, fetched_at=utcnow(),
            meta={**(meta or {}), "lingua": lingua} if (meta or lingua) else None,
        )
        self._session.add(documento)
        await self._session.flush()

        vettori = await self._embedder.documenti([p.testo for p in passaggi])

        for passaggio, vettore in zip(passaggi, vettori):
            chunk = Chunk(
                document_id=documento.id,
                kb_id=kb.id,
                ordinal=passaggio.ordinale,
                text=passaggio.testo,
                section=passaggio.sezione,
                tokens=passaggio.token_stimati,
            )
            self._session.add(chunk)
            await self._session.flush()

            # Il tsvector si calcola nel database: la funzione sta lì, con i
            # suoi dizionari, e replicarla in Python darebbe una forma che non
            # corrisponde a quella con cui la domanda verrà confrontata.
            await self._session.execute(
                text(
                    "UPDATE chunks SET tsv = to_tsvector(:conf, :testo) WHERE id = :id"
                ),
                {"conf": configurazione, "testo": passaggio.testo, "id": chunk.id},
            )

            self._session.add(ChunkVector(
                chunk_id=chunk.id, kb_id=kb.id, embedding=vettore,
                embedded_with=self._embedder.modello,
            ))

        await self._session.flush()
        await self._aggiorna_statistiche(kb)

        logger.info(
            "Indicizzato «%s» in %s: %d passaggi", titolo, kb.slug, len(passaggi),
        )
        return EsitoIndicizzazione(documento_id=documento.id, passaggi=len(passaggi))

    async def _aggiorna_statistiche(self, kb: KnowledgeBase) -> None:
        """Quanto contiene la base. Si legge dalla console, non dal codice."""
        conteggi = await self._session.execute(
            select(
                func.count(func.distinct(Document.id)),
                func.count(Chunk.id),
                func.coalesce(func.sum(Chunk.tokens), 0),
            )
            .select_from(Document)
            .outerjoin(Chunk, Chunk.document_id == Document.id)
            .where(Document.kb_id == kb.id)
        )
        documenti, passaggi, token = conteggi.one()
        kb.stats = {
            "documenti": int(documenti),
            "passaggi": int(passaggi),
            "token_stimati": int(token),
            "aggiornate_il": utcnow().isoformat(),
        }
