"""Digestione di un corpus intero.

Tre passate, in ordine di costo crescente: prima ciò che si decide a vista,
poi la classificazione, infine la deduplicazione semantica — che ha senso solo
su ciò che è sopravvissuto alle prime due.

**Riprendibile.** Un corpus grande richiede ore, e in quelle ore può succedere
di tutto: un'interruzione, un riavvio, un modello che smette di rispondere.
Ogni passaggio classificato viene salvato subito, e una nuova esecuzione parte
da quelli senza etichetta. Ripartire da zero dopo tre ore di lavoro sarebbe la
ragione per cui nessuno riesegue mai la digestione.

**La deduplicazione viene per ultima** perché confronta ciò che resta: due
passaggi identici di cui uno è apparato editoriale non sono un duplicato da
fondere, sono uno da tenere e uno da buttare.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.knowledge_models import Chunk, ChunkVector, KnowledgeBase
from ..observability.tracing import traccia
from .digestion import (
    PER_LOTTO, SOGLIA_QUALITA, Digestore, EsitoDigestione, scarto_a_vista,
)
from .embedding import somiglianza_coseno
from .taxonomy import Categoria, Etichettatura, Provenienza

logger = logging.getLogger(__name__)

#: Sopra questa somiglianza due passaggi dicono la stessa cosa.
#:
#: Più alta della soglia usata per le memorie (0,92): qui si confrontano testi
#: lunghi, dove la somiglianza coseno è naturalmente più alta, e due paragrafi
#: diversi dello stesso autore sullo stesso tema arrivano facilmente a 0,90
#: senza essere duplicati.
SOGLIA_DUPLICATO = 0.96

#: Come arriva l'avanzamento a chi guarda: (fatti, totale, messaggio).
Avanzamento = Callable[[int, int, str], None]


class DigestioneCorpus:
    def __init__(
        self,
        session: AsyncSession,
        digestore: Digestore,
        *,
        embedder=None,
        per_lotto: int = PER_LOTTO,
    ) -> None:
        self._session = session
        self._digestore = digestore
        # Serve a rivettorizzare i passaggi ripuliti: un testo cambiato con il
        # vettore vecchio continuerebbe a essere cercato per com'era prima, e
        # il difetto sarebbe muto — il passaggio si troverebbe, semplicemente
        # con la pertinenza sbagliata.
        self._embedder = embedder
        self._per_lotto = per_lotto

    async def digerisci(
        self,
        kb: KnowledgeBase,
        *,
        chi: str = "",
        rifai: bool = False,
        avanzamento: Optional[Avanzamento] = None,
    ) -> EsitoDigestione:
        """Analizza i passaggi di una base.

        `rifai` rianalizza anche quelli già etichettati: serve quando cambia
        il classificatore o la tassonomia, e senza si resterebbe con un corpus
        metà vecchio e metà nuovo — incoerente e impossibile da interpretare.
        """
        esito = EsitoDigestione()

        da_fare = await self._da_fare(kb, rifai=rifai)
        totale = len(da_fare)
        if not totale:
            logger.info("Base «%s»: niente da digerire", kb.slug)
            return esito

        logger.info("Base «%s»: %d passaggi da digerire", kb.slug, totale)

        # Passata 1: a vista, senza costo.
        restanti: List[Chunk] = []
        for chunk in da_fare:
            motivo = scarto_a_vista(chunk.text)
            if motivo:
                _segna_scartato(chunk, motivo, provenienza=Provenienza.EDITORIALE)
                esito.scartati_a_vista += 1
            else:
                restanti.append(chunk)

        await self._session.flush()
        if avanzamento:
            avanzamento(
                esito.scartati_a_vista, totale,
                f"{esito.scartati_a_vista} scartati a vista",
            )

        # Passata 2: classificazione, a lotti e salvando man mano.
        fatti = esito.scartati_a_vista
        for inizio in range(0, len(restanti), self._per_lotto):
            lotto = restanti[inizio : inizio + self._per_lotto]
            etichette = await self._digestore.classifica(
                [c.text for c in lotto], chi=chi,
            )

            for chunk, etichettatura in zip(lotto, etichette):
                ripulito = _applica(chunk, etichettatura)
                if ripulito:
                    esito.ripuliti += 1
                    await self._rivettorizza(chunk)
                if etichettatura.scartato:
                    esito.scartati_dal_giudizio += 1
                else:
                    esito.classificati += 1
                    _conta(esito, etichettatura)

            # Dopo ogni lotto e non alla fine: un'interruzione a metà di tre
            # ore non deve costare tutto il lavoro fatto.
            await self._session.commit()

            fatti += len(lotto)
            if avanzamento:
                avanzamento(fatti, totale, f"{fatti} di {totale} passaggi")

        esito.esaminati = totale

        # Passata 3: duplicati, solo fra ciò che è sopravvissuto.
        fusi = await self._togli_duplicati(kb)
        esito.scartati_dal_giudizio += fusi

        await self._aggiorna_statistiche(kb, esito)
        await self._session.commit()

        logger.info("Base «%s» digerita: %s", kb.slug, esito.to_dict())
        return esito

    async def _rivettorizza(self, chunk: Chunk) -> None:
        """Ricalcola il vettore e l'indice lessicale di un passaggio ripulito.

        Senza, il passaggio resterebbe cercabile per il testo che aveva prima
        — apparato compreso — e il difetto non darebbe alcun segno: si
        troverebbe comunque, solo con la pertinenza sbagliata.
        """
        if self._embedder is None:
            return

        vettori = await self._embedder.documenti([chunk.text])
        esistente = await self._session.get(ChunkVector, chunk.id)
        if esistente is not None:
            esistente.embedding = vettori[0]
            esistente.embedded_with = self._embedder.modello

        kb = await self._session.get(KnowledgeBase, chunk.kb_id)
        await self._session.execute(
            text("UPDATE chunks SET tsv = to_tsvector(:conf, :testo) WHERE id = :id"),
            {
                "conf": kb.text_config if kb else "simple",
                "testo": chunk.text,
                "id": chunk.id,
            },
        )

    async def _da_fare(
        self, kb: KnowledgeBase, *, rifai: bool
    ) -> Sequence[Chunk]:
        query = select(Chunk).where(Chunk.kb_id == kb.id).order_by(Chunk.ordinal)
        if not rifai:
            # Solo quelli mai visti: è ciò che rende l'operazione riprendibile
            # dopo un'interruzione.
            query = query.where(Chunk.labels.is_(None), Chunk.discarded.is_(False))
        return (await self._session.execute(query)).scalars().all()

    async def _togli_duplicati(self, kb: KnowledgeBase) -> int:
        """Scarta i passaggi che ne ripetono un altro già tenuto.

        Solo fra i sopravvissuti: due passaggi identici di cui uno è apparato
        editoriale non sono un duplicato da fondere, sono uno da tenere e uno
        da buttare — e la passata precedente ha già deciso quale.
        """
        with traccia("digestione.duplicati", kb=kb.slug):
            righe = (await self._session.execute(
                select(Chunk, ChunkVector.embedding)
                .join(ChunkVector, ChunkVector.chunk_id == Chunk.id)
                .where(Chunk.kb_id == kb.id, Chunk.discarded.is_(False))
                .order_by(Chunk.ordinal)
            )).all()

        if len(righe) < 2:
            return 0

        # Dal più utile: il sopravvissuto dev'essere quello che insegna di
        # più, non quello che capita prima nel documento.
        ordinati = sorted(
            righe, key=lambda r: (r[0].quality or 0.0), reverse=True,
        )

        scartati = 0
        tenuti: List[tuple] = []

        for chunk, embedding in ordinati:
            duplicato_di = None
            for altro, altro_embedding in tenuti:
                if somiglianza_coseno(embedding, altro_embedding) >= SOGLIA_DUPLICATO:
                    duplicato_di = altro
                    break

            if duplicato_di is not None:
                chunk.discarded = True
                chunk.discard_reason = (
                    f"ripete il passaggio {duplicato_di.ordinal} dello stesso corpus"
                )
                scartati += 1
            else:
                tenuti.append((chunk, embedding))

        if scartati:
            await self._session.flush()
            logger.info("Base «%s»: %d duplicati scartati", kb.slug, scartati)
        return scartati

    async def _aggiorna_statistiche(
        self, kb: KnowledgeBase, esito: EsitoDigestione
    ) -> None:
        vivi = await self._session.scalar(
            select(func.count()).select_from(Chunk).where(
                Chunk.kb_id == kb.id, Chunk.discarded.is_(False),
            )
        )
        kb.stats = {
            **(kb.stats or {}),
            "passaggi_vivi": int(vivi or 0),
            "digestione": esito.to_dict(),
        }


def _applica(chunk: Chunk, etichettatura: Etichettatura) -> bool:
    """Scrive il verdetto sul passaggio. Vero se il testo è stato ripulito."""
    ripulito = False

    da_togliere = etichettatura.da_togliere.strip()
    if da_togliere:
        motivo = _perche_non_rimuovere(chunk.text, da_togliere)
        if motivo:
            logger.warning(
                "Rimozione rifiutata (%s): %r", motivo, da_togliere[:70],
            )
        else:
            # L'originale si conserva prima di toccare il testo: è l'unica
            # operazione della digestione che non si può disfare guardando le
            # etichette.
            if chunk.text_original is None:
                chunk.text_original = chunk.text
            nuovo = chunk.text[len(da_togliere):].strip()
            chunk.text = nuovo
            chunk.tokens = max(1, round(len(nuovo) / 3.6))
            ripulito = True

    chunk.labels = etichettatura.to_dict()
    principale = etichettatura.principale
    chunk.label_main = principale.value if principale else None
    chunk.provenance = etichettatura.provenienza.value
    chunk.quality = etichettatura.qualita

    if etichettatura.scartato:
        chunk.discarded = True
        chunk.discard_reason = etichettatura.motivo_scarto
    else:
        chunk.discarded = False
        chunk.discard_reason = None

    return ripulito


def _perche_non_rimuovere(testo: str, da_togliere: str) -> str:
    """Il motivo per cui una rimozione non va applicata, o stringa vuota.

    **L'apparato è sempre in testa.** Questa è la difesa che conta: su un
    corpus reale il modello ha proposto di togliere la prosa dell'autore
    lasciando l'incipit latino — la stringa c'era davvero nel testo, e una
    verifica sulla sola presenza l'avrebbe accettata. Pretendere che sia un
    prefisso rende quell'errore impossibile invece che improbabile.
    """
    pulito = testo.strip()
    inizio = pulito[:len(da_togliere) + 4]

    if not inizio.startswith(da_togliere):
        return "non è in testa al passaggio"
    if len(da_togliere) >= len(pulito) * 0.35:
        # Sopra un terzo non è un richiamo bibliografico: è il modello che ha
        # deciso di riassumere.
        return f"troppo lunga ({len(da_togliere)} di {len(pulito)} caratteri)"

    # Due difese e non tre: ne avevo scritta una terza — «non deve restare
    # meno di ottanta caratteri» — e scrivendone il test si è visto che non
    # può mai scattare. Perché resti così poco, con una rimozione sotto un
    # terzo, il passaggio dovrebbe essere più corto di centoventitré
    # caratteri: sotto la soglia con cui `scarto_a_vista` lo ha già tolto di
    # mezzo. Una regola che non si attiva mai non protegge nulla e fa credere
    # il contrario.
    return ""


def _segna_scartato(
    chunk: Chunk, motivo: str, *, provenienza: Provenienza,
) -> None:
    chunk.discarded = True
    chunk.discard_reason = motivo
    chunk.quality = 0.0
    chunk.provenance = provenienza.value
    chunk.labels = Etichettatura(
        categorie={Categoria.APPARATO: 1.0},
        provenienza=provenienza,
        qualita=0.0,
        motivo_scarto=motivo,
    ).to_dict()
    chunk.label_main = Categoria.APPARATO.value


def _conta(esito: EsitoDigestione, etichettatura: Etichettatura) -> None:
    principale = etichettatura.principale
    if principale is not None:
        esito.per_categoria[principale.value] = (
            esito.per_categoria.get(principale.value, 0) + 1
        )
    chiave = etichettatura.provenienza.value
    esito.per_provenienza[chiave] = esito.per_provenienza.get(chiave, 0) + 1
