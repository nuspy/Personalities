"""Conoscenza e personalità.

Due gruppi di tabelle che arrivano insieme perché servono l'uno all'altro: una
personalità senza corpus risponde a memoria, e un corpus senza personalità non
ha voce.

**Sul versionamento.** `personality_versions` è immutabile e ogni conversazione
registra quale versione l'ha prodotta. Senza, una conversazione di tre mesi fa
non è riproducibile: il prompt che l'ha generata sarebbe già cambiato, e il
debug guarderebbe un testo diverso da quello che ha causato il problema.

**Sui vettori.** Stanno in una tabella separata dai chunk e non in una colonna
accanto al testo. La ragione è pratica: cambiare modello di embedding
significa ricalcolare tutti i vettori, e con la separazione lo si fa senza
toccare — né bloccare — la tabella che contiene il testo. La dimensione fa
parte della definizione della colonna, quindi modelli con dimensioni diverse
non possono convivere nella stessa tabella: è un vincolo del tipo `vector`, e
va saputo prima di sceglierne uno.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index,
    Integer, String, Text, UniqueConstraint, Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, OwnedMixin, TimestampMixin

#: Dimensione dei vettori. Legata al modello di embedding scelto
#: (`qwen3-embedding-0.6b`, multilingue, 1024 dimensioni): cambiarla richiede
#: una migrazione e il ricalcolo dell'intero indice, quindi è una decisione di
#: piattaforma e non di configurazione.
DIMENSIONI_EMBEDDING = 1024


class KnowledgeBase(Base, TimestampMixin, OwnedMixin):
    """Un corpus interrogabile.

    `embed_model` è registrato sulla base e non dedotto dalla configurazione
    corrente: se un giorno il modello predefinito cambia, le basi già
    indicizzate devono continuare a dire con quale modello sono state
    costruite — altrimenti si interrogherebbero vettori di un modello con le
    query di un altro, e il risultato sarebbe silenziosamente sbagliato.
    """

    __tablename__ = "knowledge_bases"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    #: `corpus` (gli scritti della persona), `reference` (materiale di
    #: contesto), `news` (fonti che cambiano nel tempo).
    kind: Mapped[str] = mapped_column(String(20), default="corpus", nullable=False)
    visibility: Mapped[str] = mapped_column(String(20), default="private", nullable=False)

    embed_model: Mapped[str] = mapped_column(String(120), nullable=False)
    embed_dim: Mapped[int] = mapped_column(Integer, default=DIMENSIONI_EMBEDDING)

    #: La configurazione di ricerca testuale con cui i passaggi sono stati
    #: indicizzati (`italian`, `english`, `simple`…).
    #:
    #: Va registrata qui per la stessa ragione di `embed_model`: la domanda
    #: dev'essere interpretata **esattamente come** i documenti, o non li
    #: incontra mai. Con i passaggi indicizzati in italiano — dove «amici»
    #: diventa la radice «amic» — una domanda interpretata in `simple` cerca
    #: la parola intera e trova zero risultati su venti. Il guasto è muto: la
    #: ricerca lessicale smette semplicemente di contribuire, e l'ibrido
    #: diventa un vettoriale con più codice attorno.
    text_config: Mapped[str] = mapped_column(
        String(30), default="simple", nullable=False,
    )
    chunk_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    stats: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    documents: Mapped[List["Document"]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "kind in ('corpus', 'reference', 'news')", name="ck_kb_kind",
        ),
        CheckConstraint(
            "visibility in ('private', 'shared', 'public')", name="ck_kb_visibility",
        ),
        Index("ix_knowledge_bases_owner_slug", "owner_id", "slug"),
    )


class Document(Base, TimestampMixin):
    """Un testo intero, prima di essere spezzato.

    `sha256` con vincolo di unicità per base: lo stesso documento raccolto due
    volte — un aggiornamento di una fonte, un file caricato di nuovo — non deve
    moltiplicare i passaggi recuperabili, o il modello leggerebbe la stessa
    cosa tre volte credendo di avere tre conferme.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    kb_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False,
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    uri: Mapped[Optional[str]] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    #: Quando il testo è stato scritto, non quando l'abbiamo raccolto. Serve a
    #: distinguere una fonte antica da una recente quando entrambe rispondono
    #: alla stessa domanda.
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    #: Autore, lingua, tipo di fonte: ciò che serve a dire in una citazione da
    #: dove viene un passaggio.
    meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")
    chunks: Mapped[List["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("kb_id", "sha256", name="uq_documents_kb_sha"),
        Index("ix_documents_kb", "kb_id"),
    )


class Chunk(Base):
    """Un passaggio recuperabile.

    `ordinal` conserva l'ordine nel documento: un passaggio isolato perde il
    filo del discorso, e conoscere i vicini permette di restituirli insieme
    quando la domanda lo richiede.
    """

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False,
    )
    #: Ripetuto rispetto a `document.kb_id`, e deliberatamente: il recupero
    #: filtra per base a ogni interrogazione, e passare per la tabella dei
    #: documenti aggiungerebbe una giunzione alla query più frequente.
    kb_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False,
    )

    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    #: Il titolo della sezione da cui proviene: entra nella citazione e aiuta
    #: chi legge a collocare il passaggio.
    section: Mapped[Optional[str]] = mapped_column(String(300))
    tokens: Mapped[Optional[int]] = mapped_column(Integer)

    #: La forma indicizzabile del testo per la ricerca lessicale.
    #:
    #: È una colonna vera e non un'espressione calcolata nell'indice, perché la
    #: configurazione linguistica cambia da documento a documento: un corpus
    #: latino e uno italiano vogliono stemming diversi, e un indice su
    #: `to_tsvector('italian', text)` li tratterebbe tutti come italiano. Qui
    #: la costruisce chi indicizza, che sa in che lingua è il testo.
    #:
    #: Serve perché il recupero è ibrido: il vettoriale trova ciò che è detto
    #: con altre parole, il lessicale trova il nome proprio o la citazione
    #: esatta — e da solo nessuno dei due basta.
    tsv: Mapped[Optional[Any]] = mapped_column(TSVECTOR)

    document: Mapped[Document] = relationship(back_populates="chunks")
    vector: Mapped[Optional["ChunkVector"]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan", uselist=False,
    )

    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_doc_ordinal"),
        Index("ix_chunks_kb", "kb_id"),
        # GIN sul tsvector: è l'indice che rende praticabile la ricerca
        # lessicale. Senza, ogni interrogazione scandisce l'intera tabella dei
        # passaggi, e il costo si vede solo quando il corpus è già grande.
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )


class ChunkVector(Base):
    """La rappresentazione vettoriale di un passaggio.

    Separata dal testo perché ha un ciclo di vita proprio: si ricalcola quando
    cambia il modello di embedding, mentre il testo resta.
    """

    __tablename__ = "chunk_vectors"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True,
    )
    kb_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False,
    )
    embedding: Mapped[List[float]] = mapped_column(
        Vector(DIMENSIONI_EMBEDDING), nullable=False,
    )
    #: Con quale modello è stato calcolato. Un vettore senza questa
    #: informazione è inutilizzabile il giorno in cui il modello cambia: non si
    #: saprebbe quali ricalcolare.
    embedded_with: Mapped[str] = mapped_column(String(120), nullable=False)

    chunk: Mapped[Chunk] = relationship(back_populates="vector")

    __table_args__ = (
        Index("ix_chunk_vectors_kb", "kb_id"),
        # HNSW con distanza coseno. Approssimato e non esatto: su un corpus di
        # qualche migliaio di passaggi la differenza nei risultati è
        # trascurabile, mentre quella nei tempi non lo è — e l'alternativa
        # esatta degrada linearmente col corpus.
        #
        # `m` e `ef_construction` sono i valori consigliati da pgvector:
        # costruzione più lenta in cambio di un grafo che richiama meglio.
        Index(
            "ix_chunk_vectors_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Personality(Base, TimestampMixin, OwnedMixin):
    """Una voce.

    Il contenuto vero sta nelle versioni: questa riga è l'identità stabile a
    cui puntano le conversazioni, e ciò che cambia — prompt, regole, corpora —
    cambia in una versione nuova.
    """

    __tablename__ = "personalities"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="draft", nullable=False)

    current_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid,
        # Nessun vincolo di chiave esterna: le due tabelle si puntano a
        # vicenda, e una coppia di vincoli reciproci renderebbe impossibile
        # inserire la prima riga senza rinviare i controlli. L'integrita' e'
        # garantita dal repository, che pubblica una versione solo dopo averla
        # scritta.
        nullable=True,
    )

    versions: Mapped[List["PersonalityVersion"]] = relationship(
        back_populates="personality",
        cascade="all, delete-orphan",
        foreign_keys="PersonalityVersion.personality_id",
    )

    __table_args__ = (
        CheckConstraint(
            "status in ('draft', 'published', 'archived')",
            name="ck_personalities_status",
        ),
        # «Le personalità di questo utente, quelle pubblicate»: è l'unica query
        # che la console e il catalogo fanno davvero.
        Index("ix_personalities_owner_status", "owner_id", "status"),
    )


class PersonalityVersion(Base, TimestampMixin):
    """Una versione immutabile di una personalità.

    Immutabile per disciplina: modificarla renderebbe irriproducibili le
    conversazioni che la citano, che è esattamente ciò da cui il versionamento
    deve proteggere. Le modifiche producono una versione nuova.
    """

    __tablename__ = "personality_versions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    personality_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("personalities.id", ondelete="CASCADE"), nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    #: Lo strato stabile del prompt: carattere, lessico, modo di argomentare.
    #: È la parte che deve restare byte-identica fra richieste, o il prompt
    #: caching non riconosce il prefisso e lo sconto sparisce senza che nulla
    #: lo segnali.
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    behavior_rules: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    llm_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    rag_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    memory_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    guard_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    #: Il profilo stilistico da cui il prompt è stato generato. Conservarlo
    #: permette di rigenerare il prompt con regole diverse senza rifare
    #: l'analisi del corpus, che è la parte costosa.
    style_profile: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    author_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
    )

    personality: Mapped[Personality] = relationship(
        back_populates="versions", foreign_keys=[personality_id],
    )

    __table_args__ = (
        UniqueConstraint("personality_id", "version", name="uq_version_per_personality"),
    )


class PersonalityKnowledgeBase(Base):
    """Quali corpora una personalità consulta, e con quale peso."""

    __tablename__ = "personality_kb"

    personality_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("personalities.id", ondelete="CASCADE"), primary_key=True,
    )
    kb_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), primary_key=True,
    )

    #: `voice` è il corpus da cui viene lo stile, `knowledge` quello da cui
    #: vengono i fatti. La distinzione conta per il groundcheck della fase 2:
    #: una frase in carattere non va verificata come se fosse un'affermazione
    #: di fatto, o il sistema segnalerebbe come inventata ogni risposta ben
    #: riuscita.
    role: Mapped[str] = mapped_column(String(20), default="knowledge", nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    max_chunks: Mapped[int] = mapped_column(Integer, default=6, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (
        CheckConstraint("role in ('voice', 'knowledge')", name="ck_personality_kb_role"),
    )


class AnswerTrace(Base):
    """Come è nata una risposta.

    Non è un registro di debug: è la prova di ciò che il modello ha letto. Senza,
    «perché ha detto questo?» non ha risposta, e il vantaggio del prompt caching
    resta un'ipotesi — `usage` è l'unico posto dove si vede quanti token sono
    stati letti da cache invece che riletti da capo.
    """

    __tablename__ = "answer_traces"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, unique=True,
    )
    personality_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("personality_versions.id", ondelete="SET NULL"),
    )

    #: I passaggi scelti e — altrettanto importante — quelli scartati con il
    #: loro punteggio: senza i secondi non si capisce mai perché il recupero
    #: abbia mancato ciò che sarebbe servito.
    retrieved: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    grounding: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    guard_events: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    usage: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    latency_ms: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
    )
