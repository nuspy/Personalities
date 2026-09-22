"""Le realizzazioni: adapter, modelli, esportazioni.

**«Realizzazione multipla, uso singolo».** Una personalità può avere molte
build — un LoRA su un modello, uno su un altro, un fine-tuning, un GGUF — ma
ne serve una sola per volta. `Build` registra tutte quelle costruite;
`RuntimeBinding` dice quale è in uso. Tenerle separate evita la domanda «se
cancello questa, cosa risponde?», perché la risposta è scritta da qualche
parte invece che dedotta.

**Perché lo stato è una colonna e non un calcolo.** Un job può essere in coda
per ore — se il worker con acceleratore è a zero repliche, è lo stato normale,
non un guasto — e dedurre «in coda» dall'assenza di un risultato renderebbe
indistinguibile un lavoro che aspetta da uno che è fallito senza dirlo.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger, CheckConstraint, DateTime, ForeignKey, Index, Integer, String,
    Text, Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, OwnedMixin, TimestampMixin

#: Cosa si può costruire.
#:
#: `digestione` non produce un artefatto ma riusa questo meccanismo: è un
#: lavoro lungo, con avanzamento e un esito, esattamente come un
#: addestramento. Duplicare coda, stati e avanzamento per farne un secondo
#: significherebbe mantenerne due che si comportano allo stesso modo.
TIPI_BUILD = ("lora", "finetune", "gguf", "merge", "digestione")

#: Gli stati di un job, in ordine di avanzamento.
#:
#: `in_coda` non è uno stato transitorio: con il worker GPU a zero repliche un
#: job ci resta finché qualcuno non accende una macchina, ed è un
#: comportamento corretto — la coda è il posto dove il lavoro aspetta, non
#: dove muore.
STATI = ("in_coda", "in_corso", "riuscita", "fallita", "annullata")


class Build(Base, TimestampMixin, OwnedMixin):
    """Una realizzazione richiesta, in corso o conclusa."""

    __tablename__ = "builds"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    #: Nullo per le digestioni: quel lavoro riguarda un corpus e non una
    #: voce, e inventare una personalità fittizia per riempire la colonna
    #: renderebbe il dato bugiardo invece che mancante.
    personality_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("personalities.id", ondelete="CASCADE"),
    )
    #: La versione da cui si parte: il profilo stilistico e il corpus di quella
    #: versione sono ciò che l'addestramento consuma. Senza, ripetere una build
    #: a distanza di mesi darebbe un risultato diverso senza spiegazione.
    personality_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("personality_versions.id", ondelete="SET NULL"),
    )

    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="in_coda", nullable=False)

    #: Il modello base, gli iperparametri, gli adapter di partenza.
    params: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    #: Dove è finito il risultato. Un percorso sul nodo che l'ha prodotto, e va
    #: letto così: con più worker, un artefatto è raggiungibile solo da chi
    #: l'ha scritto finché non viene caricato altrove.
    artifact_path: Mapped[Optional[str]] = mapped_column(Text)
    artifact_meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    #: Avanzamento 0-100 e ultima riga di stato: è ciò che la console mostra
    #: mentre il job gira.
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text)

    #: Perché è fallita. Il messaggio dell'eccezione, non la traccia: quella
    #: sta nei log del worker, e riportarla qui la farebbe finire in una
    #: interfaccia dove non serve a nessuno.
    error: Mapped[Optional[str]] = mapped_column(Text)

    #: Chi l'ha eseguita. Serve a ritrovare i log e a sapere dove sia
    #: l'artefatto.
    worker_id: Mapped[Optional[str]] = mapped_column(String(80))

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    eventi: Mapped[List["BuildEvent"]] = relationship(
        back_populates="build", cascade="all, delete-orphan",
        order_by="BuildEvent.created_at",
    )

    __table_args__ = (
        CheckConstraint(
            "kind in ('lora', 'finetune', 'gguf', 'merge', 'digestione')",
            name="ck_builds_kind",
        ),
        CheckConstraint(
            "status in ('in_coda', 'in_corso', 'riuscita', 'fallita', 'annullata')",
            name="ck_builds_status",
        ),
        Index("ix_builds_owner_creazione", "owner_id", "created_at"),
        Index("ix_builds_personalita", "personality_id", "created_at"),
        Index("ix_builds_stato", "status"),
    )

    @property
    def conclusa(self) -> bool:
        return self.status in ("riuscita", "fallita", "annullata")


class BuildEvent(Base):
    """Una riga di avanzamento.

    Conservate e non solo trasmesse: chi apre la console dopo che un job è
    finito deve poter leggere com'è andato. Un avanzamento che esiste solo nel
    flusso in tempo reale è visibile a chi stava guardando, e a nessun altro.
    """

    __tablename__ = "build_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    build_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("builds.id", ondelete="CASCADE"), nullable=False,
    )

    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    #: `info` o `errore`. Un errore non conclude sempre il job — uno stadio può
    #: fallire e lasciar proseguire il successivo — quindi va distinto dallo
    #: stato finale.
    level: Mapped[str] = mapped_column(String(10), default="info", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )

    build: Mapped[Build] = relationship(back_populates="eventi")

    __table_args__ = (
        Index("ix_build_events_build", "build_id", "created_at"),
    )


class RuntimeBinding(Base, TimestampMixin):
    """Quale realizzazione una personalità sta effettivamente usando.

    Una riga per personalità, e lo impone la chiave primaria: due build attive
    contemporaneamente per la stessa voce sarebbero una domanda senza risposta
    al momento di servirla.

    `mode` può valere `rag` anche con una build presente: è la degradazione
    dichiarata — se il worker che serviva quel LoRA non c'è più, si risponde in
    RAG e lo si scrive, invece di fallire o di far finta di niente.
    """

    __tablename__ = "runtime_bindings"

    personality_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("personalities.id", ondelete="CASCADE"), primary_key=True,
    )
    build_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("builds.id", ondelete="SET NULL"),
    )

    mode: Mapped[str] = mapped_column(String(20), default="rag", nullable=False)
    #: Dove gira: `local` (un worker di questa piattaforma) o l'identificativo
    #: di un provider remoto.
    serving: Mapped[Optional[str]] = mapped_column(String(80))

    # Nessun vincolo di unicita' esplicito: `personality_id` e' gia' chiave
    # primaria, quindi «una riga per personalita'» e' gia' garantito.
    # Dichiararlo due volte faceva riprovare ad Alembic la stessa creazione a
    # ogni autogenerazione successiva.
    __table_args__ = (
        CheckConstraint(
            "mode in ('rag', 'lora', 'finetune')", name="ck_runtime_mode",
        ),
    )
