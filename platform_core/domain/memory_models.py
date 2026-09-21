"""La memoria: cosa la piattaforma ricorda di chi le parla.

**I fatti non si cancellano, si chiudono.** Quando qualcuno dice «mi sono
trasferito a Vienna», la memoria «vive a Budapest» non sparisce: le si mette
una data di fine e la si collega alla nuova con `superseded_by`. La differenza
conta in due momenti — alla domanda «dove vivevo l'anno scorso?», che con la
cancellazione non avrebbe risposta, e davanti a una contraddizione, dove
sapere cosa è stato creduto prima è ciò che permette di capire cosa è
cambiato.

**Ogni memoria appartiene a un utente, sempre.** `personality_id` invece può
essere nullo: alcune cose valgono ovunque — il nome di chi scrive, la sua
lingua — e altre solo dentro una conversazione con una voce precisa. Tenerle
separate evita che un dettaglio confidato a una personalità riaffiori in una
conversazione con un'altra, che è il modo in cui una memoria utile diventa
inquietante.

**Le scadenze non sono una rifinitura.** Senza, dopo mesi il recupero peggiora
invece di migliorare: mille ricordi mediocri seppelliscono i dieci che
contano. `expires_at` e il consolidamento sono ciò che tiene in piedi il
sottosistema, non un abbellimento da aggiungere dopo.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin
from .knowledge_models import DIMENSIONI_EMBEDDING

#: I generi di memoria. La distinzione non è tassonomica: decide quanto a
#: lungo una cosa resta e quanto pesa nel recupero.
GENERI = (
    # Chi è chi parla: nome, lavoro, dove vive. Cambia di rado e serve spesso.
    "identita",
    # Cosa preferisce: come vuole essere trattato, cosa gli interessa.
    "preferenza",
    # Un fatto della sua vita: un viaggio, una decisione, una persona.
    "fatto",
    # Un impegno preso nella conversazione: «ricordamelo la prossima volta».
    "impegno",
    # Il riassunto di una conversazione conclusa. Scade prima degli altri:
    # serve a riprendere il filo, non a essere ricordato per sempre.
    "sessione",
)


class Memory(Base, TimestampMixin):
    """Una cosa che la piattaforma ricorda."""

    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    #: Non `OwnedMixin`: una memoria senza proprietario non ha senso, mentre il
    #: mixin ammette il nullo. Qui è obbligatorio, e il vincolo lo dice.
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    #: Nullo: la memoria vale con qualunque personalità.
    personality_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("personalities.id", ondelete="CASCADE"),
    )

    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    #: Quanto conta. Assegnata all'estrazione e ritoccata dal consolidamento:
    #: una cosa ripetuta più volte sale, una mai più richiamata scende.
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    #: Quanto si è sicuri che sia vera. Un'affermazione esplicita vale più di
    #: una dedotta, e tenerle distinte evita di trattare un'inferenza come un
    #: fatto dichiarato.
    confidence: Mapped[float] = mapped_column(Float, default=0.8, nullable=False)

    #: Quante volte è stata usata in una risposta. È uno dei quattro segnali
    #: del recupero: ciò che serve spesso è probabile serva ancora.
    times_referenced: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    last_referenced_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
    )

    # -- validità nel tempo ------------------------------------------------

    #: Da quando il fatto è vero. Non è `created_at`: qualcuno può raccontare
    #: oggi qualcosa di tre anni fa.
    valid_from: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    #: Fino a quando lo è stato. Valorizzato quando il fatto viene superato:
    #: è ciò che trasforma una cancellazione in una storia.
    valid_to: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    #: La memoria che ha preso il posto di questa.
    superseded_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("memories.id", ondelete="SET NULL"),
    )

    #: Quando smettere di tenerla. Nullo: resta finché non viene superata.
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    #: Il messaggio da cui è stata estratta: è la risposta a «perché ti
    #: ricordi questo?», e senza di essa la memoria è un'affermazione senza
    #: provenienza.
    source_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"),
    )

    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(DIMENSIONI_EMBEDDING),
    )

    meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    __table_args__ = (
        CheckConstraint(
            "kind in ('identita', 'preferenza', 'fatto', 'impegno', 'sessione')",
            name="ck_memories_kind",
        ),
        CheckConstraint(
            "importance >= 0 and importance <= 1", name="ck_memories_importance",
        ),
        CheckConstraint(
            "confidence >= 0 and confidence <= 1", name="ck_memories_confidence",
        ),
        # La query del recupero: «le memorie vive di questo utente». Parte dal
        # proprietario perché è il filtro che non manca mai.
        Index("ix_memories_utente_valide", "user_id", "valid_to"),
        Index("ix_memories_utente_personalita", "user_id", "personality_id"),
        Index("ix_memories_scadenza", "expires_at"),
        # HNSW sul vettore, come per i passaggi: il recupero delle memorie è
        # anzitutto semantico.
        Index(
            "ix_memories_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    @property
    def viva(self) -> bool:
        """Vera se il fatto vale ancora.

        Una memoria superata resta leggibile — serve a raccontare cosa è
        cambiato — ma non entra più nelle risposte.
        """
        return self.valid_to is None and self.superseded_by is None


class MemoryAccess(Base):
    """Chi ha guardato le memorie di qualcun altro.

    Esiste per una ragione sola: la specifica chiede che l'accesso di un
    amministratore alle memorie di un utente lasci traccia. È materia
    delicata — sono cose che qualcuno ha confidato a un servizio, non dati
    operativi — e un accesso senza registro è indistinguibile da un abuso.

    Separata da `audit_log` perché la domanda «chi ha letto le mie memorie?»
    deve avere una risposta rapida e completa, senza filtrare un registro che
    contiene tutto il resto.
    """

    __tablename__ = "memory_accesses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #: Di chi sono le memorie guardate.
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    #: Chi le ha guardate. Nullo per il sistema — il consolidamento le legge
    #: tutte, ed è un accesso legittimo che va comunque distinto da quello di
    #: una persona.
    actor_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
    )

    action: Mapped[str] = mapped_column(String(40), nullable=False)
    #: Quante ne ha viste: «ha letto una memoria» e «le ha scaricate tutte»
    #: sono due fatti diversi.
    count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    correlation_id: Mapped[Optional[str]] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
    )

    __table_args__ = (
        Index("ix_memory_accesses_soggetto", "subject_id", "created_at"),
    )
