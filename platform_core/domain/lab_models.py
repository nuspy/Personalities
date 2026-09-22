"""Il laboratorio: come vanno le risposte, e quale versione va meglio.

Due tabelle, una per ciascuna delle domande che una piattaforma di voci si
fa dopo il lancio.

**`answer_feedback` — la risposta è piaciuta?** Un voto per risposta, che chi
l'ha ricevuta può cambiare. Una riga per messaggio e non uno storico dei
voti: conta cosa pensa adesso, e il voto che ha tolto non deve contare
contro la versione che l'ha prodotto.

**`experiments` — quale versione va meglio?** Un confronto fra versioni della
stessa personalità, con i pesi con cui si assegnano. L'assegnazione è per
utente e stabile: chi ha cominciato con la variante A la ritrova a ogni nuova
conversazione, o il confronto misurerebbe il disorientamento invece della
versione. La conversazione registra l'esperimento che l'ha assegnata —
insieme alla versione, che registrava già — ed è da lì che si leggono i
risultati.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger, CheckConstraint, DateTime, ForeignKey, Index, SmallInteger, String,
    Text, Uuid, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


class AnswerFeedback(Base, TimestampMixin):
    __tablename__ = "answer_feedback"

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True,
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
    )
    #: +1 o -1. Niente scala da uno a cinque: nessuno sa dire se una risposta
    #: meriti un tre o un quattro, e la media di numeri inventati è un numero
    #: inventato.
    vote: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    #: Facoltativo, e solo per chi ha qualcosa da dire: chiederlo sempre
    #: farebbe smettere di votare.
    reason: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("vote in (-1, 1)", name="ck_answer_feedback_vote"),
        Index("ix_answer_feedback_utente", "user_id"),
        Index("ix_answer_feedback_creazione", "created_at"),
    )


STATI_ESPERIMENTO = ("attivo", "concluso")


class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    personality_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("personalities.id", ondelete="CASCADE"), nullable=False,
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    #: Che cosa si vuole sapere, scritto prima di guardare i numeri: un
    #: esperimento senza ipotesi finisce per confermare quella che fa comodo.
    hypothesis: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="attivo", nullable=False)
    #: `[{"version_id": "...", "peso": 50, "etichetta": "A"}, ...]`. La prima
    #: è il riferimento, contro cui si confrontano le altre.
    variants: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="SET NULL"),
    )
    #: `RESTRICT` come per le conversazioni: la versione scelta non deve
    #: poter sparire mentre l'esperimento la indica.
    winner_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("personality_versions.id", ondelete="RESTRICT"),
    )
    conclusion: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("status in ('attivo', 'concluso')", name="ck_experiments_status"),
        # Un solo esperimento attivo per personalità: con due, una stessa
        # conversazione apparterrebbe a entrambi, e nessuno dei due
        # misurerebbe ciò che dice.
        Index(
            "uq_experiments_attivo_per_personalita", "personality_id",
            unique=True, postgresql_where=text("status = 'attivo'"),
        ),
    )
