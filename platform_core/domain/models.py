"""Le tabelle che lo scheletro verticale attraversa davvero.

Qui stanno soltanto `users`, `conversations`, `messages` e `audit_log`: le
quattro toccate dal percorso «accedo, chiedo, ricevo, resta traccia». Le altre
della specifica — personalità, knowledge base, memorie, crediti — arrivano con
la fase che le usa. Scriverle adesso significherebbe indovinare i campi senza
codice che li legge, e uno schema inventato in anticipo invecchia male: le
colonne sbagliate restano, perché toglierle costa una migrazione.

**Sulle chiavi primarie.** Due convenzioni, per una ragione sola: ciò che
compare in una URL non deve essere indovinabile. `conversations` e `messages`
usano UUID perché finiscono in `/conversations/{id}`; `users` e `audit_log`
usano interi perché non compaiono mai in un percorso — l'utente si identifica
dal token, e il registro si legge per filtro, non per id.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger, CheckConstraint, DateTime, ForeignKey, Index, String, Text, Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, OwnedMixin, TimestampMixin


class User(Base, TimestampMixin):
    """Il riflesso locale di un'identità che vive in Keycloak.

    La riga non nasce da una registrazione: nasce la prima volta che un token
    valido arriva con un `sub` mai visto. Qui non si conservano credenziali —
    nemmeno un hash — perché l'unico posto dove una password può stare è il
    servizio che la verifica.

    `deleted_at` invece di una `DELETE`: le conversazioni e il registro crediti
    di un utente cancellato servono ancora per la contabilità e per l'audit, e
    una cancellazione fisica li porterebbe via in cascata.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # L'identificativo stabile del soggetto in Keycloak. Non l'email: quella
    # l'utente può cambiarla, e due account diversi possono averne avuta la
    # stessa in momenti diversi.
    keycloak_sub: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    email: Mapped[Optional[str]] = mapped_column(String(320), index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(200))
    locale: Mapped[str] = mapped_column(String(10), default="it", nullable=False)

    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan",
    )

    @property
    def is_active(self) -> bool:
        return self.deleted_at is None

    def __repr__(self) -> str:  # pragma: no cover - diagnostica
        return f"<User id={self.id} sub={self.keycloak_sub[:8]}>"


class Conversation(Base, TimestampMixin, OwnedMixin):
    """Uno scambio continuato fra un utente e una personalità.

    `personality_version_id` non c'è ancora — le personalità arrivano nella
    fase 3 — ma è prevista dal piano fin d'ora, perché senza quella colonna una
    conversazione di tre mesi fa non è riproducibile: il prompt che l'ha
    prodotta sarebbe già cambiato.

    `last_message_at` è ridondante rispetto a `max(messages.created_at)`, e lo
    è apposta: l'elenco delle conversazioni si ordina per attività recente, e
    quell'ordinamento non può costare una scansione dei messaggi.
    """

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    title: Mapped[Optional[str]] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    last_message_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    owner: Mapped[Optional[User]] = relationship(back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    __table_args__ = (
        CheckConstraint(
            "status in ('active', 'archived', 'deleted')",
            name="ck_conversations_status",
        ),
        # L'unica query che conta davvero: «le mie conversazioni, dalla più
        # recente». Un indice sul solo proprietario obbligherebbe comunque a
        # ordinare.
        Index("ix_conversations_owner_recent", "owner_id", "last_message_at"),
    )


class Message(Base):
    """Un turno. Immutabile: non si corregge, si aggiunge.

    Non eredita da `TimestampMixin` perché un messaggio non viene mai
    modificato, e una colonna `updated_at` che resta uguale a `created_at` per
    sempre invita qualcuno, prima o poi, a modificarlo.

    Non eredita da `OwnedMixin`: il proprietario è quello della conversazione.
    Duplicarlo qui darebbe due verità sulla stessa cosa, e la seconda sarebbe
    quella sbagliata il giorno in cui una conversazione cambia mano.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False,
    )

    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Consumo e costo del turno. JSONB e non colonne fisse perché ogni provider
    # conta diversamente — i token letti da cache, per dire, esistono su
    # Anthropic e non altrove — e perché è qui che si verifica se il risparmio
    # promesso dal CAG sia reale, invece di restare un'ipotesi.
    tokens: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    # Lega il messaggio alla traccia distribuita che l'ha prodotto: da un turno
    # sbagliato si risale agli span di tutti gli stadi.
    correlation_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        CheckConstraint(
            "role in ('user', 'assistant', 'system')", name="ck_messages_role",
        ),
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )


class AuditLog(Base):
    """Chi ha fatto cosa, a cosa, e com'era prima.

    Append-only per disciplina: l'applicazione non espone aggiornamenti né
    cancellazioni su questa tabella. Un registro che si può correggere non è un
    registro.

    Serve già nella fase 0 perché la specifica chiede che l'accesso di un
    amministratore alle memorie di un utente lasci traccia, e quel genere di
    requisito va nello schema dall'inizio: aggiunto dopo, resta scoperto tutto
    ciò che è stato scritto prima.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Nullo per le azioni di sistema (consolidamento memorie, scadenze): «non è
    # stato nessuno» è un'informazione, e va distinta da un utente ignoto.
    actor_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True,
    )

    action: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    target_type: Mapped[Optional[str]] = mapped_column(String(50))
    target_id: Mapped[Optional[str]] = mapped_column(String(64))

    before: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    after: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    ip: Mapped[Optional[str]] = mapped_column(String(45))  # anche IPv6
    correlation_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
    )

    __table_args__ = (
        Index("ix_audit_log_target", "target_type", "target_id"),
    )
