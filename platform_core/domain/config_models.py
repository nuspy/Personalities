"""Le scelte di configurazione che si fanno dalla console.

Una tabella sola, e per una ragione: qui sta **quale** modello serve un
compito, mai **dove** si trova né con quale chiave. Gli indirizzi restano
nell'ambiente (`PERSONA_MODELLI`), e questa tabella conserva un nome scelto
in un elenco chiuso. La differenza non è di stile: un endpoint modificabile
dall'interfaccia è traffico dirottabile con le chiavi appresso.

Una riga per compito, non uno storico: conta l'assegnazione di adesso. Chi
ha cambiato cosa sta nel registro di audit, che è il posto delle storie.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


class ModelAssignment(Base, TimestampMixin):
    """A quale modello è assegnato un compito."""

    __tablename__ = "model_assignments"

    #: Il valore di `Compito`: conversazione, digestione, recupero, giudizio.
    task: Mapped[str] = mapped_column(String(40), primary_key=True)
    #: Il nome del modello nell'elenco configurato. Non c'è vincolo di chiave
    #: esterna perché l'elenco non è una tabella: se il nome sparisce
    #: dall'ambiente il compito ricade sul predefinito, e il registro lo dice
    #: nei log invece di far fallire l'avvio.
    model_name: Mapped[str] = mapped_column(String(60), nullable=False)
    #: Chi l'ha deciso. Nullo se l'ha fatto un processo automatico.
    updated_by: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="SET NULL"),
    )
