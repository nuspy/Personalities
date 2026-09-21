"""Fondamenta del livello dati.

Due scelte che attraversano tutte le tabelle e conviene fissare qui una volta
sola, perché ripeterle su ogni modello significa prima o poi dimenticarle.

**Il tempo è sempre in UTC e sempre esplicito.** Un `TIMESTAMP` senza fuso è
ambiguo appena due processi girano in zone diverse, e la memoria temporale
della piattaforma — «quando te ne ho parlato l'ultima volta?» — si regge
interamente su confronti fra istanti.

**La proprietà di un dato non è un campo fra gli altri.** `OwnedMixin` esiste
perché l'isolamento fra utenti si possa verificare guardando la classe, non
leggendo ogni query che la tocca.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Adesso, con fuso esplicito."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base dichiarativa comune."""


class TimestampMixin:
    """Quando la riga è nata e quando è stata toccata l'ultima volta.

    I valori di default sono calcolati dal database (`func.now()`) e non da
    Python: con più processi che scrivono, l'orologio del database è l'unico
    riferimento condiviso.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class OwnedMixin:
    """Marca una tabella come contenente dati di un utente.

    Chi eredita da questo mixin **deve** essere letto e scritto attraverso un
    repository che filtra per proprietario. Il filtro non può vivere nei
    router: basta un endpoint scritto in fretta perché i dati di un utente
    finiscano a un altro, e quel genere di difetto non si manifesta in un test
    che usa un solo utente.

    L'indice sul proprietario non è un'ottimizzazione accessoria: ogni query
    su queste tabelle comincia da lì. Non viene però creato qui, perché quasi
    sempre la query reale non è «le righe di questo utente» ma «le righe di
    questo utente, ordinate per data» — e un indice sul solo `owner_id`
    resterebbe accanto al composito senza essere mai usato, pagato a ogni
    scrittura. Ogni tabella dichiara il proprio con `owner_index()`, e che
    nessuna se ne dimentichi lo verifica un test, non la buona memoria.
    """

    owner_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )


def owner_index(table_name: str, *extra_columns: str) -> Index:
    """Indice composito che parte dal proprietario.

    Le query reali non sono «tutte le conversazioni»: sono «le conversazioni
    di questo utente, dalla più recente». Un indice sul solo `owner_id`
    costringe comunque a ordinare.
    """
    return Index(
        f"ix_{table_name}_owner_{'_'.join(extra_columns)}",
        "owner_id",
        *extra_columns,
    )
