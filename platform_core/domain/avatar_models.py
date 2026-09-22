"""Gli avatar, come righe.

**Una tabella e non una colonna JSONB sulla personalità.** Un avatar si
riusa: lo stesso ritratto serve a una personalità e alla sua variante in
un'altra lingua, e un modello 3D commissionato una volta non si duplica per
ogni voce che lo indossa. Con una colonna, cambiare il file significherebbe
ritoccare ogni personalità che lo mostra, e scoprire per esclusione quali
fossero.

**Non è versionato con la personalità**, a differenza del prompt. Il volto è
identità presentazionale: sostituire il ritratto non cambia il senso delle
conversazioni passate, mentre cambiare il prompt sì. Legarlo a
`personality_versions` costringerebbe a pubblicare una versione nuova per
correggere un'immagine storta.

La configurazione resta JSONB perché è **diversa per tipo** — un'immagine ha
un indirizzo, un video ha una clip per stato, un modello ha la corrispondenza
fra visemi e morph target — e tre insiemi di colonne quasi sempre nulle
sarebbero una tabella che descrive male tutti e tre. La validazione sta in
`platform_core/avatar`, e avviene al salvataggio: un avatar rotto scoperto
mentre qualcuno conversa è un volto che non compare e nessuna informazione
per chi l'ha configurato.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, OwnedMixin, TimestampMixin, owner_index


class Avatar(Base, TimestampMixin, OwnedMixin):
    """Un volto che una o più personalità possono indossare."""

    __tablename__ = "avatars"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)

    #: `immagine`, `video` o `modello`. Decide come leggere `config` e cosa il
    #: client sa fare con questo avatar — in particolare se può muovere la
    #: bocca sui visemi o no.
    kind: Mapped[str] = mapped_column(String(20), nullable=False)

    description: Mapped[Optional[str]] = mapped_column(Text)
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    __table_args__ = (
        CheckConstraint(
            "kind in ('immagine', 'video', 'modello')", name="ck_avatars_kind",
        ),
        owner_index("avatars", "created_at"),
        Index("ix_avatars_kind", "kind"),
    )
