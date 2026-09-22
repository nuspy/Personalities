"""Assegnazione dei modelli ai compiti

Quale modello serve la conversazione, la digestione, il recupero e il
giudizio. Solo il **nome**, scelto in un elenco che sta nell'ambiente:
indirizzi e chiavi non passano mai dal database, e quindi nemmeno dalla
console che lo scrive.

Revision ID: bf7597c1985b
Revisione precedente: 3d7a2c9e5b18
Creata: 2026-09-22 20:50:57.016086+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'bf7597c1985b'
down_revision: Union[str, None] = '3d7a2c9e5b18'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_assignments",
        sa.Column("task", sa.String(length=40), primary_key=True),
        sa.Column("model_name", sa.String(length=60), nullable=False),
        # Nullo quando a decidere è stato un processo, non una persona.
        sa.Column("updated_by", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.ForeignKeyConstraint(["updated_by"], ["users.id"], ondelete="SET NULL"),
    )


def downgrade() -> None:
    op.drop_table("model_assignments")
