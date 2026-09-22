"""personalità private

Le personalità create da un utente per sé: pubblicate — si usano — ma fuori
dal catalogo. Le esistenti sono tutte del catalogo, e restano pubbliche.

Revision ID: 3d7a2c9e5b18
Revisione precedente: 8415f60b5014
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "3d7a2c9e5b18"
down_revision: Union[str, None] = "8415f60b5014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "personalities",
        sa.Column("visibility", sa.String(length=20), server_default="pubblica", nullable=False),
    )
    op.create_check_constraint(
        "ck_personalities_visibility", "personalities", "visibility in ('pubblica', 'privata')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_personalities_visibility", "personalities", type_="check")
    op.drop_column("personalities", "visibility")
