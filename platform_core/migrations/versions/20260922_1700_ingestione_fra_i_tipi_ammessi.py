"""ingestione fra i tipi ammessi

I file caricati dalla console diventano un lavoro del worker, come la
digestione: il vincolo `ck_builds_kind` va allargato a mano, perché
l'autogenerazione di Alembic non confronta il testo dei vincoli di controllo.

Revision ID: 8e41f6c2a7d3
Revisione precedente: 5c2e7a91d4f0
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "8e41f6c2a7d3"
down_revision: Union[str, None] = "5c2e7a91d4f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VECCHIO = "kind in ('lora', 'finetune', 'gguf', 'merge', 'digestione')"
NUOVO = "kind in ('lora', 'finetune', 'gguf', 'merge', 'digestione', 'ingestione')"


def upgrade() -> None:
    op.drop_constraint("ck_builds_kind", "builds", type_="check")
    op.create_check_constraint("ck_builds_kind", "builds", NUOVO)


def downgrade() -> None:
    # Le ingestioni registrate violerebbero il vincolo vecchio: sono righe di
    # lavoro, non dati di nessuno — i documenti che hanno aggiunto restano.
    op.execute("delete from builds where kind = 'ingestione'")
    op.drop_constraint("ck_builds_kind", "builds", type_="check")
    op.create_check_constraint("ck_builds_kind", "builds", VECCHIO)
