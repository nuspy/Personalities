"""digestione fra i tipi ammessi

Il vincolo `ck_builds_kind` elencava i quattro tipi di addestramento e non
`digestione`: accodarne una falliva sul database vero con una violazione di
check, mentre i test passavano perché costruiscono lo schema dai modelli —
dove il tipo c'era già.

Alembic non se ne accorge da solo: l'autogenerazione confronta tabelle,
colonne e indici, **non** il testo dei vincoli di controllo. Ogni volta che
una tupla come `TIPI_BUILD` cresce, la migrazione va scritta a mano.

Revision ID: b7c1d3e9f204
Revisione precedente: a98d1bcae6b5
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "b7c1d3e9f204"
down_revision: Union[str, None] = "a98d1bcae6b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VECCHIO = "kind in ('lora', 'finetune', 'gguf', 'merge')"
NUOVO = "kind in ('lora', 'finetune', 'gguf', 'merge', 'digestione')"


def upgrade() -> None:
    op.drop_constraint("ck_builds_kind", "builds", type_="check")
    op.create_check_constraint("ck_builds_kind", "builds", NUOVO)


def downgrade() -> None:
    # Le digestioni già registrate violerebbero il vincolo vecchio: si
    # cancellano, perché sono righe di lavoro e non dati di nessuno, e
    # lasciarle renderebbe la revisione impossibile da applicare.
    op.execute("delete from builds where kind = 'digestione'")
    op.drop_constraint("ck_builds_kind", "builds", type_="check")
    op.create_check_constraint("ck_builds_kind", "builds", VECCHIO)
