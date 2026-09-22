"""segno di estrazione delle memorie sulle conversazioni

Revision ID: 5c2e7a91d4f0
Revisione precedente: 93be1f9481b8
Creata: 2026-09-22 15:30:00+00:00

Le conversazioni già estratte prima di questa migrazione si riconoscono
dalle memorie che vi rimandano: il segno si deduce da lì, o la prima passata
dopo l'aggiornamento le riestrarrebbe tutte.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '5c2e7a91d4f0'
down_revision: Union[str, None] = '93be1f9481b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'conversations',
        sa.Column('memories_extracted_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        """
        UPDATE conversations c
           SET memories_extracted_at = c.last_message_at
         WHERE EXISTS (
               SELECT 1
                 FROM memories m
                 JOIN messages g ON g.id = m.source_message_id
                WHERE g.conversation_id = c.id
         )
        """
    )


def downgrade() -> None:
    op.drop_column('conversations', 'memories_extracted_at')
