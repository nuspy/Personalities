"""avatar e voce

La tabella degli avatar e il volto sulla personalità.

Il vincolo di chiave esterna porta un nome esplicito: l'autogenerazione lo
lascia anonimo e il `downgrade` che ne risulta fallisce — `drop_constraint(None,
...)` non può funzionare. Un difetto che si scopre solo tornando indietro, cioè
nel momento in cui una migrazione reversibile serve davvero.

Revision ID: d957fd1a8d29
Revisione precedente: b7c1d3e9f204
Creata: 2026-09-22 00:24:06.100814+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
# L'autogenerazione scrive `pgvector.sqlalchemy.Vector(...)` nelle colonne di
# embedding ma non ne emette l'import: senza questa riga la migrazione
# fallisce con `NameError` al momento di applicarla, non di generarla.
import pgvector.sqlalchemy
from sqlalchemy.dialects import postgresql

revision: str = 'd957fd1a8d29'
down_revision: Union[str, None] = 'b7c1d3e9f204'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('avatars',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('slug', sa.String(length=80), nullable=False),
    sa.Column('name', sa.String(length=160), nullable=False),
    sa.Column('kind', sa.String(length=20), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('owner_id', sa.BigInteger(), nullable=True),
    sa.CheckConstraint("kind in ('immagine', 'video', 'modello')", name='ck_avatars_kind'),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('slug')
    )
    op.create_index('ix_avatars_kind', 'avatars', ['kind'], unique=False)
    op.create_index('ix_avatars_owner_created_at', 'avatars', ['owner_id', 'created_at'], unique=False)
    op.add_column('personalities', sa.Column('avatar_id', sa.Uuid(), nullable=True))
    op.add_column(
        'personality_versions',
        sa.Column('voice_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_foreign_key(
        'fk_personalities_avatar', 'personalities', 'avatars',
        ['avatar_id'], ['id'], ondelete='SET NULL',
    )


def downgrade() -> None:
    op.drop_constraint(
        'fk_personalities_avatar', 'personalities', type_='foreignkey',
    )
    op.drop_column('personality_versions', 'voice_config')
    op.drop_column('personalities', 'avatar_id')
    op.drop_index('ix_avatars_owner_created_at', table_name='avatars')
    op.drop_index('ix_avatars_kind', table_name='avatars')
    op.drop_table('avatars')
