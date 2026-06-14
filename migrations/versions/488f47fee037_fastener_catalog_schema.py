"""fastener catalog schema

Revision ID: 488f47fee037
Revises: a0e34dad1e9a
Create Date: 2026-06-13 17:10:27.870271

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '488f47fee037'
down_revision: Union[str, None] = 'a0e34dad1e9a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # catalog_items is reloaded wholesale from CSV after migration (app.ingest.catalog),
    # so the retarget to fastener attributes recreates the table rather than
    # ALTER-ing columns (SQLite can't add NOT NULL columns without defaults to a
    # populated table without batch-recreating anyway).
    op.drop_table('catalog_items')
    op.create_table('catalog_items',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sku', sa.String(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=False),
    sa.Column('diameter', sa.String(), nullable=False),
    sa.Column('thread_pitch_or_tpi', sa.String(), nullable=False),
    sa.Column('length', sa.String(), nullable=False),
    sa.Column('grade_or_class', sa.String(), nullable=False),
    sa.Column('material', sa.String(), nullable=False),
    sa.Column('finish_coating', sa.String(), nullable=False),
    sa.Column('head_type', sa.String(), nullable=False),
    sa.Column('drive_type', sa.String(), nullable=False),
    sa.Column('standard', sa.String(), nullable=False),
    sa.Column('thread_direction', sa.String(), nullable=False),
    sa.Column('units', sa.String(), nullable=False),
    sa.Column('price_usd', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_catalog_items_sku'), 'catalog_items', ['sku'], unique=True)
    op.create_index(op.f('ix_catalog_items_type'), 'catalog_items', ['type'], unique=False)
    op.create_index(op.f('ix_catalog_items_units'), 'catalog_items', ['units'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_catalog_items_units'), table_name='catalog_items')
    op.drop_index(op.f('ix_catalog_items_type'), table_name='catalog_items')
    op.drop_index(op.f('ix_catalog_items_sku'), table_name='catalog_items')
    op.drop_table('catalog_items')
    op.create_table('catalog_items',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sku', sa.String(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=False),
    sa.Column('size', sa.String(), nullable=False),
    sa.Column('thread_standard', sa.String(), nullable=False),
    sa.Column('material_grade', sa.String(), nullable=False),
    sa.Column('pressure_rating_psi', sa.Integer(), nullable=False),
    sa.Column('price_usd', sa.Float(), nullable=False),
    sa.Column('common_end_markets', sa.JSON(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_catalog_items_sku'), 'catalog_items', ['sku'], unique=True)
    op.create_index(op.f('ix_catalog_items_type'), 'catalog_items', ['type'], unique=False)
