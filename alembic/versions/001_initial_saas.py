"""initial saas tables

Revision ID: 001
Revises:
Create Date: 2026-04-19

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "oauth_accounts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("provider", sa.String(32)),
        sa.Column("provider_user_id", sa.String(255)),
        sa.Column("email", sa.String(320), nullable=True),
        sa.UniqueConstraint("provider", "provider_user_id", name="uq_oauth_provider_uid"),
    )
    op.create_table(
        "teams",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200)),
        sa.Column("slug", sa.String(200), unique=True),
        sa.Column("owner_id", sa.String(36), sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_table(
        "team_memberships",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("team_id", sa.String(36), sa.ForeignKey("teams.id", ondelete="CASCADE")),
        sa.Column("role", sa.String(20)),
        sa.UniqueConstraint("user_id", "team_id", name="uq_membership_user_team"),
    )
    op.create_table(
        "team_invitations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("team_id", sa.String(36), sa.ForeignKey("teams.id", ondelete="CASCADE")),
        sa.Column("email", sa.String(320)),
        sa.Column("token_hash", sa.String(128)),
        sa.Column("role", sa.String(20)),
        sa.Column("expires_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("team_id", sa.String(36), sa.ForeignKey("teams.id", ondelete="CASCADE")),
        sa.Column("name", sa.String(200)),
        sa.Column("description", sa.Text()),
        sa.Column("storage_subpath", sa.String(500)),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_table(
        "training_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("team_id", sa.String(36), sa.ForeignKey("teams.id", ondelete="CASCADE")),
        sa.Column("created_by", sa.String(36), sa.ForeignKey("users.id")),
        sa.Column("status", sa.String(32)),
        sa.Column("config_name", sa.String(500)),
        sa.Column("mode", sa.String(32)),
        sa.Column("checkpoint", sa.String(2000), nullable=True),
        sa.Column("allow_overwrite", sa.Boolean()),
        sa.Column("log_path", sa.String(2000), nullable=True),
        sa.Column("rq_job_id", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("team_id", sa.String(36), sa.ForeignKey("teams.id", ondelete="CASCADE")),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("name", sa.String(200)),
        sa.Column("key_prefix", sa.String(16)),
        sa.Column("key_hash", sa.String(128)),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("team_id", sa.String(36), sa.ForeignKey("teams.id", ondelete="SET NULL"), nullable=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(128)),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime()),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("api_keys")
    op.drop_table("training_jobs")
    op.drop_table("datasets")
    op.drop_table("team_invitations")
    op.drop_table("team_memberships")
    op.drop_table("teams")
    op.drop_table("oauth_accounts")
    op.drop_table("users")
