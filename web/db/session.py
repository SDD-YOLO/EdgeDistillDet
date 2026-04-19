from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from web.core.saas_settings import get_database_url
from web.db.base import Base

_engine: Engine | None = None
SessionLocal = None


def get_engine() -> Engine:
    global _engine, SessionLocal
    if _engine is None:
        url = get_database_url()
        connect_args: dict = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def init_db() -> None:
    """创建表（开发/SQLite 便捷路径；生产建议 Alembic）。"""
    eng = get_engine()
    from web.db import models  # noqa: F401

    Base.metadata.create_all(bind=eng)


def get_db() -> Generator[Session, None, None]:
    get_engine()
    assert SessionLocal is not None
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
