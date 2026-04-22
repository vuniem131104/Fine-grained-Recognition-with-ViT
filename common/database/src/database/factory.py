from __future__ import annotations

import os

from database.interface import (
    IModelPredictionRepository,
    IUserRepository,
    IWikiDocumentRepository,
    IConversationRepository,
    IMessageRepository,
)
from database.providers.postgres_db import (
    ConversationRepository,
    MessageRepository,
    ModelPredictionRepository,
    UserRepository,
    WikiDocumentRepository,
    get_session,
)

_PROVIDER = os.getenv("DB_PROVIDER")


def get_user_repository() -> tuple[IUserRepository, object]:
    """Return (repo, session_ctx) where session_ctx is used as a context manager."""
    if _PROVIDER == "postgres":
        session_ctx = get_session()
        return UserRepository, session_ctx
    raise NotImplementedError(f"Unknown DB_PROVIDER: {_PROVIDER!r}")


def make_user_repository(session: object) -> IUserRepository:
    if _PROVIDER == "postgres":
        return UserRepository(session)  # type: ignore[arg-type]
    raise NotImplementedError(f"Unknown DB_PROVIDER: {_PROVIDER!r}")


def make_prediction_repository(session: object) -> IModelPredictionRepository:
    if _PROVIDER == "postgres":
        return ModelPredictionRepository(session)  # type: ignore[arg-type]
    raise NotImplementedError(f"Unknown DB_PROVIDER: {_PROVIDER!r}")


def make_wiki_repository(session: object) -> IWikiDocumentRepository:
    if _PROVIDER == "postgres":
        return WikiDocumentRepository(session)  # type: ignore[arg-type]
    raise NotImplementedError(f"Unknown DB_PROVIDER: {_PROVIDER!r}")


def make_conversation_repository(session: object) -> IConversationRepository:
    if _PROVIDER == "postgres":
        return ConversationRepository(session)  # type: ignore[arg-type]
    raise NotImplementedError(f"Unknown DB_PROVIDER: {_PROVIDER!r}")


def make_message_repository(session: object) -> IMessageRepository:
    if _PROVIDER == "postgres":
        return MessageRepository(session)  # type: ignore[arg-type]
    raise NotImplementedError(f"Unknown DB_PROVIDER: {_PROVIDER!r}")
