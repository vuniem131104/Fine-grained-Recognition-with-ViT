from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.interface import (
    IConversationRepository,
    IMessageRepository,
    IModelPredictionRepository,
    IUserRepository,
    IWikiDocumentRepository,
)
from database.models.conversation_model import Conversation
from database.models.message_model import Message
from database.models.model_prediction_model import ModelPrediction
from database.models.user_model import User
from database.models.wiki_document_model import WikiDocument

load_dotenv()


def _build_url() -> str:
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("MAIN_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


class PostgresDatabase:
    def __init__(self) -> None:
        self.engine = create_engine(
            _build_url(),
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        self._session_factory = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False
        )

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        s = self._session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()


# Module-level singleton used by get_session() and repositories
_db = PostgresDatabase()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    with _db.session() as s:
        yield s


# ── Repositories ────────────────────────────────────────────────────────────

class UserRepository(IUserRepository):
    def __init__(self, session: Session) -> None:
        self._s = session

    def get_by_id(self, record_id: int) -> User | None:
        return self._s.get(User, record_id)

    def get_by_email(self, email: str) -> User | None:
        return self._s.query(User).filter(User.email == email).first()

    def list(self, limit: int = 100, offset: int = 0) -> list[User]:
        return self._s.query(User).offset(offset).limit(limit).all()

    def create(self, **kwargs: Any) -> User:
        user = User(**kwargs)
        self._s.add(user)
        self._s.flush()
        self._s.refresh(user)
        return user

    def update(self, record_id: int, **kwargs: Any) -> User | None:
        user = self.get_by_id(record_id)
        if user is None:
            return None
        for k, v in kwargs.items():
            setattr(user, k, v)
        self._s.flush()
        self._s.refresh(user)
        return user

    def delete(self, record_id: int) -> bool:
        user = self.get_by_id(record_id)
        if user is None:
            return False
        self._s.delete(user)
        self._s.flush()
        return True

    def update_last_login(self, user_id: int) -> None:
        user = self.get_by_id(user_id)
        if user is not None:
            user.last_login = datetime.now(tz=timezone.utc)
            self._s.flush()


class ModelPredictionRepository(IModelPredictionRepository):
    def __init__(self, session: Session) -> None:
        self._s = session

    def get_by_id(self, record_id: int) -> ModelPrediction | None:
        return self._s.get(ModelPrediction, record_id)

    def list(self, limit: int = 100, offset: int = 0) -> list[ModelPrediction]:
        return self._s.query(ModelPrediction).offset(offset).limit(limit).all()

    def create(self, **kwargs: Any) -> ModelPrediction:
        obj = ModelPrediction(**kwargs)
        self._s.add(obj)
        self._s.flush()
        self._s.refresh(obj)
        return obj

    def update(self, record_id: int, **kwargs: Any) -> ModelPrediction | None:
        obj = self.get_by_id(record_id)
        if obj is None:
            return None
        for k, v in kwargs.items():
            setattr(obj, k, v)
        self._s.flush()
        self._s.refresh(obj)
        return obj

    def delete(self, record_id: int) -> bool:
        obj = self.get_by_id(record_id)
        if obj is None:
            return False
        self._s.delete(obj)
        self._s.flush()
        return True


class WikiDocumentRepository(IWikiDocumentRepository):
    def __init__(self, session: Session) -> None:
        self._s = session

    def get_by_id(self, record_id: int) -> WikiDocument | None:
        return self._s.get(WikiDocument, record_id)

    def get_by_url(self, url: str) -> WikiDocument | None:
        return self._s.query(WikiDocument).filter(WikiDocument.url == url).first()

    def get_by_species(self, species: str) -> list[WikiDocument]:
        return self._s.query(WikiDocument).filter(WikiDocument.species == species).all()

    def list(self, limit: int = 100, offset: int = 0) -> list[WikiDocument]:
        return self._s.query(WikiDocument).offset(offset).limit(limit).all()

    def create(self, **kwargs: Any) -> WikiDocument:
        obj = WikiDocument(**kwargs)
        self._s.add(obj)
        self._s.flush()
        self._s.refresh(obj)
        return obj

    def update(self, record_id: int, **kwargs: Any) -> WikiDocument | None:
        obj = self.get_by_id(record_id)
        if obj is None:
            return None
        for k, v in kwargs.items():
            setattr(obj, k, v)
        self._s.flush()
        self._s.refresh(obj)
        return obj

    def delete(self, record_id: int) -> bool:
        obj = self.get_by_id(record_id)
        if obj is None:
            return False
        self._s.delete(obj)
        self._s.flush()
        return True

class ConversationRepository(IConversationRepository):
    def __init__(self, session: Session) -> None:
        self._s = session

    def get_by_id(self, record_id: uuid.UUID) -> Conversation | None:
        return self._s.get(Conversation, record_id)

    def get_by_user_id(self, user_id: int) -> list[Conversation]:
        return self._s.query(Conversation).filter(Conversation.user_id == user_id).all()

    def list(self, limit: int = 100, offset: int = 0) -> list[Conversation]:
        return self._s.query(Conversation).offset(offset).limit(limit).all()

    def create(self, **kwargs: Any) -> Conversation:
        obj = Conversation(**kwargs)
        self._s.add(obj)
        self._s.flush()
        self._s.refresh(obj)
        return obj

    def update(self, record_id: uuid.UUID, **kwargs: Any) -> Conversation | None:
        obj = self.get_by_id(record_id)
        if obj is None:
            return None
        for k, v in kwargs.items():
            setattr(obj, k, v)
        self._s.flush()
        self._s.refresh(obj)
        return obj

    def delete(self, record_id: uuid.UUID) -> bool:
        obj = self.get_by_id(record_id)
        if obj is None:
            return False
        self._s.delete(obj)
        self._s.flush()
        return True


class MessageRepository(IMessageRepository):
    def __init__(self, session: Session) -> None:
        self._s = session

    def get_by_id(self, record_id: uuid.UUID) -> Message | None:
        return self._s.get(Message, record_id)

    def get_by_conversation_id(self, conversation_id: uuid.UUID) -> list[Message]:
        return (
            self._s.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .all()
        )

    def list(self, limit: int = 100, offset: int = 0) -> list[Message]:
        return self._s.query(Message).offset(offset).limit(limit).all()

    def create(self, **kwargs: Any) -> Message:
        obj = Message(**kwargs)
        self._s.add(obj)
        self._s.flush()
        self._s.refresh(obj)
        return obj

    def update(self, record_id: uuid.UUID, **kwargs: Any) -> Message | None:
        obj = self.get_by_id(record_id)
        if obj is None:
            return None
        for k, v in kwargs.items():
            setattr(obj, k, v)
        self._s.flush()
        self._s.refresh(obj)
        return obj

    def delete(self, record_id: uuid.UUID) -> bool:
        obj = self.get_by_id(record_id)
        if obj is None:
            return False
        self._s.delete(obj)
        self._s.flush()
        return True
