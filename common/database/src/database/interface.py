from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

ModelT = TypeVar("ModelT")


class IRepository(ABC, Generic[ModelT]):
    @abstractmethod
    def get_by_id(self, record_id: Any) -> ModelT | None: ...

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> list[ModelT]: ...

    @abstractmethod
    def create(self, **kwargs: Any) -> ModelT: ...

    @abstractmethod
    def update(self, record_id: Any, **kwargs: Any) -> ModelT | None: ...

    @abstractmethod
    def delete(self, record_id: Any) -> bool: ...


class IUserRepository(IRepository["User"]):  # type: ignore[name-defined]
    @abstractmethod
    def get_by_email(self, email: str) -> Any | None: ...

    @abstractmethod
    def update_last_login(self, user_id: int) -> None: ...


class IModelPredictionRepository(IRepository["ModelPrediction"]):  # type: ignore[name-defined]
    pass


class IWikiDocumentRepository(IRepository["WikiDocument"]):  # type: ignore[name-defined]
    @abstractmethod
    def get_by_url(self, url: str) -> Any | None: ...

    @abstractmethod
    def get_by_species(self, species: str) -> list[Any]: ...


class IConversationRepository(IRepository["Conversation"]):  # type: ignore[name-defined]
    @abstractmethod
    def get_by_id(self, record_id: uuid.UUID) -> Any | None: ...

    @abstractmethod
    def get_by_user_id(self, user_id: int) -> list[Any]: ...

    @abstractmethod
    def update(self, record_id: uuid.UUID, **kwargs: Any) -> Any | None: ...

    @abstractmethod
    def delete(self, record_id: uuid.UUID) -> bool: ...


class IMessageRepository(IRepository["Message"]):  # type: ignore[name-defined]
    @abstractmethod
    def get_by_id(self, record_id: uuid.UUID) -> Any | None: ...

    @abstractmethod
    def get_by_conversation_id(self, conversation_id: uuid.UUID) -> list[Any]: ...

    @abstractmethod
    def update(self, record_id: uuid.UUID, **kwargs: Any) -> Any | None: ...

    @abstractmethod
    def delete(self, record_id: uuid.UUID) -> bool: ...
