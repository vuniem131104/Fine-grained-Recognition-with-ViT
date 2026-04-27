import uuid
from structlog import get_logger
from database import (
    ConversationRepository,
    MessageRepository,
    get_session,
)
from chatbot.agents.state import ChatbotState

logger = get_logger(__name__)

HISTORY_LIMIT = 5


class MemoryManagementService:
    async def gprocess(self, state: ChatbotState) -> dict:
        """Load conversation history for the user and inject it into state."""
        user_id: int | None = state.get("user_id")
        conversation_id: uuid.UUID | None = state.get("conversation_id")
        if not user_id:
            return {"conversation_history": [], "conversation_id": None}

        try:
            history, resolved_conversation_id = self._load_history(user_id, conversation_id)
            return {"conversation_history": history, "conversation_id": resolved_conversation_id}
        except Exception as e:
            logger.exception("Failed to load conversation history", error=str(e))
            return {"conversation_history": [], "conversation_id": conversation_id}

    def _load_history(
        self, user_id: int, conversation_id: uuid.UUID | None
    ) -> tuple[list[dict], uuid.UUID | None]:
        """Return last HISTORY_LIMIT messages for the given conversation."""
        with get_session() as session:
            conv_repo = ConversationRepository(session)
            msg_repo = MessageRepository(session)

            if conversation_id:
                conv = conv_repo.get_by_id(conversation_id)
                if conv and conv.user_id == user_id:
                    messages = msg_repo.get_by_conversation_id(conv.id)
                    history = []
                    for msg in messages[-HISTORY_LIMIT:]:
                        history.append({"role": "user", "content": msg.user_message})
                        history.append({"role": "assistant", "content": msg.assistant_message[:500]})
                    return history, conv.id
                logger.warning(
                    "conversation_id not found or belongs to different user",
                    conversation_id=str(conversation_id),
                    user_id=user_id,
                )

            return [], None

    def _save_message(
        self,
        user_id: int,
        conversation_id: uuid.UUID | None,
        user_message: str,
        assistant_message: str,
    ) -> uuid.UUID:
        """Save message to the given conversation (or create a new one). Returns conversation_id."""
        with get_session() as session:
            conv_repo = ConversationRepository(session)
            msg_repo = MessageRepository(session)

            if conversation_id:
                conv = conv_repo.get_by_id(conversation_id)
                if conv is None:
                    conv = conv_repo.create(id=conversation_id, user_id=user_id)
                elif conv.user_id != user_id:
                    logger.warning(
                        "conversation_id belongs to different user",
                        conversation_id=str(conversation_id),
                        user_id=user_id,
                    )
                    conv = conv_repo.create(user_id=user_id)
            else:
                conv = conv_repo.create(user_id=user_id)

            msg_repo.create(
                conversation_id=conv.id,
                user_message=user_message,
                assistant_message=assistant_message,
            )
            logger.info("Message saved", user_id=user_id, conversation_id=str(conv.id))
            return conv.id