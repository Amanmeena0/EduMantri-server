# src/infrastructure/cache/in_memory_store.py
from typing import List, Dict, Any
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from src.core.logger import logger

class SessionManager:
    """
    Manages session chat histories in memory.
    Designed with a clean interface for future Redis/database transition.
    """
    def __init__(self):
        self.store: Dict[str, ChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            logger.info(f"Created new conversation session: {session_id}")
        return self.store[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.store:
            del self.store[session_id]
            logger.info(f"Cleared session history for: {session_id}")

    def get_active_sessions(self) -> List[str]:
        return list(self.store.keys())

session_manager = SessionManager()
