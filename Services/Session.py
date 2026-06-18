
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import logging
from typing import List
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self.store = {}
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            logger.info(f"Created new session: {session_id}")
        return self.store[session_id]
    
    def clear_session(self, session_id: str):
        if session_id in self.store:
            del self.store[session_id]
            logger.info(f"Cleared session: {session_id}")
    
    def get_active_sessions(self) -> List[str]:
        return list(self.store.keys())
    

session_manager = SessionManager()