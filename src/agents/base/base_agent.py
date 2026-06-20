# src/agents/base/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    session_id: str
    query: str
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_parameters: Dict[str, Any] = Field(default_factory=dict)
    current_agent: str = "host"
    agent_responses: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    response: str = ""
    followups: List[str] = Field(default_factory=list)

class BaseAgent(ABC):
    """
    Abstract Base Class for all agents.
    All specialized agents must inherit from this class and implement the run method.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        """
        Executes the agent logic.
        Args:
            state: The current conversation state.
        Returns:
            The updated conversation state.
        """
        raise NotImplementedError
