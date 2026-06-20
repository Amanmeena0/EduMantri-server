# src/agents/orchestrator/orchestrator.py
import logging
from typing import Dict
from src.agents.base.base_agent import BaseAgent, AgentState
from src.agents.orchestrator.routing import Router
from src.agents.orchestrator.workflow import WorkflowStateManager
from src.core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    """
    Orchestrates execution of the Multi-Agent EduMantri student platform.
    Coordinates sequential execution and intent routing.
    """
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.router = Router()

    def register_agent(self, agent: BaseAgent):
        self.agents[agent.name] = agent
        logger.info(f"Registered Agent: '{agent.name}'")

    async def execute(self, initial_state: AgentState) -> AgentState:
        state = initial_state
        visited_agents = set()
        max_steps = 6  # Loop safety counter

        # Pre-process state parameters
        state = WorkflowStateManager.extract_parameters(state)

        while state.current_agent != "end" and len(visited_agents) < max_steps:
            current_name = state.current_agent
            
            # Check if agent is registered
            if current_name == "general_chat":
                # Handle general chat fallback inline to save LLM roundtrips
                state.response = "Hello! I am your EduMantri AI Study Companion. How can I help you study today?"
                state.followups = ["Explain photosynthesis", "Solve a math problem", "Create a study plan"]
                state.current_agent = "end"
                break
                
            if current_name not in self.agents:
                logger.warning(f"Target agent '{current_name}' not found. Defaulting to host.")
                current_name = "host"
                state.current_agent = "host"

            logger.info(f"Orchestrator running agent: '{current_name}'")
            visited_agents.add(current_name)
            
            # Invoke agent
            try:
                agent = self.agents[current_name]
                state = await agent.run(state)
            except Exception as e:
                logger.exception(f"Error executing agent {current_name}: {e}")
                raise AgentExecutionError(f"Agent {current_name} failed: {e}")
            
            # Retrieve routing decision
            state.current_agent = self.router.determine_next(state)

        logger.info("Multi-agent flow execution complete.")
        return state
