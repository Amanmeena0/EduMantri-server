# src/tests/test_agents.py
import unittest
import asyncio
from unittest.mock import MagicMock
from src.agents.base.base_agent import BaseAgent, AgentState
from src.agents.orchestrator.orchestrator import MultiAgentOrchestrator

class DummyAgent(BaseAgent):
    async def run(self, state: AgentState) -> AgentState:
        state.response = "dummy response"
        state.metadata["run_executed"] = True
        return state

class TestMultiAgentOrchestrator(unittest.TestCase):
    def test_agent_registration(self):
        orchestrator = MultiAgentOrchestrator()
        agent = DummyAgent("dummy")
        orchestrator.register_agent(agent)
        
        self.assertIn("dummy", orchestrator.agents)
        self.assertEqual(orchestrator.agents["dummy"], agent)

    def test_orchestrator_execution(self):
        orchestrator = MultiAgentOrchestrator()
        dummy = DummyAgent("concept_explainer") # Registered under router mapping
        orchestrator.register_agent(dummy)
        
        # Setup host mock agent
        host_mock = MagicMock(spec=BaseAgent)
        host_mock.name = "host"
        
        # Async run method mock returning state
        async def mock_host_run(state: AgentState) -> AgentState:
            state.metadata["classified_intent"] = "explain_concept"
            state.current_agent = "concept_explainer"
            return state
            
        host_mock.run = mock_host_run
        orchestrator.register_agent(host_mock)
        
        initial_state = AgentState(
            session_id="test_sess",
            query="Explain photosynthesis",
            current_agent="host"
        )
        
        final_state = asyncio.run(orchestrator.execute(initial_state))
        
        self.assertEqual(final_state.response, "dummy response")
        self.assertTrue(final_state.metadata.get("run_executed"))

if __name__ == "__main__":
    unittest.main()
