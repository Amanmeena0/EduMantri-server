# src/agents/doubts/active_recall.py
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.ai.prompts.academic_prompts import active_recall_prompt
from src.core.logger import logger

class ActiveRecallDrillAgent(BaseAgent):
    """
    Quizzes the student interactively without giving answers immediately.
    """
    def __init__(self):
        super().__init__("active_recall_drill")
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        logger.info("Executing ActiveRecallDrillAgent...")
        
        # Extract variables with sensible defaults
        topic = state.extracted_parameters.get("topic") or state.query
        familiarity = state.extracted_parameters.get("familiarity") or "5"
        previous_mistakes = state.extracted_parameters.get("previous_mistakes") or "none"

        # Format prompt template
        prompt_val = active_recall_prompt.format_prompt(
            topic=topic,
            familiarity=familiarity,
            previous_mistakes=previous_mistakes
        )

        response = await self.llm.ainvoke(prompt_val.to_messages())
        state.response = response.content.strip()
        state.agent_responses[self.name] = state.response
        
        return state
