# src/agents/generation/study_planner.py
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.ai.prompts.academic_prompts import study_planner_prompt
from src.core.logger import logger

class StudyPlannerAgent(BaseAgent):
    """
    Constructs a realistic study plan layout matching deadline constraints.
    """
    def __init__(self):
        super().__init__("study_planner")
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        logger.info("Executing StudyPlannerAgent...")
        
        # Extract planner params
        topic = state.extracted_parameters.get("topic") or state.query
        hours_available = state.extracted_parameters.get("hours_available") or "2"
        deadline = state.extracted_parameters.get("deadline") or "1 week"
        current_level = state.extracted_parameters.get("current_level") or "3"
        target_level = state.extracted_parameters.get("target_level") or "8"

        prompt_val = study_planner_prompt.format_prompt(
            topic=topic,
            hours_available=hours_available,
            deadline=deadline,
            current_level=current_level,
            target_level=target_level
        )

        response = await self.llm.ainvoke(prompt_val.to_messages())
        state.response = response.content.strip()
        state.agent_responses[self.name] = state.response
        
        return state
