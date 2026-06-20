# src/agents/problem_solver/concept_solver.py
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.ai.prompts.academic_prompts import concept_explainer_prompt
from src.core.logger import logger

class ConceptExplainerAgent(BaseAgent):
    """
    Tutors the student on specific concepts using simple, engaging analogies
    tailored to the student's difficulty level.
    """
    def __init__(self):
        super().__init__("concept_explainer")
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        logger.info("Executing ConceptExplainerAgent...")
        
        # Extract variables
        subject = state.extracted_parameters.get("subject") or "General Studies"
        concept = state.extracted_parameters.get("concept") or state.query
        difficulty_level = state.extracted_parameters.get("difficulty_level") or "intermediate"

        prompt_val = concept_explainer_prompt.format_prompt(
            subject=subject,
            concept=concept,
            difficulty_level=difficulty_level
        )

        response = await self.llm.ainvoke(prompt_val.to_messages())
        state.response = response.content.strip()
        state.agent_responses[self.name] = state.response
        
        return state
