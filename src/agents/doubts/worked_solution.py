# src/agents/doubts/worked_solution.py
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.ai.prompts.academic_prompts import worked_solution_prompt
from src.core.logger import logger

class WorkedSolutionProviderAgent(BaseAgent):
    """
    Generates step-by-step worked solutions for math/science questions,
    optionally evaluating student's prior attempted steps.
    """
    def __init__(self):
        super().__init__("worked_solution_provider")
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        logger.info("Executing WorkedSolutionProviderAgent...")
        
        # Extract variables with sensible defaults
        problem = state.extracted_parameters.get("problem") or state.query
        problem_type = state.extracted_parameters.get("problem_type") or "academic"
        student_work = state.extracted_parameters.get("student_work") or "no student work provided"

        # Format prompt template
        prompt_val = worked_solution_prompt.format_prompt(
            problem=problem,
            problem_type=problem_type,
            student_work=student_work
        )

        response = await self.llm.ainvoke(prompt_val.to_messages())
        state.response = response.content.strip()
        state.agent_responses[self.name] = state.response
        
        return state
