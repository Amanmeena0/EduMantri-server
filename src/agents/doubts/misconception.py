# src/agents/doubts/misconception.py
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.ai.prompts.academic_prompts import misconception_prompt
from src.core.logger import logger

class MisconceptionCorrectorAgent(BaseAgent):
    """
    Gently corrects misunderstandings by validating why the misconception is reasonable,
    clarifying the error, providing the correct concept, and offering mnemonics.
    """
    def __init__(self):
        super().__init__("misconception_corrector")
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        logger.info("Executing MisconceptionCorrectorAgent...")
        
        # Extract variables with sensible defaults
        student_belief = state.extracted_parameters.get("student_belief") or state.query
        correct_concept = state.extracted_parameters.get("correct_concept") or "the scientifically accepted explanation"
        subject = state.extracted_parameters.get("subject") or "general studies"

        # Format prompt template
        prompt_val = misconception_prompt.format_prompt(
            student_belief=student_belief,
            correct_concept=correct_concept,
            subject=subject
        )

        response = await self.llm.ainvoke(prompt_val.to_messages())
        state.response = response.content.strip()
        state.agent_responses[self.name] = state.response
        
        return state
