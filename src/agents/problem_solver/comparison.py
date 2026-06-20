# src/agents/problem_solver/comparison.py
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.ai.prompts.academic_prompts import comparison_prompt
from src.core.logger import logger

class ComparisonAgent(BaseAgent):
    """
    Compares and contrasts two academic items in a clean markdown table.
    """
    def __init__(self):
        super().__init__("comparison_agent")
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        logger.info("Executing ComparisonAgent...")
        
        # Extract variables with sensible fallback parsing from the query (e.g. "mitosis vs meiosis")
        query = state.query
        item_a = state.extracted_parameters.get("item_a")
        item_b = state.extracted_parameters.get("item_b")
        
        if not item_a or not item_b:
            logger.info("Extracting items from query containing 'vs' or 'compare'")
            if " vs " in query.lower():
                parts = query.lower().split(" vs ")
                item_a = parts[0].replace("compare", "").strip()
                item_b = parts[1].strip()
            else:
                item_a = "Concept A"
                item_b = "Concept B"

        focus = state.extracted_parameters.get("focus") or "key features, differences, and use cases"

        prompt_val = comparison_prompt.format_prompt(
            item_a=item_a,
            item_b=item_b,
            focus=focus
        )

        response = await self.llm.ainvoke(prompt_val.to_messages())
        state.response = response.content.strip()
        state.agent_responses[self.name] = state.response
        
        return state
