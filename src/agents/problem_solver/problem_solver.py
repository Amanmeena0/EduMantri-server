# src/agents/problem_solver/problem_solver.py
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.ai.prompts.academic_prompts import problem_solver_prompt
from src.ai.retrievers.faiss_retriever import retriever
from src.core.logger import logger

class ProblemSolverAgent(BaseAgent):
    """
    Solves science, math, or logic problems step-by-step.
    Queries the vector store for formula sheets if needed.
    """
    def __init__(self):
        super().__init__("problem_solver")
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        logger.info("Executing ProblemSolverAgent...")
        
        # 1. Fetch formula context
        context = state.extracted_parameters.get("context")
        if not context:
            logger.info("Formula context missing. Searching FAISS database for relevant equations...")
            try:
                retrieved_docs = retriever.invoke(state.query)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            except Exception as e:
                logger.warning(f"Failed to retrieve formula reference files: {e}")
                context = "no additional formula reference context found"

        problem = state.extracted_parameters.get("problem") or state.query
        subject = state.extracted_parameters.get("subject") or "General Mathematics"

        # 2. Format and invoke LLM
        prompt_val = problem_solver_prompt.format_prompt(
            problem=problem,
            subject=subject,
            context=context
        )

        response = await self.llm.ainvoke(prompt_val.to_messages())
        state.response = response.content.strip()
        state.agent_responses[self.name] = state.response
        
        return state
