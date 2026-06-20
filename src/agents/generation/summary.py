# src/agents/generation/summary.py
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.ai.prompts.academic_prompts import summary_prompt
from src.ai.retrievers.faiss_retriever import retriever
from src.core.logger import logger

class SummaryGeneratorAgent(BaseAgent):
    """
    Creates structural revisions summaries from source articles/documents.
    """
    def __init__(self):
        super().__init__("summary_generator")
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        logger.info("Executing SummaryGeneratorAgent...")
        
        # 1. Fetch content
        content = state.extracted_parameters.get("content")
        if not content:
            logger.info("Content missing. Querying FAISS to fetch summary source...")
            try:
                retrieved_docs = retriever.invoke(state.query)
                content = "\n\n".join([doc.page_content for doc in retrieved_docs])
            except Exception as e:
                logger.warning(f"Failed to query FAISS retriever: {e}")
                content = state.query

        content_type = state.extracted_parameters.get("content_type") or "academic text"
        length = state.extracted_parameters.get("length") or "3"

        # 2. Format and invoke LLM
        prompt_val = summary_prompt.format_prompt(
            content=content,
            content_type=content_type,
            length=length
        )

        response = await self.llm.ainvoke(prompt_val.to_messages())
        state.response = response.content.strip()
        state.agent_responses[self.name] = state.response
        
        return state
