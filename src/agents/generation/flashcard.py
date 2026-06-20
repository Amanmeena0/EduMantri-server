# src/agents/generation/flashcard.py
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.ai.prompts.academic_prompts import flashcard_prompt
from src.ai.retrievers.faiss_retriever import retriever
from src.core.logger import logger

class FlashcardGeneratorAgent(BaseAgent):
    """
    Creates study flashcards (front/back) using source documents retrieved from database.
    """
    def __init__(self):
        super().__init__("flashcard_generator")
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        logger.info("Executing FlashcardGeneratorAgent...")
        
        # 1. Gather context
        content = state.extracted_parameters.get("content")
        
        # Fallback to FAISS RAG if content is not pre-extracted
        if not content:
            logger.info("Content missing. Fetching relevant documents via FAISS database...")
            try:
                retrieved_docs = retriever.invoke(state.query)
                content = "\n\n".join([doc.page_content for doc in retrieved_docs])
                logger.info(f"Retrieved {len(retrieved_docs)} context documents from FAISS.")
            except Exception as e:
                logger.warning(f"Failed to query FAISS retriever: {e}. Falling back to default query content.")
                content = state.query
                
        num_cards = state.extracted_parameters.get("num_cards") or 3

        # 2. Format and invoke LLM
        prompt_val = flashcard_prompt.format_prompt(
            content=content,
            num_cards=num_cards
        )

        response = await self.llm.ainvoke(prompt_val.to_messages())
        state.response = response.content.strip()
        state.agent_responses[self.name] = state.response
        
        return state
