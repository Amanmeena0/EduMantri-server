# src/agents/generation/practice_questions.py
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.ai.prompts.academic_prompts import practice_questions_prompt
from src.ai.retrievers.faiss_retriever import retriever
from src.core.logger import logger

class PracticeQuestionGeneratorAgent(BaseAgent):
    """
    Generates practice questions (MCQs/Short Answers) based on context documents.
    """
    def __init__(self):
        super().__init__("practice_question_generator")
        self.llm = get_llm()

    async def run(self, state: AgentState) -> AgentState:
        logger.info("Executing PracticeQuestionGeneratorAgent...")
        
        # 1. Fetch content
        content = state.extracted_parameters.get("content")
        if not content:
            logger.info("Content missing. Fetching source documents via FAISS database...")
            try:
                retrieved_docs = retriever.invoke(state.query)
                content = "\n\n".join([doc.page_content for doc in retrieved_docs])
            except Exception as e:
                logger.warning(f"Failed to query FAISS retriever: {e}")
                content = state.query

        num_questions = state.extracted_parameters.get("num_questions") or 3
        question_type = state.extracted_parameters.get("question_type") or "multiple-choice"

        # 2. Format and invoke LLM
        prompt_val = practice_questions_prompt.format_prompt(
            content=content,
            num_questions=num_questions,
            question_type=question_type
        )

        response = await self.llm.ainvoke(prompt_val.to_messages())
        state.response = response.content.strip()
        state.agent_responses[self.name] = state.response
        
        return state
