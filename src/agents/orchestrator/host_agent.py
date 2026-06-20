# src/agents/orchestrator/host_agent.py
import json
from src.agents.base.base_agent import BaseAgent, AgentState
from src.ai.llms.gemini import get_llm
from src.core.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage

class HostAgent(BaseAgent):
    """
    Main Entry Agent (Host). Classifies student intent and extracts key variables
    to route execution to the appropriate specialized sub-agent.
    """
    def __init__(self):
        super().__init__("host")
        self.llm = get_llm(temperature=0.0) # Low temperature for accurate classification

    async def run(self, state: AgentState) -> AgentState:
        logger.info("HostAgent analyzing student intent...")
        
        system_instruction = (
            "You are the Intent Classifier for EduMantri study platform.\n"
            "Analyze the student's query and classify it into exactly one of the following intent labels:\n"
            "- 'explain_concept'\n"
            "- 'solve_problem'\n"
            "- 'create_plan'\n"
            "- 'make_flashcards'\n"
            "- 'correct_misconception'\n"
            "- 'summarize'\n"
            "- 'compare'\n"
            "- 'generate_questions'\n"
            "- 'show_worked_solution'\n"
            "- 'active_recall'\n"
            "- 'general_chat'\n\n"
            "Return a JSON block containing:\n"
            "{\n"
            "  \"intent\": \"<intent_label>\",\n"
            "  \"confidence\": 0.0 to 1.0,\n"
            "  \"is_ambiguous\": true/false,\n"
            "  \"clarifying_question\": \"<question if ambiguous, else empty>\"\n"
            "}\n"
            "Output only the JSON code block. No extra text."
        )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Student Query: '{state.query}'")
        ]

        try:
            response = await self.llm.ainvoke(messages)
            content = response.content.strip()
            
            # Clean json fences if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            logger.info(f"Host classification response: {content}")
            result = json.loads(content)
            
            if result.get("is_ambiguous", False) and result.get("clarifying_question"):
                logger.info("Query is ambiguous. Prompting clarifying question.")
                state.response = result["clarifying_question"]
                state.metadata["classified_intent"] = "end"
                state.current_agent = "end"
            else:
                intent = result.get("intent", "general_chat")
                state.metadata["classified_intent"] = intent
                logger.info(f"Classified student query intent: '{intent}'")
                
        except Exception as e:
            logger.error(f"Failed to classify intent using LLM: {e}. Defaulting to general_chat.")
            state.metadata["classified_intent"] = "general_chat"

        return state
