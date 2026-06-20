# src/agents/orchestrator/workflow.py
import re
from typing import Dict, Any
from src.agents.base.base_agent import AgentState
from src.core.logger import logger

class WorkflowStateManager:
    """
    Manages transformations, extractions, and validations on AgentState.
    """
    @staticmethod
    def extract_parameters(state: AgentState) -> AgentState:
        """
        Regex and pattern parsing on user query to extract parameters.
        """
        query = state.query.lower()
        params: Dict[str, Any] = {}

        # 1. Subject extraction (simple heuristic mapping)
        subjects = ["math", "physics", "chemistry", "biology", "history", "programming", "science", "english"]
        for subj in subjects:
            if subj in query:
                params["subject"] = subj
                break
        
        # 2. Difficulty Level extraction
        if "easy" in query or "beginner" in query:
            params["difficulty_level"] = "beginner"
        elif "hard" in query or "advanced" in query:
            params["difficulty_level"] = "advanced"
        else:
            params["difficulty_level"] = "intermediate"

        # 3. Numeric questions counts extraction
        num_match = re.search(r'(\d+)\s+(flashcard|question|card)', query)
        if num_match:
            params["num_cards"] = int(num_match.group(1))
            params["num_questions"] = int(num_match.group(1))
        else:
            params["num_cards"] = 3
            params["num_questions"] = 3

        # Merge extracted params with existing ones
        state.extracted_parameters.update(params)
        logger.info(f"Extracted workflow parameters: {state.extracted_parameters}")
        return state
