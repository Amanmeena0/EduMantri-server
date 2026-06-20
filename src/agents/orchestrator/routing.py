# src/agents/orchestrator/routing.py
from typing import Dict
from src.agents.base.base_agent import AgentState
from src.core.logger import logger

class Router:
    """
    Decides the next execution step based on AgentState intent and parameters.
    """
    def __init__(self):
        # Map classifier outputs to unified agent names
        self.intent_map: Dict[str, str] = {
            "explain_concept": "concept_explainer",
            "solve_problem": "problem_solver",
            "create_plan": "study_planner",
            "make_flashcards": "flashcard_generator",
            "correct_misconception": "misconception_corrector",
            "summarize": "summary_generator",
            "compare": "comparison_agent",
            "generate_questions": "practice_question_generator",
            "show_worked_solution": "worked_solution_provider",
            "active_recall": "active_recall_drill",
            "general_chat": "general_chat",
            "end": "end"
        }

    def determine_next(self, state: AgentState) -> str:
        """
        Determines which agent should execute next based on current state.
        Returns "end" if execution is complete.
        """
        current = state.current_agent
        
        # If current agent has run and produced a response, we usually route to end
        if current in self.intent_map.values() and current != "host":
            logger.info(f"Agent {current} has executed successfully. Ending flow.")
            return "end"
            
        # If current agent is host, route to the classified intent agent
        if current == "host":
            target_intent = state.metadata.get("classified_intent", "end")
            next_agent = self.intent_map.get(target_intent, "end")
            logger.info(f"Router resolved intent '{target_intent}' to agent '{next_agent}'")
            return next_agent

        return "end"
