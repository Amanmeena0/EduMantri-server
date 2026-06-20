# src/guardrails/guardrails.py
import logging
from typing import Dict, Any, List
from src.core.logger import logger
from src.core.exceptions import GuardrailTriggeredException

class GuardrailsManager:
    """
    Validation engine to evaluate inputs and outputs against safety and context constraints.
    """
    @staticmethod
    def validate_input(query: str) -> str:
        """
        Validates input queries. Returns the query if valid, or raises an exception.
        """
        query_strip = query.strip()
        if not query_strip:
            raise GuardrailTriggeredException("Query input cannot be empty.")
            
        # Example safety/academic scope validation:
        # For a production application, check if the input contains harmful content or injection attempts.
        return query_strip

    @staticmethod
    def enhanced_context_guard(inputs: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates generated answer against retrieved context.
        Ensures the model doesn't hallucinate general assumptions when no context is present.
        """
        answer = output.get("answer", "")
        context = inputs.get("context", [])
        
        # If context is empty, force fallback response
        if not context or (isinstance(context, list) and len(context) == 0):
            logger.info("Guardrail triggered: No retrieval context available. Overriding answer.")
            output["answer"] = "I cannot find sufficient information in the available documents to answer your question."
            return output

        # Identify generalized hallucination phrases
        generic_phrases = [
            "in general", "typically", "usually", "most likely", 
            "it is common", "generally speaking"
        ]
        
        if any(phrase in answer.lower() for phrase in generic_phrases):
            if "according to" not in answer.lower() and "based on" not in answer.lower():
                logger.info("Guardrail triggered: Generalized statement without source citations. Overriding answer.")
                output["answer"] = "I cannot find sufficient information in the available documents to answer your question."
                
        return output

# Static convenience handlers
validate_input = GuardrailsManager.validate_input
enhanced_context_guard = GuardrailsManager.enhanced_context_guard
