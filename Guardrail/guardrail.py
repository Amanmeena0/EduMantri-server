import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Guardrail:

    def enhanced_context_guard(inputs: dict, output: dict) -> dict:
        """Enhanced guardrail with better context validation"""
        answer = output.get("answer", "")
        context = inputs.get("context", [])
        if not context or (isinstance(context, list) and len(context) == 0):
            output["answer"] = "I cannot find sufficient information in the available DGFT documents to answer your question."
            logger.info("Guardrail triggered: No context available")
            return output
        generic_phrases = [
            "in general", "typically", "usually", "most likely", 
            "it is common", "generally speaking"
        ]
        if any(phrase in answer.lower() for phrase in generic_phrases):
            if "according to" not in answer.lower() and "based on" not in answer.lower():
                output["answer"] = "I cannot find sufficient information in the available DGFT documents to answer your question."
                logger.info("Guardrail triggered: Generic answer detected")
        return output


guard = Guardrail()