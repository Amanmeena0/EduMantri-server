# src/api/routes/chat.py
import uuid
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from src.schemas.api_schemas import QueryInput, BotResponse
from src.agents.base.base_agent import AgentState
from src.api.dependencies.orchestrator_dep import get_orchestrator
from src.agents.orchestrator.orchestrator import MultiAgentOrchestrator
from src.infrastructure.cache.in_memory_store import session_manager
from src.guardrails.guardrails import validate_input, enhanced_context_guard
from src.core.logger import logger
from src.core.exceptions import EduMantriException

router = APIRouter(prefix="", tags=["chat"])

@router.post("/chat", response_model=BotResponse)
async def post_chat_query(
    input_data: QueryInput,
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """
    Processes the user query through the multi-agent study helper system,
    managing state routing, retrieval contexts, and guardrail validations.
    """
    try:
        # 1. Run Input Guardrails
        validated_query = validate_input(input_data.query)
    except Exception as e:
        logger.warning(f"Input validation guardrail triggered: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # Generate or resolve session
    session_id = input_data.session_id or str(uuid.uuid4())
    logger.info(f"Incoming request for session: {session_id}, query: '{validated_query}'")

    # 2. Retrieve history and format
    session_history = session_manager.get_session_history(session_id)
    history_list: List[Dict[str, Any]] = []
    for msg in session_history.messages:
        role = "human" if msg.type == "human" else "ai"
        history_list.append({"role": role, "content": msg.content})

    # 3. Construct Initial Agent State
    state = AgentState(
        session_id=session_id,
        query=validated_query,
        chat_history=history_list,
        current_agent="host"
    )

    try:
        # 4. Execute Multi-Agent Orchestrator Graph
        final_state = await orchestrator.execute(state)
        
        # 5. Apply Output Guardrails (context fidelity checks)
        # Wrap outputs to match expected dict structure by retriever guards
        inputs_dict = {"context": [doc.page_content for doc in session_history.messages] if session_history.messages else ["fallback"]}
        output_dict = {"answer": final_state.response}
        
        # Only run context guard checks if the response isn't a direct chat greeting
        if final_state.metadata.get("classified_intent") not in ["general_chat", "create_plan"]:
            checked_output = enhanced_context_guard(inputs_dict, output_dict)
            final_response = checked_output["answer"]
        else:
            final_response = final_state.response

        # 6. Synchronize Session History Store
        session_history.add_user_message(validated_query)
        session_history.add_ai_message(final_response)

        # Fallback default suggestions for follow-ups
        followups = final_state.followups
        if not followups:
            # Smart default suggestions depending on intent
            intent = final_state.metadata.get("classified_intent", "")
            if intent == "explain_concept":
                followups = ["Test my knowledge on this", "Give me a worked solution", "Summarize this explanation"]
            elif intent == "solve_problem":
                followups = ["Explain the core concept here", "Show a worked solution", "Give me a similar problem"]
            else:
                followups = ["Explain this topic", "Generate practice questions", "Make flashcards"]

        return BotResponse(
            response=final_response,
            followups=followups
        )

    except EduMantriException as e:
        logger.error(f"EduMantri execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected system error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
