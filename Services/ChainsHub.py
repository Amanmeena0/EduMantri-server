from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from models.models import get_llms
from utils.retrival import retriver
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from Guardrail.guardrail import guard
from Session import session_manager

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

history_aware_retriever = create_history_aware_retriever(
    llm=get_llms,
    retriever=retriver,
    prompt="contextualize_q_prompt"
)

followup_chain = "followup_prompt" | get_llms | StrOutputParser()

question_answer_chain = create_stuff_documents_chain(get_llms, "qa_prompt")


rag_chain_hist = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain_hist.with_config(run_postprocess_fn=guard),
    session_manager.get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)