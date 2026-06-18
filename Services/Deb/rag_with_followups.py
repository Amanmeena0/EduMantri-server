



# Function to generate follow-up questions based on context


def retrieve_and_split_docs(inputs: dict) -> dict:
    all_docs = history_aware_retriever.invoke(inputs)
    logger.info(f"Retrieved {len(all_docs)} total documents")
    
    # Ensure we have enough documents
    if len(all_docs) >= 3:
        primary_docs = all_docs[:3]
        # Use remaining docs for follow-up, or overlap if needed
        followup_docs = all_docs[2:5] if len(all_docs) > 3 else all_docs[1:3]
    elif len(all_docs) >= 2:
        primary_docs = all_docs[:2]
        followup_docs = all_docs[1:2]
    elif len(all_docs) == 1:
        primary_docs = all_docs
        followup_docs = all_docs  # Use the same doc for both
    else:
        primary_docs = []
        followup_docs = []
    
    logger.info(f"Split into {len(primary_docs)} primary docs and {len(followup_docs)} followup docs")
    return {"primary_docs": primary_docs, "followup_docs": followup_docs}



def generate_followups(query, docs):
    if not docs:
        logger.info("No documents available for follow-up generation")
        return []
    
    try:
        context_text = "\n\n".join([doc.page_content for doc in docs])
        if not context_text.strip():
            logger.info("Empty context text for follow-up generation")
            return []
            
        response = followup_chain.invoke({"context": context_text, "input": query})
        
        # Better parsing of follow-up questions
        followups = []
        if response and response.strip():
            # Split by newlines and filter out empty lines
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Clean up and format questions
            for line in lines:
                # Remove numbering (1., 2., etc.) and clean up
                cleaned = line.strip()
                if cleaned:
                    # Remove common prefixes like "1.", "2.", "-", "*", etc.
                    import re
                    cleaned = re.sub(r'^[\d\.\-\*\s]*', '', cleaned).strip()
                    if cleaned and len(cleaned) > 10:  # Ensure meaningful questions
                        followups.append(cleaned)
        
        logger.info(f"Generated {len(followups)} follow-up questions")
        return followups[:2]  # Limit to 2 questions as intended
        
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return []

def rag_with_followups(input_text: str, chat_history: List[dict]) -> dict:
    inputs = {"input": input_text, "chat_history": chat_history}
    doc_dict = split_docs_chain.invoke(inputs)
    
    answer = question_answer_chain.invoke({
        "input": input_text,
        "chat_history": chat_history,
        "context": doc_dict["primary_docs"]
    })
    
    # More robust fallback detection
    fallback_phrases = [
        "I cannot find sufficient information",
        "cannot find information",
        "no information available",
        "insufficient information",
        "not enough information"
    ]
    
    fallback = any(phrase.lower() in answer.lower() for phrase in fallback_phrases)
    
    followups = []
    if not fallback:
        # Ensure we have documents for follow-up generation
        followup_docs = doc_dict.get("followup_docs", [])
        
        # If followup_docs is empty, use primary_docs for follow-up generation
        if not followup_docs and doc_dict.get("primary_docs"):
            logger.info("Using primary docs for follow-up generation as followup_docs is empty")
            followup_docs = doc_dict["primary_docs"]
        
        if followup_docs:
            followups = generate_followups(input_text, followup_docs)
        else:
            logger.info("No documents available for follow-up generation")
    else:
        logger.info("Fallback response detected, skipping follow-up generation")
    
    return {
        "answer": answer,
        "followup_questions": followups
    }



def debug_followup_generation(query: str, chat_history: List[dict] = None) -> dict:
    """Debug function to test follow-up question generation"""
    if chat_history is None:
        chat_history = []
    
    logger.info(f"Debug follow-up generation for query: '{query}'")
    
    inputs = {"input": query, "chat_history": chat_history}
    doc_dict = split_docs_chain.invoke(inputs)
    
    print(f"\n{'='*50}")
    print(f"FOLLOW-UP GENERATION DEBUG: '{query}'")
    print(f"{'='*50}")
    print(f"Primary docs count: {len(doc_dict['primary_docs'])}")
    print(f"Followup docs count: {len(doc_dict['followup_docs'])}")
    
    if doc_dict['followup_docs']:
        print("\nFollowup docs content preview:")
        for i, doc in enumerate(doc_dict['followup_docs'], 1):
            print(f"Doc {i}: {doc.page_content[:150]}...")
    
    followups = generate_followups(query, doc_dict['followup_docs'])
    print(f"\nGenerated follow-ups: {followups}")
    print(f"{'='*50}")
    
    return {
        "followup_questions": followups,
        "doc_counts": {
            "primary": len(doc_dict['primary_docs']),
            "followup": len(doc_dict['followup_docs'])
        }
    }