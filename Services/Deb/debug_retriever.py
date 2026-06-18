





def debug_retriever(inputs):
    """Enhanced debug function to inspect retrieved documents"""
    logger.info(f"Debug retriever invoked with inputs: {inputs}")
    docs = history_aware_retriever.invoke(inputs)
    logger.info(f"Debug retriever returned {len(docs)} documents.")
    print("\n" + "="*50)
    print("RETRIEVED DOCUMENTS DEBUG")
    print("="*50)
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"File Type: {doc.metadata.get('file_type', 'Unknown')}")
        print(f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
        print(f"Word Count: {doc.metadata.get('word_count', 'N/A')}")
        print(f"Content Preview: {doc.page_content[:300]}...")
        print("-" * 40)
    return docs