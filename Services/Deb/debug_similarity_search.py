









def debug_similarity_search(query: str, k: int = 3):
    """Debug similarity search with scores"""
    logger.info(f"Debug similarity search for query: '{query}' with k={k}")
    debug_info = enhanced_retriever.debug_retrieval(query, k)
    logger.info(f"Similarity search returned {len(debug_info)} results.")
    print(f"\n{'='*50}")
    print(f"SIMILARITY SEARCH DEBUG: '{query}'")
    print(f"{'='*50}")
    for i, info in enumerate(debug_info, 1):
        print(f"\n--- Result {i} (Score: {info['score']:.4f}) ---")
        print(f"Source: {info['source']}")
        print(f"Chunk Index: {info['chunk_info']['chunk_index']}")
        print(f"Word Count: {info['chunk_info']['word_count']}")
        print(f"File Type: {info['chunk_info']['file_type']}")
        print(f"Content: {info['content']}")
        print("-" * 40)
    return debug_info