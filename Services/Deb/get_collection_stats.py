




def get_collection_stats():
    """Get statistics about the vector store collection"""
    try:
        logger.info("Getting collection stats...")
        collection = enhanced_retriever.vector_store._collection
        count = collection.count()
        logger.info(f"Collection '{enhanced_retriever.collection_name}' has {count} documents.")
        return {
            "document_count": count,
            "collection_name": enhanced_retriever.collection_name,
            "persist_directory": enhanced_retriever.persist_directory
        }
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return None
    


def get_collection_stats():
    """Get statistics about the vector store collection"""
    try:
        logger.info("Getting collection stats...")
        collection = enhanced_retriever.vector_store._collection
        count = collection.count()
        logger.info(f"Collection '{enhanced_retriever.collection_name}' has {count} documents.")
        return {
            "document_count": count,
            "collection_name": enhanced_retriever.collection_name,
            "persist_directory": enhanced_retriever.persist_directory
        }
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return None