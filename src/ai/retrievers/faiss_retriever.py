# src/ai/retrievers/faiss_retriever.py
import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.core.logger import logger
from src.core.config import settings
from src.ai.embeddings.hf_embeddings import get_embeddings

class FAISSRetrieverManager:
    """
    Manages loading and querying of the FAISS vector database.
    """
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or settings.PERSIST_DIR
        self.embeddings = get_embeddings()
        self.vector_store: Optional[FAISS] = None
        self.retriever = None
        self.initialize_retriever()

    def initialize_retriever(self):
        """Initializes the FAISS vector store and MMR retriever."""
        try:
            logger.info(f"Loading local FAISS database from: {self.persist_directory}")
            if not os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
                logger.warning("FAISS index files not found. The database builder needs to be run first.")
                # We can dynamically initialize a stub database if none exists to avoid API crashes.
                # However, for production it should fail fast or load a stub.
                from langchain_core.documents import Document
                stub_docs = [Document(page_content="EduMantri Study Companion database loading.", metadata={"source": "system"})]
                self.vector_store = FAISS.from_documents(stub_docs, self.embeddings)
            else:
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            
            # Setup retriever with MMR search
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.7,
                }
            )
            logger.info("FAISS vector store and retriever successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load FAISS database: {e}")
            self.vector_store = None
            self.retriever = None

    def get_retriever_with_source_filtering(self, source_types: List[str] = None) -> Any:
        """Get retriever with optional source type filtering"""
        if source_types and self.vector_store:
            logger.info(f"Creating filtered retriever for source_types: {source_types}")
            def filtered_retriever(query: str) -> List[Document]:
                logger.info(f"Filtered retrieval for query: '{query}' with source_types: {source_types}")
                docs = self.vector_store.similarity_search(
                    query, 
                    k=5,
                    filter={"source_type": {"$in": source_types}}
                )
                return docs
            return filtered_retriever
        
        logger.info("Returning standard FAISS retriever")
        return self.retriever

    def debug_retrieval(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Debug function to inspect retrieved documents and similarity scores"""
        if not self.vector_store:
            logger.warning("Vector store not initialized. Cannot perform search.")
            return []
            
        logger.info(f"Running similarity search query: '{query}' with k={k}")
        docs = self.vector_store.similarity_search_with_score(query, k=k)
        debug_info = []
        for doc, score in docs:
            debug_info.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
                "source": doc.metadata.get('source', 'unknown'),
                "chunk_info": {
                    "chunk_index": doc.metadata.get('chunk_index', 'N/A'),
                    "word_count": doc.metadata.get('word_count', 'N/A'),
                    "file_type": doc.metadata.get('file_type', 'N/A')
                }
            })
        return debug_info

# Global retriever manager instance
retriever_manager = FAISSRetrieverManager()
retriever = retriever_manager.retriever
