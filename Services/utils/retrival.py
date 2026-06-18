import os 
import logging 

from langchain_community.vectorstores import FAISS
from Services.models.models import get_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRetriever:
    """Enhanced Retriever"""

    def __init__(self, persist_directory: str= "./faiss_index"):
        self.presist_directory = persist_directory
        self.embeddings = get_embedding
        self.vector_store = None
        self.retriver = None
        self.retriver()


    def initilize_retriver_mmr(self):
        """Initialize the FAISS vector store and retriever"""

        try:
            logger.info(
                f"Initializing FAISS vector store from: {self.persist_directory}"
            )

            # 🔹 Load FAISS index
            self.vector_store = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # 🔹 Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.7,
                }
            )

            logger.info("Vector store and retriever initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise


    
    def get_retriever_with_source_filtering(self, source_types: List[str] = None) -> Any:
            """Get retriever with optional source type filtering"""
            if source_types:
                logger.info(f"Creating filtered retriever for source_types: {source_types}")
                def filtered_retriever(query: str) -> List[Document]:
                    logger.info(f"Filtered retrieval for query: '{query}' with source_types: {source_types}")
                    docs = self.vector_store.similarity_search(
                        query, 
                        k=5,
                        filter={"source_type": {"$in": source_types}}
                    )
                    logger.info(f"Filtered retriever returned {len(docs)} documents.")
                    return docs
                return filtered_retriever
            logger.info("Returning default retriever (no source filtering)")
            return self.retriever

    def debug_retrieval(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
            """Debug function to inspect retrieved documents"""
            logger.info(f"Running debug retrieval for query: '{query}' with k={k}")
            docs = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Retrieved {len(docs)} documents with scores.")
            debug_info = []
            for doc, score in docs:
                logger.info(f"Doc source: {doc.metadata.get('source', 'unknown')}, Score: {score}")
                debug_info.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "score": score,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get('source', 'unknown'),
                    "chunk_info": {
                        "chunk_index": doc.metadata.get('chunk_index', 'N/A'),
                        "word_count": doc.metadata.get('word_count', 'N/A'),
                        "file_type": doc.metadata.get('file_type', 'N/A')
                    }
                })
            return debug_info







enhanced_retriver = EnhancedRetriever()
retriver = enhanced_retriver.retriever