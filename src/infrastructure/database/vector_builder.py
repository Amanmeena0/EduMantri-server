# src/infrastructure/database/vector_builder.py
import os
from langchain_community.vectorstores import FAISS
from src.core.logger import logger
from src.core.config import settings
from src.ai.embeddings.hf_embeddings import get_embeddings
from src.infrastructure.storage.text_chunker import process_documents_with_chunking
from src.infrastructure.mcp.docs import LocalDocumentLoader
from src.infrastructure.mcp.web import WebDocumentScraper

def create_vector_store() -> FAISS:
    """
    Ingests files from local directories and web pages,
    chunks them, indexes them in FAISS, and persists the vector index.
    """
    logger.info("Initializing document loading services...")
    
    # 1. Load Local Documents
    local_loader = LocalDocumentLoader()
    local_docs = local_loader.load_documents()
    
    # 2. Load Web Documents
    web_scraper = WebDocumentScraper()
    web_docs = web_scraper.load_documents()
    
    all_docs = local_docs + web_docs
    if not all_docs:
        logger.warning("No documents loaded! Creating an empty database fallback. Please place documents in data directories.")
        # Create a single fallback document so FAISS doesn't crash on empty input
        from langchain_core.documents import Document
        all_docs = [Document(page_content="EduMantri Study Companion Platform active.", metadata={"source": "system"})]
        
    logger.info(f"Loaded {len(all_docs)} total documents. Beginning chunking...")
    
    # 3. Process Chunking
    chunks = process_documents_with_chunking(all_docs)
    logger.info(f"Generated {len(chunks)} chunks from loaded content.")
    
    # 4. Generate Embeddings & FAISS store
    logger.info("Instantiating text embeddings model...")
    embeddings = get_embeddings()
    
    logger.info(f"Indexing chunks and building FAISS database stored in {settings.PERSIST_DIR}...")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # 5. Persist FAISS index locally
    os.makedirs(settings.PERSIST_DIR, exist_ok=True)
    vector_store.save_local(settings.PERSIST_DIR)
    logger.info("FAISS vector store persisted successfully.")
    
    return vector_store

if __name__ == "__main__":
    create_vector_store()
