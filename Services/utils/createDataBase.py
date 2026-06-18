from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
import logging
from Services.models.models import get_embedding
from Services.utils.Chunking import process_documents_with_chunking
from mcp_tools.docTool.fetch import LocalDocuments
from mcp_tools.webTool.scraper import WebDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def create_vector_store():
    """Create and persist vector store from documents"""
    
    logger.info("Loading local documents...")
    local_documents = LocalDocuments.load_documents()

    logger.info("Loading Web Documents...")
    web_docs = WebDocument.web_documents()
    
    all_docs = local_documents + web_docs
    logger.info(f"Total documents Loaded: {len(all_docs)}")

    logger.info("Starting Chuning processes...")
    chunks = process_documents_with_chunking(all_docs)
    logger.info(f"Total Chunks created: {len(chunks)}")

    

   # Create vector store
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=get_embedding
    )

    # Save locally (FAISS does not use persist_directory like Chroma)
    vectorstore.save_local("faiss_index")

    logger.info(f"Created vector store with {len(chunks)} document chunks")
    
    return vectorstore

if __name__ == "__main__":
    create_vector_store()