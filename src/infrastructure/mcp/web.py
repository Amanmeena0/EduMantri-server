# src/infrastructure/mcp/web.py
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from src.core.logger import logger

class WebDocumentScraper:
    """
    Crawls and parses specified URLs into LangChain Documents.
    """
    def __init__(self, urls: List[str] = None):
        self.urls = urls or [
            "https://www.dgft.gov.in/CP/?opt=dgft-ra-details",
            "https://www.dgft.gov.in/CP/?opt=dgft-hq-details"
        ]

    def load_documents(self) -> List[Document]:
        logger.info(f"Starting web document ingestion for {len(self.urls)} URLs")
        all_documents: List[Document] = []

        for url in self.urls:
            try:
                logger.info(f"Loading web page: {url}")
                loader = WebBaseLoader(url)
                docs = loader.load()
                # Filter out empty or whitespace-only documents
                valid_docs = [doc for doc in docs if doc.page_content.strip()]

                for doc in valid_docs:
                    doc.metadata['source_url'] = url
                    doc.metadata['source_type'] = 'web'
                    doc.metadata['source'] = url

                all_documents.extend(valid_docs)
                logger.info(f"Loaded {len(valid_docs)} web documents from {url}")
            except Exception as e:
                logger.error(f"Web ingestion failed for URL {url}: {e}")

        logger.info(f"Web ingestion complete. Total documents: {len(all_documents)}")
        return all_documents
