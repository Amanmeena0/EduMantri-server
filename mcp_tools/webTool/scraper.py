import os 
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
import logging
from typing import List
from langchain_core.documents import Document

# bs4 and requests 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebDocument:
    def web_documents() -> List[Document]:
        web_urls = [
            "https://www.dgft.gov.in/CP/?opt=dgft-ra-details",
            "https://www.dgft.gov.in/CP/?opt=dgft-hq-details"
        ]
        logger.info("Starting web document ingestion for %d URLs", len(web_urls))
        all_documents: List[Document] = []

        for url in web_urls:
            try:
                logger.info("Loading URL: %s", url)
                web_loader = WebBaseLoader(url)
                web_Docs = web_loader.load()
                web_Docs = [doc for doc in web_Docs if doc.page_content.strip()]


                for doc in web_Docs:
                    doc.metadata['source_url'] = url
                    doc.metadata['source_type'] = 'web'

                all_documents.extend(web_Docs)
                logger.info("Loaded %d web documents from %s", len(web_Docs), url)
            except Exception as e:
                logger.exception("Web load failed for %s: %s", url, e)

        logger.info("Web ingestion complete. Total documents loaded: %d", len(all_documents))
        return all_documents
