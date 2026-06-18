import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import logging
from typing import List
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalDocuments:

    @staticmethod
    def load_documents() -> List[Document]:
        """Load all documents from specified directories"""

        base_dirs = [
            "./data/FAQ",
            "./data/Hand_Of_Procedure",
            "./data/Foregin_Trade_Policy",
            "./data/Txt_files/"
        ]

        logger.info("Starting local document ingestion from %d folders", len(base_dirs))
        all_documents: List[Document] = []

        for folder in base_dirs:
            path = Path(folder)

            if not path.exists():
                logger.warning("Folder not found, skipping: %s", folder)
                continue

            logger.info("Scanning folder: %s", folder)

            for file_path in path.rglob("*"):
                if file_path.suffix.lower() in ['.pdf', '.txt']:
                    try:
                        if file_path.suffix.lower() == ".pdf":
                            loader = PyPDFLoader(str(file_path))
                        else:
                            loader = TextLoader(str(file_path))

                        docs = loader.load()

                        for doc in docs:
                            doc.metadata["file_path"] = str(file_path)
                            doc.metadata["file_type"] = file_path.suffix.lower()

                        all_documents.extend(docs)

                        logger.info("Loaded %d chunks from %s", len(docs), file_path)

                    except Exception as e:
                        logger.exception("Failed to load %s: %s", file_path, e)

        logger.info("Local ingestion complete. Total documents loaded: %d", len(all_documents))

        return all_documents