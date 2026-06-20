# src/infrastructure/mcp/docs.py
import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.core.logger import logger

class LocalDocumentLoader:
    """
    Ingests and loads PDF and TXT documents from designated local directories.
    """
    def __init__(self, base_dirs: List[str] = None):
        self.base_dirs = base_dirs or [
            "./data/FAQ",
            "./data/Hand_Of_Procedure",
            "./data/Foregin_Trade_Policy",
            "./data/Txt_files/"
        ]

    def load_documents(self) -> List[Document]:
        logger.info(f"Starting local document loading from {len(self.base_dirs)} directories")
        all_documents: List[Document] = []

        for folder in self.base_dirs:
            path = Path(folder)
            if not path.exists():
                logger.warning(f"Ingestion directory not found (skipping): {folder}")
                continue

            logger.info(f"Scanning folder: {folder}")
            # Scan files inside directories
            for file_path in path.rglob("*"):
                if file_path.suffix.lower() in ['.pdf', '.txt']:
                    try:
                        if file_path.suffix.lower() == ".pdf":
                            loader = PyPDFLoader(str(file_path))
                        else:
                            loader = TextLoader(str(file_path), encoding='utf-8')

                        docs = loader.load()

                        for doc in docs:
                            doc.metadata["file_path"] = str(file_path)
                            doc.metadata["file_type"] = file_path.suffix.lower()
                            doc.metadata["source"] = file_path.name

                        all_documents.extend(docs)
                        logger.info(f"Loaded {len(docs)} chunks from {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to parse file {file_path}: {e}")

        logger.info(f"Local ingestion complete. Loaded {len(all_documents)} total pages/documents.")
        return all_documents
