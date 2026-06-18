import os 
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_core.documents import Document
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class Chunk:
    """Represents a chunk of text with metadata"""


    content: str
    metadata: Dict[str, Any]
    start_index: int
    end_index: int
    chunk_id: str

class SemanticChunker:
    """
        Token-aware chunking implementation optimized for document processing
    """
    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 100,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0:
            raise ValueError("overlap must be greater than or equal to 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk_document(self, document: Document) -> List[Document]:
        """Chunk a LangChain Document into smaller semantic chunks"""

        text = document.page_content
        source_metadata = document.metadata

        chunks = self._semantic_chunk(text, source_metadata.get('source','unknown'))

        langchain_docs = []
        for chunk in chunks:
            combined_metadata = {**source_metadata,**chunk.metadata}

            doc = Document(
                page_content=chunk.content,
                metadata=combined_metadata
            )
            langchain_docs.append(doc)

        return langchain_docs

    def _semantic_chunk(self, text: str, source_id: str) -> List[Chunk]:
        """Chunk text using token windows with token overlap."""
        if not text.strip():
            return []

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        token_ids = encoded.get("input_ids", [])
        offsets = encoded.get("offset_mapping", [])
        if not token_ids:
            return []

        chunks = []
        step_size = self.chunk_size - self.overlap

        for start_tok in range(0, len(token_ids), step_size):
            end_tok = min(start_tok + self.chunk_size, len(token_ids))
            chunk_token_ids = token_ids[start_tok:end_tok]
            content = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True).strip()
            if not content:
                if end_tok == len(token_ids):
                    break
                continue

            start_char = offsets[start_tok][0]
            end_char = offsets[end_tok - 1][1]

            chunk = self._create_chunk(
                content,
                start_char,
                end_char,
                source_id,
                len(chunks),
                start_tok,
                end_tok
            )
            chunks.append(chunk)

            if end_tok == len(token_ids):
                break

        return chunks

    def _create_chunk(
        self,
        content: str,
        start: int,
        end: int,
        source_id: str,
        chunk_idx: int,
        token_start: int,
        token_end: int
    ) -> Chunk:
        """Create a chunk with metadata"""

        metadata = {
            "source_id": source_id,
            "chunk_index": chunk_idx,
            "char_count": len(content),
            "word_count": len(content.split()),
            "token_count": token_end - token_start,
            "token_start": token_start,
            "token_end": token_end,
            "chunk_start": start,
            "chunk_end": end
        }
            
        return Chunk(
            content=content,
            metadata=metadata,
            start_index=start,
            end_index=end,
            chunk_id=f"{source_id}_chunk_{chunk_idx}"
        )
    

def process_documents_with_chunking(documents: List[Document]) -> List[Document]:
    """Process documents with semantic chunking"""
    
    chunker = SemanticChunker(chunk_size=400, overlap=70)
    
    all_chunks = []
    for doc in documents:
        try:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from document: {doc.metadata.get('source', 'unknown')}")
        except Exception as e:
            logger.warning(f"Failed to chunk document {doc.metadata.get('source', 'unknown')}: {e}")
    
    return all_chunks


