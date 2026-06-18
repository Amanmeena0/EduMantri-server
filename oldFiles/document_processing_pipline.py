import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Ensure GOOGLE_API_KEY is set
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_KEY:
    raise EnvironmentError("Missing Google API Key. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env")
os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY

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
        """
        Chunk a LangChain Document into smaller semantic chunks
        """
        text = document.page_content
        source_metadata = document.metadata
        
        # Use semantic chunking strategy
        chunks = self._semantic_chunk(text, source_metadata.get('source', 'unknown'))
        
        # Convert chunks to LangChain Documents
        langchain_docs = []
        for chunk in chunks:
            # Merge original metadata with chunk metadata
            combined_metadata = {**source_metadata, **chunk.metadata}
            
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

def load_documents() -> List[Document]:
    """Load all documents from specified directories and web URLs"""
    
    # Load PDF and text files from specified directories
    base_dirs = [
        "./dat/FAQ",
        "./data/Hand_of_Procedure", 
        "./data/Foregin_Trade_Policy",
        "./data/Txt_files/"
    ]
    
    all_documents = []
    
    # Collect and load PDF/text files
    for folder in base_dirs:
        for file_path in Path(folder).rglob("*"):
            if file_path.suffix.lower() in ['.pdf', '.txt']:
                try:
                    loader = PyPDFLoader(str(file_path)) if file_path.suffix == ".pdf" else TextLoader(str(file_path))
                    docs = loader.load()
                    
                    # Add file path to metadata for better tracking
                    for doc in docs:
                        doc.metadata['file_path'] = str(file_path)
                        doc.metadata['file_type'] = file_path.suffix
                        
                    all_documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
    
    # Load web content
    web_urls = [
        "https://www.dgft.gov.in/CP/?opt=dgft-ra-details",
        "https://www.dgft.gov.in/CP/?opt=dgft-hq-details"
    ]
    
    for url in web_urls:
        try:
            web_loader = WebBaseLoader(url)   
            web_docs = web_loader.load()
            web_docs = [doc for doc in web_docs if doc.page_content.strip()]
            
            # Add URL to metadata
            for doc in web_docs:
                doc.metadata['source_url'] = url
                doc.metadata['source_type'] = 'web'
                
            all_documents.extend(web_docs)
            logger.info(f"Loaded {len(web_docs)} web documents from {url}")
        except Exception as e:
            logger.warning(f"Web load failed for {url}: {e}")
    
    return all_documents

def process_documents_with_chunking(documents: List[Document]) -> List[Document]:
    """Process documents with semantic chunking"""
    
    chunker = SemanticChunker(chunk_size=800, overlap=100)
    
    all_chunks = []
    for doc in documents:
        try:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from document: {doc.metadata.get('source', 'unknown')}")
        except Exception as e:
            logger.warning(f"Failed to chunk document {doc.metadata.get('source', 'unknown')}: {e}")
    
    return all_chunks

def create_vector_store(documents: List[Document]) -> Chroma:
    """Create and persist vector store from documents"""
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,  
        embedding=embeddings,
        persist_directory="./vahei_db",
        collection_name="dgft_collection"
    )
    
    # Persist the vector store
    vectorstore.persist()
    logger.info(f"Created vector store with {len(documents)} document chunks")
    
    return vectorstore

def main():
    """Main pipeline execution"""
    logger.info("Starting document processing pipeline...")
    
    # Load all documents
    logger.info("Loading documents...")
    documents = load_documents()
    logger.info(f"Loaded {len(documents)} total documents")
    
    # Process documents with semantic chunking
    logger.info("Processing documents with semantic chunking...")
    chunked_documents = process_documents_with_chunking(documents)
    logger.info(f"Created {len(chunked_documents)} total chunks")
    
    # Create and persist vector storeaa
    logger.info("Creating vector store...")
    vectorstore = create_vector_store(chunked_documents)
    
    logger.info("Pipeline completed successfully!")
    
    # Optional: Save sample for inspection
    if chunked_documents:
        with open("sample_chunks.txt", "w", encoding="utf-8") as f:
            for i, doc in enumerate(chunked_documents[:-5]):  # Save first 5 chunks
                f.write(f"=== CHUNK {i+1} ===\n")
                f.write(f"Content: {doc.page_content}\n")
                f.write(f"Metadata: {doc.metadata}\n\n")

if __name__ == "__main__":
    main()