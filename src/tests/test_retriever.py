# src/tests/test_retriever.py
import os
# Set dummy API key before imports to prevent configuration failures
os.environ["GOOGLE_API_KEY"] = "mock_key_for_testing"

import unittest
from unittest.mock import MagicMock, patch
from src.ai.retrievers.faiss_retriever import FAISSRetrieverManager

class TestFAISSRetriever(unittest.TestCase):
    @patch("src.ai.retrievers.faiss_retriever.get_embeddings")
    @patch("src.ai.retrievers.faiss_retriever.FAISS")
    @patch("os.path.exists")
    def test_initialization_no_crash(self, mock_exists, mock_faiss, mock_get_embeddings):
        """Verifies FAISSRetrieverManager initializes without the legacy infinite recursion crashes."""
        mock_exists.return_value = True
        mock_get_embeddings.return_value = MagicMock()
        mock_vector_store = MagicMock()
        mock_faiss.load_local.return_value = mock_vector_store
        
        # Instantiate retriever manager
        manager = FAISSRetrieverManager(persist_directory="/tmp/fake_faiss")
        
        self.assertIsNotNone(manager)
        mock_faiss.load_local.assert_called_once()
        self.assertIsNotNone(manager.retriever)

if __name__ == "__main__":
    unittest.main()
