"""Tests for the RAG system components."""

import os
import tempfile
import shutil
from unittest.mock import Mock, patch

import pytest

from rag import DocumentIndexer, RAGRetriever, RAGChain


class TestDocumentIndexer:
    """Tests for DocumentIndexer."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def indexer(self, temp_dir):
        """Create a DocumentIndexer instance."""
        return DocumentIndexer(
            collection_name="test_collection",
            persist_directory=os.path.join(temp_dir, "chroma_db"),
        )

    def test_indexer_initialization(self, indexer):
        """Test that indexer initializes correctly."""
        assert indexer.collection_name == "test_collection"
        assert indexer.index is None

    def test_index_text(self, indexer):
        """Test indexing text documents."""
        texts = ["This is a test document.", "Another test document."]
        metadata = [{"source": "test1"}, {"source": "test2"}]

        index = indexer.index_text(texts, metadata)

        assert index is not None
        assert indexer.get_index() is not None

    def test_index_text_without_metadata(self, indexer):
        """Test indexing text without metadata."""
        texts = ["Document without metadata."]

        index = indexer.index_text(texts)

        assert index is not None

    def test_load_index(self, indexer):
        """Test loading an existing index."""
        # First create an index
        indexer.index_text(["Test document for loading."])

        # Create a new indexer pointing to the same location
        new_indexer = DocumentIndexer(
            collection_name=indexer.collection_name,
            persist_directory=indexer.persist_directory,
        )

        # Load the index
        loaded_index = new_indexer.load_index()

        assert loaded_index is not None


class TestRAGRetriever:
    """Tests for RAGRetriever."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def retriever_with_docs(self, temp_dir):
        """Create a retriever with indexed documents."""
        persist_dir = os.path.join(temp_dir, "chroma_db")

        # First index some documents
        indexer = DocumentIndexer(
            collection_name="test_retriever",
            persist_directory=persist_dir,
        )
        indexer.index_text([
            "Python is a programming language.",
            "Machine learning uses algorithms to learn from data.",
            "RAG combines retrieval with generation.",
        ])

        # Create retriever
        return RAGRetriever(
            collection_name="test_retriever",
            persist_directory=persist_dir,
            top_k=2,
        )

    def test_retriever_initialization(self, retriever_with_docs):
        """Test that retriever initializes correctly."""
        assert retriever_with_docs.top_k == 2

    def test_retrieve(self, retriever_with_docs):
        """Test retrieving documents."""
        results = retriever_with_docs.retrieve("What is Python?")

        assert len(results) > 0
        assert "content" in results[0]
        assert "score" in results[0]

    def test_get_context(self, retriever_with_docs):
        """Test getting combined context."""
        context = retriever_with_docs.get_context("machine learning")

        assert len(context) > 0
        assert "[Document" in context

    def test_get_context_with_max_length(self, retriever_with_docs):
        """Test context truncation."""
        context = retriever_with_docs.get_context("machine learning", max_length=50)

        assert len(context) <= 53  # 50 + "..."


class TestRAGChain:
    """Tests for RAGChain."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = Mock(spec=RAGRetriever)
        retriever.retrieve.return_value = [
            {"content": "Test content", "score": 0.9, "metadata": {}},
        ]
        retriever.get_context.return_value = "[Document 1]\nTest content"
        return retriever

    @patch("rag.chain.ChatOpenAI")
    def test_chain_initialization(self, mock_llm, mock_retriever):
        """Test that chain initializes correctly."""
        mock_llm.return_value = Mock()

        chain = RAGChain(retriever=mock_retriever)

        assert chain.retriever == mock_retriever

    @patch("rag.chain.ChatOpenAI")
    def test_query(self, mock_llm, mock_retriever):
        """Test querying the chain."""
        # Setup mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.__or__ = Mock(return_value=mock_llm_instance)
        mock_llm.return_value = mock_llm_instance

        chain = RAGChain(retriever=mock_retriever)
        chain.chain = Mock()
        chain.chain.invoke.return_value = "This is a test answer."

        answer = chain.query("What is this about?")

        assert answer == "This is a test answer."

    @patch("rag.chain.ChatOpenAI")
    def test_query_with_sources(self, mock_llm, mock_retriever):
        """Test querying with sources."""
        mock_llm_instance = Mock()
        mock_llm_instance.__or__ = Mock(return_value=mock_llm_instance)
        mock_llm.return_value = mock_llm_instance

        chain = RAGChain(retriever=mock_retriever)
        chain.chain = Mock()
        chain.chain.invoke.return_value = "Test answer."

        result = chain.query_with_sources("What is this?")

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "Test answer."
