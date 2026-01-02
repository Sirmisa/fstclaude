"""Document indexer using LlamaIndex and ChromaDB."""

import os
from typing import List, Optional

import chromadb
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.chroma import ChromaVectorStore


class DocumentIndexer:
    """Indexes documents using LlamaIndex and stores vectors in ChromaDB."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
    ):
        """Initialize the document indexer.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the ChromaDB data.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

        # Initialize ChromaDB vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        self.index: Optional[VectorStoreIndex] = None

    def index_documents(self, documents_path: str) -> VectorStoreIndex:
        """Index documents from a directory.

        Args:
            documents_path: Path to the directory containing documents.

        Returns:
            VectorStoreIndex: The created index.
        """
        if not os.path.exists(documents_path):
            raise ValueError(f"Documents path does not exist: {documents_path}")

        # Load documents using LlamaIndex SimpleDirectoryReader
        documents = SimpleDirectoryReader(documents_path).load_data()

        # Create index
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
        )

        return self.index

    def index_text(self, texts: List[str], metadata: Optional[List[dict]] = None) -> VectorStoreIndex:
        """Index a list of text strings.

        Args:
            texts: List of text strings to index.
            metadata: Optional list of metadata dicts for each text.

        Returns:
            VectorStoreIndex: The created index.
        """
        documents = []
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            documents.append(Document(text=text, metadata=meta))

        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
        )

        return self.index

    def load_index(self) -> VectorStoreIndex:
        """Load an existing index from ChromaDB.

        Returns:
            VectorStoreIndex: The loaded index.
        """
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
        )
        return self.index

    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the current index.

        Returns:
            The current VectorStoreIndex or None if not initialized.
        """
        return self.index
