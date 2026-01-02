"""RAG Retriever using ChromaDB."""

from typing import List, Optional

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore


class RAGRetriever:
    """Retrieves relevant documents using ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        top_k: int = 3,
    ):
        """Initialize the retriever.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory where ChromaDB data is persisted.
            top_k: Number of top results to retrieve.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.top_k = top_k

        # Initialize ChromaDB client and collection
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

        # Initialize vector store and index
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

        # Initialize retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )

    def retrieve(self, query: str) -> List[dict]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query string.

        Returns:
            List of retrieved documents with their content and metadata.
        """
        nodes = self.retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "content": node.node.get_content(),
                "score": node.score,
                "metadata": node.node.metadata,
            })

        return results

    def get_context(self, query: str, max_length: Optional[int] = None) -> str:
        """Get combined context from retrieved documents.

        Args:
            query: The query string.
            max_length: Optional maximum length for the combined context.

        Returns:
            Combined context string from retrieved documents.
        """
        results = self.retrieve(query)

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i}]\n{result['content']}")

        context = "\n\n".join(context_parts)

        if max_length and len(context) > max_length:
            context = context[:max_length] + "..."

        return context
