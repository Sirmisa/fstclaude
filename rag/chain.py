"""RAG Chain using LangChain for LLM integration."""

import os
from typing import Optional

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI

from .retriever import RAGRetriever


class RAGChain:
    """RAG Chain combining retrieval with LLM generation using LangChain."""

    DEFAULT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question.
If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
    ):
        """Initialize the RAG chain.

        Args:
            retriever: Optional RAGRetriever instance. If not provided, one will be created.
            model_name: Name of the OpenAI model to use.
            temperature: Temperature for the LLM.
            collection_name: Name of the ChromaDB collection (used if retriever not provided).
            persist_directory: ChromaDB persist directory (used if retriever not provided).
        """
        self.retriever = retriever or RAGRetriever(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(self.DEFAULT_TEMPLATE)

        # Build the chain
        self.chain = self._build_chain()

    def _build_chain(self):
        """Build the RAG chain using LangChain LCEL."""

        def get_context(question: str) -> str:
            return self.retriever.get_context(question)

        chain = (
            {"context": get_context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str) -> str:
        """Query the RAG system.

        Args:
            question: The question to answer.

        Returns:
            The generated answer.
        """
        return self.chain.invoke(question)

    def query_with_sources(self, question: str) -> dict:
        """Query the RAG system and return answer with sources.

        Args:
            question: The question to answer.

        Returns:
            Dict containing the answer and source documents.
        """
        # Get retrieved documents
        sources = self.retriever.retrieve(question)

        # Generate answer
        answer = self.query(question)

        return {
            "answer": answer,
            "sources": sources,
        }
