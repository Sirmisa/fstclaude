"""Gradio web interface for the RAG system."""

import os
from typing import List

import gradio as gr
from dotenv import load_dotenv

from rag import DocumentIndexer, RAGChain, RAGRetriever

# Load environment variables
load_dotenv()

# Global instances
indexer = None
rag_chain = None

COLLECTION_NAME = "rag_documents"
PERSIST_DIR = "./chroma_db"


def initialize_system():
    """Initialize the RAG system."""
    global indexer, rag_chain

    indexer = DocumentIndexer(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    # Try to load existing index
    try:
        indexer.load_index()
        rag_chain = RAGChain(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )
        return "RAG system initialized with existing index."
    except Exception:
        return "RAG system initialized. Please upload documents to create an index."


def upload_documents(files) -> str:
    """Handle document upload and indexing.

    Args:
        files: List of uploaded files.

    Returns:
        Status message.
    """
    global indexer, rag_chain

    if not files:
        return "No files uploaded."

    try:
        # Save uploaded files temporarily
        texts = []
        metadata = []

        for file in files:
            with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                texts.append(content)
                metadata.append({"filename": os.path.basename(file.name)})

        # Index the documents
        if indexer is None:
            indexer = DocumentIndexer(
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR,
            )

        indexer.index_text(texts, metadata)

        # Reinitialize the RAG chain
        rag_chain = RAGChain(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )

        return f"Successfully indexed {len(files)} document(s)."

    except Exception as e:
        return f"Error indexing documents: {str(e)}"


def add_text_document(text: str, doc_name: str) -> str:
    """Add a text document directly.

    Args:
        text: Document text content.
        doc_name: Name for the document.

    Returns:
        Status message.
    """
    global indexer, rag_chain

    if not text.strip():
        return "Please enter some text."

    try:
        if indexer is None:
            indexer = DocumentIndexer(
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR,
            )

        indexer.index_text(
            [text],
            [{"filename": doc_name or "manual_input"}]
        )

        # Reinitialize the RAG chain
        rag_chain = RAGChain(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )

        return "Document added successfully."

    except Exception as e:
        return f"Error adding document: {str(e)}"


def query_rag(
    question: str,
    chat_history: List[dict],
    show_sources: bool
) -> tuple[List[dict], str]:
    """Query the RAG system.

    Args:
        question: User question.
        chat_history: Current chat history.
        show_sources: Whether to show source documents.

    Returns:
        Updated chat history and sources text.
    """
    global rag_chain

    if not question.strip():
        return chat_history, ""

    if rag_chain is None:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": "Please upload documents first to initialize the RAG system."})
        return chat_history, ""

    try:
        if show_sources:
            result = rag_chain.query_with_sources(question)
            answer = result["answer"]
            sources_text = "\n\n".join([
                f"**Source {i+1}** (score: {s['score']:.3f}):\n{s['content'][:200]}..."
                for i, s in enumerate(result["sources"])
            ])
        else:
            answer = rag_chain.query(question)
            sources_text = ""

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        return chat_history, sources_text

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history, ""


def clear_chat() -> tuple[List, str]:
    """Clear the chat history."""
    return [], ""


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(title="RAG System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# RAG System")
        gr.Markdown("A simple RAG system using ChromaDB, LlamaIndex, LangChain, and Gradio.")

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                label="Chat",
                height=400,
                type="messages",
            )

            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about your documents...",
                    scale=4,
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)

            with gr.Row():
                show_sources = gr.Checkbox(label="Show Sources", value=True)
                clear_btn = gr.Button("Clear Chat")

            sources_output = gr.Markdown(label="Retrieved Sources")

            # Event handlers
            submit_btn.click(
                fn=query_rag,
                inputs=[question_input, chatbot, show_sources],
                outputs=[chatbot, sources_output],
            ).then(
                fn=lambda: "",
                outputs=question_input,
            )

            question_input.submit(
                fn=query_rag,
                inputs=[question_input, chatbot, show_sources],
                outputs=[chatbot, sources_output],
            ).then(
                fn=lambda: "",
                outputs=question_input,
            )

            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot, sources_output],
            )

        with gr.Tab("Upload Documents"):
            gr.Markdown("### Upload Files")
            file_upload = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".txt", ".md", ".pdf"],
            )
            upload_btn = gr.Button("Index Documents", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)

            upload_btn.click(
                fn=upload_documents,
                inputs=file_upload,
                outputs=upload_status,
            )

            gr.Markdown("### Or Add Text Directly")
            with gr.Row():
                doc_name_input = gr.Textbox(
                    label="Document Name",
                    placeholder="my_document",
                    scale=1,
                )
                text_input = gr.Textbox(
                    label="Document Content",
                    placeholder="Paste your document text here...",
                    lines=5,
                    scale=3,
                )

            add_text_btn = gr.Button("Add Document")
            add_text_status = gr.Textbox(label="Status", interactive=False)

            add_text_btn.click(
                fn=add_text_document,
                inputs=[text_input, doc_name_input],
                outputs=add_text_status,
            )

        with gr.Tab("Settings"):
            gr.Markdown("### Configuration")
            gr.Markdown("""
            **Environment Variables Required:**
            - `OPENAI_API_KEY`: Your OpenAI API key

            **Default Settings:**
            - Collection Name: `rag_documents`
            - Vector Store: ChromaDB (persistent)
            - LLM Model: `gpt-3.5-turbo`
            """)

            init_status = gr.Textbox(label="System Status", interactive=False)
            init_btn = gr.Button("Reinitialize System")

            init_btn.click(
                fn=initialize_system,
                outputs=init_status,
            )

        # Initialize on load
        demo.load(fn=initialize_system, outputs=[])

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
