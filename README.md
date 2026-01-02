# RAG System

A simple Retrieval-Augmented Generation (RAG) system using ChromaDB, LlamaIndex, LangChain, and Gradio.

## Features

- **Document Indexing**: Upload and index documents using LlamaIndex
- **Vector Storage**: Store embeddings in ChromaDB for efficient retrieval
- **LLM Integration**: Generate answers using LangChain with OpenAI models
- **Web Interface**: Interactive chat interface built with Gradio

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fstclaude
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:7860`

3. Upload documents in the "Upload Documents" tab

4. Ask questions in the "Chat" tab

## Project Structure

```
fstclaude/
├── app.py                 # Gradio web interface
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── documents/            # Sample documents
│   └── sample.txt
└── rag/                  # RAG module
    ├── __init__.py
    ├── indexer.py        # Document indexing with LlamaIndex
    ├── retriever.py      # Document retrieval with ChromaDB
    └── chain.py          # LangChain integration
```

## Technologies

- **ChromaDB**: Vector database for storing and retrieving embeddings
- **LlamaIndex**: Framework for document indexing and retrieval
- **LangChain**: LLM orchestration and chain building
- **Gradio**: Web interface framework

## License

MIT
