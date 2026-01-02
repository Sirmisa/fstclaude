# RAG System

[![CI](https://github.com/Sirmisa/fstclaude/actions/workflows/ci.yml/badge.svg)](https://github.com/Sirmisa/fstclaude/actions/workflows/ci.yml)

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
├── .github/workflows/    # CI/CD pipelines
│   └── ci.yml
├── app.py                # Gradio web interface
├── requirements.txt      # Python dependencies
├── requirements-dev.txt  # Development dependencies
├── pyproject.toml        # Project configuration
├── .env.example          # Environment variables template
├── documents/            # Sample documents
│   └── sample.txt
├── tests/                # Test suite
│   └── test_rag.py
└── rag/                  # RAG module
    ├── __init__.py
    ├── indexer.py        # Document indexing with LlamaIndex
    ├── retriever.py      # Document retrieval with ChromaDB
    └── chain.py          # LangChain integration
```

## CI/CD

This project uses GitHub Actions for continuous integration. The pipeline runs on every push and pull request to main/master branches.

### Pipeline Jobs

1. **Test**: Runs the test suite with pytest and generates coverage reports
2. **Lint**: Checks code quality with ruff
3. **Build**: Verifies that all imports work correctly

### Running Tests Locally

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=rag

# Run linter
ruff check .
```

### Required Secrets

Add the following secrets to your GitHub repository for full CI functionality:
- `OPENAI_API_KEY`: Your OpenAI API key (required for integration tests)

## Technologies

- **ChromaDB**: Vector database for storing and retrieving embeddings
- **LlamaIndex**: Framework for document indexing and retrieval
- **LangChain**: LLM orchestration and chain building
- **Gradio**: Web interface framework

## License

MIT
