# 🐘 Elephant-Mind (pg-brain)

Elephant-Mind is a lightweight **RAG (Retrieval-Augmented Generation)** system designed to ingest PDF documents and provide accurate answers using a local LLM and vector database. It leverages **PostgreSQL** with the `pgvector` extension for efficient semantic search and **Ollama** for embeddings and text generation.

## 🚀 Features

- **PDF Ingestion**: Automatically extracts text from PDFs and chunks it with overlap for better context preservation.
- **Vector Search**: Uses `pgvector` to store and query high-dimensional embeddings.
- **Local-First**: Runs entirely on your local machine using Docker and Ollama.
- **RAG Pipeline**: Combines document retrieval with LLM generation to provide context-aware answers.

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

- **Docker & Docker Compose**: To run the PostgreSQL database.
- **Ollama**: For local LLM and embedding models.
- **Python 3.10+**: To run the ingestion and query scripts.

## ⚙️ Local Setup & Configuration

### 1. Database Setup
Start the PostgreSQL container with `pgvector`:

```bash
docker compose -f docker/docker-compose.yml up -d
```

This will start a Postgres instance on `localhost:5432` with:
- **User**: `admin`
- **Password**: `secret`
- **Database**: `vectordb`

### 2. Ollama Configuration
Ensure Ollama is running and pull the necessary models:

```bash
# Pull the embedding model
ollama pull nomic-embed-text

# Pull the LLM model
ollama pull llama3.2
```

### 3. Python Environment
Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configuration
The system uses environment variables for configuration. Create a `.env` file in the root directory (a template is provided below):

```env
# Database Configuration
POSTGRES_USER=admin
POSTGRES_PASSWORD=secret
POSTGRES_DB=vectordb
DB_HOST=localhost
DB_PORT=5432

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3.2
```

The Python scripts (`src/ingest.py` and `src/query.py`) and `docker-compose.yml` will automatically load these values.

## 📖 Usage

### 📥 Ingesting Documents
Place your PDF files in the `data/pdf/` directory.

To run the ingestion:
```bash
python src/ingest.py
```
This script will:
1. Extract text from the PDF.
2. Generate embeddings using `nomic-embed-text`.
3. Save the chunks and vectors into the `document_embeddings` table.

### 🔍 Querying the System
Start the interactive query session:

```bash
python src/query.py
```
You can then ask questions based on the ingested documents. Type `exit` to quit.
