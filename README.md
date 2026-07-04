# Pure CLI RAG Chatbot

A minimal, high-utility end-to-end Retrieval-Augmented Generation (RAG) pipeline running completely inside the terminal. This project ingests a custom text knowledge base, splits it into overlapping semantic chunks, converts the text into numeric vectors using OpenAI, and stores them in a local ChromaDB database. When a user asks a question, the backend queries the database for context and passes it to an LLM to generate an accurate, grounded answer.

Built as a hands-on architectural deep dive to master the underlying mechanics of document ingestion, semantic search vector math, and LLM context management.

---

## Tech Stack & Core Components

- **Orchestration Framework:** LangChain (`langchain-community`, `langchain-openai`)
- **Vector Database:** ChromaDB (Running completely locally on disk)
- **Embedding Model:** OpenAI `text-embedding-3-small` (1,536-dimensional semantic spaces)
- **Large Language Model:** OpenAI `gpt-4o-mini` (Factually locked with `temperature=0`)
- **Interface:** Pure CLI (Command Line Interface)

---

## Project Architecture

```text
basicRag/
│
├── app.py              # CLI Application: Handles chunking, embedding, database storage, and queries
├── .env                # Local Environment Variables (Contains OpenAI API Keys - Hidden)
├── .gitignore          # Safeguard file protecting sensitive folders and API keys
└── venv/               # Isolated Python Virtual Environment
```

---

## Getting Started (Local Setup)

Follow these steps to configure and run this application on your local machine.

### 1. Clone the Repository & Navigate In
```bash
git clone https://github.com
cd basicRag
```

### 2. Configure Your Virtual Environment
```bash
# Create the environment
python3 -m venv venv

# Activate the environment (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install langchain langchain-community langchain-openai chromadb pypdf python-dotenv
```

### 4. Setup Your Environment Variables
Create a file named `.env` in the root project directory and paste your OpenAI developer key inside:
```text
OPENAI_API_KEY=your_actual_openai_sk_api_key_here
```

---

## How to Run the Application

To run the full text processing pipeline, generate vector embeddings, and execute a semantic search query directly inside your Mac terminal:
```bash
python app.py
```

---

## Security Configuration
This repository utilizes a defensive `.gitignore` policy. Hidden configurations (`.env`), package modules (`venv/`), and local vector binaries (`chroma_db/`) are excluded from version tracking to completely protect production private keys and prevent deployment data leaks.
