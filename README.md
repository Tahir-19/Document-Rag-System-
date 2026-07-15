RAG API with FastAPI, Qdrant & Ollama

A Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, Qdrant, Hugging Face Embeddings, and Ollama (Llama 3.2). This project allows users to upload .txt files, automatically index them into Qdrant, and ask questions that are answered using only the uploaded knowledge base.

🚀 Features
Upload .txt files through FastAPI.
Automatic text chunking.
Generate embeddings using Hugging Face.
Store embeddings in Qdrant Cloud.
Semantic search using vector similarity.
BM25 keyword retrieval.
Hybrid Retrieval (BM25 + Vector Search).
Answer generation using Ollama (Llama 3.2).
Unique filename generation using UUID.
Multiple file support.
FastAPI REST API.
Swagger API documentation.
Prevent duplicate filename conflicts.
UTF-8 file validation.
Empty file validation.
🛠 Tech Stack
Python 3.x
FastAPI
LangChain
LangChain Community
Hugging Face Embeddings
sentence-transformers
Qdrant Cloud
Ollama
Llama 3.2
BM25 Retriever
Uvicorn
python-dotenv
Pydantic
📁 Project Structure
RAG-API/
│
├── static/
│   └── uploads/
│
├── .env
├── main.py
├── requirements.txt
├── README.md
└── venv/
⚙ Environment Variables

Create a .env file inside the project root.

QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
📦 Installation

Clone the repository

git clone https://github.com/yourusername/rag-api.git

Move into the project directory

cd rag-api

Create virtual environment

python -m venv venv

Activate virtual environment

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

Install dependencies

pip install -r requirements.txt
▶ Run Ollama

Start Ollama

ollama serve

Run Llama 3.2

ollama run llama3.2
▶ Run FastAPI
python main.py api

or

uvicorn main:app --reload
📄 API Endpoints
Upload File
POST /upload

Upload a UTF-8 .txt file.

Response
{
    "status":"success",
    "file":"rag_file_4fce4a6d.txt",
    "message":"File uploaded aur Qdrant mein store ho gaya!",
    "chunks_created":25
}
Ask Question
POST /ask
Request
{
    "query":"What is Artificial Intelligence?"
}
Response
{
    "answer":"Artificial Intelligence is..."
}
⚡ Workflow
Upload TXT File
        │
        ▼
UTF-8 Validation
        │
        ▼
Generate UUID Filename
        │
        ▼
Read File Content
        │
        ▼
Chunk Text
        │
        ▼
Generate Embeddings
        │
        ▼
Store in Qdrant
        │
        ▼
User Question
        │
        ▼
Hybrid Retrieval
(BM25 + Vector Search)
        │
        ▼
Top Relevant Chunks
        │
        ▼
Ollama (Llama 3.2)
        │
        ▼
Final Answer
📚 Libraries Used
fastapi
uvicorn
langchain
langchain-community
langchain-qdrant
qdrant-client
sentence-transformers
python-dotenv
pydantic
rank-bm25
ollama
✨ Implemented Features
File Upload API
Ask API
Qdrant Cloud Integration
Hugging Face Embeddings
Hybrid Retrieval
BM25 Retriever
Vector Retriever
Recursive Character Text Splitter
UUID-based Unique File Names
FastAPI Validation
Error Handling
UTF-8 Validation
Empty File Validation
Environment Variable Support (.env)
Ollama Local LLM Integration
Swagger Documentation
📌 Future Improvements
PDF support
DOCX support
Chat history
Multiple collections
Streaming responses
User authentication
Delete uploaded files
File management dashboard
Conversation memory
Citation/source references
Docker support
Deployment on Render/AWS/Azure
