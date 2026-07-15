import re
import sys
import uuid
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import Optional
from uuid import uuid4

from sympy import content
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os
bm25_retriever = None
bm25 = None
bm25_documents = []
load_dotenv()
KNOWLEDGE_FILE = Path("knowledge.txt")
STATIC_DIR = Path("static/uploads")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "knowledge_txt_rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NO_CONTEXT_MESSAGE = "File ma apka question k related answer nai ha." 
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
   
    
)   

app = FastAPI(title="RAG API", version="1.0.0")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
print('qdrant url: ', QDRANT_URL)
print('qdrant api key: ', QDRANT_API_KEY)
vector_store: Optional[QdrantVectorStore] = None



qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False,
    timeout=120,
)
class AskRequest(BaseModel):
    query: str  


class RagChain:
    def __init__(self, store: QdrantVectorStore):
        self.store = store

    def invoke(self, query: str) -> str:
        context_docs = retrieve_context(self.store, query)
        return generate_answer(query, context_docs)


def load_documents(file_path: Path) -> list[Document]:
    """Load: knowledge.txt ko LangChain Document object ma convert karta ha."""
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{file_path} empty ha.")
    return [Document(page_content=text, metadata={"source": str(file_path)})]


def split_documents(documents: list[Document]) -> list[Document]:
    """Split: long text ko searchable chunks ma divide karta ha."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_size = len(embeddings.embed_query("dimension check"))

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    vector_store.add_documents(
        chunks,
        ids=[str(uuid4()) for _ in chunks]
    )

    return vector_store


def retrieve_context(
    vector_store: QdrantVectorStore,
    question: str,
    k: int = 4,
    min_score: float = 0.0,
) -> list[Document]:
    """Retrieve: user question se related chunks search karta ha."""
    results = vector_store.similarity_search_with_score(question, k=k)
    return [doc for doc, score in results if score >= min_score]

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()



def generate_answer(question: str, context_docs: list[Document]) -> str:
    """Retrieved tamam chunks ko mila kar answer generate kare."""

    if not context_docs:
        print('no context found.. ')
        return NO_CONTEXT_MESSAGE

    print('generating anwser..')
    # Sab retrieved chunks combine karo
    context = ' '.join([doc.page_content for doc in context_docs])

    print("\n COMBINED CONTEXT ")
    print(context)
    print("\n===========")

    try:
        llm = ChatOpenAI(
            model="llama3.2",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_template("""
You are an information extraction assistant.

Your task is NOT to summarize the context.
Your task is NOT to repeat the context.

Your task is to extract ONLY the exact answer to the user's question.

Rules:
- Return only one sentence.
- Do not copy more than one sentence from the context.
- If the answer is a number, return only the number with its unit.
- If the answer is a name, return only the name.
- Ignore every other line in the context.
- Never reproduce the whole chunk.
- If the answer is missing, reply:
I don't know based on the provided context.
                                                  context: {context}
question: {question}
Answer: 
""")
        chain = prompt | llm

        response = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )
        print('final response: ', response.content.strip())
        return response.content.strip()
    except Exception as exc:
        print(f" Error in /ask: {exc}")
        raise HTTPException(status_code=500, detail=f"Error: {str(exc)}") from exc

def build_rag():
    global bm25_retriever

    print("Loading knowledge.txt...")
    documents = load_documents(KNOWLEDGE_FILE)

    print("Chunking text...")
    chunks = split_documents(documents)

    print("Embedding and storing in Qdrant...")
    store = create_vector_store(chunks)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    build_bm25(chunks)

    print(f"Ready. {len(chunks)} chunks store ho gaye.\n")

    return store
def build_bm25(chunks):
    global bm25, bm25_documents

    bm25_documents = chunks

    tokenized = [
        doc.page_content.lower().split()
        for doc in chunks
    ]
    bm25 = BM25Okapi(tokenized)

def bm25_search(question, k=4):
    global bm25
    if bm25 is None:
        return[]

    query = question.lower().split()

    scores = bm25.get_scores(query)

    top = sorted(
        zip(scores, bm25_documents),
        key=lambda x: x[0],
        reverse=True
    )[:k]

    return [doc for score, doc in top]

def get_or_create_vector_store():
    global vector_store

    if vector_store is None:
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )

    return vector_store


def process_and_index_file(content: bytes, text: str, filename: str) -> int:

    print(f"\nLoading file: {filename}...")

    document = Document(
        page_content=text,
        metadata={
            "source": filename,
            "file_size": len(content),
            "file_length": len(text),
            "word_count": len(text.split())
        }
    )

    print("Chunking text...")
    chunks = split_documents([document])

    print(f"Total chunks created: {len(chunks)}")

    print("Embedding and storing in Qdrant...")

    store = get_or_create_vector_store()

    store.add_documents(
        chunks,
        ids=[str(uuid4()) for _ in chunks]
    )

    print("Building BM25 index...")
    build_bm25(chunks)

    print(f"Ready. {len(chunks)} chunks stored successfully.\n")

    return len(chunks)
def hybrid_search(store, question, k=4):

    dense = retrieve_context(store, question, k)
    print("Dense:",len(dense))

    sparse = bm25_search(question, k)
    print("BM25:", len(sparse))

    final = []

    seen = set()

    for doc in dense + sparse:

        if doc.page_content not in seen:

            final.append(doc)

            seen.add(doc.page_content)

    return final


def get_rag_chain() -> RagChain:
    return RagChain(get_or_create_vector_store())


@app.get("/knowledge_files")
def debug():

    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return {"message": "Collection is empty"}

    return points[0].payload


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename or not str(file.filename).lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    content = await file.read()

    # Original filename aur extension
    original_name = Path(file.filename).stem
    extension = Path(file.filename).suffix

    # Unique filename
    unique_filename = f"{original_name}_{uuid.uuid4().hex}{extension}"

    # Save path
    file_path = STATIC_DIR / unique_filename

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Only UTF-8 text files supported")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File content is empty")

    # Process using unique filename
    chunks = process_and_index_file(text, content, unique_filename)

    return {
        "status": "success",
        "file": unique_filename,
        "message": "File uploaded aur Qdrant mein store ho gaya!",
        "chunks_created": chunks,
    }


@app.post("/ask")
async def ask_question(payload: AskRequest):
    print("Question from Swagger:")
    print(payload.query)

    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        chain = get_rag_chain()
        context_docs = hybrid_search(
    chain.store,
    payload.query.strip()
)

        print("\n===== Retrieved Chunks =====")
        for i, doc in enumerate(context_docs, 1):
            print(f"\nChunk {i}")
            print(doc.page_content)
            print("-" * 60)

        answer = generate_answer(
            payload.query.strip(),
            context_docs
        )

        return {
            "query": payload.query,
            "answer": answer,
            "status": "success",
        }

    except Exception as exc:
        print(f"Error in /ask: {exc}")
        raise HTTPException(status_code=500, detail=f"Error: {str(exc)}")

def main():
    vector_store = build_rag()

    while True:
        question = input(
            "Apna question likhain (exit likh kar band karain): "
        ).strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Allah Hafiz!")
            break

        context_docs = hybrid_search(vector_store, question)

        answer = generate_answer(question, context_docs)

        print("\nAnswer:")
        print(answer)
        print()
    while True:
        question = input("Apna question likhain (exit likh kar band karain): ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Allah Hafiz!")
            break

        context_docs = retrieve_context(vector_store, question)
        answer = generate_answer(question, context_docs)
        print("\nAnswer:")
        print(answer)
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "api":
        print(" Server starting on http://127.0.0.1:8004")
        uvicorn.run(app, host="127.0.0.1", port=8004)
    else:
        print("Run with: python main.py api")
