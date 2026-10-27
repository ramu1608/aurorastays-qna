
# Import necessary libraries
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import RAG utility functions
from rag_utils import retrieve, rag


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"


# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")


# ---------------------------------------------------------
# Load the embedding model
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ---------------------------------------------------------
# Load the saved FAISS vector store
# ---------------------------------------------------------

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Vectors stored: {vectorstore.index.ntotal}")


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """
    Handles GET requests to the root URL.
    """
    return "Welcome to the Aurora Stay RAG API!"


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """
    Handles POST requests to /v1/relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 3
    }

    Returns the top-k relevant chunks retrieved from
    the vector store.
    """

    # Get JSON data from request body
    data = request.get_json()

    # Extract parameters
    query = data["query"]
    k = data.get("k", 2)

    # Retrieve relevant chunks
    chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore
    )

    # Return the retrieved chunks
    return jsonify({
        "query": query,
        "k": k,
        "relevant_chunks": chunks
    })


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """
    Handles POST requests to /v1/answer_with_relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 3,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns both the generated answer and the relevant
    chunks used to generate the answer.
    """

    # Get JSON data from request body
    data = request.get_json()

    # Extract parameters
    query = data["query"]
    k = data.get("k", 2)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)

    # Run the complete RAG pipeline
    answer, chunks = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    # Return answer, chunks, and parameters used
    return jsonify({
        "query": query,
        "k": k,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "relevant_chunks": chunks,
        "answer": answer
    })


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """
    Handles POST requests to /v1/answer.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 3,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns only the generated answer.
    """

    # Get JSON data from request body
    data = request.get_json()

    # Extract parameters
    query = data["query"]
    k = data.get("k", 2)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)

    # Run the complete RAG pipeline
    generated_answer, _ = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    # Return only the answer
    return jsonify({
        "answer": generated_answer
    })


# ---------------------------------------------------------
# Run the Flask application
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_api.run(debug=True)
