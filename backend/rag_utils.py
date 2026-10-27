
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
import numpy as np
import os

# Initialize the Cross-Encoder model (lightweight and highly effective for QA)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieve the top-k relevant chunks using two-stage Cross-Encoder re-ranking
def retrieve(query: str, initial_k: int = 10, final_k: int = 3, vectorstore=None) -> list[str]:
    if vectorstore is None:
        raise ValueError("A valid vectorstore must be provided.")

    # Step 1: Broad retrieval using the FAISS index
    initial_results = vectorstore.similarity_search(query, k=initial_k)
    if not initial_results:
        return []

    # Step 2: Prepare pairs of (query, document) for the Cross-Encoder
    pairs = [[query, doc.page_content] for doc in initial_results]

    # Step 3: Get relevance scores from the Cross-Encoder
    scores = reranker.predict(pairs)

    # Step 4: Sort the indices based on the highest scores
    best_indices = np.argsort(scores)[::-1][:final_k]

    # Step 5: Extract the top re-ranked chunks
    top_results = [initial_results[i] for i in best_indices]

    # Return formatted list
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in top_results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about AURORASTAY HOTELS.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exclusions, limitations, exceptions, or coverage details.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent benefits, exclusions, limits, waiting periods, conditions, or coverage details.
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 3,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks using the updated two-stage pipeline
    retrieved_chunks = retrieve(
        query=query,
        initial_k=10,
        final_k=k,
        vectorstore=vectorstore
    )

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
