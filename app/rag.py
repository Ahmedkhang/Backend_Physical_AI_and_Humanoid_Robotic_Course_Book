from fastapi import APIRouter, Depends
from pydantic import BaseModel
from qdrant_client import QdrantClient
from cohere import Client as CohereClient  # ← Cohere for embeddings
from openai import OpenAI  # ← OpenAI SDK for Gemini generation
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Pydantic model (make sure this is defined here or imported)
class QueryRequest(BaseModel):
    query: str
    selected_text: str | None = None

# Dependency: Qdrant client
def get_qdrant():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

# Dependency: Cohere client (for embeddings)
def get_cohere() -> CohereClient:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not set in .env")
    return CohereClient(api_key=api_key)

# Dependency: OpenAI client pointed at Gemini
def get_gemini() -> OpenAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # ← Critical: ends with /openai/
    )
@router.post("/ask")
async def ask_book(
    request: QueryRequest,
    qdrant: QdrantClient = Depends(get_qdrant),
    cohere: CohereClient = Depends(get_cohere),
    gemini: OpenAI = Depends(get_gemini)
):
    collection_name = os.getenv("COLLECTION_NAME", "physical-ai-book")

    # Embed the query using Cohere
    query_embedding = cohere.embed(
        texts=[request.query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    # Search in Qdrant
    search_results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=6,
        with_payload=True
    )

    if not search_results:
        return {"answer": "No relevant content found in the book.", "sources": []}

    context = "\n\n".join([hit.payload["text"] for hit in search_results])

    if request.selected_text:
        context = request.selected_text + "\n\nAdditional context from book:\n" + context

    # ← THIS IS WHERE YOUR CODE GOES — Gemini generation
    response = gemini.chat.completions.create(
        model="gemini-2.5-flash",  # ← Correct current model name
        messages=[
            {"role": "system", "content": "You are a precise teaching assistant for the 'Physical AI and Humanoid Robotics' textbook. Answer the user's question accurately and completely using ONLY the provided book context. If the exact information is not in the context, say 'This information is not specified in the available book content.' Do not guess or add external knowledge. Be concise and structured."},
            {"role": "user", "content": f"Question: {request.query}\n\nContext:\n{context}"}
        ],
        temperature=0.3,
        max_tokens=1000
    )

    answer = response.choices[0].message.content.strip()

    sources = [
        {
            "url": hit.payload["url"],
            "snippet": hit.payload["text"][:300] + "..."
        }
        for hit in search_results
    ]

    return {"answer": answer, "sources": sources}
    # Rest of your code...