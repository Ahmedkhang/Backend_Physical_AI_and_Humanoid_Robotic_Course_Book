from fastapi import FastAPI, Depends
from pydantic import BaseModel
from qdrant_client import QdrantClient
import cohere
from openai import OpenAI
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Physical AI & Humanoid Robotics RAG Backend",
    description="RAG chatbot for the course book",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    selected_text: str | None = None

# Health endpoint for HF
@app.get("/")
async def root():
    return {"status": "RAG backend ready! 🚀", "message": "API is live"}

# Dependencies
def get_qdrant():
    return QdrantClient(
        # url=os.getenv("QDRANT_URL"),
        # api_key=os.getenv("QDRANT_API_KEY")
        
          url = "https://3828422c-46dd-423e-ae04-60d1c8c180ac.europe-west3-0.gcp.cloud.qdrant.io:6333",
          api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.O5cbwPZSZHB5wY6MzkWLjNcVIFcf7GrHbZHONgEh1fY"
    )

def get_cohere() -> cohere.Client:
    api_key = "EZeCjZvxDjpFwNf59ycDKnui2r7n20DMGSIjP25W"
    # print("Cohere key length:", len(os.getenv("COHERE_API_KEY") or ""))
    # print("Cohere key:", api_key)
    if not api_key:
        raise ValueError("COHERE_API_KEY missing")
    return cohere.Client(api_key=api_key)

def get_gemini() -> OpenAI:
    # api_key = os.getenv("GEMINI_API_KEY")
    api_key = "AIzaSyAViKui7Ub53Z30wHZWu6RGfuM2kHNG9XM"
    if not api_key:
        raise ValueError("GEMINI_API_KEY missing")
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
@app.post("/api/ask")
async def ask_book(
    request: QueryRequest,
    qdrant: QdrantClient = Depends(get_qdrant),
    cohere: cohere.Client = Depends(get_cohere),
    gemini: OpenAI = Depends(get_gemini)
):
    collection_name = os.getenv("COLLECTION_NAME", "physical-ai-book")

    # Embed query
    query_embedding = cohere.embed(
        texts=[request.query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    # Search Qdrant (updated for newer qdrant-client)
    search_results = qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=6,
        with_payload=True
    ).points

    if not search_results:
        return {"answer": "No relevant content found in the book.", "sources": []}

    # Build context
    context = "\n\n".join([hit.payload["text"] for hit in search_results])
    if request.selected_text:
        context = request.selected_text + "\n\nBook context:\n" + context

    # Generate answer with Gemini
    response = gemini.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": """"You are an expert teaching assistant for the "Physical AI & Humanoid Robotics" course. Your knowledge is strictly limited to the content of the official course book. You must answer questions using ONLY the provided context from the book — never hallucinate or add external information.
Key guidelines:
1. Always base your answer on the retrieved context. Quote or paraphrase directly from it when possible.
2. If the exact topic (e.g., a specific module, chapter, or concept) is not present in the context, do NOT say "I don't know" abruptly. Instead, respond helpfully:
   - Politely inform the user that this specific topic does not appear in the current book content.
   - If related concepts ARE present in the context (e.g., motion planning, whole-body control, reinforcement learning, simulation), briefly mention them and suggest they might be relevant.
   - Suggest possible reasons: the topic might be covered in a different module, assumed as prerequisite knowledge, or planned for future updates.
   - Recommend searching with related keywords or checking specific modules (e.g., "Try asking about 'whole-body locomotion' in Module 5" or "This might be covered under reinforcement learning in Module 6").
3. Be encouraging and educational: Frame responses to help the student learn, even when direct information is missing.
   - Example: "Forward kinematics for manipulators isn't explicitly detailed in the current book sections, but the course heavily focuses on locomotion and whole-body dynamics. You might find related ideas in chapters on inverse kinematics for balancing or trajectory optimization."
4. When sources are provided, reference them naturally (e.g., "As described in the chapter on MPC control..." or "According to the multi-robot coordination section...").
5. Keep answers clear, structured, and concise unless the user asks for depth.
Your goal is to help students deeply understand the course material while staying faithful to the book content."""},
            {"role": "user", "content": f"Question: {request.query}\n\nContext:\n{context}"}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    answer = response.choices[0].message.content.strip()

    sources = [
        {"url": hit.payload.get("url", ""), "snippet": hit.payload["text"][:300] + "..."}
        for hit in search_results
    ]

    return {"answer": answer, "sources": sources}
