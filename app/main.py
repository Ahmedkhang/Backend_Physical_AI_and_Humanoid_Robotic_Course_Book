from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os


app = FastAPI(title="Physical AI Book RAG Backend")

class QueryRequest(BaseModel):
    query: str
    selected_text: str | None = None  # for highlighted text mode

@app.get("/")
async def root():
    return {"message": "Physical AI RAG Backend is alive and ready! 🚀", "status": "healthy"}
from app.rag import router as rag_router
app.include_router(rag_router, prefix="/api")
# We'll add /ask endpoint in rag.py
