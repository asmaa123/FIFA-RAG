#!/usr/bin/env python3
"""
FastAPI Backend for FIFA World Cup RAG System
Endpoints: /api/query, /api/health, /api/evaluate
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="FIFA World Cup RAG System",
    description="Retrieve and generate answers about FIFA World Cup using RAG",
    version="1.0.0"
)

# Initialize RAG Pipeline (global)
rag_pipeline = None

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    metrics: dict

class EvaluateRequest(BaseModel):
    questions: List[dict]  # List of {"question": str, "answer": str}

class HealthResponse(BaseModel):
    status: str
    model: str
    chunks_loaded: int

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    
    print("🚀 Starting FastAPI Server...")
    print("📥 Loading RAG Pipeline...")
    
    chunks_path = os.path.join(os.path.dirname(__file__), "data", "chunks.json")
    embeddings_path = os.path.join(os.path.dirname(__file__), "data", "embeddings.npy")
    
    try:
        rag_pipeline = RAGPipeline(
            chunks_json=chunks_path,
            embeddings_npy=embeddings_path,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            top_k=5
        )
        print("✅ RAG Pipeline loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load RAG Pipeline: {e}")
        raise

# Routes

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    return HealthResponse(
        status="ok",
        model="mixtral-8x7b-32768",
        chunks_loaded=8  # Hardcoded based on our chunks
    )

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main RAG query endpoint"""
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag_pipeline.query(request.question, top_k=request.top_k)
        
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            metrics=result["metrics"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/evaluate")
async def evaluate_endpoint(request: EvaluateRequest):
    """Evaluate RAG system on Q&A pairs"""
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    if not request.questions:
        raise HTTPException(status_code=400, detail="Questions list cannot be empty")
    
    try:
        evaluation_results = rag_pipeline.evaluate_qa(request.questions)
        
        return {
            "status": "ok",
            "evaluation": evaluation_results,
            "summary": {
                "total_questions": evaluation_results["total_questions"],
                "correct_answers": evaluation_results["correct_answers"],
                "accuracy": evaluation_results["accuracy"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating: {str(e)}")

@app.post("/api/batch-query")
async def batch_query_endpoint(questions: List[str]):
    """Process multiple questions in batch"""
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    if not questions:
        raise HTTPException(status_code=400, detail="Questions list cannot be empty")
    
    try:
        results = []
        for question in questions:
            result = rag_pipeline.query(question)
            results.append(result)
        
        return {
            "status": "ok",
            "total_questions": len(questions),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "ok",
        "message": "FIFA World Cup RAG API",
        "endpoints": {
            "health": "/api/health",
            "query": "/api/query (POST)",
            "evaluate": "/api/evaluate (POST)",
            "batch-query": "/api/batch-query (POST)",
            "docs": "/docs"
        }
    }

# Run
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
