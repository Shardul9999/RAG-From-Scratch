import os
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import embed, generate_answer
from db import (
    search_chunks,
    insert_audit_log,
    get_db_stats,
    get_document_stats,
    delete_chunks_by_source,
    truncate_chunks,
    get_recent_audit_logs,
)
from worker import index_document
from cache import (
    get_cached_answer,
    store_answer,
    flush_semantic_cache,
    get_cache_stats,
)
from reranker import rerank

app = FastAPI(title="RAG Service API")

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FlushRequest(BaseModel):
    source: Optional[str] = None

@app.post("/upload")
async def upload(file: UploadFile):
    os.makedirs("temp", exist_ok=True)
    path = f"temp/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())
    
    index_document.delay(path, "default")

    return {
        "status": "indexing",
        "filename": file.filename,
        "path": path
    }

@app.get("/query")
def query(q: str):
    start_time = time.time()

    # Query embedding
    vector = embed(q)

    # Semantic cache check
    cached_response = get_cached_answer(vector)
    if cached_response is not None:
        latency_ms = round((time.time() - start_time) * 1000)
        return {
            "source": "semantic_cache",
            "latency_ms": latency_ms,
            **cached_response
        }

    # Retrieve top 20 candidate chunks from pgvector
    results = search_chunks(
        tenant="default",
        embedding=vector,
        limit=20
    )

    # Similarity threshold 
    results = [
        r
        for r in results
        if r.score >= 0.50
    ]

    # Cross-Encoder Reranking
    results = rerank(
        q, results, top_k=5
    )

    # Generate answer with Gemini via LiteLLM
    answer, llm_response = generate_answer(
        q,
        results
    )

    citations = []
    for citation_number in answer.citations:
        index = citation_number - 1

        if 0 <= index < len(results):
            chunk = results[index]

            citations.append({
                "chunk": citation_number,
                "source": chunk.source,
                "title": chunk.title,
                "content": getattr(chunk, "content", ""),
                "score": round(float(chunk.score), 4) if hasattr(chunk, "score") else None
            })

    usage = getattr(llm_response, "usage", None)
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    hidden_params = getattr(llm_response, "_hidden_params", {}) or {}
    cost = hidden_params.get("response_cost", 0)

    insert_audit_log(
        tenant="default",
        query=q,
        model="gemini-2.5-flash",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        chunks_used=len(results)
    )
    
    latency_ms = round((time.time() - start_time) * 1000)
    response_data = {
        "answer": answer.answer,
        "citations": citations,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "cost": cost
        } if input_tokens else None,
        "candidate_chunks_count": len(results)
    }

    # Store in semantic cache
    if results:
        store_answer(
            query=q,
            query_embedding=vector,
            response=response_data
        )

    return {
        "source": "llm",
        "latency_ms": latency_ms,
        **response_data
    }

@app.get("/api/stats")
def api_stats():
    return {
        "postgres": get_db_stats("default"),
        "redis": get_cache_stats(),
        "models": {
            "llm": "gemini-2.5-flash",
            "embedding": "gemini-embedding-2 (3072 dims)",
            "reranker": "BAAI/bge-reranker-base"
        }
    }

@app.get("/api/documents")
def api_documents():
    return get_document_stats("default")

@app.post("/api/flush-cache")
def api_flush_cache():
    count = flush_semantic_cache()
    return {"status": "success", "deleted_keys": count}

@app.post("/api/flush-chunks")
def api_flush_chunks(req: Optional[FlushRequest] = None):
    source = req.source if req else None
    if source:
        deleted = delete_chunks_by_source(source, "default")
        return {"status": "success", "message": f"Deleted chunks for {source}", "count": deleted}
    else:
        truncate_chunks()
        return {"status": "success", "message": "All chunks flushed"}

@app.get("/api/audit-logs")
def api_audit_logs(limit: int = 15):
    return get_recent_audit_logs("default", limit=limit)

# Mount compiled React SPA if built
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

