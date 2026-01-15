# Week 7 Assignment: Production-Ready RAG System

## Learning Goal

Harden your RAG system for production with caching, graceful degradation, and cost tracking.

## Setup

```bash
uv add chromadb redis prometheus-client structlog fastapi uvicorn sentence-transformers
```

## Success Criteria

- Cache hit rate > 30% on repeated queries
- Graceful degradation handles primary retriever timeout
- Cost tracking logs tokens and estimated cost per query
- FastAPI endpoint returns response, source, latency, and cost

## Why This Works

- **Caching attacks your biggest cost**: generation is usually the largest spend in production RAG.
- **Degradation prevents outages**: a “good enough” fallback beats a hard failure.
- **Cost per successful query is the real KPI**: it forces you to balance quality and spend.

## Common Mistakes

- **Caching the wrong thing**: caching bad answers locks in failures. Cache only after validation or good feedback.
- **No TTL strategy**: without TTLs, you serve stale answers after docs change.
- **No measurement**: if you don’t log hit rate and cost/query, you can’t prove ROI.

## Dataset

Use a dataset from earlier weeks (SQuAD, MS MARCO, Natural Questions, or your own).

## Requirements

### Part 1: Multi-Level Caching

```python
import hashlib
import json
import redis
from sentence_transformers import SentenceTransformer

class RAGCache:
    def __init__(self) -> None:
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.embedding_cache: dict[str, list[float]] = {}
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _key(self, prefix: str, value: str) -> str:
        return f"{prefix}:{hashlib.md5(value.encode()).hexdigest()}"

    def get_embedding(self, text: str) -> list[float]:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        key = self._key("embed", text)
        cached = self.redis_client.get(key)
        if cached:
            embedding = json.loads(cached)
            self.embedding_cache[text] = embedding
            return embedding
        embedding = self.embedding_model.encode(text).tolist()
        self.embedding_cache[text] = embedding
        self.redis_client.setex(key, 3600, json.dumps(embedding))
        return embedding

    def get_response(self, query: str) -> str | None:
        key = self._key("response", query)
        cached = self.redis_client.get(key)
        return json.loads(cached) if cached else None

    def set_response(self, query: str, response: str, ttl: int = 3600) -> None:
        key = self._key("response", query)
        self.redis_client.setex(key, ttl, json.dumps(response))
```

### Part 2: Graceful Degradation

```python
import asyncio
from typing import Any

class RobustRetriever:
    def __init__(self, cache: RAGCache) -> None:
        self.cache = cache

    async def retrieve(self, query: str) -> dict[str, Any]:
        cached = self.cache.get_response(query)
        if cached:
            return {"source": "cache", "results": cached}

        try:
            results = await asyncio.wait_for(self._primary_retriever(query), timeout=2.0)
            return {"source": "primary", "results": results}
        except asyncio.TimeoutError:
            return {"source": "fallback", "results": await self._fallback_retriever(query)}

    async def _primary_retriever(self, query: str) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return {"documents": ["Primary result"], "scores": [0.92]}

    async def _fallback_retriever(self, query: str) -> dict[str, Any]:
        await asyncio.sleep(0.02)
        return {"documents": ["Fallback result"], "scores": [0.70]}
```

### Part 3: Cost Tracking

```python
import structlog
from prometheus_client import Counter, Histogram

logger = structlog.get_logger()

query_counter = Counter("rag_queries_total", "Total queries", ["model", "status"])
query_cost = Counter("rag_cost_total", "Total cost in USD", ["model"])
query_latency = Histogram("rag_latency_seconds", "Query latency in seconds", ["model"])

PRICING = {
    "gpt-5.2": {"input": 0.15, "output": 0.60},
}

def calculate_cost(tokens_in: int, tokens_out: int, model: str) -> float:
    pricing = PRICING.get(model)
    if not pricing:
        return 0.0
    return (tokens_in / 1_000_000) * pricing["input"] + (tokens_out / 1_000_000) * pricing["output"]
```

### Part 4: Production API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

app = FastAPI()
cache = RAGCache()
retriever = RobustRetriever(cache)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_endpoint(request: QueryRequest) -> dict[str, str | float]:
    start = time.time()
    result = await retriever.retrieve(request.query)
    response_text = str(result["results"])
    cache.set_response(request.query, response_text)

    latency = time.time() - start
    cost = calculate_cost(tokens_in=len(request.query), tokens_out=len(response_text), model="gpt-5.2")

    query_counter.labels(model="gpt-5.2", status="success").inc()
    query_cost.labels(model="gpt-5.2").inc(cost)
    query_latency.labels(model="gpt-5.2").observe(latency)

    return {
        "response": response_text,
        "source": result["source"],
        "latency": latency,
        "cost": cost,
    }
```

## Deliverable

- Multi-level cache with Redis and in-memory layers
- Graceful degradation path (primary to fallback)
- Cost tracking with Prometheus metrics
- FastAPI endpoint for production use

## Bonus

Add a metrics endpoint and a simple dashboard to visualize latency and cost trends.

## Extra Notes (Token Economics)

Use this simple mental model:

- **Cost scales with tokens**: fewer prompt tokens and fewer output tokens usually means lower spend.
- **Cost scales with calls**: caching and batching reduce how often you call the model.

If you know your traffic and rough token counts, you can estimate monthly cost:

1. queries/day × (1 - cache_hit_rate)
2. Multiply by cost/query
3. Multiply by 30 for a month

This is not perfect, but it helps you decide if an optimization is worth the engineering time.
