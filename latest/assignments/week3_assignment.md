# Week 3 Assignment: Streaming RAG with Citations

## Learning Goal

Build a RAG system with Server-Sent Events (SSE) streaming and interactive citations, improving perceived latency while collecting user feedback.

## Setup

```bash
uv add fastapi uvicorn httpx openai chromadb sentence-transformers
```

## Dataset

Use **SQuAD 2.0** or your documents from Week 0:

```python
from datasets import load_dataset

# Load SQuAD 2.0 passages
squad = load_dataset("rajpurkar/squad_v2", split="train[:500]")
documents = list(set(squad["context"]))  # ~150 unique passages

# Or load from your Week 0 database
import sqlite3
conn = sqlite3.connect("rag_metrics.db")
# ... load logged queries and documents
```

## Success Criteria

- Time to first source (TTFS): < 200ms
- Time to first token (TTFT): < 1000ms
- Streaming reduces perceived latency by 50%+ vs non-streaming
- Citation feedback logged to database

## Why This Works

- **Perceived performance matters**: users wait longer if they see progress instead of silence.
- **Sources-first builds trust**: showing citations early makes the answer feel grounded.
- **Streaming creates feedback moments**: you can ask for feedback while the user is still engaged.

## Common Mistakes

- **Streaming only tokens**: if you don’t stream “what’s happening” (sources, steps), it still feels slow.
- **No retry strategy**: one flaky network call can break the whole experience.
- **No cancellation handling**: users leave tabs; your server should stop work when they disconnect.

## Implementation Note (Retry + Degradation)

In production, you usually add:

- **Retry on the initial request** (not on every streamed chunk)
- **Fallback response** if streaming fails
- **Cancellation handling** so you stop work when the user disconnects

## Part 1: Streaming RAG API

Build a FastAPI backend that streams responses with citations:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
import json

app = FastAPI()
client = OpenAI()

async def stream_rag_response(query: str, sources: list[dict]):
    """Stream RAG response with interleaved citations."""
    
    # First, stream the sources
    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
    
    # Build context from sources
    context = "\n\n".join(
        f"[{i+1}] {s['text']}" 
        for i, s in enumerate(sources)
    )
    
    # Stream the LLM response
    stream = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": f"Answer using these sources:\n{context}"},
            {"role": "user", "content": query}
        ],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield f"data: {json.dumps({'type': 'text', 'data': chunk.choices[0].delta.content})}\n\n"
    
    yield "data: [DONE]\n\n"

@app.get("/stream")
async def stream_query(q: str):
    # Retrieve sources (use your retriever here)
    sources = retrieve_sources(q)
    
    return StreamingResponse(
        stream_rag_response(q, sources),
        media_type="text/event-stream"
    )
```

## Part 2: Frontend with Progressive Loading

```javascript
// Connect to SSE endpoint
const eventSource = new EventSource(`/stream?q=${encodeURIComponent(query)}`);

eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
        eventSource.close();
        return;
    }
    
    const parsed = JSON.parse(event.data);
    
    if (parsed.type === 'sources') {
        // Show sources immediately (before answer)
        renderSources(parsed.data);
    } else if (parsed.type === 'text') {
        // Append streaming text
        appendText(parsed.data);
    }
};
```

## Part 3: Citation Feedback System

Track which citations users click or find helpful:

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CitationFeedback:
    query_id: str
    citation_index: int
    action: str  # "clicked", "helpful", "not_helpful"
    timestamp: datetime

@app.post("/feedback/citation")
async def record_citation_feedback(
    query_id: str,
    citation_index: int,
    action: str
):
    feedback = CitationFeedback(
        query_id=query_id,
        citation_index=citation_index,
        action=action,
        timestamp=datetime.now()
    )
    # Log to database for analysis
    save_feedback(feedback)
    return {"status": "recorded"}
```

## Part 4: Chain of Thought Validation

Add a validation layer to check answer quality:

```python
async def validate_response(query: str, answer: str, sources: list[str]) -> dict:
    """Validate that the answer is grounded in sources."""
    
    validation_prompt = f"""
    Query: {query}
    Answer: {answer}
    Sources: {sources}
    
    Check if the answer:
    1. Is factually grounded in the sources
    2. Doesn't hallucinate information
    3. Properly cites relevant sources
    
    Return JSON: {{"valid": bool, "issues": [str], "confidence": float}}
    """
    
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": validation_prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

## Requirements

1. **Streaming API**
   - FastAPI endpoint with SSE
   - Stream sources before answer text
   - Handle connection drops gracefully

2. **Citation System**
   - Numbered citations in responses
   - Click tracking on citations
   - Thumbs up/down per citation

3. **Validation Layer**
   - Check groundedness after generation
   - Flag potential hallucinations
   - Log validation results for analysis

4. **Latency Metrics**
   - Time to first source (TTFS)
   - Time to first token (TTFT)
   - Total response time
   - Perceived vs actual latency

## Deliverables

1. Working streaming API with SSE
2. Citation feedback collection endpoint
3. Validation pipeline that runs post-generation
4. Latency comparison report (streaming vs non-streaming)

## Bonus

- Add skeleton loading states
- Implement citation highlighting on hover
- Build a feedback dashboard showing citation helpfulness
- Add retry logic for failed streams
