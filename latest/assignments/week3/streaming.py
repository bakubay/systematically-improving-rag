"""
Week 3: Streaming RAG with Citations

Implements SSE streaming, citation tracking, and response validation.
"""

import json
import time
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Optional
import asyncio

# Check dependencies
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("chromadb not installed. Run: uv add chromadb")

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


async def async_retry(
    fn,
    *,
    retries: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 4.0,
    retriable_exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    """Retry an async function with exponential backoff.

    Why this matters (Chapter 3.2):
    - Streaming feels fast, but networks fail.
    - A small retry policy prevents random one-off failures from breaking UX.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except retriable_exceptions:
            attempt += 1
            if attempt > retries:
                raise
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            await asyncio.sleep(delay)


@dataclass
class Source:
    """A retrieved source document."""
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class CitationFeedback:
    """User feedback on a citation."""
    query_id: str
    citation_index: int
    action: str  # "clicked", "helpful", "not_helpful"
    timestamp: datetime


@dataclass
class StreamMetrics:
    """Latency metrics for streaming."""
    query_start: float
    time_to_first_source: Optional[float] = None
    time_to_first_token: Optional[float] = None
    total_time: Optional[float] = None
    tokens_streamed: int = 0


class CitationTracker:
    """Track citation interactions and feedback."""
    
    def __init__(self, db_path: str = "citations.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                citation_index INTEGER NOT NULL,
                action TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citation_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                citation_index INTEGER NOT NULL,
                was_used_in_answer BOOLEAN,
                timestamp TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_feedback(self, feedback: CitationFeedback):
        """Record user feedback on a citation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO citation_feedback (query_id, citation_index, action, timestamp) VALUES (?, ?, ?, ?)",
            (feedback.query_id, feedback.citation_index, feedback.action, feedback.timestamp.isoformat())
        )
        
        conn.commit()
        conn.close()
    
    def record_usage(self, query_id: str, source_id: str, citation_index: int, was_used: bool):
        """Record which citations were used in the answer."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO citation_usage (query_id, source_id, citation_index, was_used_in_answer, timestamp) VALUES (?, ?, ?, ?, ?)",
            (query_id, source_id, citation_index, was_used, datetime.now().isoformat())
        )
        
        conn.commit()
        conn.close()
    
    def get_citation_stats(self) -> dict:
        """Get statistics on citation feedback."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count by action
        cursor.execute("""
            SELECT action, COUNT(*) as count
            FROM citation_feedback
            GROUP BY action
        """)
        
        action_counts = dict(cursor.fetchall())
        
        # Count total citations used
        cursor.execute("SELECT COUNT(*) FROM citation_usage WHERE was_used_in_answer = 1")
        used_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "feedback_counts": action_counts,
            "citations_used": used_count,
        }


class StreamingRAG:
    """RAG system with streaming responses."""
    
    def __init__(self):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb required")
        
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.create_collection(
            name="documents",
            embedding_function=self.embedding_fn,
        )
        
        self.citation_tracker = CitationTracker()
        self.openai_client = AsyncOpenAI() if OPENAI_AVAILABLE else None
    
    def add_documents(self, documents: list[dict]):
        """Add documents to the collection."""
        self.collection.add(
            documents=[d["text"] for d in documents],
            ids=[d["id"] for d in documents],
            metadatas=[d.get("metadata", {}) for d in documents],
        )
    
    def retrieve(self, query: str, k: int = 3) -> list[Source]:
        """Retrieve relevant sources."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )
        
        sources = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                sources.append(Source(
                    id=doc_id,
                    text=results["documents"][0][i],
                    score=1 - results["distances"][0][i] if results["distances"] else 0.0,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))
        
        return sources
    
    async def stream_response(
        self,
        query: str,
        query_id: str,
    ) -> AsyncIterator[dict]:
        """Stream a RAG response with citations."""
        metrics = StreamMetrics(query_start=time.time())
        
        # Retrieve sources
        sources = self.retrieve(query, k=3)
        
        metrics.time_to_first_source = time.time() - metrics.query_start
        
        # Yield sources first
        yield {
            "type": "sources",
            "data": [
                {
                    "id": s.id,
                    "text": s.text[:200] + "..." if len(s.text) > 200 else s.text,
                    "score": s.score,
                }
                for s in sources
            ]
        }
        
        # Build context with numbered citations
        context = "\n\n".join(
            f"[{i+1}] {s.text}"
            for i, s in enumerate(sources)
        )
        
        # Stream LLM response
        if self.openai_client:
            try:
                async def _start_stream():
                    return await self.openai_client.chat.completions.create(
                        model="gpt-5.2",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Answer the question using these sources. "
                                    "Cite sources with [1], [2], etc.\n\n"
                                    f"Sources:\n{context}"
                                ),
                            },
                            {"role": "user", "content": query},
                        ],
                        stream=True,
                    )

                # Retry the initial request (not every streamed chunk).
                stream = await async_retry(_start_stream, retries=2, base_delay_s=0.5)
                
                first_token = True
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        if first_token:
                            metrics.time_to_first_token = time.time() - metrics.query_start
                            first_token = False
                        
                        metrics.tokens_streamed += 1
                        yield {
                            "type": "text",
                            "data": chunk.choices[0].delta.content
                        }
            except asyncio.CancelledError:
                # Client disconnected or task cancelled. Stop work fast.
                raise
            except Exception as e:
                # Graceful degradation: fall back to a simple mock answer.
                yield {
                    "type": "warning",
                    "data": f"Streaming failed, using fallback. Error: {e}",
                }
                mock_response = (
                    f"I had trouble streaming the full answer. Based on the sources [1], "
                    f"here is a short response to '{query}': "
                )
                for word in mock_response.split():
                    metrics.tokens_streamed += 1
                    yield {"type": "text", "data": word + " "}
                    await asyncio.sleep(0.01)
        else:
            # Mock streaming for testing
            metrics.time_to_first_token = time.time() - metrics.query_start
            
            mock_response = f"Based on the sources provided [1], the answer to '{query}' involves "
            mock_response += "multiple factors [2]. The key insight is that this topic "
            mock_response += "requires understanding the underlying concepts [3]."
            
            for word in mock_response.split():
                metrics.tokens_streamed += 1
                yield {"type": "text", "data": word + " "}
                await asyncio.sleep(0.05)  # Simulate streaming delay
        
        # Final metrics
        metrics.total_time = time.time() - metrics.query_start
        
        yield {
            "type": "metrics",
            "data": {
                "time_to_first_source_ms": metrics.time_to_first_source * 1000,
                "time_to_first_token_ms": metrics.time_to_first_token * 1000 if metrics.time_to_first_token else None,
                "total_time_ms": metrics.total_time * 1000,
                "tokens_streamed": metrics.tokens_streamed,
            }
        }
        
        yield {"type": "done", "data": None}
    
    def non_streaming_response(self, query: str) -> tuple[str, list[Source], float]:
        """Non-streaming response for comparison."""
        start = time.time()
        
        sources = self.retrieve(query, k=3)
        
        # Mock response
        response = f"Based on the sources provided [1], the answer to '{query}' involves "
        response += "multiple factors [2]. The key insight is that this topic "
        response += "requires understanding the underlying concepts [3]."
        
        elapsed = time.time() - start
        
        return response, sources, elapsed


class ResponseValidator:
    """Validate RAG responses for groundedness."""
    
    def __init__(self):
        self.client = AsyncOpenAI() if OPENAI_AVAILABLE else None
    
    async def validate(
        self,
        query: str,
        answer: str,
        sources: list[str],
    ) -> dict:
        """Validate that the answer is grounded in sources."""
        
        if not self.client:
            # Mock validation
            return {
                "valid": True,
                "confidence": 0.85,
                "issues": [],
                "citation_coverage": 0.67,
            }
        
        try:
            validation_prompt = f"""
Analyze if this answer is properly grounded in the sources.

Query: {query}

Answer: {answer}

Sources:
{chr(10).join(f'[{i+1}] {s}' for i, s in enumerate(sources))}

Return JSON with:
- valid: boolean (true if answer is grounded)
- confidence: float 0-1
- issues: list of strings (any problems found)
- citation_coverage: float 0-1 (what % of claims are cited)
"""
            
            response = await self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": validation_prompt}],
                response_format={"type": "json_object"},
            )
            
            return json.loads(response.choices[0].message.content)
        
        except Exception as e:
            return {
                "valid": False,
                "confidence": 0.0,
                "issues": [str(e)],
                "citation_coverage": 0.0,
            }


def create_sample_documents() -> list[dict]:
    """Create sample documents for testing."""
    return [
        {
            "id": "doc1",
            "text": "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "metadata": {"topic": "programming"}
        },
        {
            "id": "doc2",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. Common techniques include supervised learning, unsupervised learning, and reinforcement learning.",
            "metadata": {"topic": "ai"}
        },
        {
            "id": "doc3",
            "text": "RAG (Retrieval-Augmented Generation) combines retrieval systems with language models. It first retrieves relevant documents, then uses them as context for generating accurate, grounded responses.",
            "metadata": {"topic": "ai"}
        },
        {
            "id": "doc4",
            "text": "Vector databases store and search high-dimensional embeddings. They enable semantic similarity search, which finds documents based on meaning rather than exact keyword matches.",
            "metadata": {"topic": "databases"}
        },
        {
            "id": "doc5",
            "text": "Server-Sent Events (SSE) is a technology for streaming data from server to client over HTTP. Unlike WebSockets, SSE is unidirectional but simpler to implement and works well for real-time updates.",
            "metadata": {"topic": "web"}
        },
    ]


async def demo_streaming():
    """Demo the streaming RAG system."""
    print("=" * 60)
    print("WEEK 3: STREAMING RAG WITH CITATIONS")
    print("=" * 60)
    
    # Initialize
    rag = StreamingRAG()
    validator = ResponseValidator()
    
    # Add documents
    docs = create_sample_documents()
    rag.add_documents(docs)
    print(f"\nIndexed {len(docs)} documents")
    
    # Test queries
    test_queries = [
        "What is Python?",
        "How does RAG work?",
        "What are vector databases?",
    ]
    
    for query in test_queries:
        query_id = f"q_{hash(query) % 10000}"
        
        print(f"\n{'-' * 60}")
        print(f"Query: {query}")
        print("-" * 60)
        
        # Streaming response
        print("\nStreaming response:")
        full_response = ""
        sources_data = []
        
        async for event in rag.stream_response(query, query_id):
            if event["type"] == "sources":
                sources_data = event["data"]
                print(f"  [Sources received: {len(sources_data)}]")
            elif event["type"] == "text":
                print(event["data"], end="", flush=True)
                full_response += event["data"]
            elif event["type"] == "metrics":
                print("\n\n  Metrics:")
                print(f"    Time to first source: {event['data']['time_to_first_source_ms']:.1f}ms")
                if event['data']['time_to_first_token_ms']:
                    print(f"    Time to first token: {event['data']['time_to_first_token_ms']:.1f}ms")
                print(f"    Total time: {event['data']['total_time_ms']:.1f}ms")
                print(f"    Tokens streamed: {event['data']['tokens_streamed']}")
        
        # Validate response
        print("\n  Validation:")
        validation = await validator.validate(
            query=query,
            answer=full_response,
            sources=[s["text"] for s in sources_data],
        )
        print(f"    Valid: {validation['valid']}")
        print(f"    Confidence: {validation['confidence']:.2f}")
        print(f"    Citation coverage: {validation.get('citation_coverage', 'N/A')}")
        if validation.get("issues"):
            print(f"    Issues: {validation['issues']}")
    
    # Compare streaming vs non-streaming
    print(f"\n{'=' * 60}")
    print("LATENCY COMPARISON")
    print("=" * 60)
    
    query = "What is machine learning?"
    
    # Non-streaming
    response, sources, non_stream_time = rag.non_streaming_response(query)
    print(f"\nNon-streaming total time: {non_stream_time * 1000:.1f}ms")
    print(f"  (User sees nothing until: {non_stream_time * 1000:.1f}ms)")
    
    # Streaming
    print("\nStreaming times:")
    async for event in rag.stream_response(query, "comparison"):
        if event["type"] == "sources":
            print(f"  User sees sources at: {event.get('time_to_first_source_ms', 'N/A')}ms")
        elif event["type"] == "metrics":
            print(f"  User sees first token at: {event['data']['time_to_first_token_ms']:.1f}ms")
            print(f"  Total time: {event['data']['total_time_ms']:.1f}ms")
    
    # Citation stats
    print(f"\n{'=' * 60}")
    print("CITATION FEEDBACK STATS")
    print("=" * 60)
    
    # Record some mock feedback
    tracker = rag.citation_tracker
    tracker.record_feedback(CitationFeedback(
        query_id="q_test",
        citation_index=1,
        action="clicked",
        timestamp=datetime.now()
    ))
    tracker.record_feedback(CitationFeedback(
        query_id="q_test",
        citation_index=1,
        action="helpful",
        timestamp=datetime.now()
    ))
    tracker.record_feedback(CitationFeedback(
        query_id="q_test",
        citation_index=2,
        action="not_helpful",
        timestamp=datetime.now()
    ))
    
    stats = tracker.get_citation_stats()
    print(f"\nFeedback counts: {stats['feedback_counts']}")
    
    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print("=" * 60)


def main():
    """Run the demo."""
    asyncio.run(demo_streaming())


if __name__ == "__main__":
    main()
