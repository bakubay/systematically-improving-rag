"""
Week 0: Basic RAG Pipeline with Metrics Dashboard

Build a RAG pipeline with ChromaDB and logging for metrics tracking.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Check for optional dependencies
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("chromadb not installed. Run: uv add chromadb")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("openai not installed. Run: uv add openai")


@dataclass
class QueryLog:
    """Log entry for a query."""
    query: str
    retrieved_chunks: list[str]
    distances: list[float]
    response: str
    feedback: Optional[int] = None  # 1 = good, -1 = bad, None = no feedback
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class MetricsLogger:
    """Log queries and metrics to SQLite."""
    
    def __init__(self, db_path: str = "rag_metrics.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                retrieved_chunks TEXT,
                distances TEXT,
                response TEXT,
                feedback INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def log(self, entry: QueryLog):
        """Log a query entry."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO queries 
               (timestamp, query, retrieved_chunks, distances, response, feedback)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entry.timestamp,
                entry.query,
                json.dumps(entry.retrieved_chunks),
                json.dumps(entry.distances),
                entry.response,
                entry.feedback,
            )
        )
        conn.commit()
        conn.close()
    
    def update_feedback(self, query_id: int, feedback: int):
        """Update feedback for a query."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE queries SET feedback = ? WHERE id = ?",
            (feedback, query_id)
        )
        conn.commit()
        conn.close()
    
    def get_stats(self) -> dict:
        """Get aggregate statistics."""
        conn = sqlite3.connect(self.db_path)
        
        # Total queries
        total = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        
        # Feedback stats
        positive = conn.execute(
            "SELECT COUNT(*) FROM queries WHERE feedback = 1"
        ).fetchone()[0]
        negative = conn.execute(
            "SELECT COUNT(*) FROM queries WHERE feedback = -1"
        ).fetchone()[0]
        
        # Queries by day
        daily = conn.execute("""
            SELECT DATE(timestamp) as day, COUNT(*) as count
            FROM queries
            GROUP BY DATE(timestamp)
            ORDER BY day DESC
            LIMIT 7
        """).fetchall()
        
        conn.close()
        
        total_feedback = positive + negative
        
        return {
            "total_queries": total,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "satisfaction_rate": positive / total_feedback if total_feedback > 0 else 0.0,
            "queries_by_day": [{"day": d[0], "count": d[1]} for d in daily],
        }
    
    def get_recent_queries(self, limit: int = 10) -> list[dict]:
        """Get recent queries."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT id, timestamp, query, response, feedback 
               FROM queries ORDER BY timestamp DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        conn.close()
        
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "query": r[2],
                "response": r[3],
                "feedback": r[4],
            }
            for r in rows
        ]


class RAGPipeline:
    """Basic RAG pipeline with ChromaDB."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_path: Optional[str] = None,
        db_path: str = "rag_metrics.db",
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb required. Run: uv add chromadb")
        
        # Initialize ChromaDB
        if persist_path:
            self.client = chromadb.PersistentClient(path=persist_path)
        else:
            self.client = chromadb.Client()
        
        # Use default embedding function
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize logger
        self.logger = MetricsLogger(db_path)
        
        # OpenAI client (optional)
        self.openai_client = OpenAI() if OPENAI_AVAILABLE else None
    
    def add_documents(self, documents: list[str], ids: Optional[list[str]] = None):
        """Add documents to the collection."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(documents=documents, ids=ids)
        print(f"Added {len(documents)} documents")
    
    def retrieve(self, query: str, k: int = 5) -> tuple[list[str], list[float]]:
        """Retrieve top-k documents for a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "distances"]
        )
        
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        
        return documents, distances
    
    def generate_response(self, query: str, context: list[str]) -> str:
        """Generate a response using retrieved context."""
        if not self.openai_client:
            # Fallback: simple concatenation
            return "Based on the context:\n\n" + "\n---\n".join(context[:3])
        
        # Use OpenAI to generate response
        context_text = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(context))
        
        response = self.openai_client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question based on the provided context. Cite sources using [1], [2], etc."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}"
                }
            ],
            max_tokens=500,
        )
        
        return response.choices[0].message.content
    
    def query(self, query: str, k: int = 5, log: bool = True) -> dict:
        """Run a full RAG query."""
        # Retrieve
        documents, distances = self.retrieve(query, k)
        
        # Generate
        response = self.generate_response(query, documents)
        
        # Log
        if log:
            entry = QueryLog(
                query=query,
                retrieved_chunks=documents,
                distances=distances,
                response=response,
            )
            self.logger.log(entry)
        
        return {
            "query": query,
            "response": response,
            "sources": documents,
            "distances": distances,
        }
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return self.logger.get_stats()


def create_sample_documents() -> list[str]:
    """Create sample documents for testing."""
    return [
        "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to produce more accurate and grounded responses.",
        "Vector databases store data as high-dimensional vectors, enabling fast similarity search for applications like semantic search and recommendation systems.",
        "ChromaDB is an open-source embedding database designed for AI applications. It provides simple APIs for storing and querying embeddings.",
        "Embeddings are dense vector representations of data (text, images, etc.) that capture semantic meaning in a numerical format.",
        "Large Language Models (LLMs) like GPT are trained on vast amounts of text data to understand and generate human-like text.",
        "Fine-tuning is the process of adapting a pre-trained model to a specific task or domain by training it on additional data.",
        "Semantic search goes beyond keyword matching to understand the meaning and intent behind queries, providing more relevant results.",
        "The attention mechanism in transformers allows models to focus on relevant parts of the input when generating each part of the output.",
    ]


def main():
    """Demo the RAG pipeline."""
    print("=" * 60)
    print("WEEK 0: RAG PIPELINE WITH METRICS")
    print("=" * 60)
    
    # Initialize pipeline
    db_path = str(Path(__file__).parent / "rag_metrics.db")
    pipeline = RAGPipeline(
        collection_name="demo_docs",
        db_path=db_path,
    )
    
    # Add sample documents
    print("\nAdding sample documents...")
    documents = create_sample_documents()
    pipeline.add_documents(documents)
    
    # Run some queries
    test_queries = [
        "What is Python?",
        "How does RAG work?",
        "What are embeddings?",
        "Explain machine learning",
        "What is ChromaDB?",
    ]
    
    print("\n" + "-" * 60)
    print("RUNNING QUERIES")
    print("-" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = pipeline.query(query, k=3)
        print(f"Response: {result['response'][:200]}...")
        print(f"Top source distance: {result['distances'][0]:.4f}")
    
    # Show stats
    print("\n" + "-" * 60)
    print("METRICS DASHBOARD")
    print("-" * 60)
    
    stats = pipeline.get_stats()
    print(f"\nTotal Queries: {stats['total_queries']}")
    print(f"Positive Feedback: {stats['positive_feedback']}")
    print(f"Negative Feedback: {stats['negative_feedback']}")
    print(f"Satisfaction Rate: {stats['satisfaction_rate']:.1%}")
    
    if stats['queries_by_day']:
        print("\nQueries by Day:")
        for day in stats['queries_by_day']:
            print(f"  {day['day']}: {day['count']} queries")
    
    # Show recent queries
    print("\nRecent Queries:")
    recent = pipeline.logger.get_recent_queries(5)
    for q in recent:
        feedback_str = "+" if q['feedback'] == 1 else ("-" if q['feedback'] == -1 else "?")
        print(f"  [{feedback_str}] {q['query'][:50]}...")
    
    print("\n" + "=" * 60)
    print(f"Metrics saved to: {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
