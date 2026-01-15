# Week 0 Assignment: Build a RAG Metrics Dashboard

## Learning Goal

Shift from "did it work in the demo" to "can I measure if it's improving." You'll build the foundation for systematic RAG improvement by implementing logging and visualization.

## Setup

```bash
uv add chromadb sentence-transformers openai sqlite-utils streamlit
```

## Success Criteria

- ChromaDB collection indexed with 100+ documents
- SQLite logging captures query, response, and feedback
- Dashboard shows query volume and satisfaction trends
- At least 20 logged queries with feedback

## Why This Works

- **You get leading metrics early**: query volume, satisfaction rate, and failure patterns show what to fix next.
- **You build the flywheel**: logging turns usage into data, and data turns into better retrieval and prompts.
- **You avoid “vibes-based” changes**: you can compare changes over time instead of guessing.

## Common Mistakes

- **Not logging retrieval details**: if you only log the final answer, you can’t tell if failures are retrieval or generation.
- **Over-trusting a single number**: “overall satisfaction” hides bad segments; always slice by query type or topic.
- **Mixing feedback labels**: be consistent (e.g., `1`, `-1`, `None`) so your charts don’t lie.

## Dataset

Use the **SQuAD 2.0** dataset from Hugging Face:

```python
from datasets import load_dataset

# Load SQuAD 2.0 - contains questions, contexts, and answers
squad = load_dataset("rajpurkar/squad_v2", split="train[:1000]")

# Each item has: id, title, context, question, answers
# Use 'context' as your document corpus
documents = list(set(squad["context"]))  # ~300 unique passages
```

## Requirements

### Part 1: Build RAG Pipeline with ChromaDB

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB with persistence
client = chromadb.PersistentClient(path="./chroma_db")

# Use sentence-transformers for embeddings
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = client.get_or_create_collection(
    name="squad_docs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"source": "squad"} for _ in documents]
)

# Query
results = collection.query(
    query_texts=["What is the capital of France?"],
    n_results=5,
    include=["documents", "distances", "metadatas"]
)
```

### Part 2: Implement Query Logging

Log every query to SQLite:

```python
import sqlite3
from datetime import datetime
import json

def log_query(
    query: str,
    retrieved_chunks: list[str],
    distances: list[float],
    response: str,
    feedback: int | None = None  # 1 = good, -1 = bad, None = no feedback
):
    conn = sqlite3.connect("rag_metrics.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            query TEXT,
            retrieved_chunks TEXT,
            distances TEXT,
            response TEXT,
            feedback INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO queries (timestamp, query, retrieved_chunks, distances, response, feedback) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), query, json.dumps(retrieved_chunks), json.dumps(distances), response, feedback)
    )
    conn.commit()
    conn.close()
```

### Part 3: Build Streamlit Dashboard

Create `dashboard.py`:

```python
import streamlit as st
import sqlite3
import pandas as pd

st.title("RAG Metrics Dashboard")

conn = sqlite3.connect("rag_metrics.db")
df = pd.read_sql("SELECT * FROM queries", conn)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Queries", len(df))
col2.metric("Positive Feedback %", f"{(df['feedback'] == 1).mean() * 100:.1f}%")
col3.metric("Queries Today", len(df[df['timestamp'].str[:10] == pd.Timestamp.now().strftime('%Y-%m-%d')]))

# Query volume over time
st.subheader("Query Volume by Day")
df['date'] = pd.to_datetime(df['timestamp']).dt.date
st.bar_chart(df.groupby('date').size())

# Satisfaction trend
st.subheader("Satisfaction Rate by Day")
daily_satisfaction = df.groupby('date')['feedback'].apply(lambda x: (x == 1).mean())
st.line_chart(daily_satisfaction)
```

Run with: `streamlit run dashboard.py`

## Deliverable

A working system with:

1. ChromaDB collection with SQuAD documents indexed
2. Query logging to SQLite database
3. Streamlit dashboard showing query volume and satisfaction metrics
4. At least 20 logged queries with mixed feedback

## Bonus

Add "inventory vs capability" classification:

```python
def classify_failure(query: str, response: str, retrieved_chunks: list[str]) -> str:
    """
    Inventory issue: relevant data exists but wasn't retrieved
    Capability issue: the data doesn't exist in our corpus
    """
    # Use an LLM to classify based on query and retrieved context
    ...
```
