# Week 1 Assignment: Build a Retrieval Evaluation Pipeline

## Learning Goal

Build evaluation metrics from scratch to understand how retrieval quality is measured. You'll generate synthetic test data and implement precision, recall, MRR, and NDCG.

## Setup

```bash
uv add chromadb sentence-transformers pandas matplotlib openai datasets
```

## Success Criteria

- Custom metric implementations match expected behavior (test with known inputs)
- Evaluation runs on 100+ queries
- Visualizations show clear precision/recall tradeoff across k values
- Analysis identifies optimal k for your use case

## Why This Works

- **Retrieval is cheaper to measure than generation**: you can run hundreds of tests fast without reading long answers.
- **Precision and recall tell different stories**: precision measures “how clean” results are; recall measures “did we find the right stuff.”
- **Multiple \(k\) values reveal tradeoffs**: \(k=3\) might look great on precision but miss important docs.

## Common Mistakes

- **Evaluating at one \(k\) only**: you’ll overfit to a single number and miss the real tradeoff curve.
- **Confusing MRR with recall**: MRR cares about the first relevant result’s rank, not how many relevant docs you got.
- **Bad ground truth**: if your “relevant_ids” are incomplete, you will “punish” good retrieval.

## Dataset

Use **MS MARCO** passage ranking dataset:

```python
from datasets import load_dataset

# Load MS MARCO - real web search queries with relevance labels
msmarco = load_dataset("microsoft/ms_marco", "v1.1", split="train[:2000]")

# Structure: query, passages (list), answers
# passages contains: is_selected (relevance), passage_text

# Extract documents and create ground truth
documents = []
ground_truth = {}  # query_id -> set of relevant doc_ids

for i, item in enumerate(msmarco):
    query_id = f"q_{i}"
    relevant_docs = set()
    
    for j, passage in enumerate(item["passages"]["passage_text"]):
        doc_id = f"doc_{i}_{j}"
        documents.append({"id": doc_id, "text": passage})
        
        if item["passages"]["is_selected"][j]:
            relevant_docs.add(doc_id)
    
    if relevant_docs:
        ground_truth[query_id] = relevant_docs
```

## Requirements

### Part 1: Implement Evaluation Metrics

Build these metrics from scratch (no external libraries):

```python
import math

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """What fraction of retrieved docs are relevant?"""
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return relevant_retrieved / k if k > 0 else 0.0

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """What fraction of relevant docs did we find?"""
    retrieved_k = set(retrieved_ids[:k])
    relevant_retrieved = len(retrieved_k & relevant_ids)
    return relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """How high is the first relevant result?"""
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain - rewards relevant docs ranked higher"""
    dcg = sum(
        (1 if doc_id in relevant_ids else 0) / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
    )
    ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

### Part 2: Generate Synthetic Questions

Use an LLM to generate diverse test questions:

```python
import json
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def generate_synthetic_questions(document: str, n: int = 3) -> list[dict]:
    """Generate questions that can be answered by this document."""
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[{
            "role": "user",
            "content": f"""Generate {n} diverse questions that can be answered using this document.

Document:
{document}

Return as JSON array: [{{"question": "...", "type": "factual|inferential|comparative"}}]"""
        }],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)["questions"]
```

If you want a runnable reference implementation, see `latest/assignments/week1/synthetic_data.py`.

### Part 3: Run Evaluation

```python
import chromadb
from chromadb.utils import embedding_functions

# Setup ChromaDB
client = chromadb.PersistentClient(path="./eval_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection("msmarco", embedding_function=embedding_fn)

# Index documents
collection.add(
    documents=[d["text"] for d in documents],
    ids=[d["id"] for d in documents]
)

# Evaluate at different k values
k_values = [3, 5, 10, 20]
results = {k: {"precision": [], "recall": [], "mrr": [], "ndcg": []} for k in k_values}

for query_id, query_text in queries.items():
    relevant_ids = ground_truth[query_id]
    
    # Retrieve
    search_results = collection.query(query_texts=[query_text], n_results=max(k_values))
    retrieved_ids = search_results["ids"][0]
    
    # Calculate metrics at each k
    for k in k_values:
        results[k]["precision"].append(precision_at_k(retrieved_ids, relevant_ids, k))
        results[k]["recall"].append(recall_at_k(retrieved_ids, relevant_ids, k))
        results[k]["mrr"].append(mean_reciprocal_rank(retrieved_ids, relevant_ids))
        results[k]["ndcg"].append(ndcg_at_k(retrieved_ids, relevant_ids, k))

# Print averages
for k in k_values:
    print(f"\n=== k={k} ===")
    print(f"Precision@{k}: {sum(results[k]['precision'])/len(results[k]['precision']):.3f}")
    print(f"Recall@{k}: {sum(results[k]['recall'])/len(results[k]['recall']):.3f}")
    print(f"MRR: {sum(results[k]['mrr'])/len(results[k]['mrr']):.3f}")
    print(f"NDCG@{k}: {sum(results[k]['ndcg'])/len(results[k]['ndcg']):.3f}")
```

### Part 4: Visualize Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

metrics = ["precision", "recall", "mrr", "ndcg"]
for ax, metric in zip(axes.flat, metrics):
    values = [sum(results[k][metric])/len(results[k][metric]) for k in k_values]
    ax.bar([str(k) for k in k_values], values)
    ax.set_title(f"{metric.upper()} at different k")
    ax.set_xlabel("k")
    ax.set_ylabel(metric)

plt.tight_layout()
plt.savefig("evaluation_results.png")
```

## Deliverable

A Jupyter notebook showing:

1. MS MARCO dataset loaded and indexed in ChromaDB
2. Custom implementation of precision@k, recall@k, MRR, NDCG
3. Evaluation results across k=3, 5, 10, 20
4. Visualizations comparing metrics at different k values
5. Brief analysis: Which k provides the best precision/recall tradeoff?

## Bonus: Hard Negative Mining

Generate adversarial test cases using high-similarity but irrelevant documents:

```python
def find_hard_negatives(query: str, relevant_ids: set[str], collection, n: int = 5) -> list[str]:
    """Find documents that are similar to query but NOT relevant."""
    results = collection.query(query_texts=[query], n_results=n * 3)
    hard_negatives = [
        doc_id for doc_id in results["ids"][0] 
        if doc_id not in relevant_ids
    ][:n]
    return hard_negatives
```
