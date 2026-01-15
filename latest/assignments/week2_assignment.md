# Week 2 Assignment: Fine-tune an Embedding Model

## Learning Goal

Convert evaluation failures into training data. You'll mine hard negatives from retrieval results and fine-tune an embedding model to improve domain-specific performance.

## Setup

```bash
uv add sentence-transformers chromadb cohere datasets torch
```

## Success Criteria

- Hard negatives mined from retrieval results (not random)
- Training with triplet loss converges (loss decreases)
- Before/after comparison on held-out test set
- Target: 5-15% improvement on domain-specific queries

## Why This Works

- **Hard negatives teach the model the boundary**: easy negatives don’t help because the model already separates them.
- **Embedding fine-tuning is practical**: you can get real wins without training a full language model.
- **You measure before/after**: you only keep the change if it moves metrics from Week 1.

## Common Mistakes

- **Mining “hard” negatives that are actually positives**: this poisons training and makes the model worse.
- **Testing on your training set**: you’ll think you improved, but it won’t generalize.
- **Changing too many knobs at once**: tune one thing (mining range, loss, batch size) and re-measure.

## Dataset

Use **Natural Questions** (simplified) from Hugging Face:

```python
from datasets import load_dataset

# Natural Questions - real Google search queries
nq = load_dataset("google-research-datasets/natural_questions", "default", split="train[:500]")

# Or use the simplified version
nq_simple = load_dataset("sentence-transformers/natural-questions", split="train[:1000]")

# Structure: query, positive (relevant passage)
# We'll mine hard negatives from our retrieval system
```

## Requirements

### Part 1: Create Training Data with Hard Negatives

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
from datasets import Dataset

# Load base model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare dataset with anchor-positive pairs
train_data = Dataset.from_dict({
    "anchor": [item["query"] for item in nq_simple],
    "positive": [item["positive"] for item in nq_simple]
})

# Mine hard negatives
# These are passages that look similar but aren't relevant
hard_train_dataset = mine_hard_negatives(
    dataset=train_data,
    model=model,
    anchor_column_name="anchor",
    positive_column_name="positive",
    num_negatives=5,           # 5 hard negatives per pair
    range_min=10,              # Skip top 10 (too easy)
    range_max=100,             # Consider top 100 candidates
    sampling_strategy="top",   # Take hardest negatives
    batch_size=256,
    use_faiss=True             # Faster similarity search
)

print(f"Training examples: {len(hard_train_dataset)}")
print(f"Sample: {hard_train_dataset[0]}")
```

### Part 2: Fine-tune with Triplet Loss

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Choose loss function
# BatchHardTripletLoss: Uses hardest negatives within each batch
loss = losses.BatchHardTripletLoss(model=model)

# Alternative: MultipleNegativesRankingLoss (often works better)
# loss = losses.MultipleNegativesRankingLoss(model=model)

# Training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
)

# Create trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=hard_train_dataset,
    loss=loss,
)

# Train
trainer.train()

# Save
model.save("./finetuned_model/final")
```

### Part 3: Evaluate Before/After

```python
import chromadb
from chromadb.utils import embedding_functions

def evaluate_model(model_name: str, test_queries: list, ground_truth: dict) -> dict:
    """Evaluate retrieval performance with a given model."""
    
    # Create embedding function
    if model_name.startswith("./"):
        # Local fine-tuned model
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    else:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    
    # Create fresh collection
    client = chromadb.Client()
    collection = client.create_collection(
        name=f"eval_{model_name.replace('/', '_')}",
        embedding_function=ef
    )
    
    # Index documents
    collection.add(documents=documents, ids=doc_ids)
    
    # Evaluate
    metrics = {"precision@5": [], "recall@5": [], "mrr": []}
    
    for query_id, query_text in test_queries.items():
        results = collection.query(query_texts=[query_text], n_results=10)
        retrieved = results["ids"][0]
        relevant = ground_truth[query_id]
        
        metrics["precision@5"].append(precision_at_k(retrieved, relevant, 5))
        metrics["recall@5"].append(recall_at_k(retrieved, relevant, 5))
        metrics["mrr"].append(mean_reciprocal_rank(retrieved, relevant))
    
    return {k: sum(v)/len(v) for k, v in metrics.items()}

# Compare models
baseline = evaluate_model("all-MiniLM-L6-v2", test_queries, ground_truth)
finetuned = evaluate_model("./finetuned_model/final", test_queries, ground_truth)

print("=== Baseline (all-MiniLM-L6-v2) ===")
for metric, value in baseline.items():
    print(f"{metric}: {value:.3f}")

print("\n=== Fine-tuned ===")
for metric, value in finetuned.items():
    improvement = ((value - baseline[metric]) / baseline[metric]) * 100
    print(f"{metric}: {value:.3f} ({improvement:+.1f}%)")
```

## Alternative: Re-ranker Evaluation

If fine-tuning is too resource-intensive, compare semantic search vs Cohere re-ranking:

```python
import cohere

co = cohere.Client("your_api_key")

def retrieve_and_rerank(query: str, collection, top_k: int = 20, rerank_top: int = 5):
    """Retrieve candidates, then rerank with Cohere."""
    
    # Initial retrieval
    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results["documents"][0]
    doc_ids = results["ids"][0]
    
    # Rerank
    rerank_response = co.rerank(
        query=query,
        documents=documents,
        top_n=rerank_top,
        model="rerank-english-v3.0"
    )
    
    # Return reranked IDs
    reranked_ids = [doc_ids[r.index] for r in rerank_response.results]
    return reranked_ids

# Compare
print("Without reranking:")
baseline_metrics = evaluate_without_rerank(test_queries, ground_truth)

print("\nWith Cohere reranking:")
reranked_metrics = evaluate_with_rerank(test_queries, ground_truth)
```

## Deliverable

A training script and report showing:

1. Hard negative mining from retrieval results
2. Fine-tuning with BatchHardTripletLoss or MultipleNegativesRankingLoss
3. Before/after comparison on held-out test set
4. Improvement percentages for precision@5, recall@5, MRR

Target: 5-15% improvement on domain-specific queries.

## Bonus: Analyze Hard Negatives

```python
def analyze_hard_negatives(dataset, n_samples: int = 10):
    """Understand what makes negatives 'hard'."""
    for i in range(n_samples):
        sample = dataset[i]
        print(f"\n=== Sample {i} ===")
        print(f"Query: {sample['anchor'][:100]}...")
        print(f"Positive: {sample['positive'][:100]}...")
        print(f"Hard Negative: {sample['negative'][:100]}...")
        print("---")
        # Why is the negative confusing? Topic overlap? Keyword match?
```
