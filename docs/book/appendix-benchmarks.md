---
title: "Appendix C: Benchmarking Your RAG System"
description: "Comprehensive guide to benchmarking RAG systems including standard datasets, evaluation methodology, and running your own benchmarks."
authors:
  - Jason Liu
date: 2025-01-18
tags:
  - reference
  - benchmarking
  - evaluation
  - datasets
  - methodology
---

# Appendix C: Benchmarking Your RAG System

This appendix provides a comprehensive guide to benchmarking RAG systems. Use this to establish baselines, compare approaches, and measure improvements systematically.

---

## Why Benchmark?

!!! tip "For Product Managers"
    Benchmarking answers critical business questions:
    
    - **How does our system compare to alternatives?** Justify build vs buy decisions
    - **Are we improving?** Track progress over time
    - **Where should we invest?** Identify the weakest components
    - **What is the ROI of changes?** Quantify improvement value

!!! tip "For Engineers"
    Benchmarking provides technical clarity:
    
    - **Reproducible comparisons**: Eliminate confounding variables
    - **Component isolation**: Test retrieval separate from generation
    - **Regression detection**: Catch degradation before production
    - **Architecture decisions**: Data-driven technology choices

---

## Standard Datasets

### BEIR (Benchmarking IR)

BEIR is a heterogeneous benchmark covering 18 datasets across diverse domains.

| Dataset | Domain | Queries | Documents | Task |
|---------|--------|---------|-----------|------|
| MS MARCO | Web | 6,980 | 8.8M | Passage retrieval |
| TREC-COVID | Biomedical | 50 | 171K | Scientific search |
| NFCorpus | Nutrition | 323 | 3.6K | Expert search |
| NQ | Wikipedia | 3,452 | 2.7M | Question answering |
| HotpotQA | Wikipedia | 7,405 | 5.2M | Multi-hop QA |
| FiQA | Finance | 648 | 57K | Financial QA |
| ArguAna | Arguments | 1,406 | 8.7K | Argument retrieval |
| Touche-2020 | Arguments | 49 | 382K | Argument search |
| CQADupStack | StackExchange | 13,145 | 457K | Duplicate detection |
| Quora | Social | 10,000 | 523K | Duplicate detection |
| DBPedia | Wikipedia | 400 | 4.6M | Entity search |
| SCIDOCS | Scientific | 1,000 | 25K | Citation prediction |
| FEVER | Wikipedia | 6,666 | 5.4M | Fact verification |
| Climate-FEVER | Wikipedia | 1,535 | 5.4M | Climate claims |
| SciFact | Scientific | 300 | 5K | Scientific claims |

**When to use BEIR**: Evaluating general-purpose embedding models, comparing retrieval approaches across domains.

### MS MARCO

The most widely used passage retrieval benchmark.

| Split | Queries | Relevant Passages |
|-------|---------|-------------------|
| Train | 502,939 | ~1 per query |
| Dev | 6,980 | ~1 per query |
| Eval | 6,837 | Hidden |

**Characteristics**:
- Real Bing queries
- Sparse relevance labels (typically 1 relevant passage per query)
- Large corpus (8.8M passages)

**When to use MS MARCO**: Training and evaluating passage retrieval models, especially for web-style queries.

### Domain-Specific Datasets

| Domain | Dataset | Description |
|--------|---------|-------------|
| Legal | CaseHOLD | Legal case holdings |
| Medical | PubMedQA | Biomedical question answering |
| Code | CodeSearchNet | Code retrieval |
| Finance | FiQA | Financial opinion QA |
| Scientific | SCIDOCS | Scientific document retrieval |

!!! tip "For Product Managers"
    **Choosing the right benchmark**: Select datasets that match your domain. A legal RAG system should prioritize CaseHOLD over MS MARCO. Generic benchmarks show general capability; domain benchmarks show production relevance.

---

## Benchmark Methodology

### Experimental Setup

```python
from dataclasses import dataclass
from typing import Callable
import time
import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    dataset_path: str
    embedding_model: str
    retrieval_k: list[int]  # e.g., [1, 5, 10, 20]
    num_runs: int = 3  # For statistical significance
    warmup_queries: int = 100


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    metrics: dict[str, float]  # metric_name -> value
    latencies: list[float]  # Per-query latencies
    timestamp: str


def run_benchmark(
    config: BenchmarkConfig,
    retriever: Callable,
    dataset: dict,
) -> BenchmarkResult:
    """Run a complete benchmark evaluation."""
    
    # Warmup
    for query in dataset["queries"][:config.warmup_queries]:
        _ = retriever(query["text"], k=max(config.retrieval_k))
    
    # Collect results
    all_results = []
    latencies = []
    
    for query in dataset["queries"]:
        start = time.perf_counter()
        retrieved = retriever(query["text"], k=max(config.retrieval_k))
        latencies.append(time.perf_counter() - start)
        
        all_results.append({
            "query_id": query["id"],
            "retrieved": retrieved,
            "relevant": query["relevant_docs"],
        })
    
    # Calculate metrics
    metrics = calculate_metrics(all_results, config.retrieval_k)
    metrics["latency_p50"] = np.percentile(latencies, 50)
    metrics["latency_p95"] = np.percentile(latencies, 95)
    metrics["latency_p99"] = np.percentile(latencies, 99)
    
    return BenchmarkResult(
        config=config,
        metrics=metrics,
        latencies=latencies,
        timestamp=datetime.now().isoformat(),
    )
```

### Metrics Calculation

```python
def calculate_metrics(
    results: list[dict],
    k_values: list[int],
) -> dict[str, float]:
    """Calculate retrieval metrics at multiple k values."""
    
    metrics = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        reciprocal_ranks = []
        
        for result in results:
            retrieved_k = set(result["retrieved"][:k])
            relevant = set(result["relevant"])
            
            # Precision@k
            if len(retrieved_k) > 0:
                precision = len(retrieved_k & relevant) / len(retrieved_k)
            else:
                precision = 0.0
            precisions.append(precision)
            
            # Recall@k
            if len(relevant) > 0:
                recall = len(retrieved_k & relevant) / len(relevant)
            else:
                recall = 1.0  # No relevant docs means perfect recall
            recalls.append(recall)
            
            # Reciprocal rank
            rr = 0.0
            for i, doc_id in enumerate(result["retrieved"][:k], 1):
                if doc_id in relevant:
                    rr = 1.0 / i
                    break
            reciprocal_ranks.append(rr)
        
        metrics[f"precision@{k}"] = np.mean(precisions)
        metrics[f"recall@{k}"] = np.mean(recalls)
        metrics[f"mrr@{k}"] = np.mean(reciprocal_ranks)
    
    # NDCG calculation
    for k in k_values:
        ndcg_scores = []
        for result in results:
            ndcg = calculate_ndcg(
                result["retrieved"][:k],
                result["relevant"],
                k,
            )
            ndcg_scores.append(ndcg)
        metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)
    
    return metrics


def calculate_ndcg(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Calculate NDCG@k for a single query."""
    
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / np.log2(i + 1)
    
    # Ideal DCG
    ideal_rels = [1.0] * min(len(relevant), k)
    ideal_rels.extend([0.0] * (k - len(ideal_rels)))
    
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, 1):
        idcg += rel / np.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg
```

### Statistical Significance

```python
from scipy import stats


def compare_systems(
    results_a: list[BenchmarkResult],
    results_b: list[BenchmarkResult],
    metric: str,
    alpha: float = 0.05,
) -> dict:
    """Compare two systems with statistical significance testing."""
    
    scores_a = [r.metrics[metric] for r in results_a]
    scores_b = [r.metrics[metric] for r in results_b]
    
    # Paired t-test (if same queries)
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Effect size (Cohen's d)
    diff = np.array(scores_a) - np.array(scores_b)
    cohens_d = np.mean(diff) / np.std(diff)
    
    # Confidence interval for difference
    mean_diff = np.mean(diff)
    se = stats.sem(diff)
    ci = stats.t.interval(
        1 - alpha,
        len(diff) - 1,
        loc=mean_diff,
        scale=se,
    )
    
    return {
        "mean_a": np.mean(scores_a),
        "mean_b": np.mean(scores_b),
        "difference": mean_diff,
        "p_value": p_value,
        "significant": p_value < alpha,
        "cohens_d": cohens_d,
        "confidence_interval": ci,
    }
```

---

## Running Your Own Benchmarks

### Step 1: Define Your Evaluation Set

!!! tip "For Product Managers"
    Work with domain experts to create evaluation queries that represent real user needs:
    
    1. **Sample production queries** (if available)
    2. **Interview users** about their search patterns
    3. **Identify edge cases** that matter for your domain
    4. **Balance query types** (simple lookups, complex reasoning, multi-hop)

```python
@dataclass
class EvaluationQuery:
    """A single evaluation query with ground truth."""
    id: str
    text: str
    relevant_docs: list[str]  # Document IDs
    category: str  # For segmented analysis
    difficulty: str  # easy, medium, hard
    source: str  # production, synthetic, expert


def create_evaluation_set(
    queries: list[dict],
    documents: list[dict],
    labeling_strategy: str = "expert",
) -> list[EvaluationQuery]:
    """Create an evaluation set with relevance labels."""
    
    evaluation_queries = []
    
    for query in queries:
        if labeling_strategy == "expert":
            # Manual labeling by domain experts
            relevant = get_expert_labels(query, documents)
        elif labeling_strategy == "synthetic":
            # Generate queries from documents
            relevant = [query["source_doc"]]
        elif labeling_strategy == "click":
            # Use click data as proxy for relevance
            relevant = get_clicked_docs(query["id"])
        
        evaluation_queries.append(EvaluationQuery(
            id=query["id"],
            text=query["text"],
            relevant_docs=relevant,
            category=query.get("category", "unknown"),
            difficulty=query.get("difficulty", "medium"),
            source=labeling_strategy,
        ))
    
    return evaluation_queries
```

### Step 2: Establish Baselines

```python
def establish_baselines(
    evaluation_set: list[EvaluationQuery],
    documents: list[dict],
) -> dict[str, BenchmarkResult]:
    """Run baseline retrievers for comparison."""
    
    baselines = {}
    
    # BM25 baseline
    bm25_retriever = create_bm25_retriever(documents)
    baselines["bm25"] = run_benchmark(
        config=BenchmarkConfig(name="BM25", ...),
        retriever=bm25_retriever,
        dataset=evaluation_set,
    )
    
    # Dense retrieval baseline
    dense_retriever = create_dense_retriever(
        documents,
        model="all-MiniLM-L6-v2",
    )
    baselines["dense_minilm"] = run_benchmark(
        config=BenchmarkConfig(name="Dense-MiniLM", ...),
        retriever=dense_retriever,
        dataset=evaluation_set,
    )
    
    # OpenAI embeddings baseline
    openai_retriever = create_dense_retriever(
        documents,
        model="text-embedding-3-small",
    )
    baselines["openai_small"] = run_benchmark(
        config=BenchmarkConfig(name="OpenAI-Small", ...),
        retriever=openai_retriever,
        dataset=evaluation_set,
    )
    
    return baselines
```

### Step 3: Run Comparative Experiments

```python
def run_experiment(
    name: str,
    retriever: Callable,
    evaluation_set: list[EvaluationQuery],
    baselines: dict[str, BenchmarkResult],
    num_runs: int = 3,
) -> dict:
    """Run an experiment and compare to baselines."""
    
    # Run multiple times for statistical significance
    results = []
    for run in range(num_runs):
        result = run_benchmark(
            config=BenchmarkConfig(name=name, ...),
            retriever=retriever,
            dataset=evaluation_set,
        )
        results.append(result)
    
    # Compare to each baseline
    comparisons = {}
    for baseline_name, baseline_result in baselines.items():
        comparison = compare_systems(
            results,
            [baseline_result] * num_runs,
            metric="ndcg@10",
        )
        comparisons[baseline_name] = comparison
    
    return {
        "results": results,
        "comparisons": comparisons,
        "summary": summarize_experiment(results, comparisons),
    }
```

### Step 4: Analyze Results by Segment

```python
def analyze_by_segment(
    results: list[dict],
    evaluation_set: list[EvaluationQuery],
) -> dict[str, dict]:
    """Analyze performance by query category."""
    
    # Group queries by category
    categories = {}
    for query in evaluation_set:
        if query.category not in categories:
            categories[query.category] = []
        categories[query.category].append(query.id)
    
    # Calculate metrics per category
    segment_metrics = {}
    for category, query_ids in categories.items():
        category_results = [
            r for r in results
            if r["query_id"] in query_ids
        ]
        segment_metrics[category] = calculate_metrics(
            category_results,
            k_values=[1, 5, 10],
        )
    
    return segment_metrics
```

---

## Benchmark Reporting

### Standard Report Format

```python
def generate_benchmark_report(
    experiment_name: str,
    results: list[BenchmarkResult],
    baselines: dict[str, BenchmarkResult],
    segment_analysis: dict,
) -> str:
    """Generate a standardized benchmark report."""
    
    report = f"""
# Benchmark Report: {experiment_name}

## Summary

| Metric | Value | vs BM25 | vs Dense |
|--------|-------|---------|----------|
| NDCG@10 | {results[0].metrics['ndcg@10']:.3f} | +{delta_bm25:.1%} | +{delta_dense:.1%} |
| Recall@10 | {results[0].metrics['recall@10']:.3f} | +{delta_bm25_r:.1%} | +{delta_dense_r:.1%} |
| P95 Latency | {results[0].metrics['latency_p95']*1000:.1f}ms | - | - |

## Statistical Significance

All improvements significant at p < 0.05 (paired t-test).

## Segment Analysis

| Category | NDCG@10 | Recall@10 | Count |
|----------|---------|-----------|-------|
{segment_table}

## Recommendations

{generate_recommendations(results, segment_analysis)}
"""
    return report
```

!!! tip "For Product Managers"
    **Key metrics to track**:
    
    - **NDCG@10**: Overall ranking quality
    - **Recall@10**: Coverage of relevant documents
    - **P95 Latency**: User experience impact
    - **Segment performance**: Where are we weakest?
    
    **Questions to answer**:
    
    1. Are we better than baselines? By how much?
    2. Is the improvement statistically significant?
    3. Which query types benefit most/least?
    4. What is the latency cost of improvements?

---

## Common Pitfalls

!!! warning "PM Pitfall"
    **Benchmark shopping**: Choosing benchmarks that make your system look good rather than benchmarks that reflect your use case. Always include domain-relevant benchmarks alongside standard ones.

!!! warning "Engineering Pitfall"
    **Overfitting to benchmarks**: Optimizing specifically for benchmark queries rather than general retrieval quality. Use held-out test sets and production sampling to detect this.

!!! warning "PM Pitfall"
    **Ignoring latency**: A 10% NDCG improvement that doubles latency may not be worth it. Always report latency alongside quality metrics.

!!! warning "Engineering Pitfall"
    **Single-run comparisons**: Running each system once and comparing results. Always run multiple times and report statistical significance.

---

## Quick Reference

### Minimum Viable Benchmark

For quick comparisons, use this minimal setup:

1. **100+ evaluation queries** with relevance labels
2. **BM25 baseline** (always include)
3. **One dense baseline** (e.g., all-MiniLM-L6-v2)
4. **3 runs** for statistical significance
5. **Report NDCG@10, Recall@10, P95 latency**

### Comprehensive Benchmark

For thorough evaluation:

1. **500+ evaluation queries** across categories
2. **Multiple baselines** (BM25, 2-3 dense models)
3. **5+ runs** per configuration
4. **Segment analysis** by query type
5. **Statistical significance testing**
6. **Latency profiling** at multiple percentiles

---

## Navigation

**Previous**: [Appendix B: Algorithms Reference](appendix-algorithms.md) - Algorithm pseudocode and complexity

**Next**: [Appendix D: Debugging RAG Systems](appendix-debugging.md) - Systematic debugging methodology

**Related Chapters**:

- [Chapter 1: Evaluation-First Development](chapter1.md) - Evaluation fundamentals
- [Chapter 2: Training Data and Fine-Tuning](chapter2.md) - Measuring fine-tuning impact
