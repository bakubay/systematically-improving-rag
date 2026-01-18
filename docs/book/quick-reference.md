---
title: Quick Reference
description: One-page reference for key metrics, formulas, decision frameworks, and checklists from the RAG improvement book series
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - reference
  - cheatsheet
  - metrics
  - formulas
---

# Quick Reference

A condensed reference for the key concepts, metrics, decision frameworks, and checklists from the book. Use this as a quick lookup when building and improving RAG systems.

---

## Chapter Summaries

### Chapter 0: Introduction - The Product Mindset

**Core Concept**: Treat RAG as a product that improves continuously, not a project that ships once.

**Key Takeaways**:

- The improvement flywheel: Measure → Analyze → Improve → Deploy → Repeat
- Distinguish inventory problems (missing data) from capability problems (cannot find existing data)
- Embeddings capture meaning; vector databases enable fast similarity search
- Semantic search finds meaning; lexical search finds exact terms; hybrid combines both

### Chapter 1: Evaluation-First Development

**Core Concept**: You cannot improve what you cannot measure—and you can measure before you have users.

**Key Takeaways**:

- Leading metrics (experiment velocity) predict success; lagging metrics (satisfaction) measure it
- Prioritize recall over precision with modern LLMs—they handle irrelevant context well
- Synthetic data bootstraps evaluation before real users arrive
- Statistical significance requires proper sample sizes (typically 200-400 examples)

### Chapter 2: Training Data and Fine-Tuning

**Core Concept**: Fine-tune embedding models (cheap, fast) not language models (expensive, complex).

**Key Takeaways**:

- Bi-encoders are fast (precomputed); cross-encoders are accurate (computed per pair)
- Re-rankers give 12-20% improvement with no training; fine-tuning gives 6-10% additional improvement
- Hard negatives are the most valuable training examples—mine them from retrieval failures
- 6,000+ examples enable effective fine-tuning; fewer examples work better for re-rankers

### Chapter 3: Feedback Systems and UX

**Core Concept**: Feedback is the fuel for the improvement flywheel—collect it intentionally.

**Key Takeaways**:

- Specific feedback prompts ("Did we answer your question?") get 5x more responses
- Implicit signals (query refinement, abandonment) reveal failures explicit feedback misses
- Streaming reduces perceived latency; citations build trust
- Negative feedback requires follow-up to be actionable

### Chapter 4: Query Understanding and Prioritization

**Core Concept**: Not all queries are equal—prioritize improvements by volume, satisfaction gap, and strategic value.

**Key Takeaways**:

- Cluster queries by embedding similarity to discover patterns
- Prioritization score = Volume x (1 - Satisfaction) x Achievable Delta x Strategic Relevance
- High volume + low satisfaction = fix first; low volume + high satisfaction = maintain
- Topic modeling reveals what users actually ask about vs what you expected

### Chapter 5: Specialized Retrieval Systems

**Core Concept**: One retriever cannot excel at everything—build specialized systems for different content types.

**Key Takeaways**:

- RAPTOR creates hierarchical summaries for long documents
- Metadata extraction enables filtering before semantic search
- Synthetic text generation describes non-text content (images, tables) for embedding
- Multimodal retrieval requires unified or specialized embedding models

### Chapter 6: Query Routing and Orchestration

**Core Concept**: Success = P(selecting right retriever) x P(retriever finding data).

**Key Takeaways**:

- Few-shot classification: 10 examples = 85%, 40 examples = 95% accuracy
- Three router architectures: classifier-based (fast), embedding-based (flexible), LLM-based (powerful)
- Tools-as-APIs pattern enables parallel team development
- Avoid data leakage: never include test examples in few-shot prompts

### Chapter 7: Production Operations

**Core Concept**: Shipping is the starting line—production success requires cost-aware design and graceful degradation.

**Key Takeaways**:

- LLM generation is 60-75% of costs; optimize context size first
- Write-time computation for stable content; read-time for dynamic content
- Semantic caching returns similar (not just identical) query results
- Monitor retrieval metrics, not just generation quality

### Chapter 8: Hybrid Search

**Core Concept**: Semantic search fails on exact terms and rare vocabulary—hybrid search combines the best of both.

**Key Takeaways**:

- BM25 excels at exact matches, rare terms, and specific identifiers
- Reciprocal Rank Fusion (RRF) combines results without score normalization
- Typical hybrid improvement: 10-25% over semantic-only
- Start with equal weights (0.5/0.5), then tune based on evaluation

### Chapter 9: Context Window Management

**Core Concept**: Models pay less attention to information in the middle—position matters.

**Key Takeaways**:

- "Lost in the Middle" effect: models attend to beginning and end more than middle
- Token budgeting: allocate fixed portions to system prompt, context, history, generation
- Dynamic context assembly: build context at query time based on relevance
- Summarization reduces tokens while preserving key information

---

## Core Metrics

### Retrieval Metrics

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **Precision@K** | Relevant in top K / K | Are your results relevant? |
| **Recall@K** | Relevant in top K / Total relevant | Are you finding everything? |
| **F1 Score** | 2 x (P x R) / (P + R) | Balance of precision and recall |
| **MRR** | 1 / Rank of first relevant | How quickly do you find something useful? |
| **NDCG@K** | DCG@K / IDCG@K | Quality of ranking with graded relevance |
| **MAP** | Mean of average precision per query | Overall ranking quality |

**Rule of thumb**: With modern LLMs, prioritize recall over precision. They handle irrelevant context well.

### System Performance

| Metric | Formula | Target |
|--------|---------|--------|
| **End-to-end success** | P(router correct) x P(retrieval correct) | 75%+ |
| **Feedback rate** | Feedback submissions / Total queries | 0.5%+ (5x better than typical) |
| **Experiment velocity** | Experiments run per week | 5-10 for early systems |
| **Cache hit rate** | Cached responses / Total queries | 20-40% for semantic cache |

### Typical Performance Benchmarks

| Metric | Typical | Good | Excellent |
|--------|---------|------|-----------|
| Feedback rate | 0.1% | 0.5% | 2%+ |
| Recall@10 | 50% | 75% | 90%+ |
| Router accuracy | 70% | 90% | 95%+ |
| Re-ranker improvement | 5% | 12% | 20%+ |
| Fine-tuning improvement | 3% | 6% | 10%+ |
| Hard negative boost | 6% | 15% | 30%+ |

---

## Decision Frameworks

### Is It an Inventory Problem or Capability Problem?

```
Can a human expert find the answer by manually searching?
    |
    +-- NO --> Inventory Problem
    |          Fix: Add missing content
    |
    +-- YES --> Capability Problem
               Fix: Improve retrieval/routing
```

### Should You Fine-tune or Use a Re-ranker?

```
Do you have 6,000+ labeled examples?
    |
    +-- NO --> Use re-ranker (12-20% improvement, no training needed)
    |
    +-- YES --> Do you have hard negatives?
                    |
                    +-- NO --> Mine hard negatives first, then fine-tune
                    |
                    +-- YES --> Fine-tune embeddings (6-10% improvement)
```

### Bi-encoder vs Cross-encoder Selection

```
Is latency critical (<100ms)?
    |
    +-- YES --> Bi-encoder only
    |
    +-- NO --> Is precision critical (legal, medical)?
                    |
                    +-- YES --> Bi-encoder + Cross-encoder re-ranking
                    |
                    +-- NO --> Bi-encoder with optional re-ranking
```

### Write-time vs Read-time Computation

| Factor | Write-time (Preprocess) | Read-time (On-demand) |
|--------|------------------------|----------------------|
| Content changes | Rarely | Frequently |
| Latency requirements | Strict (<100ms) | Flexible (1-2s OK) |
| Storage budget | Available | Constrained |
| Query patterns | Predictable | Unpredictable |

### Hybrid Search Decision

```
Does your domain have specialized vocabulary or identifiers?
    |
    +-- YES --> Use hybrid search (semantic + BM25)
    |
    +-- NO --> Do users search for exact phrases or codes?
                    |
                    +-- YES --> Use hybrid search
                    |
                    +-- NO --> Semantic search may be sufficient
```

### Vector Database Selection

```
Do you have existing PostgreSQL expertise?
    |
    +-- YES --> Is your dataset < 1M vectors?
    |               |
    |               +-- YES --> pgvector
    |               +-- NO --> pgvector_scale or migrate
    |
    +-- NO --> Do you want managed infrastructure?
                    |
                    +-- YES --> Pinecone
                    |
                    +-- NO --> Want hybrid search experiments?
                                    |
                                    +-- YES --> LanceDB
                                    +-- NO --> ChromaDB (prototypes) or Turbopuffer (performance)
```

---

## Key Formulas

### Retrieval Metrics

| Metric | Formula |
|--------|---------|
| Precision@K | `relevant_in_top_k / k` |
| Recall@K | `relevant_in_top_k / total_relevant` |
| F1 | `2 * (precision * recall) / (precision + recall)` |
| MRR | `mean(1 / rank_of_first_relevant)` |
| Cosine Similarity | `(A · B) / (||A|| × ||B||)` |

### System Performance

| Metric | Formula |
|--------|---------|
| End-to-end success | `P(router_correct) × P(retrieval_correct)` |
| Prioritization score | `Volume × (1 - Satisfaction) × Delta × Relevance` |
| RRF score | `Σ 1/(k + rank_i(d))` where k=60 typically |

### Statistical Testing

| Calculation | Formula |
|-------------|---------|
| Sample size (proportions) | `n = (z² × p × (1-p)) / e²` |
| Confidence interval | `p ± z × sqrt(p(1-p)/n)` |
| Chi-square statistic | `Σ (observed - expected)² / expected` |

### Cost Estimation

```
Monthly cost = 
    (Documents × Tokens/doc × Embedding cost)           # One-time
  + (Queries/day × 30 × Input tokens × Input cost)      # Recurring
  + (Queries/day × 30 × Output tokens × Output cost)    # Recurring
  + Infrastructure                                       # Fixed
```

---

## Cost Optimization

### Typical Cost Breakdown

| Component | Percentage | Optimization Potential |
|-----------|------------|----------------------|
| Embedding generation | 5-10% | Medium (batch, cache) |
| Retrieval infrastructure | 10-20% | High (right-size, cache) |
| LLM generation | 60-75% | High (context size, caching) |
| Logging/monitoring | 5-10% | Low (sample, aggregate) |

### Cost Reduction Techniques

| Technique | Typical Savings | Complexity |
|-----------|----------------|------------|
| Prompt caching | 70-90% on repeat queries | Low |
| Semantic caching | 20-30% | Medium |
| Self-hosted embeddings | 50-80% on embedding costs | High |
| Smaller context windows | 30-50% on generation | Low |
| Batch processing | 20-40% on embeddings | Low |

---

## Prioritization Matrix

### The 2x2 for Query Segments

```
                    High Volume
                         |
         +---------------+---------------+
         |   DANGER      |   STRENGTH    |
         |   Fix first   |   Maintain    |
         |               |               |
Low -----+---------------+---------------+----- High
Satisfaction             |               Satisfaction
         |               |               |
         |   MONITOR     |   OPPORTUNITY |
         |   Low priority|   Expand      |
         |               |               |
         +---------------+---------------+
                         |
                    Low Volume
```

### Prioritization Score

```
Score = Volume% × (1 - Satisfaction%) × Achievable Delta × Strategic Relevance
```

**Example**: Scheduling queries are 8% of volume, 25% satisfaction, 50% achievable improvement, high strategic relevance = High priority fix

---

## Routing Performance

### Few-shot Examples Impact

| Examples | Typical Accuracy |
|----------|-----------------|
| 5 | 75-80% |
| 10 | 85-88% |
| 20 | 90-92% |
| 40 | 94-96% |

### End-to-end Impact

| Router Accuracy | Retrieval Accuracy | Overall Success |
|-----------------|-------------------|-----------------|
| 67% | 80% | 54% |
| 85% | 80% | 68% |
| 95% | 82% | 78% |
| 98% | 85% | 83% |

---

## Chunking Defaults

| Content Type | Chunk Size | Overlap | Notes |
|--------------|-----------|---------|-------|
| General text | 800 tokens | 50% | Good starting point |
| Legal/regulatory | 1500-2000 tokens | 30% | Preserve full clauses |
| Technical docs | 400-600 tokens | 40% | Precise retrieval |
| Conversations | Page-level | Minimal | Maintain context |

**Warning**: Chunk optimization rarely gives >10% improvement. Focus on query understanding and metadata filtering first.

---

## Feedback Copy That Works

### Do Use

- "Did we answer your question?" (5x better than generic)
- "Did this run do what you expected?"
- "Was this information helpful for your task?"

### Do Not Use

- "How did we do?" (too vague)
- "Rate your experience" (users think you mean UI)
- "Was this helpful?" (without context)

### After Negative Feedback

Ask specific follow-up:

- "Was the information wrong?"
- "Was something missing?"
- "Was it hard to understand?"

---

## Production Checklists

### Before Launch

- [ ] Baseline metrics established (Recall@5, Precision@5)
- [ ] 50+ evaluation examples covering main query types
- [ ] Feedback mechanism visible and specific
- [ ] Error handling and fallbacks implemented
- [ ] Cost monitoring in place
- [ ] Graceful degradation tested

### Weekly Review

- [ ] Check retrieval metrics for degradation
- [ ] Review negative feedback submissions
- [ ] Analyze new query patterns
- [ ] Run at least 2 experiments
- [ ] Update evaluation set with edge cases
- [ ] Review cost trends

### Monthly Review

- [ ] Cost trend analysis
- [ ] Query segment performance comparison
- [ ] Model/embedding update evaluation
- [ ] Roadmap prioritization refresh
- [ ] Review routing accuracy
- [ ] Update training data with new examples

---

## Common Pitfalls by Role

### PM Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Vague metrics | "Make it better" | Define specific, measurable targets |
| Premature optimization | Tweaking before measuring | Establish baselines first |
| Ignoring retrieval | Focus only on generation | Measure retrieval separately |
| Underinvesting in feedback | Low response rates | Specific prompts, strategic placement |

### Engineering Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Data leakage | Inflated test metrics | Separate train/test splits |
| Absence blindness | Missing retrieval failures | Log and review retrieval results |
| Over-engineering | Complex systems, slow iteration | Start simple, add complexity as needed |
| Ignoring hard negatives | Slow improvement | Mine failures for training data |

---

## Quick Lookup: Key Numbers

| What | Value | Source |
|------|-------|--------|
| Minimum evaluation examples | 50 | Chapter 1 |
| Statistical significance sample | 200-400 | Chapter 1 |
| Fine-tuning minimum examples | 6,000 | Chapter 2 |
| Few-shot examples for 90% routing | 20 | Chapter 6 |
| Typical re-ranker improvement | 12-20% | Chapter 2 |
| Typical fine-tuning improvement | 6-10% | Chapter 2 |
| Typical hybrid search improvement | 10-25% | Chapter 8 |
| Target feedback rate | 0.5%+ | Chapter 3 |
| LLM cost percentage | 60-75% | Chapter 7 |
| Semantic cache hit rate target | 20-40% | Chapter 7 |

---

## Navigation

- **Reference**: [Glossary](glossary.md) | [How to Use This Book](how-to-use.md)
- **Book Index**: [Book Overview](index.md)
