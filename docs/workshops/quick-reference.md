---
title: Quick Reference
description: One-page reference for key metrics, formulas, and decision frameworks
authors:
  - Jason Liu
date: 2025-04-18
tags:
  - reference
  - cheatsheet
---

# Quick Reference

A condensed reference for the key concepts, metrics, and decision frameworks from the workshops.

---

## Core Metrics

### Retrieval Metrics

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **Precision@K** | Relevant in top K ÷ K | Are your results relevant? |
| **Recall@K** | Relevant in top K ÷ Total relevant | Are you finding everything? |
| **MRR** | 1 ÷ Rank of first relevant | How quickly do you find something useful? |

**Rule of thumb**: With modern LLMs, prioritize recall over precision. They handle irrelevant context well.

### System Performance

| Metric | Formula | Target |
|--------|---------|--------|
| **End-to-end success** | P(router correct) × P(retrieval correct) | 75%+ |
| **Feedback rate** | Feedback submissions ÷ Total queries | 0.5%+ (5x better than typical) |
| **Experiment velocity** | Experiments run per week | 5-10 for early systems |

---

## Decision Frameworks

### Is It an Inventory Problem or Capabilities Problem?

```
Can a human expert find the answer by manually searching?
    │
    ├── NO → Inventory Problem
    │        Fix: Add missing content
    │
    └── YES → Capabilities Problem
             Fix: Improve retrieval/routing
```

### Should You Fine-tune or Use a Re-ranker?

```
Do you have 5,000+ labeled examples?
    │
    ├── NO → Use re-ranker (12-20% improvement, no training needed)
    │
    └── YES → Do you have hard negatives?
                  │
                  ├── NO → Mine hard negatives first, then fine-tune
                  │
                  └── YES → Fine-tune embeddings (6-10% improvement)
```

### Write-time vs Read-time Computation

| Factor | Write-time (Preprocess) | Read-time (On-demand) |
|--------|------------------------|----------------------|
| Content changes | Rarely | Frequently |
| Latency requirements | Strict (<100ms) | Flexible (1-2s OK) |
| Storage budget | Available | Constrained |
| Query patterns | Predictable | Unpredictable |

---

## Cost Estimation

### Quick Cost Formula

```
Monthly cost = 
    (Documents × Tokens/doc × Embedding cost)           # One-time
  + (Queries/day × 30 × Input tokens × Input cost)      # Recurring
  + (Queries/day × 30 × Output tokens × Output cost)    # Recurring
  + Infrastructure                                       # Fixed
```

### Typical Cost Breakdown

- Embedding generation: 5-10%
- Retrieval infrastructure: 10-20%
- LLM generation: 60-75%
- Logging/monitoring: 5-10%

### Cost Reduction Levers

| Technique | Typical Savings | Complexity |
|-----------|----------------|------------|
| Prompt caching | 70-90% on repeat queries | Low |
| Semantic caching | 20-30% | Medium |
| Self-hosted embeddings | 50-80% on embedding costs | High |
| Smaller context windows | 30-50% on generation | Low |

---

## Prioritization Matrix

### The 2x2 for Query Segments

```
                    High Volume
                         │
         ┌───────────────┼───────────────┐
         │   DANGER      │   STRENGTH    │
         │   Fix first   │   Maintain    │
         │               │               │
Low ─────┼───────────────┼───────────────┼───── High
Satisfaction             │               Satisfaction
         │               │               │
         │   MONITOR     │   OPPORTUNITY │
         │   Low priority│   Expand      │
         │               │               │
         └───────────────┼───────────────┘
                         │
                    Low Volume
```

### Prioritization Score

```
Score = Volume% × (1 - Satisfaction%) × Achievable Delta × Strategic Relevance
```

**Example**: Scheduling queries are 8% of volume, 25% satisfaction, 50% achievable improvement, high strategic relevance → High priority fix

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

## Chunking Defaults

| Content Type | Chunk Size | Overlap | Notes |
|--------------|-----------|---------|-------|
| General text | 800 tokens | 50% | Good starting point |
| Legal/regulatory | 1500-2000 tokens | 30% | Preserve full clauses |
| Technical docs | 400-600 tokens | 40% | Precise retrieval |
| Conversations | Page-level | Minimal | Maintain context |

**Warning**: Chunk optimization rarely gives >10% improvement. Focus on query understanding and metadata filtering first.

---

## Vector Database Selection

```
Do you have existing PostgreSQL expertise?
    │
    ├── YES → Is your dataset < 1M vectors?
    │             │
    │             ├── YES → pgvector
    │             └── NO → pgvector_scale or migrate
    │
    └── NO → Do you want managed infrastructure?
                  │
                  ├── YES → Pinecone
                  │
                  └── NO → Want hybrid search experiments?
                                │
                                ├── YES → LanceDB
                                └── NO → ChromaDB (prototypes) or Turbopuffer (performance)
```

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

## Production Checklist

### Before Launch

- [ ] Baseline metrics established (Recall@5, Precision@5)
- [ ] 50+ evaluation examples covering main query types
- [ ] Feedback mechanism visible and specific
- [ ] Error handling and fallbacks implemented
- [ ] Cost monitoring in place

### Weekly Review

- [ ] Check retrieval metrics for degradation
- [ ] Review negative feedback submissions
- [ ] Analyze new query patterns
- [ ] Run at least 2 experiments
- [ ] Update evaluation set with edge cases

### Monthly Review

- [ ] Cost trend analysis
- [ ] Query segment performance comparison
- [ ] Model/embedding update evaluation
- [ ] Roadmap prioritization refresh

---

## Key Numbers to Remember

| Metric | Typical | Good | Excellent |
|--------|---------|------|-----------|
| Feedback rate | 0.1% | 0.5% | 2%+ |
| Recall@10 | 50% | 75% | 90%+ |
| Router accuracy | 70% | 90% | 95%+ |
| Re-ranker improvement | 5% | 12% | 20%+ |
| Fine-tuning improvement | 3% | 6% | 10%+ |
| Hard negative boost | 6% | 15% | 30%+ |

---

*Return to [Workshop Index](index.md) | See [Glossary](glossary.md) for term definitions*
