---
title: Hands-On Exercises
description: Practical exercises to apply workshop concepts to your own RAG system
authors:
  - Jason Liu
date: 2025-04-18
tags:
  - exercises
  - practice
  - hands-on
---

# Hands-On Exercises

These exercises help you apply workshop concepts to your own RAG system. Each exercise includes clear objectives, step-by-step instructions, and expected outcomes.

---

## Chapter 1: Evaluation Foundations

### Exercise 1.1: Build Your First Evaluation Set

**Objective**: Create 20 evaluation examples for your RAG system.

**Time**: 1-2 hours

**Steps**:

1. Select 10 representative documents from your corpus
2. For each document, write 2 questions it should answer
3. Record the expected document(s) for each question
4. Format as JSON:

```json
{
  "question": "What is the refund policy for digital products?",
  "expected_docs": ["policies/refunds.md"],
  "difficulty": "easy",
  "category": "policy"
}
```

**Success criteria**: You have 20 question-document pairs covering at least 3 different query types.

---

### Exercise 1.2: Measure Baseline Recall

**Objective**: Establish your current retrieval performance.

**Time**: 30 minutes

**Steps**:

1. Run your 20 evaluation questions through your retrieval system
2. For each question, check if the expected document appears in top-5 results
3. Calculate Recall@5 = (questions where expected doc found) / 20

```python
def calculate_recall_at_k(results, k=5):
    found = 0
    for item in results:
        retrieved_ids = [doc['id'] for doc in item['retrieved'][:k]]
        if any(expected in retrieved_ids for expected in item['expected_docs']):
            found += 1
    return found / len(results)
```

**Success criteria**: You have a baseline Recall@5 number (e.g., "Our current system achieves 65% Recall@5").

---

### Exercise 1.3: Generate Synthetic Questions

**Objective**: Expand your evaluation set using LLM-generated questions.

**Time**: 1 hour

**Steps**:

1. Select 5 documents you haven't used yet
2. Use this prompt to generate questions:

```text
Given this document:
[DOCUMENT TEXT]

Generate 3 questions that this document answers:
1. A factual question about specific information
2. A question requiring inference from the content
3. A question using different terminology than the document

For each question, explain why this document is the correct answer.
```

3. Validate that your system can retrieve the source document for each question
4. Add passing questions to your evaluation set

**Success criteria**: You have 15+ additional synthetic questions with validated retrievability.

---

## Chapter 2: Fine-Tuning Foundations

### Exercise 2.1: Create Training Triplets

**Objective**: Build a dataset of (query, positive, negative) triplets for embedding fine-tuning.

**Time**: 1-2 hours

**Steps**:

1. Take your evaluation questions from Exercise 1.1
2. For each question, identify:
   - **Positive**: The correct document
   - **Easy negative**: A completely unrelated document
   - **Hard negative**: A document that seems related but doesn't answer the question

```python
triplet = {
    "query": "What is the refund policy?",
    "positive": "policies/refunds.md",
    "easy_negative": "blog/company-history.md",
    "hard_negative": "policies/returns.md"  # Related but different
}
```

**Success criteria**: You have 20+ triplets with at least 10 hard negatives.

---

### Exercise 2.2: Test a Re-ranker

**Objective**: Measure the impact of adding a re-ranker to your pipeline.

**Time**: 1 hour

**Steps**:

1. Install a re-ranker (Cohere, cross-encoder, etc.)
2. Modify your retrieval to:
   - Retrieve top-50 documents
   - Re-rank to top-10
3. Re-run your evaluation set
4. Compare Recall@10 before and after re-ranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, documents, top_k=10):
    pairs = [(query, doc['text']) for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]
```

**Success criteria**: You can quantify the re-ranker's impact (e.g., "Re-ranking improved Recall@10 from 72% to 84%").

---

## Chapter 3: Feedback Collection

### Exercise 3.1: Audit Your Feedback Copy

**Objective**: Identify and improve feedback request language.

**Time**: 30 minutes

**Steps**:

1. Screenshot your current feedback UI
2. List all feedback-related text (buttons, prompts, follow-ups)
3. For each piece of text, rate it:
   - Is it specific to the task? (not generic "How did we do?")
   - Is it visible? (not hidden in a corner)
   - Is it actionable? (can you improve based on responses?)
4. Rewrite any text that scores poorly

**Before/After example**:

- Before: "Rate this response"
- After: "Did this answer your question? [Yes] [Partially] [No]"

**Success criteria**: All feedback copy is specific, visible, and actionable.

---

### Exercise 3.2: Implement Implicit Signal Tracking

**Objective**: Capture user behavior signals beyond explicit feedback.

**Time**: 2 hours

**Steps**:

1. Identify 3 implicit signals relevant to your application:
   - Query refinement (user rephrases immediately)
   - Citation clicks (which sources users check)
   - Copy actions (what users copy to clipboard)
   - Abandonment (user leaves without action)

2. Add logging for each signal:

```python
def log_implicit_signal(session_id, signal_type, data):
    event = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "signal_type": signal_type,  # "refinement", "citation_click", "copy", "abandon"
        "data": data
    }
    # Store in your logging system
```

3. After 1 week, analyze the signals

**Success criteria**: You're capturing at least 3 implicit signals and can query them for analysis.

---

## Chapter 4: Query Segmentation

### Exercise 4.1: Manual Query Clustering

**Objective**: Identify natural query categories in your data.

**Time**: 1-2 hours

**Steps**:

1. Export 100 recent queries from your system
2. Read through them and create categories as you go (open coding)
3. Assign each query to a category
4. Calculate the distribution

```python
categories = {
    "product_lookup": 35,
    "policy_questions": 28,
    "troubleshooting": 22,
    "comparison": 10,
    "other": 5
}
```

5. For each category, note:
   - Estimated satisfaction (from feedback if available)
   - Typical query patterns
   - Current system performance

**Success criteria**: You have 4-6 categories with distribution percentages and performance estimates.

---

### Exercise 4.2: Build a 2x2 Prioritization Matrix

**Objective**: Identify which query segments to improve first.

**Time**: 30 minutes

**Steps**:

1. Take your categories from Exercise 4.1
2. For each category, estimate:
   - Volume (% of total queries)
   - Satisfaction (% of positive feedback)
3. Plot on a 2x2 matrix:

```text
                    High Volume
                         │
         ┌───────────────┼───────────────┐
         │   DANGER      │   STRENGTH    │
         │   [category]  │   [category]  │
Low ─────┼───────────────┼───────────────┼───── High
Satisfaction             │               Satisfaction
         │   MONITOR     │   OPPORTUNITY │
         │   [category]  │   [category]  │
         └───────────────┼───────────────┘
                         │
                    Low Volume
```

4. Identify your top priority (high volume, low satisfaction)

**Success criteria**: You have a clear #1 priority segment with justification.

---

## Chapter 5: Specialized Retrieval

### Exercise 5.1: Identify Specialization Opportunities

**Objective**: Determine which query types need specialized retrievers.

**Time**: 1 hour

**Steps**:

1. Review your query categories from Chapter 4
2. For each category, answer:
   - Does standard semantic search work well? (>80% recall)
   - Does it need exact matching? (product IDs, codes)
   - Does it need structured data? (dates, numbers, comparisons)
   - Does it need multimodal content? (images, tables)

3. Create a specialization plan:

| Category | Current Recall | Needs | Proposed Solution |
|----------|---------------|-------|-------------------|
| Product lookup | 45% | Exact matching | Hybrid search with SKU index |
| Policy questions | 78% | - | Keep current approach |
| Troubleshooting | 52% | Step-by-step | Structured procedure index |

**Success criteria**: You have a prioritized list of specialization opportunities.

---

### Exercise 5.2: Build a Metadata Index

**Objective**: Create a specialized index using extracted metadata.

**Time**: 2-3 hours

**Steps**:

1. Choose one document type that would benefit from metadata extraction
2. Define the metadata schema:

```python
schema = {
    "product_id": str,
    "category": str,
    "price_range": str,  # "budget", "mid", "premium"
    "features": list[str]
}
```

3. Extract metadata from 50 documents (manually or with LLM)
4. Create a filtered search function:

```python
def search_with_filters(query, filters):
    # First filter by metadata
    candidates = filter_by_metadata(filters)
    # Then semantic search within candidates
    return semantic_search(query, candidates)
```

5. Test on relevant queries and measure improvement

**Success criteria**: Filtered search improves recall for the target query type by 15%+.

---

## Chapter 6: Query Routing

### Exercise 6.1: Build a Simple Router

**Objective**: Create a few-shot classifier that routes queries to tools.

**Time**: 1-2 hours

**Steps**:

1. Define your tools (from Chapter 5 specialization):

```python
tools = [
    {"name": "product_search", "description": "Find products by name, ID, or features"},
    {"name": "policy_lookup", "description": "Answer questions about policies and procedures"},
    {"name": "troubleshoot", "description": "Help diagnose and fix problems"}
]
```

2. Create 5 examples per tool:

```python
examples = [
    {"query": "What's the SKU for the blue widget?", "tool": "product_search"},
    {"query": "Can I return an opened item?", "tool": "policy_lookup"},
    # ... more examples
]
```

3. Build the router:

```python
def route_query(query, examples, tools):
    prompt = f"""Given these tools: {tools}
    
    Examples:
    {format_examples(examples)}
    
    Which tool should handle: "{query}"?
    
    Respond with just the tool name."""
    
    return llm.complete(prompt)
```

4. Test on 20 queries and measure accuracy

**Success criteria**: Router achieves 85%+ accuracy on test queries.

---

### Exercise 6.2: Measure End-to-End Performance

**Objective**: Calculate your system's overall success rate.

**Time**: 1 hour

**Steps**:

1. Run 50 queries through your full pipeline (router + retrievers)
2. For each query, record:
   - Was it routed correctly? (manual judgment)
   - Did retrieval find the right document?
3. Calculate:
   - Router accuracy = correct routes / total
   - Retrieval accuracy (per tool) = correct retrievals / queries to that tool
   - Overall = router accuracy × average retrieval accuracy

```python
results = {
    "router_accuracy": 0.92,  # 46/50 correct routes
    "retrieval_by_tool": {
        "product_search": 0.85,
        "policy_lookup": 0.78,
        "troubleshoot": 0.72
    },
    "overall": 0.92 * 0.78  # = 0.72 or 72%
}
```

**Success criteria**: You have quantified end-to-end performance and identified the limiting factor (routing vs retrieval).

---

## Chapter 7: Production Readiness

### Exercise 7.1: Cost Analysis

**Objective**: Calculate your per-query cost and identify optimization opportunities.

**Time**: 1 hour

**Steps**:

1. List all cost components:
   - Embedding API calls
   - LLM generation calls
   - Vector database queries
   - Infrastructure (servers, storage)

2. Calculate cost per query:

```python
costs = {
    "embedding": 0.0001,  # $0.0001 per query embedding
    "retrieval": 0.0005,  # Vector DB query cost
    "generation": 0.003,  # LLM generation (avg tokens)
    "infrastructure": 0.001  # Amortized server cost
}
total_per_query = sum(costs.values())  # $0.0046
```

3. Project monthly costs at different scales:

| Daily Queries | Monthly Cost |
|--------------|--------------|
| 100 | $14 |
| 1,000 | $140 |
| 10,000 | $1,400 |

4. Identify top optimization opportunities

**Success criteria**: You have a cost model and identified the top 2 cost reduction opportunities.

---

### Exercise 7.2: Build a Monitoring Dashboard

**Objective**: Create visibility into production performance.

**Time**: 2-3 hours

**Steps**:

1. Define key metrics to track:
   - Query volume (per hour/day)
   - Latency (p50, p95, p99)
   - Feedback rate (positive/negative)
   - Error rate
   - Cost per query

2. Set up logging for each metric
3. Create a simple dashboard (Grafana, custom, or spreadsheet)
4. Define alert thresholds:

```python
alerts = {
    "latency_p95": {"threshold": 3000, "unit": "ms"},
    "error_rate": {"threshold": 0.05, "unit": "ratio"},
    "feedback_rate": {"threshold": 0.001, "unit": "ratio", "direction": "below"}
}
```

**Success criteria**: You have a dashboard showing key metrics and alerts for anomalies.

---

## Capstone Exercise: Full Improvement Cycle

**Objective**: Complete one full iteration of the RAG improvement flywheel.

**Time**: 4-8 hours over 1-2 weeks

**Steps**:

1. **Measure** (Chapter 1): Establish baseline metrics
2. **Analyze** (Chapter 4): Identify lowest-performing query segment
3. **Improve** (Chapters 2, 5, 6): Implement one targeted improvement
4. **Measure Again**: Quantify the impact
5. **Document**: Write up what you learned

**Deliverable**: A one-page summary including:
- Baseline metrics
- Problem identified
- Solution implemented
- Results achieved
- Next improvement planned

**Success criteria**: You have completed one measurable improvement cycle and documented the process.

---

*Return to [Workshop Index](index.md) | [How to Use This Book](how-to-use.md)*
