---
title: "Chapter 1: Evaluation-First Development"
description: "Learn how to build evaluation frameworks using synthetic data, measure retrieval with precision and recall, and establish the metrics infrastructure that powers continuous improvement."
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - evaluation
  - metrics
  - synthetic data
  - precision
  - recall
  - data flywheel
---

# Chapter 1: Evaluation-First Development

## Chapter at a Glance

**Prerequisites**: Chapter 0 (foundational concepts), basic Python, familiarity with embeddings

**What You Will Learn**:

- The difference between leading and lagging metrics and why it matters
- How to measure retrieval quality with precision and recall
- How to generate synthetic evaluation data before you have real users
- How to build evaluation infrastructure that enables rapid experimentation
- Statistical methods for confident decision-making

**Case Study Reference**: Consulting firm (50% to 90% recall), Blueprint search (27% to 85% in 4 days)

**Time to Complete**: 60-90 minutes

---

## Key Insight

**You cannot improve what you cannot measure—and you can measure before you have users.** Synthetic data is not just a stopgap until real users arrive. It is a powerful tool for establishing baselines, testing edge cases, and building the evaluation infrastructure that powers continuous improvement. Start with retrieval metrics (precision and recall), not generation quality, because they are faster, cheaper, and more objective to measure.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Distinguish between leading and lagging metrics and focus on what you can control
2. Calculate and interpret precision, recall, F1, MRR, and NDCG for retrieval systems
3. Generate diverse synthetic evaluation data that reflects real user behavior
4. Build evaluation pipelines that run automatically with every change
5. Apply statistical methods to determine if improvements are significant
6. Avoid common pitfalls like vague metrics, intervention bias, and absence blindness

---

## Introduction

In Chapter 0, we established the product mindset for RAG systems—treating them as continuously improving products rather than static implementations. We introduced the improvement flywheel and explained why systematic measurement matters more than choosing the "best" model.

This chapter puts that philosophy into practice. Before you optimize anything, you need to know what you are optimizing for. Before you claim improvement, you need to prove it with data. Before you have users, you need synthetic data to bootstrap the flywheel.

Most teams get stuck in an unproductive loop: make random changes based on intuition, get unclear results, feel frustrated, make more random changes. The solution is evaluation-first development—establishing clear metrics and measurement infrastructure before making any changes.

!!! tip "For Product Managers"
    This chapter establishes the measurement foundation for all future improvements. Focus on understanding leading vs lagging metrics, how to interpret precision and recall tradeoffs, and the ROI of evaluation infrastructure. You do not need to implement the code yourself, but understanding what the metrics mean will help you make better prioritization decisions.

!!! tip "For Engineers"
    This chapter provides the technical foundation for everything that follows. Pay close attention to the mathematical definitions, code examples, and statistical methods. You will use these concepts in every subsequent chapter. The evaluation pipeline you build here becomes the backbone of your improvement process.

---

## Core Content

### Leading vs Lagging Metrics

Understanding the difference between leading and lagging metrics fundamentally changes how you approach system improvement.

!!! tip "For Product Managers"
    **Why this distinction matters**: Lagging metrics are what you care about but cannot directly control—user satisfaction, retention, revenue. Leading metrics are what you can control that predict future performance—experiment velocity, evaluation coverage, retrieval precision.

    **The weight loss analogy**: Weight is a lagging metric. You cannot directly control it today. But you can control calories consumed and workouts completed (leading metrics). Obsessing over the scale does not help. Tracking what you eat does.

    **For RAG systems**: You cannot directly make users happy. But you can run more experiments, improve retrieval metrics, and collect more feedback. Teams that focus on experiment velocity—how many experiments they run per week—consistently outperform teams that obsess over user satisfaction scores.

    **The #1 leading metric**: Experiment velocity. Instead of asking "did the last change improve things?" ask "how can we run twice as many experiments next week?" What infrastructure would enable this? What blocks rapid testing?

!!! tip "For Engineers"
    **How to implement leading metrics**:

    | Lagging Metric | Leading Metrics |
    |----------------|-----------------|
    | User satisfaction | Feedback collection rate, response time |
    | Task completion | Retrieval recall, answer relevance |
    | Retention | Experiment velocity, evaluation coverage |
    | Revenue | Feature adoption, query success rate |

    **Tracking experiment velocity**:

    ```python
    from datetime import datetime, timedelta
    from typing import List
    from pydantic import BaseModel

    class Experiment(BaseModel):
        id: str
        hypothesis: str
        metric_target: str
        baseline_value: float
        result_value: float | None = None
        started_at: datetime
        completed_at: datetime | None = None

    def calculate_experiment_velocity(
        experiments: List[Experiment],
        window_days: int = 7
    ) -> float:
        """Calculate experiments completed per week."""
        cutoff = datetime.now() - timedelta(days=window_days)
        completed = [
            e for e in experiments
            if e.completed_at and e.completed_at > cutoff
        ]
        return len(completed) / (window_days / 7)
    ```

    **Key insight**: Teams that focus on experiment velocity often see 6-10% improvements in recall with hundreds of dollars in API calls—work that previously required tens of thousands in data labeling costs.

---

### Precision vs Recall

Before diving into formulas, let us build intuition about what precision and recall actually mean for RAG systems.

!!! tip "For Product Managers"
    **Intuitive explanation**:

    - **Recall**: Of all the documents that could answer the user's question, what percentage did we find? If there are 10 relevant documents and we found 4, that is 40% recall.
    - **Precision**: Of all the documents we returned, what percentage were actually relevant? If we returned 10 documents but only 2 were relevant, that is 20% precision.

    **The tradeoff**: You can always improve recall by returning more documents—but that hurts precision. You can always improve precision by returning fewer documents—but that hurts recall. The art is finding the right balance for your use case.

    **Business implications**:

    | High Recall Priority | High Precision Priority |
    |---------------------|------------------------|
    | Legal research (cannot miss relevant cases) | Customer support (avoid confusing users) |
    | Medical diagnosis (safety-critical) | Product search (clean results matter) |
    | Compliance audits (completeness required) | Quick answers (minimal reading required) |

    **Modern model guidance**: With GPT-4, Claude, and similar models, prioritize recall. These models are trained to handle irrelevant context well (the "needle in haystack" capability). They can filter out noise. Older or smaller models are more sensitive to low precision—they get confused by irrelevant documents.

!!! tip "For Engineers"
    **Visual intuition**:

    ```mermaid
    graph TD
        subgraph "Document Universe"
            subgraph "All Relevant Documents"
                A["Relevant &<br>Retrieved<br>(True Positives)"]
                B["Relevant but<br>Not Retrieved<br>(False Negatives)"]
            end

            subgraph "All Retrieved Documents"
                A
                C["Retrieved but<br>Not Relevant<br>(False Positives)"]
            end

            D["Not Relevant &<br>Not Retrieved<br>(True Negatives)"]
        end

        P["Precision = TP / (TP + FP)"]
        R["Recall = TP / (TP + FN)"]

        classDef relevant fill:#90EE90,stroke:#006400
        classDef retrieved fill:#ADD8E6,stroke:#00008B
        classDef both fill:#9370DB,stroke:#4B0082
        classDef neither fill:#DCDCDC,stroke:#696969
        classDef formula fill:#FFFACD,stroke:#8B8B00,stroke-width:2px

        class A both
        class B relevant
        class C retrieved
        class D neither
        class P,R formula
    ```

    **Mathematical definitions**:

    $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} = \frac{|\text{Relevant} \cap \text{Retrieved}|}{|\text{Retrieved}|}$$

    $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} = \frac{|\text{Relevant} \cap \text{Retrieved}|}{|\text{Relevant}|}$$

    **Implementation**:

    ```python
    def calculate_precision_recall(
        retrieved_ids: set[str],
        relevant_ids: set[str]
    ) -> tuple[float, float]:
        """Calculate precision and recall for a single query."""
        if not retrieved_ids:
            return 0.0, 0.0 if relevant_ids else 1.0

        true_positives = len(retrieved_ids & relevant_ids)
        precision = true_positives / len(retrieved_ids)
        recall = true_positives / len(relevant_ids) if relevant_ids else 1.0

        return precision, recall
    ```

    **Testing different K values**:

    ```python
    def evaluate_at_k(
        query: str,
        relevant_ids: set[str],
        retriever_fn,
        k_values: list[int] = [3, 5, 10, 20]
    ) -> dict[int, dict[str, float]]:
        """Evaluate precision and recall at different K values."""
        results = {}
        max_k = max(k_values)
        all_retrieved = retriever_fn(query, top_k=max_k)

        for k in k_values:
            retrieved_k = set(doc['id'] for doc in all_retrieved[:k])
            precision, recall = calculate_precision_recall(retrieved_k, relevant_ids)
            results[k] = {
                'precision': precision,
                'recall': recall,
                'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            }

        return results
    ```

    **Why score thresholds are dangerous**: Score distributions vary wildly by query type. A threshold that works for one category fails for others. Re-ranker scores are not true probabilities—do not treat a 0.5 threshold as "50% confidence." Better approach: always return top K, let the LLM filter.

---

### Evaluation Frameworks

Building a robust evaluation framework is the foundation of systematic improvement.

!!! tip "For Product Managers"
    **ROI of evaluation infrastructure**:

    - **Without evaluation**: Weeks of changes, unclear results, frustrated team, no learning
    - **With evaluation**: Clear baselines, measurable improvements, confident decisions, compound learning

    **When to invest**: Invest in evaluation infrastructure before your first optimization attempt. The cost of building evaluation (days) is far less than the cost of optimizing blindly (months).

    **What good evaluation looks like**:

    | Metric | Poor | Adequate | Good |
    |--------|------|----------|------|
    | Evaluation examples | <20 | 50-100 | 200+ |
    | Query type coverage | 1-2 types | Major types | All types |
    | Difficulty distribution | All easy | Mostly easy | Easy/medium/hard |
    | Run frequency | Manual | Weekly | Every change |

!!! tip "For Engineers"
    **Core evaluation pipeline**:

    ```python
    from typing import List, Dict, Any, Callable
    from dataclasses import dataclass
    import json
    from datetime import datetime

    @dataclass
    class EvaluationExample:
        query_id: str
        query: str
        relevant_doc_ids: set[str]
        metadata: dict[str, Any] = None

    @dataclass
    class EvaluationResult:
        query_id: str
        query: str
        precision: float
        recall: float
        mrr: float
        retrieved_ids: list[str]
        relevant_ids: set[str]

    class EvaluationPipeline:
        def __init__(
            self,
            retriever_fn: Callable[[str, int], List[Dict]],
            k: int = 10
        ):
            self.retriever_fn = retriever_fn
            self.k = k

        def evaluate_single(
            self,
            example: EvaluationExample
        ) -> EvaluationResult:
            """Evaluate a single query."""
            retrieved = self.retriever_fn(example.query, top_k=self.k)
            retrieved_ids = [doc['id'] for doc in retrieved]

            precision, recall = calculate_precision_recall(
                set(retrieved_ids),
                example.relevant_doc_ids
            )

            # Calculate MRR
            mrr = 0.0
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in example.relevant_doc_ids:
                    mrr = 1.0 / rank
                    break

            return EvaluationResult(
                query_id=example.query_id,
                query=example.query,
                precision=precision,
                recall=recall,
                mrr=mrr,
                retrieved_ids=retrieved_ids,
                relevant_ids=example.relevant_doc_ids
            )

        def evaluate_all(
            self,
            examples: List[EvaluationExample]
        ) -> Dict[str, Any]:
            """Evaluate all examples and aggregate metrics."""
            results = [self.evaluate_single(ex) for ex in examples]

            return {
                'timestamp': datetime.now().isoformat(),
                'k': self.k,
                'num_examples': len(examples),
                'avg_precision': sum(r.precision for r in results) / len(results),
                'avg_recall': sum(r.recall for r in results) / len(results),
                'avg_mrr': sum(r.mrr for r in results) / len(results),
                'detailed_results': results
            }
    ```

    **Integrating with CI/CD**:

    ```python
    def run_evaluation_check(
        baseline_file: str,
        current_results: Dict[str, Any],
        regression_threshold: float = 0.02
    ) -> bool:
        """Check if current results regress from baseline."""
        with open(baseline_file) as f:
            baseline = json.load(f)

        for metric in ['avg_precision', 'avg_recall', 'avg_mrr']:
            baseline_value = baseline[metric]
            current_value = current_results[metric]
            if current_value < baseline_value - regression_threshold:
                print(f"REGRESSION: {metric} dropped from {baseline_value:.3f} to {current_value:.3f}")
                return False

        return True
    ```

---

### Synthetic Data Generation

No user data yet? No problem. Synthetic data bootstraps the flywheel.

!!! tip "For Product Managers"
    **Business value of synthetic data**:

    - **Before launch**: Establish baselines, test edge cases, build confidence
    - **After launch**: Supplement real data, test rare scenarios, expand coverage
    - **Ongoing**: Generate adversarial examples, test new features

    **When synthetic data works well**:

    | Scenario | Synthetic Data Value |
    |----------|---------------------|
    | Pre-launch | Essential—only option |
    | Low traffic | High—supplements sparse data |
    | High traffic | Medium—tests edge cases |
    | Specialized domains | High—experts can validate |

    **Key insight from Chroma research**: Good performance on public benchmarks like MTEB does not guarantee good performance on your specific data. Custom evaluation sets from your own documents reveal the true performance of embedding models for your use case.

!!! tip "For Engineers"
    **Basic question generation**:

    ```python
    async def generate_questions_from_chunk(
        chunk: str,
        llm_client,
        num_questions: int = 3
    ) -> list[str]:
        """Generate questions that would be answered by this chunk."""
        prompt = f"""Based on this text, generate {num_questions} questions
        that could be answered using this information.

        Text: {chunk}

        Requirements:
        - Questions should be realistic (what a user might actually ask)
        - Vary the complexity (simple factual, inferential, comparative)
        - Use natural language, not formal queries

        Format: One question per line, no numbering."""

        response = await llm_client.generate(prompt)
        return [q.strip() for q in response.strip().split('\n') if q.strip()]
    ```

    **Diverse question generation with few-shot examples**:

    ```python
    async def generate_diverse_questions(
        chunk: str,
        llm_client,
        example_queries: list[str] = None,
        domain_context: str = None
    ) -> list[str]:
        """Generate diverse, realistic questions."""
        examples_section = ""
        if example_queries:
            examples_section = f"""
    Here are examples of real user questions in this domain:
    {chr(10).join(f'- {q}' for q in example_queries[:5])}

    Generate questions similar in style and intent to these examples.
    """

        context_section = ""
        if domain_context:
            context_section = f"""
    Context: {domain_context}
    """

        prompt = f"""Generate 5 diverse questions about this text passage.
    {context_section}
    {examples_section}

    Text passage:
    {chunk}

    Generate questions that:
    1. Vary in complexity (factual, inferential, comparative)
    2. Use different phrasings (questions, commands, implied questions)
    3. Reflect how real users search (sometimes incomplete or ambiguous)
    4. Target different aspects of the information

    For each question, the text passage should contain the answer.

    Format: One question per line."""

        response = await llm_client.generate(prompt)
        return [q.strip() for q in response.strip().split('\n') if q.strip()]
    ```

    **Adversarial question generation**:

    ```python
    async def generate_adversarial_questions(
        chunk: str,
        llm_client
    ) -> list[str]:
        """Generate challenging questions that test retrieval robustness."""
        prompt = f"""Given this text passage:
    {chunk}

    Generate 3 challenging questions that:
    1. Use different terminology than what appears in the passage
    2. Require understanding implications, not just matching keywords
    3. Would confuse a basic keyword search system
    4. Are still reasonable questions a user might ask

    These questions should still be answerable by the passage.

    Format: One question per line."""

        response = await llm_client.generate(prompt)
        return [q.strip() for q in response.strip().split('\n') if q.strip()]
    ```

    **Building comprehensive evaluation sets**:

    ```python
    import random
    from enum import Enum

    class Difficulty(str, Enum):
        EASY = "easy"      # Direct keyword match expected
        MEDIUM = "medium"  # Requires some semantic understanding
        HARD = "hard"      # Different terminology, inference required

    @dataclass
    class SyntheticExample:
        query: str
        source_chunk_id: str
        difficulty: Difficulty
        query_type: str  # factual, inferential, comparative, etc.

    async def build_evaluation_set(
        chunks: list[dict],
        llm_client,
        target_size: int = 200
    ) -> list[SyntheticExample]:
        """Build a balanced evaluation set."""
        examples = []

        # Sample chunks proportionally
        sample_size = min(len(chunks), target_size // 3)
        sampled_chunks = random.sample(chunks, sample_size)

        for chunk in sampled_chunks:
            # Generate easy questions (direct)
            easy_qs = await generate_questions_from_chunk(
                chunk['text'], llm_client, num_questions=1
            )
            for q in easy_qs:
                examples.append(SyntheticExample(
                    query=q,
                    source_chunk_id=chunk['id'],
                    difficulty=Difficulty.EASY,
                    query_type='factual'
                ))

            # Generate hard questions (adversarial)
            hard_qs = await generate_adversarial_questions(
                chunk['text'], llm_client
            )
            for q in hard_qs[:1]:
                examples.append(SyntheticExample(
                    query=q,
                    source_chunk_id=chunk['id'],
                    difficulty=Difficulty.HARD,
                    query_type='inferential'
                ))

        return examples
    ```

---

### Retrieval Metrics Deep Dive

Beyond precision and recall, several metrics provide deeper insight into retrieval quality.

!!! tip "For Product Managers"
    **When to use each metric**:

    | Metric | Best For | Interpretation |
    |--------|----------|----------------|
    | Precision@K | User experience (clean results) | "X% of shown results are relevant" |
    | Recall@K | Completeness (finding everything) | "We found X% of relevant documents" |
    | MRR | Single-answer queries | "Relevant result appears at position X on average" |
    | NDCG | Ranked relevance matters | "Results are well-ordered by relevance" |
    | MAP | Multiple relevant documents | "Average precision across all relevant docs" |

    **Practical guidance**: Start with Recall@10 as your primary metric. It tells you whether your system can find the right documents. Add Precision@K if users complain about irrelevant results. Add MRR if users care about the first result being correct.

!!! tip "For Engineers"
    **Mean Reciprocal Rank (MRR)**:

    MRR measures where the first relevant document appears in the results.

    $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

    Where $\text{rank}_i$ is the position of the first relevant document for query $i$.

    ```python
    def calculate_mrr(
        queries_results: list[tuple[list[str], set[str]]]
    ) -> float:
        """Calculate Mean Reciprocal Rank.

        Args:
            queries_results: List of (retrieved_ids, relevant_ids) tuples
        """
        reciprocal_ranks = []

        for retrieved_ids, relevant_ids in queries_results:
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_ids:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    ```

    **Normalized Discounted Cumulative Gain (NDCG)**:

    NDCG accounts for graded relevance (not just binary) and position.

    $$\text{DCG}_p = \sum_{i=1}^{p} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

    $$\text{NDCG}_p = \frac{\text{DCG}_p}{\text{IDCG}_p}$$

    Where IDCG is the DCG of the ideal ranking.

    ```python
    import math

    def calculate_ndcg(
        retrieved_ids: list[str],
        relevance_scores: dict[str, float],
        k: int = 10
    ) -> float:
        """Calculate NDCG@K.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevance_scores: Dict mapping doc_id to relevance score (0-1)
            k: Number of results to consider
        """
        def dcg(scores: list[float]) -> float:
            return sum(
                (2**score - 1) / math.log2(i + 2)
                for i, score in enumerate(scores)
            )

        # Get relevance scores for retrieved docs
        retrieved_scores = [
            relevance_scores.get(doc_id, 0.0)
            for doc_id in retrieved_ids[:k]
        ]

        # Calculate ideal ordering
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]

        dcg_score = dcg(retrieved_scores)
        idcg_score = dcg(ideal_scores)

        return dcg_score / idcg_score if idcg_score > 0 else 0.0
    ```

    **Statistical significance testing**:

    When comparing two retrieval systems, you need to know if differences are real or due to chance.

    ```python
    from scipy import stats
    import numpy as np

    def paired_t_test(
        scores_a: list[float],
        scores_b: list[float],
        alpha: float = 0.05
    ) -> dict:
        """Perform paired t-test to compare two systems.

        Args:
            scores_a: Per-query scores for system A
            scores_b: Per-query scores for system B
            alpha: Significance level
        """
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'mean_difference': np.mean(scores_a) - np.mean(scores_b),
            'confidence_interval': stats.t.interval(
                1 - alpha,
                len(scores_a) - 1,
                loc=np.mean(np.array(scores_a) - np.array(scores_b)),
                scale=stats.sem(np.array(scores_a) - np.array(scores_b))
            )
        }
    ```

    **Sample size calculation**:

    How many evaluation examples do you need to detect a meaningful difference?

    ```python
    from scipy.stats import norm

    def required_sample_size(
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> int:
        """Calculate required sample size for detecting an effect.

        Args:
            effect_size: Expected difference / standard deviation (Cohen's d)
            alpha: Significance level (Type I error rate)
            power: Statistical power (1 - Type II error rate)
        """
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))

    # Example: To detect a 0.05 improvement in recall (medium effect)
    # with standard deviation 0.15:
    # effect_size = 0.05 / 0.15 = 0.33
    # required_sample_size(0.33) ≈ 145 examples per group
    ```

---

### Evaluation Infrastructure

Building infrastructure that makes evaluation automatic and continuous.

!!! tip "For Product Managers"
    **Resource requirements**:

    | Component | Initial Investment | Ongoing Cost |
    |-----------|-------------------|--------------|
    | Evaluation dataset | 2-3 days | 1 day/month |
    | Pipeline code | 1-2 days | Minimal |
    | CI/CD integration | 1 day | Minimal |
    | Dashboard | 1-2 days | Minimal |

    **ROI calculation**: If evaluation prevents one week of wasted optimization work per month, it pays for itself immediately. Most teams report 3-5x ROI within the first quarter.

!!! tip "For Engineers"
    **Production monitoring beyond traditional error tracking**:

    Traditional error monitoring (like Sentry) does not work for AI systems—there is no exception when the model produces bad output. RAG systems require a fundamentally different monitoring approach.

    **Critical insight**: Track changes in metrics, not absolute values. The absolute cosine distance between queries and documents matters less than whether that distance is shifting over time.

    ```python
    from datetime import datetime, timedelta
    from collections import defaultdict

    class RetrievalMonitor:
        def __init__(self, alert_threshold: float = 0.1):
            self.metrics_history = defaultdict(list)
            self.alert_threshold = alert_threshold

        def log_query(
            self,
            query: str,
            retrieved_docs: list[dict],
            metadata: dict = None
        ):
            """Log a query for monitoring."""
            avg_score = sum(d['score'] for d in retrieved_docs) / len(retrieved_docs)

            self.metrics_history['avg_score'].append({
                'timestamp': datetime.now(),
                'value': avg_score,
                'metadata': metadata or {}
            })

        def check_drift(self, window_hours: int = 24) -> dict:
            """Check for metric drift over time window."""
            cutoff = datetime.now() - timedelta(hours=window_hours)

            recent = [
                m['value'] for m in self.metrics_history['avg_score']
                if m['timestamp'] > cutoff
            ]
            historical = [
                m['value'] for m in self.metrics_history['avg_score']
                if m['timestamp'] <= cutoff
            ]

            if not recent or not historical:
                return {'drift_detected': False, 'reason': 'insufficient_data'}

            recent_mean = sum(recent) / len(recent)
            historical_mean = sum(historical) / len(historical)
            drift = abs(recent_mean - historical_mean) / historical_mean

            return {
                'drift_detected': drift > self.alert_threshold,
                'drift_magnitude': drift,
                'recent_mean': recent_mean,
                'historical_mean': historical_mean
            }
    ```

    **Segment analysis**:

    Do not just track aggregate metrics. Segment by user cohorts, query categories, and time periods.

    ```python
    def analyze_by_segment(
        results: list[EvaluationResult],
        segment_fn: Callable[[EvaluationResult], str]
    ) -> dict[str, dict]:
        """Analyze metrics by segment."""
        segments = defaultdict(list)

        for result in results:
            segment = segment_fn(result)
            segments[segment].append(result)

        analysis = {}
        for segment, segment_results in segments.items():
            analysis[segment] = {
                'count': len(segment_results),
                'avg_recall': sum(r.recall for r in segment_results) / len(segment_results),
                'avg_precision': sum(r.precision for r in segment_results) / len(segment_results),
                'avg_mrr': sum(r.mrr for r in segment_results) / len(segment_results)
            }

        return analysis
    ```

---

## Case Study Deep Dive

### Case Study 1: Consulting Firm Report Generation

A consulting firm generates reports from user research interviews. Consultants conduct 15-30 interviews per project and need AI-generated summaries that capture all relevant insights.

!!! tip "For Product Managers"
    **Business context**: Reports were missing critical quotes. A consultant knew 6 experts said something similar, but the report only cited 3. That 50% recall rate destroyed trust. Consultants started spending hours manually verifying reports, defeating the automation's purpose.

    **Business outcomes**:

    | Stage | Recall | Business Impact |
    |-------|--------|-----------------|
    | Baseline | 50% | Manual verification required |
    | Week 1 | 70% | Reduced verification time |
    | Week 2 | 85% | Occasional spot checks |
    | Week 3 | 90% | Trust restored, full adoption |

    **Key decisions**:

    1. Invested in evaluation before optimization
    2. Measured recall specifically (not vague "quality")
    3. Focused on chunking (the actual problem) not prompts

!!! tip "For Engineers"
    **Technical investigation**:

    The team built manual evaluation sets from problematic examples. The issues turned out to be surprisingly straightforward—text chunking was breaking mid-quote and splitting speaker attributions from their statements.

    **Root cause analysis**:

    ```python
    # Problem: Chunks split mid-quote
    # Before:
    chunk_1 = "John said: 'The most important thing is"
    chunk_2 = "to maintain consistency across all touchpoints.'"

    # After: Respect interview structure
    chunk = """John said: 'The most important thing is
    to maintain consistency across all touchpoints.'"""
    ```

    **Solution implemented**:

    1. Redesigned chunking to respect interview structure
    2. Kept questions and answers together
    3. Preserved speaker attributions with complete statements
    4. Added overlap between chunks to catch context spanning sections

    **Evaluation-driven iteration**:

    ```python
    # Each iteration measured against same evaluation set
    results_v1 = evaluate_retrieval(eval_set, retriever_v1)  # 50% recall
    results_v2 = evaluate_retrieval(eval_set, retriever_v2)  # 70% recall
    results_v3 = evaluate_retrieval(eval_set, retriever_v3)  # 85% recall
    results_v4 = evaluate_retrieval(eval_set, retriever_v4)  # 90% recall
    ```

### Case Study 2: Blueprint Search for Construction

A construction technology company needed AI search for building blueprints. Workers asked questions like "Which rooms have north-facing windows?" or "Show me all electrical outlet locations."

!!! tip "For Product Managers"
    **Business context**: Only 27% recall when finding the right blueprint sections. Workers would ask simple spatial questions and get completely unrelated blueprint segments. The system was essentially useless—workers abandoned it and went back to manually scrolling through PDFs.

    **Business outcomes**:

    | Stage | Recall | Time | Key Change |
    |-------|--------|------|------------|
    | Baseline | 27% | - | Text embeddings on blueprints |
    | Vision captions | 85% | 4 days | Added spatial descriptions |
    | Counting queries | 92% | +2 weeks | Bounding box detection |

    **Key insight**: Test subsystems independently for rapid improvements. Do not try to solve everything at once.

!!! tip "For Engineers"
    **Technical approach**:

    Standard text embeddings could not handle the spatial and visual nature of blueprint queries. "North-facing windows" and "electrical outlets" are visual concepts that do not translate well to text chunks.

    **Solution: Vision-to-text transformation**:

    ```python
    async def generate_blueprint_caption(
        image: bytes,
        vision_model
    ) -> str:
        """Generate searchable caption from blueprint image."""
        prompt = """Describe this blueprint section in detail:

        1. Room identification and purpose
        2. Spatial relationships (north side, adjacent to, etc.)
        3. Visible features (windows, doors, outlets, fixtures)
        4. Measurements if visible

        Also generate 5 hypothetical questions users might ask
        about this section."""

        return await vision_model.analyze(image, prompt)
    ```

    **Evaluation revealed new opportunity**:

    Once live, usage data revealed that 20% of queries involved counting objects ("How many outlets in this room?"). This justified investing in bounding box detection models for those specific counting use cases.

    ```python
    # Query analysis revealed counting pattern
    counting_queries = [q for q in queries if is_counting_query(q)]
    print(f"Counting queries: {len(counting_queries) / len(queries):.1%}")
    # Output: Counting queries: 20.3%
    ```

---

## Implementation Guide

### Quick Start for PMs

**Week 1: Establish Baseline**

1. Define what "success" means for your RAG system
2. Work with engineering to generate 50-100 synthetic evaluation examples
3. Run baseline evaluation and document current metrics
4. Identify the top 3 failure patterns

**Week 2: Build Measurement Habit**

1. Schedule weekly metric reviews
2. Create a simple dashboard showing trends
3. Start tracking experiment velocity
4. Document every change and its measured impact

**Week 3: Expand Coverage**

1. Add evaluation examples for edge cases
2. Include different difficulty levels
3. Segment metrics by query type
4. Identify gaps in coverage

**Ongoing: Maintain the Flywheel**

- Weekly: Review metrics, identify issues
- Monthly: Analyze query clusters, update priorities
- Quarterly: Assess overall progress, adjust strategy

### Detailed Implementation for Engineers

**Step 1: Set Up Evaluation Dataset**

```python
import json
from pathlib import Path

async def create_evaluation_dataset(
    chunks: list[dict],
    output_path: str,
    llm_client,
    target_size: int = 100
):
    """Create and save evaluation dataset."""
    examples = []

    for chunk in chunks[:target_size]:
        questions = await generate_questions_from_chunk(chunk['text'], llm_client)

        for q in questions:
            examples.append({
                'query_id': f"{chunk['id']}_{len(examples)}",
                'query': q,
                'relevant_doc_ids': [chunk['id']],
                'metadata': {
                    'source_chunk': chunk['id'],
                    'difficulty': 'synthetic'
                }
            })

    Path(output_path).write_text(json.dumps(examples, indent=2))
    return examples
```

**Step 2: Implement Evaluation Pipeline**

```python
class RAGEvaluator:
    def __init__(self, retriever, k: int = 10):
        self.retriever = retriever
        self.k = k
        self.results_history = []

    def evaluate(
        self,
        eval_dataset: list[dict],
        experiment_name: str = None
    ) -> dict:
        """Run full evaluation."""
        results = []

        for example in eval_dataset:
            retrieved = self.retriever.search(example['query'], top_k=self.k)
            retrieved_ids = [doc['id'] for doc in retrieved]

            precision, recall = calculate_precision_recall(
                set(retrieved_ids),
                set(example['relevant_doc_ids'])
            )

            results.append({
                'query_id': example['query_id'],
                'precision': precision,
                'recall': recall,
                'retrieved': retrieved_ids,
                'expected': example['relevant_doc_ids']
            })

        summary = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'k': self.k,
            'num_examples': len(results),
            'avg_precision': sum(r['precision'] for r in results) / len(results),
            'avg_recall': sum(r['recall'] for r in results) / len(results),
            'zero_recall_count': sum(1 for r in results if r['recall'] == 0),
            'perfect_recall_count': sum(1 for r in results if r['recall'] == 1.0),
            'detailed_results': results
        }

        self.results_history.append(summary)
        return summary

    def compare_experiments(
        self,
        experiment_a: str,
        experiment_b: str
    ) -> dict:
        """Compare two experiments statistically."""
        results_a = next(r for r in self.results_history if r['experiment_name'] == experiment_a)
        results_b = next(r for r in self.results_history if r['experiment_name'] == experiment_b)

        recalls_a = [r['recall'] for r in results_a['detailed_results']]
        recalls_b = [r['recall'] for r in results_b['detailed_results']]

        return paired_t_test(recalls_a, recalls_b)
```

**Step 3: Integrate with Development Workflow**

```python
# In your CI/CD pipeline or pre-commit hook
def run_evaluation_gate():
    """Run evaluation and fail if regression detected."""
    evaluator = RAGEvaluator(retriever, k=10)

    # Load evaluation dataset
    eval_data = json.loads(Path('eval_dataset.json').read_text())

    # Run evaluation
    results = evaluator.evaluate(eval_data, experiment_name='current')

    # Compare to baseline
    baseline = json.loads(Path('baseline_metrics.json').read_text())

    if results['avg_recall'] < baseline['avg_recall'] - 0.02:
        raise ValueError(
            f"Recall regression: {baseline['avg_recall']:.3f} -> {results['avg_recall']:.3f}"
        )

    print(f"Evaluation passed: recall={results['avg_recall']:.3f}")
```

---

## Common Pitfalls

### PM Pitfalls

!!! warning "PM Pitfall: Vague Metrics"
    **The mistake**: Evaluating success through subjective assessment—"does it look better?" or "does it feel right?"

    **Why it happens**: Concrete metrics require upfront investment. Subjective assessment feels faster.

    **The consequence**: Teams spend weeks making changes without knowing if they help. When performance shifts, they cannot identify what changed or why.

    **How to avoid**: Define specific, measurable metrics before any optimization work. "Recall@10 should improve from 65% to 75%" not "search should be better."

!!! warning "PM Pitfall: Optimizing Lagging Metrics"
    **The mistake**: Obsessing over user satisfaction scores while ignoring experiment velocity.

    **Why it happens**: Lagging metrics are what stakeholders ask about.

    **The consequence**: Teams feel stuck because they cannot directly control outcomes.

    **How to avoid**: Track and celebrate leading metrics. Report experiment velocity alongside satisfaction scores.

!!! warning "PM Pitfall: Skipping Synthetic Data"
    **The mistake**: Waiting for real user data before building evaluation infrastructure.

    **Why it happens**: Synthetic data feels like extra work before launch.

    **The consequence**: No baseline to measure against. Months of guessing what might work.

    **How to avoid**: Generate synthetic evaluation data before launch. Even 50 synthetic queries provide valuable signal.

### Engineering Pitfalls

!!! warning "Engineering Pitfall: Intervention Bias"
    **The mistake**: Constantly switching models, tweaking prompts, or adding features without measuring impact.

    **Why it happens**: Doing something feels like progress. Measuring feels slow.

    **The consequence**: No learning. Changes may help, hurt, or do nothing—you will never know.

    **How to avoid**: Every change should target a specific metric and test a clear hypothesis. Eliminate exploratory changes without measurement.

!!! warning "Engineering Pitfall: Absence Blindness"
    **The mistake**: Obsessing over generation quality while completely ignoring whether retrieval works.

    **Why it happens**: Generation is visible. Retrieval failures are invisible.

    **The consequence**: Teams spend weeks fine-tuning prompts, only to discover retrieval returns completely irrelevant documents.

    **How to avoid**: Always evaluate retrieval separately from generation. Check retrieval metrics before touching prompts.

!!! warning "Engineering Pitfall: Score Threshold Traps"
    **The mistake**: Setting a fixed similarity score threshold (e.g., 0.7) to filter results.

    **Why it happens**: It seems intuitive—only return "confident" results.

    **The consequence**: Score distributions vary by query type. A threshold that works for one category fails for others. Re-ranker scores are not probabilities.

    **How to avoid**: Return top K results, let the LLM filter. Set thresholds based on diminishing recall returns, not absolute scores.

---

## Related Content

### Transcript

The full lecture transcript is available at `docs/workshops/chapter1-transcript.txt`. Key insights from the lecture:

- "You cannot improve what you cannot measure—and you can measure before you have users."
- "Over 90% of complexity additions to RAG systems perform worse than simpler approaches when properly evaluated."
- "The goal is not chasing the latest AI techniques. It is building a flywheel of continuous improvement driven by clear metrics."

### Talk: Generative Evals (Kelly Hong, Chroma)

Full talk available at `docs/talks/embedding-performance-generative-evals-kelly-hong.md`. Key insights:

- **Public benchmarks do not reflect real-world performance**: "If you have really good performance on a public benchmark for a given embedding model, that doesn't necessarily guarantee that you'll also get that good performance for your specific production pipeline."
- **Generate realistic queries**: Naive query generation produces perfectly formed questions that make retrieval too easy. Real users search with incomplete, ambiguous queries.
- **Human involvement is critical**: "If you want really good evals, I think it applies to basically any case where you're working with AI as well. I think it's very rare that your system is going to work well with absolutely no human in the loop."

### Talk: Zapier Feedback Systems (Vitor)

Full talk available at `docs/talks/zapier-vitor-evals.md`. Key insights:

- **4x feedback improvement**: By changing feedback button placement and wording, Zapier increased submissions from 10 to 40 per day.
- **Mine implicit signals**: Workflow activations, validation errors, and follow-up messages all provide feedback signals.
- **Turn feedback into evaluations**: Zapier grew from 23 to 383 evaluations based on real user interactions.

### Office Hours

Relevant office hours sessions:

- **Cohort 2 Week 1** (`docs/office-hours/cohort2/week1-summary.md`): Discussion of precision vs recall, bi-encoders vs cross-encoders, graph database skepticism
- **Cohort 3 Week 1** (`docs/office-hours/cohort3/week-1-1.md`): Deep dive on precision sensitivity, small language models in RAG, business value analysis

---

## Action Items

### For Product Teams

1. **This week**: Define 3-5 specific, measurable metrics for your RAG system
2. **This week**: Work with engineering to generate 50+ synthetic evaluation examples
3. **This month**: Establish baseline metrics and start tracking experiment velocity
4. **This quarter**: Build a dashboard showing metric trends over time
5. **Ongoing**: Review metrics weekly, celebrate experiment velocity improvements

### For Engineering Teams

1. **This week**: Implement the basic evaluation pipeline from this chapter
2. **This week**: Generate synthetic evaluation data (minimum 50 examples)
3. **This month**: Integrate evaluation into CI/CD (fail on regression)
4. **This month**: Implement retrieval monitoring for production
5. **This quarter**: Add statistical significance testing to experiment comparisons
6. **Ongoing**: Run evaluation with every significant change

---

## Reflection Questions

1. What are your current leading and lagging metrics? How do they connect? If you do not have leading metrics, what would be the most valuable ones to track?

2. How would you generate synthetic evaluation data for your specific domain? What makes a "realistic" query in your context?

3. Think of a recent change to your RAG system. Did you measure its impact? If not, how would you design an experiment to test it?

4. Where might absence blindness be affecting your team? What retrieval failures might be invisible to you?

5. If you could only track one metric, what would it be and why?

---

## Summary

### Key Takeaways for Product Managers

- **Leading metrics over lagging metrics**: Focus on what you can control (experiment velocity, evaluation coverage) rather than what you cannot (user satisfaction, retention).
- **Evaluation-first development**: Establish metrics and baselines before any optimization work. The cost of evaluation is far less than the cost of optimizing blindly.
- **Synthetic data is powerful**: You can measure and improve before you have users. Do not wait for real data to start the flywheel.
- **Precision vs recall tradeoffs**: Understand which matters more for your use case. Modern models handle low precision well, so prioritize recall.
- **The teams that iterate fastest win**: Build infrastructure that enables rapid experimentation.

### Key Takeaways for Engineers

- **Implement evaluation before optimization**: Every change should target a specific metric. Eliminate changes without measurement.
- **Retrieval metrics first**: Precision, recall, MRR, and NDCG are faster, cheaper, and more objective than generation quality metrics.
- **Statistical rigor matters**: Use paired t-tests and confidence intervals to determine if improvements are real.
- **Monitor for drift**: Track metric changes over time, not just absolute values. Segment by user cohorts and query types.
- **Avoid common traps**: Score thresholds are dangerous, absence blindness is real, and intervention bias kills learning.

---

## Further Reading

### Academic Papers

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) - The original RAG paper
- "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models" (Thakur et al., 2021) - Standard evaluation benchmark
- "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset" (Nguyen et al., 2016) - Large-scale retrieval benchmark

### Tools and Libraries

- **RAGAS**: Open-source framework for evaluating RAG applications
- **LangSmith**: Evaluation and monitoring for LLM applications
- **Braintrust**: Experiment tracking and evaluation platform
- **MLflow**: Open-source platform for managing ML lifecycle

### Related Appendices

- **Appendix A: Mathematical Foundations** - Full derivations of retrieval metrics
- **Appendix C: Benchmarking Your RAG System** - Standard datasets and methodology

---

## Navigation

- **Previous**: [Chapter 0: Introduction - The Product Mindset for RAG](chapter0.md) - Foundational concepts and the improvement flywheel
- **Next**: [Chapter 2: Training Data and Fine-Tuning](chapter2.md) - Converting evaluations into training data
- **Reference**: [Glossary](glossary.md) | [Quick Reference](quick-reference.md)
- **Book Index**: [Book 1: Foundations](book1-index.md)
