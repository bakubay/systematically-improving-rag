"""
Chapter 1: Evaluation-First Development - Code Examples

This module provides the evaluation infrastructure for RAG systems, including:
- Precision, recall, MRR, and NDCG calculations
- Evaluation pipeline for running experiments
- Statistical significance testing
- Production monitoring utilities
- Synthetic data generation helpers

Usage:
    from chapter1_evaluation import (
        EvaluationPipeline,
        calculate_precision_recall,
        calculate_mrr,
        calculate_ndcg,
        paired_t_test,
    )

    # Create evaluation pipeline
    pipeline = EvaluationPipeline(retriever_fn=my_retriever, k=10)

    # Run evaluation
    results = pipeline.evaluate_all(examples)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel
from scipy import stats
from scipy.stats import norm


# =============================================================================
# Data Models
# =============================================================================


class Experiment(BaseModel):
    """Model for tracking experiments."""

    id: str
    hypothesis: str
    metric_target: str
    baseline_value: float
    result_value: float | None = None
    started_at: datetime
    completed_at: datetime | None = None


class Difficulty(str, Enum):
    """Difficulty levels for evaluation examples."""

    EASY = "easy"  # Direct keyword match expected
    MEDIUM = "medium"  # Requires some semantic understanding
    HARD = "hard"  # Different terminology, inference required


@dataclass
class EvaluationExample:
    """A single evaluation example with query and expected relevant documents."""

    query_id: str
    query: str
    relevant_doc_ids: set[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""

    query_id: str
    query: str
    precision: float
    recall: float
    mrr: float
    retrieved_ids: list[str]
    relevant_ids: set[str]


@dataclass
class SyntheticExample:
    """A synthetic evaluation example with metadata."""

    query: str
    source_chunk_id: str
    difficulty: Difficulty
    query_type: str  # factual, inferential, comparative, etc.


# =============================================================================
# Core Metrics
# =============================================================================


def calculate_precision_recall(
    retrieved_ids: set[str],
    relevant_ids: set[str],
) -> tuple[float, float]:
    """
    Calculate precision and recall for a single query.

    Precision = |Relevant ∩ Retrieved| / |Retrieved|
    Recall = |Relevant ∩ Retrieved| / |Relevant|

    Args:
        retrieved_ids: Set of document IDs returned by the retriever
        relevant_ids: Set of document IDs that are actually relevant

    Returns:
        Tuple of (precision, recall)

    Examples:
        >>> calculate_precision_recall({'a', 'b', 'c'}, {'a', 'b', 'd'})
        (0.6666666666666666, 0.6666666666666666)
        >>> calculate_precision_recall(set(), {'a', 'b'})
        (0.0, 0.0)
        >>> calculate_precision_recall({'a', 'b'}, set())
        (0.0, 1.0)
    """
    if not retrieved_ids:
        return 0.0, 0.0 if relevant_ids else 1.0

    true_positives = len(retrieved_ids & relevant_ids)
    precision = true_positives / len(retrieved_ids)
    recall = true_positives / len(relevant_ids) if relevant_ids else 1.0

    return precision, recall


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        precision: Precision value (0-1)
        recall: Recall value (0-1)

    Returns:
        F1 score (0-1)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_mrr(
    queries_results: list[tuple[list[str], set[str]]],
) -> float:
    """
    Calculate Mean Reciprocal Rank across multiple queries.

    MRR = (1/|Q|) * Σ(1/rank_i)

    Where rank_i is the position of the first relevant document for query i.

    Args:
        queries_results: List of (retrieved_ids, relevant_ids) tuples

    Returns:
        Mean Reciprocal Rank (0-1)

    Examples:
        >>> calculate_mrr([(['a', 'b', 'c'], {'a'})])  # First result is relevant
        1.0
        >>> calculate_mrr([(['a', 'b', 'c'], {'b'})])  # Second result is relevant
        0.5
        >>> calculate_mrr([(['a', 'b', 'c'], {'d'})])  # No relevant results
        0.0
    """
    if not queries_results:
        return 0.0

    reciprocal_ranks = []

    for retrieved_ids, relevant_ids in queries_results:
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_ndcg(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: int = 10,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    DCG_p = Σ(2^rel_i - 1) / log2(i + 1)
    NDCG_p = DCG_p / IDCG_p

    Where IDCG is the DCG of the ideal ranking.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevance_scores: Dict mapping doc_id to relevance score (0-1)
        k: Number of results to consider

    Returns:
        NDCG@K score (0-1)

    Examples:
        >>> calculate_ndcg(['a', 'b', 'c'], {'a': 1.0, 'b': 0.5, 'c': 0.0}, k=3)
        1.0  # Perfect ordering
    """

    def dcg(scores: list[float]) -> float:
        return sum(
            (2**score - 1) / math.log2(i + 2) for i, score in enumerate(scores)
        )

    # Get relevance scores for retrieved docs
    retrieved_scores = [
        relevance_scores.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]
    ]

    # Calculate ideal ordering
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]

    dcg_score = dcg(retrieved_scores)
    idcg_score = dcg(ideal_scores)

    return dcg_score / idcg_score if idcg_score > 0 else 0.0


def evaluate_at_k(
    query: str,
    relevant_ids: set[str],
    retriever_fn: Callable[[str, int], list[dict[str, Any]]],
    k_values: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """
    Evaluate precision and recall at different K values.

    Args:
        query: The query string
        relevant_ids: Set of relevant document IDs
        retriever_fn: Function that takes (query, top_k) and returns list of docs
        k_values: List of K values to evaluate at (default: [3, 5, 10, 20])

    Returns:
        Dict mapping K to metrics dict with precision, recall, f1
    """
    if k_values is None:
        k_values = [3, 5, 10, 20]

    results = {}
    max_k = max(k_values)
    all_retrieved = retriever_fn(query, max_k)

    for k in k_values:
        retrieved_k = set(doc["id"] for doc in all_retrieved[:k])
        precision, recall = calculate_precision_recall(retrieved_k, relevant_ids)
        results[k] = {
            "precision": precision,
            "recall": recall,
            "f1": calculate_f1(precision, recall),
        }

    return results


# =============================================================================
# Statistical Methods
# =============================================================================


def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Perform paired t-test to compare two retrieval systems.

    Use this to determine if the difference between two systems is
    statistically significant.

    Args:
        scores_a: Per-query scores for system A
        scores_b: Per-query scores for system B
        alpha: Significance level (default: 0.05)

    Returns:
        Dict with t_statistic, p_value, significant, mean_difference,
        and confidence_interval

    Raises:
        ValueError: If score lists have different lengths

    Examples:
        >>> result = paired_t_test([0.8, 0.7, 0.9], [0.6, 0.5, 0.7])
        >>> result['significant']  # True if p_value < alpha
        True
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have the same length")

    scores_a_arr = np.array(scores_a)
    scores_b_arr = np.array(scores_b)

    t_stat, p_value = stats.ttest_rel(scores_a_arr, scores_b_arr)

    diff = scores_a_arr - scores_b_arr
    confidence_interval = stats.t.interval(
        1 - alpha,
        len(scores_a) - 1,
        loc=np.mean(diff),
        scale=stats.sem(diff),
    )

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "mean_difference": float(np.mean(diff)),
        "confidence_interval": (float(confidence_interval[0]), float(confidence_interval[1])),
    }


def required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculate required sample size for detecting an effect.

    Use Cohen's d for effect size: expected_difference / standard_deviation

    Args:
        effect_size: Expected difference / standard deviation (Cohen's d)
        alpha: Significance level (Type I error rate)
        power: Statistical power (1 - Type II error rate)

    Returns:
        Required sample size per group

    Examples:
        >>> # To detect a 0.05 improvement in recall with std dev 0.15:
        >>> # effect_size = 0.05 / 0.15 = 0.33
        >>> required_sample_size(0.33)
        145
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))


# =============================================================================
# Evaluation Pipeline
# =============================================================================


class EvaluationPipeline:
    """
    Pipeline for running retrieval evaluations.

    This class provides methods to evaluate a retriever against a set of
    evaluation examples and aggregate metrics.

    Attributes:
        retriever_fn: Function that takes (query, top_k) and returns list of docs
        k: Number of results to retrieve for each query

    Examples:
        >>> pipeline = EvaluationPipeline(my_retriever, k=10)
        >>> results = pipeline.evaluate_all(examples)
        >>> print(f"Average recall: {results['avg_recall']:.2%}")
    """

    def __init__(
        self,
        retriever_fn: Callable[[str, int], list[dict[str, Any]]],
        k: int = 10,
    ):
        """
        Initialize the evaluation pipeline.

        Args:
            retriever_fn: Function that takes (query, top_k) and returns
                          list of dicts with at least 'id' key
            k: Number of results to retrieve for each query
        """
        self.retriever_fn = retriever_fn
        self.k = k

    def evaluate_single(self, example: EvaluationExample) -> EvaluationResult:
        """
        Evaluate a single query.

        Args:
            example: EvaluationExample with query and relevant doc IDs

        Returns:
            EvaluationResult with precision, recall, MRR, and retrieved IDs
        """
        retrieved = self.retriever_fn(example.query, self.k)
        retrieved_ids = [doc["id"] for doc in retrieved]

        precision, recall = calculate_precision_recall(
            set(retrieved_ids),
            example.relevant_doc_ids,
        )

        # Calculate MRR for this single query
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
            relevant_ids=example.relevant_doc_ids,
        )

    def evaluate_all(
        self,
        examples: list[EvaluationExample],
    ) -> dict[str, Any]:
        """
        Evaluate all examples and aggregate metrics.

        Args:
            examples: List of EvaluationExample objects

        Returns:
            Dict with timestamp, k, num_examples, avg_precision, avg_recall,
            avg_mrr, and detailed_results
        """
        if not examples:
            return {
                "timestamp": datetime.now().isoformat(),
                "k": self.k,
                "num_examples": 0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_mrr": 0.0,
                "detailed_results": [],
            }

        results = [self.evaluate_single(ex) for ex in examples]

        return {
            "timestamp": datetime.now().isoformat(),
            "k": self.k,
            "num_examples": len(examples),
            "avg_precision": sum(r.precision for r in results) / len(results),
            "avg_recall": sum(r.recall for r in results) / len(results),
            "avg_mrr": sum(r.mrr for r in results) / len(results),
            "detailed_results": results,
        }


# =============================================================================
# Experiment Tracking
# =============================================================================


def calculate_experiment_velocity(
    experiments: list[Experiment],
    window_days: int = 7,
) -> float:
    """
    Calculate experiments completed per week.

    Experiment velocity is a key leading metric for RAG improvement.
    Teams that run more experiments per week consistently outperform
    teams that run fewer.

    Args:
        experiments: List of Experiment objects
        window_days: Time window in days (default: 7)

    Returns:
        Number of experiments completed per week
    """
    cutoff = datetime.now() - timedelta(days=window_days)
    completed = [
        e
        for e in experiments
        if e.completed_at and e.completed_at > cutoff
    ]
    return len(completed) / (window_days / 7)


# =============================================================================
# CI/CD Integration
# =============================================================================


def run_evaluation_check(
    baseline_file: str | Path,
    current_results: dict[str, Any],
    regression_threshold: float = 0.02,
) -> bool:
    """
    Check if current results regress from baseline.

    Use this in CI/CD pipelines to prevent merging changes that
    degrade retrieval quality.

    Args:
        baseline_file: Path to JSON file with baseline metrics
        current_results: Dict with current evaluation results
        regression_threshold: Maximum allowed regression (default: 0.02)

    Returns:
        True if no regression detected, False otherwise

    Examples:
        >>> results = evaluator.evaluate_all(examples)
        >>> if not run_evaluation_check('baseline.json', results):
        ...     raise ValueError("Regression detected!")
    """
    baseline_path = Path(baseline_file)
    with baseline_path.open() as f:
        baseline = json.load(f)

    for metric in ["avg_precision", "avg_recall", "avg_mrr"]:
        baseline_value = baseline.get(metric, 0.0)
        current_value = current_results.get(metric, 0.0)
        if current_value < baseline_value - regression_threshold:
            print(
                f"REGRESSION: {metric} dropped from {baseline_value:.3f} "
                f"to {current_value:.3f}"
            )
            return False

    return True


# =============================================================================
# Production Monitoring
# =============================================================================


class RetrievalMonitor:
    """
    Monitor retrieval metrics in production.

    Traditional error monitoring (like Sentry) doesn't work for AI systems
    because there's no exception when the model produces bad output.
    This class tracks metric changes over time to detect drift.

    Attributes:
        alert_threshold: Threshold for drift detection (default: 0.1)
        metrics_history: History of logged metrics

    Examples:
        >>> monitor = RetrievalMonitor(alert_threshold=0.1)
        >>> monitor.log_query("what is RAG?", retrieved_docs)
        >>> drift_info = monitor.check_drift(window_hours=24)
        >>> if drift_info['drift_detected']:
        ...     send_alert(drift_info)
    """

    def __init__(self, alert_threshold: float = 0.1):
        """
        Initialize the retrieval monitor.

        Args:
            alert_threshold: Threshold for drift detection (0-1)
        """
        self.metrics_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.alert_threshold = alert_threshold

    def log_query(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a query for monitoring.

        Args:
            query: The query string
            retrieved_docs: List of retrieved documents with 'score' key
            metadata: Optional metadata to store with the log
        """
        if not retrieved_docs:
            return

        avg_score = sum(d.get("score", 0.0) for d in retrieved_docs) / len(
            retrieved_docs
        )

        self.metrics_history["avg_score"].append(
            {
                "timestamp": datetime.now(),
                "value": avg_score,
                "query": query,
                "metadata": metadata or {},
            }
        )

    def check_drift(self, window_hours: int = 24) -> dict[str, Any]:
        """
        Check for metric drift over time window.

        Compares recent metrics to historical metrics to detect
        significant changes that might indicate problems.

        Args:
            window_hours: Time window for "recent" metrics

        Returns:
            Dict with drift_detected, drift_magnitude, recent_mean,
            historical_mean, and reason
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)

        recent = [
            m["value"]
            for m in self.metrics_history["avg_score"]
            if m["timestamp"] > cutoff
        ]
        historical = [
            m["value"]
            for m in self.metrics_history["avg_score"]
            if m["timestamp"] <= cutoff
        ]

        if not recent or not historical:
            return {"drift_detected": False, "reason": "insufficient_data"}

        recent_mean = sum(recent) / len(recent)
        historical_mean = sum(historical) / len(historical)

        if historical_mean == 0:
            return {"drift_detected": False, "reason": "historical_mean_zero"}

        drift = abs(recent_mean - historical_mean) / historical_mean

        return {
            "drift_detected": drift > self.alert_threshold,
            "drift_magnitude": drift,
            "recent_mean": recent_mean,
            "historical_mean": historical_mean,
            "reason": "drift_check_complete",
        }


# =============================================================================
# Segment Analysis
# =============================================================================


def analyze_by_segment(
    results: list[EvaluationResult],
    segment_fn: Callable[[EvaluationResult], str],
) -> dict[str, dict[str, Any]]:
    """
    Analyze metrics by segment.

    Don't just track aggregate metrics. Segment by user cohorts,
    query categories, and time periods to find specific issues.

    Args:
        results: List of EvaluationResult objects
        segment_fn: Function that takes an EvaluationResult and returns
                    a segment name string

    Returns:
        Dict mapping segment names to metric dicts

    Examples:
        >>> def by_query_length(r):
        ...     return "short" if len(r.query) < 50 else "long"
        >>> analysis = analyze_by_segment(results, by_query_length)
        >>> print(analysis['short']['avg_recall'])
    """
    segments: dict[str, list[EvaluationResult]] = defaultdict(list)

    for result in results:
        segment = segment_fn(result)
        segments[segment].append(result)

    analysis = {}
    for segment, segment_results in segments.items():
        if not segment_results:
            continue
        analysis[segment] = {
            "count": len(segment_results),
            "avg_recall": sum(r.recall for r in segment_results) / len(segment_results),
            "avg_precision": sum(r.precision for r in segment_results)
            / len(segment_results),
            "avg_mrr": sum(r.mrr for r in segment_results) / len(segment_results),
        }

    return analysis


# =============================================================================
# Evaluation Dataset Management
# =============================================================================


async def create_evaluation_dataset(
    chunks: list[dict[str, Any]],
    output_path: str | Path,
    llm_client: Any,
    target_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Create and save evaluation dataset from document chunks.

    This is an async function that generates synthetic questions
    from document chunks using an LLM.

    Args:
        chunks: List of chunk dicts with 'id' and 'text' keys
        output_path: Path to save the evaluation dataset JSON
        llm_client: LLM client with async generate() method
        target_size: Target number of chunks to process

    Returns:
        List of evaluation example dicts
    """
    examples = []

    for chunk in chunks[:target_size]:
        questions = await generate_questions_from_chunk(
            chunk["text"], llm_client
        )

        for q in questions:
            examples.append(
                {
                    "query_id": f"{chunk['id']}_{len(examples)}",
                    "query": q,
                    "relevant_doc_ids": [chunk["id"]],
                    "metadata": {
                        "source_chunk": chunk["id"],
                        "difficulty": "synthetic",
                    },
                }
            )

    output = Path(output_path)
    output.write_text(json.dumps(examples, indent=2))
    return examples


# =============================================================================
# Synthetic Data Generation
# =============================================================================


async def generate_questions_from_chunk(
    chunk: str,
    llm_client: Any,
    num_questions: int = 3,
) -> list[str]:
    """
    Generate questions that would be answered by this chunk.

    Args:
        chunk: Text content of the chunk
        llm_client: LLM client with async generate() method
        num_questions: Number of questions to generate

    Returns:
        List of generated question strings
    """
    prompt = f"""Based on this text, generate {num_questions} questions
that could be answered using this information.

Text: {chunk}

Requirements:
- Questions should be realistic (what a user might actually ask)
- Vary the complexity (simple factual, inferential, comparative)
- Use natural language, not formal queries

Format: One question per line, no numbering."""

    response = await llm_client.generate(prompt)
    return [q.strip() for q in response.strip().split("\n") if q.strip()]


async def generate_diverse_questions(
    chunk: str,
    llm_client: Any,
    example_queries: list[str] | None = None,
    domain_context: str | None = None,
) -> list[str]:
    """
    Generate diverse, realistic questions.

    Args:
        chunk: Text content of the chunk
        llm_client: LLM client with async generate() method
        example_queries: Optional list of real user queries for style matching
        domain_context: Optional domain context string

    Returns:
        List of generated question strings
    """
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
    return [q.strip() for q in response.strip().split("\n") if q.strip()]


async def generate_adversarial_questions(
    chunk: str,
    llm_client: Any,
) -> list[str]:
    """
    Generate challenging questions that test retrieval robustness.

    These questions use different terminology than what appears in
    the passage and require understanding implications.

    Args:
        chunk: Text content of the chunk
        llm_client: LLM client with async generate() method

    Returns:
        List of generated adversarial question strings
    """
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
    return [q.strip() for q in response.strip().split("\n") if q.strip()]


async def build_evaluation_set(
    chunks: list[dict[str, Any]],
    llm_client: Any,
    target_size: int = 200,
) -> list[SyntheticExample]:
    """
    Build a balanced evaluation set with varying difficulty.

    Args:
        chunks: List of chunk dicts with 'id' and 'text' keys
        llm_client: LLM client with async generate() method
        target_size: Target total number of examples

    Returns:
        List of SyntheticExample objects
    """
    import random

    examples = []

    # Sample chunks proportionally
    sample_size = min(len(chunks), target_size // 3)
    sampled_chunks = random.sample(chunks, sample_size)

    for chunk in sampled_chunks:
        # Generate easy questions (direct)
        easy_qs = await generate_questions_from_chunk(
            chunk["text"], llm_client, num_questions=1
        )
        for q in easy_qs:
            examples.append(
                SyntheticExample(
                    query=q,
                    source_chunk_id=chunk["id"],
                    difficulty=Difficulty.EASY,
                    query_type="factual",
                )
            )

        # Generate hard questions (adversarial)
        hard_qs = await generate_adversarial_questions(chunk["text"], llm_client)
        for q in hard_qs[:1]:
            examples.append(
                SyntheticExample(
                    query=q,
                    source_chunk_id=chunk["id"],
                    difficulty=Difficulty.HARD,
                    query_type="inferential",
                )
            )

    return examples


# =============================================================================
# RAG Evaluator (Higher-Level Interface)
# =============================================================================


class RAGEvaluator:
    """
    Higher-level evaluator with experiment tracking and comparison.

    This class extends the basic EvaluationPipeline with features for
    tracking multiple experiments and comparing them statistically.

    Attributes:
        retriever: The retriever object with a search() method
        k: Number of results to retrieve
        results_history: List of past evaluation results

    Examples:
        >>> evaluator = RAGEvaluator(my_retriever, k=10)
        >>> results_v1 = evaluator.evaluate(eval_data, 'baseline')
        >>> # Make changes to retriever...
        >>> results_v2 = evaluator.evaluate(eval_data, 'improved')
        >>> comparison = evaluator.compare_experiments('baseline', 'improved')
        >>> print(f"Significant: {comparison['significant']}")
    """

    def __init__(self, retriever: Any, k: int = 10):
        """
        Initialize the RAG evaluator.

        Args:
            retriever: Retriever object with search(query, top_k) method
            k: Number of results to retrieve
        """
        self.retriever = retriever
        self.k = k
        self.results_history: list[dict[str, Any]] = []

    def evaluate(
        self,
        eval_dataset: list[dict[str, Any]],
        experiment_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Run full evaluation on a dataset.

        Args:
            eval_dataset: List of dicts with 'query_id', 'query',
                          and 'relevant_doc_ids' keys
            experiment_name: Optional name for this experiment

        Returns:
            Dict with experiment metrics and detailed results
        """
        results = []

        for example in eval_dataset:
            retrieved = self.retriever.search(example["query"], top_k=self.k)
            retrieved_ids = [doc["id"] for doc in retrieved]

            precision, recall = calculate_precision_recall(
                set(retrieved_ids),
                set(example["relevant_doc_ids"]),
            )

            results.append(
                {
                    "query_id": example["query_id"],
                    "precision": precision,
                    "recall": recall,
                    "retrieved": retrieved_ids,
                    "expected": example["relevant_doc_ids"],
                }
            )

        summary = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "k": self.k,
            "num_examples": len(results),
            "avg_precision": sum(r["precision"] for r in results) / len(results)
            if results
            else 0.0,
            "avg_recall": sum(r["recall"] for r in results) / len(results)
            if results
            else 0.0,
            "zero_recall_count": sum(1 for r in results if r["recall"] == 0),
            "perfect_recall_count": sum(1 for r in results if r["recall"] == 1.0),
            "detailed_results": results,
        }

        self.results_history.append(summary)
        return summary

    def compare_experiments(
        self,
        experiment_a: str,
        experiment_b: str,
    ) -> dict[str, Any]:
        """
        Compare two experiments statistically.

        Args:
            experiment_a: Name of first experiment
            experiment_b: Name of second experiment

        Returns:
            Dict with statistical comparison results

        Raises:
            StopIteration: If experiment name not found
        """
        results_a = next(
            r for r in self.results_history if r["experiment_name"] == experiment_a
        )
        results_b = next(
            r for r in self.results_history if r["experiment_name"] == experiment_b
        )

        recalls_a = [r["recall"] for r in results_a["detailed_results"]]
        recalls_b = [r["recall"] for r in results_b["detailed_results"]]

        return paired_t_test(recalls_a, recalls_b)


# =============================================================================
# Evaluation Gate for CI/CD
# =============================================================================


def run_evaluation_gate(
    retriever: Any,
    eval_data_path: str | Path,
    baseline_path: str | Path,
    k: int = 10,
) -> None:
    """
    Run evaluation and fail if regression detected.

    Use this in CI/CD pipelines to prevent merging changes that
    degrade retrieval quality.

    Args:
        retriever: Retriever object with search(query, top_k) method
        eval_data_path: Path to evaluation dataset JSON
        baseline_path: Path to baseline metrics JSON
        k: Number of results to retrieve

    Raises:
        ValueError: If regression detected
    """
    evaluator = RAGEvaluator(retriever, k=k)

    # Load evaluation dataset
    eval_data = json.loads(Path(eval_data_path).read_text())

    # Run evaluation
    results = evaluator.evaluate(eval_data, experiment_name="current")

    # Compare to baseline
    baseline = json.loads(Path(baseline_path).read_text())

    if results["avg_recall"] < baseline["avg_recall"] - 0.02:
        raise ValueError(
            f"Recall regression: {baseline['avg_recall']:.3f} -> "
            f"{results['avg_recall']:.3f}"
        )

    print(f"Evaluation passed: recall={results['avg_recall']:.3f}")
