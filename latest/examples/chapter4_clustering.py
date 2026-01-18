"""
Chapter 4: Query Understanding and Prioritization - Code Examples

This module provides query clustering and prioritization infrastructure:
- Query clustering with K-means and embeddings
- Cluster labeling with LLMs
- 2x2 prioritization matrix implementation
- Expected Value formula for prioritization
- Topic and capability detection
- Production classification pipelines

Usage:
    from chapter4_clustering import (
        QueryCluster,
        cluster_queries,
        label_clusters_with_llm,
        classify_segment,
        calculate_priority_score,
        detect_capabilities,
    )

    # Cluster queries
    clusters = await cluster_queries(queries, satisfaction_scores)

    # Label clusters
    labels = await label_clusters_with_llm(clusters)

    # Prioritize segments
    priority = calculate_priority_score(impact=8, volume_pct=0.15, success_rate=0.4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class Quadrant(str, Enum):
    """Quadrants in the 2x2 prioritization matrix."""

    MONITOR = "monitor"  # High volume, high satisfaction
    PROMOTE = "promote"  # Low volume, high satisfaction
    DANGER_ZONE = "danger_zone"  # High volume, low satisfaction
    COST_BENEFIT = "cost_benefit"  # Low volume, low satisfaction


class Capability(str, Enum):
    """Common query capabilities."""

    SUMMARIZE = "summarize"
    COMPARE = "compare"
    EXPLAIN = "explain"
    STEP_BY_STEP = "step_by_step"
    LOOKUP = "lookup"
    CALCULATE = "calculate"
    FILTER = "filter"
    TEMPORAL = "temporal"
    AGGREGATE = "aggregate"


class IssueType(str, Enum):
    """Types of retrieval issues."""

    INVENTORY = "inventory"  # Missing data
    CAPABILITY = "capability"  # Missing feature/processing


class QueryCluster(BaseModel):
    """A cluster of similar queries.

    Attributes:
        cluster_id: Unique identifier for the cluster
        queries: List of queries in this cluster
        centroid: Cluster centroid embedding
        size: Number of queries in cluster
        avg_satisfaction: Average satisfaction score
        label: Human-readable label (set by LLM)
        description: Description of the cluster
    """

    cluster_id: int
    queries: list[str]
    centroid: list[float]
    size: int
    avg_satisfaction: float
    label: str | None = None
    description: str | None = None


class SegmentPriority(BaseModel):
    """Priority assessment for a query segment.

    Attributes:
        cluster_id: Cluster identifier
        cluster_name: Human-readable name
        volume_pct: Percentage of total queries
        satisfaction: Satisfaction score (0-1)
        quadrant: Position in 2x2 matrix
        priority_score: Calculated priority score
        recommended_action: Suggested action
    """

    cluster_id: int
    cluster_name: str
    volume_pct: float
    satisfaction: float
    quadrant: Quadrant
    priority_score: float
    recommended_action: str


class QueryAnalysis(BaseModel):
    """Analysis of a single query.

    Attributes:
        query: The original query
        topic: Detected topic
        capabilities: Detected capabilities
        complexity: Estimated complexity (0-1)
        issue_type: Type of issue if retrieval failed
    """

    query: str
    topic: str | None = None
    capabilities: list[Capability] = []
    complexity: float = 0.5
    issue_type: IssueType | None = None


@dataclass
class ClusteringResult:
    """Result of query clustering.

    Attributes:
        clusters: List of QueryCluster objects
        silhouette_score: Clustering quality metric
        total_queries: Total number of queries clustered
        optimal_k: Optimal number of clusters (if determined)
    """

    clusters: list[QueryCluster]
    silhouette_score: float
    total_queries: int
    optimal_k: int | None = None


# =============================================================================
# Embedding Protocol
# =============================================================================


class EmbeddingModel:
    """Protocol for embedding models."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        raise NotImplementedError


class SimpleEmbeddingModel:
    """Simple embedding model using sentence-transformers.

    Attributes:
        model: The sentence transformer model
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)


# =============================================================================
# Query Clustering
# =============================================================================


async def cluster_queries(
    queries: list[str],
    satisfaction_scores: list[float],
    embedding_model: EmbeddingModel | None = None,
    n_clusters: int = 20,
) -> list[QueryCluster]:
    """
    Cluster queries using embeddings and K-means.

    Start with 20 clusters and adjust based on results.
    More clusters = more specific segments.
    Fewer clusters = broader patterns.

    Args:
        queries: List of query strings
        satisfaction_scores: Satisfaction score for each query (0-1)
        embedding_model: Model to generate embeddings
        n_clusters: Number of clusters

    Returns:
        List of QueryCluster objects

    Examples:
        >>> clusters = await cluster_queries(
        ...     queries=["password reset", "forgot password", "pricing info"],
        ...     satisfaction_scores=[0.8, 0.7, 0.9],
        ...     n_clusters=2
        ... )
    """
    if embedding_model is None:
        embedding_model = SimpleEmbeddingModel()

    logger.info(f"Generating embeddings for {len(queries)} queries...")
    embeddings = embedding_model.encode(queries)

    logger.info(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Build cluster objects
    clusters = []
    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_queries = [q for q, m in zip(queries, mask) if m]
        cluster_scores = [s for s, m in zip(satisfaction_scores, mask) if m]

        clusters.append(
            QueryCluster(
                cluster_id=i,
                queries=cluster_queries,
                centroid=kmeans.cluster_centers_[i].tolist(),
                size=len(cluster_queries),
                avg_satisfaction=float(np.mean(cluster_scores)) if cluster_scores else 0.0,
            )
        )

    logger.info(f"Created {len(clusters)} clusters")
    return clusters


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: tuple[int, int] = (5, 30),
) -> int:
    """
    Find optimal number of clusters using silhouette score.

    Args:
        embeddings: Query embeddings
        k_range: Range of k values to try

    Returns:
        Optimal number of clusters
    """
    best_k = k_range[0]
    best_score = -1

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)

        if score > best_score:
            best_score = score
            best_k = k

    logger.info(f"Optimal k={best_k} with silhouette score={best_score:.3f}")
    return best_k


async def cluster_queries_with_optimization(
    queries: list[str],
    satisfaction_scores: list[float],
    embedding_model: EmbeddingModel | None = None,
    k_range: tuple[int, int] = (5, 30),
) -> ClusteringResult:
    """
    Cluster queries with automatic k optimization.

    Args:
        queries: List of query strings
        satisfaction_scores: Satisfaction scores
        embedding_model: Embedding model
        k_range: Range of k values to try

    Returns:
        ClusteringResult with clusters and metadata
    """
    if embedding_model is None:
        embedding_model = SimpleEmbeddingModel()

    embeddings = embedding_model.encode(queries)
    optimal_k = find_optimal_k(embeddings, k_range)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    sil_score = silhouette_score(embeddings, cluster_labels)

    clusters = []
    for i in range(optimal_k):
        mask = cluster_labels == i
        cluster_queries = [q for q, m in zip(queries, mask) if m]
        cluster_scores = [s for s, m in zip(satisfaction_scores, mask) if m]

        clusters.append(
            QueryCluster(
                cluster_id=i,
                queries=cluster_queries,
                centroid=kmeans.cluster_centers_[i].tolist(),
                size=len(cluster_queries),
                avg_satisfaction=float(np.mean(cluster_scores)) if cluster_scores else 0.0,
            )
        )

    return ClusteringResult(
        clusters=clusters,
        silhouette_score=sil_score,
        total_queries=len(queries),
        optimal_k=optimal_k,
    )


# =============================================================================
# Cluster Labeling
# =============================================================================


async def label_clusters_with_llm(
    clusters: list[QueryCluster],
    llm_client: Any = None,
    samples_per_cluster: int = 10,
) -> dict[int, dict[str, str]]:
    """
    Use an LLM to generate meaningful names for each cluster.

    Args:
        clusters: List of QueryCluster objects
        llm_client: LLM client with chat.completions.create method
        samples_per_cluster: Number of sample queries to show LLM

    Returns:
        Dict mapping cluster_id to label info

    Examples:
        >>> labels = await label_clusters_with_llm(clusters)
        >>> print(labels[0]['name'])
        'Password Reset Queries'
    """
    labels = {}

    for cluster in clusters:
        # Sample queries from cluster
        sample = cluster.queries[:samples_per_cluster]

        prompt = f"""Analyze these queries and provide:
1. A short name (2-4 words) for this cluster
2. A one-sentence description
3. 3 good examples of what belongs in this cluster
4. 2 examples of what does NOT belong

Queries:
{chr(10).join(f'- {q}' for q in sample)}

Respond in this exact format:
NAME: <cluster name>
DESCRIPTION: <one sentence description>
BELONGS: <example 1>, <example 2>, <example 3>
NOT_BELONGS: <example 1>, <example 2>
"""

        if llm_client:
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
        else:
            # Demo response
            content = f"""NAME: Cluster {cluster.cluster_id}
DESCRIPTION: Queries related to common topics
BELONGS: example1, example2, example3
NOT_BELONGS: unrelated1, unrelated2"""

        # Parse response
        lines = content.strip().split("\n")
        label_info = {}
        for line in lines:
            if line.startswith("NAME:"):
                label_info["name"] = line.replace("NAME:", "").strip()
            elif line.startswith("DESCRIPTION:"):
                label_info["description"] = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("BELONGS:"):
                label_info["belongs"] = line.replace("BELONGS:", "").strip()
            elif line.startswith("NOT_BELONGS:"):
                label_info["not_belongs"] = line.replace("NOT_BELONGS:", "").strip()

        labels[cluster.cluster_id] = label_info

    return labels


# =============================================================================
# Prioritization
# =============================================================================


def classify_segment(
    volume_pct: float,
    satisfaction: float,
    volume_threshold: float = 0.10,
    satisfaction_threshold: float = 0.60,
) -> Quadrant:
    """
    Classify a segment into the 2x2 matrix.

    Args:
        volume_pct: Percentage of total queries (0-1)
        satisfaction: Satisfaction score (0-1)
        volume_threshold: Threshold for high volume
        satisfaction_threshold: Threshold for high satisfaction

    Returns:
        Quadrant classification

    Examples:
        >>> classify_segment(0.15, 0.35)  # High volume, low satisfaction
        <Quadrant.DANGER_ZONE: 'danger_zone'>
    """
    high_volume = volume_pct >= volume_threshold
    high_satisfaction = satisfaction >= satisfaction_threshold

    if high_volume and high_satisfaction:
        return Quadrant.MONITOR
    elif not high_volume and high_satisfaction:
        return Quadrant.PROMOTE
    elif high_volume and not high_satisfaction:
        return Quadrant.DANGER_ZONE
    else:
        return Quadrant.COST_BENEFIT


def calculate_priority_score(
    impact: float,
    volume_pct: float,
    success_rate: float,
    effort: float = 5.0,
    risk: float = 2.0,
) -> float:
    """
    Calculate priority score using the Expected Value formula.

    Priority = (Impact x Volume %) / (Effort x Risk)

    Higher scores = higher priority.

    Args:
        impact: Expected impact if fixed (1-10 scale)
        volume_pct: Percentage of total queries (0-1)
        success_rate: Current success rate (0-1)
        effort: Estimated effort to fix (1-10 scale)
        risk: Risk of failure (1-5 scale)

    Returns:
        Priority score (higher = more important)

    Examples:
        >>> calculate_priority_score(impact=8, volume_pct=0.15, success_rate=0.4)
        0.072
    """
    # Opportunity = volume * (1 - success_rate)
    opportunity = volume_pct * (1 - success_rate)
    return (impact * opportunity) / (effort * risk)


def get_recommended_action(quadrant: Quadrant) -> str:
    """
    Get recommended action for a quadrant.

    Args:
        quadrant: The quadrant classification

    Returns:
        Recommended action string
    """
    actions = {
        Quadrant.MONITOR: "Monitor for regressions. Use as examples of what works.",
        Quadrant.PROMOTE: "Add UI hints showing these capabilities. Include in onboarding.",
        Quadrant.DANGER_ZONE: "Immediate priority. Conduct user research and set sprint goals.",
        Quadrant.COST_BENEFIT: "Evaluate if this is in scope. Consider explicitly rejecting.",
    }
    return actions[quadrant]


def prioritize_clusters(
    clusters: list[QueryCluster],
    total_queries: int,
    impact_estimator: Callable[[QueryCluster], float] | None = None,
) -> list[SegmentPriority]:
    """
    Prioritize clusters using the 2x2 matrix and Expected Value formula.

    Args:
        clusters: List of QueryCluster objects
        total_queries: Total number of queries
        impact_estimator: Optional function to estimate impact

    Returns:
        List of SegmentPriority objects, sorted by priority
    """
    priorities = []

    for cluster in clusters:
        volume_pct = cluster.size / total_queries
        satisfaction = cluster.avg_satisfaction
        quadrant = classify_segment(volume_pct, satisfaction)

        # Estimate impact (default: inverse of satisfaction)
        if impact_estimator:
            impact = impact_estimator(cluster)
        else:
            impact = (1 - satisfaction) * 10

        priority_score = calculate_priority_score(
            impact=impact,
            volume_pct=volume_pct,
            success_rate=satisfaction,
        )

        priorities.append(
            SegmentPriority(
                cluster_id=cluster.cluster_id,
                cluster_name=cluster.label or f"Cluster {cluster.cluster_id}",
                volume_pct=volume_pct,
                satisfaction=satisfaction,
                quadrant=quadrant,
                priority_score=priority_score,
                recommended_action=get_recommended_action(quadrant),
            )
        )

    # Sort by priority score (descending)
    priorities.sort(key=lambda x: x.priority_score, reverse=True)
    return priorities


# =============================================================================
# Capability Detection
# =============================================================================


def detect_capabilities(query: str) -> list[Capability]:
    """
    Detect capabilities requested in a query.

    Args:
        query: The user's query

    Returns:
        List of detected Capability enums

    Examples:
        >>> detect_capabilities("Compare the Pro and Basic plans")
        [<Capability.COMPARE: 'compare'>]
    """
    query_lower = query.lower()
    capabilities = []

    # Summarization patterns
    if any(word in query_lower for word in ["summarize", "summary", "overview", "brief"]):
        capabilities.append(Capability.SUMMARIZE)

    # Comparison patterns
    if any(word in query_lower for word in ["compare", "difference", "vs", "versus", "between"]):
        capabilities.append(Capability.COMPARE)

    # Explanation patterns
    if any(word in query_lower for word in ["explain", "why", "how does", "what is"]):
        capabilities.append(Capability.EXPLAIN)

    # Step-by-step patterns
    if any(word in query_lower for word in ["how to", "steps", "guide", "tutorial", "process"]):
        capabilities.append(Capability.STEP_BY_STEP)

    # Lookup patterns
    if any(word in query_lower for word in ["find", "search", "look up", "where is", "what is the"]):
        capabilities.append(Capability.LOOKUP)

    # Calculation patterns
    if any(word in query_lower for word in ["calculate", "compute", "how much", "total", "sum"]):
        capabilities.append(Capability.CALCULATE)

    # Filter patterns
    if any(word in query_lower for word in ["filter", "only", "just", "specific", "particular"]):
        capabilities.append(Capability.FILTER)

    # Temporal patterns
    if any(word in query_lower for word in ["2024", "2023", "last year", "this month", "recent", "latest"]):
        capabilities.append(Capability.TEMPORAL)

    # Aggregation patterns
    if any(word in query_lower for word in ["all", "every", "list all", "aggregate", "combine"]):
        capabilities.append(Capability.AGGREGATE)

    return capabilities


async def analyze_query(
    query: str,
    topic_classifier: Callable[[str], str] | None = None,
) -> QueryAnalysis:
    """
    Analyze a query for topic, capabilities, and complexity.

    Args:
        query: The user's query
        topic_classifier: Optional function to classify topic

    Returns:
        QueryAnalysis object
    """
    capabilities = detect_capabilities(query)

    # Estimate complexity based on query length and capabilities
    word_count = len(query.split())
    complexity = min(1.0, (word_count / 20) * 0.5 + len(capabilities) * 0.1)

    # Classify topic if classifier provided
    topic = None
    if topic_classifier:
        topic = topic_classifier(query)

    return QueryAnalysis(
        query=query,
        topic=topic,
        capabilities=capabilities,
        complexity=complexity,
    )


# =============================================================================
# Issue Classification
# =============================================================================


def classify_issue_type(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    response_quality: float,
) -> IssueType | None:
    """
    Determine if a retrieval failure is an inventory or capability issue.

    Inventory issues: The information doesn't exist in your corpus
    Capability issues: The information exists but wasn't retrieved/processed correctly

    Args:
        query: The user's query
        retrieved_docs: Documents that were retrieved
        response_quality: Quality score of the response (0-1)

    Returns:
        IssueType if there's an issue, None if no issue

    Examples:
        >>> classify_issue_type("latest Q4 earnings", [], 0.2)
        <IssueType.INVENTORY: 'inventory'>
    """
    if response_quality >= 0.7:
        return None  # No significant issue

    # If no documents were retrieved, likely inventory issue
    if not retrieved_docs:
        return IssueType.INVENTORY

    # If documents were retrieved but quality is low, likely capability issue
    if len(retrieved_docs) >= 3 and response_quality < 0.5:
        return IssueType.CAPABILITY

    # Check if retrieved docs seem relevant (simple heuristic)
    query_words = set(query.lower().split())
    relevant_docs = 0
    for doc in retrieved_docs:
        doc_text = doc.get("text", "").lower()
        overlap = len(query_words & set(doc_text.split()))
        if overlap >= 2:
            relevant_docs += 1

    if relevant_docs < len(retrieved_docs) * 0.3:
        return IssueType.INVENTORY
    else:
        return IssueType.CAPABILITY


def get_issue_solution(issue_type: IssueType) -> str:
    """
    Get recommended solution for an issue type.

    Args:
        issue_type: The type of issue

    Returns:
        Recommended solution string
    """
    solutions = {
        IssueType.INVENTORY: (
            "Add missing data to your corpus. Consider: "
            "1) Identifying data sources that contain this information, "
            "2) Creating synthetic documents if data doesn't exist, "
            "3) Explicitly rejecting these queries if out of scope."
        ),
        IssueType.CAPABILITY: (
            "Improve retrieval or processing. Consider: "
            "1) Fine-tuning embeddings on this query type, "
            "2) Adding a re-ranker for better precision, "
            "3) Improving chunking strategy, "
            "4) Adding query expansion or rewriting."
        ),
    }
    return solutions[issue_type]


# =============================================================================
# User Adaptation Detection
# =============================================================================


@dataclass
class AdaptationPattern:
    """Pattern indicating user adaptation to system limitations.

    Attributes:
        pattern_type: Type of adaptation detected
        evidence: Evidence for the pattern
        masked_issue: The underlying issue being masked
        recommendation: Recommended action
    """

    pattern_type: str
    evidence: str
    masked_issue: str
    recommendation: str


def detect_user_adaptation(
    query_history: list[dict[str, Any]],
    time_window_days: int = 30,
) -> list[AdaptationPattern]:
    """
    Detect patterns where users have adapted to system limitations.

    User adaptation can mask system failures - users learn what
    doesn't work and stop trying, leading to artificially high
    satisfaction scores.

    Args:
        query_history: List of query records with timestamps and satisfaction
        time_window_days: Time window to analyze

    Returns:
        List of detected AdaptationPattern objects

    Examples:
        >>> patterns = detect_user_adaptation(query_history)
        >>> for p in patterns:
        ...     print(f"{p.pattern_type}: {p.masked_issue}")
    """
    patterns = []

    # Group queries by user
    user_queries: dict[str, list[dict]] = {}
    for q in query_history:
        user_id = q.get("user_id", "anonymous")
        if user_id not in user_queries:
            user_queries[user_id] = []
        user_queries[user_id].append(q)

    for user_id, queries in user_queries.items():
        # Sort by timestamp
        queries.sort(key=lambda x: x.get("timestamp", datetime.now()))

        # Pattern 1: Query simplification over time
        if len(queries) >= 5:
            early_avg_length = np.mean([len(q.get("query", "").split()) for q in queries[:3]])
            late_avg_length = np.mean([len(q.get("query", "").split()) for q in queries[-3:]])

            if late_avg_length < early_avg_length * 0.6:
                patterns.append(
                    AdaptationPattern(
                        pattern_type="query_simplification",
                        evidence=f"User {user_id}: query length dropped from {early_avg_length:.1f} to {late_avg_length:.1f} words",
                        masked_issue="Complex queries may be failing, users learned to simplify",
                        recommendation="Investigate complex query handling, improve capability for longer queries",
                    )
                )

        # Pattern 2: Topic avoidance
        early_topics = set()
        late_topics = set()
        for q in queries[:len(queries) // 2]:
            caps = detect_capabilities(q.get("query", ""))
            early_topics.update(c.value for c in caps)
        for q in queries[len(queries) // 2:]:
            caps = detect_capabilities(q.get("query", ""))
            late_topics.update(c.value for c in caps)

        avoided_topics = early_topics - late_topics
        if avoided_topics:
            patterns.append(
                AdaptationPattern(
                    pattern_type="topic_avoidance",
                    evidence=f"User {user_id} stopped asking about: {', '.join(avoided_topics)}",
                    masked_issue="Users may have learned these capabilities don't work well",
                    recommendation=f"Investigate performance for: {', '.join(avoided_topics)}",
                )
            )

        # Pattern 3: Decreasing engagement
        if len(queries) >= 10:
            early_frequency = len(queries[:5]) / 5
            late_frequency = len(queries[-5:]) / 5

            # This is a simplified check - in production you'd use actual timestamps
            if late_frequency < early_frequency * 0.5:
                patterns.append(
                    AdaptationPattern(
                        pattern_type="decreasing_engagement",
                        evidence=f"User {user_id}: query frequency dropped significantly",
                        masked_issue="User may be disengaging due to poor experiences",
                        recommendation="Conduct user research to understand why engagement dropped",
                    )
                )

    return patterns


# =============================================================================
# Production Classification
# =============================================================================


class QueryClassifier:
    """Production classifier for routing queries.

    Attributes:
        cluster_centroids: Centroids of known clusters
        cluster_labels: Labels for each cluster
        embedding_model: Model for generating embeddings
    """

    def __init__(
        self,
        cluster_centroids: np.ndarray,
        cluster_labels: list[str],
        embedding_model: EmbeddingModel | None = None,
    ):
        self.cluster_centroids = cluster_centroids
        self.cluster_labels = cluster_labels
        self.embedding_model = embedding_model or SimpleEmbeddingModel()

    def classify(self, query: str) -> tuple[str, float]:
        """
        Classify a query into a known cluster.

        Args:
            query: The query to classify

        Returns:
            Tuple of (cluster_label, confidence)
        """
        embedding = self.embedding_model.encode([query])[0]

        # Find nearest centroid
        distances = np.linalg.norm(self.cluster_centroids - embedding, axis=1)
        nearest_idx = np.argmin(distances)
        min_distance = distances[nearest_idx]

        # Convert distance to confidence (smaller distance = higher confidence)
        # Using exponential decay
        confidence = np.exp(-min_distance)

        return self.cluster_labels[nearest_idx], float(confidence)

    def classify_batch(self, queries: list[str]) -> list[tuple[str, float]]:
        """
        Classify multiple queries.

        Args:
            queries: List of queries to classify

        Returns:
            List of (cluster_label, confidence) tuples
        """
        embeddings = self.embedding_model.encode(queries)
        results = []

        for embedding in embeddings:
            distances = np.linalg.norm(self.cluster_centroids - embedding, axis=1)
            nearest_idx = np.argmin(distances)
            min_distance = distances[nearest_idx]
            confidence = np.exp(-min_distance)
            results.append((self.cluster_labels[nearest_idx], float(confidence)))

        return results


def build_classifier_from_clusters(
    clusters: list[QueryCluster],
    embedding_model: EmbeddingModel | None = None,
) -> QueryClassifier:
    """
    Build a production classifier from clustering results.

    Args:
        clusters: List of QueryCluster objects
        embedding_model: Embedding model to use

    Returns:
        QueryClassifier ready for production use
    """
    centroids = np.array([c.centroid for c in clusters])
    labels = [c.label or f"Cluster {c.cluster_id}" for c in clusters]

    return QueryClassifier(
        cluster_centroids=centroids,
        cluster_labels=labels,
        embedding_model=embedding_model,
    )


# =============================================================================
# Visualization Helpers
# =============================================================================


def generate_priority_report(
    priorities: list[SegmentPriority],
    top_n: int = 10,
) -> str:
    """
    Generate a text report of prioritized segments.

    Args:
        priorities: List of SegmentPriority objects
        top_n: Number of top priorities to include

    Returns:
        Formatted report string
    """
    lines = ["# Query Segment Priority Report", ""]

    # Summary by quadrant
    quadrant_counts = {}
    for p in priorities:
        quadrant_counts[p.quadrant] = quadrant_counts.get(p.quadrant, 0) + 1

    lines.append("## Summary by Quadrant")
    for quadrant, count in quadrant_counts.items():
        lines.append(f"- {quadrant.value}: {count} segments")
    lines.append("")

    # Top priorities
    lines.append(f"## Top {top_n} Priorities")
    for i, p in enumerate(priorities[:top_n], 1):
        lines.append(f"### {i}. {p.cluster_name}")
        lines.append(f"- **Quadrant**: {p.quadrant.value}")
        lines.append(f"- **Volume**: {p.volume_pct:.1%}")
        lines.append(f"- **Satisfaction**: {p.satisfaction:.1%}")
        lines.append(f"- **Priority Score**: {p.priority_score:.4f}")
        lines.append(f"- **Action**: {p.recommended_action}")
        lines.append("")

    # Danger zones
    danger_zones = [p for p in priorities if p.quadrant == Quadrant.DANGER_ZONE]
    if danger_zones:
        lines.append("## DANGER ZONES (Immediate Attention)")
        for p in danger_zones:
            lines.append(f"- **{p.cluster_name}**: {p.volume_pct:.1%} volume, {p.satisfaction:.1%} satisfaction")
        lines.append("")

    return "\n".join(lines)
