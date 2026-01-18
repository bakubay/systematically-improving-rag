"""
Chapter 2: Training Data and Fine-Tuning - Code Examples

This module provides fine-tuning infrastructure for RAG systems, including:
- Training data preparation (triplets with hard negatives)
- Embedding model fine-tuning with Sentence Transformers
- Re-ranker implementation and fine-tuning
- Loss functions (triplet loss, InfoNCE)
- Training utilities (gradient accumulation, early stopping)

Usage:
    from chapter2_finetuning import (
        TrainingExample,
        Triplet,
        prepare_training_data,
        fine_tune_embedding_model,
        ReRanker,
        TwoStageRetriever,
    )

    # Prepare training data
    triplets = prepare_training_data(eval_examples, corpus, embedding_model)

    # Fine-tune embedding model
    model = fine_tune_embedding_model(triplets, base_model="BAAI/bge-base-en-v1.5")

    # Use re-ranker
    reranker = ReRanker()
    reranked_docs = reranker.rerank(query, documents)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class TrainingExample(BaseModel):
    """A training example with anchor, positive, and optional negative."""

    anchor: str
    positive: str
    negative: str | None = None


@dataclass
class Triplet:
    """A triplet for contrastive learning.

    Attributes:
        anchor: The query text
        positive: A document that is relevant to the query
        negative: A document that is not relevant to the query
    """

    anchor: str
    positive: str
    negative: str


class Difficulty(str, Enum):
    """Difficulty levels for training examples."""

    EASY = "easy"  # Random negatives
    MEDIUM = "medium"  # In-batch negatives
    HARD = "hard"  # Mined hard negatives


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning.

    Attributes:
        base_model: Name of the base model to fine-tune
        output_dir: Directory to save the fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        warmup_ratio: Ratio of warmup steps
        margin: Margin for triplet loss
        temperature: Temperature for InfoNCE loss
        gradient_accumulation_steps: Number of steps to accumulate gradients
        early_stopping_patience: Number of epochs without improvement before stopping
        save_best_only: Whether to save only the best model
    """

    base_model: str = "BAAI/bge-base-en-v1.5"
    output_dir: str = "models/fine_tuned"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    margin: float = 1.0
    temperature: float = 0.07
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 2
    save_best_only: bool = True


@dataclass
class TrainingMetrics:
    """Metrics tracked during training.

    Attributes:
        epoch: Current epoch number
        train_loss: Training loss for the epoch
        val_recall: Validation recall at k
        best_val_recall: Best validation recall seen so far
        learning_rate: Current learning rate
        timestamp: When metrics were recorded
    """

    epoch: int
    train_loss: float
    val_recall: float | None = None
    best_val_recall: float = 0.0
    learning_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Protocols for Type Safety
# =============================================================================


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode text(s) into embeddings."""
        ...


class CrossEncoderModel(Protocol):
    """Protocol for cross-encoder models."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Predict relevance scores for query-document pairs."""
        ...


# =============================================================================
# Similarity Functions
# =============================================================================


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score between -1 and 1

    Examples:
        >>> a = np.array([1.0, 0.0])
        >>> b = np.array([1.0, 0.0])
        >>> cosine_similarity(a, b)
        1.0
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between a query and multiple documents.

    Args:
        query_embedding: Query embedding vector (1D)
        doc_embeddings: Document embeddings matrix (2D: num_docs x embedding_dim)

    Returns:
        Array of similarity scores
    """
    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)

    # Avoid division by zero
    query_norm = max(query_norm, 1e-8)
    doc_norms = np.maximum(doc_norms, 1e-8)

    similarities = np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm)
    return similarities


# =============================================================================
# Hard Negative Mining
# =============================================================================


def mine_hard_negatives(
    query: str,
    positive_doc: str,
    corpus: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    num_negatives: int = 5,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """
    Mine hard negatives using embedding similarity.

    Hard negatives are documents that are similar to the query but not
    actually relevant. They are more informative for training than
    random negatives.

    Args:
        query: The query text
        positive_doc: The positive (relevant) document text
        corpus: List of document dicts with 'id' and 'text' keys
        embedding_model: Model to generate embeddings
        num_negatives: Number of hard negatives to return
        exclude_ids: Set of document IDs to exclude (e.g., positive doc IDs)

    Returns:
        List of hard negative document texts

    Examples:
        >>> negatives = mine_hard_negatives(
        ...     query="password reset",
        ...     positive_doc="How to reset your password...",
        ...     corpus=documents,
        ...     embedding_model=model,
        ...     num_negatives=3
        ... )
    """
    if exclude_ids is None:
        exclude_ids = set()

    # Encode query
    query_embedding = embedding_model.encode(query)

    # Score all documents
    candidates: list[tuple[str, str, float]] = []

    for doc in corpus:
        doc_id = doc.get("id", "")
        doc_text = doc.get("text", "")

        # Skip positive document and excluded IDs
        if doc_text == positive_doc or doc_id in exclude_ids:
            continue

        doc_embedding = embedding_model.encode(doc_text)
        similarity = cosine_similarity(query_embedding, doc_embedding)
        candidates.append((doc_id, doc_text, similarity))

    # Sort by similarity (highest first) - these are hard negatives
    candidates.sort(key=lambda x: x[2], reverse=True)

    return [text for _, text, _ in candidates[:num_negatives]]


def mine_hard_negatives_batch(
    queries: list[str],
    positive_docs: list[str],
    corpus: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    num_negatives: int = 5,
) -> list[list[str]]:
    """
    Mine hard negatives for multiple queries efficiently.

    Args:
        queries: List of query texts
        positive_docs: List of positive document texts (one per query)
        corpus: List of document dicts
        embedding_model: Model to generate embeddings
        num_negatives: Number of hard negatives per query

    Returns:
        List of lists of hard negative texts
    """
    # Pre-compute corpus embeddings
    logger.info(f"Computing embeddings for {len(corpus)} documents...")
    corpus_texts = [doc.get("text", "") for doc in corpus]
    corpus_embeddings = embedding_model.encode(corpus_texts)

    all_negatives = []

    for query, positive_doc in zip(queries, positive_docs):
        query_embedding = embedding_model.encode(query)

        # Compute similarities
        similarities = batch_cosine_similarity(query_embedding, corpus_embeddings)

        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # Filter out positive document and get top negatives
        negatives = []
        for idx in sorted_indices:
            if corpus_texts[idx] != positive_doc and len(negatives) < num_negatives:
                negatives.append(corpus_texts[idx])

        all_negatives.append(negatives)

    return all_negatives


# =============================================================================
# Training Data Preparation
# =============================================================================


def prepare_training_data(
    evaluation_examples: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    num_negatives_per_positive: int = 3,
) -> list[Triplet]:
    """
    Convert evaluation data to training triplets with hard negatives.

    This function takes evaluation examples (query + relevant doc IDs) and
    creates training triplets by:
    1. Pairing each query with its relevant documents (positives)
    2. Mining hard negatives for each positive

    Args:
        evaluation_examples: List of dicts with 'query' and 'relevant_doc_ids'
        corpus: List of document dicts with 'id' and 'text'
        embedding_model: Model for mining hard negatives
        num_negatives_per_positive: Number of negatives per positive document

    Returns:
        List of Triplet objects for training

    Examples:
        >>> triplets = prepare_training_data(
        ...     evaluation_examples=[
        ...         {"query": "password reset", "relevant_doc_ids": ["doc1"]}
        ...     ],
        ...     corpus=[{"id": "doc1", "text": "How to reset..."}],
        ...     embedding_model=model
        ... )
    """
    # Create corpus lookup
    corpus_lookup = {doc["id"]: doc for doc in corpus}

    triplets: list[Triplet] = []
    total_examples = len(evaluation_examples)

    for i, example in enumerate(evaluation_examples):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing example {i + 1}/{total_examples}")

        query = example["query"]
        relevant_ids = set(example.get("relevant_doc_ids", []))

        # Get positive documents
        positives = [
            corpus_lookup[doc_id]
            for doc_id in relevant_ids
            if doc_id in corpus_lookup
        ]

        if not positives:
            continue

        # Mine hard negatives for each positive
        for positive in positives:
            negatives = mine_hard_negatives(
                query=query,
                positive_doc=positive["text"],
                corpus=corpus,
                embedding_model=embedding_model,
                num_negatives=num_negatives_per_positive,
                exclude_ids=relevant_ids,
            )

            for negative in negatives:
                triplets.append(
                    Triplet(
                        anchor=query,
                        positive=positive["text"],
                        negative=negative,
                    )
                )

    logger.info(f"Created {len(triplets)} training triplets")
    return triplets


def create_triplets_from_rag_logs(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    cited_doc_ids: set[str],
) -> list[Triplet]:
    """
    Create triplets from RAG interaction logs.

    Documents that were cited are positives. Documents that were
    retrieved but not cited are hard negatives (the system thought
    they were relevant, but the user/LLM didn't use them).

    Args:
        query: The user query
        retrieved_docs: List of retrieved documents with 'id' and 'text'
        cited_doc_ids: Set of document IDs that were actually cited

    Returns:
        List of Triplet objects
    """
    triplets = []

    positives = [d for d in retrieved_docs if d["id"] in cited_doc_ids]
    negatives = [d for d in retrieved_docs if d["id"] not in cited_doc_ids]

    for positive in positives:
        for negative in negatives:
            triplets.append(
                Triplet(
                    anchor=query,
                    positive=positive["text"],
                    negative=negative["text"],
                )
            )

    return triplets


def save_training_data(
    triplets: list[Triplet],
    output_path: str | Path,
) -> None:
    """
    Save training triplets to JSON file.

    Args:
        triplets: List of Triplet objects
        output_path: Path to save the JSON file
    """
    data = [
        {
            "anchor": t.anchor,
            "positive": t.positive,
            "negative": t.negative,
        }
        for t in triplets
    ]

    Path(output_path).write_text(json.dumps(data, indent=2))
    logger.info(f"Saved {len(triplets)} triplets to {output_path}")


def load_training_data(input_path: str | Path) -> list[Triplet]:
    """
    Load training triplets from JSON file.

    Args:
        input_path: Path to the JSON file

    Returns:
        List of Triplet objects
    """
    data = json.loads(Path(input_path).read_text())
    triplets = [
        Triplet(
            anchor=item["anchor"],
            positive=item["positive"],
            negative=item["negative"],
        )
        for item in data
    ]
    logger.info(f"Loaded {len(triplets)} triplets from {input_path}")
    return triplets


# =============================================================================
# Loss Functions
# =============================================================================


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Compute triplet loss.

    The loss ensures the anchor is closer to the positive than to the
    negative by at least a margin.

    L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin)

    Args:
        anchor: Anchor embeddings (batch_size x embedding_dim)
        positive: Positive embeddings (batch_size x embedding_dim)
        negative: Negative embeddings (batch_size x embedding_dim)
        margin: Minimum margin between positive and negative distances

    Returns:
        Scalar loss tensor

    Examples:
        >>> anchor = torch.randn(32, 768)
        >>> positive = torch.randn(32, 768)
        >>> negative = torch.randn(32, 768)
        >>> loss = triplet_loss(anchor, positive, negative)
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Compute InfoNCE (contrastive) loss.

    Treats the problem as classification: given an anchor, identify
    the positive among many negatives.

    L = -log(exp(sim(a,p)/τ) / Σ exp(sim(a,n_i)/τ))

    Args:
        anchor: Anchor embeddings (batch_size x embedding_dim)
        positive: Positive embeddings (batch_size x embedding_dim)
        negatives: Negative embeddings (num_negatives x embedding_dim)
        temperature: Temperature parameter (lower = sharper distribution)

    Returns:
        Scalar loss tensor

    Examples:
        >>> anchor = torch.randn(32, 768)
        >>> positive = torch.randn(32, 768)
        >>> negatives = torch.randn(100, 768)
        >>> loss = info_nce_loss(anchor, positive, negatives)
    """
    # Normalize embeddings
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Compute similarities
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature
    neg_sim = torch.matmul(anchor, negatives.T) / temperature

    # Concatenate positive and negative similarities
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

    # Labels: positive is always at index 0
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

    return F.cross_entropy(logits, labels)


def multiple_negatives_ranking_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Compute Multiple Negatives Ranking Loss.

    Uses other examples in the batch as negatives, making training
    more efficient (no need to explicitly mine negatives).

    Args:
        anchor: Anchor embeddings (batch_size x embedding_dim)
        positive: Positive embeddings (batch_size x embedding_dim)
        temperature: Temperature parameter

    Returns:
        Scalar loss tensor
    """
    # Normalize embeddings
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)

    # Compute similarity matrix (anchor x all positives)
    similarity_matrix = torch.matmul(anchor, positive.T) / temperature

    # Labels: diagonal elements are the correct matches
    labels = torch.arange(anchor.size(0), device=anchor.device)

    return F.cross_entropy(similarity_matrix, labels)


# =============================================================================
# Fine-Tuning
# =============================================================================


def fine_tune_embedding_model(
    training_data: list[Triplet],
    config: TrainingConfig | None = None,
    validation_data: list[dict[str, Any]] | None = None,
    corpus: list[dict[str, Any]] | None = None,
) -> Any:
    """
    Fine-tune an embedding model with triplet loss.

    This function uses Sentence Transformers to fine-tune a bi-encoder
    model on triplet data.

    Args:
        training_data: List of Triplet objects
        config: Training configuration (uses defaults if None)
        validation_data: Optional validation examples for early stopping
        corpus: Optional corpus for validation evaluation

    Returns:
        Fine-tuned SentenceTransformer model

    Examples:
        >>> triplets = [Triplet("query", "relevant doc", "irrelevant doc")]
        >>> model = fine_tune_embedding_model(triplets)
    """
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    if config is None:
        config = TrainingConfig()

    logger.info(f"Loading base model: {config.base_model}")
    model = SentenceTransformer(config.base_model)

    # Convert to InputExamples
    train_examples = [
        InputExample(texts=[t.anchor, t.positive, t.negative])
        for t in training_data
    ]

    logger.info(f"Created {len(train_examples)} training examples")

    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.batch_size,
    )

    # Use triplet loss
    train_loss = losses.TripletLoss(model=model)

    # Calculate warmup steps
    num_training_steps = len(train_dataloader) * config.epochs
    warmup_steps = int(num_training_steps * config.warmup_ratio)

    logger.info(f"Training for {config.epochs} epochs")
    logger.info(f"Total steps: {num_training_steps}, Warmup steps: {warmup_steps}")

    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        optimizer_params={"lr": config.learning_rate},
        show_progress_bar=True,
    )

    logger.info(f"Model saved to {config.output_dir}")
    return model


def fine_tune_with_validation(
    training_data: list[Triplet],
    validation_examples: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    config: TrainingConfig | None = None,
    k: int = 10,
) -> Any:
    """
    Fine-tune with validation-based early stopping.

    Evaluates on validation set after each epoch and stops if
    recall doesn't improve for `patience` epochs.

    Args:
        training_data: List of Triplet objects
        validation_examples: List of dicts with 'query' and 'relevant_doc_ids'
        corpus: List of document dicts for validation retrieval
        config: Training configuration
        k: Number of results for recall calculation

    Returns:
        Best fine-tuned model based on validation recall
    """
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    if config is None:
        config = TrainingConfig()

    logger.info(f"Loading base model: {config.base_model}")
    model = SentenceTransformer(config.base_model)

    # Convert to InputExamples
    train_examples = [
        InputExample(texts=[t.anchor, t.positive, t.negative])
        for t in training_data
    ]

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.batch_size,
    )

    train_loss = losses.TripletLoss(model=model)

    # Pre-compute corpus embeddings for validation
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_ids = [doc["id"] for doc in corpus]

    best_recall = 0.0
    patience_counter = 0
    best_model_path = Path(config.output_dir) / "best_model"
    best_model_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")

        # Training
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=0,
            output_path=None,
            show_progress_bar=True,
        )

        # Validation
        corpus_embeddings = model.encode(corpus_texts, convert_to_numpy=True)
        recalls = []

        for example in validation_examples:
            query = example["query"]
            relevant_ids = set(example["relevant_doc_ids"])

            query_embedding = model.encode(query, convert_to_numpy=True)
            similarities = batch_cosine_similarity(query_embedding, corpus_embeddings)

            # Get top-k
            top_indices = np.argsort(similarities)[::-1][:k]
            retrieved_ids = {corpus_ids[i] for i in top_indices}

            recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
            recalls.append(recall)

        val_recall = sum(recalls) / len(recalls)
        logger.info(f"Validation Recall@{k}: {val_recall:.4f}")

        if val_recall > best_recall:
            best_recall = val_recall
            patience_counter = 0
            model.save(str(best_model_path))
            logger.info(f"New best model saved (recall: {best_recall:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load best model
    return SentenceTransformer(str(best_model_path))


# =============================================================================
# Re-Ranker
# =============================================================================


class ReRanker:
    """
    Re-ranker using cross-encoder model.

    Cross-encoders process query-document pairs together, allowing
    full attention between both texts for more accurate relevance scoring.

    Attributes:
        model: The cross-encoder model
        model_name: Name of the model being used

    Examples:
        >>> reranker = ReRanker()
        >>> reranked = reranker.rerank("password reset", documents)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the re-ranker.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        logger.info(f"Loaded re-ranker model: {model_name}")

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Re-rank documents using cross-encoder.

        Args:
            query: The query string
            documents: List of document dicts with 'text' key
            top_k: Number of top documents to return

        Returns:
            List of documents sorted by relevance score
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def rerank_batch(
        self,
        queries: list[str],
        documents_list: list[list[dict[str, Any]]],
        top_k: int = 10,
    ) -> list[list[dict[str, Any]]]:
        """
        Re-rank multiple query-document sets.

        Args:
            queries: List of query strings
            documents_list: List of document lists (one per query)
            top_k: Number of top documents to return per query

        Returns:
            List of re-ranked document lists
        """
        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)
        return results


class TwoStageRetriever:
    """
    Two-stage retrieval: bi-encoder then cross-encoder.

    This is the standard production pattern: use a fast bi-encoder
    to retrieve candidates, then use a slower but more accurate
    cross-encoder to re-rank the top results.

    Attributes:
        bi_encoder: The bi-encoder model for initial retrieval
        cross_encoder: The cross-encoder model for re-ranking
        initial_k: Number of candidates from bi-encoder
        final_k: Number of final results after re-ranking

    Examples:
        >>> retriever = TwoStageRetriever(bi_encoder, cross_encoder)
        >>> results = retriever.retrieve("password reset", documents)
    """

    def __init__(
        self,
        bi_encoder: EmbeddingModel,
        cross_encoder: CrossEncoderModel,
        initial_k: int = 100,
        final_k: int = 10,
    ):
        """
        Initialize the two-stage retriever.

        Args:
            bi_encoder: Bi-encoder model for initial retrieval
            cross_encoder: Cross-encoder model for re-ranking
            initial_k: Number of candidates from bi-encoder
            final_k: Number of final results after re-ranking
        """
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.initial_k = initial_k
        self.final_k = final_k

    def retrieve(
        self,
        query: str,
        documents: list[dict[str, Any]],
        doc_embeddings: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Two-stage retrieval: bi-encoder then cross-encoder.

        Args:
            query: The query string
            documents: List of document dicts with 'text' key
            doc_embeddings: Pre-computed document embeddings (optional)

        Returns:
            List of top documents after re-ranking
        """
        # Stage 1: Fast retrieval with bi-encoder
        query_embedding = self.bi_encoder.encode(query)

        if doc_embeddings is None:
            doc_texts = [doc["text"] for doc in documents]
            doc_embeddings = self.bi_encoder.encode(doc_texts)

        similarities = batch_cosine_similarity(query_embedding, doc_embeddings)

        # Get top initial_k candidates
        top_indices = np.argsort(similarities)[::-1][: self.initial_k]
        candidates = [documents[i].copy() for i in top_indices]

        for i, idx in enumerate(top_indices):
            candidates[i]["bi_encoder_score"] = float(similarities[idx])

        # Stage 2: Re-rank with cross-encoder
        pairs = [(query, doc["text"]) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[: self.final_k]


# =============================================================================
# Training Utilities
# =============================================================================


def train_with_gradient_accumulation(
    model: torch.nn.Module,
    dataloader: Any,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int = 4,
    device: str = "cuda",
) -> float:
    """
    Train with gradient accumulation for larger effective batch size.

    Gradient accumulation allows training with larger effective batch
    sizes on limited GPU memory.

    Args:
        model: The model to train
        dataloader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        accumulation_steps: Number of steps to accumulate gradients
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    model.to(device)
    optimizer.zero_grad()

    total_loss = 0.0
    num_batches = 0

    for i, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        loss = loss_fn(batch) / accumulation_steps

        # Backward pass
        loss.backward()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining gradients
    if num_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / num_batches


class EarlyStopping:
    """
    Early stopping handler.

    Stops training when validation metric doesn't improve for
    a specified number of epochs.

    Attributes:
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum change to qualify as improvement
        best_score: Best validation score seen
        counter: Number of epochs without improvement
        should_stop: Whether training should stop

    Examples:
        >>> early_stopping = EarlyStopping(patience=3)
        >>> for epoch in range(100):
        ...     val_score = evaluate(model)
        ...     if early_stopping(val_score):
        ...         break
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (higher is better)

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_fine_tuning(
    original_model: EmbeddingModel,
    fine_tuned_model: EmbeddingModel,
    eval_examples: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    k: int = 10,
) -> dict[str, Any]:
    """
    Compare original vs fine-tuned model performance.

    Args:
        original_model: The original embedding model
        fine_tuned_model: The fine-tuned embedding model
        eval_examples: List of dicts with 'query' and 'relevant_doc_ids'
        corpus: List of document dicts
        k: Number of results for recall calculation

    Returns:
        Dict with original and fine-tuned recall, and improvement
    """
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_ids = [doc["id"] for doc in corpus]

    results = {"original": [], "fine_tuned": []}

    # Pre-compute embeddings
    logger.info("Computing original model embeddings...")
    orig_embeddings = original_model.encode(corpus_texts)

    logger.info("Computing fine-tuned model embeddings...")
    ft_embeddings = fine_tuned_model.encode(corpus_texts)

    for example in eval_examples:
        query = example["query"]
        relevant_ids = set(example["relevant_doc_ids"])

        # Evaluate original
        orig_query_emb = original_model.encode(query)
        orig_sims = batch_cosine_similarity(orig_query_emb, orig_embeddings)
        orig_top_k = np.argsort(orig_sims)[::-1][:k]
        orig_retrieved = {corpus_ids[i] for i in orig_top_k}
        orig_recall = len(orig_retrieved & relevant_ids) / len(relevant_ids)
        results["original"].append(orig_recall)

        # Evaluate fine-tuned
        ft_query_emb = fine_tuned_model.encode(query)
        ft_sims = batch_cosine_similarity(ft_query_emb, ft_embeddings)
        ft_top_k = np.argsort(ft_sims)[::-1][:k]
        ft_retrieved = {corpus_ids[i] for i in ft_top_k}
        ft_recall = len(ft_retrieved & relevant_ids) / len(relevant_ids)
        results["fine_tuned"].append(ft_recall)

    orig_avg = sum(results["original"]) / len(results["original"])
    ft_avg = sum(results["fine_tuned"]) / len(results["fine_tuned"])

    return {
        "original_avg_recall": orig_avg,
        "fine_tuned_avg_recall": ft_avg,
        "improvement": ft_avg - orig_avg,
        "improvement_pct": (ft_avg - orig_avg) / orig_avg * 100 if orig_avg > 0 else 0,
        "num_examples": len(eval_examples),
    }


def compare_models(
    original_retriever: Any,
    fine_tuned_retriever: Any,
    reranker: ReRanker,
    eval_examples: list[dict[str, Any]],
    k: int = 10,
) -> dict[str, Any]:
    """
    Compare original, fine-tuned, and re-ranked retrieval.

    Args:
        original_retriever: Original retriever with search(query, top_k) method
        fine_tuned_retriever: Fine-tuned retriever
        reranker: ReRanker instance
        eval_examples: List of evaluation examples
        k: Number of results for recall calculation

    Returns:
        Dict with recall for each approach
    """
    results = {
        "original": [],
        "fine_tuned": [],
        "fine_tuned_reranked": [],
    }

    for example in eval_examples:
        query = example["query"]
        relevant_ids = set(example["relevant_doc_ids"])

        # Original
        orig_results = original_retriever.search(query, top_k=k)
        orig_recall = len({r["id"] for r in orig_results} & relevant_ids) / len(
            relevant_ids
        )
        results["original"].append(orig_recall)

        # Fine-tuned
        ft_results = fine_tuned_retriever.search(query, top_k=k)
        ft_recall = len({r["id"] for r in ft_results} & relevant_ids) / len(
            relevant_ids
        )
        results["fine_tuned"].append(ft_recall)

        # Fine-tuned + Re-ranked
        ft_candidates = fine_tuned_retriever.search(query, top_k=k * 3)
        reranked = reranker.rerank(query, ft_candidates, top_k=k)
        rr_recall = len({r["id"] for r in reranked} & relevant_ids) / len(relevant_ids)
        results["fine_tuned_reranked"].append(rr_recall)

    return {
        "original_recall": sum(results["original"]) / len(results["original"]),
        "fine_tuned_recall": sum(results["fine_tuned"]) / len(results["fine_tuned"]),
        "reranked_recall": sum(results["fine_tuned_reranked"])
        / len(results["fine_tuned_reranked"]),
    }
