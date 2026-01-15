"""
Week 2: Embedding Fine-tuning with Hard Negative Mining

Fine-tune sentence transformers using triplet loss and hard negatives.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import random

# Check dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("sentence-transformers not installed. Run: uv add sentence-transformers")

DATASETS_AVAILABLE = False  # Not currently used, but available via: uv add datasets


@dataclass
class Triplet:
    """A training triplet: anchor, positive, negative."""
    anchor: str
    positive: str
    negative: str


def create_mock_training_data(n_samples: int = 100) -> list[dict]:
    """Create mock anchor-positive pairs for training."""
    random.seed(42)
    
    # Topics with related queries and documents
    topics = {
        "python": {
            "queries": [
                "How to install Python?",
                "Python syntax basics",
                "What is Python used for?",
                "Python vs JavaScript",
                "Best Python libraries",
            ],
            "documents": [
                "Python is a high-level programming language. To install Python, download it from python.org.",
                "Python syntax uses indentation for code blocks. Variables don't need type declarations.",
                "Python is used for web development, data science, AI, and automation.",
                "Python is dynamically typed while JavaScript runs in browsers. Both are popular.",
                "Popular Python libraries include NumPy, Pandas, TensorFlow, and Django.",
            ]
        },
        "machine_learning": {
            "queries": [
                "What is machine learning?",
                "Types of machine learning",
                "How to start with ML?",
                "Machine learning vs AI",
                "Best ML frameworks",
            ],
            "documents": [
                "Machine learning is a subset of AI where systems learn from data without explicit programming.",
                "There are three types: supervised, unsupervised, and reinforcement learning.",
                "Start with Python, learn math basics, then try scikit-learn and simple datasets.",
                "ML is a technique within AI. AI is broader and includes rule-based systems too.",
                "Popular ML frameworks include TensorFlow, PyTorch, scikit-learn, and XGBoost.",
            ]
        },
        "databases": {
            "queries": [
                "SQL vs NoSQL databases",
                "How to choose a database?",
                "What is PostgreSQL?",
                "Database indexing explained",
                "Vector databases overview",
            ],
            "documents": [
                "SQL databases are relational with structured schemas. NoSQL is flexible and scales horizontally.",
                "Choose based on data structure, scale needs, consistency requirements, and team expertise.",
                "PostgreSQL is an open-source relational database known for reliability and features.",
                "Indexes speed up queries by creating data structures for faster lookups. B-trees are common.",
                "Vector databases store embeddings for similarity search. Examples: Pinecone, Chroma, Weaviate.",
            ]
        },
    }
    
    pairs = []
    for topic, data in topics.items():
        for query, doc in zip(data["queries"], data["documents"]):
            pairs.append({
                "anchor": query,
                "positive": doc,
                "topic": topic,
            })
    
    # Expand to n_samples with variations
    while len(pairs) < n_samples:
        base = random.choice(pairs[:15])  # Original pairs
        pairs.append({
            "anchor": base["anchor"] + " " + random.choice(["please", "thanks", "help"]),
            "positive": base["positive"],
            "topic": base["topic"],
        })
    
    return pairs[:n_samples]


class HardNegativeMiner:
    """Mine hard negatives from a corpus."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers required")
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required")
        
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.corpus_texts: list[str] = []
    
    def index_corpus(self, documents: list[str]):
        """Index documents for mining."""
        print(f"Indexing {len(documents)} documents...")
        self.corpus_texts = documents
        self.corpus_embeddings = self.model.encode(
            documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    def mine_hard_negatives(
        self,
        anchor: str,
        positive: str,
        n_negatives: int = 5,
        range_min: int = 5,
        range_max: int = 50,
    ) -> list[str]:
        """
        Mine hard negatives for an anchor-positive pair.
        
        Hard negatives are documents that are similar to the anchor
        but not the positive document.
        """
        if self.corpus_embeddings is None:
            raise ValueError("Must index corpus first")
        
        # Embed anchor
        anchor_embedding = self.model.encode([anchor], convert_to_numpy=True)[0]
        
        # Calculate similarities
        similarities = np.dot(self.corpus_embeddings, anchor_embedding)
        
        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Find positive index to exclude
        try:
            positive_idx = self.corpus_texts.index(positive)
        except ValueError:
            positive_idx = -1
        
        # Select hard negatives from range
        negatives = []
        for idx in sorted_indices[range_min:range_max]:
            if idx != positive_idx and len(negatives) < n_negatives:
                negatives.append(self.corpus_texts[idx])
        
        return negatives
    
    def create_triplets(
        self,
        pairs: list[dict],
        n_negatives: int = 3,
    ) -> list[Triplet]:
        """Create triplets from anchor-positive pairs."""
        triplets = []
        
        print(f"Creating triplets for {len(pairs)} pairs...")
        for i, pair in enumerate(pairs):
            negatives = self.mine_hard_negatives(
                anchor=pair["anchor"],
                positive=pair["positive"],
                n_negatives=n_negatives,
            )
            
            for neg in negatives:
                triplets.append(Triplet(
                    anchor=pair["anchor"],
                    positive=pair["positive"],
                    negative=neg,
                ))
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(pairs)} pairs")
        
        print(f"Created {len(triplets)} triplets")
        return triplets


class TripletEvaluator:
    """Evaluate triplet quality and model improvement."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers required")
        
        self.model = SentenceTransformer(model_name)
    
    def evaluate_triplets(self, triplets: list[Triplet]) -> dict:
        """
        Evaluate triplet quality by checking if positive is closer than negative.
        """
        correct = 0
        margins = []
        
        for triplet in triplets:
            embeddings = self.model.encode([
                triplet.anchor,
                triplet.positive,
                triplet.negative,
            ])
            
            anchor_emb, pos_emb, neg_emb = embeddings
            
            # Calculate cosine similarities
            pos_sim = np.dot(anchor_emb, pos_emb) / (
                np.linalg.norm(anchor_emb) * np.linalg.norm(pos_emb)
            )
            neg_sim = np.dot(anchor_emb, neg_emb) / (
                np.linalg.norm(anchor_emb) * np.linalg.norm(neg_emb)
            )
            
            margin = pos_sim - neg_sim
            margins.append(margin)
            
            if pos_sim > neg_sim:
                correct += 1
        
        return {
            "accuracy": correct / len(triplets) if triplets else 0.0,
            "avg_margin": float(np.mean(margins)) if margins else 0.0,
            "min_margin": float(np.min(margins)) if margins else 0.0,
            "max_margin": float(np.max(margins)) if margins else 0.0,
        }
    
    def compare_models(
        self,
        model_names: list[str],
        triplets: list[Triplet],
    ) -> dict:
        """Compare multiple models on the same triplets."""
        results = {}
        
        for model_name in model_names:
            print(f"\nEvaluating {model_name}...")
            self.model = SentenceTransformer(model_name)
            results[model_name] = self.evaluate_triplets(triplets)
        
        return results


def simulate_fine_tuning(triplets: list[Triplet]) -> dict:
    """
    Simulate fine-tuning results.
    
    In production, you would use:
    ```python
    from sentence_transformers import SentenceTransformerTrainer, losses
    
    loss = losses.TripletLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=triplet_dataset,
        loss=loss,
    )
    trainer.train()
    ```
    """
    # Simulate improvement
    baseline_accuracy = 0.75
    improved_accuracy = 0.88
    
    return {
        "baseline": {
            "accuracy": baseline_accuracy,
            "avg_margin": 0.12,
        },
        "fine_tuned": {
            "accuracy": improved_accuracy,
            "avg_margin": 0.25,
        },
        "improvement": {
            "accuracy_gain": improved_accuracy - baseline_accuracy,
            "relative_improvement": (improved_accuracy - baseline_accuracy) / baseline_accuracy,
        }
    }


def main():
    """Run the fine-tuning pipeline."""
    print("=" * 60)
    print("WEEK 2: EMBEDDING FINE-TUNING")
    print("=" * 60)
    
    # Create training data
    print("\nCreating mock training data...")
    pairs = create_mock_training_data(n_samples=50)
    print(f"Created {len(pairs)} anchor-positive pairs")
    
    # Extract all documents for corpus
    all_docs = list(set(p["positive"] for p in pairs))
    print(f"Corpus size: {len(all_docs)} unique documents")
    
    # Initialize miner
    print("\nInitializing hard negative miner...")
    miner = HardNegativeMiner()
    miner.index_corpus(all_docs)
    
    # Create triplets
    print("\nMining hard negatives...")
    triplets = miner.create_triplets(pairs, n_negatives=2)
    
    # Show sample triplets
    print("\n" + "-" * 60)
    print("SAMPLE TRIPLETS")
    print("-" * 60)
    
    for i, t in enumerate(triplets[:3]):
        print(f"\nTriplet {i + 1}:")
        print(f"  Anchor:   {t.anchor[:60]}...")
        print(f"  Positive: {t.positive[:60]}...")
        print(f"  Negative: {t.negative[:60]}...")
    
    # Evaluate triplet quality
    print("\n" + "-" * 60)
    print("TRIPLET QUALITY EVALUATION")
    print("-" * 60)
    
    evaluator = TripletEvaluator()
    eval_results = evaluator.evaluate_triplets(triplets[:50])
    
    print(f"\nTriplet Accuracy: {eval_results['accuracy']:.1%}")
    print(f"Average Margin: {eval_results['avg_margin']:.4f}")
    print(f"Margin Range: [{eval_results['min_margin']:.4f}, {eval_results['max_margin']:.4f}]")
    
    # Simulate fine-tuning
    print("\n" + "-" * 60)
    print("FINE-TUNING SIMULATION")
    print("-" * 60)
    
    ft_results = simulate_fine_tuning(triplets)
    
    print("\nBaseline Model:")
    print(f"  Accuracy: {ft_results['baseline']['accuracy']:.1%}")
    print(f"  Avg Margin: {ft_results['baseline']['avg_margin']:.4f}")
    
    print("\nFine-tuned Model:")
    print(f"  Accuracy: {ft_results['fine_tuned']['accuracy']:.1%}")
    print(f"  Avg Margin: {ft_results['fine_tuned']['avg_margin']:.4f}")
    
    print("\nImprovement:")
    print(f"  Accuracy Gain: +{ft_results['improvement']['accuracy_gain']:.1%}")
    print(f"  Relative: +{ft_results['improvement']['relative_improvement']:.1%}")
    
    # Save triplets
    output_dir = Path(__file__).parent
    triplets_path = output_dir / "triplets.json"
    
    with open(triplets_path, "w") as f:
        json.dump(
            [{"anchor": t.anchor, "positive": t.positive, "negative": t.negative} 
             for t in triplets],
            f, indent=2
        )
    print(f"\nTriplets saved to: {triplets_path}")
    
    print("\n" + "=" * 60)
    print("Fine-tuning pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
