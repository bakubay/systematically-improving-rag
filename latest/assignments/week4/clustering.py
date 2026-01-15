"""
Week 4: Query Clustering System

Cluster queries to identify patterns and improvement opportunities.
Uses K-means clustering with UMAP visualization.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("sentence-transformers not installed. Run: uv add sentence-transformers")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not installed. Run: uv add scikit-learn")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("umap-learn not installed. Run: uv add umap-learn")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not installed. Run: uv add matplotlib")


@dataclass
class QueryData:
    """Container for query with metadata."""
    text: str
    satisfaction: float = 0.5  # 0.0 to 1.0
    query_id: Optional[str] = None
    # Optional: if you have it, add query count/frequency from logs.
    # If this is None, we treat each query as one “unit” of volume.
    frequency: Optional[int] = None


def create_mock_queries(n: int = 200) -> list[QueryData]:
    """Create mock query data for testing."""
    np.random.seed(42)
    
    # Define query templates by category
    categories = {
        "factual": [
            "What is {}?",
            "Define {}",
            "Who invented {}?",
            "When was {} created?",
            "Where is {} located?",
        ],
        "how_to": [
            "How do I {}?",
            "How to {} step by step?",
            "What's the best way to {}?",
            "Can you explain how to {}?",
        ],
        "comparison": [
            "What's the difference between {} and {}?",
            "Compare {} vs {}",
            "{} or {}: which is better?",
        ],
        "troubleshooting": [
            "Why is {} not working?",
            "{} error: how to fix?",
            "Problem with {}: help needed",
            "{} fails when I try to {}",
        ],
    }
    
    topics = [
        "Python", "JavaScript", "machine learning", "RAG", "embeddings",
        "vector database", "LLM", "API", "authentication", "deployment",
        "Docker", "Kubernetes", "database", "caching", "optimization",
        "testing", "debugging", "monitoring", "logging", "security",
    ]
    
    queries = []
    for i in range(n):
        # Pick random category and template
        category = np.random.choice(list(categories.keys()))
        template = np.random.choice(categories[category])
        
        # Fill in topics
        if "{}" in template:
            n_slots = template.count("{}")
            selected_topics = np.random.choice(topics, size=n_slots, replace=False)
            query_text = template.format(*selected_topics)
        else:
            query_text = template
        
        # Assign satisfaction score (some categories have lower satisfaction)
        base_satisfaction = {
            "factual": 0.8,
            "how_to": 0.7,
            "comparison": 0.6,
            "troubleshooting": 0.4,
        }[category]
        satisfaction = np.clip(
            base_satisfaction + np.random.normal(0, 0.15),
            0.0, 1.0
        )
        
        queries.append(QueryData(
            text=query_text,
            satisfaction=satisfaction,
            query_id=f"q_{i}"
        ))
    
    return queries


class QueryClusterer:
    """Cluster queries and analyze patterns."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers required. Run: uv add sentence-transformers")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. Run: uv add scikit-learn")
        
        self.model = SentenceTransformer(model_name)
        self.queries: list[QueryData] = []
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_analysis: dict = {}
    
    def fit(self, queries: list[QueryData], n_clusters: int = 10):
        """Embed queries and cluster them."""
        self.queries = queries
        
        # Embed all queries
        print(f"Embedding {len(queries)} queries...")
        texts = [q.text for q in queries]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Cluster
        print(f"Clustering into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # Calculate silhouette score
        silhouette = silhouette_score(self.embeddings, self.cluster_labels)
        print(f"Silhouette score: {silhouette:.3f}")
        
        # Analyze clusters
        self._analyze_clusters()
        
        return self
    
    def find_optimal_k(self, k_range: range = range(5, 25)) -> int:
        """Find optimal number of clusters using elbow method."""
        if self.embeddings is None:
            raise ValueError("Must embed queries first. Call fit() with some n_clusters.")
        
        inertias = []
        silhouettes = []
        
        print("Finding optimal k...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.embeddings, labels))
        
        # Find elbow (simplified: max silhouette)
        optimal_k = k_range[np.argmax(silhouettes)]
        print(f"Optimal k by silhouette: {optimal_k}")
        
        return optimal_k
    
    def _analyze_clusters(self):
        """Analyze each cluster for patterns."""
        self.cluster_analysis = {}
        n_clusters = len(set(self.cluster_labels))
        
        for cluster_id in range(n_clusters):
            # Get queries in this cluster
            mask = self.cluster_labels == cluster_id
            cluster_queries = [q for q, m in zip(self.queries, mask) if m]
            
            # Calculate statistics
            satisfactions = [q.satisfaction for q in cluster_queries]
            avg_satisfaction = np.mean(satisfactions) if satisfactions else 0.0
            
            # Find failure examples (low satisfaction)
            failures = sorted(cluster_queries, key=lambda q: q.satisfaction)[:10]
            
            self.cluster_analysis[cluster_id] = {
                "volume": len(cluster_queries),
                "volume_pct": len(cluster_queries) / len(self.queries) * 100,
                "avg_satisfaction": avg_satisfaction,
                "sample_queries": [q.text for q in cluster_queries[:5]],
                "failure_examples": [q.text for q in failures[:5]],
            }

    @staticmethod
    def expected_value(volume_pct: float, success_rate: float, impact: float) -> float:
        """Expected Value (EV) = impact × volume% × success_rate.

        Why this is useful:
        - Volume% tells you how often users hit this segment.
        - Success rate tells you how often you already succeed.
        - Impact is how much value you get when you fix it (0-1).

        In many products, “impact” is a business guess:
        - Revenue-impacting queries might be 1.0
        - Nice-to-have queries might be 0.3
        """
        return impact * (volume_pct / 100.0) * success_rate

    @staticmethod
    def detect_user_adaptation_signals(queries: list[str]) -> dict:
        """Detect simple signals that users are adapting around system limits.

        This is not perfect. It’s a cheap heuristic that helps you decide
        which clusters to inspect first.
        """
        lowered = [q.lower() for q in queries]
        signals = {
            "contains_please_help": sum("help" in q or "please" in q for q in lowered),
            "contains_error_terms": sum("error" in q or "not working" in q or "fails" in q for q in lowered),
            "very_short_queries": sum(len(q.split()) <= 3 for q in lowered),
            "many_question_marks": sum(q.count("?") >= 2 for q in lowered),
        }
        signals["total"] = len(queries)
        return signals
    
    def get_prioritization_matrix(self) -> list[dict]:
        """Generate prioritization matrix (volume vs satisfaction)."""
        if not self.cluster_analysis:
            raise ValueError("Must fit clusters first")
        
        median_volume = np.median([c["volume_pct"] for c in self.cluster_analysis.values()])
        median_satisfaction = np.median([c["avg_satisfaction"] for c in self.cluster_analysis.values()])
        
        results = []
        for cluster_id, data in self.cluster_analysis.items():
            # Classify quadrant
            high_volume = data["volume_pct"] > median_volume
            low_satisfaction = data["avg_satisfaction"] < median_satisfaction
            
            if high_volume and low_satisfaction:
                quadrant = "PRIORITY"  # High volume, low satisfaction
                priority_score = data["volume_pct"] * (1 - data["avg_satisfaction"])
            elif high_volume and not low_satisfaction:
                quadrant = "MAINTAIN"  # High volume, high satisfaction
                priority_score = 0.3 * data["volume_pct"]
            elif not high_volume and low_satisfaction:
                quadrant = "MONITOR"  # Low volume, low satisfaction
                priority_score = 0.5 * data["volume_pct"] * (1 - data["avg_satisfaction"])
            else:
                quadrant = "NICHE"  # Low volume, high satisfaction
                priority_score = 0.1 * data["volume_pct"]
            
            results.append({
                "cluster_id": cluster_id,
                "volume_pct": data["volume_pct"],
                "avg_satisfaction": data["avg_satisfaction"],
                "quadrant": quadrant,
                "priority_score": priority_score,
                "sample_queries": data["sample_queries"],
                # New: Expected Value (EV) style score.
                # We treat success_rate as avg_satisfaction (a proxy).
                # We treat impact as (1 - avg_satisfaction), meaning the worse it is,
                # the more upside there is if you fix it.
                "expected_value": self.expected_value(
                    volume_pct=data["volume_pct"],
                    success_rate=data["avg_satisfaction"],
                    impact=(1.0 - data["avg_satisfaction"]),
                ),
                "adaptation_signals": self.detect_user_adaptation_signals(data["sample_queries"]),
            })
        
        # Sort by priority
        results.sort(key=lambda x: x["priority_score"], reverse=True)
        return results
    
    def print_report(self):
        """Print analysis report."""
        print("\n" + "=" * 70)
        print("QUERY CLUSTERING ANALYSIS REPORT")
        print("=" * 70)
        print(f"Total queries: {len(self.queries)}")
        print(f"Number of clusters: {len(self.cluster_analysis)}")
        
        # Prioritization matrix
        print("\n" + "-" * 70)
        print("PRIORITIZATION MATRIX")
        print("-" * 70)
        
        matrix = self.get_prioritization_matrix()
        
        print(f"\n{'Rank':<5} {'Cluster':<8} {'Volume%':<10} {'Satisf.':<10} {'Quadrant':<12} {'Priority':<10}")
        print("-" * 70)
        
        for i, item in enumerate(matrix, 1):
            print(f"{i:<5} {item['cluster_id']:<8} {item['volume_pct']:<10.1f} "
                  f"{item['avg_satisfaction']:<10.2f} {item['quadrant']:<12} "
                  f"{item['priority_score']:<10.2f}")

        print("\n" + "-" * 70)
        print("EXPECTED VALUE (EV) VIEW (Impact × Volume% × Success Rate)")
        print("-" * 70)
        print("EV is a ranking helper. It is not a truth machine. Use it to decide what to inspect.")

        top_ev = sorted(matrix, key=lambda x: x["expected_value"], reverse=True)[:5]
        for item in top_ev:
            print(
                f"Cluster {item['cluster_id']}: EV={item['expected_value']:.4f} "
                f"(volume={item['volume_pct']:.1f}%, success≈{item['avg_satisfaction']:.2f})"
            )
        
        # Top priority clusters
        print("\n" + "-" * 70)
        print("TOP 3 PRIORITY CLUSTERS (High Volume, Low Satisfaction)")
        print("-" * 70)
        
        priority_clusters = [c for c in matrix if c["quadrant"] == "PRIORITY"][:3]
        
        for item in priority_clusters:
            print(f"\nCluster {item['cluster_id']}:")
            print(f"  Volume: {item['volume_pct']:.1f}%")
            print(f"  Satisfaction: {item['avg_satisfaction']:.2f}")
            print("  Sample queries:")
            for q in item["sample_queries"][:3]:
                print(f"    - {q}")
        
        print("\n" + "=" * 70)
    
    def plot_clusters(self, output_path: Optional[str] = None):
        """Create UMAP visualization of clusters."""
        if not UMAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print("Skipping visualization (umap-learn or matplotlib not installed)")
            return
        
        print("\nCreating UMAP visualization...")
        
        # Reduce to 2D
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding_2d = reducer.fit_transform(self.embeddings)
        
        # Get satisfaction scores
        satisfactions = [q.satisfaction for q in self.queries]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Colored by cluster
        scatter1 = axes[0].scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=self.cluster_labels,
            cmap='tab20',
            s=30,
            alpha=0.6
        )
        axes[0].set_title("Queries Colored by Cluster")
        axes[0].set_xlabel("UMAP Dimension 1")
        axes[0].set_ylabel("UMAP Dimension 2")
        plt.colorbar(scatter1, ax=axes[0], label="Cluster")
        
        # Plot 2: Colored by satisfaction
        scatter2 = axes[1].scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=satisfactions,
            cmap='RdYlGn',
            s=30,
            alpha=0.6
        )
        axes[1].set_title("Queries Colored by Satisfaction")
        axes[1].set_xlabel("UMAP Dimension 1")
        axes[1].set_ylabel("UMAP Dimension 2")
        plt.colorbar(scatter2, ax=axes[1], label="Satisfaction")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Run the clustering analysis."""
    print("=" * 70)
    print("WEEK 4: QUERY CLUSTERING SYSTEM")
    print("=" * 70)
    
    # Create mock data
    print("\nGenerating mock query data...")
    queries = create_mock_queries(n=200)
    print(f"Generated {len(queries)} queries")
    
    # Print sample
    print("\nSample queries:")
    for q in queries[:5]:
        print(f"  - {q.text} (satisfaction: {q.satisfaction:.2f})")
    
    # Initialize clusterer
    clusterer = QueryClusterer()
    
    # Fit clusters
    clusterer.fit(queries, n_clusters=8)
    
    # Print report
    clusterer.print_report()
    
    # Create visualization
    output_dir = Path(__file__).parent
    clusterer.plot_clusters(output_path=str(output_dir / "cluster_visualization.png"))
    
    # Save results
    results = {
        "total_queries": len(queries),
        "n_clusters": len(clusterer.cluster_analysis),
        "prioritization": clusterer.get_prioritization_matrix(),
    }
    
    with open(output_dir / "clustering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'clustering_results.json'}")
    
    return results


if __name__ == "__main__":
    main()
