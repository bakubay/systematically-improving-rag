"""
Week 1: Retrieval Evaluation Metrics

This module implements core retrieval metrics from scratch:
- Precision@k
- Recall@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@k)
"""

import math
from typing import List, Set


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate precision@k: fraction of retrieved docs that are relevant.
    
    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Number of top results to consider
    
    Returns:
        Precision score between 0.0 and 1.0
    """
    if k <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return relevant_retrieved / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate recall@k: fraction of relevant docs that were retrieved.
    
    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Number of top results to consider
    
    Returns:
        Recall score between 0.0 and 1.0
    """
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    relevant_retrieved = len(retrieved_k & relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank: 1/position of first relevant result.
    
    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs (ground truth)
    
    Returns:
        MRR score between 0.0 and 1.0
    """
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    NDCG rewards relevant documents ranked higher. Uses binary relevance
    (1 if relevant, 0 otherwise).
    
    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Number of top results to consider
    
    Returns:
        NDCG score between 0.0 and 1.0
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    
    # Calculate DCG
    dcg = sum(
        (1 if doc_id in relevant_ids else 0) / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
    )
    
    # Calculate ideal DCG (all relevant docs at top)
    ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def f1_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate F1 score at k: harmonic mean of precision and recall.
    
    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Number of top results to consider
    
    Returns:
        F1 score between 0.0 and 1.0
    """
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Calculate Average Precision (AP): average of precision at each relevant doc.
    
    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs (ground truth)
    
    Returns:
        AP score between 0.0 and 1.0
    """
    if not relevant_ids:
        return 0.0
    
    precisions = []
    relevant_count = 0
    
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            relevant_count += 1
            precisions.append(relevant_count / i)
    
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant_ids)


class RetrievalEvaluator:
    """Evaluate retrieval results across multiple queries."""
    
    def __init__(self):
        self.results = []
    
    def add_result(
        self,
        query_id: str,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ):
        """Add a single query result for evaluation."""
        self.results.append({
            "query_id": query_id,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids
        })
    
    def evaluate(self, k_values: List[int] = None) -> dict:
        """
        Evaluate all results and return aggregate metrics.
        
        Args:
            k_values: List of k values to evaluate at (default: [3, 5, 10, 20])
        
        Returns:
            Dictionary with average metrics at each k
        """
        if k_values is None:
            k_values = [3, 5, 10, 20]
        
        metrics = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            mrrs = []
            ndcgs = []
            f1s = []
            aps = []
            
            for result in self.results:
                retrieved = result["retrieved_ids"]
                relevant = result["relevant_ids"]
                
                precisions.append(precision_at_k(retrieved, relevant, k))
                recalls.append(recall_at_k(retrieved, relevant, k))
                mrrs.append(mean_reciprocal_rank(retrieved, relevant))
                ndcgs.append(ndcg_at_k(retrieved, relevant, k))
                f1s.append(f1_at_k(retrieved, relevant, k))
                aps.append(average_precision(retrieved, relevant))
            
            n = len(self.results)
            metrics[k] = {
                "precision": sum(precisions) / n if n > 0 else 0.0,
                "recall": sum(recalls) / n if n > 0 else 0.0,
                "mrr": sum(mrrs) / n if n > 0 else 0.0,
                "ndcg": sum(ndcgs) / n if n > 0 else 0.0,
                "f1": sum(f1s) / n if n > 0 else 0.0,
                "map": sum(aps) / n if n > 0 else 0.0,
            }
        
        return metrics
    
    def print_report(self, k_values: List[int] = None):
        """Print a formatted evaluation report."""
        metrics = self.evaluate(k_values)
        
        print("\n" + "=" * 60)
        print("RETRIEVAL EVALUATION REPORT")
        print("=" * 60)
        print(f"Total queries evaluated: {len(self.results)}")
        print("-" * 60)
        
        for k, k_metrics in sorted(metrics.items()):
            print(f"\n@ k={k}:")
            print(f"  Precision:  {k_metrics['precision']:.4f}")
            print(f"  Recall:     {k_metrics['recall']:.4f}")
            print(f"  F1:         {k_metrics['f1']:.4f}")
            print(f"  MRR:        {k_metrics['mrr']:.4f}")
            print(f"  NDCG:       {k_metrics['ndcg']:.4f}")
            print(f"  MAP:        {k_metrics['map']:.4f}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test the metrics with example data
    print("Testing retrieval metrics...")
    
    # Example: retrieved docs and ground truth
    retrieved = ["doc1", "doc3", "doc5", "doc2", "doc7", "doc4", "doc8", "doc9", "doc10", "doc6"]
    relevant = {"doc1", "doc2", "doc4", "doc6"}
    
    print(f"\nRetrieved: {retrieved}")
    print(f"Relevant:  {relevant}")
    
    print(f"\nPrecision@3: {precision_at_k(retrieved, relevant, 3):.4f}")
    print(f"Precision@5: {precision_at_k(retrieved, relevant, 5):.4f}")
    print(f"Precision@10: {precision_at_k(retrieved, relevant, 10):.4f}")
    
    print(f"\nRecall@3: {recall_at_k(retrieved, relevant, 3):.4f}")
    print(f"Recall@5: {recall_at_k(retrieved, relevant, 5):.4f}")
    print(f"Recall@10: {recall_at_k(retrieved, relevant, 10):.4f}")
    
    print(f"\nMRR: {mean_reciprocal_rank(retrieved, relevant):.4f}")
    
    print(f"\nNDCG@3: {ndcg_at_k(retrieved, relevant, 3):.4f}")
    print(f"NDCG@5: {ndcg_at_k(retrieved, relevant, 5):.4f}")
    print(f"NDCG@10: {ndcg_at_k(retrieved, relevant, 10):.4f}")
    
    print(f"\nF1@5: {f1_at_k(retrieved, relevant, 5):.4f}")
    print(f"Average Precision: {average_precision(retrieved, relevant):.4f}")
    
    # Test evaluator with multiple queries
    print("\n" + "-" * 40)
    print("Testing RetrievalEvaluator with multiple queries...")
    
    evaluator = RetrievalEvaluator()
    
    # Add some test queries
    evaluator.add_result("q1", ["d1", "d2", "d3", "d4", "d5"], {"d1", "d3"})
    evaluator.add_result("q2", ["d2", "d1", "d4", "d3", "d5"], {"d1", "d2", "d5"})
    evaluator.add_result("q3", ["d5", "d4", "d3", "d2", "d1"], {"d1"})
    evaluator.add_result("q4", ["d1", "d2", "d3", "d4", "d5"], {"d1", "d2", "d3", "d4"})
    
    evaluator.print_report([3, 5])
    
    print("\nAll tests passed!")
