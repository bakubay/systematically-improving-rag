"""
Week 1: Retrieval Evaluation Pipeline

Complete evaluation pipeline using ChromaDB and custom metrics.
Uses MS MARCO dataset for evaluation.
"""

import json
from pathlib import Path
from typing import Optional

from metrics import RetrievalEvaluator, recall_at_k

# Check for optional dependencies
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not installed. Run: uv add chromadb")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("datasets not installed. Run: uv add datasets")


def create_mock_data():
    """Create mock data for testing without external dependencies."""
    documents = [
        {"id": "doc_0", "text": "Python is a programming language known for its simplicity."},
        {"id": "doc_1", "text": "Machine learning uses algorithms to learn from data."},
        {"id": "doc_2", "text": "RAG combines retrieval with generation for better answers."},
        {"id": "doc_3", "text": "Vector databases store embeddings for similarity search."},
        {"id": "doc_4", "text": "ChromaDB is an open-source embedding database."},
        {"id": "doc_5", "text": "Embeddings represent text as numerical vectors."},
        {"id": "doc_6", "text": "Semantic search finds documents by meaning, not keywords."},
        {"id": "doc_7", "text": "LLMs like GPT can generate human-like text."},
        {"id": "doc_8", "text": "Fine-tuning adapts pre-trained models to specific tasks."},
        {"id": "doc_9", "text": "Evaluation metrics measure retrieval system performance."},
    ]
    
    # Test queries with ground truth
    queries = [
        {
            "id": "q_0",
            "text": "What is Python?",
            "relevant_ids": {"doc_0"}
        },
        {
            "id": "q_1",
            "text": "How does machine learning work?",
            "relevant_ids": {"doc_1"}
        },
        {
            "id": "q_2",
            "text": "What is RAG and retrieval augmented generation?",
            "relevant_ids": {"doc_2"}
        },
        {
            "id": "q_3",
            "text": "What are vector databases used for?",
            "relevant_ids": {"doc_3", "doc_4"}
        },
        {
            "id": "q_4",
            "text": "How do embeddings represent text?",
            "relevant_ids": {"doc_5", "doc_6"}
        },
        {
            "id": "q_5",
            "text": "What is semantic search?",
            "relevant_ids": {"doc_6"}
        },
        {
            "id": "q_6",
            "text": "How to evaluate retrieval systems?",
            "relevant_ids": {"doc_9"}
        },
    ]
    
    return documents, queries


def load_msmarco_sample(n_samples: int = 100):
    """Load a sample from MS MARCO dataset."""
    if not DATASETS_AVAILABLE:
        print("Using mock data instead of MS MARCO")
        return create_mock_data()
    
    try:
        dataset = load_dataset("microsoft/ms_marco", "v1.1", split=f"train[:{n_samples}]")
    except Exception as e:
        print(f"Could not load MS MARCO: {e}")
        print("Using mock data instead")
        return create_mock_data()
    
    documents = []
    queries = []
    doc_id_counter = 0
    
    for i, item in enumerate(dataset):
        query_id = f"q_{i}"
        relevant_doc_ids = set()
        
        # Extract passages
        for j, (passage_text, is_selected) in enumerate(
            zip(item["passages"]["passage_text"], item["passages"]["is_selected"])
        ):
            doc_id = f"doc_{doc_id_counter}"
            documents.append({"id": doc_id, "text": passage_text})
            
            if is_selected:
                relevant_doc_ids.add(doc_id)
            
            doc_id_counter += 1
        
        if relevant_doc_ids:
            queries.append({
                "id": query_id,
                "text": item["query"],
                "relevant_ids": relevant_doc_ids
            })
    
    return documents, queries


class EvaluationPipeline:
    """Complete evaluation pipeline for retrieval systems."""
    
    def __init__(self, persist_path: Optional[str] = None):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB required. Run: uv add chromadb")
        
        if persist_path:
            self.client = chromadb.PersistentClient(path=persist_path)
        else:
            self.client = chromadb.Client()
        
        # Use default embedding function (all-MiniLM-L6-v2)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = None
        self.documents = []
        self.queries = []
    
    def index_documents(self, documents: list[dict], collection_name: str = "eval_docs"):
        """Index documents into ChromaDB."""
        self.documents = documents
        
        # Delete existing collection if exists
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.collection.add(
                documents=[d["text"] for d in batch],
                ids=[d["id"] for d in batch]
            )
        
        print(f"Indexed {len(documents)} documents")
    
    def set_queries(self, queries: list[dict]):
        """Set test queries with ground truth."""
        self.queries = queries
        print(f"Loaded {len(queries)} test queries")
    
    def retrieve(self, query_text: str, k: int = 10) -> list[str]:
        """Retrieve top-k documents for a query."""
        if self.collection is None:
            raise ValueError("No documents indexed. Call index_documents first.")
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k
        )
        
        return results["ids"][0]
    
    def evaluate(self, k_values: list[int] = None) -> dict:
        """Run evaluation on all queries."""
        if k_values is None:
            k_values = [3, 5, 10, 20]
        
        max_k = max(k_values)
        evaluator = RetrievalEvaluator()
        
        for query in self.queries:
            retrieved_ids = self.retrieve(query["text"], k=max_k)
            evaluator.add_result(
                query_id=query["id"],
                retrieved_ids=retrieved_ids,
                relevant_ids=query["relevant_ids"]
            )
        
        return evaluator.evaluate(k_values)
    
    def run_full_evaluation(self, k_values: list[int] = None):
        """Run and print full evaluation report."""
        if k_values is None:
            k_values = [3, 5, 10]
        
        max_k = max(k_values)
        evaluator = RetrievalEvaluator()
        
        print("\nRunning evaluation...")
        for i, query in enumerate(self.queries):
            retrieved_ids = self.retrieve(query["text"], k=max_k)
            evaluator.add_result(
                query_id=query["id"],
                retrieved_ids=retrieved_ids,
                relevant_ids=query["relevant_ids"]
            )
            
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(self.queries)} queries")
        
        evaluator.print_report(k_values)
        return evaluator.evaluate(k_values)
    
    def analyze_failures(self, k: int = 5, n_failures: int = 5):
        """Analyze queries with low recall."""
        print(f"\n{'='*60}")
        print(f"FAILURE ANALYSIS (Queries with Recall@{k} < 0.5)")
        print('='*60)
        
        failures = []
        
        for query in self.queries:
            retrieved_ids = self.retrieve(query["text"], k=k)
            recall = recall_at_k(retrieved_ids, query["relevant_ids"], k)
            
            if recall < 0.5:
                failures.append({
                    "query": query,
                    "retrieved": retrieved_ids,
                    "recall": recall
                })
        
        failures.sort(key=lambda x: x["recall"])
        
        print(f"\nFound {len(failures)} queries with recall < 0.5")
        print(f"\nTop {n_failures} worst performing queries:")
        
        for i, failure in enumerate(failures[:n_failures]):
            query = failure["query"]
            print(f"\n{i+1}. Query: {query['text'][:80]}...")
            print(f"   Relevant docs: {query['relevant_ids']}")
            print(f"   Retrieved: {failure['retrieved'][:5]}")
            print(f"   Recall@{k}: {failure['recall']:.2f}")


def main():
    """Run the evaluation pipeline."""
    print("=" * 60)
    print("WEEK 1: RETRIEVAL EVALUATION PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    documents, queries = create_mock_data()  # Use mock data for testing
    
    print(f"Loaded {len(documents)} documents and {len(queries)} queries")
    
    # Initialize pipeline
    pipeline = EvaluationPipeline()
    
    # Index documents
    print("\nIndexing documents...")
    pipeline.index_documents(documents)
    
    # Set queries
    pipeline.set_queries(queries)
    
    # Run evaluation
    metrics = pipeline.run_full_evaluation(k_values=[3, 5, 10])
    
    # Analyze failures
    pipeline.analyze_failures(k=5, n_failures=3)
    
    # Save results
    output_path = Path(__file__).parent / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return metrics


if __name__ == "__main__":
    main()
