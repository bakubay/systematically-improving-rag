---
title: "Chapter 8: Hybrid Search"
description: "Combine semantic and lexical search for robust retrieval. Learn when hybrid approaches outperform pure vector search, how to implement BM25 with embeddings, and when to use Reciprocal Rank Fusion for result merging."
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - hybrid-search
  - lexical-search
  - bm25
  - semantic-search
  - reciprocal-rank-fusion
  - retrieval
  - information-retrieval
---

# Chapter 8: Hybrid Search

## Chapter at a Glance

**Prerequisites**: Chapter 1 (evaluation framework), Chapter 0 (semantic vs lexical search basics), familiarity with embeddings and vector databases

**What You Will Learn**:

- Why semantic search alone fails for exact matches, product IDs, and domain-specific jargon
- How lexical search works: tokenization, inverted indices, and TF-IDF/BM25 scoring
- When to use hybrid search vs pure semantic or pure lexical approaches
- How to implement Reciprocal Rank Fusion (RRF) to combine search results
- How to evaluate hybrid search performance against baselines
- Implementation patterns for production hybrid search systems

**Case Study Reference**: E-commerce product search improved from 67% to 89% recall by adding lexical search for product IDs and brand names that embeddings missed

**Time to Complete**: 60-75 minutes

---

## Key Insight

**Neither semantic nor lexical search is universally better—the right approach depends on your data and queries.** Semantic search excels at understanding meaning and handling synonyms, but struggles with exact matches, product IDs, and specialized terminology. Lexical search handles exact matching and filtering efficiently, but misses semantic relationships. Hybrid approaches combine both strengths, typically improving recall by 10-25% over either approach alone. The key is measuring which approach works for your specific use case rather than assuming one is always better.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Identify when semantic search fails** and recognize query types that benefit from lexical search
2. **Understand lexical search fundamentals** including tokenization, inverted indices, and BM25 scoring
3. **Choose between semantic, lexical, and hybrid approaches** based on query characteristics and data types
4. **Implement Reciprocal Rank Fusion** to combine results from multiple retrieval methods
5. **Evaluate hybrid search performance** using the evaluation framework from Chapter 1
6. **Design production hybrid search architectures** that balance latency, cost, and quality

---

## Introduction

Chapter 0 introduced the distinction between semantic and lexical search. Chapter 1 established evaluation frameworks for measuring retrieval quality. This chapter brings these concepts together, showing when and how to combine search approaches for better results.

**The Limitation of Pure Semantic Search**:

Semantic search has become the default approach for RAG systems, but it has significant blind spots:

- **Exact matches**: Product IDs like "SKU-12345" or model numbers like "iPhone 15 Pro Max" may not embed distinctly
- **Specialized terminology**: Domain jargon not present in embedding training data gets poor representations
- **Filtering**: Vector databases struggle with efficient filtering—you either search then filter (risking empty results) or filter then search (expensive for large datasets)
- **Negation**: "I love coffee" and "I hate coffee" have similar embeddings because embedding models don't fully understand negation

**The Case for Hybrid Search**:

Lexical search addresses these exact weaknesses. By combining both approaches, you get:

- Semantic understanding for meaning-based queries
- Exact matching for identifiers and specific terms
- Efficient filtering through inverted indices
- Better recall across diverse query types

!!! tip "For Product Managers"
    Hybrid search typically improves recall by 10-25% over pure semantic search, with the largest gains on queries involving exact terms, product identifiers, or domain-specific vocabulary. The investment is moderate—most vector databases now support hybrid search with minimal additional configuration.

!!! tip "For Engineers"
    This chapter covers both the theory and implementation of hybrid search. Pay attention to the Reciprocal Rank Fusion algorithm for combining results, and the evaluation methodology for measuring improvement. The code examples use LanceDB, which supports hybrid search with a single parameter change.

---

## Core Content

### When Semantic Search Fails

Understanding failure modes helps identify when hybrid search adds value.

!!! tip "For Product Managers"
    **Common semantic search failures**:

    | Query Type | Example | Why Semantic Fails |
    |------------|---------|-------------------|
    | Product IDs | "SKU-12345" | IDs are arbitrary strings, not semantic concepts |
    | Model numbers | "iPhone 15 Pro Max 256GB" | Specific configurations embed similarly |
    | People's names | "John Smith contract" | Names don't carry semantic meaning |
    | Exact phrases | "force majeure clause" | Legal terms need exact matching |
    | Negation | "contracts without arbitration" | Embeddings don't understand "without" |

    **Business impact**: A customer support system found that 15% of queries involved product IDs or order numbers. Pure semantic search had 23% recall on these queries. Adding lexical search improved recall to 91%—a 68 percentage point improvement for this query segment.

    **Decision framework**: If more than 10% of your queries involve exact terms, identifiers, or domain jargon, hybrid search likely provides meaningful improvement.

!!! tip "For Engineers"
    **Technical reasons for semantic search failures**:

    1. **Out-of-vocabulary terms**: Embedding models are trained on general text. Domain-specific terms like "HIPAA" or "SOC2" may not have meaningful representations.

    2. **Bag-of-words limitations in embeddings**: While embeddings capture semantic meaning, they lose word order context. "The cat sat on the mat" and "The mat sat on the cat" have similar embeddings.

    3. **Sparse vs dense representations**: Embeddings are dense vectors where every dimension has a value. Lexical representations are sparse—most dimensions are zero. Sparse representations naturally handle exact matching.

    4. **Score distribution variance**: Semantic similarity scores vary wildly by query type. A threshold that works for one category fails for others.

    **Example: Score distribution problem**

    ```python
    # Semantic search scores for different query types
    query_scores = {
        "What is machine learning?": [0.89, 0.87, 0.85, 0.82],  # High scores
        "SKU-12345": [0.34, 0.33, 0.31, 0.29],  # Low scores even for correct match
        "John Smith contract": [0.45, 0.44, 0.43, 0.42],  # Similar scores, hard to rank
    }

    # A 0.7 threshold would:
    # - Return results for "machine learning" query
    # - Return nothing for SKU query (even if correct doc exists)
    # - Return nothing for name query
    ```

### How Lexical Search Works

Understanding lexical search fundamentals helps you use it effectively.

!!! tip "For Product Managers"
    **Lexical search in plain terms**:

    Lexical search finds documents containing the exact words in your query. It works by:

    1. **Building an index**: When documents are added, each word is recorded with pointers to documents containing it
    2. **Matching queries**: When you search, the system finds documents containing your query words
    3. **Ranking results**: Documents are scored by how often and how uniquely they contain query terms

    **Key advantages**:

    - **Speed**: Inverted indices make lookup extremely fast
    - **Exact matching**: Perfect for IDs, names, and specific terms
    - **Filtering**: Can combine search with filters in a single operation
    - **Transparency**: Easy to understand why a document matched

    **Key limitations**:

    - **No semantic understanding**: "car" won't match "automobile"
    - **Word order ignored**: "dress shoe" might match "shoe dress"
    - **Vocabulary mismatch**: Users must use the same words as documents

!!! tip "For Engineers"
    **Lexical search components**:

    **1. Text Analysis Pipeline**:

    ```python
    def analyze_text(text: str) -> list[str]:
        """
        Standard text analysis pipeline for lexical search.

        Steps:
        1. Lowercase - normalize case
        2. Tokenize - split into words
        3. Remove stopwords - filter common words
        4. Stem/lemmatize - normalize word forms
        """
        # Lowercase
        text = text.lower()

        # Tokenize (simple whitespace split)
        tokens = text.split()

        # Remove stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at"}
        tokens = [t for t in tokens if t not in stopwords]

        # Stem (simplified - production would use Porter or Snowball stemmer)
        # Example: "running" -> "run", "contracts" -> "contract"
        # In production, use: from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()  # Requires: pip install nltk
        tokens = [stemmer.stem(t) for t in tokens]

        return tokens
    ```

    **2. Inverted Index Structure**:

    ```python
    # Inverted index maps terms to documents
    inverted_index = {
        "contract": [
            {"doc_id": 1, "positions": [5, 23, 45], "frequency": 3},
            {"doc_id": 7, "positions": [12], "frequency": 1},
        ],
        "arbitration": [
            {"doc_id": 1, "positions": [67], "frequency": 1},
            {"doc_id": 3, "positions": [8, 34], "frequency": 2},
        ],
    }

    # Query: "contract arbitration"
    # Find intersection of posting lists
    # doc_id 1 contains both terms
    ```

    **3. BM25 Scoring**:

    BM25 (Best Match 25) is the standard scoring algorithm for lexical search:

    ```python
    import math
    from typing import Dict, List

    def bm25_score(
        query_terms: List[str],
        doc_terms: List[str],
        doc_length: int,
        avg_doc_length: float,
        doc_frequencies: Dict[str, int],
        total_docs: int,
        k1: float = 1.2,
        b: float = 0.75
    ) -> float:
        """
        Calculate BM25 score for a document given a query.

        Args:
            query_terms: Tokenized query
            doc_terms: Tokenized document
            doc_length: Number of terms in document
            avg_doc_length: Average document length in corpus
            doc_frequencies: Number of docs containing each term
            total_docs: Total documents in corpus
            k1: Term frequency saturation parameter
            b: Document length normalization parameter

        Returns:
            BM25 relevance score
        """
        score = 0.0

        for term in query_terms:
            if term not in doc_terms:
                continue

            # Term frequency in document
            tf = doc_terms.count(term)

            # Inverse document frequency
            df = doc_frequencies.get(term, 0)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

            # Length normalization
            length_norm = 1 - b + b * (doc_length / avg_doc_length)

            # BM25 formula
            term_score = idf * (tf * (k1 + 1)) / (tf + k1 * length_norm)
            score += term_score

        return score
    ```

    **Key BM25 properties**:

    - **Term frequency saturation**: More occurrences help, but with diminishing returns (controlled by k1)
    - **Document length normalization**: Longer documents don't automatically score higher (controlled by b)
    - **IDF weighting**: Rare terms are more important than common terms

### Hybrid Search Approaches

Several strategies exist for combining semantic and lexical search.

!!! tip "For Product Managers"
    **Hybrid search strategies**:

    | Strategy | Description | Best For |
    |----------|-------------|----------|
    | Parallel + Fusion | Run both searches, combine results | General purpose, balanced queries |
    | Lexical then Re-rank | Lexical search first, semantic re-ranking | Large datasets, filtering-heavy |
    | Semantic then Lexical boost | Semantic search with lexical score boost | Meaning-focused with exact term needs |
    | Query-dependent routing | Choose approach based on query type | Diverse query patterns |

    **ROI analysis**:

    | Approach | Implementation Effort | Latency Impact | Typical Recall Improvement |
    |----------|----------------------|----------------|---------------------------|
    | Parallel + RRF | Low (if DB supports) | +20-50ms | 10-25% |
    | Lexical + Re-rank | Medium | +100-300ms | 15-30% |
    | Query routing | High | Variable | 20-40% |

    **Recommendation**: Start with parallel search and Reciprocal Rank Fusion. Most modern vector databases support this with minimal configuration, and it provides good improvement with low implementation effort.

!!! tip "For Engineers"
    **Strategy 1: Parallel Search with Reciprocal Rank Fusion**

    The most common approach runs both searches in parallel and combines results:

    ```python
    from typing import List, Dict, Any
    import asyncio

    async def hybrid_search(
        query: str,
        semantic_searcher,
        lexical_searcher,
        k: int = 10,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with Reciprocal Rank Fusion.

        Args:
            query: Search query
            semantic_searcher: Semantic search function
            lexical_searcher: Lexical search function
            k: Number of results to return
            rrf_k: RRF constant (typically 60)

        Returns:
            Combined and re-ranked results
        """
        # Run searches in parallel
        semantic_results, lexical_results = await asyncio.gather(
            semantic_searcher.search(query, top_k=k * 2),
            lexical_searcher.search(query, top_k=k * 2)
        )

        # Apply Reciprocal Rank Fusion
        combined = reciprocal_rank_fusion(
            [semantic_results, lexical_results],
            k=rrf_k
        )

        return combined[:k]


    def reciprocal_rank_fusion(
        result_lists: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across all lists

        Args:
            result_lists: List of ranked result lists
            k: Constant to prevent high scores for top ranks (typically 60)

        Returns:
            Combined results sorted by RRF score
        """
        scores: Dict[str, float] = {}
        docs: Dict[str, Dict[str, Any]] = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc["id"]

                # RRF formula: 1 / (k + rank)
                # rank is 0-indexed, so rank 0 gets score 1/(k+0) = 1/k
                rrf_score = 1.0 / (k + rank)

                if doc_id in scores:
                    scores[doc_id] += rrf_score
                else:
                    scores[doc_id] = rrf_score
                    docs[doc_id] = doc

        # Sort by combined RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [
            {**docs[doc_id], "rrf_score": scores[doc_id]}
            for doc_id in sorted_ids
        ]
    ```

    **Strategy 2: Lexical Search with Semantic Re-ranking**

    For large datasets where semantic search over everything is expensive:

    ```python
    from typing import List, Dict, Any

    async def lexical_then_rerank(
        query: str,
        lexical_searcher,
        reranker,
        initial_k: int = 100,
        final_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Lexical search for initial retrieval, semantic re-ranking for precision.

        This approach:
        1. Uses fast lexical search to get candidate set
        2. Applies expensive semantic re-ranking only to candidates

        Args:
            query: Search query
            lexical_searcher: Lexical search function
            reranker: Semantic re-ranking model
            initial_k: Number of lexical results to retrieve
            final_k: Number of final results to return

        Returns:
            Re-ranked results
        """
        # Fast lexical retrieval
        candidates = await lexical_searcher.search(query, top_k=initial_k)

        # Semantic re-ranking on smaller candidate set
        reranked = await reranker.rerank(
            query=query,
            documents=[c["text"] for c in candidates],
            top_k=final_k
        )

        # Map back to original documents
        results = []
        for idx, score in reranked:
            doc = candidates[idx].copy()
            doc["rerank_score"] = score
            results.append(doc)

        return results
    ```

    **Strategy 3: Query-Dependent Routing**

    Route queries to the best search method based on characteristics:

    ```python
    from typing import List, Dict, Any
    from enum import Enum
    import re

    class SearchMethod(Enum):
        SEMANTIC = "semantic"
        LEXICAL = "lexical"
        HYBRID = "hybrid"

    def classify_query(query: str) -> SearchMethod:
        """
        Classify query to determine best search method.

        Heuristics:
        - Contains IDs/codes -> lexical
        - Contains quotes (exact phrase) -> lexical
        - Short keyword query -> lexical
        - Natural language question -> semantic
        - Mixed -> hybrid
        """
        # Check for product IDs, SKUs, order numbers
        if re.search(r'\b[A-Z]{2,}-?\d{3,}\b', query):
            return SearchMethod.LEXICAL

        # Check for quoted phrases (exact match intent)
        if '"' in query or "'" in query:
            return SearchMethod.LEXICAL

        # Check for very short queries (likely keywords)
        if len(query.split()) <= 2:
            return SearchMethod.LEXICAL

        # Check for question words (semantic intent)
        question_words = {"what", "how", "why", "when", "where", "who", "which"}
        if query.lower().split()[0] in question_words:
            return SearchMethod.SEMANTIC

        # Default to hybrid for mixed queries
        return SearchMethod.HYBRID


    async def adaptive_search(
        query: str,
        semantic_searcher,
        lexical_searcher,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Adaptively choose search method based on query characteristics.
        """
        method = classify_query(query)

        if method == SearchMethod.SEMANTIC:
            return await semantic_searcher.search(query, top_k=k)
        elif method == SearchMethod.LEXICAL:
            return await lexical_searcher.search(query, top_k=k)
        else:
            return await hybrid_search(query, semantic_searcher, lexical_searcher, k)
    ```

### Implementation with LanceDB

LanceDB makes hybrid search straightforward with built-in support.

!!! tip "For Product Managers"
    **Why LanceDB for hybrid search**:

    - Single line of code to switch between semantic, lexical, and hybrid
    - Built-in re-ranking support
    - SQL-compatible for complex filtering
    - No separate infrastructure for lexical search

    **Cost comparison**:

    | Approach | Infrastructure | Maintenance |
    |----------|---------------|-------------|
    | Separate Elasticsearch + Vector DB | High | High |
    | PostgreSQL + pgvector + pg_search | Medium | Medium |
    | LanceDB (all-in-one) | Low | Low |

!!! tip "For Engineers"
    **LanceDB hybrid search implementation**:

    ```python
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from lancedb.rerankers import CohereReranker

    # Define schema
    class Document(LanceModel):
        id: str
        text: str
        source: str
        vector: Vector(1536)  # OpenAI embedding dimension

    # Connect and create table
    db = lancedb.connect("./hybrid_search_db")
    table = db.create_table("documents", schema=Document)

    # Add documents (embeddings generated automatically if configured)
    table.add([
        Document(id="1", text="Contract for services...", source="legal"),
        Document(id="2", text="SKU-12345 product specification...", source="catalog"),
    ])

    # Create full-text search index
    table.create_fts_index("text")

    # Search comparison
    query = "SKU-12345 specifications"

    # Pure semantic search
    semantic_results = table.search(query).limit(10).to_list()

    # Pure lexical search
    lexical_results = table.search(query, query_type="fts").limit(10).to_list()

    # Hybrid search (combines both)
    hybrid_results = table.search(query, query_type="hybrid").limit(10).to_list()

    # Hybrid with re-ranking
    reranker = CohereReranker()
    reranked_results = (
        table.search(query, query_type="hybrid")
        .limit(50)  # Get more candidates
        .rerank(reranker)
        .limit(10)  # Final results
        .to_list()
    )
    ```

    **Comparing search methods**:

    ```python
    from typing import List, Dict, Any

    async def compare_search_methods(
        table,
        queries: List[str],
        ground_truth: Dict[str, List[str]],
        k: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare semantic, lexical, and hybrid search on evaluation set.

        Args:
            table: LanceDB table
            queries: List of test queries
            ground_truth: Dict mapping query to list of relevant doc IDs
            k: Number of results to retrieve

        Returns:
            Metrics for each search method
        """
        methods = {
            "semantic": lambda q: table.search(q).limit(k).to_list(),
            "lexical": lambda q: table.search(q, query_type="fts").limit(k).to_list(),
            "hybrid": lambda q: table.search(q, query_type="hybrid").limit(k).to_list(),
        }

        results = {}

        for method_name, search_fn in methods.items():
            total_recall = 0
            total_precision = 0

            for query in queries:
                retrieved = search_fn(query)
                retrieved_ids = {r["id"] for r in retrieved}
                relevant_ids = set(ground_truth.get(query, []))

                if relevant_ids:
                    recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
                    total_recall += recall

                if retrieved_ids:
                    precision = len(retrieved_ids & relevant_ids) / len(retrieved_ids)
                    total_precision += precision

            results[method_name] = {
                "avg_recall": total_recall / len(queries),
                "avg_precision": total_precision / len(queries),
            }

        return results
    ```

### Evaluating Hybrid Search

Use the evaluation framework from Chapter 1 to measure hybrid search improvement.

!!! tip "For Product Managers"
    **Evaluation methodology**:

    1. **Segment queries by type**: Identify which queries benefit from lexical search
    2. **Measure baseline**: Run pure semantic search on your evaluation set
    3. **Measure hybrid**: Run hybrid search on the same set
    4. **Analyze by segment**: Some query types may improve dramatically while others stay flat

    **Example results from e-commerce search**:

    | Query Type | Semantic Recall | Hybrid Recall | Improvement |
    |------------|-----------------|---------------|-------------|
    | Natural language | 78% | 81% | +3% |
    | Product names | 65% | 82% | +17% |
    | SKU/ID queries | 23% | 91% | +68% |
    | Brand + feature | 71% | 85% | +14% |
    | **Overall** | **67%** | **84%** | **+17%** |

    **Key insight**: Hybrid search provides the largest improvements on exact-match queries, but also helps with product names and brand queries where exact terms matter.

!!! tip "For Engineers"
    **Comprehensive evaluation**:

    ```python
    from typing import List, Dict, Any, Tuple
    from dataclasses import dataclass
    import json

    @dataclass
    class EvaluationResult:
        method: str
        query_type: str
        recall: float
        precision: float
        mrr: float  # Mean Reciprocal Rank
        query_count: int

    def evaluate_by_query_type(
        table,
        evaluation_data: List[Dict[str, Any]],
        k: int = 10
    ) -> List[EvaluationResult]:
        """
        Evaluate search methods segmented by query type.

        Args:
            table: LanceDB table
            evaluation_data: List of dicts with 'query', 'relevant_docs', 'query_type'
            k: Number of results to retrieve

        Returns:
            Evaluation results by method and query type
        """
        methods = {
            "semantic": lambda q: table.search(q).limit(k).to_list(),
            "lexical": lambda q: table.search(q, query_type="fts").limit(k).to_list(),
            "hybrid": lambda q: table.search(q, query_type="hybrid").limit(k).to_list(),
        }

        # Group by query type
        by_type: Dict[str, List[Dict]] = {}
        for item in evaluation_data:
            query_type = item.get("query_type", "unknown")
            if query_type not in by_type:
                by_type[query_type] = []
            by_type[query_type].append(item)

        results = []

        for method_name, search_fn in methods.items():
            for query_type, items in by_type.items():
                recalls = []
                precisions = []
                mrrs = []

                for item in items:
                    retrieved = search_fn(item["query"])
                    retrieved_ids = [r["id"] for r in retrieved]
                    relevant_ids = set(item["relevant_docs"])

                    # Recall
                    if relevant_ids:
                        recall = len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)
                        recalls.append(recall)

                    # Precision
                    if retrieved_ids:
                        precision = len(set(retrieved_ids) & relevant_ids) / len(retrieved_ids)
                        precisions.append(precision)

                    # MRR
                    mrr = 0.0
                    for rank, doc_id in enumerate(retrieved_ids):
                        if doc_id in relevant_ids:
                            mrr = 1.0 / (rank + 1)
                            break
                    mrrs.append(mrr)

                results.append(EvaluationResult(
                    method=method_name,
                    query_type=query_type,
                    recall=sum(recalls) / len(recalls) if recalls else 0,
                    precision=sum(precisions) / len(precisions) if precisions else 0,
                    mrr=sum(mrrs) / len(mrrs) if mrrs else 0,
                    query_count=len(items)
                ))

        return results


    def print_evaluation_report(results: List[EvaluationResult]) -> None:
        """Print formatted evaluation report."""
        # Group by query type
        by_type: Dict[str, List[EvaluationResult]] = {}
        for r in results:
            if r.query_type not in by_type:
                by_type[r.query_type] = []
            by_type[r.query_type].append(r)

        for query_type, type_results in by_type.items():
            print(f"\n=== {query_type} (n={type_results[0].query_count}) ===")
            print(f"{'Method':<12} {'Recall':>8} {'Precision':>10} {'MRR':>8}")
            print("-" * 40)

            for r in sorted(type_results, key=lambda x: x.method):
                print(f"{r.method:<12} {r.recall:>8.1%} {r.precision:>10.1%} {r.mrr:>8.2f}")
    ```

    **Statistical significance testing**:

    ```python
    from scipy import stats
    from typing import List, Tuple

    def paired_t_test(
        baseline_scores: List[float],
        treatment_scores: List[float]
    ) -> Tuple[float, float]:
        """
        Perform paired t-test to check if improvement is significant.

        Args:
            baseline_scores: Per-query scores for baseline method
            treatment_scores: Per-query scores for treatment method

        Returns:
            (t_statistic, p_value)
        """
        t_stat, p_value = stats.ttest_rel(treatment_scores, baseline_scores)
        return t_stat, p_value


    def is_improvement_significant(
        semantic_recalls: List[float],
        hybrid_recalls: List[float],
        alpha: float = 0.05
    ) -> bool:
        """
        Check if hybrid search significantly improves over semantic.

        Args:
            semantic_recalls: Per-query recall for semantic search
            hybrid_recalls: Per-query recall for hybrid search
            alpha: Significance level

        Returns:
            True if improvement is statistically significant
        """
        t_stat, p_value = paired_t_test(semantic_recalls, hybrid_recalls)

        # One-tailed test (hybrid > semantic)
        p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2

        return p_value_one_tailed < alpha
    ```

### Advanced Hybrid Patterns

More sophisticated approaches for specific use cases.

!!! tip "For Product Managers"
    **Advanced patterns and when to use them**:

    | Pattern | Use Case | Complexity |
    |---------|----------|------------|
    | Weighted fusion | When one method is consistently better | Low |
    | Query expansion | Improve lexical recall with synonyms | Medium |
    | SPLADE | Best of both worlds in single model | High |
    | Learned fusion | Optimize weights from data | High |

    **Recommendation**: Start with standard RRF. Only invest in advanced patterns if evaluation shows specific weaknesses that simpler approaches don't address.

!!! tip "For Engineers"
    **Weighted Reciprocal Rank Fusion**:

    ```python
    def weighted_rrf(
        result_lists: List[List[Dict[str, Any]]],
        weights: List[float],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        RRF with weights for each result list.

        Useful when one search method is more reliable than another.

        Args:
            result_lists: List of ranked result lists
            weights: Weight for each result list (should sum to 1)
            k: RRF constant

        Returns:
            Combined results sorted by weighted RRF score
        """
        assert len(result_lists) == len(weights)

        scores: Dict[str, float] = {}
        docs: Dict[str, Dict[str, Any]] = {}

        for results, weight in zip(result_lists, weights):
            for rank, doc in enumerate(results):
                doc_id = doc["id"]
                rrf_score = weight * (1.0 / (k + rank))

                if doc_id in scores:
                    scores[doc_id] += rrf_score
                else:
                    scores[doc_id] = rrf_score
                    docs[doc_id] = doc

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [{**docs[doc_id], "rrf_score": scores[doc_id]} for doc_id in sorted_ids]


    # Example: Weight semantic higher for natural language queries
    hybrid_results = weighted_rrf(
        [semantic_results, lexical_results],
        weights=[0.7, 0.3],  # 70% semantic, 30% lexical
        k=60
    )
    ```

    **Query expansion for lexical search**:

    ```python
    from typing import List, Set

    class QueryExpander:
        """Expand queries with synonyms to improve lexical recall."""

        def __init__(self, synonym_dict: Dict[str, List[str]]):
            self.synonyms = synonym_dict

        def expand(self, query: str) -> str:
            """
            Expand query with synonyms.

            Example: "car insurance" -> "car automobile vehicle insurance"
            """
            tokens = query.lower().split()
            expanded_tokens = []

            for token in tokens:
                expanded_tokens.append(token)
                if token in self.synonyms:
                    expanded_tokens.extend(self.synonyms[token])

            return " ".join(expanded_tokens)

        @classmethod
        def from_embeddings(
            cls,
            terms: List[str],
            embedding_model,
            similarity_threshold: float = 0.85,
            max_synonyms: int = 3
        ) -> "QueryExpander":
            """
            Build synonym dictionary from embedding similarity.

            Args:
                terms: List of terms to find synonyms for
                embedding_model: Model to generate embeddings
                similarity_threshold: Minimum similarity for synonyms
                max_synonyms: Maximum synonyms per term

            Returns:
                QueryExpander with learned synonyms
            """
            import numpy as np

            # Embed all terms
            embeddings = [embedding_model.embed(t) for t in terms]

            # Find similar terms
            synonym_dict = {}
            for i, term in enumerate(terms):
                similarities = []
                for j, other_term in enumerate(terms):
                    if i != j:
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        if sim >= similarity_threshold:
                            similarities.append((other_term, sim))

                # Keep top synonyms
                similarities.sort(key=lambda x: x[1], reverse=True)
                synonym_dict[term] = [s[0] for s in similarities[:max_synonyms]]

            return cls(synonym_dict)
    ```

---

## Case Study Deep Dive

### E-Commerce Product Search: From 67% to 89% Recall

An e-commerce company selling electronics struggled with search quality for product-specific queries.

!!! tip "For Product Managers"
    **The problem**:

    - Overall search recall: 67%
    - Customer complaints about "can't find products I know exist"
    - High bounce rate on search results pages (45%)

    **Query analysis revealed three segments**:

    | Query Type | % of Queries | Semantic Recall | Example |
    |------------|--------------|-----------------|---------|
    | Natural language | 45% | 78% | "wireless headphones for running" |
    | Product name | 35% | 65% | "Sony WH-1000XM5" |
    | SKU/Model number | 20% | 23% | "SKU-12345" |

    **The insight**: 55% of queries involved specific product names or identifiers where semantic search performed poorly.

    **Solution**: Implemented hybrid search with weighted RRF:

    - Natural language queries: 70% semantic, 30% lexical
    - Product name queries: 50% semantic, 50% lexical
    - SKU queries: 20% semantic, 80% lexical

    **Results after 4 weeks**:

    | Metric | Before | After | Change |
    |--------|--------|-------|--------|
    | Overall recall | 67% | 89% | +22% |
    | Search bounce rate | 45% | 28% | -17% |
    | Add-to-cart rate | 3.2% | 4.1% | +28% |
    | Revenue per search | $0.45 | $0.58 | +29% |

    **ROI**: Implementation took 2 weeks of engineering time. Revenue increase paid for the investment within the first week of deployment.

!!! tip "For Engineers"
    **Implementation details**:

    ```python
    from enum import Enum
    from typing import List, Dict, Any
    import re

    class QueryCategory(Enum):
        NATURAL_LANGUAGE = "natural_language"
        PRODUCT_NAME = "product_name"
        SKU = "sku"

    def categorize_query(query: str) -> QueryCategory:
        """Categorize query to determine search weights."""
        # SKU pattern: letters followed by numbers
        if re.search(r'\b[A-Z]{2,}-?\d{4,}\b', query.upper()):
            return QueryCategory.SKU

        # Product name pattern: brand + model (e.g., "Sony WH-1000XM5")
        brand_pattern = r'\b(Sony|Apple|Samsung|LG|Bose|JBL|Beats)\b'
        if re.search(brand_pattern, query, re.IGNORECASE):
            return QueryCategory.PRODUCT_NAME

        return QueryCategory.NATURAL_LANGUAGE


    class AdaptiveHybridSearch:
        """Hybrid search with query-dependent weights."""

        def __init__(self, table):
            self.table = table
            self.weights = {
                QueryCategory.NATURAL_LANGUAGE: (0.7, 0.3),  # semantic, lexical
                QueryCategory.PRODUCT_NAME: (0.5, 0.5),
                QueryCategory.SKU: (0.2, 0.8),
            }

        async def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
            """Search with adaptive weights based on query type."""
            category = categorize_query(query)
            semantic_weight, lexical_weight = self.weights[category]

            # Get results from both methods
            semantic_results = self.table.search(query).limit(k * 2).to_list()
            lexical_results = self.table.search(query, query_type="fts").limit(k * 2).to_list()

            # Combine with weighted RRF
            combined = weighted_rrf(
                [semantic_results, lexical_results],
                weights=[semantic_weight, lexical_weight],
                k=60
            )

            # Add metadata
            for result in combined:
                result["query_category"] = category.value
                result["weights_used"] = (semantic_weight, lexical_weight)

            return combined[:k]
    ```

    **Monitoring and iteration**:

    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, List

    @dataclass
    class SearchMetrics:
        query: str
        category: str
        results_count: int
        clicked_position: int  # -1 if no click
        converted: bool
        timestamp: datetime

    class HybridSearchMonitor:
        """Monitor hybrid search performance for continuous improvement."""

        def __init__(self):
            self.metrics: List[SearchMetrics] = []

        def record(self, metrics: SearchMetrics) -> None:
            self.metrics.append(metrics)

        def get_category_performance(self) -> Dict[str, Dict[str, float]]:
            """Calculate performance metrics by query category."""
            by_category: Dict[str, List[SearchMetrics]] = {}

            for m in self.metrics:
                if m.category not in by_category:
                    by_category[m.category] = []
                by_category[m.category].append(m)

            results = {}
            for category, category_metrics in by_category.items():
                clicks = [m for m in category_metrics if m.clicked_position >= 0]
                conversions = [m for m in category_metrics if m.converted]

                results[category] = {
                    "query_count": len(category_metrics),
                    "click_rate": len(clicks) / len(category_metrics) if category_metrics else 0,
                    "avg_click_position": (
                        sum(m.clicked_position for m in clicks) / len(clicks)
                        if clicks else -1
                    ),
                    "conversion_rate": len(conversions) / len(category_metrics) if category_metrics else 0,
                }

            return results
    ```

---

## Implementation Guide

### Quick Start for PMs: Hybrid Search Assessment

**Step 1: Query analysis**

```
Total queries analyzed: ____
Queries with exact terms (IDs, names): ____% 
Queries with domain jargon: ____%
Natural language queries: ____%
```

**Step 2: Current performance baseline**

| Query Type | Current Recall | Target Recall |
|------------|----------------|---------------|
| Natural language | ___% | ___% |
| Exact terms | ___% | ___% |
| Domain jargon | ___% | ___% |
| Overall | ___% | ___% |

**Step 3: Expected ROI**

```
Current search success rate: ____%
Expected improvement with hybrid: ____%
Business impact per 1% improvement: $____
Implementation cost: $____
Expected ROI: ____x
```

### Detailed Implementation for Engineers

**Step 1: Set up hybrid search infrastructure**

```python
import lancedb
from lancedb.pydantic import LanceModel, Vector

# Define schema with text for both semantic and lexical
class Document(LanceModel):
    id: str
    text: str
    title: str
    category: str
    vector: Vector(1536)

# Create table and indices
db = lancedb.connect("./hybrid_db")
table = db.create_table("documents", schema=Document)

# Create full-text search index on text field
table.create_fts_index("text")
```

**Step 2: Implement evaluation pipeline**

```python
# Load evaluation data
evaluation_data = [
    {"query": "wireless headphones", "relevant_docs": ["doc1", "doc2"], "query_type": "natural_language"},
    {"query": "SKU-12345", "relevant_docs": ["doc3"], "query_type": "sku"},
    # ... more examples
]

# Run evaluation
results = evaluate_by_query_type(table, evaluation_data, k=10)
print_evaluation_report(results)
```

**Step 3: Implement adaptive hybrid search**

```python
# Create adaptive searcher
searcher = AdaptiveHybridSearch(table)

# Test on sample queries
test_queries = [
    "best noise canceling headphones",
    "Sony WH-1000XM5",
    "SKU-78901",
]

for query in test_queries:
    results = await searcher.search(query, k=5)
    print(f"\nQuery: {query}")
    print(f"Category: {results[0]['query_category']}")
    print(f"Top result: {results[0]['title']}")
```

**Step 4: Set up monitoring**

```python
# Initialize monitor
monitor = HybridSearchMonitor()

# Record search events (integrate with your application)
monitor.record(SearchMetrics(
    query="Sony headphones",
    category="product_name",
    results_count=10,
    clicked_position=0,
    converted=True,
    timestamp=datetime.now()
))

# Review performance weekly
performance = monitor.get_category_performance()
for category, metrics in performance.items():
    print(f"{category}: CTR={metrics['click_rate']:.1%}, CVR={metrics['conversion_rate']:.1%}")
```

---

## Common Pitfalls

### PM Pitfalls

!!! warning "PM Pitfall: Assuming hybrid is always better"
    **The mistake**: Implementing hybrid search without measuring whether it actually improves your specific use case.

    **Why it happens**: Hybrid search sounds sophisticated and is often recommended as a best practice.

    **The fix**: Always measure baseline performance first. If semantic search already achieves 90%+ recall on your queries, hybrid may add complexity without meaningful improvement.

!!! warning "PM Pitfall: Ignoring query distribution"
    **The mistake**: Optimizing for the average query when different query types have very different needs.

    **Why it happens**: Aggregate metrics hide segment-specific problems.

    **The fix**: Segment queries by type and measure performance separately. A 67% overall recall might hide 90% recall on common queries and 10% recall on important edge cases.

!!! warning "PM Pitfall: Underestimating maintenance"
    **The mistake**: Treating hybrid search as a one-time implementation.

    **Why it happens**: Initial results look good, so the system is left unchanged.

    **The fix**: Monitor query patterns over time. As your product evolves, the optimal weights and strategies may need adjustment.

### Engineering Pitfalls

!!! warning "Engineering Pitfall: Not creating full-text index"
    **The mistake**: Attempting lexical search without proper indexing, resulting in slow queries.

    **Why it happens**: Vector databases handle semantic search automatically, but lexical search requires explicit index creation.

    **The fix**: Always create full-text search indices on text fields before enabling hybrid search.

    ```python
    # Required for efficient lexical search
    table.create_fts_index("text")
    ```

!!! warning "Engineering Pitfall: Using wrong RRF constant"
    **The mistake**: Using k=1 or very small k values in RRF, causing top results to dominate unfairly.

    **Why it happens**: The RRF formula isn't intuitive, and smaller k seems like it would give more weight to top results.

    **The fix**: Use k=60 (the standard value from the original RRF paper). This provides balanced fusion across ranks.

!!! warning "Engineering Pitfall: Not handling empty results"
    **The mistake**: Assuming both search methods always return results.

    **Why it happens**: During development, test queries usually return results from both methods.

    **The fix**: Handle cases where one method returns empty results gracefully.

    ```python
    def safe_hybrid_search(query, semantic_fn, lexical_fn, k=10):
        semantic_results = semantic_fn(query) or []
        lexical_results = lexical_fn(query) or []

        if not semantic_results and not lexical_results:
            return []
        if not semantic_results:
            return lexical_results[:k]
        if not lexical_results:
            return semantic_results[:k]

        return reciprocal_rank_fusion([semantic_results, lexical_results])[:k]
    ```

!!! warning "Engineering Pitfall: Inconsistent text preprocessing"
    **The mistake**: Using different preprocessing for indexing vs querying.

    **Why it happens**: Semantic and lexical search have different preprocessing requirements.

    **The fix**: Ensure consistent preprocessing, especially for lexical search where exact token matching matters.

---

## Related Content

### Source Materials

This chapter synthesizes content from multiple sources:

- **Workshop Content**: [Chapter 1 - Evaluation Framework](../workshops/chapter1.md) - Hybrid search mentions
- **Expert Talk**: [Lexical Search - John Berryman](../talks/john-lexical-search.md) - Deep dive on lexical search fundamentals

### Expert Talks

!!! info "Lexical Search in RAG Applications - John Berryman"
    **Key insights**:

    - Semantic search struggles with exact matches, product IDs, and domain jargon
    - Lexical search can process filtering and relevance scoring simultaneously
    - SPLADE uses language models to add synthetic synonyms to lexical indices
    - The ideal hybrid solution is still evolving, but combining approaches is clearly beneficial

    **Practical application**: John demonstrated using the Wayfair Annotation Dataset to show how lexical search enables efficient filtering and faceted search that semantic search cannot match.

    [Read the full talk summary](../talks/john-lexical-search.md)

### Office Hours

!!! info "Cohort 2 Week 1 Summary"
    **Key discussions**:

    - LanceDB chosen for this course because it makes comparing retrieval strategies trivial
    - One line of code to switch between lexical, vector, and hybrid search
    - Re-rankers typically improve performance by 6-12% while adding 400-500ms latency

    [Read the full summary](../office-hours/cohort2/week1-summary.md)

!!! info "Cohort 3 Week 2-1 Summary"
    **Key discussions**:

    - For documentation, consider page-level chunking combined with semantic and lexical search
    - Postgres with pgvector provides good balance, add pg_search for BM25 implementation
    - Combine vector search and lexical search in same database for filtering by metadata

    [Read the full summary](../office-hours/cohort3/week-2-1.md)

---

## Action Items

### For Product Teams

1. **Analyze query distribution** (Week 1)
   - Categorize recent queries by type (natural language, exact terms, identifiers)
   - Identify what percentage involve exact matching needs
   - Estimate potential improvement from hybrid search

2. **Establish baseline metrics** (Week 1)
   - Measure current recall by query type
   - Document search-related user complaints
   - Track search bounce rate and conversion

3. **Define success criteria** (Week 1)
   - Set target recall improvement by query type
   - Define acceptable latency increase
   - Calculate ROI threshold for implementation

4. **Plan rollout** (Week 2)
   - Decide on A/B test vs full rollout
   - Define monitoring metrics
   - Plan iteration cycle based on results

### For Engineering Teams

1. **Set up hybrid search infrastructure** (Week 1)
   - Create full-text search index on text fields
   - Implement basic hybrid search with RRF
   - Verify latency is acceptable

2. **Build evaluation pipeline** (Week 1)
   - Create evaluation dataset with query types labeled
   - Implement comparison of semantic vs lexical vs hybrid
   - Add statistical significance testing

3. **Implement adaptive search** (Week 2)
   - Build query classifier
   - Implement weighted RRF based on query type
   - Add monitoring for category-specific performance

4. **Deploy and monitor** (Week 2)
   - Deploy hybrid search to production
   - Set up dashboards for performance by query type
   - Create alerts for performance degradation

---

## Reflection Questions

1. **What percentage of your queries involve exact terms, identifiers, or domain jargon?** Consider whether this percentage justifies hybrid search investment.

2. **How would you measure the business impact of improved search recall?** Think about conversion rates, user satisfaction, and support ticket reduction.

3. **What is the right balance between semantic and lexical search for your use case?** Consider whether query-dependent weighting would help.

4. **How would you handle queries that perform worse with hybrid search?** Some queries may get better results from pure semantic search.

5. **What monitoring would help you continuously improve hybrid search performance?** Think about query categorization accuracy and per-category metrics.

---

## Summary

### Key Takeaways for Product Managers

- **Hybrid search addresses semantic search blind spots**: Exact matches, product IDs, and domain jargon often need lexical search. If more than 10% of queries involve these patterns, hybrid search likely provides meaningful improvement.

- **Measure before implementing**: Segment queries by type and measure baseline performance. Hybrid search typically improves recall by 10-25%, but the improvement varies significantly by query type.

- **ROI is usually positive**: Implementation effort is moderate (1-2 weeks), and the business impact of improved search can be substantial. The e-commerce case study showed 29% revenue improvement.

- **Continuous monitoring matters**: Query patterns evolve over time. Set up monitoring by query type to catch performance changes and optimize weights accordingly.

### Key Takeaways for Engineers

- **Reciprocal Rank Fusion is the standard approach**: Use k=60 for balanced fusion. Start with equal weights, then optimize based on evaluation data.

- **LanceDB makes hybrid search simple**: Single parameter change to switch between semantic, lexical, and hybrid. Create full-text index before enabling lexical search.

- **Evaluate by query type**: Aggregate metrics hide segment-specific problems. Build evaluation pipelines that measure performance separately for different query categories.

- **Handle edge cases**: Empty results from one method, inconsistent preprocessing, and query classification errors all need graceful handling.

---

## Further Reading

### Academic Papers

- [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Original RRF paper
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720) - Learned sparse representations
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832) - Token-level interaction model

### Tools and Libraries

- [LanceDB](https://lancedb.com/) - Vector database with built-in hybrid search
- [Elasticsearch](https://www.elastic.co/) - Industry-standard lexical search with vector support
- [PostgreSQL + pgvector + pg_search](https://github.com/pgvector/pgvector) - SQL database with vector and full-text search

### Related Chapters

- [Chapter 0: Introduction](chapter0.md) - Semantic vs lexical search basics
- [Chapter 1: Evaluation-First Development](chapter1.md) - Evaluation framework for measuring improvement
- [Chapter 2: Training Data and Fine-Tuning](chapter2.md) - Fine-tuning embeddings for better semantic search

---

## Navigation

**Previous**: [Chapter 7: Production Operations](chapter7.md) - Deploying and maintaining RAG systems at scale

**Next**: [Chapter 9: Context Window Management](chapter9.md) - Managing context effectively for better generation

**Reference Materials**:

- Appendix A: Mathematical Foundations - BM25 formula derivation (coming soon)
- Appendix B: Algorithms Reference - RRF algorithm details (coming soon)
