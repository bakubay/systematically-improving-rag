---
title: "Appendix B: Algorithms Reference"
description: "Comprehensive reference for algorithms used in RAG systems including RAPTOR, hierarchical clustering, router selection, and complexity analysis."
authors:
  - Jason Liu
date: 2025-01-18
tags:
  - reference
  - algorithms
  - RAPTOR
  - clustering
  - routing
---

# Appendix B: Algorithms Reference

This appendix provides complete algorithm definitions, pseudocode, and complexity analysis for all algorithms used throughout the book. Use this as a reference when implementing or optimizing RAG system components.

---

## RAPTOR Algorithm

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds hierarchical document representations by recursively clustering and summarizing text chunks.

### Overview

RAPTOR addresses the problem of retrieving information that spans multiple sections or requires understanding document structure. It builds a tree where:

- **Leaf nodes**: Original text chunks
- **Internal nodes**: Summaries of clustered children
- **Root**: High-level document summary

### Pseudocode

```
Algorithm: RAPTOR Tree Construction
Input: Document D, chunk_size, max_cluster_size, similarity_threshold
Output: Tree T with leaf and summary nodes

1. CHUNK(D, chunk_size) → chunks[]
2. EMBED(chunks) → embeddings[]
3. T.leaves ← chunks
4. current_level ← chunks

5. while |current_level| > 1:
6.     clusters ← CLUSTER(current_level, max_cluster_size, similarity_threshold)
7.     next_level ← []
8.     for cluster in clusters:
9.         summary ← SUMMARIZE(cluster)
10.        summary_embedding ← EMBED(summary)
11.        node ← CREATE_NODE(summary, summary_embedding, children=cluster)
12.        T.add_node(node)
13.        next_level.append(node)
14.    current_level ← next_level

15. return T
```

### Clustering Subroutine

```
Algorithm: CLUSTER
Input: nodes[], max_cluster_size, similarity_threshold
Output: clusters[][]

1. embeddings ← [node.embedding for node in nodes]
2. similarity_matrix ← COSINE_SIMILARITY(embeddings, embeddings)

3. # Hierarchical clustering with Ward linkage
4. linkage_matrix ← WARD_LINKAGE(1 - similarity_matrix)
5. cluster_labels ← CUT_TREE(linkage_matrix, threshold=similarity_threshold)

6. clusters ← GROUP_BY(nodes, cluster_labels)

7. # Split oversized clusters
8. final_clusters ← []
9. for cluster in clusters:
10.    if |cluster| > max_cluster_size:
11.        sub_clusters ← RECURSIVE_SPLIT(cluster, max_cluster_size)
12.        final_clusters.extend(sub_clusters)
13.    else:
14.        final_clusters.append(cluster)

15. return final_clusters
```

### Retrieval Algorithm

```
Algorithm: RAPTOR Retrieval
Input: Query q, Tree T, k (number of results)
Output: Retrieved nodes[]

1. query_embedding ← EMBED(q)
2. all_nodes ← T.get_all_nodes()  # Both leaves and summaries

3. # Score all nodes
4. scores ← []
5. for node in all_nodes:
6.     score ← COSINE_SIMILARITY(query_embedding, node.embedding)
7.     scores.append((node, score))

8. # Sort by score descending
9. scores.sort(key=lambda x: x[1], reverse=True)

10. # Return top-k
11. return [node for node, score in scores[:k]]
```

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Chunking | O(n) | O(n) |
| Embedding | O(n × d) | O(n × d) |
| Clustering (per level) | O(m² log m) | O(m²) |
| Summarization | O(m × s) | O(s) |
| Tree construction | O(n² log n) | O(n × d) |
| Retrieval | O(N × d) | O(N) |

Where:
- n = number of original chunks
- d = embedding dimension
- m = nodes at current level
- s = summary length
- N = total nodes in tree

### Implementation Notes

!!! tip "For Engineers"
    **Memory optimization**: For large documents, compute similarity matrices in batches rather than all at once. A 10,000-chunk document with 1536-dimensional embeddings requires ~900MB for the full similarity matrix.

    **Summarization quality**: Use a capable LLM for summarization. Poor summaries at lower levels propagate errors upward. Consider using structured prompts that preserve key entities and relationships.

    **Cluster size tuning**: Start with max_cluster_size=10 and similarity_threshold=0.5. Adjust based on document structure—technical documents may need smaller clusters, narrative documents can use larger ones.

---

## Hierarchical Clustering

Hierarchical clustering builds a tree of clusters by iteratively merging the most similar pairs.

### Ward Linkage

Ward's method minimizes the total within-cluster variance at each merge step.

**Merge criterion**: When merging clusters A and B, the increase in total variance is:

$$\Delta(A, B) = \frac{|A| \cdot |B|}{|A| + |B|} \cdot \|c_A - c_B\|^2$$

Where $c_A$ and $c_B$ are cluster centroids.

### Pseudocode

```
Algorithm: Agglomerative Hierarchical Clustering (Ward)
Input: points[], n_clusters or distance_threshold
Output: cluster_labels[]

1. # Initialize: each point is its own cluster
2. clusters ← [{i} for i in range(|points|)]
3. centroids ← points.copy()
4. linkage_matrix ← []

5. while |clusters| > n_clusters:
6.     # Find pair with minimum merge cost
7.     min_cost ← infinity
8.     merge_i, merge_j ← -1, -1
9.     
10.    for i in range(|clusters|):
11.        for j in range(i+1, |clusters|):
12.            cost ← WARD_DISTANCE(clusters[i], clusters[j], centroids)
13.            if cost < min_cost:
14.                min_cost ← cost
15.                merge_i, merge_j ← i, j
16.    
17.    # Merge clusters
18.    new_cluster ← clusters[merge_i] ∪ clusters[merge_j]
19.    new_centroid ← COMPUTE_CENTROID(points, new_cluster)
20.    
21.    # Update data structures
22.    clusters.remove(clusters[merge_j])
23.    clusters.remove(clusters[merge_i])
24.    clusters.append(new_cluster)
25.    centroids[merge_i] ← new_centroid
26.    
27.    # Record merge
28.    linkage_matrix.append([merge_i, merge_j, min_cost, |new_cluster|])

29. return EXTRACT_LABELS(clusters)
```

### Dendrogram Interpretation

The linkage matrix can be visualized as a dendrogram:

```
Height (distance)
    |
    |     ┌───────┐
    |  ┌──┤       │
    |  │  │   ┌───┤
    |  │  └───┤   │
    |  │      │   │
    └──┴──────┴───┴── Points
       1  2   3   4
```

**Cutting the dendrogram**: Choose a height threshold to determine cluster assignments. Lower cuts produce more clusters, higher cuts produce fewer.

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Naive implementation | O(n³) | O(n²) |
| Optimized (nearest-neighbor chain) | O(n² log n) | O(n²) |
| Distance matrix computation | O(n² × d) | O(n²) |

---

## Router Selection Algorithms

Query routing directs queries to appropriate specialized retrievers or tools.

### Classifier-Based Router

```
Algorithm: Classifier Router
Input: Query q, classifier model M, tools[]
Output: Selected tool, confidence

1. features ← EXTRACT_FEATURES(q)  # or EMBED(q)
2. probabilities ← M.predict_proba(features)
3. 
4. best_tool_idx ← argmax(probabilities)
5. confidence ← probabilities[best_tool_idx]
6. 
7. if confidence < threshold:
8.     return FALLBACK_TOOL, confidence
9. 
10. return tools[best_tool_idx], confidence
```

### Embedding-Based Router

```
Algorithm: Embedding Router
Input: Query q, tool_descriptions[], tool_embeddings[], k
Output: Selected tools[], scores[]

1. query_embedding ← EMBED(q)
2. 
3. # Compute similarities to all tool descriptions
4. similarities ← []
5. for i, tool_emb in enumerate(tool_embeddings):
6.     sim ← COSINE_SIMILARITY(query_embedding, tool_emb)
7.     similarities.append((i, sim))
8. 
9. # Sort by similarity
10. similarities.sort(key=lambda x: x[1], reverse=True)
11. 
12. # Return top-k tools
13. selected ← [(tools[i], sim) for i, sim in similarities[:k]]
14. return selected
```

### LLM-Based Router with Few-Shot Examples

```
Algorithm: LLM Router
Input: Query q, tools[], examples[], llm
Output: Selected tool, extracted_params

1. # Build prompt with tool descriptions
2. tool_descriptions ← FORMAT_TOOLS(tools)
3. 
4. # Select relevant examples
5. relevant_examples ← SELECT_EXAMPLES(q, examples, k=5)
6. examples_text ← FORMAT_EXAMPLES(relevant_examples)
7. 
8. # Construct prompt
9. prompt ← f"""
10. Available tools:
11. {tool_descriptions}
12. 
13. Examples:
14. {examples_text}
15. 
16. Query: {q}
17. 
18. Select the appropriate tool and extract parameters.
19. """
20. 
21. # Get structured response
22. response ← llm.generate(prompt, response_format=ToolSelection)
23. 
24. return response.tool, response.params
```

### Dynamic Example Selection

```
Algorithm: Dynamic Example Selection
Input: Query q, example_pool[], k
Output: Selected examples[]

1. query_embedding ← EMBED(q)
2. 
3. # Embed all examples (can be pre-computed)
4. example_embeddings ← [EMBED(ex.query) for ex in example_pool]
5. 
6. # Find most similar examples
7. similarities ← []
8. for i, ex_emb in enumerate(example_embeddings):
9.     sim ← COSINE_SIMILARITY(query_embedding, ex_emb)
10.    similarities.append((i, sim))
11. 
12. similarities.sort(key=lambda x: x[1], reverse=True)
13. 
14. # Select top-k, ensuring diversity
15. selected ← []
16. selected_tools ← set()
17. 
18. for i, sim in similarities:
19.    ex ← example_pool[i]
20.    # Ensure tool diversity
21.    if ex.tool not in selected_tools or |selected| < k // 2:
22.        selected.append(ex)
23.        selected_tools.add(ex.tool)
24.    if |selected| >= k:
25.        break
26. 
27. return selected
```

### Complexity Analysis

| Router Type | Inference Time | Training Time | Space |
|-------------|---------------|---------------|-------|
| Classifier | O(d) | O(n × d × epochs) | O(d × c) |
| Embedding | O(t × d) | None | O(t × d) |
| LLM | O(prompt_tokens) | None | O(1) |

Where:
- d = feature/embedding dimension
- c = number of classes/tools
- t = number of tools
- n = training examples

---

## Reciprocal Rank Fusion (RRF)

RRF combines results from multiple retrieval systems by aggregating ranks.

### Algorithm

```
Algorithm: Reciprocal Rank Fusion
Input: ranked_lists[][], k (constant, typically 60)
Output: fused_ranking[]

1. scores ← {}  # document_id → score
2. 
3. for ranked_list in ranked_lists:
4.     for rank, doc_id in enumerate(ranked_list, start=1):
5.         if doc_id not in scores:
6.             scores[doc_id] ← 0
7.         scores[doc_id] += 1 / (k + rank)
8. 
9. # Sort by aggregated score
10. fused_ranking ← sorted(scores.items(), key=lambda x: x[1], reverse=True)
11. 
12. return [doc_id for doc_id, score in fused_ranking]
```

### Weighted RRF

```
Algorithm: Weighted Reciprocal Rank Fusion
Input: ranked_lists[][], weights[], k
Output: fused_ranking[]

1. scores ← {}
2. 
3. for i, ranked_list in enumerate(ranked_lists):
4.     weight ← weights[i]
5.     for rank, doc_id in enumerate(ranked_list, start=1):
6.         if doc_id not in scores:
7.             scores[doc_id] ← 0
8.         scores[doc_id] += weight / (k + rank)
9. 
10. fused_ranking ← sorted(scores.items(), key=lambda x: x[1], reverse=True)
11. return [doc_id for doc_id, score in fused_ranking]
```

### Why k=60?

The constant k controls how much weight is given to lower-ranked results:

| k value | Effect |
|---------|--------|
| k=1 | Top result gets 50% of weight, steep dropoff |
| k=60 | Top result gets ~1.6% of weight, gradual dropoff |
| k=100 | Top result gets ~1% of weight, very gradual |

k=60 was empirically determined to work well across diverse retrieval tasks.

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Single list processing | O(n) | O(n) |
| m lists fusion | O(m × n) | O(m × n) |
| Final sorting | O(N log N) | O(N) |

Where n = documents per list, N = unique documents across all lists.

---

## K-Means Clustering for Query Segmentation

K-means partitions queries into k clusters based on embedding similarity.

### Algorithm

```
Algorithm: K-Means Clustering
Input: embeddings[][], k, max_iterations
Output: cluster_labels[], centroids[][]

1. # Initialize centroids (k-means++)
2. centroids ← KMEANS_PLUS_PLUS_INIT(embeddings, k)
3. 
4. for iteration in range(max_iterations):
5.     # Assignment step
6.     labels ← []
7.     for emb in embeddings:
8.         distances ← [EUCLIDEAN(emb, c) for c in centroids]
9.         labels.append(argmin(distances))
10.    
11.    # Update step
12.    new_centroids ← []
13.    for i in range(k):
14.        cluster_points ← [emb for emb, l in zip(embeddings, labels) if l == i]
15.        if |cluster_points| > 0:
16.            new_centroids.append(MEAN(cluster_points))
17.        else:
18.            new_centroids.append(centroids[i])  # Keep old centroid
19.    
20.    # Check convergence
21.    if centroids == new_centroids:
22.        break
23.    centroids ← new_centroids
24. 
25. return labels, centroids
```

### K-Means++ Initialization

```
Algorithm: K-Means++ Initialization
Input: embeddings[][], k
Output: initial_centroids[][]

1. # Choose first centroid uniformly at random
2. centroids ← [RANDOM_CHOICE(embeddings)]
3. 
4. for i in range(1, k):
5.     # Compute distances to nearest centroid
6.     distances ← []
7.     for emb in embeddings:
8.         min_dist ← min([EUCLIDEAN(emb, c) for c in centroids])
9.         distances.append(min_dist ** 2)
10.    
11.    # Sample proportional to squared distance
12.    probabilities ← distances / sum(distances)
13.    new_centroid ← WEIGHTED_RANDOM_CHOICE(embeddings, probabilities)
14.    centroids.append(new_centroid)
15. 
16. return centroids
```

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Initialization (k-means++) | O(n × k × d) | O(k × d) |
| Single iteration | O(n × k × d) | O(n) |
| Total (t iterations) | O(t × n × k × d) | O(n + k × d) |

---

## Semantic Caching

Semantic caching stores and retrieves responses based on query similarity.

### Cache Lookup Algorithm

```
Algorithm: Semantic Cache Lookup
Input: Query q, cache, similarity_threshold
Output: Cached response or None

1. query_embedding ← EMBED(q)
2. 
3. # Search cache for similar queries
4. candidates ← cache.search(query_embedding, k=10)
5. 
6. for cached_query, cached_response, similarity in candidates:
7.     if similarity >= similarity_threshold:
8.         # Additional validation
9.         if VALIDATE_CACHE_HIT(q, cached_query):
10.            cache.update_access_time(cached_query)
11.            return cached_response
12. 
13. return None
```

### Cache Update Algorithm

```
Algorithm: Semantic Cache Update
Input: Query q, response r, cache, max_size
Output: Updated cache

1. query_embedding ← EMBED(q)
2. 
3. # Check if similar query already cached
4. existing ← cache.search(query_embedding, k=1)
5. if existing and existing[0].similarity > 0.99:
6.     # Update existing entry
7.     cache.update(existing[0].key, response=r)
8.     return cache
9. 
10. # Add new entry
11. if cache.size >= max_size:
12.    # Eviction: remove least recently used
13.    cache.evict_lru()
14. 
15. cache.add(query=q, embedding=query_embedding, response=r)
16. return cache
```

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Embedding | O(d) | O(d) |
| Cache search (ANN) | O(log n) | O(1) |
| Cache update | O(log n) | O(d + r) |

Where d = embedding dimension, n = cache size, r = response size.

---

## Quick Reference Tables

### Algorithm Selection Guide

| Problem | Recommended Algorithm | When to Use |
|---------|----------------------|-------------|
| Long document retrieval | RAPTOR | Documents > 1500 pages, cross-section queries |
| Query segmentation | K-Means | Understanding query distribution, 1000+ queries |
| Multi-source fusion | RRF | Combining semantic + lexical search |
| Query routing | LLM Router | < 1000 queries/day, complex routing logic |
| Query routing | Classifier | > 10000 queries/day, stable categories |
| Response caching | Semantic Cache | Repetitive queries, cost optimization |

### Complexity Summary

| Algorithm | Time | Space | Key Parameter |
|-----------|------|-------|---------------|
| RAPTOR Construction | O(n² log n) | O(n × d) | chunk_size |
| RAPTOR Retrieval | O(N × d) | O(N) | k (results) |
| Hierarchical Clustering | O(n² log n) | O(n²) | distance_threshold |
| K-Means | O(t × n × k × d) | O(n + k × d) | k (clusters) |
| RRF | O(m × n) | O(m × n) | k (constant=60) |
| Semantic Cache | O(log n) | O(n × d) | similarity_threshold |

---

## Navigation

- **Previous**: [Appendix A: Mathematical Foundations](appendix-math.md) - Retrieval metrics and statistical methods
- **Next**: [Appendix C: Benchmarking Your RAG System](appendix-benchmarks.md) - Evaluation methodology and datasets
- **Reference**: [Glossary](glossary.md) | [Quick Reference](quick-reference.md)
- **Book Index**: [Book Overview](index.md)
