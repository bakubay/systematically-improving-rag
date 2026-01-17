# Technical Concepts That Need Proper Introduction

This document identifies technical concepts that are used throughout the book but may not be properly introduced before they're referenced. These should be added to Chapter 0 (Introduction) or early chapters as foundational concepts.

## Foundational RAG Concepts

### Currently Missing or Under-Explained

1. **Embeddings and Vector Representations**
   - **Status**: Mentioned in prerequisites ("familiarity with embeddings") but not explained
   - **Where Used**: Throughout all chapters
   - **What's Needed**:
     - What embeddings are (dense vector representations)
     - How they capture semantic meaning
     - Why similar texts have similar embeddings
     - Basic intuition: "words with similar meanings are close in vector space"
   - **Add To**: Chapter 0 or early Chapter 1

2. **Vector Databases**
   - **Status**: Mentioned but not explained conceptually
   - **Where Used**: Chapter 1, throughout book
   - **What's Needed**:
     - What vector databases are (specialized storage for embeddings)
     - Why they're needed (efficient similarity search)
     - Approximate nearest neighbor (ANN) search concept
     - Basic tradeoffs (accuracy vs speed)
   - **Add To**: Chapter 0 or early Chapter 1

3. **Semantic Search vs Lexical Search**
   - **Status**: Lexical search mentioned but not contrasted with semantic
   - **Where Used**: Chapter 1, Chapter 5, hybrid search discussions
   - **What's Needed**:
     - Lexical search: keyword matching (BM25, exact match)
     - Semantic search: meaning-based (embeddings, vector similarity)
     - When each works best
     - Why hybrid combines both
   - **Add To**: Chapter 0 or early Chapter 1

4. **BM25 (Best Matching 25)**
   - **Status**: Mentioned in talks/office hours but not explained
   - **Where Used**: Hybrid search, lexical search discussions
   - **What's Needed**:
     - What BM25 is (probabilistic ranking function)
     - How it differs from TF-IDF
     - When to use it (exact matches, rare terms, filtering)
   - **Add To**: Chapter 1 or hybrid search chapter

5. **Chunking Strategies**
   - **Status**: Mentioned but strategies not explained
   - **Where Used**: Throughout, especially Chapter 5
   - **What's Needed**:
     - What chunking is (breaking documents into smaller pieces)
     - Common strategies: fixed-size, sentence-based, semantic, page-level
     - Tradeoffs: chunk size vs context preservation
     - Why it matters for retrieval
   - **Add To**: Chapter 0 or early Chapter 1

6. **Cosine Similarity**
   - **Status**: Used but not explained
   - **Where Used**: Throughout (embedding comparisons)
   - **What's Needed**:
     - What cosine similarity measures (angle between vectors)
     - Why it's used (normalized, works well for embeddings)
     - Range: -1 to 1, with 1 being identical
   - **Add To**: Chapter 0 or early Chapter 1

## Evaluation Concepts

7. **Precision vs Recall Tradeoff**
   - **Status**: Formulas given but intuition may be missing
   - **Where Used**: Chapter 1 extensively
   - **What's Needed**:
     - Intuitive explanation: precision = "are results good?", recall = "did we find everything?"
     - Why they trade off (retrieving more increases recall but may decrease precision)
     - When to optimize for each
   - **Add To**: Early Chapter 1 (before formulas)

8. **Leading vs Lagging Metrics**
   - **Status**: In glossary but may need earlier introduction
   - **Where Used**: Chapter 1, throughout book
   - **What's Needed**:
     - Leading metrics: actionable, predictive (experiment velocity, evaluation coverage)
     - Lagging metrics: outcomes, hard to control (user satisfaction, revenue)
     - Why leading metrics matter more for improvement
     - Analogy: calories (leading) vs weight (lagging)
   - **Add To**: Chapter 0 or early Chapter 1

9. **Statistical Significance Testing**
   - **Status**: Mentioned in deep dives but not introduced
   - **Where Used**: Chapter 1, evaluation sections
   - **What's Needed**:
     - What statistical significance means
     - Why it matters (distinguishing real improvements from noise)
     - Basic concepts: p-values, confidence intervals
     - When to use (A/B tests, metric comparisons)
   - **Add To**: Chapter 1 (before deep dive section)

## Fine-Tuning Concepts

10. **Bi-Encoder vs Cross-Encoder**
    - **Status**: In glossary but may need earlier introduction
    - **Where Used**: Chapter 2, re-ranking discussions
    - **What's Needed**:
      - Bi-encoder: independent encoding (fast, pre-computable)
      - Cross-encoder: joint encoding (slow, more accurate)
      - When to use each
      - Why re-rankers use cross-encoders
    - **Add To**: Early Chapter 2

11. **Contrastive Learning**
    - **Status**: In glossary, mentioned in deep dives
    - **Where Used**: Chapter 2 (fine-tuning)
    - **What's Needed**:
      - What contrastive learning is (learning by comparison)
      - Positive/negative pairs concept
      - Why it works for embeddings
      - Intuition: "pull similar things together, push different things apart"
    - **Add To**: Early Chapter 2 (before loss function deep dive)

12. **InfoNCE Loss**
    - **Status**: Mentioned in deep dives but not explained conceptually
    - **Where Used**: Chapter 2 (mathematical section)
    - **What's Needed**:
      - What InfoNCE is (contrastive loss function)
      - Why it's used (effective for embeddings)
      - Basic intuition before mathematical formulation
    - **Add To**: Early Chapter 2 (before mathematical deep dive)

13. **Re-Ranking**
    - **Status**: Mentioned but concept not fully explained
    - **Where Used**: Chapter 2, throughout
    - **What's Needed**:
      - What re-ranking is (re-scoring retrieved results)
      - Why it's used (improve precision at top-K)
      - Two-stage approach: retrieve many, re-rank to top few
      - Typical improvements (12-20%)
    - **Add To**: Early Chapter 2

## Specialized Retrieval Concepts

14. **RAPTOR Algorithm**
    - **Status**: Explained in Chapter 5 but may need earlier intuition
    - **Where Used**: Chapter 5 extensively
    - **What's Needed**:
      - What RAPTOR solves (long documents, information spread)
      - Basic concept: hierarchical summarization
      - When to use (documents >1500 pages, related info across sections)
    - **Add To**: Early Chapter 5 (before implementation)

15. **Hierarchical Clustering**
    - **Status**: Mentioned in RAPTOR but not explained
    - **Where Used**: Chapter 5 (RAPTOR implementation)
    - **What's Needed**:
      - What hierarchical clustering is (grouping similar items)
      - Dendrograms concept
      - Why it's used in RAPTOR (group related chunks)
    - **Add To**: Chapter 5 (before RAPTOR section)

16. **Metadata Extraction**
    - **Status**: Mentioned but concept not fully explained
    - **Where Used**: Chapter 5 (specialized retrieval)
    - **What's Needed**:
      - What metadata extraction is (structured info from unstructured text)
      - Why it matters (enables filtering, structured search)
      - Examples: dates, entities, categories
    - **Add To**: Early Chapter 5

17. **Synthetic Text Generation**
    - **Status**: Mentioned but not contrasted with extraction
    - **Where Used**: Chapter 5 (specialized retrieval strategies)
    - **What's Needed**:
      - What synthetic text generation is (creating new text from documents)
      - When to use vs metadata extraction
      - Examples: captions for images, summaries for tables
    - **Add To**: Early Chapter 5

## Query Understanding Concepts

18. **Query Clustering**
    - **Status**: Explained in Chapter 4 but may need earlier intuition
    - **Where Used**: Chapter 4 extensively
    - **What's Needed**:
      - What query clustering is (grouping similar queries)
      - Why it matters (identify patterns, prioritize improvements)
      - Basic approach: embed queries, cluster embeddings
    - **Add To**: Early Chapter 4

19. **Topic Modeling**
    - **Status**: Mentioned but not explained
    - **Where Used**: Chapter 4
    - **What's Needed**:
      - What topic modeling is (discovering themes in queries)
      - How it differs from clustering
      - When to use each
    - **Add To**: Chapter 4 (if used extensively)

## Routing Concepts

20. **Query Routing**
    - **Status**: Concept explained but may need earlier foundation
    - **Where Used**: Chapter 6 extensively
    - **What's Needed**:
      - What routing is (directing queries to specialized systems)
      - Why it's needed (different queries need different approaches)
      - Basic architecture: router → specialized retrievers
    - **Add To**: Early Chapter 6

21. **Few-Shot Classification**
    - **Status**: Mentioned but not explained
    - **Where Used**: Chapter 6 (routing)
    - **What's Needed**:
      - What few-shot learning is (learning from examples in prompt)
      - Why it's used for routing (quick setup, no training)
      - How many examples needed (10-40 for 88-95% accuracy)
    - **Add To**: Early Chapter 6

22. **Tools-as-APIs Pattern**
    - **Status**: Mentioned but concept not fully explained
    - **Where Used**: Chapter 6 (agentic RAG)
    - **What's Needed**:
      - What tools-as-APIs means (exposing retrieval as callable functions)
      - Why it's powerful (LLMs can choose when to retrieve)
      - Examples: grep, find, SQL queries
    - **Add To**: Early Chapter 6

## Production Concepts

23. **Semantic Caching**
    - **Status**: In glossary but may need earlier introduction
    - **Where Used**: Chapter 7
    - **What's Needed**:
      - What semantic caching is (caching by meaning, not exact match)
      - Why it matters (cost reduction, latency improvement)
      - Similarity thresholds (e.g., 0.95 cosine similarity)
    - **Add To**: Early Chapter 7

24. **Write-Time vs Read-Time Computation**
    - **Status**: In glossary but may need earlier introduction
    - **Where Used**: Chapter 7
    - **What's Needed**:
      - Write-time: preprocess at ingestion (higher storage, lower latency)
      - Read-time: compute on-demand (lower storage, higher latency)
      - When to choose each
    - **Add To**: Early Chapter 7

25. **Graceful Degradation**
    - **Status**: Mentioned but not explained
    - **Where Used**: Chapter 7 (production operations)
    - **What's Needed**:
      - What graceful degradation is (system continues with reduced functionality)
      - Why it matters (availability, user experience)
      - Examples: fallback to simpler retrieval, cached results
    - **Add To**: Chapter 7

## Advanced Concepts (May Need Appendices)

26. **Approximate Nearest Neighbor (ANN) Search**
    - **Status**: Mentioned but not explained
    - **Where Used**: Vector database discussions
    - **What's Needed**:
      - What ANN is (fast but approximate similarity search)
      - Why it's needed (exact search too slow for large datasets)
      - Tradeoffs: accuracy vs speed
      - Common algorithms: HNSW, IVF, LSH
    - **Add To**: Appendix or Chapter 1 deep dive

27. **Quantization**
    - **Status**: Mentioned in deep dives but not explained
    - **Where Used**: Chapter 2 (deployment)
    - **What's Needed**:
      - What quantization is (reducing precision: FP32 → INT8)
      - Why it's used (smaller models, faster inference)
      - Tradeoffs: accuracy vs size/speed
    - **Add To**: Chapter 2 (before deep dive) or appendix

28. **Gradient Accumulation**
    - **Status**: Mentioned in deep dives but not explained
    - **Where Used**: Chapter 2 (fine-tuning)
    - **What's Needed**:
      - What gradient accumulation is (simulating larger batches)
      - Why it's used (memory constraints)
      - How it works (accumulate gradients over multiple steps)
    - **Add To**: Chapter 2 (before deep dive) or appendix

## Concepts That Need Better Integration

29. **The Alignment Problem**
    - **Status**: Mentioned in glossary and case studies but may need earlier introduction
    - **Where Used**: Throughout, especially Chapter 5
    - **What's Needed**:
      - What alignment means (match between what you embed and what you search for)
      - Why it's critical (misalignment causes retrieval failures)
      - Examples: embedding product descriptions but searching for purchase patterns
    - **Add To**: Chapter 0 or early Chapter 1

30. **Inventory vs Capability Problem**
    - **Status**: Mentioned but may need clearer introduction
    - **Where Used**: Chapter 4, office hours
    - **What's Needed**:
      - Inventory problem: answer doesn't exist in knowledge base
      - Capability problem: answer exists but system can't find it
      - Why distinction matters (different solutions)
    - **Add To**: Chapter 0 or early Chapter 1

## Recommendations

### Priority 1: Add to Chapter 0 (Introduction)

These are foundational concepts used throughout:

- Embeddings and vector representations
- Vector databases
- Semantic vs lexical search
- Chunking strategies
- Cosine similarity
- The alignment problem
- Inventory vs capability problem

### Priority 2: Add to Early Chapters

Introduce before first use:

- Leading vs lagging metrics (Chapter 0 or early Chapter 1)
- Precision vs recall intuition (early Chapter 1)
- Bi-encoder vs cross-encoder (early Chapter 2)
- Contrastive learning (early Chapter 2)
- Re-ranking (early Chapter 2)
- Query clustering (early Chapter 4)
- Query routing (early Chapter 6)

### Priority 3: Add to Appendices or Deep Dives

More advanced concepts:

- ANN search algorithms
- Quantization techniques
- Gradient accumulation
- Statistical significance testing (if not in main text)

### Format Suggestions

- **For PMs**: Intuitive explanations, analogies, when to care
- **For Engineers**: Technical details, tradeoffs, implementation considerations
- Use visual diagrams where helpful (vector space, clustering, routing architecture)
- Include "Key Insight" callouts for critical concepts
- Cross-reference to glossary for detailed definitions
