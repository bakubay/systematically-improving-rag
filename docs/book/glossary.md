---
title: Glossary
description: Key terms and concepts used throughout the RAG improvement book series
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - reference
  - glossary
---

# Glossary

This glossary defines key terms used throughout the book. Terms are organized alphabetically for quick reference. Each entry includes the primary chapter where the concept is introduced.

---

## A

### Absence Blindness

The tendency to focus on what you can see (like generation quality) while ignoring what you cannot easily observe (like retrieval failures). Teams often spend weeks fine-tuning prompts without checking whether retrieval returns relevant documents in the first place.

**Example**: A team optimizes their prompt for three weeks, only to discover their retrieval system returns completely irrelevant documents for 40% of queries.

**See**: [Chapter 1](chapter1.md)

### Alignment Problem

The mismatch between what your queries ask about and what information your embeddings capture. If you embed only the first message of conversations but search for conversation patterns, you have an alignment problem.

**Example**: Embedding product descriptions but searching for "products similar to what I bought last month" fails because purchase history is not in the embeddings.

**See**: [Chapter 0](chapter0.md), [Chapter 5](chapter5.md)

### ANN (Approximate Nearest Neighbor)

A search algorithm that finds vectors similar to a query vector without exhaustively comparing every vector. Trades some accuracy for dramatically faster search times. Essential for scaling vector search to millions of documents.

**See**: [Chapter 0](chapter0.md)

---

## B

### Bi-encoder

An embedding model architecture where queries and documents are encoded independently into vectors, then compared using similarity metrics like cosine distance. Fast at query time because document embeddings can be precomputed, but less accurate than cross-encoders for ranking.

**Contrast with**: Cross-encoder, Re-ranker

**See**: [Chapter 2](chapter2.md)

### BM25

A lexical ranking function that scores documents based on term frequency and inverse document frequency. Does not understand semantics but excels at exact keyword matching and rare term retrieval.

**See**: [Chapter 0](chapter0.md), [Chapter 8](chapter8.md)

---

## C

### Chain-of-Thought

A technique where the model shows its reasoning process before providing a final answer. Improves accuracy on complex tasks and helps users verify the reasoning behind responses.

**See**: [Chapter 3](chapter3.md)

### Chunking

The process of dividing documents into smaller pieces for embedding and retrieval. Chunk size affects both retrieval quality and context utilization.

**Strategies**: Fixed-size, sentence-based, semantic, page-level

**See**: [Chapter 0](chapter0.md)

### Cold Start Problem

The challenge of building and improving a RAG system before you have real user data. Solved through synthetic data generation—creating realistic test queries from your document corpus.

**Example**: Generating 200 synthetic queries from legal case documents to establish baseline metrics before launching to users.

**See**: [Chapter 1](chapter1.md)

### Confidence Interval

A range of values that likely contains the true population parameter. For retrieval metrics, a 95% confidence interval means if you repeated the experiment 100 times, about 95 of those intervals would contain the true value.

**See**: [Chapter 1](chapter1.md), [Appendix A](appendix-math.md)

### Contrastive Learning

A training approach where models learn to distinguish between similar and dissimilar examples. For embeddings, this means training on triplets of (query, positive document, negative document) so the model learns to place queries closer to relevant documents in vector space.

**See**: [Chapter 2](chapter2.md)

### Cosine Similarity

A measure of similarity between two vectors based on the angle between them. Ranges from -1 (opposite) to 1 (identical). The standard similarity metric for comparing embeddings.

**Formula**: cos(θ) = (A · B) / (||A|| × ||B||)

**See**: [Chapter 0](chapter0.md), [Appendix A](appendix-math.md)

### Cross-encoder

A model architecture that processes query and document together as a single input, producing a relevance score. More accurate than bi-encoders but much slower because it cannot precompute document representations.

**Contrast with**: Bi-encoder

**See**: [Chapter 2](chapter2.md)

---

## D

### Data Flywheel

A self-reinforcing cycle where user interactions generate data that improves the system, which attracts more users, generating more data. The core concept of this book series.

```
User Interactions → Data Collection → System Improvements → Better UX → More Users → ...
```

**See**: [Chapter 0](chapter0.md), [Chapter 1](chapter1.md)

### Dynamic Context Assembly

Building the context window at query time based on the specific query and retrieved documents, rather than using a fixed template. Allows optimal use of limited token budgets.

**See**: [Chapter 9](chapter9.md)

---

## E

### Embedding

A dense vector representation of text (or other content) that captures semantic meaning. Similar texts have similar embeddings, enabling semantic search through vector similarity.

**Related**: Vector database, Cosine similarity

**See**: [Chapter 0](chapter0.md), [Chapter 2](chapter2.md)

### Experiment Velocity

The rate at which you can test hypotheses about your RAG system. The most important leading metric for early-stage systems. Teams that run 10 experiments per week improve faster than teams that run 1 experiment per month.

**See**: [Chapter 1](chapter1.md)

### Explicit Feedback

Direct user signals about satisfaction, such as thumbs up/down buttons, star ratings, or written comments.

**Contrast with**: Implicit feedback

**See**: [Chapter 3](chapter3.md)

---

## F

### F1 Score

The harmonic mean of precision and recall. Balances both metrics into a single number. Useful when you need to optimize for both finding relevant documents and avoiding irrelevant ones.

**Formula**: F1 = 2 × (Precision × Recall) / (Precision + Recall)

**See**: [Chapter 1](chapter1.md), [Appendix A](appendix-math.md)

### Few-shot Learning

Providing examples in the prompt to guide model behavior. For routing, 10 examples might achieve 88% accuracy while 40 examples reach 95%.

**See**: [Chapter 6](chapter6.md)

### Fine-tuning

Adapting a pre-trained model to your specific domain or task by training on your own data. For embeddings, this typically involves contrastive learning on query-document pairs.

**See**: [Chapter 2](chapter2.md)

---

## G

### Graceful Degradation

Designing systems to provide reduced but acceptable functionality when components fail, rather than failing completely.

**Example**: Returning cached results when the embedding service is unavailable.

**See**: [Chapter 7](chapter7.md)

### Gradient Accumulation

A training technique that simulates larger batch sizes by accumulating gradients over multiple forward passes before updating weights. Useful when GPU memory is limited.

**See**: [Chapter 2](chapter2.md)

---

## H

### Hard Negative

A document that appears relevant based on surface features (keywords, topic) but is actually not helpful for answering a specific query. Hard negatives are the most valuable training examples for improving retrieval because they teach the model subtle distinctions.

**Example**: For the query "Python memory management," a document about "Python snake habitats" is an easy negative (obviously wrong). A document about "Python garbage collection in version 2.7" when the user needs Python 3.11 information is a hard negative (seems relevant but is not).

**Contrast with**: Easy negative (completely unrelated documents)

**See**: [Chapter 2](chapter2.md), [Chapter 3](chapter3.md)

### Hybrid Search

Combining lexical search (keyword matching) with semantic search (embedding similarity). Often outperforms either approach alone because lexical search handles exact matches and rare terms while semantic search handles paraphrasing and conceptual similarity.

**See**: [Chapter 0](chapter0.md), [Chapter 8](chapter8.md)

---

## I

### Implicit Feedback

Signals about user satisfaction derived from behavior rather than explicit ratings. Includes query refinements (user rephrases immediately), abandonment, dwell time, citation clicks, and copy actions.

**Contrast with**: Explicit feedback (thumbs up/down, ratings)

**See**: [Chapter 3](chapter3.md)

### InfoNCE Loss

A contrastive loss function commonly used for training embedding models. Maximizes the similarity between positive pairs while minimizing similarity with negative examples in the batch.

**See**: [Chapter 2](chapter2.md), [Appendix A](appendix-math.md)

### Intervention Bias

The tendency to make changes just to feel like progress is being made, without measuring impact. Manifests as constantly switching models, tweaking prompts, or adding features without clear hypotheses.

**See**: [Chapter 1](chapter1.md)

### Inventory Problem

When a RAG system fails because the answer does not exist in the knowledge base—not because retrieval failed. No amount of better embeddings or re-ranking can fix missing data.

**Contrast with**: Capability problem (answer exists but system cannot find it)

**See**: [Chapter 0](chapter0.md)

---

## L

### Lagging Metric

An outcome metric you care about but cannot directly control: user satisfaction, churn rate, revenue. Like body weight—easy to measure, hard to change directly.

**Contrast with**: Leading metric

**See**: [Chapter 1](chapter1.md)

### Leading Metric

An actionable metric that predicts future performance and that you can directly influence: experiment velocity, evaluation coverage, feedback collection rate. Like calories consumed—you have direct control.

**Contrast with**: Lagging metric

**See**: [Chapter 1](chapter1.md)

### Lost in the Middle

A phenomenon where language models pay less attention to information in the middle of their context window compared to the beginning and end. Affects how you should order retrieved documents.

**See**: [Chapter 9](chapter9.md)

---

## M

### MAP (Mean Average Precision)

The mean of average precision scores across all queries. Considers both relevance and ranking position. Higher values indicate better ranking quality.

**See**: [Chapter 1](chapter1.md), [Appendix A](appendix-math.md)

### Metadata Filtering

Narrowing search results based on structured attributes (date, category, author) before or after semantic search. Can dramatically improve precision for queries with clear categorical constraints.

**See**: [Chapter 5](chapter5.md)

### MRR (Mean Reciprocal Rank)

The average of reciprocal ranks of the first relevant result across queries. If the first relevant result is at position 3, the reciprocal rank is 1/3.

**See**: [Chapter 1](chapter1.md), [Appendix A](appendix-math.md)

### Multimodal Retrieval

Retrieving content across different modalities (text, images, tables, code) using unified or specialized embedding models.

**See**: [Chapter 5](chapter5.md)

---

## N

### NDCG (Normalized Discounted Cumulative Gain)

A ranking metric that accounts for the position of relevant results and allows for graded relevance (not just binary). Discounts the value of results that appear lower in the ranking.

**See**: [Chapter 1](chapter1.md), [Appendix A](appendix-math.md)

---

## P

### Precision

Of the documents you retrieved, what percentage were actually relevant? If you returned 10 documents but only 2 were relevant, precision is 20%.

**Formula**: Precision = (Relevant ∩ Retrieved) / Retrieved

**Contrast with**: Recall

**See**: [Chapter 1](chapter1.md)

### Precision@K

Precision calculated for the top K results. Precision@5 means: of the top 5 documents returned, how many were relevant?

**See**: [Chapter 1](chapter1.md), [Appendix A](appendix-math.md)

---

## Q

### Query Clustering

Grouping similar user queries together to identify patterns, prioritize improvements, and understand user intent at scale.

**See**: [Chapter 4](chapter4.md)

### Query Expansion

Augmenting the original query with related terms or rephrased versions to improve recall. Can be done with synonyms, LLM-generated variations, or user feedback.

**See**: [Chapter 8](chapter8.md)

### Query Routing

Directing user queries to the appropriate specialized retriever or tool based on query characteristics. A router that achieves 95% accuracy with retrievers at 82% accuracy yields 78% end-to-end success (0.95 × 0.82).

**See**: [Chapter 6](chapter6.md)

---

## R

### RAG (Retrieval-Augmented Generation)

A pattern where relevant documents are retrieved from a knowledge base and provided as context to a language model for generating responses. Combines the knowledge storage of search systems with the language capabilities of LLMs.

**See**: [Chapter 0](chapter0.md)

### RAPTOR

Recursive Abstractive Processing for Tree-Organized Retrieval. A technique for handling long documents by creating hierarchical summaries—summaries of summaries—enabling retrieval at different levels of abstraction.

**See**: [Chapter 5](chapter5.md), [Appendix B](appendix-algorithms.md)

### Recall

Of all the relevant documents that exist, what percentage did you find? If there are 10 relevant documents and you found 4, recall is 40%.

**Formula**: Recall = (Relevant ∩ Retrieved) / Relevant

**Contrast with**: Precision

**See**: [Chapter 1](chapter1.md)

### Recall@K

Recall calculated when retrieving K documents. Recall@10 means: if you retrieve 10 documents, what percentage of all relevant documents did you find?

**See**: [Chapter 1](chapter1.md), [Appendix A](appendix-math.md)

### Reciprocal Rank Fusion (RRF)

A method for combining results from multiple retrieval systems. Scores each result based on its rank in each system, then combines scores. Simple but effective for hybrid search.

**Formula**: RRF(d) = Σ 1/(k + rank_i(d))

**See**: [Chapter 8](chapter8.md), [Appendix B](appendix-algorithms.md)

### Re-ranker

A model that re-scores retrieved documents to improve ranking. Typically a cross-encoder that is more accurate but slower than the initial bi-encoder retrieval. Applied to top-N results (e.g., retrieve 50, re-rank to top 10).

**Typical improvement**: 12-20% at top-5

**See**: [Chapter 2](chapter2.md)

---

## S

### Sample Size Calculation

Determining how many test cases you need to detect a meaningful difference between systems with statistical confidence. Depends on effect size, variance, and desired confidence level.

**See**: [Chapter 1](chapter1.md), [Appendix A](appendix-math.md)

### Semantic Cache

A cache that returns stored responses for queries that are semantically similar (not just identical) to previous queries. Requires setting a similarity threshold (e.g., 0.95 cosine similarity).

**See**: [Chapter 7](chapter7.md)

### Semantic Search

Finding documents based on meaning rather than exact keyword matches. Uses embeddings to represent queries and documents as vectors, then finds similar vectors.

**Contrast with**: Lexical search (keyword-based)

**See**: [Chapter 0](chapter0.md)

### Statistical Significance

The probability that an observed difference between systems is not due to random chance. Typically requires p < 0.05 (less than 5% probability the difference is random).

**See**: [Chapter 1](chapter1.md), [Appendix A](appendix-math.md)

### Streaming

Sending response tokens to the user as they are generated rather than waiting for the complete response. Reduces perceived latency and improves user experience.

**See**: [Chapter 3](chapter3.md)

### Synthetic Data

Artificially generated evaluation data, typically created by having an LLM generate questions that a document chunk should answer. Used to overcome the cold start problem and establish baselines before real user data exists.

**See**: [Chapter 1](chapter1.md)

---

## T

### Token Budget

The maximum number of tokens available in a context window, and how to allocate them between system prompt, retrieved context, conversation history, and generation.

**See**: [Chapter 9](chapter9.md)

### Topic Modeling

Automatically discovering themes or topics in a collection of documents or queries. Used for understanding query patterns and prioritizing improvements.

**See**: [Chapter 4](chapter4.md)

### Trellis Framework

A framework for organizing production monitoring of AI systems: (1) Discretize infinite outputs into specific buckets, (2) Prioritize by Volume × Negative Sentiment × Achievable Delta × Strategic Relevance, (3) Recursively refine within buckets.

**See**: [Chapter 1](chapter1.md)

### Triplet Loss

A loss function for training embeddings using (anchor, positive, negative) triplets. Pushes the anchor closer to the positive example and farther from the negative example.

**See**: [Chapter 2](chapter2.md), [Appendix A](appendix-math.md)

### Two-Level Performance Formula

For systems with routing to specialized retrievers, overall success = P(correct router) × P(correct retrieval | correct router). A 95% router with 82% retrieval yields 78% overall, while a 67% router with 80% retrieval yields only 54%.

**See**: [Chapter 6](chapter6.md)

---

## V

### Vector Database

A database optimized for storing and querying high-dimensional vectors (embeddings). Supports approximate nearest neighbor search to find similar vectors efficiently.

**Examples**: Pinecone, ChromaDB, pgvector, LanceDB, Weaviate

**See**: [Chapter 0](chapter0.md)

---

## W

### Write-time vs Read-time Computation

A fundamental architectural trade-off. Write-time computation (preprocessing) increases storage costs but improves query latency. Read-time computation (on-demand) reduces storage but increases latency. Choose based on content stability and latency requirements.

**See**: [Chapter 7](chapter7.md)

---

## Quick Reference: Key Formulas

| Metric | Formula | Use Case |
|--------|---------|----------|
| Precision@K | Relevant in top K / K | Measuring result quality |
| Recall@K | Relevant in top K / Total relevant | Measuring coverage |
| F1 Score | 2 × (P × R) / (P + R) | Balancing precision and recall |
| Cosine Similarity | (A · B) / (\|\|A\|\| × \|\|B\|\|) | Comparing embeddings |
| End-to-end success | P(router) × P(retrieval) | System performance |
| RRF Score | Σ 1/(k + rank_i(d)) | Combining search results |
| Prioritization score | Volume × (1 - Satisfaction) × Delta × Relevance | Roadmap planning |

---

*Return to [Chapter 0: Introduction](chapter0.md)*
