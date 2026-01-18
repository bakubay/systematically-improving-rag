---
title: "Chapter 2: Training Data and Fine-Tuning"
description: "Learn how to convert evaluation data into training datasets, fine-tune embedding models and re-rankers, and apply contrastive learning techniques to achieve 6-20% improvements in retrieval performance."
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - fine-tuning
  - embeddings
  - re-rankers
  - contrastive learning
  - training data
  - bi-encoder
  - cross-encoder
---

# Chapter 2: Training Data and Fine-Tuning

## Chapter at a Glance

**Prerequisites**: Chapter 0 (foundational concepts), Chapter 1 (evaluation frameworks), basic Python, familiarity with embeddings

**What You Will Learn**:

- Why off-the-shelf embeddings fail for specialized applications
- The difference between bi-encoders and cross-encoders and when to use each
- How contrastive learning works and why hard negatives matter
- How to convert evaluation data into training datasets
- Practical workflows for fine-tuning embedding models and re-rankers
- Loss functions and training strategies for retrieval systems

**Case Study Reference**: Glean (20% improvement with custom embeddings), LanceDB (12% improvement with re-rankers), Healthcare RAG (72% to 89% recall)

**Time to Complete**: 75-90 minutes

---

## Key Insight

**The goal is not to fine-tune language models (expensive and complex), but to fine-tune embedding models that move toward your specific data distributions and improve retrieval.** With just 6,000 examples, you can achieve 6-10% better performance. Training takes 40 minutes on a laptop and costs around $1.50 in compute. This is the machine learning playbook that used to cost hundreds of thousands of dollars in data labeling—now accessible to any team with a few prompts and a for loop.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain why generic embeddings fail for specialized applications and identify the hidden assumptions in provider models
2. Distinguish between bi-encoders and cross-encoders and select the appropriate architecture for your use case
3. Apply contrastive learning techniques using triplet structures with hard negatives
4. Convert evaluation examples from Chapter 1 into training datasets for fine-tuning
5. Implement practical fine-tuning workflows for embedding models and re-rankers
6. Understand loss functions (InfoNCE, triplet loss) and training strategies at a conceptual and mathematical level

---

## Introduction

In Chapter 1, we established evaluation-first development—the practice of measuring before optimizing. You built evaluation datasets, calculated precision and recall, and learned to run systematic experiments. That evaluation data is not just for measurement. It becomes the foundation for improvement.

This chapter transforms your evaluation data into training data. The synthetic questions and relevance judgments from Chapter 1 become the triplets that teach your embedding models what "similar" actually means for your application. The flywheel continues: evaluation data becomes training data, training improves retrieval, better retrieval generates better data.

**Building on Chapter 1's Foundation:**

- 20 examples became evaluation baselines
- 30 examples became few-shot prompts
- 1,000+ examples become fine-tuning datasets

Every piece of data serves multiple purposes. The key insight is that you are not throwing away data—you are using it differently as you accumulate more.

!!! tip "For Product Managers"
    This chapter explains why generic embedding models underperform and what it takes to improve them. Focus on understanding the business value of fine-tuning (6-20% recall improvements), the cost-benefit tradeoffs (hundreds of dollars vs tens of thousands), and when fine-tuning makes sense for your use case. You do not need to understand the mathematical details, but knowing what is possible will help you prioritize investments.

!!! tip "For Engineers"
    This chapter provides the technical foundation for improving retrieval through fine-tuning. Pay close attention to the contrastive learning concepts, loss function derivations, and training strategies. The code examples show practical implementations you can adapt for your systems. Understanding bi-encoders vs cross-encoders is essential for architectural decisions.

---

## Core Content

### Bi-Encoders vs Cross-Encoders

Before diving into fine-tuning, you need to understand the two fundamental architectures for retrieval: bi-encoders and cross-encoders. This distinction determines your system's speed, accuracy, and fine-tuning approach.

!!! tip "For Product Managers"
    **The fundamental tradeoff**: Bi-encoders are fast but less accurate. Cross-encoders are accurate but slow. Most production systems use both—bi-encoders for initial retrieval, cross-encoders for re-ranking.

    **Cost and speed implications**:

    | Architecture | Speed | Accuracy | Use Case |
    |--------------|-------|----------|----------|
    | Bi-encoder | Fast (pre-computed) | Good | First-pass retrieval |
    | Cross-encoder | Slow (computed per pair) | Better | Re-ranking top results |
    | Combined | Fast + accurate | Best | Production systems |

    **Business decision framework**:

    - **Bi-encoder only**: Acceptable when latency is critical (<100ms) and precision requirements are moderate
    - **Cross-encoder re-ranking**: Worth the latency cost (30-500ms) when precision matters (legal, medical, financial)
    - **Combined approach**: Standard for most production systems—retrieve 50-100 candidates with bi-encoder, re-rank top 10-20 with cross-encoder

    **Real numbers from production**:

    - Re-rankers typically add 30ms (GPU) to 300-500ms (CPU) latency
    - Re-rankers typically improve recall by 10-20%
    - The ROI is almost always positive for non-latency-critical applications

!!! tip "For Engineers"
    **Bi-encoders (embedding models)**:

    Bi-encoders process queries and documents independently. Each text is encoded into a fixed-size vector, and similarity is computed via cosine distance or dot product.

    ```python
    # Bi-encoder: Independent encoding
    query_embedding = model.encode(query)        # Computed at query time
    doc_embedding = model.encode(document)       # Pre-computed, stored in vector DB
    similarity = cosine_similarity(query_embedding, doc_embedding)
    ```

    **Key characteristics**:

    - Documents can be pre-embedded and stored in vector databases
    - Query-time computation is minimal (one encoding + similarity search)
    - Scales to millions of documents
    - Examples: OpenAI text-embedding, Cohere embed, SBERT, BGE, E5

    **Cross-encoders (re-rankers)**:

    Cross-encoders process query-document pairs together, allowing full attention between both texts.

    ```python
    # Cross-encoder: Joint processing
    relevance_score = model.predict([query, document])  # Computed per pair
    ```

    **Key characteristics**:

    - Cannot pre-compute scores (depends on query)
    - Must evaluate each candidate document separately
    - Computationally expensive: O(n) model calls for n documents
    - Much more accurate due to cross-attention
    - Examples: Cohere Rerank, monoT5, cross-encoder models

    **Why cross-encoders are more accurate**:

    Cross-encoders can attend to relationships between query and document tokens. A bi-encoder encoding "medication side effects" cannot know it will be compared to a document about "adverse reactions"—it must encode a general representation. A cross-encoder sees both texts and can recognize the semantic equivalence.

    **Architecture comparison**:

    ```mermaid
    graph LR
        subgraph "Bi-Encoder"
            Q1[Query] --> E1[Encoder]
            D1[Document] --> E2[Encoder]
            E1 --> V1[Query Vector]
            E2 --> V2[Doc Vector]
            V1 --> S1[Similarity]
            V2 --> S1
        end

        subgraph "Cross-Encoder"
            Q2[Query] --> C1[Combined Input]
            D2[Document] --> C1
            C1 --> E3[Encoder with Cross-Attention]
            E3 --> S2[Relevance Score]
        end
    ```

    **Practical implementation pattern**:

    ```python
    from typing import List, Dict
    import numpy as np

    class TwoStageRetriever:
        def __init__(
            self,
            bi_encoder,
            cross_encoder,
            initial_k: int = 100,
            final_k: int = 10
        ):
            self.bi_encoder = bi_encoder
            self.cross_encoder = cross_encoder
            self.initial_k = initial_k
            self.final_k = final_k

        def retrieve(self, query: str, documents: List[Dict]) -> List[Dict]:
            """Two-stage retrieval: bi-encoder then cross-encoder."""
            # Stage 1: Fast retrieval with bi-encoder
            query_embedding = self.bi_encoder.encode(query)
            candidates = self._vector_search(query_embedding, self.initial_k)

            # Stage 2: Re-rank with cross-encoder
            pairs = [(query, doc['text']) for doc in candidates]
            scores = self.cross_encoder.predict(pairs)

            # Sort by cross-encoder scores
            for doc, score in zip(candidates, scores):
                doc['rerank_score'] = score

            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
            return reranked[:self.final_k]
    ```

---

### Contrastive Learning

Contrastive learning is the foundation of embedding fine-tuning. Instead of teaching a model what things are, you teach it what things are similar to and different from.

!!! tip "For Product Managers"
    **Why contrastive learning works**:

    Traditional supervised learning says "this is a cat." Contrastive learning says "this cat is more similar to that cat than to this dog." For retrieval, this is exactly what we need—we want queries to be closer to relevant documents than irrelevant ones.

    **Business value**:

    - Training with only positive examples: ~6% improvement
    - Training with hard negatives: ~30% improvement (5x multiplier)
    - The quality of your negative examples determines the quality of your fine-tuned model

    **What this means for your team**:

    1. **Data collection priority**: Collect both positive and negative signals from users
    2. **UX design opportunity**: Design interfaces that capture what users reject, not just what they accept
    3. **Investment justification**: Hard negative mining is worth the engineering effort—it multiplies your training ROI

!!! tip "For Engineers"
    **Triplet structure**:

    The most common contrastive learning setup uses triplets:

    1. **Anchor**: The query
    2. **Positive**: A document that is relevant to the query
    3. **Negative**: A document that is not relevant to the query

    The training objective: make the anchor closer to the positive than to the negative.

    ```python
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class Triplet:
        anchor: str      # Query
        positive: str    # Relevant document
        negative: str    # Irrelevant document

    # Example triplet for healthcare RAG
    triplet = Triplet(
        anchor="What are the side effects of medication X?",
        positive="Medication X may cause drowsiness, nausea, and in rare cases, allergic reactions.",
        negative="Medication X is used to treat high blood pressure and should be taken with food."
    )
    ```

    **Visual representation**:

    ```mermaid
    graph LR
        A[Anchor: Query] --- P[Positive: Relevant Doc]
        A --- N[Negative: Irrelevant Doc]
        P -.- |"Pull Closer"| A
        N -.- |"Push Away"| A
    ```

    **Creating triplets from RAG data**:

    ```python
    def create_triplets_from_rag_logs(
        query: str,
        retrieved_docs: List[Dict],
        cited_doc_ids: set[str]
    ) -> List[Triplet]:
        """Create triplets from RAG interaction logs.

        Positive: Documents that were cited in the response
        Negative: Documents that were retrieved but not cited
        """
        triplets = []

        positives = [d for d in retrieved_docs if d['id'] in cited_doc_ids]
        negatives = [d for d in retrieved_docs if d['id'] not in cited_doc_ids]

        for positive in positives:
            for negative in negatives:
                triplets.append(Triplet(
                    anchor=query,
                    positive=positive['text'],
                    negative=negative['text']
                ))

        return triplets
    ```

    **The critical importance of hard negatives**:

    Easy negatives (completely unrelated documents) teach the model nothing—it already knows car maintenance is not relevant to medication questions. Hard negatives (similar but wrong documents) teach the boundaries between concepts.

    **Hard negative characteristics**:

    1. Surface-level similarity to the query (same domain, related concepts)
    2. NOT actually relevant to the user's intent
    3. The distinction is meaningful and teachable

    **Hard negative mining strategies**:

    ```python
    def mine_hard_negatives(
        query: str,
        positive_doc: str,
        corpus: List[Dict],
        embedding_model,
        num_negatives: int = 5
    ) -> List[str]:
        """Mine hard negatives using embedding similarity.

        Find documents that are similar to the query but not the positive.
        """
        query_embedding = embedding_model.encode(query)

        # Score all documents
        candidates = []
        for doc in corpus:
            if doc['text'] == positive_doc:
                continue
            doc_embedding = embedding_model.encode(doc['text'])
            similarity = cosine_similarity(query_embedding, doc_embedding)
            candidates.append((doc['text'], similarity))

        # Sort by similarity (highest first) - these are hard negatives
        candidates.sort(key=lambda x: x[1], reverse=True)

        return [text for text, _ in candidates[:num_negatives]]
    ```

    **Real-world hard negative examples**:

    | Domain | Query | Hard Negative | Why It's Hard |
    |--------|-------|---------------|---------------|
    | Finance | "Employee fuel reimbursement" | "Equipment fuel for company tractors" | Same term "fuel", different category |
    | Medical | "MS symptoms" | "MS (mitral stenosis) diagnosis" | Same abbreviation, different condition |
    | Legal | "2024 tax rates" | "2023 tax rates" | Same topic, wrong time period |
    | E-commerce | "Red running shoes" | "Red dress shoes" | Same color, different intent |

    **UX patterns for collecting hard negatives**:

    1. **Document-level feedback**: Thumbs up/down on each retrieved document
    2. **Click tracking**: Documents retrieved but not clicked are potential negatives
    3. **Dwell time**: Quick returns from documents indicate irrelevance
    4. **User deletion signals**: Documents users actively remove are perfect hard negatives
    5. **Query reformulation**: When users rephrase and get better results, the original results become negatives

---

### Re-Ranking

Re-rankers are cross-encoders applied after initial retrieval. They are one of the highest-ROI improvements you can make to a RAG system.

!!! tip "For Product Managers"
    **ROI of re-rankers**:

    | Metric | Typical Improvement |
    |--------|---------------------|
    | Top-5 results | 12% better |
    | Top-10 results | 6-10% better |
    | Full-text ranking | Up to 20% better |

    **When re-rankers provide the most value**:

    - Initial retrieval returns many "close but not quite" candidates
    - Subtle relevance distinctions matter (medical, legal, technical)
    - User queries are complex or ambiguous
    - Cost of showing wrong results is high

    **When to skip re-rankers**:

    - Initial retrieval already achieves 90%+ precision
    - Latency requirements are strict (<100ms total)
    - Query patterns are simple and well-defined
    - Document corpus is small and homogeneous

    **Cost-benefit analysis**:

    ```
    Without re-ranker:
    - Latency: 100ms
    - Recall@10: 65%

    With re-ranker:
    - Latency: 130-400ms (depending on GPU/CPU)
    - Recall@10: 75-82%

    Question: Is 10-17% better recall worth 30-300ms latency?
    Answer: Almost always yes, unless you're building real-time systems.
    ```

    **Success story**: One team debated whether to invest in fine-tuning embeddings or implementing a re-ranker. Testing showed:

    - Fine-tuning embeddings: 65% to 78% recall
    - Adding re-ranker (no fine-tuning): 65% to 82% recall
    - Both combined: 65% to 91% recall

!!! tip "For Engineers"
    **Re-ranker implementation**:

    ```python
    from typing import List, Dict, Tuple

    class ReRanker:
        def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)

        def rerank(
            self,
            query: str,
            documents: List[Dict],
            top_k: int = 10
        ) -> List[Dict]:
            """Re-rank documents using cross-encoder."""
            # Create query-document pairs
            pairs = [(query, doc['text']) for doc in documents]

            # Get relevance scores
            scores = self.model.predict(pairs)

            # Attach scores and sort
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)

            reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
            return reranked[:top_k]
    ```

    **Using Cohere Rerank API**:

    ```python
    import cohere

    class CohereReRanker:
        def __init__(self, api_key: str):
            self.client = cohere.Client(api_key)

        def rerank(
            self,
            query: str,
            documents: List[str],
            top_k: int = 10
        ) -> List[Dict]:
            """Re-rank using Cohere's API."""
            response = self.client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model="rerank-english-v3.0"
            )

            return [
                {
                    'text': documents[result.index],
                    'score': result.relevance_score,
                    'original_index': result.index
                }
                for result in response.results
            ]
    ```

    **Graded relevance for re-ranker training**:

    Binary relevance (yes/no) works, but graded relevance (0-5 scale) produces better re-rankers.

    ```python
    # Binary relevance
    training_data_binary = [
        {"query": "password reset", "document": "Step-by-step password reset guide", "label": 1},
        {"query": "password reset", "document": "About our company", "label": 0},
    ]

    # Graded relevance (better)
    training_data_graded = [
        {"query": "password reset", "document": "Step-by-step password reset guide", "score": 5},
        {"query": "password reset", "document": "General account management info", "score": 3},
        {"query": "password reset", "document": "Creating a strong password", "score": 2},
        {"query": "password reset", "document": "About our company", "score": 0},
    ]
    ```

    **Latency considerations**:

    | Deployment | Latency per Query | Notes |
    |------------|-------------------|-------|
    | GPU (L4/T4) | ~30ms | Recommended for production |
    | CPU | 100-500ms | 4-5x slower than GPU |
    | API (Cohere) | 50-200ms | Includes network latency |

---

### Embedding Fine-Tuning

Fine-tuning embedding models customizes what "similarity" means for your application.

!!! tip "For Product Managers"
    **When to fine-tune vs use off-the-shelf**:

    | Scenario | Recommendation |
    |----------|----------------|
    | Generic domain, small corpus | Use off-the-shelf |
    | Specialized terminology | Fine-tune |
    | Domain-specific similarity | Fine-tune |
    | 6,000+ labeled examples | Fine-tune |
    | Cost at scale matters | Fine-tune (self-host) |

    **The hidden assumptions in provider models**:

    When you use OpenAI or Cohere embeddings, you inherit their definition of similarity. They do not know:

    - That in your legal application, procedural questions need different treatment than factual questions
    - That in your e-commerce application, "similar" means complementary products, not substitutes
    - That in your healthcare application, "MS" means multiple sclerosis, not mitral stenosis

    **Real-world example**: A dating app asks whether "I love coffee" and "I hate coffee" should be similar. The words are opposites, but both indicate strong food preferences. Generic embeddings see them as opposites. For matching people who care about food and drink, they might actually be similar.

    **Cost comparison**:

    | Approach | Cost | Time | Infrastructure |
    |----------|------|------|----------------|
    | Off-the-shelf API | $0 upfront | Immediate | None |
    | Fine-tuned embedding | ~$1.50 | 40 minutes | Laptop GPU |
    | Fine-tuned LLM | $100-1000s | Hours to days | Multiple GPUs |

    **Glean's approach**: Glean builds custom embedding models for every customer. After six months of learning from user feedback, they typically see 20% improvement in search performance. Their insight: smaller, fine-tuned models often outperform large general-purpose models for specific enterprise contexts.

!!! tip "For Engineers"
    **Fine-tuning workflow**:

    **Step 1: Prepare training data**

    ```python
    from dataclasses import dataclass
    from typing import List
    import json

    @dataclass
    class TrainingExample:
        anchor: str
        positive: str
        negative: str | None = None

    def prepare_training_data(
        evaluation_examples: List[dict],
        corpus: List[dict],
        embedding_model
    ) -> List[TrainingExample]:
        """Convert evaluation data to training triplets."""
        training_data = []

        for example in evaluation_examples:
            query = example['query']
            positive_ids = set(example['relevant_doc_ids'])

            # Get positive documents
            positives = [d for d in corpus if d['id'] in positive_ids]

            # Mine hard negatives
            for positive in positives:
                negatives = mine_hard_negatives(
                    query=query,
                    positive_doc=positive['text'],
                    corpus=corpus,
                    embedding_model=embedding_model,
                    num_negatives=3
                )

                for negative in negatives:
                    training_data.append(TrainingExample(
                        anchor=query,
                        positive=positive['text'],
                        negative=negative
                    ))

        return training_data
    ```

    **Step 2: Choose base model**

    ```python
    # For English-only applications
    base_models_english = [
        "sentence-transformers/all-MiniLM-L6-v2",      # Small, fast
        "BAAI/bge-base-en-v1.5",                        # Good balance
        "sentence-transformers/all-mpnet-base-v2",     # Higher quality
    ]

    # For multilingual applications
    base_models_multilingual = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-large",
    ]

    # Modern BERT models (8,000 token context)
    modern_models = [
        "answerdotai/ModernBERT-base",
        "answerdotai/ModernBERT-large",
    ]
    ```

    **Step 3: Fine-tune with Sentence Transformers**

    ```python
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    def fine_tune_embedding_model(
        training_data: List[TrainingExample],
        base_model: str = "BAAI/bge-base-en-v1.5",
        output_path: str = "fine_tuned_model",
        epochs: int = 3,
        batch_size: int = 16
    ):
        """Fine-tune embedding model with triplet loss."""
        # Load base model
        model = SentenceTransformer(base_model)

        # Convert to InputExamples
        train_examples = [
            InputExample(texts=[ex.anchor, ex.positive, ex.negative])
            for ex in training_data
            if ex.negative is not None
        ]

        # Create DataLoader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )

        # Use triplet loss
        train_loss = losses.TripletLoss(model=model)

        # Train
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path,
            show_progress_bar=True
        )

        return model
    ```

    **Step 4: Evaluate improvement**

    ```python
    def evaluate_fine_tuning(
        original_model,
        fine_tuned_model,
        eval_examples: List[dict],
        corpus: List[dict],
        k: int = 10
    ) -> dict:
        """Compare original vs fine-tuned model."""
        results = {'original': [], 'fine_tuned': []}

        for example in eval_examples:
            query = example['query']
            relevant_ids = set(example['relevant_doc_ids'])

            # Evaluate original
            orig_retrieved = retrieve_with_model(original_model, query, corpus, k)
            orig_recall = len(set(orig_retrieved) & relevant_ids) / len(relevant_ids)
            results['original'].append(orig_recall)

            # Evaluate fine-tuned
            ft_retrieved = retrieve_with_model(fine_tuned_model, query, corpus, k)
            ft_recall = len(set(ft_retrieved) & relevant_ids) / len(relevant_ids)
            results['fine_tuned'].append(ft_recall)

        return {
            'original_avg_recall': sum(results['original']) / len(results['original']),
            'fine_tuned_avg_recall': sum(results['fine_tuned']) / len(results['fine_tuned']),
            'improvement': (
                sum(results['fine_tuned']) / len(results['fine_tuned']) -
                sum(results['original']) / len(results['original'])
            )
        }
    ```

---

### Loss Functions

Understanding loss functions helps you choose the right training objective and debug training issues.

!!! tip "For Product Managers"
    **What loss functions do**: Loss functions measure how wrong the model is during training. Lower loss means the model is learning to put relevant documents closer to queries.

    **Key insight**: You do not need to understand the math, but knowing that different loss functions exist for different scenarios helps when discussing technical tradeoffs with your engineering team.

    | Loss Function | Best For | Data Requirement |
    |---------------|----------|------------------|
    | Triplet Loss | General fine-tuning | Triplets (anchor, positive, negative) |
    | InfoNCE | Large batch training | Pairs with in-batch negatives |
    | Multiple Negatives Ranking | Efficient training | Pairs (negatives from batch) |

!!! tip "For Engineers"
    **Triplet Loss**:

    Triplet loss ensures the anchor is closer to the positive than to the negative by at least a margin.

    $$\mathcal{L}_{\text{triplet}} = \max(0, \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + \alpha)$$

    Where:
    - $f(a)$ is the anchor embedding
    - $f(p)$ is the positive embedding
    - $f(n)$ is the negative embedding
    - $\alpha$ is the margin (typically 0.5-1.0)

    ```python
    import torch
    import torch.nn.functional as F

    def triplet_loss(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float = 1.0
    ) -> torch.Tensor:
        """Compute triplet loss."""
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + margin)
        return loss.mean()
    ```

    **InfoNCE Loss (Contrastive Loss)**:

    InfoNCE treats the problem as classification: given an anchor, identify the positive among many negatives.

    $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(a, p) / \tau)}{\sum_{i=1}^{N} \exp(\text{sim}(a, n_i) / \tau)}$$

    Where:
    - $\text{sim}(a, p)$ is similarity between anchor and positive
    - $\tau$ is temperature (typically 0.05-0.1)
    - $N$ is the number of negatives

    ```python
    def info_nce_loss(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """Compute InfoNCE loss."""
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
    ```

    **Multiple Negatives Ranking Loss**:

    This loss uses other examples in the batch as negatives, making training more efficient.

    ```python
    from sentence_transformers import losses

    # Using Sentence Transformers
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # This automatically uses other positives in the batch as negatives
    # Very efficient: no need to explicitly mine negatives
    ```

    **Choosing the right loss**:

    | Scenario | Recommended Loss |
    |----------|------------------|
    | Have explicit negatives | Triplet Loss |
    | Large batches, pairs only | Multiple Negatives Ranking |
    | Need fine-grained control | InfoNCE |
    | Limited compute | Multiple Negatives Ranking |

---

### Training Strategies

Practical considerations for training embedding models effectively.

!!! tip "For Product Managers"
    **Resource requirements**:

    | Dataset Size | Training Time | Hardware | Cost |
    |--------------|---------------|----------|------|
    | 1,000 examples | 10-15 min | Laptop GPU | ~$0.50 |
    | 6,000 examples | 30-45 min | Laptop GPU | ~$1.50 |
    | 50,000 examples | 2-4 hours | Cloud GPU | ~$10-20 |

    **Key decisions**:

    1. **When to start**: Wait until you have 1,000+ examples minimum, 6,000+ for best results
    2. **How often to retrain**: Monthly or when significant new data accumulates
    3. **Whether to re-embed**: Fine-tuning requires re-embedding your entire corpus

!!! tip "For Engineers"
    **Learning rate schedules**:

    ```python
    from sentence_transformers import SentenceTransformer
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

    def create_optimizer_and_scheduler(
        model: SentenceTransformer,
        num_training_steps: int,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1
    ):
        """Create optimizer with warmup and decay."""
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        warmup_steps = int(num_training_steps * warmup_ratio)

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Decay scheduler
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps
        )

        return optimizer, warmup_scheduler, decay_scheduler
    ```

    **Gradient accumulation for limited memory**:

    ```python
    def train_with_gradient_accumulation(
        model,
        dataloader,
        loss_fn,
        optimizer,
        accumulation_steps: int = 4
    ):
        """Train with gradient accumulation for larger effective batch size."""
        model.train()
        optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # Forward pass
            loss = loss_fn(batch) / accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    ```

    **Validation during training**:

    ```python
    def train_with_validation(
        model,
        train_dataloader,
        val_examples,
        corpus,
        epochs: int = 3,
        patience: int = 2
    ):
        """Train with early stopping based on validation recall."""
        best_recall = 0
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            model.train()
            for batch in train_dataloader:
                # ... training step ...
                pass

            # Validation
            model.eval()
            val_recall = evaluate_recall(model, val_examples, corpus)

            print(f"Epoch {epoch + 1}: Validation Recall@10 = {val_recall:.3f}")

            if val_recall > best_recall:
                best_recall = val_recall
                patience_counter = 0
                model.save("best_model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        return SentenceTransformer("best_model")
    ```

    **Quantization for deployment**:

    ```python
    from sentence_transformers import SentenceTransformer
    import torch

    def quantize_model(model_path: str, output_path: str):
        """Quantize model for faster inference."""
        model = SentenceTransformer(model_path)

        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        quantized_model.save(output_path)
        return quantized_model
    ```

---

## Case Study Deep Dive

### Case Study 1: Glean's Custom Embedding Models

Glean builds custom embedding models for every enterprise customer, achieving 20% search performance improvements.

!!! tip "For Product Managers"
    **Business context**: Glean provides enterprise AI search across applications like Google Drive, GitHub, Jira, and Confluence. Generic embeddings fail because enterprise data has company-specific language (project names, internal terminology) that general models do not understand.

    **Approach**:

    1. Start with a high-performance base model (BERT-based)
    2. Perform continued pre-training on company data
    3. Convert to embedding model through fine-tuning
    4. Continuously update as the company evolves

    **Results**: After six months of learning from user feedback, 20% improvement in search performance.

    **Key insight**: "When you're thinking about building really performant enterprise AI, you want to think about using smaller embedding models when you can, because small embedding models when fine-tuned to the domain can give you a lot better performance compared to just using large LLMs."

!!! tip "For Engineers"
    **Training data sources Glean uses**:

    - **Title-body pairs**: Document titles mapped to body passages
    - **Anchor data**: Documents that reference other documents create relevance pairs
    - **Co-access data**: Documents accessed together in short time periods
    - **Public datasets**: MS MARCO and similar high-quality datasets
    - **Synthetic data**: LLM-generated question-answer pairs

    **Application-specific intelligence**: For Slack data, they create "conversation documents" from threads, using the first message as title and rest as body. This understanding of application-specific nuances produces higher quality training data.

    **Feedback loop**:

    - Search product: Query-click pairs provide direct relevance signals
    - RAG assistant: Citation clicks, upvotes/downvotes, interaction patterns

### Case Study 2: Healthcare RAG Fine-Tuning

A healthcare company fine-tuned embeddings on medical abbreviations where generic models confused similar acronyms.

!!! tip "For Product Managers"
    **Business context**: Medical abbreviations like "MS" can mean multiple sclerosis or mitral stenosis. Generic embeddings cannot distinguish context-dependent meanings, leading to dangerous retrieval errors.

    **Results**:

    | Metric | Before | After |
    |--------|--------|-------|
    | Recall@10 | 72% | 89% |
    | Abbreviation confusion | High | Near-zero |
    | Training cost | - | $1.50 |
    | Training time | - | 45 minutes |

    **ROI**: Prevented multiple medical documentation errors that could have had serious consequences.

!!! tip "For Engineers"
    **Training approach**:

    1. Identified abbreviations with context-dependent meanings
    2. Created positives using abbreviation in correct context
    3. Created hard negatives using same abbreviation in different medical context
    4. Trained model to learn that context clues (symptoms, conditions, patient history) determine meaning

    ```python
    # Example training data for medical abbreviations
    training_examples = [
        Triplet(
            anchor="Patient presents with MS symptoms including fatigue and vision problems",
            positive="Multiple sclerosis diagnosis confirmed via MRI showing lesions",
            negative="Mitral stenosis detected during cardiac examination"
        ),
        Triplet(
            anchor="MS patient with cardiac history",
            positive="Mitral stenosis severity assessed via echocardiogram",
            negative="Multiple sclerosis treatment with disease-modifying therapy"
        ),
    ]
    ```

### Case Study 3: LanceDB Re-Ranker Benchmarks

Ayush from LanceDB demonstrated systematic improvements from re-ranker fine-tuning.

!!! tip "For Product Managers"
    **Results summary**:

    | Approach | Top-5 Improvement | Top-10 Improvement |
    |----------|-------------------|-------------------|
    | Off-the-shelf re-ranker | 10% | 6% |
    | Fine-tuned re-ranker | 12-14% | 8-10% |
    | Full-text search + re-ranker | Up to 20% | 15% |

    **Key insight**: Even a small re-ranker model (MiniLM, 6M parameters) provides significant improvements. Start small to validate the approach before investing in larger models.

!!! tip "For Engineers"
    **Benchmark methodology**:

    - Dataset: Google QA dataset (3M query-context pairs)
    - Training: 2M examples, evaluation: 5K examples
    - Base models tested: MiniLM (6M params), Modern BERT (150M params)
    - Architectures: Cross-encoder, ColBERT

    **ColBERT architecture advantage**: Late interaction model that calculates document embeddings offline but compares token-level embeddings at query time. Offers balance between bi-encoder speed and cross-encoder accuracy.

---

## Implementation Guide

### Quick Start for PMs

**Week 1: Assess Current State**

1. Review your evaluation data from Chapter 1
2. Identify how many examples you have (need 1,000+ for fine-tuning)
3. Assess whether your domain has specialized terminology
4. Calculate potential ROI: current recall vs expected improvement

**Week 2: Decide on Approach**

1. If <1,000 examples: Focus on data collection, try off-the-shelf re-ranker
2. If 1,000-6,000 examples: Start with re-ranker fine-tuning
3. If 6,000+ examples: Consider embedding fine-tuning

**Week 3: Implement and Measure**

1. Work with engineering to implement chosen approach
2. Run A/B test against baseline
3. Measure recall improvement
4. Calculate actual ROI

**Ongoing: Build the Data Flywheel**

- Design UX to capture relevance signals (clicks, citations, feedback)
- Log top 20-40 retrieved chunks per query
- Use LLM judges to mark relevance when human annotation is not practical
- Retrain models monthly or when significant data accumulates

### Detailed Implementation for Engineers

**Step 1: Prepare Your Data**

```python
import json
from pathlib import Path
from typing import List, Dict

def load_and_prepare_data(
    eval_data_path: str,
    corpus_path: str,
    output_path: str
) -> List[TrainingExample]:
    """Load evaluation data and prepare training triplets."""
    # Load data
    eval_data = json.loads(Path(eval_data_path).read_text())
    corpus = json.loads(Path(corpus_path).read_text())

    # Create corpus lookup
    corpus_lookup = {doc['id']: doc for doc in corpus}

    # Generate triplets
    training_examples = []

    for example in eval_data:
        query = example['query']
        relevant_ids = set(example['relevant_doc_ids'])

        # Get positives
        positives = [corpus_lookup[id] for id in relevant_ids if id in corpus_lookup]

        # Get hard negatives (documents retrieved but not relevant)
        if 'retrieved_ids' in example:
            hard_neg_ids = set(example['retrieved_ids']) - relevant_ids
            hard_negatives = [corpus_lookup[id] for id in hard_neg_ids if id in corpus_lookup]
        else:
            hard_negatives = []

        # Create triplets
        for positive in positives:
            for negative in hard_negatives[:3]:  # Limit negatives per positive
                training_examples.append(TrainingExample(
                    anchor=query,
                    positive=positive['text'],
                    negative=negative['text']
                ))

    # Save
    Path(output_path).write_text(json.dumps([
        {'anchor': e.anchor, 'positive': e.positive, 'negative': e.negative}
        for e in training_examples
    ], indent=2))

    return training_examples
```

**Step 2: Fine-Tune Embedding Model**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def fine_tune_embeddings(
    training_examples: List[TrainingExample],
    base_model: str = "BAAI/bge-base-en-v1.5",
    output_dir: str = "models/fine_tuned_embeddings",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """Fine-tune embedding model."""
    # Load model
    model = SentenceTransformer(base_model)

    # Prepare data
    train_examples = [
        InputExample(texts=[e.anchor, e.positive, e.negative])
        for e in training_examples
    ]

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )

    # Configure loss
    train_loss = losses.TripletLoss(model=model)

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=output_dir,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True
    )

    print(f"Model saved to {output_dir}")
    return model
```

**Step 3: Fine-Tune Re-Ranker (using Cohere)**

```python
import cohere

def fine_tune_cohere_reranker(
    training_data: List[Dict],
    model_name: str = "my-fine-tuned-reranker"
) -> str:
    """Fine-tune Cohere re-ranker."""
    co = cohere.Client()

    # Format data for Cohere
    # Each example needs: query, relevant_passages, hard_negatives
    formatted_data = []
    for example in training_data:
        formatted_data.append({
            "query": example['anchor'],
            "relevant_passages": [example['positive']],
            "hard_negatives": [example['negative']] if example['negative'] else []
        })

    # Create fine-tuning job
    response = co.finetuning.create_finetuned_model(
        request={
            "name": model_name,
            "settings": {
                "base_model": {"base_type": "BASE_TYPE_RERANK"},
                "dataset": {"reranker_dataset": {"data": formatted_data}}
            }
        }
    )

    return response.finetuned_model.id
```

**Step 4: Evaluate and Compare**

```python
def compare_models(
    original_retriever,
    fine_tuned_retriever,
    reranker,
    eval_examples: List[Dict],
    k: int = 10
) -> Dict:
    """Compare original, fine-tuned, and re-ranked retrieval."""
    results = {
        'original': [],
        'fine_tuned': [],
        'fine_tuned_reranked': []
    }

    for example in eval_examples:
        query = example['query']
        relevant_ids = set(example['relevant_doc_ids'])

        # Original
        orig_results = original_retriever.search(query, top_k=k)
        orig_recall = len(set(r['id'] for r in orig_results) & relevant_ids) / len(relevant_ids)
        results['original'].append(orig_recall)

        # Fine-tuned
        ft_results = fine_tuned_retriever.search(query, top_k=k)
        ft_recall = len(set(r['id'] for r in ft_results) & relevant_ids) / len(relevant_ids)
        results['fine_tuned'].append(ft_recall)

        # Fine-tuned + Re-ranked
        ft_candidates = fine_tuned_retriever.search(query, top_k=k*3)
        reranked = reranker.rerank(query, ft_candidates, top_k=k)
        rr_recall = len(set(r['id'] for r in reranked) & relevant_ids) / len(relevant_ids)
        results['fine_tuned_reranked'].append(rr_recall)

    return {
        'original_recall': sum(results['original']) / len(results['original']),
        'fine_tuned_recall': sum(results['fine_tuned']) / len(results['fine_tuned']),
        'reranked_recall': sum(results['fine_tuned_reranked']) / len(results['fine_tuned_reranked']),
    }
```

---

## Common Pitfalls

### PM Pitfalls

!!! warning "PM Pitfall: Fine-Tuning Too Early"
    **The mistake**: Investing in fine-tuning before having enough data or establishing baselines.

    **Why it happens**: Fine-tuning sounds sophisticated and promising.

    **The consequence**: Wasted engineering time, unclear results, no way to measure improvement.

    **How to avoid**: Wait until you have 1,000+ examples minimum. Establish baseline metrics first. Try off-the-shelf re-rankers before custom fine-tuning.

!!! warning "PM Pitfall: Ignoring Data Collection"
    **The mistake**: Not logging relevance signals from day one.

    **Why it happens**: Data collection feels like overhead before you need it.

    **The consequence**: When you are ready to fine-tune, you have no data. Teams wait 3-6 months to collect enough.

    **How to avoid**: Start logging now. Save top 20-40 chunks per query. Use LLM judges to mark relevance. Design UX to capture feedback.

!!! warning "PM Pitfall: Underestimating Re-Ranker ROI"
    **The mistake**: Skipping re-rankers because they add latency.

    **Why it happens**: Latency is visible and measurable. Recall improvement is harder to see.

    **The consequence**: Missing 10-20% recall improvement for 30-300ms latency cost.

    **How to avoid**: Run the experiment. Measure actual recall improvement. Calculate business value of better results vs latency cost.

### Engineering Pitfalls

!!! warning "Engineering Pitfall: Easy Negatives"
    **The mistake**: Using random documents as negatives instead of hard negatives.

    **Why it happens**: Hard negative mining requires extra work.

    **The consequence**: Model learns nothing useful. Training with only positives gives 6% improvement. Hard negatives give 30%.

    **How to avoid**: Mine hard negatives using embedding similarity. Use retrieved-but-not-cited documents. Track user deletion signals.

!!! warning "Engineering Pitfall: Catastrophic Forgetting"
    **The mistake**: Fine-tuning too long on domain-specific data, losing general capabilities.

    **Why it happens**: More training seems like it should help.

    **The consequence**: Model performs worse than baseline on general queries.

    **How to avoid**: Use validation sets. Implement early stopping. Mix domain-specific data with general data. Monitor performance on diverse query types.

!!! warning "Engineering Pitfall: Ignoring Evaluation During Training"
    **The mistake**: Training for fixed epochs without monitoring validation metrics.

    **Why it happens**: Seems simpler to just train and evaluate at the end.

    **The consequence**: Overfitting, wasted compute, suboptimal models.

    **How to avoid**: Evaluate on validation set every epoch. Implement early stopping. Save best model, not final model.

!!! warning "Engineering Pitfall: Re-Embedding Blindness"
    **The mistake**: Fine-tuning embeddings without planning for re-embedding the corpus.

    **Why it happens**: Focus on model training, not deployment.

    **The consequence**: Fine-tuned model sits unused because re-embedding millions of documents is expensive.

    **How to avoid**: Plan re-embedding before fine-tuning. Consider re-rankers first (no re-embedding needed). Budget for re-embedding compute.

---

## Related Content

### Transcript

The full lecture transcript is available at `docs/workshops/chapter2-transcript.txt`. Key insights from the lecture:

- "If you're not fine-tuning, you're Blockbuster, not Netflix."
- "The goal isn't to fine-tune a language model—those are pretty hard and expensive. The goal is to train and fine-tune embedding models."
- "With just 6,000 examples, we will ultimately be able to do six or ten percent better."
- "Everything we're doing today with language models is what I used to have to pay data labeling teams hundreds of thousands of dollars to do every year."

### Talk: Glean's Custom Embedding Models (Manav)

Full talk available at `docs/talks/glean-manav.md`. Key insights:

- **20% improvement**: Custom embedding models achieve 20% search performance improvement after six months
- **Smaller models win**: "Small embedding models when fine-tuned to the domain can give you a lot better performance compared to just using large LLMs"
- **Training data sources**: Title-body pairs, anchor data, co-access signals, synthetic data
- **60-70% rule**: For 60-70% of enterprise queries, basic lexical search with recency signals works well

### Talk: Re-Rankers and Fine-Tuning (Ayush, LanceDB)

Full talk available at `docs/talks/fine-tuning-rerankers-embeddings-ayush-lancedb.md`. Key insights:

- **12% improvement at top-5**: Re-rankers consistently improve retrieval by 10-20%
- **Hard negatives matter**: Training with hard negatives provides 5x better results than easy negatives
- **Start small**: Even MiniLM (6M parameters) provides significant improvements
- **Combine approaches**: Fine-tuned embeddings + re-ranker yields best results

### Office Hours

Relevant office hours sessions:

- **Cohort 2 Week 2** (`docs/office-hours/cohort2/week2-summary.md`): Fine-tuning vs prompting decisions, synthetic data distribution mismatch, embedding vs re-ranker fine-tuning
- **Cohort 3 Week 2** (`docs/office-hours/cohort3/week-2-1.md`): Medical RAG systems, citation handling, graph-based vs traditional RAG

---

## Action Items

### For Product Teams

1. **This week**: Audit your current data collection—are you logging relevance signals?
2. **This week**: Count your evaluation examples—do you have 1,000+ for fine-tuning?
3. **This month**: Design UX changes to capture more relevance signals (document-level feedback, click tracking)
4. **This month**: Try an off-the-shelf re-ranker (Cohere Rerank) and measure improvement
5. **This quarter**: If you have 6,000+ examples, plan embedding fine-tuning project
6. **Ongoing**: Review monthly whether data volume justifies fine-tuning investment

### For Engineering Teams

1. **This week**: Implement relevance logging (top 20-40 chunks per query)
2. **This week**: Add Cohere Rerank to your pipeline and measure recall improvement
3. **This month**: Convert evaluation data to training triplets with hard negatives
4. **This month**: Run fine-tuning experiment on small model (MiniLM) to validate approach
5. **This quarter**: If validation succeeds, fine-tune production embedding model
6. **This quarter**: Set up automated retraining pipeline for monthly updates
7. **Ongoing**: Monitor for catastrophic forgetting, maintain validation sets

---

## Reflection Questions

1. What definition of "similarity" is most important for your application? How does it differ from generic text similarity?

2. Where could you collect hard negatives in your current system? What user actions indicate "retrieved but not relevant"?

3. If you had to choose between fine-tuning embeddings or adding a re-ranker, which would provide more value for your use case? Why?

4. How would you design your UX to capture more relevance signals without disrupting user experience?

5. What is the cost of a retrieval failure in your application? How does that compare to the cost of fine-tuning?

---

## Summary

### Key Takeaways for Product Managers

- **Fine-tuning is accessible**: 6,000 examples, 40 minutes, $1.50. This is not the expensive ML project it used to be.
- **Start with re-rankers**: They provide 10-20% improvement without re-embedding your corpus. Low risk, high reward.
- **Data collection is the bottleneck**: Start logging relevance signals now. Teams that wait lose 3-6 months.
- **Generic embeddings have hidden assumptions**: Provider models define similarity in ways that may not match your domain.
- **Hard negatives multiply ROI**: 5x better results from training with hard negatives vs easy negatives.

### Key Takeaways for Engineers

- **Bi-encoders for speed, cross-encoders for accuracy**: Use both in a two-stage retrieval system.
- **Contrastive learning with triplets**: Anchor, positive, negative. Pull similar things closer, push dissimilar things apart.
- **Hard negative mining is critical**: Use embedding similarity to find documents that are similar but wrong.
- **Loss functions matter**: Triplet loss for explicit negatives, Multiple Negatives Ranking for efficient batch training.
- **Validate during training**: Early stopping, validation sets, and monitoring prevent overfitting and catastrophic forgetting.

---

## Further Reading

### Academic Papers

- "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers & Gurevych, 2019) - Foundation for sentence transformers
- "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction" (Khattab & Zaharia, 2020) - Late interaction architecture
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020) - DPR approach
- "Contrastive Learning for Neural Text Generation" (An et al., 2022) - Contrastive learning techniques

### Tools and Libraries

- **Sentence Transformers**: [sbert.net](https://www.sbert.net/) - Fine-tuning embedding models
- **Cohere Rerank**: [cohere.com/rerank](https://cohere.com/rerank) - Production re-ranking API with fine-tuning
- **LanceDB**: [lancedb.com](https://lancedb.com/) - Vector database with built-in re-ranking
- **Modal**: [modal.com](https://modal.com/) - Parallel training and embedding at scale

### Related Appendices

- **Appendix A: Mathematical Foundations** - Full derivations of loss functions
- **Appendix B: Algorithms Reference** - Training algorithms and complexity analysis

---

## Navigation

- **Previous**: [Chapter 1: Evaluation-First Development](chapter1.md) - Synthetic data and evaluation frameworks
- **Next**: [Chapter 3: Feedback Systems and UX](chapter3.md) - Collecting user feedback and building fast interfaces
- **Reference**: [Glossary](glossary.md) | [Quick Reference](quick-reference.md)
- **Book Index**: [Book 1: Foundations](book1-index.md)
