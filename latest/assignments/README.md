# Coding Assignments

This directory contains hands-on coding assignments for each week of the course. Each assignment reinforces key RAG concepts through practical implementation.

## Assignment Structure

Each assignment includes:

- **Documentation** (`.md` files): Learning goals, setup, requirements, deliverables
- **Working Code** (`.py` files): Tested, runnable implementations

## Quick Start

Run any assignment with:

```bash
cd /path/to/systematically-improving-rag
uv run python latest/assignments/week1/metrics.py
```

## Assignments by Week

| Week | Documentation | Code | Focus Area |
|------|---------------|------|------------|
| 0 | [RAG Metrics Dashboard](week0_assignment.md) | [rag_pipeline.py](week0/rag_pipeline.py) | Logging, dashboards, ChromaDB |
| 1 | [Retrieval Evaluation](week1_assignment.md) | [metrics.py](week1/metrics.py), [evaluation_pipeline.py](week1/evaluation_pipeline.py) | Precision, recall, MRR, NDCG |
| 2 | [Fine-tune Embeddings](week2_assignment.md) | [fine_tuning.py](week2/fine_tuning.py) | Triplet loss, hard negatives |
| 3 | [Streaming RAG](week3_assignment.md) | [streaming.py](week3/streaming.py) | SSE, citations, validation |
| 4 | [Query Clustering](week4_assignment.md) | [clustering.py](week4/clustering.py) | K-means, UMAP, prioritization |
| 5 | [Multimodal Search](week5_assignment.md) | [multimodal.py](week5/multimodal.py) | Tables, images, rich descriptions |
| 6 | [Tool Routing](week6_assignment.md) | [router.py](week6/router.py) | OpenAI tool calling, few-shot |
| 7 | [Production RAG](week7_assignment.md) | [caching.py](week7/caching.py) | Multi-level caching, cost tracking |
| Capstone | [End-to-End System](capstone_assignment.md) | [system.py](capstone/system.py) | Full RAG flywheel |

## Code Overview

### Week 1: Evaluation Metrics
- `metrics.py`: Precision@k, Recall@k, MRR, NDCG implementations
- `evaluation_pipeline.py`: Full evaluation pipeline with ChromaDB

### Week 2: Fine-tuning
- `fine_tuning.py`: Hard negative mining, triplet creation, evaluation

### Week 3: Streaming
- `streaming.py`: SSE streaming, citation tracking, response validation

### Week 4: Query Analysis
- `clustering.py`: K-means clustering, UMAP visualization, prioritization matrix

### Week 6: Query Routing
- `router.py`: OpenAI function calling, dynamic example selection

### Week 7: Production
- `caching.py`: Multi-level cache (memory/Redis/semantic), cost tracking

### Capstone
- `system.py`: Complete RAG system with improvement flywheel

## Recommended Datasets

These public datasets are used across assignments:

- **SQuAD 2.0**: Question answering on Wikipedia (`rajpurkar/squad_v2`)
- **MS MARCO**: Web search queries and passages (`microsoft/ms_marco`)
- **HotpotQA**: Multi-hop reasoning questions (`hotpot_qa`)
- **Natural Questions**: Real Google search queries (`google-research-datasets/natural_questions`)
- **COCO**: Image captioning dataset (`HuggingFaceM4/COCO`)

## Getting Started

1. Ensure you have the course environment set up (see `latest/README.md`)
2. Start with Week 0 if you're new to RAG evaluation
3. Complete assignments in order - they build on each other
4. Use the weekly notebooks as reference implementations
