---
title: "Systematically Improving RAG Applications"
description: "A comprehensive technical reference for building and improving Retrieval-Augmented Generation systems, with separate guidance for Product Managers and Engineers."
authors:
  - Jason Liu
date: 2025-01-18
tags:
  - RAG
  - retrieval
  - LLM
  - product management
  - engineering
---

# Systematically Improving RAG Applications

A comprehensive technical reference for building and improving Retrieval-Augmented Generation systems.

---

## About This Book

This book teaches a data-driven approach to building RAG systems that get better over time. Unlike tutorials that show you how to build a RAG system once, this book shows you how to build systems that improve continuously based on real user behavior.

The content is designed for two audiences:

- **Product Managers** who need to understand RAG capabilities, make strategic decisions, and measure success
- **Engineers** who need to implement, optimize, and maintain RAG systems in production

Throughout the book, content is clearly marked for each audience using admonitions. You can read the full book or focus on the sections most relevant to your role.

!!! note "Book vs Workshops"
This book is a different draft from the workshops. While the workshops came directly from the course lectures, this book synthesizes content from workshops, transcripts, talks, and office hours into a comprehensive technical reference organized for both Product Managers and Engineers.

---

## Book Structure

The book is organized into four parts, plus appendices and supporting materials.

### Book 1: Foundations

Build the mental models and infrastructure for continuous improvement.

| Chapter                  | Title                              | Description                                                              |
| ------------------------ | ---------------------------------- | ------------------------------------------------------------------------ |
| [Chapter 0](chapter0.md) | Introduction - The Product Mindset | Foundational concepts, the improvement flywheel, common failure patterns |
| [Chapter 1](chapter1.md) | Evaluation-First Development       | Synthetic data, precision/recall, statistical significance               |
| [Chapter 2](chapter2.md) | Training Data and Fine-Tuning      | Embeddings, re-rankers, contrastive learning, loss functions             |

### Book 2: User-Centric Design

Understand and serve your users better.

| Chapter                  | Title                                  | Description                                                  |
| ------------------------ | -------------------------------------- | ------------------------------------------------------------ |
| [Chapter 3](chapter3.md) | Feedback Systems and UX                | Feedback collection, streaming, citations, perceived latency |
| [Chapter 4](chapter4.md) | Query Understanding and Prioritization | Query clustering, topic modeling, economic value analysis    |

### Book 3: Architecture and Production

Build robust systems that scale.

| Chapter                  | Title                           | Description                                              |
| ------------------------ | ------------------------------- | -------------------------------------------------------- |
| [Chapter 5](chapter5.md) | Specialized Retrieval Systems   | Metadata extraction, RAPTOR, multimodal retrieval        |
| [Chapter 6](chapter6.md) | Query Routing and Orchestration | Router architectures, tool interfaces, latency analysis  |
| [Chapter 7](chapter7.md) | Production Operations           | Semantic caching, monitoring, cost optimization, scaling |

### Book 4: Advanced Topics

Techniques for complex scenarios.

| Chapter                  | Title                     | Description                                                   |
| ------------------------ | ------------------------- | ------------------------------------------------------------- |
| [Chapter 8](chapter8.md) | Hybrid Search             | Lexical search, BM25, Reciprocal Rank Fusion                  |
| [Chapter 9](chapter9.md) | Context Window Management | Lost in the middle, token budgeting, dynamic context assembly |

---

## Appendices

Technical reference materials for deeper dives.

| Appendix                             | Title                        | Description                                            |
| ------------------------------------ | ---------------------------- | ------------------------------------------------------ |
| [Appendix A](appendix-math.md)       | Mathematical Foundations     | Retrieval metrics, statistical testing, loss functions |
| [Appendix B](appendix-algorithms.md) | Algorithms Reference         | RAPTOR, clustering, router selection algorithms        |
| [Appendix C](appendix-benchmarks.md) | Benchmarking Your RAG System | Standard datasets, methodology, running benchmarks     |
| [Appendix D](appendix-debugging.md)  | Debugging RAG Systems        | Systematic methodology, failure modes, debugging tools |

---

## Supporting Materials

| Resource                              | Description                                    |
| ------------------------------------- | ---------------------------------------------- |
| [How to Use This Book](how-to-use.md) | Reading paths, prerequisites, navigation guide |
| [Glossary](glossary.md)               | Key terms and definitions                      |
| [Quick Reference](quick-reference.md) | Formulas, decision trees, checklists           |

---

## Case Studies

Real-world examples that thread through the book.

| Case Study                                         | Description                                              |
| -------------------------------------------------- | -------------------------------------------------------- |
| [Construction Company](case-study-construction.md) | Blueprint search system evolution from 27% to 85% recall |
| [Voice AI](case-study-voice-ai.md)                 | Restaurant voice assistant with real-time requirements   |
| [WildChat](case-study-wildchat.md)                 | Analysis of 1M+ real conversations                       |

---

## Reading Paths

### For Product Managers

Focus on business value, decision frameworks, and success metrics.

**Quick Start** (4-6 hours):

1. [Chapter 0](chapter0.md) - Understand the product mindset
2. [Chapter 1](chapter1.md) - Learn why evaluation comes first
3. [Chapter 3](chapter3.md) - Design feedback systems
4. [Chapter 4](chapter4.md) - Prioritize improvements

**Full Journey**: Read all chapters, focusing on "For Product Managers" sections.

### For Engineers

Focus on implementation details, code examples, and technical tradeoffs.

**Quick Start** (6-8 hours):

1. [Chapter 0](chapter0.md) - Build foundational intuition
2. [Chapter 1](chapter1.md) - Set up evaluation infrastructure
3. [Chapter 2](chapter2.md) - Implement fine-tuning
4. [Chapter 7](chapter7.md) - Production operations

**Full Journey**: Read all chapters, focusing on "For Engineers" sections.

### Full Journey

Read chapters in order for the complete picture. Each chapter builds on previous concepts.

---

## How Content Is Organized

Throughout the book, content is marked for specific audiences:

!!! tip "For Product Managers"
Business context, decision frameworks, ROI analysis, success metrics.

!!! tip "For Engineers"
Implementation details, code examples, algorithms, technical tradeoffs.

!!! warning "PM Pitfall"
Strategic mistakes to avoid.

!!! warning "Engineering Pitfall"
Technical mistakes to avoid.

!!! example "Example"
Concrete examples and case studies.

!!! info "Info"
General information and context.

---

## Getting Started

**New to RAG?** Start with [Chapter 0: Introduction](chapter0.md) to build foundational understanding.

**Have an existing system?** Start with [Chapter 1: Evaluation-First Development](chapter1.md) to establish baselines.

**Looking for something specific?** Use the [Quick Reference](quick-reference.md) or [Glossary](glossary.md).

---

_Ready to begin? Start with [Chapter 0: Introduction - The Product Mindset for RAG](chapter0.md)._
