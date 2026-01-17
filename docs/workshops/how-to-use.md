---
title: How to Use This Book
description: Reading paths, prerequisites, and guidance for getting the most from these workshops
authors:
  - Jason Liu
date: 2025-04-18
tags:
  - guide
  - getting-started
---

# How to Use This Book

This guide helps you navigate the workshops based on your goals, experience level, and available time.

---

## Three Reading Paths

### Path 1: The Full Journey (Recommended)

**Time**: 8-12 hours of reading + 10-20 hours of hands-on practice

**For**: Teams building new RAG systems or significantly improving existing ones

Read chapters in order from Introduction through Chapter 7. Each chapter builds on the previous one, and the concepts compound. The construction company case study threads through multiple chapters, showing how the same system evolves.

```
Introduction → Ch 1 → Ch 2 → Ch 3.1 → Ch 3.2 → Ch 3.3 → Ch 4.1 → Ch 4.2 → Ch 5.1 → Ch 5.2 → Ch 6.1 → Ch 6.2 → Ch 6.3 → Ch 7
```

### Path 2: Quick Wins First

**Time**: 3-4 hours of reading + 5-10 hours of implementation

**For**: Teams with existing RAG systems that need immediate improvements

Start with the chapters that typically deliver the fastest results:

1. **[Chapter 1](chapter1.md)**: Set up evaluation (you cannot improve what you cannot measure)
2. **[Chapter 3.1](chapter3-1.md)**: Fix feedback collection (often 5x improvement with copy changes)
3. **[Chapter 2](chapter2.md)**: Add re-ranking (12-20% retrieval improvement)
4. **[Chapter 4.1](chapter4-1.md)**: Identify your worst-performing query segments

Then return to fill gaps as needed.

### Path 3: Reference Mode

**Time**: As needed

**For**: Experienced practitioners looking for specific techniques

Jump directly to what you need:

- **Evaluation setup**: [Chapter 1](chapter1.md)
- **Fine-tuning embeddings**: [Chapter 2](chapter2.md)
- **Feedback collection**: [Chapter 3.1](chapter3-1.md)
- **Streaming/latency**: [Chapter 3.2](chapter3-2.md)
- **Query clustering**: [Chapter 4.1](chapter4-1.md)
- **Prioritization**: [Chapter 4.2](chapter4-2.md)
- **Multimodal retrieval**: [Chapter 5.2](chapter5-2.md)
- **Query routing**: [Chapter 6.1](chapter6-1.md), [Chapter 6.2](chapter6-2.md)
- **Production operations**: [Chapter 7](chapter7.md)

Use the [Glossary](glossary.md) for term definitions and [Quick Reference](quick-reference.md) for formulas and decision trees.

---

## Prerequisites by Chapter

| Chapter | What You Should Know |
|---------|---------------------|
| **Introduction** | What RAG is at a high level |
| **Chapter 1** | Basic Python, familiarity with embeddings |
| **Chapter 2** | Chapter 1 concepts, basic ML training concepts |
| **Chapter 3.1-3.3** | Web development basics (for UI patterns) |
| **Chapter 4.1-4.2** | Chapter 1 concepts, basic statistics |
| **Chapter 5.1-5.2** | Chapters 1-2, understanding of different data types |
| **Chapter 6.1-6.3** | Chapters 1-5, API design concepts |
| **Chapter 7** | All previous chapters, basic DevOps/infrastructure |

---

## Time Estimates

| Chapter | Reading | Hands-on Practice |
|---------|---------|-------------------|
| Introduction | 30 min | - |
| Chapter 1 | 45 min | 2-3 hours |
| Chapter 2 | 45 min | 3-4 hours |
| Chapter 3.1 | 30 min | 1-2 hours |
| Chapter 3.2 | 30 min | 2-3 hours |
| Chapter 3.3 | 30 min | 1-2 hours |
| Chapter 4.1 | 45 min | 2-3 hours |
| Chapter 4.2 | 30 min | 1-2 hours |
| Chapter 5.1 | 30 min | 1-2 hours |
| Chapter 5.2 | 45 min | 3-4 hours |
| Chapter 6.1 | 30 min | 1-2 hours |
| Chapter 6.2 | 45 min | 2-3 hours |
| Chapter 6.3 | 30 min | 1-2 hours |
| Chapter 7 | 45 min | 2-3 hours |
| **Total** | **~8 hours** | **~25 hours** |

---

## What You Will Build

By the end of the full journey, you will have:

1. **An evaluation framework** with synthetic data and retrieval metrics
2. **A feedback collection system** that gathers 5x more data than typical implementations
3. **Fine-tuned embeddings or re-rankers** tailored to your domain
4. **Query segmentation** showing which user needs are underserved
5. **Specialized retrievers** for different content types
6. **A routing layer** that directs queries to the right tools
7. **Production monitoring** that catches degradation before users notice

---

## Hands-On Practice

Each chapter includes:

- **Action Items**: Specific tasks to implement that week
- **Reflection Questions**: Prompts to apply concepts to your system
- **Code Examples**: Patterns you can adapt

For deeper hands-on practice, the [WildChat Case Study](../../latest/case_study/README.md) walks through a complete RAG improvement cycle with real data:

| Case Study Part | Related Workshop Chapter |
|-----------------|-------------------------|
| Part 1: Data Exploration | Chapter 1 |
| Part 2: The Alignment Problem | Chapter 2, Chapter 5 |
| Part 3: Solving with Summaries | Chapter 5 |
| Part 4: Advanced Techniques | Chapter 2, Chapter 6 |

---

## Common Questions

### "I already have a RAG system. Where do I start?"

Start with [Chapter 1](chapter1.md) to establish evaluation metrics. You cannot improve what you cannot measure. Even if your system is "working," you need baselines to know if changes help or hurt.

### "I do not have any users yet. Is this relevant?"

Yes. [Chapter 1](chapter1.md) specifically addresses the cold-start problem using synthetic data. You can build evaluation infrastructure and test improvements before launch.

### "My team is skeptical about investing time in evaluation."

Show them the $100M company example from [Chapter 1](chapter1.md)—companies with massive valuations operating with fewer than 30 evaluations. Then show the construction company case study: systematic evaluation led to 27% → 85% recall improvement in four days.

### "We are using [specific vector database/LLM]. Does this apply?"

Yes. The concepts are tool-agnostic. Specific code examples use common tools (OpenAI, LanceDB, ChromaDB), but the frameworks apply regardless of your stack.

### "How do I convince my manager to let me work on this?"

Frame it in business terms:
- Evaluation prevents shipping regressions (risk reduction)
- Feedback collection generates training data (asset building)
- Query segmentation reveals product opportunities (revenue potential)
- The construction company reduced unit costs from $0.09 to $0.04 per query (cost savings)

---

## Getting Help

- **Glossary**: [Key terms and definitions](glossary.md)
- **Quick Reference**: [Formulas and decision trees](quick-reference.md)
- **Chapter Index**: [Full workshop listing](index.md)

---

## Suggested Weekly Schedule

For teams working through the material together:

| Week | Focus | Chapters |
|------|-------|----------|
| 1 | Foundations | Introduction, Chapter 1 |
| 2 | Improvement Techniques | Chapter 2 |
| 3 | User Experience | Chapters 3.1, 3.2, 3.3 |
| 4 | User Understanding | Chapters 4.1, 4.2 |
| 5 | Specialized Retrieval | Chapters 5.1, 5.2 |
| 6 | System Architecture | Chapters 6.1, 6.2, 6.3 |
| 7 | Production | Chapter 7 |

Each week: Read the chapters, implement the action items, discuss reflection questions as a team.

---

*Ready to start? Begin with the [Introduction](chapter0.md) or jump to [Chapter 1](chapter1.md) if you are already familiar with the product mindset.*
