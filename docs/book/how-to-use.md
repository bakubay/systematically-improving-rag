---
title: How to Use This Book
description: Reading paths, prerequisites, and guidance for getting the most from this technical reference
authors:
  - Jason Liu
date: 2025-01-18
tags:
  - guide
  - getting-started
---

# How to Use This Book

This guide helps you navigate the book based on your goals, experience level, and role.

---

## Two Audiences, One Book

This book is designed for two audiences:

- **Product Managers** who need to understand RAG capabilities, make strategic decisions, and measure success
- **Engineers** who need to implement, optimize, and maintain RAG systems in production

Throughout the book, content is clearly marked for each audience using admonitions. You can read the full book or focus on the sections most relevant to your role.

---

## Content Markers

### Audience-Specific Content

!!! tip "For Product Managers"
    Sections marked like this contain business context, decision frameworks, ROI analysis, and success metrics. Focus here if you need to understand the "why" and "what" rather than the "how."

!!! tip "For Engineers"
    Sections marked like this contain implementation details, code examples, algorithms, and technical tradeoffs. Focus here if you need to build and maintain RAG systems.

### Warnings and Pitfalls

!!! warning "PM Pitfall"
    Strategic mistakes to avoid. These highlight common errors in planning, resource allocation, and decision-making.

!!! warning "Engineering Pitfall"
    Technical mistakes to avoid. These highlight common implementation errors, performance issues, and architectural problems.

### Other Markers

!!! example "Example"
    Concrete examples and case studies that illustrate concepts.

!!! info "Info"
    General information, context, and background.

!!! success "Success Story"
    Real-world success stories and outcomes.

---

## Three Reading Paths

### Path 1: The Full Journey (Recommended)

**Time**: 10-15 hours of reading + 20-30 hours of hands-on practice

**For**: Teams building new RAG systems or significantly improving existing ones

Read chapters in order from Chapter 0 through Chapter 9. Each chapter builds on the previous one, and the concepts compound. The construction company case study threads through multiple chapters, showing how the same system evolves.

```
Chapter 0 → Chapter 1 → Chapter 2 → Chapter 3 → Chapter 4 → Chapter 5 → Chapter 6 → Chapter 7 → Chapter 8 → Chapter 9
```

### Path 2: Quick Wins First

**Time**: 4-6 hours of reading + 10-15 hours of implementation

**For**: Teams with existing RAG systems that need immediate improvements

Start with the chapters that typically deliver the fastest results:

1. **[Chapter 1](chapter1.md)**: Set up evaluation (you cannot improve what you cannot measure)
2. **[Chapter 3](chapter3.md)**: Fix feedback collection (often 5x improvement with copy changes)
3. **[Chapter 2](chapter2.md)**: Add re-ranking (12-20% retrieval improvement)
4. **[Chapter 4](chapter4.md)**: Identify your worst-performing query segments

Then return to fill gaps as needed.

### Path 3: Reference Mode

**Time**: As needed

**For**: Experienced practitioners looking for specific techniques

Jump directly to what you need:

- **Evaluation setup**: [Chapter 1](chapter1.md)
- **Fine-tuning embeddings**: [Chapter 2](chapter2.md)
- **Feedback collection**: [Chapter 3](chapter3.md)
- **Query clustering**: [Chapter 4](chapter4.md)
- **Specialized retrieval**: [Chapter 5](chapter5.md)
- **Query routing**: [Chapter 6](chapter6.md)
- **Production operations**: [Chapter 7](chapter7.md)
- **Hybrid search**: [Chapter 8](chapter8.md)
- **Context management**: [Chapter 9](chapter9.md)

Use the [Glossary](glossary.md) for term definitions and [Quick Reference](quick-reference.md) for formulas and decision trees.

---

## Prerequisites by Chapter

| Chapter | What You Should Know |
|---------|---------------------|
| **Chapter 0** | What RAG is at a high level |
| **Chapter 1** | Basic Python, familiarity with embeddings |
| **Chapter 2** | Chapter 1 concepts, basic ML training concepts |
| **Chapter 3** | Web development basics (for UI patterns) |
| **Chapter 4** | Chapter 1 concepts, basic statistics |
| **Chapter 5** | Chapters 1-2, understanding of different data types |
| **Chapter 6** | Chapters 1-5, API design concepts |
| **Chapter 7** | All previous chapters, basic DevOps/infrastructure |
| **Chapter 8** | Chapter 1, understanding of search fundamentals |
| **Chapter 9** | Chapters 1-2, understanding of LLM context limits |

---

## Time Estimates

| Chapter | Reading | Hands-on Practice |
|---------|---------|-------------------|
| Chapter 0 | 45 min | - |
| Chapter 1 | 60 min | 3-4 hours |
| Chapter 2 | 60 min | 4-5 hours |
| Chapter 3 | 45 min | 2-3 hours |
| Chapter 4 | 45 min | 2-3 hours |
| Chapter 5 | 60 min | 3-4 hours |
| Chapter 6 | 60 min | 3-4 hours |
| Chapter 7 | 45 min | 2-3 hours |
| Chapter 8 | 45 min | 2-3 hours |
| Chapter 9 | 45 min | 2-3 hours |
| **Total** | **~9 hours** | **~28 hours** |

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
8. **Hybrid search** that combines semantic and lexical approaches
9. **Context management** that optimizes token usage

---

## Hands-On Practice

Each chapter includes:

- **Action Items**: Specific tasks to implement
- **Reflection Questions**: Prompts to apply concepts to your system
- **Code Examples**: Patterns you can adapt

For deeper hands-on practice, the case studies walk through complete RAG improvement cycles:

| Case Study | Related Chapters |
|------------|------------------|
| [Construction Company](case-study-construction.md) | Chapters 1, 4, 5, 6, 7 |
| [Voice AI](case-study-voice-ai.md) | Chapters 3, 4, 6 |
| [WildChat](case-study-wildchat.md) | Chapters 1, 2, 4 |

---

## Common Questions

### "I already have a RAG system. Where do I start?"

Start with [Chapter 1](chapter1.md) to establish evaluation metrics. You cannot improve what you cannot measure. Even if your system is "working," you need baselines to know if changes help or hurt.

### "I do not have any users yet. Is this relevant?"

Yes. [Chapter 1](chapter1.md) specifically addresses the cold-start problem using synthetic data. You can build evaluation infrastructure and test improvements before launch.

### "My team is skeptical about investing time in evaluation."

Show them the case studies. The construction company case study demonstrates how systematic evaluation led to 27% to 85% recall improvement in four days. Frame it in business terms: evaluation prevents shipping regressions (risk reduction) and generates training data (asset building).

### "We are using [specific vector database/LLM]. Does this apply?"

Yes. The concepts are tool-agnostic. Specific code examples use common tools (OpenAI, LanceDB), but the frameworks apply regardless of your stack.

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
- **Appendix A**: [Mathematical Foundations](appendix-math.md)
- **Appendix D**: [Debugging RAG Systems](appendix-debugging.md)

---

## Suggested Weekly Schedule

For teams working through the material together:

| Week | Focus | Chapters |
|------|-------|----------|
| 1 | Foundations | Chapter 0, Chapter 1 |
| 2 | Improvement Techniques | Chapter 2 |
| 3 | User Experience | Chapter 3 |
| 4 | User Understanding | Chapter 4 |
| 5 | Specialized Retrieval | Chapter 5 |
| 6 | System Architecture | Chapter 6 |
| 7 | Production | Chapter 7 |
| 8 | Advanced Topics | Chapter 8, Chapter 9 |

Each week: Read the chapters, implement the action items, discuss reflection questions as a team.

---

## Navigation

- **Next**: [Chapter 0: Introduction](chapter0.md)
- **Reference**: [Glossary](glossary.md) | [Quick Reference](quick-reference.md)
- **Book Index**: [Book Overview](index.md)

---

*Ready to start? Begin with [Chapter 0: Introduction - The Product Mindset for RAG](chapter0.md).*
