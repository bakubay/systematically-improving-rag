---
title: "Case Study: WildChat Alignment Problem"
description: "Hands-on case study demonstrating the alignment problem in RAG systems using the WildChat dataset, improving pattern queries from 11% to 55% and content queries from 62% to 82%"
authors:
  - Jason Liu
date: 2024-01-15
tags:
  - case-study
  - wildchat
  - alignment
  - embeddings
  - evaluation
---

# Case Study: WildChat Alignment Problem

## Overview

This case study uses the WildChat dataset to demonstrate the fundamental alignment problem in RAG systems. You will discover why query generation strategy and embedding strategy must be aligned—and what happens when they are not.

**Key Results**:

| Approach | v1 Queries (Content) | v2 Queries (Pattern) | Storage |
|----------|---------------------|---------------------|---------|
| First Message | 62% | 12% | 1x |
| Full Conversation | 55% | 45% | 10x |
| v1 Summary | 58% | 15% | 2x |
| v4 Summary | 52% | 42% | 3x |

**The Core Insight**: You cannot search for patterns in embeddings that do not contain pattern information. Alignment between queries and embeddings matters more than model sophistication.

**Code Location**: `latest/case_study/` in the repository

---

## Chapter Connections

This case study demonstrates concepts from the first half of the book:

| Chapter | Concept Applied | Result |
|---------|-----------------|--------|
| Chapter 0 | The alignment problem | Core discovery |
| Chapter 1 | Evaluation framework | Systematic measurement |
| Chapter 2 | Embedding strategies | Multiple approaches tested |
| Chapter 5 | Specialized retrieval | Summary-based indices |

---

## The Business Problem

!!! tip "For Product Managers"
    **The scenario**: You are building a conversation search system. Users want to find past conversations based on different criteria:

    - **Content queries**: "Find conversations about Python programming"
    - **Pattern queries**: "Find conversations where the user was frustrated"

    **The challenge**: A single embedding strategy cannot serve both query types well. Content queries work with first-message embeddings. Pattern queries require understanding the full conversation flow.

    **Business implications**:

    - If you only support content queries, users cannot find conversations by behavior patterns
    - If you only support pattern queries, simple topic searches become unreliable
    - The choice of embedding strategy determines which use cases your system can serve

!!! tip "For Engineers"
    **Technical setup**:

    The WildChat dataset contains 1 million real conversations with ChatGPT. Each conversation has:

    - Multiple turns (user messages and assistant responses)
    - Metadata (language, country, timestamps)
    - Varying lengths and complexity

    **The experiment**:

    1. Generate two types of synthetic queries:
       - **v1 queries**: Content-focused ("What topics were discussed?")
       - **v2 queries**: Pattern-focused ("What was the conversation dynamic?")

    2. Create multiple embedding strategies:
       - First message only
       - Full conversation
       - Various summary approaches

    3. Measure recall for each combination

---

## The Alignment Problem

The alignment problem is the fundamental insight of this case study: embedding strategies encode specific information, and queries can only find what is encoded.

!!! tip "For Product Managers"
    **The mental model**:

    Think of embeddings as a filing system. If you file documents by topic, you can find them by topic. If you file by date, you can find them by date. But you cannot find documents by date if they are only filed by topic.

    **In RAG systems**:

    - First-message embeddings encode: topic, initial question, user intent
    - Full-conversation embeddings encode: topic, dynamics, resolution, patterns
    - Summary embeddings encode: whatever the summary prompt extracts

    **The mismatch**: When users ask pattern queries ("frustrated conversations") but your embeddings only contain topic information, recall drops dramatically—from 62% to 12% in our experiments.

!!! tip "For Engineers"
    **Why this happens mathematically**:

    Embedding models create vector representations that capture semantic similarity. But "semantic similarity" depends on what text you embed:

    ```python
    # First message: "How do I sort a list in Python?"
    # Embedding captures: Python, sorting, lists, programming question

    # Full conversation includes:
    # - User: "How do I sort a list in Python?"
    # - Assistant: [detailed explanation]
    # - User: "That doesn't work, I'm getting an error"
    # - Assistant: [debugging help]
    # - User: "Finally! Thank you so much!"

    # Full conversation embedding captures:
    # - Python, sorting, lists (same as first message)
    # - Debugging, errors, troubleshooting (new)
    # - Resolution, satisfaction (new)
    ```

    **The v2 query**: "Find conversations where the user struggled but eventually succeeded"

    - Against first-message embedding: No signal for "struggled" or "succeeded"
    - Against full-conversation embedding: Strong signal for both

---

## Experimental Setup

!!! tip "For Product Managers"
    **What we measured**:

    - **Recall@1**: Did the correct conversation appear as the top result?
    - **Recall@5**: Did the correct conversation appear in the top 5 results?

    **Why recall matters**: In a search system, if the right answer is not in the results, no amount of reranking or post-processing will help. Recall is the ceiling on system performance.

!!! tip "For Engineers"
    **Query generation strategies**:

    ```python
    # v1: Content-focused queries
    V1_PROMPT = """Generate a search query that would help find this
    conversation based on its TOPIC and CONTENT.

    Focus on:
    - Main subject matter discussed
    - Specific questions asked
    - Technical concepts mentioned
    """

    # v2: Pattern-focused queries
    V2_PROMPT = """Generate a search query that would help find this
    conversation based on its PATTERN and DYNAMICS.

    Focus on:
    - How the conversation evolved
    - User sentiment and engagement
    - Whether issues were resolved
    - Conversation style (technical, casual, frustrated)
    """
    ```

    **Embedding strategies**:

    ```python
    async def embed_first_message(conversation: Conversation) -> list[float]:
        """Embed only the first user message."""
        first_message = conversation.messages[0].content
        return await embedding_model.embed(first_message)

    async def embed_full_conversation(conversation: Conversation) -> list[float]:
        """Embed the entire conversation."""
        full_text = "\n".join(
            f"{msg.role}: {msg.content}"
            for msg in conversation.messages
        )
        return await embedding_model.embed(full_text)

    async def embed_summary(
        conversation: Conversation,
        summary_version: str
    ) -> list[float]:
        """Embed a generated summary of the conversation."""
        summary = await generate_summary(conversation, summary_version)
        return await embedding_model.embed(summary)
    ```

---

## Results and Analysis

### First Message Embeddings

!!! tip "For Product Managers"
    **Results**:

    | Query Type | Recall@1 | Recall@5 |
    |------------|----------|----------|
    | v1 (Content) | 62% | 78% |
    | v2 (Pattern) | 12% | 24% |

    **Interpretation**: First-message embeddings work well for content queries but fail completely for pattern queries. This makes sense—the first message contains topic information but no information about how the conversation evolved.

    **Business decision**: If your users only need content search, first-message embeddings are efficient (1x storage) and effective. If users need pattern search, this approach will not work.

!!! tip "For Engineers"
    **Why v2 queries fail**:

    Consider a v2 query: "Find conversations where the user was confused and needed multiple explanations"

    The first message might be: "How do I use async/await in Python?"

    This message contains no signal for:
    - Confusion (that comes later)
    - Multiple explanations (requires seeing the full conversation)
    - Resolution status (only visible at the end)

    The embedding model cannot find what is not there.

### Full Conversation Embeddings

!!! tip "For Product Managers"
    **Results**:

    | Query Type | Recall@1 | Recall@5 |
    |------------|----------|----------|
    | v1 (Content) | 55% | 72% |
    | v2 (Pattern) | 45% | 62% |

    **Interpretation**: Full conversation embeddings improve pattern queries significantly (12% → 45%) but slightly hurt content queries (62% → 55%). The embedding now contains more information, but it is also noisier for simple topic searches.

    **Trade-off**: 10x storage cost for better pattern search but worse content search.

!!! tip "For Engineers"
    **The noise problem**:

    Full conversation embeddings include everything:
    - The original question (good for content search)
    - All the back-and-forth (noise for content search)
    - Resolution and sentiment (good for pattern search)

    For a simple content query like "Python sorting," the embedding now includes debugging discussions, thank-you messages, and other content that dilutes the topic signal.

### Summary Embeddings

!!! tip "For Product Managers"
    **Results for v4 summaries** (pattern-optimized):

    | Query Type | Recall@1 | Recall@5 |
    |------------|----------|----------|
    | v1 (Content) | 52% | 68% |
    | v2 (Pattern) | 42% | 58% |

    **Interpretation**: Summary embeddings provide a middle ground—better than first-message for patterns, more storage-efficient than full conversation. The summary prompt determines what information is captured.

    **Strategic insight**: You can design summaries to capture specific information. A pattern-focused summary prompt will improve pattern search. A content-focused summary prompt will improve content search.

!!! tip "For Engineers"
    **Summary prompt design**:

    ```python
    # v1 summary: Content-focused
    V1_SUMMARY_PROMPT = """Summarize this conversation focusing on:
    - Main topic discussed
    - Key questions asked
    - Technical concepts mentioned
    """

    # v4 summary: Pattern-focused
    V4_SUMMARY_PROMPT = """Summarize this conversation focusing on:
    - How the conversation evolved
    - User engagement and sentiment
    - Whether the user's issue was resolved
    - Key turning points in the discussion
    """
    ```

    **The alignment principle**: Match your summary prompt to your expected query types. If users will search for patterns, generate pattern-focused summaries.

---

## The Solution: Multiple Indices

The ultimate solution is to maintain multiple indices optimized for different query types.

!!! tip "For Product Managers"
    **Architecture**:

    | Index | Optimized For | Storage | Use Case |
    |-------|---------------|---------|----------|
    | First Message | Content queries | 1x | Topic search |
    | Pattern Summary | Pattern queries | 2x | Behavior search |
    | Full Conversation | Complex queries | 10x | Deep analysis |

    **Routing strategy**: Use query classification (Chapter 6) to route queries to the appropriate index. Content queries go to the first-message index. Pattern queries go to the pattern-summary index.

    **Cost-benefit**: The additional storage cost (3-13x depending on configuration) is justified by the dramatic improvement in pattern query recall (12% → 42%+).

!!! tip "For Engineers"
    **Multi-index implementation**:

    ```python
    from enum import Enum

    class QueryType(Enum):
        CONTENT = "content"
        PATTERN = "pattern"
        COMPLEX = "complex"

    async def classify_query(query: str) -> QueryType:
        """Classify query to determine which index to use."""
        # Use few-shot classification or embedding similarity
        # to determine query type
        pass

    async def search(query: str) -> list[Conversation]:
        """Search using the appropriate index."""
        query_type = await classify_query(query)

        if query_type == QueryType.CONTENT:
            return await first_message_index.search(query)
        elif query_type == QueryType.PATTERN:
            return await pattern_summary_index.search(query)
        else:
            return await full_conversation_index.search(query)
    ```

---

## Running the Case Study

The complete case study code is available in `latest/case_study/`. Here is how to run it:

!!! tip "For Engineers"
    **Setup**:

    ```bash
    cd latest/case_study
    uv sync
    cp .env.example .env
    # Edit .env with your OpenAI API key
    ```

    **Load data**:

    ```bash
    uv run python main.py load-wildchat --limit 1000
    uv run python main.py stats
    ```

    **Generate queries**:

    ```bash
    uv run python main.py generate-questions --version v1 --limit 1000
    uv run python main.py generate-questions --version v2 --limit 1000
    ```

    **Create embeddings**:

    ```bash
    uv run python main.py embed-conversations --embedding-model text-embedding-3-small
    ```

    **Evaluate**:

    ```bash
    uv run python main.py evaluate --question-version v1 --embedding-model text-embedding-3-small
    uv run python main.py evaluate --question-version v2 --embedding-model text-embedding-3-small
    ```

    **Generate and evaluate summaries**:

    ```bash
    uv run python main.py generate-summaries --versions v1,v4 --limit 1000
    uv run python main.py embed-summaries --technique v4 --embedding-model text-embedding-3-small
    uv run python main.py evaluate --question-version v2 --embedding-model text-embedding-3-small --target-type summary --target-technique v4
    ```

---

## Key Lessons Learned

!!! tip "For Product Managers"
    **Strategic insights**:

    1. **Alignment is fundamental**: The choice of embedding strategy determines which query types your system can serve. This is a product decision, not just a technical one.

    2. **Know your users**: If users primarily search by topic, first-message embeddings are efficient and effective. If users need pattern search, you must invest in richer embeddings.

    3. **Multiple indices may be necessary**: A single embedding strategy cannot serve all query types well. Plan for multiple indices with query routing.

    4. **Storage vs capability trade-off**: Richer embeddings (full conversation, summaries) cost more storage but enable new capabilities. Quantify the business value of pattern search before investing.

!!! tip "For Engineers"
    **Technical insights**:

    1. **Embeddings encode specific information**: You cannot search for what is not encoded. Design your embedding strategy around your expected query types.

    2. **Summary prompts are powerful**: A well-designed summary prompt can extract specific information for embedding. Match the prompt to your query types.

    3. **Measure before optimizing**: The 62% → 12% recall drop for v2 queries on first-message embeddings was only visible through systematic evaluation. Always measure.

    4. **Reranking cannot fix alignment**: If the relevant document is not in the candidate set, no amount of reranking will help. Alignment is a recall problem, not a ranking problem.

---

## Performance Benchmarks

### Embedding Model Comparison

| Model | Dimensions | v1 Recall@1 | v2 Recall@1 | Cost |
|-------|------------|-------------|-------------|------|
| all-MiniLM-L6-v2 | 384 | 54.8% | 10.7% | Free |
| text-embedding-3-small | 1536 | 58.7% | 11.3% | $0.02/1K |
| text-embedding-3-large | 3072 | 62.5% | 12.2% | $0.13/1K |

**Key insight**: Better embedding models improve recall slightly, but the alignment problem persists. A 3x more expensive model only improved v2 recall from 10.7% to 12.2%—still far below the 42% achieved with pattern-focused summaries.

### Processing Times (1000 conversations)

| Operation | Time |
|-----------|------|
| Question generation | ~5 minutes |
| Summary generation | ~15 minutes |
| Embedding creation | ~2 minutes |
| Evaluation | ~1 minute |

---

## Related Content

- [Chapter 0: Introduction](chapter0.md) - The alignment problem concept
- [Chapter 1: Evaluation-First Development](chapter1.md) - Evaluation methodology
- [Chapter 2: Training Data and Fine-Tuning](chapter2.md) - Embedding strategies
- [Chapter 5: Specialized Retrieval Systems](chapter5.md) - Summary-based indices
- [Appendix C: Benchmarking Your RAG System](appendix-benchmarks.md) - Evaluation methodology

---

## Navigation

[Previous: Construction Case Study](case-study-construction.md) | [Back to Case Studies Index](index.md) | [Next: Voice AI Case Study](case-study-voice-ai.md)
