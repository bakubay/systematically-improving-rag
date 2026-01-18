---
title: "Case Study: Construction Project Management"
description: "Complete case study showing systematic RAG improvement from 27% to 85% recall, 35% retention improvement, and 65% to 84% overall success over 12 months"
authors:
  - Jason Liu
date: 2024-01-15
tags:
  - case-study
  - construction
  - blueprints
  - routing
  - production
---

# Case Study: Construction Project Management

## Overview

This case study follows a construction project management company through their complete RAG improvement journey. The system serves contractors who need to search building blueprints, project documents, schedules, and compliance information.

**Key Results**:

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| Blueprint Search Recall | 27% | 85% | +58 points |
| New User Retention | - | +35% | Significant |
| Overall System Success | 65% | 84% | +19 points |
| Cost per Query | $0.09 | $0.04 | -56% |

**Timeline**: 12 months from initial deployment to mature production system

---

## Chapter Connections

This case study demonstrates concepts from every chapter in the book:

| Chapter | Concept Applied | Result |
|---------|-----------------|--------|
| Chapter 1 | Evaluation framework | Established baseline metrics |
| Chapter 2 | Fine-tuning | Improved date extraction |
| Chapter 3 | Feedback collection | 40-60 daily submissions |
| Chapter 4 | Query segmentation | Identified scheduling as danger zone |
| Chapter 5 | Specialized retrieval | Blueprint search with vision captions |
| Chapter 6 | Query routing | 95% routing accuracy |
| Chapter 7 | Production operations | Maintained flywheel at scale |

---

## The Business Problem

!!! tip "For Product Managers"
    **Initial situation**: The company built a RAG system for contractors to search project information. Workers asked questions like:

    - "Which rooms have north-facing windows?"
    - "Show me all electrical outlet locations"
    - "When is the foundation pour scheduled?"
    - "What are the safety requirements for concrete work?"

    **The symptoms**:

    - Workers abandoned the AI system and went back to manually scrolling through PDFs
    - New users tried the system once and never returned
    - Support tickets complained about "useless search results"
    - Only 8% of queries were scheduling-related, but scheduling had 25% satisfaction

    **The hidden cost**: The company was losing contractors who tried the system once, failed to find what they needed, and never came back. The 8% scheduling volume masked the fact that 90% of new users asked scheduling questions on day 1.

!!! tip "For Engineers"
    **Technical symptoms**:

    - Standard text embeddings on blueprints returned irrelevant results
    - Spatial queries ("north-facing windows") had no semantic match in text chunks
    - Date-based queries ("next week's schedule") failed because dates were embedded as text
    - No way to distinguish between document types (blueprints vs. specifications vs. schedules)

---

## Phase 1: Establishing Baseline (Chapter 1)

The first step was measuring what was actually happening, not what the team assumed was happening.

!!! tip "For Product Managers"
    **Initial metrics**:

    | Segment | Volume | Satisfaction |
    |---------|--------|--------------|
    | Document search | 52% | 70% |
    | Scheduling | 8% | 25% |
    | Cost lookup | 15% | 82% |
    | Compliance | 12% | 78% |
    | Other | 13% | 65% |

    **The revelation**: Using the 2x2 prioritization matrix from Chapter 4, scheduling appeared in the "danger zone"—low volume but extremely low satisfaction. Further analysis revealed the user adaptation pattern: new users asked 90% scheduling queries on day 1, but by day 30, only 20% were scheduling queries. Users learned the system could not handle scheduling and stopped asking.

    **Business insight**: The 8% volume was not low demand—it was learned helplessness. Fixing scheduling would not just improve that segment; it would unlock new user retention.

!!! tip "For Engineers"
    **Evaluation setup**:

    ```python
    from dataclasses import dataclass
    from typing import Literal

    @dataclass
    class EvaluationResult:
        query: str
        segment: Literal["blueprint", "document", "schedule", "compliance"]
        retrieved_docs: list[str]
        relevant_docs: list[str]
        recall_at_5: float
        user_satisfaction: float | None

    async def evaluate_segment(
        segment: str,
        test_queries: list[str],
        retriever
    ) -> dict:
        """Evaluate retrieval performance for a specific segment."""
        results = []
        for query in test_queries:
            retrieved = await retriever.search(query, k=5)
            relevant = get_ground_truth(query)
            recall = len(set(retrieved) & set(relevant)) / len(relevant)
            results.append(recall)

        return {
            "segment": segment,
            "recall_at_5": sum(results) / len(results),
            "sample_size": len(results)
        }
    ```

    **Baseline results**:

    - Blueprint search: 27% recall
    - Document search: 70% recall
    - Schedule lookup: 25% recall (capability issue—dates in text)
    - Overall system: 65% success rate

---

## Phase 2: Blueprint Search Improvement (Chapter 5)

The blueprint search problem was the most visible failure. Workers asked simple spatial questions and got completely unrelated results.

!!! tip "For Product Managers"
    **The problem**: Standard text embeddings could not handle spatial and visual queries. "North-facing windows" and "electrical outlets" are visual concepts that do not translate to text chunks.

    **The timeline**:

    | Stage | Recall | Time | Key Change |
    |-------|--------|------|------------|
    | Baseline | 27% | - | Text embeddings on blueprints |
    | Vision captions | 85% | 4 days | Added spatial descriptions |
    | Counting queries | 92% | +2 weeks | Bounding box detection |

    **The investment**: Approximately $10 in LLM calls per document for summarization. For 1,000 blueprints, total processing cost was $10,000. The ROI was immediate—workers could find relevant blueprints in seconds instead of hours.

    **Key insight**: Test subsystems independently for rapid improvements. Do not try to solve everything at once.

!!! tip "For Engineers"
    **Why standard embeddings failed**: Vision models are not trained for spatial search. CLIP embeddings understand "this looks like a blueprint" but not "this blueprint has 4 bedrooms."

    **Solution: Vision-to-text transformation**:

    ```python
    async def generate_blueprint_caption(
        image: bytes,
        vision_model
    ) -> str:
        """Generate searchable caption from blueprint image."""
        prompt = """Analyze this architectural blueprint and extract:

        1. ROOM COUNTS: Count all rooms by type (bedrooms, bathrooms, etc.)
        2. DIMENSIONS: List key dimensions (total square footage, room sizes)
        3. ORIENTATION: Identify building orientation (north-facing windows, etc.)
        4. KEY FEATURES: Note architectural features users might search for

        Format as a searchable summary that matches how construction workers
        phrase their queries.
        """

        response = await vision_model.analyze(image, prompt)
        return response.text
    ```

    **The key insight**: The summary prompt anticipated user mental models. Instead of describing what the blueprint looked like, it extracted what users actually searched for.

    **Implementation pattern**:

    ```python
    async def index_blueprint(blueprint_path: str) -> None:
        """Index a blueprint with searchable captions."""
        # Generate vision caption
        image_bytes = load_image(blueprint_path)
        caption = await generate_blueprint_caption(image_bytes, vision_model)

        # Create embedding from caption (not image)
        embedding = await embed_text(caption)

        # Store with metadata
        await vector_db.insert(
            id=blueprint_path,
            embedding=embedding,
            metadata={
                "type": "blueprint",
                "caption": caption,
                "original_path": blueprint_path
            }
        )
    ```

---

## Phase 3: Scheduling Fix (Chapter 4)

The scheduling problem was a capability issue, not an inventory issue. The scheduling information existed in documents—the system just could not process temporal queries correctly.

!!! tip "For Product Managers"
    **Diagnosis using Chapter 4 framework**:

    - **Inventory issue?** No—schedule information existed in project documents
    - **Capability issue?** Yes—the system could not parse dates or understand temporal queries

    **The solution**:

    1. Extract date metadata from all documents using LLM
    2. Build specialized calendar index for date-based queries
    3. Add explicit date range filtering to query processor
    4. Train classifier to detect scheduling queries and route appropriately

    **Results**:

    - Scheduling satisfaction: 25% → 78%
    - New user retention: +35%
    - Document search volume increased (users trusted the system more)

!!! tip "For Engineers"
    **Date extraction pipeline**:

    ```python
    from pydantic import BaseModel
    from datetime import date

    class ExtractedDates(BaseModel):
        dates: list[date]
        date_descriptions: list[str]
        is_deadline: list[bool]

    async def extract_dates_from_document(
        document: str,
        llm_client
    ) -> ExtractedDates:
        """Extract all dates and their context from a document."""
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Extract all dates from this construction document.
                    For each date, provide:
                    1. The date in ISO format
                    2. What the date refers to (e.g., "foundation pour deadline")
                    3. Whether it's a deadline or milestone"""
                },
                {"role": "user", "content": document}
            ],
            response_model=ExtractedDates
        )
        return response
    ```

    **Calendar index structure**:

    ```python
    # Specialized index for temporal queries
    calendar_index = {
        "2024-03-15": [
            {"doc_id": "proj_123", "event": "foundation pour", "is_deadline": True},
            {"doc_id": "proj_456", "event": "permit review", "is_deadline": False}
        ],
        "2024-03-22": [
            {"doc_id": "proj_123", "event": "framing start", "is_deadline": False}
        ]
    }

    async def search_by_date_range(
        start_date: date,
        end_date: date
    ) -> list[dict]:
        """Search for events within a date range."""
        results = []
        for date_key, events in calendar_index.items():
            if start_date <= date.fromisoformat(date_key) <= end_date:
                results.extend(events)
        return results
    ```

---

## Phase 4: Query Routing (Chapter 6)

With specialized retrievers for blueprints, documents, and schedules, the next challenge was routing queries to the right tool.

!!! tip "For Product Managers"
    **The routing problem**:

    Three excellent specialized retrievers:

    - Blueprint Search: 85% accuracy when used
    - Document Search: 78% accuracy on safety procedures and specifications
    - Schedule Lookup: 82% accuracy for timeline queries

    With a monolithic system routing all queries to generic search, overall performance was only 65%. Blueprint queries hit document search. Schedule questions went to blueprint search.

    **The routing journey**:

    | Week | Routing Accuracy | Overall Success | Key Change |
    |------|------------------|-----------------|------------|
    | 1 | 88% | 70% | 10 examples per tool |
    | 2 | 88% | 70% | Basic routing working |
    | 3-4 | 95% | 78% | 40 examples per tool |

    **ROI of example investment**: The team spent approximately 4 hours creating 40 examples per tool (120 total). This investment improved routing accuracy from 88% to 95%, translating to a 7 percentage point improvement in overall system success.

!!! tip "For Engineers"
    **Router implementation**:

    ```python
    from pydantic import BaseModel
    import instructor
    from openai import OpenAI

    client = instructor.from_openai(OpenAI())

    class SearchBlueprint(BaseModel):
        """Search for building plans and blueprints."""
        description: str
        start_date: str | None = None
        end_date: str | None = None

    class SearchText(BaseModel):
        """Search for text documents like contracts and proposals."""
        query: str
        document_types: list[str] | None = None

    class SearchSchedule(BaseModel):
        """Search for project timelines and schedules."""
        query: str
        date_range_start: str | None = None
        date_range_end: str | None = None

    FEW_SHOT_EXAMPLES = """
    <examples>
    Query: "Find blueprints with 4 bedrooms"
    Tool: SearchBlueprint
    Parameters: {"description": "4 bedrooms"}

    Query: "What's the safety procedure for concrete pouring?"
    Tool: SearchText
    Parameters: {"query": "concrete pouring safety procedure"}

    Query: "When is the foundation pour?"
    Tool: SearchSchedule
    Parameters: {"query": "foundation pour"}

    Query: "Show me buildings with north-facing windows from 2020"
    Tool: SearchBlueprint
    Parameters: {
        "description": "north-facing windows",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31"
    }
    </examples>
    """

    async def route_query(query: str) -> BaseModel:
        """Route a query to the appropriate tool."""
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a query router for a construction
                    information system. Analyze the user's query and decide
                    which tool should handle it.

                    {FEW_SHOT_EXAMPLES}"""
                },
                {"role": "user", "content": query}
            ],
            response_model=SearchBlueprint | SearchText | SearchSchedule
        )
        return response
    ```

    **Confusion matrix analysis** revealed that schedule queries were often misrouted to document search because both mentioned project names. Adding examples with explicit date references improved routing accuracy for temporal queries.

---

## Phase 5: Production Operations (Chapter 7)

With the system working well, the focus shifted to maintaining performance while managing costs and scaling.

!!! tip "For Product Managers"
    **Production journey over 12 months**:

    | Metric | Month 1-2 | Month 3-6 | Month 7-12 |
    |--------|-----------|-----------|------------|
    | Daily Queries | 500 | 500 | 2,500 |
    | Routing Accuracy | 95% | 95% | 96% |
    | Retrieval Accuracy | 82% | 85% | 87% |
    | Overall Success | 78% | 81% | 84% |
    | Daily Cost | $45 | $32 | $98 |
    | Cost per Query | $0.09 | $0.064 | $0.04 |
    | Feedback/Day | 40 | 45 | 60 |

    **Month 1-2 (Initial Deploy)**:

    - Baseline established with evaluation framework
    - Feedback collection generating 40 submissions daily
    - Cost per query: $0.09

    **Month 3-6 (First Improvement Cycle)**:

    - Used feedback to identify schedule search issues (dates parsed incorrectly)
    - Fine-tuned date extraction
    - Implemented prompt caching: $45/day to $32/day (29% reduction)
    - Overall success improved from 78% to 81%

    **Month 7-12 (Sustained Improvement)**:

    - 5x query growth while improving unit economics
    - Added new tool for permit search based on usage patterns
    - Updated routing with 60 examples per tool
    - Cost per query dropped to $0.04 despite increased complexity

!!! tip "For Engineers"
    **Cost optimization implementation**:

    ```python
    import time

    class ProductionPipeline:
        def __init__(self):
            self.cache = MultiLevelCache(
                semantic_threshold=0.95,
                ttl_hours=24
            )
            self.metrics = MetricsCollector()

        async def process_query(self, query: str) -> dict:
            start_time = time.time()

            # Check cache first
            cached = self.cache.get(query)
            if cached:
                self.metrics.record_cache_hit()
                return {**cached["result"], "cache_hit": True}

            # Process and cache
            result = await self._process_uncached(query)
            self.cache.set(query, result)

            # Record metrics
            latency = (time.time() - start_time) * 1000
            self.metrics.record_query(
                latency_ms=latency,
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                routing_decision=result.get("routing"),
                cache_hit=False
            )

            return result
    ```

    **Monitoring dashboard metrics**:

    - Query latency (p50, p95, p99)
    - Cache hit rate (target: >30%)
    - Routing accuracy (sampled daily)
    - Retrieval recall (sampled weekly)
    - Cost per query (daily)
    - User feedback rate

---

## Key Lessons Learned

!!! tip "For Product Managers"
    **Strategic insights**:

    1. **Volume can be misleading**: The 8% scheduling volume masked a critical retention problem. Always look at user journeys, not just aggregate metrics.

    2. **User adaptation hides failures**: Users learn to avoid broken features. Low volume in a segment might indicate learned helplessness, not low demand.

    3. **Specialized beats general**: Building three specialized retrievers (blueprint, document, schedule) with routing outperformed any attempt at a single general-purpose system.

    4. **Investment in examples pays off**: 4 hours creating routing examples delivered a 7 percentage point improvement in overall success.

    5. **Cost and quality can improve together**: Over 12 months, cost per query dropped 56% while overall success improved 6 percentage points.

!!! tip "For Engineers"
    **Technical insights**:

    1. **Match embeddings to queries**: Blueprint search failed because text embeddings could not represent spatial concepts. Vision-to-text transformation solved this.

    2. **Capability vs inventory**: The scheduling problem was capability (could not parse dates), not inventory (dates existed in documents). Different problems require different solutions.

    3. **Routing accuracy matters**: With 95% routing accuracy and 82% average retrieval accuracy, overall success was 78%. Improving routing from 88% to 95% was more impactful than improving any single retriever.

    4. **Cache aggressively**: Semantic caching with 0.95 similarity threshold achieved >30% hit rate, reducing costs significantly.

    5. **Measure continuously**: The evaluation framework from Chapter 1 remained active in production, enabling continuous improvement.

---

## Metrics Timeline

```text
Month 1:  Blueprint 27% → 85% (vision captions)
Month 2:  Scheduling 25% → 78% (date extraction)
Month 3:  Routing 88% → 95% (40 examples per tool)
Month 4:  Overall 78% → 81% (prompt caching)
Month 6:  Cost $0.09 → $0.064 (caching + optimization)
Month 12: Overall 84%, Cost $0.04 (sustained improvement)
```

---

## Related Content

- [Chapter 1: Evaluation-First Development](chapter1.md) - Evaluation framework used throughout
- [Chapter 4: Query Understanding and Prioritization](chapter4.md) - Segmentation and prioritization
- [Chapter 5: Specialized Retrieval Systems](chapter5.md) - Blueprint search implementation
- [Chapter 6: Query Routing and Orchestration](chapter6.md) - Routing implementation
- [Chapter 7: Production Operations](chapter7.md) - Production monitoring and optimization

---

## Navigation

- **Next**: [WildChat Case Study](case-study-wildchat.md)
- **Reference**: [Glossary](glossary.md) | [Quick Reference](quick-reference.md)
- **Book Index**: [Book Overview](index.md)
