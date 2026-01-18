---
title: "Chapter 6: Query Routing and Orchestration"
description: "Build intelligent query routing systems that direct queries to specialized retrievers. Learn router architectures, few-shot classification, tool interfaces, and two-level performance measurement to achieve system-wide success."
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - query routing
  - orchestration
  - tool interfaces
  - few-shot learning
  - router architecture
  - multi-agent systems
  - performance measurement
---

# Chapter 6: Query Routing and Orchestration

## Chapter at a Glance

**Prerequisites**: Chapter 5 (specialized retrieval systems), Chapter 1 (evaluation framework), understanding of embeddings and tool interfaces

**What You Will Learn**:

- Why routing matters: P(success) = P(right retriever) x P(finding data | right retriever)
- Three router architectures: classifier-based, embedding-based, and LLM-based
- How to implement few-shot classification with 10-40 examples per tool
- Building tool interfaces that work for both LLMs and humans
- Two-level performance measurement for diagnosing system bottlenecks
- When to use multi-agent versus single-agent architectures

**Case Study Reference**: Construction company improved from 65% to 78% overall success with routing, achieving 95% routing accuracy with 40 examples per tool

**Time to Complete**: 75-90 minutes

---

## Key Insight

**The best retriever is multiple retrievers—success = P(selecting right retriever) x P(retriever finding data).** Query routing is not about choosing one perfect system. It is about building a portfolio of specialized tools and letting a smart router decide. Start simple with few-shot classification, then evolve to fine-tuned models as you collect routing decisions. Without intelligent routing, even the best specialized retrievers sit unused because queries hit the wrong tools.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Design query routing systems** that direct queries to appropriate specialized retrievers, understanding the two-level performance formula
2. **Implement three router architectures**—classifier-based, embedding-based, and LLM-based—choosing the right approach for your constraints
3. **Build production-ready tool interfaces** using the tools-as-APIs pattern that enables parallel team development
4. **Master few-shot classification** with 10-40 examples per tool, including dynamic example selection and data leakage prevention
5. **Measure two-level performance** tracking both routing accuracy and retrieval quality to identify system bottlenecks
6. **Decide between multi-agent and single-agent architectures** based on your specific requirements for token efficiency, specialization, and safety

---

## Introduction

In Chapter 5, you built specialized retrieval systems for different content types—blueprint search, document search, schedule lookup. Each excels at its specific task. But building excellent specialized retrievers is only half the problem. Without intelligent routing, queries hit the wrong tools and users get irrelevant results.

This chapter shows how to build the orchestration layer that makes specialization work. The pattern mirrors how Google evolved: not one search engine, but Maps for locations, Images for visual content, YouTube for video—with intelligent routing that detects "pizza near me" should go to Maps while "how to make pizza" should go to YouTube.

**Where We Have Been:**

- **Chapter 0**: Introduced embeddings, vector search, and foundational concepts
- **Chapter 1**: Built evaluation frameworks to measure retrieval performance
- **Chapter 2**: Fine-tuned embeddings for domain-specific improvements
- **Chapter 3**: Collected user feedback revealing which queries fail
- **Chapter 4**: Segmented queries to identify patterns needing different approaches
- **Chapter 5**: Built specialized retrievers for each query pattern

**Now What?** Connect those specialized retrievers with intelligent routing. The work from Chapter 4—identifying segments like "blueprint spatial queries" or "schedule lookups"—directly informs which tools to route to. The specialized retrievers from Chapter 5 become the tools your router selects between.

!!! tip "For Product Managers"
    This chapter establishes the strategic framework for building unified RAG systems. Focus on understanding the two-level performance formula (routing accuracy x retrieval quality), the ROI of routing improvements, and how to organize teams around the tools-as-APIs pattern. The construction company case study demonstrates how routing improvements drove a 13 percentage point increase in overall success.

!!! tip "For Engineers"
    This chapter provides concrete implementation patterns for query routing. Pay attention to the three router architectures, few-shot example management, and the critical warning about data leakage. The code examples demonstrate production-ready patterns including dynamic example selection and confusion matrix analysis.

---

## Core Content

### The Query Routing Problem

Query routing means directing user queries to the right retrieval components. Without it, even excellent specialized retrievers become useless if they are never called for the right queries.

!!! tip "For Product Managers"
    **The business case for routing**:

    Consider the construction company from Chapter 4. They built three excellent specialized retrievers:

    - **Blueprint Search**: 85% accuracy when used (up from 27% baseline)
    - **Document Search**: 78% accuracy on safety procedures and specifications
    - **Schedule Lookup**: 82% accuracy for timeline queries

    **The Problem**: With a monolithic system routing all queries to generic search, overall performance was only 65%. Blueprint queries hit document search. Schedule questions went to blueprint search. The specialized tools sat mostly unused.

    **The Solution Timeline**:

    - **Week 1**: Implemented basic routing with 10 examples per tool using few-shot classification
    - **Week 2**: Routing accuracy reached 88%. Overall success = 88% routing x 80% avg retrieval = 70% (up from 65%)
    - **Week 4**: Added feedback collection tracking which routing decisions led to user satisfaction
    - **Week 6**: Expanded to 40 examples per tool. Routing accuracy improved to 95%. Overall success = 95% routing x 82% avg retrieval = 78%

    **The Key Formula**: P(success) = P(right tool | query) x P(finding data | right tool)

    This decomposition is powerful. When overall performance is 65%, you cannot tell if routing is broken (sending queries to wrong tools) or if retrievers are broken (tools cannot find answers). Measure both separately to know where to focus improvement efforts.

!!! tip "For Engineers"
    **The technical rationale for routing**:

    The mathematics support decomposition: when you have distinct query types, routing to specialized models beats general-purpose approaches. This pattern appears throughout machine learning—mixture of experts, task decomposition, modular systems.

    **Architecture Overview**:

    ```mermaid
    graph TD
        A[User Query] --> B[Query Router]
        B --> C[Tool Selection]
        C --> D[Document Tool]
        C --> E[Image Tool]
        C --> F[Table Tool]
        D --> G[Ranking]
        E --> G
        F --> G
        G --> H[Context Assembly]
        H --> I[Response Generation]
        I --> J[User Interface]
    ```

    This architecture resembles modern microservice patterns where specialized services handle specific tasks. The difference is that the "client" making API calls is often a language model rather than another service.

### Router Architectures

Three main approaches exist for building query routers, each with different tradeoffs.

!!! tip "For Product Managers"
    **Choosing the right architecture**:

    | Architecture | Best For | Cost | Accuracy | Latency |
    |-------------|----------|------|----------|---------|
    | Classifier-based | High volume, stable categories | Low | Medium-High | Very Low |
    | Embedding-based | Evolving categories, semantic matching | Medium | Medium | Low |
    | LLM-based | Complex routing, parameter extraction | High | Highest | Medium |

    **Decision framework**:

    - **Start with LLM-based**: Fastest to implement, easiest to iterate
    - **Move to classifier**: When volume justifies training cost and categories stabilize
    - **Use embedding-based**: When you need semantic flexibility without LLM costs

    Most teams should start with LLM-based routing using few-shot examples, then consider migration to classifiers once they have enough labeled data from production usage.

!!! tip "For Engineers"
    **Architecture 1: Classifier-Based Routing**

    Train a traditional classifier (logistic regression, random forest, or neural network) on labeled query-tool pairs.

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    class ClassifierRouter:
        """Route queries using a trained classifier."""

        def __init__(self):
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.classifier = LogisticRegression(multi_class='multinomial')
            self.tool_names: list[str] = []

        def train(self, queries: list[str], tools: list[str]) -> None:
            """
            Train the router on labeled query-tool pairs.

            Args:
                queries: List of user queries
                tools: List of corresponding tool names
            """
            self.tool_names = list(set(tools))
            X = self.vectorizer.fit_transform(queries)
            y = [self.tool_names.index(t) for t in tools]
            self.classifier.fit(X, y)

        def route(self, query: str) -> tuple[str, float]:
            """
            Route a query to the appropriate tool.

            Args:
                query: User query

            Returns:
                Tuple of (tool_name, confidence_score)
            """
            X = self.vectorizer.transform([query])
            probs = self.classifier.predict_proba(X)[0]
            tool_idx = np.argmax(probs)
            return self.tool_names[tool_idx], probs[tool_idx]
    ```

    **Pros**: Fast inference, low cost at scale, interpretable
    **Cons**: Requires labeled training data, fixed categories, cannot extract parameters

    **Architecture 2: Embedding-Based Routing**

    Use semantic similarity between query embeddings and tool description embeddings.

    ```python
    from typing import List, Tuple
    import numpy as np

    class EmbeddingRouter:
        """Route queries using embedding similarity."""

        def __init__(self, embedding_model):
            self.embedding_model = embedding_model
            self.tool_embeddings: dict[str, np.ndarray] = {}
            self.tool_descriptions: dict[str, str] = {}

        def register_tool(self, name: str, description: str) -> None:
            """
            Register a tool with its description.

            Args:
                name: Tool name
                description: Natural language description of when to use this tool
            """
            self.tool_descriptions[name] = description
            self.tool_embeddings[name] = self.embedding_model.embed(description)

        def route(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
            """
            Route query to most similar tools.

            Args:
                query: User query
                top_k: Number of tools to return

            Returns:
                List of (tool_name, similarity_score) tuples
            """
            query_embedding = self.embedding_model.embed(query)

            similarities = []
            for name, tool_embedding in self.tool_embeddings.items():
                sim = np.dot(query_embedding, tool_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
                )
                similarities.append((name, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
    ```

    **Pros**: No labeled data required, handles new tools easily, semantic flexibility
    **Cons**: Less precise than classifiers, sensitive to description quality

    **Architecture 3: LLM-Based Routing**

    Use a language model with structured outputs to select tools and extract parameters.

    ```python
    import instructor
    from pydantic import BaseModel
    from typing import Iterable, Literal
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
        document_type: Literal["contract", "proposal", "bid"] | None = None

    class SearchSchedule(BaseModel):
        """Search for project schedules and timelines."""
        project_name: str | None = None
        date_range: str | None = None

    class ClarifyQuestion(BaseModel):
        """Ask for clarification when the query is unclear."""
        question: str

    ToolType = SearchBlueprint | SearchText | SearchSchedule | ClarifyQuestion

    def route_query(query: str, examples: str = "") -> Iterable[ToolType]:
        """
        Route a query to appropriate tools using LLM.

        Args:
            query: User query
            examples: Few-shot examples for routing

        Returns:
            Iterable of tool objects with extracted parameters
        """
        system_prompt = f"""
        You are a query router for a construction information system.
        Analyze the user's query and decide which tool(s) should handle it.
        You can return multiple tools if the query requires different information.

        Available tools:
        - SearchBlueprint: For finding building plans and blueprints
        - SearchText: For finding text documents like contracts and proposals
        - SearchSchedule: For finding project timelines and schedules
        - ClarifyQuestion: For asking follow-up questions when unclear

        {examples}
        """

        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_model=Iterable[ToolType]
        )
    ```

    **Pros**: Highest accuracy, extracts parameters, handles complex queries
    **Cons**: Higher latency and cost, requires prompt engineering

### The Tools-as-APIs Pattern

Treat each specialized retriever as an API that language models can call. This creates separation between tool interfaces, tool implementations, and routing logic.

!!! tip "For Product Managers"
    **Benefits of the API approach**:

    - **Clear Boundaries**: Teams work independently on different tools
    - **Testability**: Components can be tested in isolation
    - **Reusability**: Tools work for both LLMs and direct API calls
    - **Scalability**: Add new capabilities without changing existing code

    **Team Organization for Scalable Development**:

    | Team | Responsibility | Success Metric |
    |------|---------------|----------------|
    | Interface Team | Design tool specifications, define contracts | API clarity, developer satisfaction |
    | Implementation Teams | Build specialized retrievers | Per-tool accuracy |
    | Router Team | Build and optimize query routing | Routing accuracy |
    | Evaluation Team | Test end-to-end performance | Overall system success |

    This separation allows teams to work independently while maintaining system coherence. The construction company achieved 40% faster feature delivery after adopting this structure.

!!! tip "For Engineers"
    **Tool interface design**:

    ```python
    from pydantic import BaseModel
    from typing import List, Optional
    from abc import ABC, abstractmethod

    class ToolResult(BaseModel):
        """Standard result format for all tools."""
        tool_name: str
        results: List[dict]
        confidence: float
        metadata: dict = {}

    class BaseTool(ABC, BaseModel):
        """Base class for all retrieval tools."""

        @abstractmethod
        async def execute(self) -> ToolResult:
            """Execute the tool and return results."""
            pass

        @property
        @abstractmethod
        def description(self) -> str:
            """Return tool description for routing."""
            pass

    class BlueprintSearchTool(BaseTool):
        """Search for building plans and blueprints."""
        description_query: str
        start_date: Optional[str] = None
        end_date: Optional[str] = None

        @property
        def description(self) -> str:
            return """
            Use this tool to search for building plans, blueprints, and
            architectural drawings. Supports filtering by date range and
            searching by description (room counts, dimensions, features).

            Example queries:
            - "Find blueprints with 4 bedrooms"
            - "Show me buildings with north-facing windows from 2020"
            - "Residential plans with open floor plans"
            """

        async def execute(self) -> ToolResult:
            """Execute blueprint search."""
            # Build query with filters
            query = self._build_query()
            results = await self._search_index(query)
            return ToolResult(
                tool_name="blueprint_search",
                results=results,
                confidence=self._calculate_confidence(results),
                metadata={"filters_applied": self._get_filters()}
            )

        def _build_query(self) -> dict:
            """Build search query with date filters."""
            query = {"description": self.description_query}
            if self.start_date:
                query["start_date"] = self.start_date
            if self.end_date:
                query["end_date"] = self.end_date
            return query
    ```

    **Tool portfolio design principle**: Tools do not map one-to-one with retrievers. Like command-line utilities, multiple tools can access the same underlying data in different ways.

    ```python
    # One retriever, multiple access patterns
    class DocumentRetriever:
        """Core retrieval engine for all documents."""
        pass

    # Tool 1: Search by keyword
    class SearchDocuments(BaseTool):
        query: str

    # Tool 2: Find by metadata
    class FindDocumentsByMetadata(BaseTool):
        author: Optional[str] = None
        date_range: Optional[str] = None
        document_type: Optional[str] = None

    # Tool 3: Get related documents
    class GetRelatedDocuments(BaseTool):
        document_id: str
        similarity_threshold: float = 0.8
    ```

### Few-Shot Classification for Routing

Good examples are critical for router effectiveness. They help the model recognize patterns that should trigger specific tools.

!!! tip "For Product Managers"
    **Example scaling guidelines**:

    | Stage | Examples per Tool | Expected Accuracy |
    |-------|------------------|-------------------|
    | Development | 5-10 | 70-80% |
    | Production (initial) | 10-20 | 80-90% |
    | Production (mature) | 20-40 | 90-95% |
    | Advanced | 40+ with dynamic selection | 95%+ |

    The quality of examples matters as much as quantity. Include edge cases, ambiguous queries, and multi-tool scenarios in your example set.

    **ROI of example investment**: The construction company spent approximately 4 hours creating 40 examples per tool (120 total). This investment improved routing accuracy from 88% to 95%, translating to a 7 percentage point improvement in overall system success.

!!! tip "For Engineers"
    **Creating effective few-shot examples**:

    ```python
    FEW_SHOT_EXAMPLES = """
    <examples>
    Query: "Find blueprints for the city hall built in 2010."
    Tool: SearchBlueprint
    Parameters: {
        "description_query": "city hall blueprints",
        "start_date": "2010-01-01",
        "end_date": "2010-12-31"
    }

    Query: "I need plans for residential buildings constructed after 2015."
    Tool: SearchBlueprint
    Parameters: {
        "description_query": "residential building plans",
        "start_date": "2015-01-01",
        "end_date": null
    }

    Query: "I need the contract for the Johnson project."
    Tool: SearchText
    Parameters: {
        "query": "Johnson project contract",
        "document_type": "contract"
    }

    Query: "When is the foundation pour scheduled for the Main Street project?"
    Tool: SearchSchedule
    Parameters: {
        "project_name": "Main Street",
        "date_range": null
    }

    Query: "Find me school building plans from 2018-2020 and any related bid documents."
    Tools: [SearchBlueprint, SearchText]
    Parameters: [
        {"description_query": "school building plans", "start_date": "2018-01-01", "end_date": "2020-12-31"},
        {"query": "school building bids", "document_type": "bid"}
    ]

    Query: "I'm not sure what kind of building plans I need for my renovation."
    Tool: ClarifyQuestion
    Parameters: {
        "question": "Could you tell me more about your renovation project? What type of building is it, what changes are you planning, and do you need plans for permits or construction guidance?"
    }
    </examples>
    """
    ```

    **Example selection principles**:

    1. **Cover edge cases**: Include ambiguous queries that test boundaries
    2. **Multi-tool examples**: Show when to use multiple tools together
    3. **Hard decisions**: Similar queries that go to different tools
    4. **Real queries**: Use actual user examples when possible
    5. **Diversity**: Cover all tools and parameter combinations

    **Dynamic example selection**:

    Once you have enough interaction data, select relevant examples dynamically for each query:

    ```python
    from typing import List
    import numpy as np

    class DynamicExampleSelector:
        """Select relevant examples based on query similarity."""

        def __init__(self, embedding_model):
            self.embedding_model = embedding_model
            self.example_database: List[dict] = []
            self.example_embeddings: List[np.ndarray] = []

        def add_example(self, query: str, tool: str, parameters: dict) -> None:
            """Add an example to the database."""
            embedding = self.embedding_model.embed(query)
            self.example_database.append({
                "query": query,
                "tool": tool,
                "parameters": parameters,
                "embedding": embedding
            })
            self.example_embeddings.append(embedding)

        def get_examples(self, query: str, num_examples: int = 5) -> List[dict]:
            """
            Get most relevant examples for a query.

            Args:
                query: User query
                num_examples: Number of examples to return

            Returns:
                List of relevant examples
            """
            query_embedding = self.embedding_model.embed(query)

            # Calculate similarities
            similarities = []
            for i, example_embedding in enumerate(self.example_embeddings):
                sim = np.dot(query_embedding, example_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(example_embedding)
                )
                similarities.append((i, sim))

            # Sort by similarity and return top examples
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [self.example_database[i] for i, _ in similarities[:num_examples]]

        def format_examples(self, examples: List[dict]) -> str:
            """Format examples for inclusion in prompt."""
            formatted = ["<examples>"]
            for ex in examples:
                formatted.append(f"Query: \"{ex['query']}\"")
                formatted.append(f"Tool: {ex['tool']}")
                formatted.append(f"Parameters: {ex['parameters']}")
                formatted.append("")
            formatted.append("</examples>")
            return "\n".join(formatted)
    ```

!!! warning "Critical Warning: Preventing Data Leakage"
    **The most common router evaluation mistake**:

    When you have limited data (20-50 examples total), it is easy for your test queries to accidentally appear in your few-shot examples. This creates artificially high performance that does not generalize.

    **Why this happens**:

    - Small datasets mean high overlap probability
    - Synthetic data generation can create similar queries
    - Teams reuse examples across different purposes

    **Consequences**:

    ```
    Development Results: 95% routing accuracy
    Production Reality: 60% routing accuracy
    User Experience: Getting few-shot examples as answers (very confusing)
    ```

    **Prevention strategy**:

    1. **Strict data splits**: Create test set first, never let it contaminate few-shot examples
    2. **Diverse synthetic data**: Generate test queries from different prompts than training examples
    3. **Regular auditing**: Check for semantic similarity between test and few-shot examples
    4. **Production validation**: Always validate performance on completely new user queries

### Two-Level Performance Measurement

With routing and specialized retrievers, you need to measure two things: routing accuracy and retrieval accuracy.

!!! tip "For Product Managers"
    **The performance formula**:

    ```
    P(success) = P(right tool | query) x P(finding data | right tool)
    ```

    **Debugging scenarios**:

    | Routing Accuracy | Retrieval Accuracy | Overall | Problem | Solution |
    |-----------------|-------------------|---------|---------|----------|
    | 95% | 40% | 38% | Retrievers need improvement | Fine-tune embeddings, improve chunks |
    | 50% | 90% | 45% | Router makes poor choices | Add few-shot examples, improve descriptions |
    | 70% | 70% | 49% | System-wide issues | May need architecture changes |

    **Strategic framework using extended formula**:

    ```
    P(success) = P(success | right tool) x P(right tool | query) x P(query)
    ```

    Where P(query) represents how often users ask questions your system can handle well. This gives you control over the query distribution through UI design and user education.

    | P(success|tool) | P(tool|query) | Strategy |
    |-----------------|---------------|----------|
    | High | High | Product strengths to highlight |
    | Low | High | Research focus: improve retrievers |
    | High | Low | Router improvement or expose tools directly |
    | Low | Low | Consider deprioritizing |

!!! tip "For Engineers"
    **Implementing two-level measurement**:

    ```python
    from dataclasses import dataclass
    from typing import List, Dict, Set

    @dataclass
    class RoutingMetrics:
        """Metrics for router evaluation."""
        precision: float  # When we select a tool, how often is it correct?
        recall: float     # How often do we select all correct tools?
        f1: float         # Harmonic mean of precision and recall
        per_tool_recall: Dict[str, float]  # Recall broken down by tool

    def evaluate_router(
        router_function,
        test_dataset: List[dict]
    ) -> RoutingMetrics:
        """
        Evaluate router performance on test dataset.

        Args:
            router_function: Function that takes query and returns tool names
            test_dataset: List of {query, expected_tools} pairs

        Returns:
            RoutingMetrics with precision, recall, F1, and per-tool breakdown
        """
        results = []
        tool_expected_count: Dict[str, int] = {}
        tool_correct_count: Dict[str, int] = {}

        for test_case in test_dataset:
            query = test_case["query"]
            expected_tools: Set[str] = set(test_case["expected_tools"])
            selected_tools: Set[str] = set(router_function(query))

            # Track expected tools
            for tool in expected_tools:
                tool_expected_count[tool] = tool_expected_count.get(tool, 0) + 1

            # Calculate metrics for this query
            correct_tools = expected_tools.intersection(selected_tools)
            for tool in correct_tools:
                tool_correct_count[tool] = tool_correct_count.get(tool, 0) + 1

            precision = len(correct_tools) / len(selected_tools) if selected_tools else 1.0
            recall = len(correct_tools) / len(expected_tools) if expected_tools else 1.0

            results.append({"precision": precision, "recall": recall})

        # Calculate overall metrics
        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_recall = sum(r["recall"] for r in results) / len(results)
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        # Calculate per-tool recall
        per_tool_recall = {}
        for tool in tool_expected_count:
            per_tool_recall[tool] = tool_correct_count.get(tool, 0) / tool_expected_count[tool]

        return RoutingMetrics(
            precision=avg_precision,
            recall=avg_recall,
            f1=avg_f1,
            per_tool_recall=per_tool_recall
        )
    ```

    **Confusion matrix analysis**:

    When tool selection fails, understand why with a confusion matrix:

    | Expected\Selected | SearchText | SearchBlueprint | SearchSchedule |
    |-------------------|------------|-----------------|----------------|
    | SearchText        | 85         | 5               | 10             |
    | SearchBlueprint   | 40         | 50              | 10             |
    | SearchSchedule    | 15         | 5               | 80             |

    This shows SearchBlueprint is frequently mistaken for SearchText. Solutions:

    - Add 10-15 specific examples for SearchBlueprint
    - Improve tool description to differentiate from SearchText
    - Create contrast examples showing similar queries going to different tools

### Multi-Agent vs Single-Agent Architectures

Modern RAG systems can use either single agents with multiple tools or multiple specialized agents. Each approach has tradeoffs.

!!! tip "For Product Managers"
    **When to use each approach**:

    | Factor | Single Agent | Multi-Agent |
    |--------|-------------|-------------|
    | Complexity | Lower | Higher |
    | Token efficiency | Lower (full context) | Higher (specialized context) |
    | Debugging | Easier | Harder |
    | Specialization | Limited | High |
    | Safety | Harder to isolate | Read/write separation possible |

    **Recommendation**: Start with single-agent architecture. Only move to multi-agent when you have specific requirements that justify the complexity, such as:

    - Need to separate read and write operations for safety
    - Different tasks require fundamentally different models
    - Token costs are prohibitive with full context

!!! tip "For Engineers"
    **Single-agent with multiple tools** (recommended starting point):

    ```python
    async def process_query(query: str) -> dict:
        """Process query using single agent with multiple tools."""
        # Step 1: Route to appropriate tools
        tools = route_query(query)

        # Step 2: Execute tools in parallel
        results = await asyncio.gather(*[
            tool.execute() for tool in tools
        ])

        # Step 3: Combine results and generate response
        combined_context = combine_results(results)
        response = await generate_response(query, combined_context)

        return {"response": response, "sources": results}
    ```

    **Multi-agent considerations**:

    ```python
    # Example: Separate read and write agents for safety
    class ReadAgent:
        """Agent that can only read data."""
        allowed_tools = [SearchBlueprint, SearchText, SearchSchedule]

    class WriteAgent:
        """Agent that can modify data."""
        allowed_tools = [UpdateDocument, CreateRecord, DeleteRecord]

    async def process_with_safety(query: str, allow_writes: bool = False):
        """Process with read/write separation."""
        # Always start with read agent
        read_results = await read_agent.process(query)

        if allow_writes and requires_write(query):
            # Only invoke write agent with explicit permission
            write_results = await write_agent.process(query, context=read_results)
            return combine(read_results, write_results)

        return read_results
    ```

    **The "bitter lesson" perspective**: As models improve, complex multi-agent orchestration often becomes unnecessary. From Cline's experience with coding agents:

    > "What we're finding now is that actually summarization just works better. In our internal testing, you can have these long-running tasks with resets of the context window many times over with just a very detailed summary."

    Simple approaches often outperform complex engineered solutions as model capabilities improve.

### Latency Analysis and Optimization

Routing adds latency to your system. Understanding where time goes helps optimize the critical path.

!!! tip "For Product Managers"
    **Latency budget allocation**:

    | Component | Typical Latency | Optimization Priority |
    |-----------|----------------|----------------------|
    | Router (LLM) | 200-500ms | High - consider caching |
    | Router (classifier) | 10-50ms | Low |
    | Tool execution | 100-1000ms | Medium - parallelize |
    | Response generation | 500-2000ms | Medium - streaming |

    **User perception**: Users perceive systems as fast when they see progress. Streaming responses and showing intermediate results (like "Searching blueprints...") improve perceived performance even when total latency is unchanged.

!!! tip "For Engineers"
    **Parallel execution pattern**:

    ```python
    import asyncio
    from typing import List

    async def execute_tools_parallel(tools: List[BaseTool]) -> List[ToolResult]:
        """Execute multiple tools in parallel."""
        return await asyncio.gather(*[tool.execute() for tool in tools])

    async def execute_with_timeout(
        tools: List[BaseTool],
        timeout: float = 5.0
    ) -> List[ToolResult]:
        """Execute tools with timeout, returning partial results if needed."""
        tasks = [asyncio.create_task(tool.execute()) for tool in tools]
        try:
            return await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            # Cancel pending tasks and return completed results
            for task in tasks:
                task.cancel()
            return [task.result() for task in tasks if task.done() and not task.cancelled()]
    ```

    **Caching strategies**:

    ```python
    import hashlib

    class CachedRouter:
        """Router with caching for repeated queries."""

        def __init__(self, router, cache_size: int = 1000):
            self.router = router
            self.cache = {}
            self.cache_size = cache_size

        def route(self, query: str) -> List[str]:
            """Route with caching."""
            cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()

            if cache_key in self.cache:
                return self.cache[cache_key]

            result = self.router.route(query)

            # LRU eviction if cache full
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            self.cache[cache_key] = result
            return result
    ```

    **Prompt caching for few-shot examples**:

    With prompt caching (available in OpenAI and Anthropic APIs), large few-shot example sets become economical:

    ```
    Cost Analysis (GPT-4 with prompt caching):
    - 40 examples per tool x 5 tools = 200 examples
    - ~8,000 tokens cached context = $0.0025 per query
    - vs Fine-tuning: $200+ upfront + retraining costs
    - Break-even: ~80,000 queries
    ```

---

## Case Study Deep Dive

### Construction Company: 65% to 78% Overall Success

The construction company from previous chapters provides a complete example of routing implementation and optimization.

!!! tip "For Product Managers"
    **The business context**:

    - 8% of queries were blueprint-related but caused 35% of user churn (from Chapter 4)
    - Three specialized retrievers built: Blueprint (85%), Document (78%), Schedule (82%)
    - Initial monolithic system: 65% overall success

    **The routing journey**:

    | Week | Routing Accuracy | Retrieval Accuracy | Overall Success | Key Change |
    |------|-----------------|-------------------|-----------------|------------|
    | 0 | N/A (monolithic) | 65% | 65% | Baseline |
    | 2 | 88% | 80% | 70% | 10 examples/tool |
    | 4 | 91% | 81% | 74% | Feedback collection |
    | 6 | 95% | 82% | 78% | 40 examples/tool |
    | 12 | 96% | 87% | 84% | Fine-tuned retrievers |

    **ROI calculation**:

    - Investment: 40 hours engineering time + $500 in LLM costs for example generation
    - Return: 13 percentage point improvement in success rate
    - User retention improved 35%
    - Workers actually started using the system daily

!!! tip "For Engineers"
    **Implementation details**:

    **Week 1-2: Basic routing setup**

    ```python
    # Initial tool definitions
    TOOLS = {
        "blueprint": SearchBlueprint,
        "document": SearchText,
        "schedule": SearchSchedule
    }

    # 10 examples per tool
    INITIAL_EXAMPLES = """
    Query: "Find blueprints with 4 bedrooms"
    Tool: blueprint

    Query: "What's the safety procedure for concrete pouring?"
    Tool: document

    Query: "When is the foundation pour?"
    Tool: schedule
    ...
    """
    ```

    **Week 4: Feedback integration**

    ```python
    async def record_routing_feedback(
        query: str,
        selected_tool: str,
        user_satisfied: bool
    ) -> None:
        """Record routing decision and outcome for improvement."""
        await feedback_db.insert({
            "query": query,
            "tool": selected_tool,
            "satisfied": user_satisfied,
            "timestamp": datetime.now()
        })

        # If user was satisfied, consider adding to examples
        if user_satisfied:
            await example_candidates.insert({
                "query": query,
                "tool": selected_tool,
                "confidence": "high"
            })
    ```

    **Week 6: Expanded examples with dynamic selection**

    ```python
    # Expanded to 40 examples per tool based on feedback data
    example_selector = DynamicExampleSelector(embedding_model)

    # Load examples from feedback
    for record in await feedback_db.find({"satisfied": True}):
        example_selector.add_example(
            query=record["query"],
            tool=record["tool"],
            parameters=record.get("parameters", {})
        )

    # Use dynamic selection in routing
    async def route_with_dynamic_examples(query: str) -> List[ToolType]:
        relevant_examples = example_selector.get_examples(query, num_examples=10)
        examples_text = example_selector.format_examples(relevant_examples)
        return route_query(query, examples=examples_text)
    ```

---

## Implementation Guide

### Quick Start for PMs: Evaluating Routing Opportunities

**Step 1: Assess current architecture**

Map your existing RAG system to the migration phases:

1. **Recognition**: Different queries need different retrieval (you are here if using monolithic search)
2. **Separation**: Breaking into specialized components (Chapter 5 work)
3. **Interface**: Defining clear contracts between components (this chapter)
4. **Orchestration**: Building routing layer (this chapter)

**Step 2: Calculate routing ROI**

```
Current success rate: ____%
Estimated routing accuracy achievable: ____%
Estimated retrieval accuracy with specialization: ____%
Projected success rate: routing x retrieval = ____%
Improvement: projected - current = ____% points
```

**Step 3: Plan team organization**

| Role | Responsibility | Headcount |
|------|---------------|-----------|
| Interface design | Tool specifications, contracts | 0.5 FTE |
| Router development | Routing logic, examples | 1 FTE |
| Tool implementation | Specialized retrievers | 1-2 FTE per tool |
| Evaluation | End-to-end testing | 0.5 FTE |

### Detailed Implementation for Engineers

**Step 1: Define tool interfaces**

```python
from pydantic import BaseModel
from typing import List, Optional, Union
from abc import ABC, abstractmethod

# Define all tools with clear descriptions
class SearchBlueprint(BaseModel):
    """
    Search for building plans and blueprints.

    Use when the user asks about:
    - Floor plans, building layouts
    - Room counts, dimensions
    - Architectural drawings
    - Building specifications
    """
    description: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class SearchText(BaseModel):
    """
    Search for text documents.

    Use when the user asks about:
    - Contracts, proposals, bids
    - Safety procedures, policies
    - Specifications, requirements
    - Meeting notes, communications
    """
    query: str
    document_type: Optional[str] = None

class SearchSchedule(BaseModel):
    """
    Search for project schedules and timelines.

    Use when the user asks about:
    - Project timelines, milestones
    - Scheduled activities, deadlines
    - Resource allocation, availability
    """
    project_name: Optional[str] = None
    date_range: Optional[str] = None
```

**Step 2: Implement router with examples**

```python
import instructor
from openai import OpenAI
from typing import Iterable

client = instructor.from_openai(OpenAI())

ToolType = Union[SearchBlueprint, SearchText, SearchSchedule]

def create_router(examples: str):
    """Create a router function with given examples."""

    def route(query: str) -> Iterable[ToolType]:
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are a query router for a construction information system.
                    Analyze queries and select appropriate tools.

                    {examples}
                    """
                },
                {"role": "user", "content": query}
            ],
            response_model=Iterable[ToolType]
        )

    return route
```

**Step 3: Build evaluation pipeline**

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    routing_metrics: RoutingMetrics
    retrieval_metrics: Dict[str, float]
    overall_success: float
    bottleneck: str

async def evaluate_system(
    router,
    tools: Dict[str, BaseTool],
    test_dataset: List[dict]
) -> EvaluationResult:
    """
    Evaluate complete system performance.

    Args:
        router: Router function
        tools: Dictionary of tool implementations
        test_dataset: Test cases with queries, expected tools, and expected results

    Returns:
        Complete evaluation results with bottleneck identification
    """
    # Evaluate routing
    routing_metrics = evaluate_router(router, test_dataset)

    # Evaluate retrieval for correctly routed queries
    retrieval_metrics = {}
    for tool_name, tool in tools.items():
        tool_cases = [t for t in test_dataset if tool_name in t["expected_tools"]]
        if tool_cases:
            retrieval_metrics[tool_name] = await evaluate_tool(tool, tool_cases)

    # Calculate overall success
    avg_retrieval = sum(retrieval_metrics.values()) / len(retrieval_metrics)
    overall_success = routing_metrics.recall * avg_retrieval

    # Identify bottleneck
    if routing_metrics.recall < 0.8 and avg_retrieval > 0.8:
        bottleneck = "routing"
    elif routing_metrics.recall > 0.8 and avg_retrieval < 0.8:
        bottleneck = "retrieval"
    else:
        bottleneck = "both"

    return EvaluationResult(
        routing_metrics=routing_metrics,
        retrieval_metrics=retrieval_metrics,
        overall_success=overall_success,
        bottleneck=bottleneck
    )
```

**Step 4: Implement feedback loop**

```python
class FeedbackLoop:
    """Continuous improvement through user feedback."""

    def __init__(self, example_selector: DynamicExampleSelector):
        self.example_selector = example_selector
        self.feedback_buffer: List[dict] = []

    async def record_interaction(
        self,
        query: str,
        selected_tools: List[str],
        user_feedback: str  # "helpful", "not_helpful", "wrong_tool"
    ) -> None:
        """Record user interaction for learning."""
        self.feedback_buffer.append({
            "query": query,
            "tools": selected_tools,
            "feedback": user_feedback,
            "timestamp": datetime.now()
        })

        # Add successful interactions to examples
        if user_feedback == "helpful":
            for tool in selected_tools:
                self.example_selector.add_example(
                    query=query,
                    tool=tool,
                    parameters={}  # Extract from actual tool call
                )

    async def analyze_failures(self) -> Dict[str, List[str]]:
        """Analyze failure patterns for improvement."""
        failures = [f for f in self.feedback_buffer if f["feedback"] != "helpful"]

        # Group by failure type
        wrong_tool = [f for f in failures if f["feedback"] == "wrong_tool"]
        not_helpful = [f for f in failures if f["feedback"] == "not_helpful"]

        return {
            "wrong_tool_queries": [f["query"] for f in wrong_tool],
            "not_helpful_queries": [f["query"] for f in not_helpful]
        }
```

---

## Common Pitfalls

### PM Pitfalls

!!! warning "PM Pitfall: Optimizing routing before measuring baseline"
    **The mistake**: Investing in routing improvements without knowing current performance breakdown.

    **Why it happens**: The two-level formula is not intuitive; teams assume low performance means bad retrieval.

    **The fix**: Always measure routing accuracy and retrieval accuracy separately before optimizing. You might discover your retrievers are excellent but routing is broken.

!!! warning "PM Pitfall: Over-engineering multi-agent systems"
    **The mistake**: Building complex multi-agent orchestration when single-agent would suffice.

    **Why it happens**: Multi-agent architectures sound sophisticated and promise better specialization.

    **The fix**: Start with single-agent architecture. Only add complexity when you have specific requirements (safety isolation, token cost reduction) that justify it. As models improve, simpler approaches often win.

!!! warning "PM Pitfall: Ignoring the P(query) factor"
    **The mistake**: Focusing only on routing and retrieval accuracy, ignoring that users might not know what to ask.

    **Why it happens**: P(query) is a product/UX concern, not a technical metric.

    **The fix**: Use the extended formula: P(success) = P(success|tool) x P(tool|query) x P(query). If you have great capabilities that users do not discover, invest in UI/UX to increase P(query).

### Engineering Pitfalls

!!! warning "Engineering Pitfall: Data leakage in router evaluation"
    **The mistake**: Using the same queries for few-shot examples and test evaluation.

    **Why it happens**: Small datasets make overlap likely; teams reuse data across purposes.

    **The fix**: Create strict data splits. Generate test queries from different prompts than training examples. Audit for semantic similarity between test and few-shot sets.

!!! warning "Engineering Pitfall: Insufficient examples for rare tools"
    **The mistake**: Having 40 examples for common tools but only 5 for rare ones.

    **Why it happens**: Example collection follows query distribution, leaving rare tools underrepresented.

    **The fix**: Ensure minimum example coverage (10+) for all tools. Generate synthetic examples for rare tools. Monitor per-tool recall to catch underperforming tools.

!!! warning "Engineering Pitfall: Not handling multi-tool queries"
    **The mistake**: Router returns single tool when query needs multiple.

    **Why it happens**: Single-tool examples dominate training data.

    **The fix**: Include explicit multi-tool examples in few-shot set. Design router to return list of tools. Test specifically on queries requiring multiple tools.

!!! warning "Engineering Pitfall: Ignoring router latency"
    **The mistake**: Using expensive LLM routing for every query without caching.

    **Why it happens**: Focus on accuracy over latency during development.

    **The fix**: Implement caching for repeated queries. Consider classifier-based routing for high-volume production. Use prompt caching to reduce LLM routing costs.

---

## Related Content

### Source Materials

This chapter synthesizes content from multiple sources:

- **Workshop Content**: [Chapter 6.1 - Query Routing Foundations](../workshops/chapter6-1.md), [Chapter 6.2 - Tool Interfaces and Implementation](../workshops/chapter6-2.md), [Chapter 6.3 - Performance Measurement](../workshops/chapter6-3.md)
- **Transcript**: [Chapter 6 Lecture Transcript](../workshops/chapter6-transcript.txt) - Contains detailed walkthrough of routing implementation

### Expert Talks

!!! info "Why I Stopped Using RAG for Coding Agents - Nik Pash, Cline"
    **Key insights**:

    - Embedding-based RAG creates unnecessary complexity for coding agents
    - Agentic exploration (reading files, using grep) often outperforms embedding search
    - Simple summarization works better than complex context management
    - The "bitter lesson": as models improve, complex application layers become unnecessary

    [Read the full talk summary](../talks/rag-is-dead-cline-nik.md)

!!! info "Why Grep Beat Embeddings in Our SWE-Bench Agent - Colin Flaherty, Augment"
    **Key insights**:

    - For SWE-Bench tasks, grep and find were sufficient; embedding search was not the bottleneck
    - Agent persistence compensates for less sophisticated tools
    - Best approach combines agentic loops with high-quality embedding models as tools
    - "Vibe-first" evaluation: start with 5-10 examples before quantitative metrics

    [Read the full talk summary](../talks/colin-rag-agents.md)

### Office Hours

!!! info "Cohort 2 Week 6 Summary"
    **Key discussions**:

    - Deep Research as RAG with strong reasoning capabilities
    - Long context windows vs chunking tradeoffs
    - Human-labeled data remains essential for high-quality systems
    - Structured reports provide more business value than ad-hoc answers

    [Read the full summary](../office-hours/cohort2/week6-summary.md)

---

## Action Items

### For Product Teams

1. **Assess current architecture** (Week 1)
   - Map system to migration phases (Recognition, Separation, Interface, Orchestration)
   - Identify which phase you are in
   - Document content types needing different retrieval approaches

2. **Calculate routing ROI** (Week 1)
   - Measure current overall success rate
   - Estimate achievable routing accuracy (typically 85-95%)
   - Calculate projected improvement: routing x retrieval - current
   - Build business case for routing investment

3. **Plan team organization** (Week 2)
   - Define roles for Interface, Implementation, Router, and Evaluation teams
   - Establish clear ownership boundaries
   - Plan coordination mechanisms (APIs, shared metrics)

4. **Design user feedback collection** (Week 2)
   - Plan how to capture routing satisfaction signals
   - Design UI for explicit feedback ("Was this helpful?")
   - Plan feedback-to-example pipeline

### For Engineering Teams

1. **Define tool interfaces** (Week 1)
   - Create Pydantic models for each tool with clear descriptions
   - Document when to use each tool with examples
   - Design parameter extraction patterns

2. **Implement basic router** (Week 1-2)
   - Start with LLM-based routing using Instructor
   - Create 10-20 few-shot examples per tool
   - Implement parallel tool execution

3. **Build evaluation pipeline** (Week 2)
   - Create test dataset with query-tool annotations
   - Implement routing accuracy measurement
   - Build confusion matrix analysis
   - Set up per-tool recall tracking

4. **Implement feedback loop** (Week 3)
   - Record routing decisions and outcomes
   - Build example candidate pipeline from successful interactions
   - Implement dynamic example selection

5. **Optimize for production** (Week 3-4)
   - Add caching for repeated queries
   - Implement prompt caching for few-shot examples
   - Consider classifier migration for high-volume tools
   - Set up latency monitoring

---

## Reflection Questions

1. **What is your current routing accuracy, and how does it compare to your retrieval accuracy?** Use the two-level formula to identify whether routing or retrieval is your bottleneck.

2. **How many few-shot examples do you have per tool, and are rare tools adequately covered?** Consider whether per-tool recall varies significantly across your tool portfolio.

3. **What would happen if you exposed your tools directly to users?** Think about whether a dual-mode interface (chat + direct tool access) would improve user experience.

4. **How would you detect and prevent data leakage in your router evaluation?** Consider the overlap between your few-shot examples and test queries.

5. **When would multi-agent architecture be justified for your use case?** Think about specific requirements (safety, token costs, specialization) that would outweigh the added complexity.

---

## Summary

### Key Takeaways for Product Managers

- **Two-level performance matters**: P(success) = P(right tool) x P(finding data | right tool). Without this breakdown, you cannot tell if problems are routing or retrieval issues.

- **Start simple, evolve with data**: Begin with LLM-based routing and 10-20 examples per tool. Expand to 40+ examples as you collect feedback. Consider classifier migration only when volume justifies it.

- **Team organization enables scale**: The tools-as-APIs pattern lets teams work independently. Interface, Implementation, Router, and Evaluation teams can develop in parallel with clear contracts.

- **User feedback drives improvement**: Successful routing interactions become training examples. Build the feedback collection infrastructure early.

### Key Takeaways for Engineers

- **Three router architectures**: Classifier (fast, needs training data), Embedding (flexible, no labels needed), LLM (highest accuracy, extracts parameters). Start with LLM, migrate as needed.

- **Few-shot examples are critical**: Quality matters as much as quantity. Include edge cases, multi-tool queries, and contrast examples. Prevent data leakage with strict splits.

- **Dynamic example selection improves accuracy**: Retrieve relevant examples based on query similarity rather than using static examples for all queries.

- **Measure both levels**: Track routing accuracy (per-tool recall, confusion matrix) and retrieval accuracy separately. The bottleneck determines where to focus improvement efforts.

---

## Further Reading

### Academic Papers

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Original RAG paper
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - Context position effects

### Tools and Implementations

- [Instructor](https://github.com/jxnl/instructor) - Structured outputs for LLMs
- [LlamaIndex Routers](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/) - Router implementations

### Related Chapters

- [Chapter 5: Specialized Retrieval Systems](chapter5.md) - Building the tools that routing selects between
- [Chapter 7: Production Operations](chapter7.md) - Operating routing systems at scale
- [Chapter 4: Query Understanding](chapter4.md) - Segmentation that informs tool design

---

## Navigation

**Previous**: [Chapter 5: Specialized Retrieval Systems](chapter5.md) - Building specialized retrievers for different content types

**Next**: [Chapter 7: Production Operations](chapter7.md) - Operating RAG systems at scale

**Reference Materials**:

- [Appendix A: Mathematical Foundations](appendix-math.md) - Performance formulas
- [Appendix B: Algorithms Reference](appendix-algorithms.md) - Router algorithm details
