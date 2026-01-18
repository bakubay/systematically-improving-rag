---
title: "Chapter 5: Specialized Retrieval Systems"
description: "Build specialized retrieval systems for different content types. Learn when to extract metadata versus generate synthetic text, implement RAPTOR for long documents, and handle multimodal content including images, tables, and SQL generation."
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - specialized retrieval
  - metadata extraction
  - synthetic text
  - RAPTOR
  - multimodal
  - image search
  - table search
  - SQL generation
---

# Chapter 5: Specialized Retrieval Systems

## Chapter at a Glance

**Prerequisites**: Chapter 1 (evaluation framework), Chapter 4 (query segmentation and prioritization), understanding of embeddings and vector search from Chapter 0

**What You Will Learn**:

- Why specialized retrieval systems outperform monolithic approaches
- Two core improvement strategies: metadata extraction and synthetic text generation
- The RAPTOR algorithm for documents exceeding 1,500 pages
- How to handle multimodal content: images, tables, and SQL generation
- The two-level measurement framework for debugging specialized systems
- Building tool portfolios that work together like Google's specialized search products

**Case Study Reference**: Construction blueprint search (16% to 85% recall improvement), Tax law RAPTOR implementation ($10 processing cost, 85% improvement in finding complete information), Hardware store multi-index system

**Time to Complete**: 75-90 minutes

---

## Key Insight

**Different queries need different retrievers—one-size-fits-all is why most RAG systems underperform.** A search for "SKU-12345" needs exact matching, "compare pricing plans" needs structured comparison, and "how do I reset my password" needs procedural knowledge. Build specialized indices for each pattern and let a router decide. This is how Google evolved: Maps for location, Images for visual, YouTube for video. Your RAG system should follow the same pattern. The two improvement strategies are simple: extract structure from unstructured text, or generate searchable text from structured data. Both create AI-processed views of your data optimized for specific retrieval patterns.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Design specialized retrieval systems** that match different query types to appropriate search strategies, understanding why monolithic approaches fail
2. **Apply the two core improvement strategies**—metadata extraction for structured filtering and synthetic text generation for semantic search—choosing the right approach for each use case
3. **Implement RAPTOR** for long documents where related information spans multiple sections, understanding the cost-benefit tradeoffs
4. **Build multimodal retrieval systems** that handle images, tables, and SQL generation with appropriate techniques for each content type
5. **Use the two-level measurement framework** (P(finding data) = P(selecting retriever) x P(finding data | retriever)) to diagnose and fix system bottlenecks
6. **Design tool portfolios** where multiple specialized tools work together, similar to how command-line tools interact with a file system

---

## Introduction

In Chapter 4, you learned to segment queries and prioritize improvements based on impact, volume, and success probability. That analysis revealed something important: different query types need fundamentally different retrieval approaches. The high-volume, low-satisfaction segments you identified often fail because a single retrieval system cannot handle diverse query patterns.

This chapter shows how to build specialized retrieval systems that excel at specific tasks rather than performing adequately at everything. The pattern mirrors Google's evolution—from one search engine to specialized tools for Maps, Images, YouTube, and Shopping, each using completely different algorithms optimized for their content type.

**Where We Have Been:**

- **Chapter 0**: Introduced embeddings, vector search, and the alignment problem
- **Chapter 1**: Built evaluation frameworks to measure retrieval performance
- **Chapter 2**: Fine-tuned embeddings for domain-specific improvements
- **Chapter 3**: Collected user feedback revealing which queries fail
- **Chapter 4**: Segmented queries to identify patterns needing different approaches

**Now What?** Build specialized indices for each query pattern, then combine them with routing logic (covered in Chapter 6). The work from Chapter 4—identifying segments like "blueprint spatial queries" or "schedule lookups"—directly informs which specialized systems to build.

!!! tip "For Product Managers"
    This chapter establishes the strategic framework for building retrieval capabilities. Focus on understanding when to invest in specialized indices versus improving general search, the ROI calculations for different approaches, and how to organize teams around specialized capabilities. The "materialized view" concept provides a powerful mental model for explaining these investments to stakeholders.

!!! tip "For Engineers"
    This chapter provides concrete implementation patterns for specialized retrieval. Pay attention to the two improvement strategies (metadata extraction vs synthetic text), the RAPTOR algorithm for long documents, and the specific techniques for handling images, tables, and SQL. The code examples demonstrate production-ready patterns you can adapt to your domain.

---

## Core Content

### Why Specialized Retrieval Beats Monolithic Approaches

Most RAG systems start with one big index that tries to handle everything. This works until it doesn't—usually when you realize users ask wildly different types of questions that need different handling.

!!! tip "For Product Managers"
    **The business case for specialization**:

    Consider a hardware store's knowledge base. Users ask three fundamentally different types of questions:

    1. **Exact product lookup**: "Do you have DeWalt DCD771C2 in stock?"
    2. **Conceptual search**: "What's the most durable power drill for heavy construction?"
    3. **Attribute filtering**: "Show me all drills under 5 pounds with at least 18V battery"

    A single embedding-based search handles all three poorly. The product code needs exact matching (lexical search), the durability question needs semantic understanding, and the attribute query needs structured filtering. Trying to solve all three with one approach means solving none of them well.

    **ROI of specialization**: Teams that build specialized indices typically see 25-40% improvement in retrieval accuracy for their target segments. More importantly, they can improve specific capabilities without breaking others—a crucial advantage for sustainable development.

!!! tip "For Engineers"
    **The technical rationale for specialization**:

    The mathematics support specialization: when you have distinct query types, specialized models beat general-purpose ones. This pattern appears throughout machine learning—mixture of experts, task decomposition, modular systems.

    **Google's Evolution as a Model**:

    Google didn't abandon general web search. They built specialized tools and developed routing logic:

    - **Google Maps**: Specialized for locations, routes, geographical queries
    - **Google Images**: Optimized for visual content with computer vision
    - **YouTube**: Built for video with engagement signals and temporal understanding
    - **Google Shopping**: Designed for products with pricing, availability, commerce
    - **Google Scholar**: Tailored for academic papers with citation networks

    Each system uses completely different algorithms, ranking signals, and interfaces optimized for their content. The breakthrough came when they figured out automatic routing—search "pizza near me" and you get Maps; search "how to make pizza" and you get YouTube.

    Apply this pattern to your RAG system. Build specialized retrievers, then route queries to the appropriate one (covered in Chapter 6).

### The Materialized View Concept

Think of specialized indices as **materialized views** of your existing data, but processed by AI rather than traditional SQL operations.

**Traditional Materialized View:**

- SQL precomputes complex joins and aggregations
- Trades storage space for query speed
- Updates when source data changes

**AI Materialized View:**

- AI precomputes structured extractions or synthetic representations
- Trades processing time and storage for retrieval accuracy
- Updates when source documents change or AI models improve

This framing helps you think systematically about what views to create and maintain. You wouldn't create a database materialized view without understanding what queries it optimizes for—the same logic applies to specialized AI indices.

### Two Core Improvement Strategies

When improving retrieval capabilities, two complementary strategies emerge. Think of them as opposite sides of the same coin—one extracting structure from the unstructured, the other creating retrieval-optimized representations of structured data.

#### Strategy 1: Extracting Metadata

Pull structured data out of your text. Instead of treating everything as a blob of text, identify the structured information hiding in there that would make search work better.

!!! tip "For Product Managers"
    **When to use metadata extraction**:

    - Users need to filter by specific attributes (dates, categories, status)
    - Structured information is buried in text
    - Queries involve "show me all X where Y"

    **Business value examples**:

    - **Finance**: Distinguishing fiscal years from calendar years dramatically improves search accuracy for financial metrics
    - **Legal**: Identifying whether contracts are signed or unsigned enables immediate filtering that saves hours of manual review
    - **Support**: Classifying call transcripts by type (job interview, standup, design review) enables type-specific metadata extraction

    **ROI calculation**: If extracting a single metadata field saves 5 minutes per query and you have 1,000 queries per month, that's 83 hours saved monthly. At $50/hour fully loaded cost, that's $4,150/month in value from one extraction.

!!! tip "For Engineers"
    **Implementation pattern**:

    ```python
    from pydantic import BaseModel
    from datetime import date
    from typing import Optional, List

    class FinancialStatement(BaseModel):
        """Structured representation of a financial statement document."""
        company: str
        period_ending: date
        revenue: float
        net_income: float
        earnings_per_share: float
        fiscal_year: bool = True  # Is this fiscal year (vs calendar year)?
        sector: Optional[str] = None
        currency: str = "USD"
        restated: bool = False

    async def extract_financial_data(document_text: str) -> FinancialStatement:
        """
        Extract structured financial data from document text using LLM.

        Args:
            document_text: Raw text from financial document

        Returns:
            Structured FinancialStatement object with extracted data
        """
        system_prompt = """
        Extract the following financial information from the document:
        - Company name
        - Period end date
        - Whether this is a fiscal year report (vs calendar year)
        - Revenue amount (with currency)
        - Net income amount
        - Earnings per share
        - Business sector
        - Whether this statement has been restated

        Format your response as a JSON object with these fields.
        """

        # Use structured outputs for reliable extraction
        result = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": document_text}
            ],
            response_model=FinancialStatement
        )
        return result
    ```

    Once extracted, store in a traditional database (Postgres, etc.) for efficient filtering. Vector search alone cannot handle "show me all tech companies with revenue growth over 10% in fiscal year 2024."

#### Strategy 2: Building Synthetic Text Chunks

Take your data (structured or not) and generate text chunks specifically designed to match how people search. These synthetic chunks act as better search targets that point back to your original content.

!!! tip "For Product Managers"
    **When to use synthetic text generation**:

    - Content is visual or multimedia (images, videos, diagrams)
    - Users search for concepts not in the original text
    - Query patterns are predictable and task-specific

    **Business value examples**:

    - **Image collections**: Generate detailed descriptions capturing searchable aspects
    - **Research interviews**: Extract common questions and answers to form searchable FAQ
    - **Numerical data**: Create natural language descriptions of key trends and outliers
    - **Customer service transcripts**: Create problem-solution pairs capturing resolution patterns

    The synthetic chunks work as a bridge—they're easier to search than your original content but point back to the source when you need full details.

!!! tip "For Engineers"
    **Implementation pattern**:

    ```python
    from pydantic import BaseModel
    from typing import List

    class SyntheticChunk(BaseModel):
        """Synthetic text chunk optimized for retrieval."""
        title: str
        category: str
        summary: str
        entities: List[str]
        source_id: str  # Pointer back to original content

    async def generate_synthetic_chunk(
        document: str,
        task_context: str
    ) -> SyntheticChunk:
        """
        Generate summary optimized for specific retrieval tasks.

        Args:
            document: Source document (image description, transcript, etc.)
            task_context: What users typically query (room counts,
                          pricing info, key dates, etc.)

        Returns:
            Synthetic chunk optimized for retrieval
        """
        prompt = f"""
        Create a searchable summary of this document optimized for these query types:
        {task_context}

        Include:
        - Explicit counts of items users will search for
        - Key dimensions and measurements
        - Important dates and timelines
        - Critical relationships between entities

        The summary should match how users phrase their searches, not how
        the document is written.
        """

        result = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": document}
            ],
            response_model=SyntheticChunk
        )
        return result
    ```

    The key insight: design your summarization prompt based on the specific tasks your system needs to perform. If users search for room counts in blueprints, your summary should explicitly count and list rooms.

### Strategy 3: RAPTOR for Long Documents

When dealing with extremely long documents (1,500-2,000+ pages), traditional chunking strategies fail to capture information that spans multiple sections. The RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) approach offers a sophisticated solution.

!!! tip "For Product Managers"
    **When to use RAPTOR**:

    - Documents where related information is scattered across many pages
    - Content with hierarchical structure (laws/exemptions, rules/exceptions)
    - Long-form documents that don't change frequently (worth the preprocessing cost)
    - Cases where missing related information has high consequences

    **Cost-benefit analysis**:

    | Factor | Value |
    |--------|-------|
    | Upfront cost | $5-20 in LLM calls per document |
    | Processing time | 10-30 minutes per document |
    | Benefit | Dramatically improved recall for cross-document concepts |
    | ROI threshold | Justified for documents accessed frequently or with high-value queries |

    **Real example**: A tax law firm implemented RAPTOR for regulatory documents. Laws appeared on pages 1-30, but exemptions were scattered throughout pages 50-200. Clustering identified related exemptions across sections, and summaries linked laws with all relevant exemptions. One-time processing cost: $10 per document. Result: 85% improvement in finding complete legal information.

!!! tip "For Engineers"
    **The RAPTOR process**:

    1. **Initial Chunking**: Start with page-level or section-level chunks
    2. **Embedding & Clustering**: Embed chunks and cluster semantically similar content
    3. **Hierarchical Summarization**: Create summaries at multiple levels of abstraction
    4. **Tree Structure**: Build a retrieval tree from detailed chunks to high-level summaries

    **Implementation outline**:

    ```python
    from typing import List
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    async def build_raptor_index(chunks: List[str]) -> dict:
        """
        Build RAPTOR hierarchical index from document chunks.

        Args:
            chunks: List of document chunks (page-level or section-level)

        Returns:
            Dictionary containing tree structure with summaries at each level
        """
        # Step 1: Embed all chunks
        embeddings = await embed_chunks(chunks)

        # Step 2: Cluster similar chunks using hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.8,
            linkage='ward'
        )
        cluster_labels = clustering.fit_predict(embeddings)

        # Step 3: Group chunks by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunks[idx])

        # Step 4: Generate summaries for each cluster
        cluster_summaries = {}
        for label, cluster_chunks in clusters.items():
            combined_text = "\n\n".join(cluster_chunks)
            summary = await generate_cluster_summary(combined_text)
            cluster_summaries[label] = summary

        # Step 5: Recursively apply to summaries if needed
        # (for very large documents, create multiple levels)

        return {
            "leaf_chunks": chunks,
            "clusters": clusters,
            "summaries": cluster_summaries,
            "embeddings": embeddings
        }

    async def raptor_retrieve(
        query: str,
        raptor_index: dict,
        top_k: int = 5
    ) -> List[str]:
        """
        Retrieve from RAPTOR index using multi-level search.

        Args:
            query: User query
            raptor_index: Built RAPTOR index
            top_k: Number of results to return

        Returns:
            List of relevant chunks
        """
        query_embedding = await embed_text(query)

        # Search summaries first for broad matching
        summary_scores = []
        for label, summary in raptor_index["summaries"].items():
            summary_embedding = await embed_text(summary)
            score = cosine_similarity(query_embedding, summary_embedding)
            summary_scores.append((label, score))

        # Get top clusters
        top_clusters = sorted(summary_scores, key=lambda x: x[1], reverse=True)[:3]

        # Search within top clusters for specific chunks
        candidate_chunks = []
        for label, _ in top_clusters:
            candidate_chunks.extend(raptor_index["clusters"][label])

        # Re-rank candidates
        final_results = await rerank_chunks(query, candidate_chunks, top_k)
        return final_results
    ```

    **Practical example for construction specifications**:

    ```
    Original Structure:
    - General requirements (pages 1-50)
    - Specific materials (pages 51-300)
    - Installation procedures (pages 301-500)
    - Exceptions and special cases (scattered throughout)

    After RAPTOR Processing:
    - Clustered related materials with their installation procedures
    - Linked all exceptions to their base requirements
    - Created summaries at project, section, and detail levels
    - Reduced average retrieval attempts from 5.2 to 1.3 per query
    ```

    For implementation details, see the [Original RAPTOR paper](https://arxiv.org/abs/2401.18059) and [LlamaIndex RAPTOR implementation](https://docs.llamaindex.ai/en/stable/examples/retrievers/raptor.html).

### Handling Multimodal Content

Different content types require different retrieval approaches. This section covers documents, images, and tables—each with specific techniques that work.

#### Document Search: Beyond Basic Chunking

!!! tip "For Product Managers"
    **Key decisions for document retrieval**:

    1. **Page-level vs arbitrary chunking**: Respect original page boundaries when authors organized content logically
    2. **Write-time vs read-time processing**: Higher storage cost for faster retrieval (usually worth it for production)
    3. **Contextual retrieval investment**: Rewriting chunks with document context improves accuracy but adds processing cost

    **When to invest in contextual retrieval**: Documents where isolated chunks lose meaning, technical documentation with cross-references, legal documents where context determines interpretation.

!!! tip "For Engineers"
    **Page-aware chunking**:

    ```python
    def chunk_by_pages(
        document: str,
        respect_sections: bool = True,
        min_size: int = 200,
        max_size: int = 2000
    ) -> List[str]:
        """
        Chunk document respecting page and section boundaries.

        Args:
            document: Full document text with page markers
            respect_sections: Whether to keep sections intact
            min_size: Minimum chunk size in tokens
            max_size: Maximum chunk size in tokens

        Returns:
            List of chunks with preserved context
        """
        # Implementation respects logical document structure
        # rather than arbitrary token counts
        pass
    ```

    **Contextual chunk rewriting**:

    Original chunk: "Jason the doctor is unhappy with Patient X"

    Without context, this is ambiguous. Is Jason a medical doctor unhappy with a patient? Is a doctor named Jason unhappy? Is someone consulting Dr. Jason about Patient X?

    ```python
    async def create_contextual_chunk(
        chunk: str,
        document_title: str,
        section: str
    ) -> str:
        """Rewrite chunk with document context."""
        prompt = f"""
        Document context: {document_title}
        Section: {section}

        Original chunk: {chunk}

        Rewrite this chunk to include necessary context
        so it can be understood in isolation.
        """
        return await llm.complete(prompt)
    ```

    Result: "In this employee feedback document, Jason (the medical doctor on our staff) expressed dissatisfaction with the Patient X project management software due to frequent crashes."

#### Image Search: Bridging Visual and Textual Understanding

The challenge with image search is that vision models were trained on image captions ("A dog playing in a park"), but users search with queries like "happy pets" or "outdoor activities." There's a fundamental mismatch between training data and search behavior.

!!! tip "For Product Managers"
    **The VLM training gap problem**:

    Vision-Language Models were trained on image-caption pairs from the web:

    - Training data: "A man in a blue shirt standing next to a car"
    - How users search: "professional headshot," "team building activities," "confident leadership pose"

    This mismatch means VLMs excel at generating accurate captions but struggle to understand the conceptual, contextual, and functional language users employ when searching.

    **When to use VLMs vs traditional CV** (from Reducto):

    - **Traditional CV excels at**: Clean structured information, precise bounding boxes, confidence scores, token-efficient processing
    - **VLMs excel at**: Handwriting, charts, figures, diagrams, visually complex layouts

    The most effective approach uses both: traditional CV for initial extraction, VLMs for grading outputs and making corrections.

!!! tip "For Engineers"
    **Rich image description prompt**:

    ```python
    async def generate_rich_image_description(
        image: bytes,
        ocr_text: str = None,
        surrounding_text: str = None
    ) -> str:
        """
        Generate comprehensive description optimized for retrieval.

        Args:
            image: Image data
            ocr_text: Optional text extracted from the image
            surrounding_text: Optional text surrounding image in original context

        Returns:
            Detailed description optimized for search
        """
        prompt = f"""
        # Image Analysis Task

        ## Context Information
        {"OCR Text from image: " + ocr_text if ocr_text else "No OCR text available."}
        {"Surrounding context: " + surrounding_text if surrounding_text else "No surrounding context."}

        ## Analysis Instructions
        Analyze this image in extreme detail:

        1. Describe the visual scene, setting, and overall composition
        2. List all people visible, their positions, actions, and expressions
        3. Enumerate all objects visible in the image
        4. Note any text visible in the image
        5. Describe colors, lighting, and visual style
        6. Identify the type of image (photograph, diagram, screenshot, etc.)
        7. Use chain-of-thought reasoning: what is happening and why
        8. Generate 5-7 potential questions someone might ask when searching for this image
        9. Suggest 5-10 relevant tags

        ## Final Description
        Provide a comprehensive 3-5 sentence description that would help people
        find this image when searching with natural language queries.
        """

        return await vision_model.analyze(image, prompt)
    ```

    **The difference in practice**:

    - Basic prompt ("Describe this image"): "Two people at a table."
    - Better prompt: "Two people arguing across a dinner table in a dimly lit room. One person appears agitated while the other looks defensive. A knife is visible on the table."
    - Optimal prompt: "This dramatic image shows two business professionals in a tense negotiation across a polished conference table in a corporate boardroom with floor-to-ceiling windows overlooking a city skyline. The older man in a gray suit appears frustrated, gesturing emphatically with papers in hand, while the younger woman in a black blazer maintains a composed but firm expression."

    The difference between basic and good descriptions meant 40% better retrieval rates in production systems.

#### Table Search: Structured Data in Context

Tables are structured data living in unstructured documents. They require special handling because they represent two-dimensional associations that can be formatted countless ways.

!!! tip "For Product Managers"
    **Key insight from testing**: Markdown tables work best for LLM lookup:

    | Format | Accuracy |
    |--------|----------|
    | Markdown | 85% |
    | CSV | 73% |
    | JSON | 71% |
    | YAML | 69% |

    The visual structure helps LLMs understand relationships better than nested JSON.

    **Two approaches to table retrieval**:

    1. **Table as Document**: Chunk the table (keep headers!), use semantic search, add summaries. Good for "Which product had the highest Q3 sales?"
    2. **Table as Database**: Treat tables as mini-databases. Create schema descriptions and sample queries, search against those. Good for "Show me all products with margin > 20%."

!!! tip "For Engineers"
    **Table processor implementation**:

    ```python
    from typing import List, Dict, Any, Optional
    import pandas as pd

    class TableProcessor:
        """Process tables for enhanced retrievability and querying."""

        async def process_table(
            self,
            table_data: pd.DataFrame,
            table_name: str,
            source_doc: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Process table for both document-like and database-like retrieval.

            Args:
                table_data: The table as a pandas DataFrame
                table_name: Name of the table
                source_doc: Optional source document information

            Returns:
                Dictionary with processed table components
            """
            # Generate schema representation
            schema = self._generate_schema_representation(table_data, table_name)

            # Generate natural language summary
            summary = await self._generate_table_summary(table_data, table_name)

            # Generate sample queries this table could answer
            sample_queries = await self._generate_sample_queries(table_data, table_name)

            # Convert to markdown for LLM consumption
            markdown = table_data.to_markdown()

            return {
                "table_name": table_name,
                "schema": schema,
                "summary": summary,
                "sample_queries": sample_queries,
                "markdown": markdown,
                "raw_data": table_data,
                "source_document": source_doc
            }

        def _generate_schema_representation(
            self, df: pd.DataFrame, table_name: str
        ) -> str:
            """Generate SQL-like schema representation."""
            types = []
            for col in df.columns:
                dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    sql_type = "NUMERIC"
                elif pd.api.types.is_datetime64_dtype(dtype):
                    sql_type = "TIMESTAMP"
                else:
                    sql_type = "TEXT"

                # Add sample values for better understanding
                sample_values = df[col].dropna().unique()[:3]
                sample_str = f"Sample: {', '.join(str(x) for x in sample_values)}"

                types.append(f"{col} {sql_type} -- {sample_str}")

            return f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(types) + "\n);"
    ```

    **Number formatting warning**: `1 234 567` tokenizes as three separate numbers. Use `1234567` or `1,234,567` instead.

#### SQL Generation: A Case Study in Capability Building

SQL generation demonstrates all these principles in action. You need to find the right tables AND write good queries.

!!! tip "For Product Managers"
    **Why naive text-to-SQL fails**:

    The same question can mean different things. "Show me month-over-month revenue growth":

    - Calendar month or 28-day period?
    - Include weekends or not?
    - Absolute dollars or percentage?
    - All revenue or just recurring?
    - Same day comparison or month-end?
    - What about partial months?

    Models cannot read your mind about business logic. But if you show them examples of how your company calculates these things, they follow that pattern.

    **The approach that works**: Retrieve similar queries from your analytics repository instead of generating from scratch. Accuracy jumps 30% immediately.

!!! tip "For Engineers"
    **What actually works for SQL generation**:

    1. Document all tables with good descriptions and sample data
    2. Generate test questions for different query patterns
    3. Check if you're finding the right tables (precision/recall)
    4. Build a library of good SQL queries that work
    5. Retrieve and include relevant examples when generating new queries

    ```python
    async def generate_sql_with_examples(
        user_query: str,
        table_schemas: List[str],
        example_queries: List[dict]
    ) -> str:
        """
        Generate SQL using retrieved examples.

        Args:
            user_query: Natural language query
            table_schemas: Relevant table CREATE statements
            example_queries: Similar queries with their SQL

        Returns:
            Generated SQL query
        """
        prompt = f"""
        Given these table schemas:
        {chr(10).join(table_schemas)}

        And these example queries from our analytics team:
        {format_examples(example_queries)}

        Generate SQL for this question: {user_query}

        Follow the patterns shown in the examples for date handling,
        aggregations, and joins.
        """

        return await llm.complete(prompt)
    ```

    **Building the query library**: Allow users to "star" SQL statements. This creates valuable training data for few-shot examples. A simple UI feature generates significant long-term value.

### The Two-Level Measurement Framework

With specialized indices, you need to measure two things:

1. Are we selecting the right retrieval method for each query?
2. Is each retrieval method finding the right information?

**The Performance Formula:**

```
P(finding correct data) = P(selecting correct retriever) × P(finding correct data | correct retriever)
```

!!! tip "For Product Managers"
    **Debugging scenarios**:

    | Routing Accuracy | Retrieval Accuracy | Overall | Problem | Solution |
    |-----------------|-------------------|---------|---------|----------|
    | 90% | 40% | 36% | Retrievers need improvement | Fine-tune embeddings, improve chunks |
    | 50% | 90% | 45% | Router makes poor choices | Improve router training, add few-shot examples |
    | 70% | 70% | 49% | System-wide issues | May need architecture changes |

    The key insight: these problems require completely different solutions. Without this breakdown, you waste time optimizing the wrong component.

!!! tip "For Engineers"
    **Implementation**:

    ```python
    @dataclass
    class RetrievalMetrics:
        routing_accuracy: float  # P(correct retriever selected)
        retrieval_accuracy: float  # P(correct data | correct retriever)

        @property
        def overall_accuracy(self) -> float:
            return self.routing_accuracy * self.retrieval_accuracy

        def diagnose(self) -> str:
            if self.routing_accuracy > 0.85 and self.retrieval_accuracy < 0.6:
                return "Focus on improving individual retrievers"
            elif self.routing_accuracy < 0.6 and self.retrieval_accuracy > 0.85:
                return "Focus on improving router accuracy"
            else:
                return "Consider system-wide improvements"
    ```

    Measuring both levels tells you where to focus your efforts.

---

## Case Study Deep Dive

### Construction Blueprint Search: 16% to 85% Recall

A construction information system needed to search architectural blueprints. Users asked questions like "Find blueprints with 4 bedrooms and 2 bathrooms" or "Show me buildings with north-facing windows."

!!! tip "For Product Managers"
    **The business problem**: Workers asked simple spatial questions and got completely unrelated blueprint segments. The system was essentially unusable, causing workers to fall back to manual search through thousands of documents.

    **The timeline**:

    - **Day 0**: Initial attempts using standard image embeddings achieved only 16% recall
    - **Day 1-2**: Created task-specific summaries that explicitly counted rooms, dimensions, key features
    - **Day 3-4**: Implemented separate "search summaries" tool that only queries the summary index
    - **Day 4**: Recall improved to 85%—a 69 percentage point improvement

    **The investment**: Approximately $10 in LLM calls per document for summarization. For a corpus of 1,000 blueprints, total processing cost was $10,000. The ROI was immediate—workers could now find relevant blueprints in seconds instead of hours.

!!! tip "For Engineers"
    **Why standard embeddings failed**: Vision models aren't trained for spatial search. CLIP embeddings understand "this looks like a blueprint" but not "this blueprint has 4 bedrooms."

    **The solution**: Task-specific summaries that anticipated user queries.

    ```python
    BLUEPRINT_SUMMARY_PROMPT = """
    Analyze this architectural blueprint and extract:

    1. ROOM COUNTS: Count all rooms by type (bedrooms, bathrooms, etc.)
    2. DIMENSIONS: List key dimensions (total square footage, room sizes)
    3. ORIENTATION: Identify building orientation (north-facing windows, etc.)
    4. KEY FEATURES: Note architectural features users might search for

    Format as a searchable summary that matches how construction workers
    phrase their queries.
    """
    ```

    **Key insight**: The summary prompt anticipated user mental models. Instead of describing what the blueprint looked like, it extracted what users actually searched for.

### Tax Law RAPTOR Implementation

A tax law firm needed to search regulatory documents where laws appeared in one section but exemptions were scattered throughout.

!!! tip "For Product Managers"
    **The challenge**: A 2,000-page regulatory document had laws on pages 1-30, but relevant exemptions appeared on pages 50, 127, 189, and 245. Traditional chunking meant users found the law but missed critical exemptions.

    **The solution**: RAPTOR clustering identified related exemptions across sections, then created summaries linking laws with all relevant exemptions.

    **Results**:

    - Processing cost: $10 per document
    - Processing time: 25 minutes per document
    - Improvement: 85% better at finding complete legal information
    - ROI: Justified by high-stakes nature of legal queries where missing an exemption could cost millions

!!! tip "For Engineers"
    **Implementation approach**:

    1. Chunked at section level (not page level) to preserve legal structure
    2. Used hierarchical clustering with ward linkage
    3. Generated summaries that explicitly linked laws to their exemptions
    4. Built three-level index: document summary → section summaries → original chunks

    The key was designing cluster summaries that preserved legal relationships, not just semantic similarity.

---

## Implementation Guide

### Quick Start for PMs: Evaluating Specialization Opportunities

**Step 1: Audit current system performance by query type**

Use the segmentation from Chapter 4 to identify which query types perform poorly:

- Which segments have lowest satisfaction scores?
- Which segments have highest volume?
- Calculate: Impact = Volume × (1 - Current Satisfaction)

**Step 2: Classify the problem type**

For each underperforming segment, determine:

- **Inventory problem**: The information exists but isn't in your index
- **Capability problem**: The information is indexed but retrieval fails

**Step 3: Choose improvement strategy**

| Problem Type | Strategy | Example |
|--------------|----------|---------|
| Users need filtering | Metadata extraction | "Show me contracts from Q3" |
| Content is visual | Synthetic text | Blueprint room counts |
| Information scattered | RAPTOR | Legal exemptions across sections |
| Exact matching needed | Lexical search | Product SKUs |

**Step 4: Calculate ROI**

```
Investment = Processing cost + Engineering time
Return = (Queries/month) × (Time saved/query) × (Hourly cost)
Payback period = Investment / Return
```

### Detailed Implementation for Engineers

**Step 1: Set up evaluation infrastructure**

Before building specialized indices, establish how you'll measure success:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class SpecializedIndexEval:
    """Evaluation framework for specialized indices."""
    index_name: str
    test_queries: List[str]
    expected_results: List[List[str]]

    async def evaluate(self, retriever) -> dict:
        """Run evaluation and return metrics."""
        results = {
            "precision": [],
            "recall": [],
            "mrr": []
        }

        for query, expected in zip(self.test_queries, self.expected_results):
            retrieved = await retriever.search(query, k=10)
            results["precision"].append(
                len(set(retrieved) & set(expected)) / len(retrieved)
            )
            results["recall"].append(
                len(set(retrieved) & set(expected)) / len(expected)
            )
            # Calculate MRR
            for i, doc in enumerate(retrieved):
                if doc in expected:
                    results["mrr"].append(1 / (i + 1))
                    break
            else:
                results["mrr"].append(0)

        return {k: sum(v) / len(v) for k, v in results.items()}
```

**Step 2: Implement metadata extraction pipeline**

```python
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class MetadataExtractor:
    """Generic metadata extraction pipeline."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    async def extract(
        self,
        document: str,
        schema: Type[T],
        extraction_prompt: str
    ) -> T:
        """
        Extract structured metadata from document.

        Args:
            document: Source document text
            schema: Pydantic model defining extraction schema
            extraction_prompt: Task-specific extraction instructions

        Returns:
            Extracted metadata as Pydantic model instance
        """
        result = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": document}
            ],
            response_model=schema
        )
        return result

    async def batch_extract(
        self,
        documents: List[str],
        schema: Type[T],
        extraction_prompt: str,
        concurrency: int = 10
    ) -> List[T]:
        """Extract metadata from multiple documents concurrently."""
        semaphore = asyncio.Semaphore(concurrency)

        async def extract_with_limit(doc):
            async with semaphore:
                return await self.extract(doc, schema, extraction_prompt)

        return await asyncio.gather(*[
            extract_with_limit(doc) for doc in documents
        ])
```

**Step 3: Implement synthetic text generation**

```python
class SyntheticTextGenerator:
    """Generate retrieval-optimized synthetic text."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    async def generate_summary(
        self,
        content: str,
        task_context: str,
        query_examples: List[str] = None
    ) -> str:
        """
        Generate summary optimized for specific query patterns.

        Args:
            content: Source content (text, image description, etc.)
            task_context: Description of what users typically search for
            query_examples: Optional example queries to optimize for

        Returns:
            Retrieval-optimized summary
        """
        examples_section = ""
        if query_examples:
            examples_section = f"""
            Example queries users might ask:
            {chr(10).join(f'- {q}' for q in query_examples)}
            """

        prompt = f"""
        Create a searchable summary optimized for these query types:
        {task_context}

        {examples_section}

        The summary should:
        1. Use language that matches how users phrase searches
        2. Include explicit counts and measurements
        3. Highlight key entities and relationships
        4. Be self-contained (understandable without original context)
        """

        result = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content}
            ]
        )
        return result.choices[0].message.content
```

**Step 4: Build tool portfolio**

```python
from pydantic import BaseModel
from typing import List, Union
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())

# Define specialized search tools
class DocumentSearch(BaseModel):
    """Search through text documents and manuals."""
    query: str

class ImageSearch(BaseModel):
    """Search through images and visual content."""
    query: str

class TableSearch(BaseModel):
    """Search through structured data and tables."""
    query: str

class SQLQuery(BaseModel):
    """Query structured databases with SQL."""
    query: str

async def route_query(user_query: str) -> List[BaseModel]:
    """Route query to appropriate retrieval tools."""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a query router. Analyze the query and
                decide which retrieval tools to use. You can call multiple
                tools if needed.

                Available tools:
                - DocumentSearch: For procedures, policies, text content
                - ImageSearch: For visual content, diagrams, photos
                - TableSearch: For data, comparisons, structured info
                - SQLQuery: For specific data queries requiring SQL
                """
            },
            {"role": "user", "content": user_query}
        ],
        response_model=List[Union[DocumentSearch, ImageSearch, TableSearch, SQLQuery]]
    )
```

---

## Common Pitfalls

### PM Pitfalls

!!! warning "PM Pitfall: Building specialized indices without measuring baseline"
    **The mistake**: Investing in specialized retrieval without knowing current performance by query type.

    **Why it happens**: Excitement about new capabilities overshadows measurement discipline.

    **The fix**: Always measure baseline performance for target query segments before building. Use the segmentation from Chapter 4 to establish clear before/after metrics.

!!! warning "PM Pitfall: Over-specializing too early"
    **The mistake**: Building many specialized indices before validating the approach works.

    **Why it happens**: The "Google has many specialized tools" analogy is compelling but misleading for early-stage systems.

    **The fix**: Start with one specialized index for your highest-impact segment. Prove the approach works, then expand. Most systems need 3-5 specialized indices, not 20.

!!! warning "PM Pitfall: Ignoring maintenance costs"
    **The mistake**: Calculating ROI based only on initial processing costs, ignoring ongoing maintenance.

    **Why it happens**: Specialized indices need updates when source data changes or models improve.

    **The fix**: Include maintenance costs in ROI calculations. Budget for periodic reprocessing (typically quarterly for active documents).

### Engineering Pitfalls

!!! warning "Engineering Pitfall: Forcing everything through text embeddings"
    **The mistake**: Stringifying JSON, numbers, and structured data to feed to text embedding models.

    **Why it happens**: Text embeddings are familiar and "work for everything."

    **The fix**: Use appropriate representations for each data type. Numbers need numerical comparisons, locations need geo-distance, categories need exact matching. As Daniel from Superlinked explains: "Text embedding models understand numbers through co-occurrence in training data, not as actual numerical values."

!!! warning "Engineering Pitfall: Relying on re-ranking to fix poor retrieval"
    **The mistake**: Using re-ranking to compensate for inadequate initial retrieval.

    **Why it happens**: Re-ranking is easier to implement than improving base retrieval.

    **The fix**: Re-ranking only applies to candidates you retrieved. If the right document isn't in your top 100 candidates, no amount of re-ranking will find it. Fix the underlying retrieval first.

!!! warning "Engineering Pitfall: Using Boolean filters instead of biases"
    **The mistake**: Implementing "near Manhattan" as a hard distance cutoff instead of a smooth preference.

    **Why it happens**: Boolean filters are simpler to implement.

    **The fix**: Most user preferences are gradual, not binary. Use smooth bias functions that gradually decrease relevance with distance rather than hard cutoffs that exclude potentially excellent results.

!!! warning "Engineering Pitfall: Extracting multiple attributes in one API call"
    **The mistake**: Extracting summaries, action items, categories, and entities in a single prompt.

    **Why it happens**: Seems more efficient to do everything at once.

    **The fix**: Prompts for some attributes affect extraction of others. When we asked for shorter action items, summaries also got shorter. Split extractions into separate jobs with specific focuses. Use prompt caching to maintain cost efficiency.

---

## Related Content

### Expert Talks

!!! info "Lexical Search in RAG Applications - John Berryman"
    **Key insights**:

    - Semantic search struggles with exact matches for product IDs, names, and specific phrases
    - Lexical search excels at filtering and can process filtering and relevance scoring simultaneously
    - The future belongs to hybrid approaches combining lexical filtering with semantic understanding

    [Read the full talk summary](../talks/john-lexical-search.md)

!!! info "Why Most Document Parsing Sucks - Adit, Reducto"
    **Key insights**:

    - VLMs excel at handwriting, charts, and figures—but traditional CV provides better precision for structured content
    - Tables are particularly challenging because they represent two-dimensional associations
    - Create separate representations optimized for embedding models versus LLMs

    [Read the full talk summary](../talks/reducto-docs-adit.md)

!!! info "Encoder Stacking and Multi-Modal Retrieval - Daniel, Superlinked"
    **Key insights**:

    - Text embedding models fundamentally cannot understand numerical relationships
    - Use specialized encoders for different data types (numerical, location, categorical)
    - Boolean filters are poor approximations of user preferences—use smooth biases instead

    [Read the full talk summary](../talks/superlinked-encoder-stacking.md)

### Office Hours

!!! info "Cohort 2 Week 5 Summary"
    **Key discussions**:

    - Excel file handling with multiple sheets and tables
    - SQL generation techniques using Claude Sonnet
    - Linear adapters for cost-effective embedding fine-tuning
    - Partitioning strategies in retrieval systems

    [Read the full summary](../office-hours/cohort2/week5-summary.md)

!!! info "Cohort 3 Week 5 Office Hours"
    **Key discussions**:

    - Fine-tuning for citation accuracy (4% to 0% error rate)
    - Tool portfolio design for specialized retrieval
    - Temporal reasoning in medical data
    - Document summarization improving recall from 16% to 85%

    [Read Office Hour 1](../office-hours/cohort3/week-5-1.md) | [Read Office Hour 2](../office-hours/cohort3/week-5-2.md)

---

## Action Items

### For Product Teams

1. **Audit current retrieval by query type** (Week 1)
   - Use Chapter 4 segmentation to identify underperforming segments
   - Calculate impact scores: Volume × (1 - Satisfaction)
   - Prioritize top 3 segments for specialization

2. **Classify problems and choose strategies** (Week 1)
   - For each segment, determine: inventory or capability problem?
   - Match problem type to improvement strategy
   - Calculate ROI for top candidates

3. **Plan team organization** (Week 2)
   - Consider organizing teams around specialized capabilities
   - Document team: PDF processing, contextual retrieval
   - Vision team: Image description, OCR enhancement
   - Structured data team: Table processing, SQL generation

4. **Establish success metrics** (Week 2)
   - Define target improvements for each specialized index
   - Set up two-level measurement (routing + retrieval accuracy)
   - Plan A/B testing approach for production validation

### For Engineering Teams

1. **Set up evaluation infrastructure** (Week 1)
   - Create test sets for target query segments
   - Implement precision/recall/MRR measurement
   - Establish baseline metrics before building

2. **Build first specialized index** (Week 1-2)
   - Choose highest-impact segment from PM prioritization
   - Implement appropriate strategy (metadata extraction, synthetic text, or RAPTOR)
   - Measure improvement against baseline

3. **Implement multimodal handling** (Week 2-3)
   - For images: Rich description prompts with chain-of-thought
   - For tables: Markdown format with schema descriptions
   - For SQL: Query library with business-specific examples

4. **Build tool portfolio** (Week 3-4)
   - Implement routing logic for specialized tools
   - Set up parallel execution for multi-tool queries
   - Implement result combination (short-term: concat + rerank)

---

## Reflection Questions

1. **For your current RAG system, which query types would benefit most from specialized retrieval?** Consider the segmentation from Chapter 4—which high-volume, low-satisfaction segments could be addressed with metadata extraction, synthetic text, or RAPTOR?

2. **How would you explain the "materialized view" concept to a non-technical stakeholder?** Practice articulating why specialized indices are investments that pay off through improved retrieval, not just additional infrastructure costs.

3. **What's the difference between an inventory problem and a capability problem in your domain?** Think of specific examples where the information exists but isn't indexed versus where it's indexed but retrieval fails.

4. **If you could only build one specialized index, which would have the highest ROI?** Use the formula: ROI = (Queries/month × Time saved × Hourly cost) / (Processing cost + Engineering time).

5. **How would you measure success for a specialized retrieval system?** Consider both the routing accuracy (selecting the right retriever) and retrieval accuracy (finding the right data given the right retriever).

---

## Summary

### Key Takeaways for Product Managers

- **Specialization beats monolithic approaches**: Different query types need different retrieval strategies. Build specialized indices for each pattern, then route queries appropriately.

- **Two improvement strategies**: Extract structure from unstructured text (metadata extraction) or generate searchable text from structured data (synthetic text). Both create AI-processed views optimized for specific retrieval patterns.

- **ROI calculation matters**: Specialized indices require investment. Calculate: Investment = Processing cost + Engineering time; Return = Queries × Time saved × Hourly cost. Most high-impact segments justify the investment.

- **Organize teams around capabilities**: As you build multiple specialized indices, consider organizing teams around content types (documents, images, structured data) rather than features.

### Key Takeaways for Engineers

- **Match data types to appropriate representations**: Don't force everything through text embeddings. Numbers need numerical comparisons, locations need geo-distance, categories need exact matching.

- **RAPTOR for long documents**: When information spans multiple sections in documents over 1,500 pages, use hierarchical clustering and summarization to link related content.

- **Two-level measurement is essential**: P(finding data) = P(selecting retriever) × P(finding data | retriever). This formula tells you whether to improve routing or individual retrievers.

- **Build tool portfolios, not mega-tools**: Multiple specialized tools that work together (like command-line tools with a file system) outperform single tools trying to do everything.

---

## Further Reading

### Academic Papers

- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) - The original RAPTOR paper with full algorithm details
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - Why position matters in retrieved context

### Tools and Implementations

- [LlamaIndex RAPTOR Implementation](https://docs.llamaindex.ai/en/stable/examples/retrievers/raptor.html) - Production-ready RAPTOR implementation
- [Superlinked](https://superlinked.com/) - Framework for encoder stacking and multi-modal retrieval
- [Reducto](https://reducto.ai/) - Document parsing with hybrid CV + VLM approach

### Related Chapters

- [Chapter 0: Introduction](chapter0.md) - Foundational concepts including embeddings and vector search
- [Chapter 4: Query Understanding](chapter4.md) - Segmentation that identifies which specialized indices to build
- [Chapter 6: Query Routing](chapter6.md) - How to route queries to appropriate specialized retrievers

---

## Navigation

- **Previous**: [Chapter 4: Query Understanding and Prioritization](chapter4.md) - Segmentation and prioritization frameworks
- **Next**: [Chapter 6: Query Routing and Orchestration](chapter6.md) - Combining specialized retrievers with intelligent routing
- **Reference**: [Glossary](glossary.md) | [Quick Reference](quick-reference.md)
- **Book Index**: [Book Overview](index.md)
