---
title: "Chapter 9: Context Window Management"
description: "Master context window management for better RAG generation. Learn about the 'Lost in the Middle' problem, token budgeting strategies, dynamic context assembly, and mitigation techniques for optimal LLM performance."
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - context-window
  - lost-in-the-middle
  - token-budgeting
  - context-assembly
  - rag-generation
  - llm-optimization
---

# Chapter 9: Context Window Management

## Chapter at a Glance

**Prerequisites**: Chapter 1 (evaluation framework), Chapter 5 (specialized retrieval), Chapter 6 (query routing), familiarity with LLM token limits and pricing

**What You Will Learn**:

- Why the "Lost in the Middle" problem causes LLMs to miss critical information in long contexts
- How to budget tokens effectively across different context components
- Strategies for dynamic context assembly based on query complexity
- Mitigation techniques including monologuing, reordering, and summarization
- When to use long context windows vs chunked retrieval
- How to evaluate and measure context management effectiveness

**Case Study Reference**: Insurance claims processing improved answer accuracy from 67% to 89% by implementing context reordering and monologuing techniques

**Time to Complete**: 50-65 minutes

---

## Key Insight

**Bigger context windows do not automatically mean better answers—how you organize and present information matters as much as what you include.** The "Lost in the Middle" phenomenon shows that LLMs struggle to attend to information in the middle of long contexts, preferring content at the beginning and end. Effective context management involves strategic ordering, token budgeting, and techniques like monologuing that help models "re-read" important information. As context windows grow from 4K to 200K+ tokens, the challenge shifts from fitting information to organizing it effectively.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand the Lost in the Middle problem** and its impact on RAG system accuracy
2. **Implement token budgeting strategies** that allocate context space effectively across system prompts, retrieved documents, and few-shot examples
3. **Design dynamic context assembly** that adapts to query complexity and available information
4. **Apply mitigation techniques** including monologuing, strategic reordering, and compression
5. **Choose between long context and chunked retrieval** based on use case characteristics
6. **Evaluate context management effectiveness** using targeted metrics

---

## Introduction

Previous chapters focused on retrieving the right information. This chapter addresses what happens after retrieval: how to present that information to the LLM for optimal generation quality.

**The Context Management Challenge**:

You have built a system with:

- Evaluation framework (Chapter 1) measuring retrieval quality
- Fine-tuned embeddings (Chapter 2) improving recall
- Feedback collection (Chapter 3) driving continuous improvement
- Query routing (Chapter 6) directing queries to specialized retrievers
- Specialized retrieval (Chapter 5) handling diverse content types

But retrieval is only half the problem. Even with perfect recall, the LLM must correctly process and reason over the retrieved information. This is where context management becomes critical.

**Why Context Management Matters**:

Consider a legal research system that retrieves 15 relevant contract clauses for a query about termination rights. Even if all 15 clauses are relevant, the LLM might:

- Focus primarily on the first few clauses (recency bias)
- Miss critical exceptions buried in the middle
- Produce inconsistent answers depending on clause ordering
- Exceed token limits when including full clause text

These problems compound as context windows grow. A 200K token context window does not solve the problem—it creates new challenges around attention allocation and information organization.

!!! tip "For Product Managers"
    Context management directly impacts answer quality and user trust. A system that retrieves the right information but presents it poorly will still produce incorrect answers. The business impact includes:
    
    - **Accuracy**: Poor context management can reduce accuracy by 15-30% even with perfect retrieval
    - **Consistency**: Users lose trust when identical queries produce different answers based on document ordering
    - **Cost**: Inefficient context use wastes tokens and increases latency
    
    Investment in context management typically yields 10-20% accuracy improvements with minimal infrastructure changes.

!!! tip "For Engineers"
    This chapter covers both the theory and implementation of context management. Pay attention to:
    
    - The Lost in the Middle research and its implications for context ordering
    - Token budgeting algorithms for different context components
    - Monologuing as a technique for improving reasoning over long contexts
    - Evaluation methods for measuring context management effectiveness

---

## Core Content

### The Lost in the Middle Problem

Research from Stanford and UC Berkeley demonstrated that LLMs struggle to use information in the middle of long contexts, even when that information is directly relevant to the query.

!!! tip "For Product Managers"
    **What the research found**:
    
    In experiments where researchers placed relevant information at different positions in a long context:
    
    - Information at the **beginning**: 75-80% accuracy
    - Information in the **middle**: 45-55% accuracy  
    - Information at the **end**: 70-75% accuracy
    
    This U-shaped curve means that simply including relevant information is not enough—where you place it matters significantly.
    
    **Business implications**:
    
    | Scenario | Risk | Mitigation |
    |----------|------|------------|
    | Legal document review | Critical clauses in middle sections missed | Reorder by relevance, use monologuing |
    | Customer support | Key troubleshooting steps skipped | Place most relevant steps first |
    | Financial analysis | Important caveats overlooked | Summarize key points at beginning |
    | Medical records | Relevant history buried in timeline | Extract and highlight key events |
    
    **Decision framework**: If your use case involves contexts longer than 4K tokens with information distributed throughout, you need explicit context management strategies.

!!! tip "For Engineers"
    **Technical explanation**:
    
    The Lost in the Middle phenomenon stems from how transformer attention mechanisms work:
    
    1. **Attention decay**: Self-attention weights naturally decay with distance, making it harder to attend to middle positions
    2. **Position encoding limitations**: Absolute position encodings can make middle positions less distinguishable
    3. **Training data bias**: Models are often trained on data where important information appears at the beginning or end
    
    **Quantifying the problem**:
    
    ```python
    from dataclasses import dataclass
    from typing import Literal
    
    @dataclass
    class PositionExperiment:
        """Track accuracy by information position."""
        position: Literal["beginning", "middle", "end"]
        context_length: int
        accuracy: float
        
    # Typical results from Lost in the Middle experiments
    experiments = [
        PositionExperiment("beginning", 4000, 0.78),
        PositionExperiment("middle", 4000, 0.52),
        PositionExperiment("end", 4000, 0.73),
        PositionExperiment("beginning", 16000, 0.71),
        PositionExperiment("middle", 16000, 0.43),
        PositionExperiment("end", 16000, 0.68),
    ]
    
    # The accuracy drop in the middle is consistent across context lengths
    # and becomes more pronounced with longer contexts
    ```
    
    **Model improvements**:
    
    Newer models have improved attention mechanisms that reduce (but do not eliminate) this problem:
    
    - GPT-4 Turbo and Claude 3 show flatter curves than earlier models
    - Models with extended context (100K+) often use techniques like sliding window attention
    - Fine-tuning on long-context tasks can improve middle-position recall
    
    However, even with improvements, the fundamental challenge remains: attention is a limited resource that must be allocated across the entire context.

### Token Budgeting Strategies

Effective context management requires explicit budgeting of available tokens across different components.

!!! tip "For Product Managers"
    **Why token budgeting matters**:
    
    A typical RAG prompt has several components competing for limited context space:
    
    | Component | Purpose | Typical Allocation |
    |-----------|---------|-------------------|
    | System prompt | Instructions, persona, constraints | 500-2,000 tokens |
    | Few-shot examples | Demonstrate expected behavior | 1,000-4,000 tokens |
    | Retrieved documents | Source information for answer | 2,000-8,000 tokens |
    | Query + history | User question and conversation context | 500-2,000 tokens |
    | Reserved for output | Space for model response | 500-2,000 tokens |
    
    **Cost implications**:
    
    Token budgeting directly affects costs:
    
    - **Over-allocation to documents**: Higher input costs, potentially lower accuracy due to Lost in the Middle
    - **Under-allocation to examples**: Lower quality outputs, more hallucinations
    - **No output reservation**: Truncated responses, incomplete answers
    
    **Decision framework**:
    
    1. Start with output reservation (never compromise this)
    2. Allocate system prompt based on task complexity
    3. Budget few-shot examples based on task novelty
    4. Fill remaining space with retrieved documents, ordered by relevance

!!! tip "For Engineers"
    **Implementation pattern**:
    
    ```python
    from dataclasses import dataclass, field
    import tiktoken
    
    @dataclass
    class TokenBudget:
        """Manage token allocation across context components."""
        
        total_limit: int = 16000  # Model's context window
        output_reserve: int = 2000  # Reserved for generation
        system_prompt_limit: int = 1500
        few_shot_limit: int = 3000
        query_limit: int = 500
        
        @property
        def document_budget(self) -> int:
            """Calculate remaining budget for retrieved documents."""
            used = (
                self.output_reserve + 
                self.system_prompt_limit + 
                self.few_shot_limit + 
                self.query_limit
            )
            return max(0, self.total_limit - used)
    
    
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """Count tokens in text for a specific model."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    
    def allocate_documents(
        documents: list[str],
        budget: TokenBudget,
        model: str = "gpt-4"
    ) -> list[str]:
        """
        Select documents that fit within token budget.
        
        Documents are assumed to be ordered by relevance (most relevant first).
        """
        selected = []
        remaining_tokens = budget.document_budget
        
        for doc in documents:
            doc_tokens = count_tokens(doc, model)
            if doc_tokens <= remaining_tokens:
                selected.append(doc)
                remaining_tokens -= doc_tokens
            else:
                # Try to include a truncated version
                if remaining_tokens > 100:  # Minimum useful length
                    truncated = truncate_to_tokens(doc, remaining_tokens, model)
                    selected.append(truncated)
                break
        
        return selected
    
    
    def truncate_to_tokens(
        text: str, 
        max_tokens: int, 
        model: str = "gpt-4"
    ) -> str:
        """Truncate text to fit within token limit."""
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens) + "..."
    ```
    
    **Dynamic budget adjustment**:
    
    ```python
    def adjust_budget_for_query(
        query: str,
        base_budget: TokenBudget,
        query_complexity: float  # 0.0 to 1.0
    ) -> TokenBudget:
        """
        Adjust token budget based on query complexity.
        
        Complex queries need more few-shot examples and document context.
        Simple queries can use smaller budgets for faster, cheaper responses.
        """
        if query_complexity < 0.3:
            # Simple query: reduce allocations
            return TokenBudget(
                total_limit=base_budget.total_limit,
                output_reserve=1000,
                system_prompt_limit=500,
                few_shot_limit=1000,
                query_limit=300
            )
        elif query_complexity > 0.7:
            # Complex query: maximize allocations
            return TokenBudget(
                total_limit=base_budget.total_limit,
                output_reserve=3000,
                system_prompt_limit=2000,
                few_shot_limit=4000,
                query_limit=1000
            )
        else:
            return base_budget
    ```

### Dynamic Context Assembly

Context assembly should adapt to the specific query and available information rather than using a fixed template.

!!! tip "For Product Managers"
    **Why dynamic assembly matters**:
    
    Different queries have different context needs:
    
    | Query Type | Context Priority | Example |
    |------------|------------------|---------|
    | Factual lookup | Retrieved documents | "What is the return policy?" |
    | Comparison | Multiple documents, structured format | "Compare Plan A vs Plan B" |
    | Reasoning | Few-shot examples, step-by-step | "Should I choose option X or Y?" |
    | Creative | System prompt, examples | "Write a summary of..." |
    
    **Business value**:
    
    - **Faster responses**: Simple queries use smaller contexts
    - **Lower costs**: Token usage matches query complexity
    - **Higher accuracy**: Context structure matches task requirements
    
    **Implementation approach**:
    
    1. Classify incoming queries by type
    2. Select appropriate context template
    3. Populate template with relevant components
    4. Validate token budget before sending

!!! tip "For Engineers"
    **Context assembly patterns**:
    
    ```python
    from enum import Enum
    from dataclasses import dataclass
    
    class QueryType(Enum):
        FACTUAL = "factual"
        COMPARISON = "comparison"
        REASONING = "reasoning"
        CREATIVE = "creative"
        MULTI_HOP = "multi_hop"
    
    
    @dataclass
    class ContextTemplate:
        """Define context structure for a query type."""
        query_type: QueryType
        system_prompt: str
        include_few_shot: bool
        few_shot_count: int
        document_ordering: str  # "relevance", "chronological", "reverse_relevance"
        include_metadata: bool
        summarize_documents: bool
    
    
    TEMPLATES = {
        QueryType.FACTUAL: ContextTemplate(
            query_type=QueryType.FACTUAL,
            system_prompt="Answer the question using only the provided documents.",
            include_few_shot=False,
            few_shot_count=0,
            document_ordering="relevance",
            include_metadata=True,
            summarize_documents=False
        ),
        QueryType.COMPARISON: ContextTemplate(
            query_type=QueryType.COMPARISON,
            system_prompt="""Compare the items using the provided documents.
            Structure your response with clear sections for each item.""",
            include_few_shot=True,
            few_shot_count=2,
            document_ordering="relevance",
            include_metadata=True,
            summarize_documents=False
        ),
        QueryType.REASONING: ContextTemplate(
            query_type=QueryType.REASONING,
            system_prompt="""Think through this problem step by step.
            Consider multiple perspectives before reaching a conclusion.""",
            include_few_shot=True,
            few_shot_count=3,
            document_ordering="relevance",
            include_metadata=True,
            summarize_documents=False
        ),
        QueryType.MULTI_HOP: ContextTemplate(
            query_type=QueryType.MULTI_HOP,
            system_prompt="""This question requires combining information from 
            multiple sources. First identify the relevant pieces of information,
            then synthesize them into a complete answer.""",
            include_few_shot=True,
            few_shot_count=2,
            document_ordering="relevance",
            include_metadata=True,
            summarize_documents=True  # Compress to fit more sources
        )
    }
    
    
    def assemble_context(
        query: str,
        query_type: QueryType,
        documents: list[dict],
        few_shot_examples: list[dict],
        budget: TokenBudget
    ) -> str:
        """
        Assemble context based on query type and template.
        """
        template = TEMPLATES[query_type]
        
        # Start with system prompt
        context_parts = [template.system_prompt]
        
        # Add few-shot examples if needed
        if template.include_few_shot:
            examples = few_shot_examples[:template.few_shot_count]
            for ex in examples:
                context_parts.append(
                    f"Example Question: {ex['question']}\n"
                    f"Example Answer: {ex['answer']}"
                )
        
        # Order and add documents
        ordered_docs = order_documents(documents, template.document_ordering)
        
        if template.summarize_documents:
            ordered_docs = [summarize_document(d) for d in ordered_docs]
        
        for doc in ordered_docs:
            doc_text = format_document(doc, include_metadata=template.include_metadata)
            context_parts.append(doc_text)
        
        # Add query
        context_parts.append(f"Question: {query}")
        
        return "\n\n".join(context_parts)
    
    
    def order_documents(
        documents: list[dict], 
        ordering: str
    ) -> list[dict]:
        """Order documents according to specified strategy."""
        if ordering == "relevance":
            # Assume documents are already sorted by relevance
            return documents
        elif ordering == "chronological":
            return sorted(documents, key=lambda d: d.get("date", ""))
        elif ordering == "reverse_relevance":
            # Put most relevant at end (for Lost in the Middle mitigation)
            return list(reversed(documents))
        return documents
    ```

### Mitigation Strategies

Several techniques can mitigate the Lost in the Middle problem and improve context utilization.

!!! tip "For Product Managers"
    **Available mitigation techniques**:
    
    | Technique | How It Works | Best For | Complexity |
    |-----------|--------------|----------|------------|
    | **Reordering** | Place important info at start/end | All use cases | Low |
    | **Monologuing** | Model restates key info before answering | Complex reasoning | Medium |
    | **Summarization** | Compress documents before inclusion | Long documents | Medium |
    | **Chunking** | Break context into smaller pieces | Very long contexts | High |
    | **Iterative refinement** | Multiple passes over context | High-stakes decisions | High |
    
    **ROI analysis**:
    
    - **Reordering**: Free to implement, 5-15% accuracy improvement
    - **Monologuing**: Increases output tokens by 20-50%, 10-20% accuracy improvement
    - **Summarization**: Requires additional LLM calls, enables 2-3x more source documents
    
    **Recommendation**: Start with reordering (free), add monologuing for complex queries, use summarization for document-heavy use cases.

!!! tip "For Engineers"
    **Technique 1: Strategic Reordering**
    
    Place the most relevant information at the beginning and end of the context:
    
    ```python
    def reorder_for_attention(
        documents: list[dict],
        strategy: str = "sandwich"
    ) -> list[dict]:
        """
        Reorder documents to optimize for attention patterns.
        
        Strategies:
        - "sandwich": Most relevant at start and end, less relevant in middle
        - "front_load": All documents ordered by relevance (most relevant first)
        - "back_load": Most relevant at end (for models with recency bias)
        """
        if not documents:
            return documents
            
        if strategy == "sandwich":
            # Split into high and low relevance
            n = len(documents)
            high_relevance = documents[:n//2]
            low_relevance = documents[n//2:]
            
            # Interleave: high at start, low in middle, high at end
            result = []
            for i, doc in enumerate(high_relevance):
                if i % 2 == 0:
                    result.insert(0, doc)  # Add to front
                else:
                    result.append(doc)  # Add to back
            
            # Insert low relevance in middle
            mid_point = len(result) // 2
            for doc in low_relevance:
                result.insert(mid_point, doc)
                mid_point += 1
            
            return result
            
        elif strategy == "front_load":
            return documents  # Assume already sorted by relevance
            
        elif strategy == "back_load":
            return list(reversed(documents))
        
        return documents
    ```
    
    **Technique 2: Monologuing**
    
    Have the model explicitly restate key information before generating the answer:
    
    ```python
    def create_monologue_prompt(
        query: str,
        documents: list[str],
        task_context: str
    ) -> str:
        """
        Create a prompt that encourages monologuing for improved comprehension.
        
        Monologuing helps the model "re-read" important information by
        requiring it to explicitly restate relevant details before answering.
        """
        doc_context = "\n\n".join([
            f"DOCUMENT {i+1}:\n{doc}" 
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""You will answer a question based on the provided documents.

    TASK CONTEXT: {task_context}

    DOCUMENTS:
    {doc_context}

    QUESTION: {query}

    Before answering, complete these steps:

    1. IDENTIFY KEY INFORMATION: List the specific facts, figures, or statements 
       from the documents that are relevant to answering this question.

    2. NOTE ANY CONFLICTS: If documents contain conflicting information, 
       identify the conflicts and how you will resolve them.

    3. ORGANIZE YOUR REASONING: Explain how the key information connects 
       to form your answer.

    4. PROVIDE YOUR ANSWER: Based on your analysis above, provide a clear, 
       well-supported answer.

    Begin your response with "KEY INFORMATION:" and proceed through each step."""
        
        return prompt
    ```
    
    **Technique 3: Compression via Summarization**
    
    ```python
    async def compress_documents(
        documents: list[str],
        query: str,
        target_tokens: int,
        llm_client
    ) -> list[str]:
        """
        Compress documents to fit within token budget while preserving
        query-relevant information.
        """
        compressed = []
        tokens_per_doc = target_tokens // len(documents)
        
        for doc in documents:
            current_tokens = count_tokens(doc)
            
            if current_tokens <= tokens_per_doc:
                compressed.append(doc)
            else:
                # Summarize with query context
                summary_prompt = f"""Summarize the following document, focusing on 
    information relevant to this query: "{query}"

    Keep your summary under {tokens_per_doc} tokens.

    DOCUMENT:
    {doc}

    SUMMARY:"""
                
                summary = await llm_client.generate(summary_prompt)
                compressed.append(summary)
        
        return compressed
    ```
    
    **Technique 4: Iterative Refinement**
    
    For high-stakes decisions, use multiple passes:
    
    ```python
    async def iterative_answer(
        query: str,
        documents: list[str],
        llm_client,
        max_iterations: int = 3
    ) -> dict:
        """
        Generate answer through iterative refinement.
        
        Each iteration reviews and improves the previous answer.
        """
        # First pass: generate initial answer
        initial_prompt = create_monologue_prompt(query, documents, "Initial analysis")
        current_answer = await llm_client.generate(initial_prompt)
        
        iterations = [{"iteration": 1, "answer": current_answer}]
        
        for i in range(2, max_iterations + 1):
            # Review and refine
            refinement_prompt = f"""Review this answer and improve it if needed.

    ORIGINAL QUESTION: {query}

    CURRENT ANSWER:
    {current_answer}

    DOCUMENTS (for reference):
    {chr(10).join(documents[:3])}  # Include top documents for reference

    Instructions:
    1. Check if the answer fully addresses the question
    2. Verify claims against the documents
    3. Identify any missing information or errors
    4. Provide an improved answer if needed, or confirm the current answer is complete

    REFINED ANSWER:"""
            
            refined = await llm_client.generate(refinement_prompt)
            
            # Check if answer changed significantly
            if refined.strip() == current_answer.strip():
                break
                
            current_answer = refined
            iterations.append({"iteration": i, "answer": current_answer})
        
        return {
            "final_answer": current_answer,
            "iterations": iterations,
            "total_iterations": len(iterations)
        }
    ```

### Long Context vs Chunked Retrieval

As context windows expand, teams must decide when to use full documents vs chunked retrieval.

!!! tip "For Product Managers"
    **The tradeoff**:
    
    | Approach | Advantages | Disadvantages |
    |----------|------------|---------------|
    | **Long context (full docs)** | Preserves document structure, simpler retrieval | Higher cost, Lost in the Middle risk, slower |
    | **Chunked retrieval** | Lower cost, faster, targeted information | Loses context, requires good chunking |
    
    **Decision framework**:
    
    Use **long context** when:
    
    - Documents are under 50 pages
    - Document structure matters (legal contracts, technical specs)
    - Queries require understanding relationships across sections
    - You have budget for higher token costs
    
    Use **chunked retrieval** when:
    
    - Documents are very long (100+ pages)
    - Queries target specific facts
    - Cost optimization is critical
    - Latency requirements are strict
    
    **Hybrid approach**: Use document-level retrieval to identify relevant documents, then include full documents in context. This simplifies retrieval while preserving document structure.

!!! tip "For Engineers"
    **Implementation: Hybrid document-level retrieval**
    
    ```python
    from dataclasses import dataclass
    
    @dataclass
    class Document:
        id: str
        title: str
        content: str
        chunks: list[str]
        chunk_embeddings: list[list[float]]
        summary: str
        summary_embedding: list[float]
    
    
    async def hybrid_retrieval(
        query: str,
        documents: list[Document],
        embedding_client,
        top_k_docs: int = 3,
        context_budget: int = 8000
    ) -> list[str]:
        """
        Retrieve at document level, then include full documents or summaries.
        
        This approach:
        1. Uses chunk embeddings to identify relevant documents
        2. Ranks documents by chunk relevance
        3. Includes full documents if they fit, otherwise summaries
        """
        query_embedding = await embedding_client.embed(query)
        
        # Score each document by its best chunk match
        doc_scores = []
        for doc in documents:
            best_chunk_score = max(
                cosine_similarity(query_embedding, chunk_emb)
                for chunk_emb in doc.chunk_embeddings
            )
            doc_scores.append((doc, best_chunk_score))
        
        # Sort by score and take top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in doc_scores[:top_k_docs]]
        
        # Fit documents within budget
        result = []
        remaining_budget = context_budget
        
        for doc in top_docs:
            doc_tokens = count_tokens(doc.content)
            
            if doc_tokens <= remaining_budget:
                # Include full document
                result.append(f"DOCUMENT: {doc.title}\n\n{doc.content}")
                remaining_budget -= doc_tokens
            else:
                # Include summary instead
                summary_tokens = count_tokens(doc.summary)
                if summary_tokens <= remaining_budget:
                    result.append(
                        f"DOCUMENT: {doc.title} (summarized)\n\n{doc.summary}"
                    )
                    remaining_budget -= summary_tokens
        
        return result
    
    
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0
    ```
    
    **When to use each approach**:
    
    ```python
    def select_retrieval_strategy(
        query_complexity: float,
        avg_doc_length: int,
        latency_requirement_ms: int,
        cost_sensitivity: float
    ) -> str:
        """
        Select retrieval strategy based on requirements.
        
        Returns: "full_document", "chunked", or "hybrid"
        """
        # Full document for complex queries on shorter documents
        if query_complexity > 0.7 and avg_doc_length < 10000:
            return "full_document"
        
        # Chunked for simple queries or very long documents
        if query_complexity < 0.3 or avg_doc_length > 50000:
            return "chunked"
        
        # Chunked for strict latency or cost requirements
        if latency_requirement_ms < 500 or cost_sensitivity > 0.8:
            return "chunked"
        
        # Hybrid for everything else
        return "hybrid"
    ```

---

## Case Study Deep Dive

### Insurance Claims Processing

An insurance company needed to process complex claims that required reviewing policy documents, claim history, and coverage details.

!!! tip "For Product Managers"
    **The challenge**:
    
    - Claims adjusters asked questions like "Is this water damage covered under policy #12345?"
    - Answering required reviewing 3-5 documents totaling 15,000+ tokens
    - Initial system achieved 67% accuracy on test cases
    - Errors often involved missing exclusions or conditions buried in policy documents
    
    **Root cause analysis**:
    
    The team analyzed 50 incorrect answers and found:
    
    - 40% missed relevant exclusions (typically in middle of documents)
    - 25% failed to connect information across multiple documents
    - 20% cited outdated policy versions
    - 15% other errors
    
    **Solution implemented**:
    
    1. **Reordering**: Placed exclusions and conditions at the beginning of context
    2. **Monologuing**: Required model to list relevant exclusions before answering
    3. **Metadata inclusion**: Added policy effective dates to prevent version confusion
    4. **Structured output**: Required answers to cite specific policy sections
    
    **Results**:
    
    | Metric | Before | After | Improvement |
    |--------|--------|-------|-------------|
    | Overall accuracy | 67% | 89% | +22 points |
    | Exclusion detection | 55% | 91% | +36 points |
    | Cross-document reasoning | 62% | 84% | +22 points |
    | User satisfaction | 3.2/5 | 4.4/5 | +1.2 points |
    
    **ROI**: The improvements reduced claim review time by 35% and decreased appeals due to incorrect initial decisions by 28%.

!!! tip "For Engineers"
    **Implementation details**:
    
    ```python
    @dataclass
    class InsuranceClaim:
        claim_id: str
        policy_id: str
        claim_type: str
        description: str
        amount: float
        date: str
    
    @dataclass
    class PolicyDocument:
        policy_id: str
        version: str
        effective_date: str
        sections: dict[str, str]  # section_name -> content
        exclusions: list[str]
        conditions: list[str]
    
    
    def build_claims_context(
        claim: InsuranceClaim,
        policy: PolicyDocument,
        claim_history: list[dict]
    ) -> str:
        """
        Build context optimized for claims processing.
        
        Key optimizations:
        1. Exclusions and conditions placed first (Lost in the Middle mitigation)
        2. Metadata included for version tracking
        3. Structured format for easy reference
        """
        context_parts = []
        
        # 1. Policy metadata (for version tracking)
        context_parts.append(f"""POLICY INFORMATION:
    Policy ID: {policy.policy_id}
    Version: {policy.version}
    Effective Date: {policy.effective_date}
    """)
        
        # 2. Exclusions FIRST (most commonly missed)
        if policy.exclusions:
            context_parts.append("POLICY EXCLUSIONS (IMPORTANT - Review carefully):")
            for i, exclusion in enumerate(policy.exclusions, 1):
                context_parts.append(f"  {i}. {exclusion}")
        
        # 3. Conditions second
        if policy.conditions:
            context_parts.append("\nPOLICY CONDITIONS:")
            for i, condition in enumerate(policy.conditions, 1):
                context_parts.append(f"  {i}. {condition}")
        
        # 4. Relevant policy sections
        relevant_sections = identify_relevant_sections(
            claim.claim_type, 
            policy.sections
        )
        context_parts.append("\nRELEVANT POLICY SECTIONS:")
        for section_name, content in relevant_sections.items():
            context_parts.append(f"\n{section_name}:\n{content}")
        
        # 5. Claim details
        context_parts.append(f"""
    CURRENT CLAIM:
    Claim ID: {claim.claim_id}
    Type: {claim.claim_type}
    Description: {claim.description}
    Amount: ${claim.amount:,.2f}
    Date: {claim.date}
    """)
        
        # 6. Claim history (at end, less critical)
        if claim_history:
            context_parts.append("\nPRIOR CLAIM HISTORY:")
            for hist in claim_history[-5:]:  # Last 5 claims
                context_parts.append(
                    f"  - {hist['date']}: {hist['type']} - {hist['outcome']}"
                )
        
        return "\n".join(context_parts)
    
    
    def create_claims_prompt(context: str, question: str) -> str:
        """Create prompt with monologuing for claims analysis."""
        return f"""You are an insurance claims analyst. Review the policy and claim 
    information to answer the question.

    {context}

    QUESTION: {question}

    Before providing your answer, complete these steps:

    1. APPLICABLE EXCLUSIONS: List any exclusions from the policy that may apply 
       to this claim. If none apply, state "No applicable exclusions identified."

    2. APPLICABLE CONDITIONS: List any conditions that must be met for coverage. 
       Note whether each condition appears to be satisfied based on the claim details.

    3. RELEVANT COVERAGE: Identify the specific policy sections that provide 
       coverage for this type of claim.

    4. ANALYSIS: Based on the above, explain whether this claim should be covered 
       and why.

    5. RECOMMENDATION: Provide your coverage recommendation with specific policy 
       section citations.

    Begin with "1. APPLICABLE EXCLUSIONS:" and proceed through each step."""
    
    
    def identify_relevant_sections(
        claim_type: str, 
        sections: dict[str, str]
    ) -> dict[str, str]:
        """Identify policy sections relevant to claim type."""
        # Map claim types to relevant section keywords
        relevance_map = {
            "water_damage": ["water", "flood", "plumbing", "property"],
            "theft": ["theft", "burglary", "personal property", "security"],
            "liability": ["liability", "injury", "damage", "third party"],
            "auto": ["vehicle", "collision", "comprehensive", "auto"]
        }
        
        keywords = relevance_map.get(claim_type, [])
        relevant = {}
        
        for section_name, content in sections.items():
            section_lower = section_name.lower()
            if any(kw in section_lower for kw in keywords):
                relevant[section_name] = content
        
        return relevant
    ```

---

## Implementation Guide

### Quick Start for PMs

**Week 1: Baseline Assessment**

1. Measure current accuracy on a test set of 50+ queries
2. Identify queries where the model misses information that was in the context
3. Categorize errors: position-related, complexity-related, or other

**Week 2: Implement Reordering**

1. Work with engineering to implement relevance-based reordering
2. Place most relevant documents at beginning and end of context
3. Re-measure accuracy on same test set

**Week 3: Add Monologuing for Complex Queries**

1. Identify query types that benefit from explicit reasoning
2. Implement monologue prompts for these query types
3. Measure accuracy improvement and latency impact

**Week 4: Optimize and Monitor**

1. Set up monitoring for context utilization metrics
2. Create dashboard showing accuracy by query complexity
3. Establish baseline for ongoing improvement

### Detailed Implementation for Engineers

**Step 1: Implement Token Counting and Budgeting**

```python
# Install required package
# uv add tiktoken

import tiktoken
from dataclasses import dataclass
from typing import Optional

@dataclass
class ContextMetrics:
    """Track context composition and utilization."""
    total_tokens: int
    system_prompt_tokens: int
    few_shot_tokens: int
    document_tokens: int
    query_tokens: int
    utilization: float  # percentage of budget used
    
    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "system_prompt_tokens": self.system_prompt_tokens,
            "few_shot_tokens": self.few_shot_tokens,
            "document_tokens": self.document_tokens,
            "query_tokens": self.query_tokens,
            "utilization": self.utilization
        }


class ContextManager:
    """Manage context assembly and token budgeting."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        total_limit: int = 16000,
        output_reserve: int = 2000
    ):
        self.model = model
        self.total_limit = total_limit
        self.output_reserve = output_reserve
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def assemble_context(
        self,
        system_prompt: str,
        few_shot_examples: list[dict],
        documents: list[str],
        query: str,
        reorder_strategy: str = "sandwich"
    ) -> tuple[str, ContextMetrics]:
        """
        Assemble context with token budgeting and reordering.
        
        Returns assembled context and metrics.
        """
        # Count fixed components
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)
        
        # Format and count few-shot examples
        few_shot_text = self._format_few_shot(few_shot_examples)
        few_shot_tokens = self.count_tokens(few_shot_text)
        
        # Calculate document budget
        fixed_tokens = system_tokens + few_shot_tokens + query_tokens
        doc_budget = self.total_limit - self.output_reserve - fixed_tokens
        
        # Select and reorder documents
        selected_docs = self._select_documents(documents, doc_budget)
        reordered_docs = self._reorder_documents(selected_docs, reorder_strategy)
        
        # Format documents
        doc_text = self._format_documents(reordered_docs)
        doc_tokens = self.count_tokens(doc_text)
        
        # Assemble final context
        context = f"""{system_prompt}

{few_shot_text}

DOCUMENTS:
{doc_text}

QUESTION: {query}"""
        
        total_tokens = self.count_tokens(context)
        
        metrics = ContextMetrics(
            total_tokens=total_tokens,
            system_prompt_tokens=system_tokens,
            few_shot_tokens=few_shot_tokens,
            document_tokens=doc_tokens,
            query_tokens=query_tokens,
            utilization=total_tokens / (self.total_limit - self.output_reserve)
        )
        
        return context, metrics
    
    def _format_few_shot(self, examples: list[dict]) -> str:
        """Format few-shot examples."""
        if not examples:
            return ""
        
        formatted = ["EXAMPLES:"]
        for i, ex in enumerate(examples, 1):
            formatted.append(f"\nExample {i}:")
            formatted.append(f"Q: {ex['question']}")
            formatted.append(f"A: {ex['answer']}")
        
        return "\n".join(formatted)
    
    def _select_documents(
        self, 
        documents: list[str], 
        budget: int
    ) -> list[str]:
        """Select documents that fit within budget."""
        selected = []
        remaining = budget
        
        for doc in documents:
            doc_tokens = self.count_tokens(doc)
            if doc_tokens <= remaining:
                selected.append(doc)
                remaining -= doc_tokens
        
        return selected
    
    def _reorder_documents(
        self, 
        documents: list[str], 
        strategy: str
    ) -> list[str]:
        """Reorder documents based on strategy."""
        if strategy == "sandwich" and len(documents) > 2:
            # Most relevant at start and end
            n = len(documents)
            mid = n // 2
            return documents[:mid:2] + documents[mid:] + documents[1:mid:2]
        elif strategy == "reverse":
            return list(reversed(documents))
        return documents
    
    def _format_documents(self, documents: list[str]) -> str:
        """Format documents for context."""
        formatted = []
        for i, doc in enumerate(documents, 1):
            formatted.append(f"[Document {i}]\n{doc}")
        return "\n\n".join(formatted)
```

**Step 2: Implement Monologuing**

```python
class MonologuePromptBuilder:
    """Build prompts that encourage monologuing."""
    
    TEMPLATES = {
        "general": """Before answering, complete these steps:

1. KEY INFORMATION: List the specific facts from the documents relevant to this question.
2. REASONING: Explain how these facts connect to form your answer.
3. ANSWER: Provide your final answer based on the above analysis.

Begin with "1. KEY INFORMATION:" and proceed through each step.""",

        "comparison": """Before answering, complete these steps:

1. ITEM A DETAILS: List relevant details about the first item from the documents.
2. ITEM B DETAILS: List relevant details about the second item from the documents.
3. COMPARISON: Identify key similarities and differences.
4. CONCLUSION: Provide your comparative analysis.

Begin with "1. ITEM A DETAILS:" and proceed through each step.""",

        "decision": """Before answering, complete these steps:

1. OPTIONS: List the available options from the documents.
2. CRITERIA: Identify the decision criteria mentioned or implied.
3. EVALUATION: Evaluate each option against the criteria.
4. RECOMMENDATION: Provide your recommendation with justification.

Begin with "1. OPTIONS:" and proceed through each step."""
    }
    
    def build_prompt(
        self,
        base_context: str,
        query: str,
        monologue_type: str = "general"
    ) -> str:
        """Build prompt with monologue instructions."""
        template = self.TEMPLATES.get(monologue_type, self.TEMPLATES["general"])
        
        return f"""{base_context}

QUESTION: {query}

{template}"""
```

**Step 3: Implement Evaluation**

```python
@dataclass
class ContextEvaluation:
    """Evaluation results for context management."""
    query_id: str
    correct: bool
    answer_position_sensitivity: float  # Did answer change with reordering?
    key_info_recalled: float  # Percentage of key info mentioned
    reasoning_quality: float  # 0-1 score for reasoning
    
    
async def evaluate_context_management(
    test_cases: list[dict],
    context_manager: ContextManager,
    llm_client,
    monologue_builder: MonologuePromptBuilder
) -> dict:
    """
    Evaluate context management effectiveness.
    
    Test cases should include:
    - query: The question
    - documents: List of relevant documents
    - expected_answer: Ground truth
    - key_facts: List of facts that should be mentioned
    """
    results = []
    
    for case in test_cases:
        # Test with different orderings
        orderings = ["relevance", "sandwich", "reverse"]
        answers_by_ordering = {}
        
        for ordering in orderings:
            context, metrics = context_manager.assemble_context(
                system_prompt="Answer based on the documents.",
                few_shot_examples=[],
                documents=case["documents"],
                query=case["query"],
                reorder_strategy=ordering
            )
            
            prompt = monologue_builder.build_prompt(
                context, 
                case["query"], 
                "general"
            )
            
            answer = await llm_client.generate(prompt)
            answers_by_ordering[ordering] = answer
        
        # Evaluate
        correct = evaluate_correctness(
            answers_by_ordering["sandwich"], 
            case["expected_answer"]
        )
        
        position_sensitivity = calculate_answer_variance(
            list(answers_by_ordering.values())
        )
        
        key_info_recalled = calculate_recall(
            answers_by_ordering["sandwich"],
            case["key_facts"]
        )
        
        results.append(ContextEvaluation(
            query_id=case.get("id", "unknown"),
            correct=correct,
            answer_position_sensitivity=position_sensitivity,
            key_info_recalled=key_info_recalled,
            reasoning_quality=0.0  # Would need LLM judge
        ))
    
    # Aggregate results
    return {
        "accuracy": sum(r.correct for r in results) / len(results),
        "avg_position_sensitivity": sum(r.answer_position_sensitivity for r in results) / len(results),
        "avg_key_info_recall": sum(r.key_info_recalled for r in results) / len(results),
        "total_cases": len(results),
        "detailed_results": results
    }


def evaluate_correctness(answer: str, expected: str) -> bool:
    """Simple correctness check - in practice, use LLM judge."""
    # Normalize and compare key terms
    answer_lower = answer.lower()
    expected_lower = expected.lower()
    
    # Check if key terms from expected appear in answer
    key_terms = expected_lower.split()
    matches = sum(1 for term in key_terms if term in answer_lower)
    
    return matches / len(key_terms) > 0.7 if key_terms else False


def calculate_answer_variance(answers: list[str]) -> float:
    """Calculate how much answers vary with different orderings."""
    if len(answers) < 2:
        return 0.0
    
    # Simple approach: compare word overlap between answers
    word_sets = [set(a.lower().split()) for a in answers]
    
    overlaps = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            overlaps.append(intersection / union if union else 1.0)
    
    # Higher variance = lower overlap = more position sensitive
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 1.0
    return 1.0 - avg_overlap


def calculate_recall(answer: str, key_facts: list[str]) -> float:
    """Calculate what percentage of key facts appear in answer."""
    if not key_facts:
        return 1.0
    
    answer_lower = answer.lower()
    found = sum(1 for fact in key_facts if fact.lower() in answer_lower)
    
    return found / len(key_facts)
```

---

## Common Pitfalls

### PM Pitfalls

!!! warning "PM Pitfall: Assuming Bigger Context Windows Solve Everything"
    **The mistake**: Upgrading to a model with 200K context and assuming all context problems are solved.
    
    **Why it fails**: Bigger windows create new problems—higher costs, slower responses, and the Lost in the Middle effect becomes more pronounced with longer contexts.
    
    **The fix**: Treat context window size as a budget to optimize, not a problem to throw tokens at. Measure accuracy at different context sizes to find the optimal point.

!!! warning "PM Pitfall: Ignoring Context Management in Evaluation"
    **The mistake**: Evaluating RAG systems only on retrieval metrics without measuring generation quality.
    
    **Why it fails**: Perfect retrieval with poor context management still produces incorrect answers. Users experience the final answer, not the retrieval quality.
    
    **The fix**: Include end-to-end accuracy metrics that measure whether the final answer is correct, not just whether the right documents were retrieved.

!!! warning "PM Pitfall: One-Size-Fits-All Context Strategy"
    **The mistake**: Using the same context template for all query types.
    
    **Why it fails**: Simple factual queries need different context than complex reasoning tasks. Over-engineering simple queries wastes tokens; under-engineering complex queries produces errors.
    
    **The fix**: Classify queries by complexity and use appropriate context strategies for each type.

### Engineering Pitfalls

!!! warning "Engineering Pitfall: Not Reserving Output Tokens"
    **The mistake**: Filling the context window completely without reserving space for the model's response.
    
    **Why it fails**: The model's response gets truncated, producing incomplete answers.
    
    **The fix**: Always reserve 1,500-3,000 tokens for output, depending on expected response length. For monologuing prompts, reserve more since the reasoning steps add length.

!!! warning "Engineering Pitfall: Static Token Counting"
    **The mistake**: Using character counts or word counts instead of actual token counts.
    
    **Why it fails**: Token counts vary significantly by content type. Code has different tokenization than prose. Non-English text often uses more tokens per character.
    
    **The fix**: Use the actual tokenizer for your model (tiktoken for OpenAI, appropriate tokenizer for other models).

!!! warning "Engineering Pitfall: Ignoring Position in Evaluation"
    **The mistake**: Evaluating accuracy without testing position sensitivity.
    
    **Why it fails**: A system might achieve 85% accuracy in testing but fail on production queries where relevant information happens to land in the middle of the context.
    
    **The fix**: Include position-varied test cases in your evaluation suite. Test the same query with relevant information at beginning, middle, and end positions.

---

## Related Content

### Source Materials

- **Workshop Content**: `docs/workshops/chapter3-3.md` - Monologuing techniques
- **Workshop Content**: `docs/workshops/chapter5-2.md` - Document summarization as compression
- **Office Hours**: `docs/office-hours/cohort2/week2-summary.md` - Long context vs RAG discussion
- **Office Hours**: `docs/office-hours/cohort3/week-5-1.md` - Position bias and shuffling

### Key Insights from Sources

**From the workshops**:

> "As context windows grow larger, one might think that managing complex information would become easier. Counterintuitively, though, larger context windows often create new challenges for language models, which can struggle to attend to the most relevant information among thousands of tokens."

**From office hours on long context**:

> "The battery analogy is apt: iPhone batteries get more powerful every year, but battery life stays the same because we build more power-hungry apps. Similarly, as context windows grow, we'll find ways to use that additional capacity rather than making everything faster or cheaper."

**From office hours on position bias**:

> "If you look at the newer models, they just have way better lost-in-the-middle sensitivity in general, and I would expect that when you fine-tune these things, they also preserve some of that ability to attend over long contexts."

### Related Talks

- **RAG Antipatterns (Skylar Payne)**: `docs/talks/rag-antipatterns-skylar-payne.md` - Common mistakes in context management

---

## Action Items

### For Product Teams

1. **Audit current context usage**: Review how context is assembled for your top 10 query types
2. **Measure position sensitivity**: Test whether answer quality varies with document ordering
3. **Define complexity tiers**: Categorize queries by complexity to enable dynamic context strategies
4. **Set context efficiency targets**: Establish metrics for context utilization vs accuracy tradeoffs
5. **Plan monologue rollout**: Identify high-value query types that would benefit from explicit reasoning

### For Engineering Teams

1. **Implement token budgeting**: Add explicit token counting and budget allocation to context assembly
2. **Add reordering logic**: Implement sandwich or relevance-based reordering for retrieved documents
3. **Build monologue prompts**: Create prompt templates that encourage step-by-step reasoning
4. **Set up position-varied evaluation**: Add test cases that vary information position
5. **Monitor context metrics**: Track token utilization, position distribution, and accuracy by context size
6. **Implement compression fallbacks**: Add summarization for when full documents exceed budget

---

## Reflection Questions

1. **For your use case**: What percentage of your queries involve contexts longer than 4K tokens? How does accuracy vary with context length?

2. **Position sensitivity**: Have you tested whether your system produces different answers when document order changes? What would be the business impact of inconsistent answers?

3. **Cost vs accuracy**: What is the optimal context size for your use case? At what point do additional tokens stop improving accuracy?

4. **Monologuing tradeoffs**: For which query types would the latency cost of monologuing be justified by accuracy improvements?

5. **Long context strategy**: Should your system use full documents or chunked retrieval? What factors drive this decision for your specific use case?

---

## Summary

### Key Takeaways for Product Managers

- **Context management is as important as retrieval**: Perfect retrieval with poor context presentation still produces incorrect answers
- **Bigger is not always better**: Larger context windows create new challenges around attention and cost
- **Position matters**: The Lost in the Middle effect means document ordering significantly impacts accuracy
- **Invest in evaluation**: Measure end-to-end accuracy, not just retrieval metrics
- **Match strategy to complexity**: Simple queries need different context than complex reasoning tasks

### Key Takeaways for Engineers

- **Implement token budgeting**: Explicitly allocate tokens across system prompt, examples, documents, and output
- **Use strategic reordering**: Place most relevant information at the beginning and end of context
- **Add monologuing for complex queries**: Having the model restate key information improves reasoning
- **Evaluate position sensitivity**: Test whether answers change with document reordering
- **Consider hybrid approaches**: Document-level retrieval with full document inclusion often outperforms pure chunking

---

## Further Reading

### Academic Papers

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - The foundational research on position effects in long contexts
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Original RAG paper

### Tools and Implementations

- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's tokenizer for accurate token counting
- [LlamaIndex Context Management](https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/) - Response synthesis patterns

### Related Concepts

- **Prompt Engineering**: Techniques for structuring prompts effectively
- **Chain of Thought**: Related to monologuing, encouraging step-by-step reasoning
- **Retrieval-Augmented Generation**: The broader framework this chapter fits within

---

## Navigation

**Previous**: [Chapter 8: Hybrid Search](chapter8.md) - Combining semantic and lexical search for robust retrieval

**Next**: Appendix A: Mathematical Foundations - Formulas and derivations for retrieval metrics (coming soon)

**Related Chapters**:

- [Chapter 1: Evaluation-First Development](chapter1.md) - Evaluation framework referenced throughout
- [Chapter 5: Specialized Retrieval Systems](chapter5.md) - RAPTOR and summarization techniques
- [Chapter 6: Query Routing and Orchestration](chapter6.md) - Query classification for dynamic context
- [Chapter 7: Production Operations](chapter7.md) - Cost optimization strategies
