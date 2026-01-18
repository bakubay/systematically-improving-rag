---
title: "Appendix D: Debugging RAG Systems"
description: A systematic guide to identifying, diagnosing, and fixing common RAG system failures
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - debugging
  - troubleshooting
  - failure-modes
  - data-quality
  - evaluation
---

# Appendix D: Debugging RAG Systems

## Introduction

Debugging RAG systems differs fundamentally from debugging traditional software. There are no stack traces when the model produces a wrong answer, no exceptions when retrieval returns irrelevant documents, and no error codes when users abandon your system in frustration. This appendix provides a systematic methodology for identifying, diagnosing, and fixing common RAG failures.

!!! tip "For Product Managers"
    Understanding failure modes helps you set realistic expectations with stakeholders, prioritize engineering work effectively, and communicate issues clearly. You don't need to debug systems yourself, but knowing the taxonomy of failures helps you ask the right questions.

!!! tip "For Engineers"
    This appendix gives you a structured approach to debugging that replaces ad-hoc investigation with systematic diagnosis. Each failure mode includes detection methods, root cause analysis, and proven fixes.

## Systematic Debugging Methodology

Before diving into specific failure modes, establish a systematic approach to debugging. Random investigation wastes time and often leads to fixing symptoms rather than root causes.

### The Five-Step Debugging Process

**Step 1: Reproduce the Failure**

Document the exact conditions that produce the failure:

- The specific query that failed
- The expected result versus actual result
- The timestamp and any relevant context
- User information if available

```python
from datetime import datetime

class FailureReport:
    """Document a RAG failure for systematic investigation."""
    
    def __init__(
        self,
        query: str,
        expected_result: str,
        actual_result: str,
        retrieved_docs: list[dict],
        timestamp: datetime,
        user_context: dict | None = None
    ):
        self.query = query
        self.expected_result = expected_result
        self.actual_result = actual_result
        self.retrieved_docs = retrieved_docs
        self.timestamp = timestamp
        self.user_context = user_context or {}
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "expected": self.expected_result,
            "actual": self.actual_result,
            "retrieved_count": len(self.retrieved_docs),
            "retrieved_ids": [d.get("id") for d in self.retrieved_docs],
            "timestamp": self.timestamp.isoformat(),
            "context": self.user_context
        }
```

**Step 2: Isolate the Component**

RAG systems have multiple components that can fail. Isolate which component is responsible:

| Component | Failure Indicator | Quick Test |
|-----------|-------------------|------------|
| Data Ingestion | Missing documents | Check document counts at each pipeline stage |
| Chunking | Broken context | Inspect chunks for the relevant document |
| Embedding | Similar queries return different results | Compare embeddings for paraphrased queries |
| Retrieval | Wrong documents returned | Check if correct document exists and its score |
| Re-ranking | Correct doc retrieved but ranked low | Examine pre and post re-ranking order |
| Generation | Correct docs but wrong answer | Test generation with known-good context |

**Step 3: Gather Evidence**

Collect data that helps identify the root cause:

```python
async def gather_debug_evidence(
    query: str,
    retriever,
    generator,
    k: int = 20
) -> dict:
    """Collect comprehensive debugging information."""
    
    # Get query embedding
    query_embedding = await retriever.embed_query(query)
    
    # Retrieve with extra context
    results = await retriever.search(query, top_k=k)
    
    # Get scores at different k values
    scores_at_k = {
        f"top_{n}": [r["score"] for r in results[:n]]
        for n in [1, 3, 5, 10, 20]
    }
    
    # Check score distribution
    all_scores = [r["score"] for r in results]
    score_stats = {
        "min": min(all_scores),
        "max": max(all_scores),
        "mean": sum(all_scores) / len(all_scores),
        "spread": max(all_scores) - min(all_scores)
    }
    
    # Generate with retrieved context
    context = "\n\n".join([r["text"] for r in results[:5]])
    response = await generator.generate(query, context)
    
    return {
        "query": query,
        "query_embedding_norm": sum(x**2 for x in query_embedding) ** 0.5,
        "results": results,
        "scores_at_k": scores_at_k,
        "score_stats": score_stats,
        "generated_response": response
    }
```

**Step 4: Form and Test Hypotheses**

Based on evidence, form specific hypotheses about the root cause:

- "The correct document was chunked incorrectly, splitting the answer across chunks"
- "The query uses terminology that doesn't match document vocabulary"
- "The embedding model doesn't understand domain-specific concepts"

Test each hypothesis with targeted experiments.

**Step 5: Implement and Verify Fix**

After identifying the root cause:

1. Implement the smallest change that fixes the issue
2. Verify the fix resolves the original failure
3. Run regression tests to ensure no new failures
4. Document the fix for future reference

### The Debugging Quadrant

Organize your investigation using this quadrant analysis:

```
                    Retrieval Correct
                    |
        Q1: Lucky   |   Q2: Working
        (Right answer,  |   (Right answer,
         wrong docs)    |    right docs)
                    |
--------------------+--------------------
                    |
        Q3: Broken  |   Q4: Unlucky
        (Wrong answer,  |   (Wrong answer,
         wrong docs)    |    right docs)
                    |
                    Retrieval Incorrect
```

!!! info "Quadrant Interpretation"
    - **Q1 (Lucky)**: The model got lucky. This is fragile and will break.
    - **Q2 (Working)**: System working as intended. Monitor for regression.
    - **Q3 (Broken)**: Retrieval problem. Focus on search improvements.
    - **Q4 (Unlucky)**: Generation problem. Focus on prompting or context.

## Failure Modes Taxonomy

### Category 1: Data Ingestion Failures

These failures occur before any user query is processed. They're particularly dangerous because they're silent - your system appears to work but is missing critical information.

#### Silent Document Loss

**Symptoms:**

- Index size smaller than expected
- Users report missing information that should exist
- Queries that should match return no results

**Root Causes:**

1. **Encoding failures**: Documents with non-UTF-8 encoding silently fail to parse

```python
# BAD: Assumes UTF-8, silently fails on other encodings
def load_document_bad(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()  # Fails silently or corrupts text

# GOOD: Detect encoding and handle failures explicitly
import chardet

def load_document_good(path: str) -> str | None:
    with open(path, 'rb') as f:
        raw = f.read()
    
    detected = chardet.detect(raw)
    encoding = detected['encoding']
    confidence = detected['confidence']
    
    if confidence < 0.7:
        logger.warning(f"Low confidence encoding detection for {path}: {encoding} ({confidence})")
    
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode {path} with {encoding}: {e}")
        return None
```

2. **Format-specific parsing failures**: PDFs with complex layouts, scanned documents, or unusual formats

3. **Pipeline stage drops**: Documents lost between collection, processing, chunking, or indexing

**Detection:**

```python
from datetime import datetime

class PipelineMonitor:
    """Track document counts through the ingestion pipeline."""
    
    def __init__(self):
        self.counts = {}
    
    def record(self, stage: str, count: int, details: dict | None = None):
        self.counts[stage] = {
            "count": count,
            "timestamp": datetime.now(),
            "details": details or {}
        }
    
    def report(self) -> dict:
        stages = list(self.counts.keys())
        report = {"stages": self.counts, "drops": []}
        
        for i in range(1, len(stages)):
            prev_stage = stages[i-1]
            curr_stage = stages[i]
            prev_count = self.counts[prev_stage]["count"]
            curr_count = self.counts[curr_stage]["count"]
            
            if curr_count < prev_count:
                drop_pct = (prev_count - curr_count) / prev_count * 100
                report["drops"].append({
                    "from": prev_stage,
                    "to": curr_stage,
                    "lost": prev_count - curr_count,
                    "percentage": drop_pct
                })
        
        return report

# Usage
monitor = PipelineMonitor()
monitor.record("collected", 1000)
monitor.record("parsed", 985)  # 15 parsing failures
monitor.record("chunked", 4200)  # Multiple chunks per doc
monitor.record("embedded", 4200)
monitor.record("indexed", 4198)  # 2 indexing failures
```

**Fixes:**

- Implement encoding detection before parsing
- Log all failures explicitly rather than silently skipping
- Set up alerts for document count drops between stages
- Validate extracted content meets minimum quality thresholds

!!! warning "PM Pitfall"
    Don't assume "the system is working" because users aren't complaining. Silent data loss means users simply can't find information they need - they may not know it should exist. Proactively audit document counts.

#### Extraction Quality Issues

**Symptoms:**

- Chunks contain garbled text or formatting artifacts
- Table data appears as disconnected text
- Important information missing from chunks

**Root Causes:**

1. **PDF parsing failures**: Tables, multi-column layouts, headers/footers
2. **OCR errors**: Scanned documents with recognition mistakes
3. **Format conversion artifacts**: HTML to text, DOCX to text

**Detection:**

```python
def validate_chunk_quality(chunk: str) -> dict:
    """Check chunk for common quality issues."""
    
    issues = []
    
    # Check for garbled text (high ratio of special characters)
    special_chars = sum(1 for c in chunk if not c.isalnum() and not c.isspace())
    special_ratio = special_chars / len(chunk) if chunk else 0
    if special_ratio > 0.3:
        issues.append(f"High special character ratio: {special_ratio:.2%}")
    
    # Check for very short chunks
    word_count = len(chunk.split())
    if word_count < 10:
        issues.append(f"Very short chunk: {word_count} words")
    
    # Check for repetitive content
    words = chunk.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    if unique_ratio < 0.3:
        issues.append(f"Highly repetitive content: {unique_ratio:.2%} unique")
    
    # Check for common OCR artifacts
    ocr_artifacts = ['|', '~', '^', '`']
    artifact_count = sum(chunk.count(a) for a in ocr_artifacts)
    if artifact_count > 10:
        issues.append(f"Possible OCR artifacts: {artifact_count} found")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "metrics": {
            "special_char_ratio": special_ratio,
            "word_count": word_count,
            "unique_word_ratio": unique_ratio
        }
    }
```

**Fixes:**

- Use specialized parsing tools for complex documents (Reducto, Llama Parse, Dockling)
- Implement chunk validation before indexing
- Consider vision models for document understanding
- Manually review a sample of chunks regularly

### Category 2: Chunking Failures

Chunking determines what information can be retrieved together. Poor chunking is a common source of retrieval failures.

#### Context Fragmentation

**Symptoms:**

- Answers span multiple chunks but only one is retrieved
- Questions about relationships between concepts fail
- "The system found part of the answer but not all of it"

**Root Causes:**

1. **Chunks too small**: Important context split across chunks
2. **Semantic boundaries ignored**: Chunking by token count rather than meaning
3. **Missing overlap**: No redundancy between adjacent chunks

**Detection:**

```python
def analyze_chunk_boundaries(
    chunks: list[str],
    questions: list[dict]
) -> dict:
    """Analyze whether chunk boundaries align with information needs."""
    
    boundary_issues = []
    
    for q in questions:
        query = q["query"]
        expected_docs = q["expected_doc_ids"]
        
        # Check if answer spans multiple chunks from same source
        source_chunks = {}
        for doc_id in expected_docs:
            source = doc_id.rsplit("_chunk_", 1)[0]
            if source not in source_chunks:
                source_chunks[source] = []
            source_chunks[source].append(doc_id)
        
        for source, chunk_ids in source_chunks.items():
            if len(chunk_ids) > 1:
                boundary_issues.append({
                    "query": query,
                    "source": source,
                    "required_chunks": chunk_ids,
                    "issue": "Answer spans multiple chunks"
                })
    
    return {
        "total_questions": len(questions),
        "boundary_issues": len(boundary_issues),
        "issue_rate": len(boundary_issues) / len(questions),
        "details": boundary_issues
    }
```

**Fixes:**

- Increase chunk size (start with 800 tokens, 50% overlap)
- Use semantic chunking that respects document structure
- Implement parent-child chunk relationships
- Consider page-level chunking for certain document types

#### Low-Value Chunks

**Symptoms:**

- Retrieval returns boilerplate content
- Headers, footers, or navigation text retrieved
- Copyright notices or disclaimers in results

**Root Causes:**

1. **No filtering of boilerplate content**
2. **Chunks created from non-content sections**
3. **Duplicate content across documents**

**Detection:**

```python
import re

def identify_low_value_chunks(chunks: list[dict]) -> list[dict]:
    """Find chunks that are likely low-value boilerplate."""
    
    low_value = []
    
    # Common boilerplate patterns
    boilerplate_patterns = [
        r"^copyright\s+\d{4}",
        r"^all rights reserved",
        r"^table of contents",
        r"^page \d+ of \d+",
        r"^confidential",
        r"^for internal use only"
    ]
    
    for chunk in chunks:
        text = chunk["text"].lower().strip()
        
        # Check against patterns
        for pattern in boilerplate_patterns:
            if re.match(pattern, text):
                low_value.append({
                    "chunk_id": chunk["id"],
                    "reason": f"Matches boilerplate pattern: {pattern}",
                    "preview": text[:100]
                })
                break
        
        # Check for very short content
        if len(text.split()) < 20:
            low_value.append({
                "chunk_id": chunk["id"],
                "reason": "Very short content",
                "preview": text
            })
    
    return low_value
```

**Fixes:**

- Filter boilerplate content during chunking
- Implement content hashing to detect duplicates
- Set minimum content thresholds for chunks
- Review shortest chunks manually

### Category 3: Retrieval Failures

Retrieval failures occur when the search system doesn't return the right documents.

#### Vocabulary Mismatch

**Symptoms:**

- Queries using different terminology than documents fail
- Synonyms don't retrieve relevant content
- Domain experts find content that search misses

**Root Causes:**

1. **Embedding model doesn't understand domain vocabulary**
2. **Documents use formal language, queries use informal**
3. **Acronyms and abbreviations not expanded**

**Detection:**

```python
async def detect_vocabulary_mismatch(
    query: str,
    expected_doc: dict,
    retriever
) -> dict:
    """Check if vocabulary differences explain retrieval failure."""
    
    # Get query terms
    query_terms = set(query.lower().split())
    
    # Get document terms
    doc_terms = set(expected_doc["text"].lower().split())
    
    # Calculate overlap
    overlap = query_terms & doc_terms
    query_only = query_terms - doc_terms
    
    # Try query expansion
    expanded_query = await expand_query_with_synonyms(query)
    expanded_results = await retriever.search(expanded_query, top_k=10)
    
    # Check if expansion helps
    original_results = await retriever.search(query, top_k=10)
    original_found = expected_doc["id"] in [r["id"] for r in original_results]
    expanded_found = expected_doc["id"] in [r["id"] for r in expanded_results]
    
    return {
        "term_overlap": len(overlap) / len(query_terms),
        "query_only_terms": list(query_only),
        "original_found": original_found,
        "expanded_found": expanded_found,
        "expansion_helped": expanded_found and not original_found
    }
```

**Fixes:**

- Implement query expansion with synonyms
- Fine-tune embeddings on domain-specific data
- Add document summaries that use varied vocabulary
- Consider hybrid search (semantic + lexical)

#### Score Distribution Issues

**Symptoms:**

- All retrieved documents have similar scores
- Correct document has low score even when retrieved
- Score thresholds don't work consistently

**Root Causes:**

1. **Embedding model produces similar vectors for different content**
2. **Query too general, matches many documents equally**
3. **Index contains too much similar content**

**Detection:**

```python
def analyze_score_distribution(
    results: list[dict],
    expected_ids: list[str]
) -> dict:
    """Analyze retrieval score distribution for issues."""
    
    scores = [r["score"] for r in results]
    expected_scores = [r["score"] for r in results if r["id"] in expected_ids]
    other_scores = [r["score"] for r in results if r["id"] not in expected_ids]
    
    # Calculate statistics
    score_range = max(scores) - min(scores)
    score_std = (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5
    
    # Check separation between expected and other
    if expected_scores and other_scores:
        separation = min(expected_scores) - max(other_scores)
    else:
        separation = None
    
    issues = []
    
    if score_range < 0.1:
        issues.append("Very narrow score range - documents too similar")
    
    if score_std < 0.05:
        issues.append("Low score variance - poor discrimination")
    
    if separation is not None and separation < 0:
        issues.append("Expected documents score lower than irrelevant ones")
    
    return {
        "score_range": score_range,
        "score_std": score_std,
        "separation": separation,
        "issues": issues,
        "expected_scores": expected_scores,
        "top_5_scores": scores[:5]
    }
```

**Fixes:**

- Use re-ranking to improve score discrimination
- Fine-tune embeddings to increase separation
- Implement query classification to route to specialized indexes
- Add metadata filtering to reduce candidate set

### Category 4: Generation Failures

Generation failures occur when the model has the right context but produces the wrong answer.

#### Hallucination

**Symptoms:**

- Model states facts not present in retrieved documents
- Citations point to documents that don't support claims
- Confident answers that are incorrect

**Root Causes:**

1. **Model relies on parametric knowledge instead of context**
2. **Prompt doesn't enforce grounding in retrieved content**
3. **No citation validation**

**Detection:**

```python
async def detect_hallucination(
    query: str,
    response: str,
    retrieved_docs: list[dict],
    llm
) -> dict:
    """Check if response contains unsupported claims."""
    
    # Extract claims from response
    claims = await extract_claims(response, llm)
    
    # Check each claim against retrieved docs
    context = "\n\n".join([d["text"] for d in retrieved_docs])
    
    unsupported_claims = []
    for claim in claims:
        support_check = await llm.generate(f"""
        Context:
        {context}
        
        Claim: {claim}
        
        Is this claim directly supported by the context? 
        Answer only YES or NO, then explain briefly.
        """)
        
        if support_check.strip().upper().startswith("NO"):
            unsupported_claims.append({
                "claim": claim,
                "explanation": support_check
            })
    
    return {
        "total_claims": len(claims),
        "unsupported_claims": len(unsupported_claims),
        "hallucination_rate": len(unsupported_claims) / len(claims) if claims else 0,
        "details": unsupported_claims
    }
```

**Fixes:**

- Require inline citations in responses
- Validate citations exist and support claims
- Use prompts that explicitly instruct grounding
- Implement response verification step

#### Lost in the Middle

**Symptoms:**

- Information at the beginning or end of context is used
- Middle content is ignored even when relevant
- Longer contexts produce worse results

**Root Causes:**

1. **Attention patterns favor beginning and end of context**
2. **Too much context provided**
3. **Relevant information buried among irrelevant content**

**Detection:**

```python
def analyze_position_bias(
    test_cases: list[dict],
    results: list[dict]
) -> dict:
    """Check if answer position in context affects accuracy."""
    
    position_accuracy = {"beginning": [], "middle": [], "end": []}
    
    for test, result in zip(test_cases, results):
        # Determine where answer was in context
        context = result["context"]
        answer_location = result["answer_source_location"]
        context_length = len(context)
        
        if answer_location < context_length * 0.33:
            position = "beginning"
        elif answer_location > context_length * 0.66:
            position = "end"
        else:
            position = "middle"
        
        position_accuracy[position].append(result["correct"])
    
    return {
        position: sum(scores) / len(scores) if scores else None
        for position, scores in position_accuracy.items()
    }
```

**Fixes:**

- Limit context to most relevant documents
- Order documents by relevance (most relevant first)
- Use summarization to compress context
- Consider multiple passes with different context subsets

### Category 5: System-Level Failures

These failures involve interactions between components or operational issues.

#### Stale Index

**Symptoms:**

- Recent documents not found
- Outdated information returned
- Users report "I know this was updated"

**Root Causes:**

1. **Index refresh not scheduled or failing**
2. **Incremental updates not working**
3. **Cache serving stale results**

**Detection:**

```python
from datetime import datetime

def check_index_freshness(
    index_metadata: dict,
    expected_freshness_hours: int = 24
) -> dict:
    """Verify index is sufficiently fresh."""
    
    last_update = datetime.fromisoformat(index_metadata["last_updated"])
    age_hours = (datetime.now() - last_update).total_seconds() / 3600
    
    issues = []
    
    if age_hours > expected_freshness_hours:
        issues.append(f"Index is {age_hours:.1f} hours old (threshold: {expected_freshness_hours})")
    
    # Check document timestamps
    newest_doc = max(index_metadata["document_timestamps"])
    oldest_doc = min(index_metadata["document_timestamps"])
    
    return {
        "last_update": last_update.isoformat(),
        "age_hours": age_hours,
        "is_fresh": age_hours <= expected_freshness_hours,
        "newest_document": newest_doc,
        "oldest_document": oldest_doc,
        "issues": issues
    }
```

**Fixes:**

- Implement automated index refresh
- Monitor refresh job success/failure
- Add freshness metadata to responses
- Implement cache invalidation on updates

#### Latency Degradation

**Symptoms:**

- System slower than usual
- Timeouts increasing
- User complaints about speed

**Root Causes:**

1. **Index size growth without scaling**
2. **Inefficient queries or missing indexes**
3. **Resource contention**

**Detection:**

```python
import time
from contextlib import contextmanager

@contextmanager
def timing(name: str, metrics: dict):
    """Context manager to track operation timing."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if name not in metrics:
        metrics[name] = []
    metrics[name].append(elapsed)

async def profile_rag_pipeline(query: str, pipeline) -> dict:
    """Profile each stage of the RAG pipeline."""
    
    metrics = {}
    
    with timing("embedding", metrics):
        query_embedding = await pipeline.embed_query(query)
    
    with timing("retrieval", metrics):
        results = await pipeline.retrieve(query_embedding, top_k=20)
    
    with timing("reranking", metrics):
        reranked = await pipeline.rerank(query, results)
    
    with timing("generation", metrics):
        response = await pipeline.generate(query, reranked[:5])
    
    return {
        stage: {
            "time_ms": times[-1] * 1000,
            "avg_ms": sum(times) / len(times) * 1000
        }
        for stage, times in metrics.items()
    }
```

**Fixes:**

- Profile each pipeline stage
- Implement caching for repeated queries
- Scale infrastructure based on load
- Optimize slow queries

## Debugging Tools and Techniques

### Building a Debug Dashboard

Create a simple dashboard to monitor system health:

```python
import time
from datetime import datetime

class RAGDebugDashboard:
    """Simple debugging dashboard for RAG systems."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.recent_queries = []
        self.failure_log = []
        self.metrics = {}
    
    async def debug_query(self, query: str, expected: str | None = None) -> dict:
        """Run a query with full debugging information."""
        
        debug_info = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        # Stage 1: Embedding
        start = time.perf_counter()
        embedding = await self.pipeline.embed_query(query)
        debug_info["stages"]["embedding"] = {
            "time_ms": (time.perf_counter() - start) * 1000,
            "embedding_norm": sum(x**2 for x in embedding) ** 0.5
        }
        
        # Stage 2: Retrieval
        start = time.perf_counter()
        results = await self.pipeline.retrieve(embedding, top_k=20)
        debug_info["stages"]["retrieval"] = {
            "time_ms": (time.perf_counter() - start) * 1000,
            "result_count": len(results),
            "score_range": [results[0]["score"], results[-1]["score"]] if results else None,
            "top_5": [{"id": r["id"], "score": r["score"]} for r in results[:5]]
        }
        
        # Stage 3: Re-ranking
        start = time.perf_counter()
        reranked = await self.pipeline.rerank(query, results)
        debug_info["stages"]["reranking"] = {
            "time_ms": (time.perf_counter() - start) * 1000,
            "order_changed": [r["id"] for r in results[:5]] != [r["id"] for r in reranked[:5]],
            "top_5": [{"id": r["id"], "score": r["score"]} for r in reranked[:5]]
        }
        
        # Stage 4: Generation
        context = "\n\n".join([r["text"] for r in reranked[:5]])
        start = time.perf_counter()
        response = await self.pipeline.generate(query, context)
        debug_info["stages"]["generation"] = {
            "time_ms": (time.perf_counter() - start) * 1000,
            "response_length": len(response),
            "context_length": len(context)
        }
        
        debug_info["response"] = response
        
        # Check against expected if provided
        if expected:
            debug_info["expected"] = expected
            debug_info["matches_expected"] = expected.lower() in response.lower()
        
        self.recent_queries.append(debug_info)
        return debug_info
    
    def get_health_summary(self) -> dict:
        """Get overall system health summary."""
        
        if not self.recent_queries:
            return {"status": "no_data"}
        
        recent = self.recent_queries[-100:]  # Last 100 queries
        
        avg_latency = sum(
            sum(q["stages"][s]["time_ms"] for s in q["stages"])
            for q in recent
        ) / len(recent)
        
        return {
            "status": "healthy" if avg_latency < 2000 else "degraded",
            "avg_latency_ms": avg_latency,
            "query_count": len(recent),
            "failure_count": len(self.failure_log)
        }
```

### Logging Best Practices

Implement structured logging for debugging:

```python
import structlog

logger = structlog.get_logger()

async def logged_retrieval(query: str, retriever, top_k: int = 10):
    """Retrieval with structured logging for debugging."""
    
    log = logger.bind(
        query=query,
        top_k=top_k,
        operation="retrieval"
    )
    
    log.info("starting_retrieval")
    
    try:
        start = time.perf_counter()
        results = await retriever.search(query, top_k=top_k)
        elapsed = time.perf_counter() - start
        
        log.info(
            "retrieval_complete",
            result_count=len(results),
            elapsed_ms=elapsed * 1000,
            top_score=results[0]["score"] if results else None,
            score_spread=results[0]["score"] - results[-1]["score"] if results else None
        )
        
        return results
        
    except Exception as e:
        log.error(
            "retrieval_failed",
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

### Creating Reproducible Test Cases

Build a test case library from failures:

```python
class DebugTestCase:
    """A reproducible test case for debugging."""
    
    def __init__(
        self,
        name: str,
        query: str,
        expected_doc_ids: list[str],
        expected_answer_contains: list[str],
        failure_mode: str,
        notes: str = ""
    ):
        self.name = name
        self.query = query
        self.expected_doc_ids = expected_doc_ids
        self.expected_answer_contains = expected_answer_contains
        self.failure_mode = failure_mode
        self.notes = notes
    
    async def run(self, pipeline) -> dict:
        """Run the test case and return results."""
        
        results = await pipeline.search(self.query, top_k=10)
        retrieved_ids = [r["id"] for r in results]
        
        # Check retrieval
        retrieval_recall = len(
            set(retrieved_ids) & set(self.expected_doc_ids)
        ) / len(self.expected_doc_ids)
        
        # Check generation
        response = await pipeline.generate_answer(self.query, results[:5])
        answer_matches = all(
            phrase.lower() in response.lower()
            for phrase in self.expected_answer_contains
        )
        
        return {
            "name": self.name,
            "passed": retrieval_recall >= 0.8 and answer_matches,
            "retrieval_recall": retrieval_recall,
            "answer_matches": answer_matches,
            "failure_mode": self.failure_mode,
            "retrieved_ids": retrieved_ids,
            "response": response
        }

# Example test cases
test_cases = [
    DebugTestCase(
        name="encoding_failure_latin1",
        query="What are the safety requirements for chemical storage?",
        expected_doc_ids=["safety_manual_ch3_001"],
        expected_answer_contains=["ventilation", "temperature"],
        failure_mode="silent_data_loss",
        notes="Document was originally Latin-1 encoded"
    ),
    DebugTestCase(
        name="vocabulary_mismatch_acronym",
        query="What does the SLA say about uptime?",
        expected_doc_ids=["service_agreement_002"],
        expected_answer_contains=["99.9%", "availability"],
        failure_mode="vocabulary_mismatch",
        notes="Document uses 'Service Level Agreement' not 'SLA'"
    )
]
```

## Common Pitfalls

!!! warning "PM Pitfall: Treating All Failures as Model Problems"
    When users report bad answers, the instinct is often to blame the LLM. In reality, most failures trace back to data quality, chunking, or retrieval issues. Before requesting model changes, investigate the full pipeline.

!!! warning "PM Pitfall: Ignoring Silent Failures"
    Systems that appear to work may be silently dropping documents, returning stale data, or missing important content. Establish monitoring for document counts, index freshness, and retrieval coverage.

!!! warning "Engineering Pitfall: Debugging in Production"
    Without a reproducible test environment, debugging becomes guesswork. Build a test harness that lets you replay queries with full visibility into each pipeline stage.

!!! warning "Engineering Pitfall: Fixing Symptoms Instead of Root Causes"
    Adding a special case to handle one failing query often creates technical debt. Investigate why the query failed and fix the underlying issue.

!!! warning "Engineering Pitfall: Over-Engineering Solutions"
    Before implementing complex solutions (graph databases, multi-stage retrieval, custom models), verify that simpler fixes don't solve the problem. Most issues trace back to data quality or basic configuration.

## Quick Reference: Debugging Checklist

When investigating a RAG failure, work through this checklist:

**1. Data Ingestion**

- [ ] Is the document in the index?
- [ ] Was it parsed correctly?
- [ ] Are there encoding issues?
- [ ] Is the content complete?

**2. Chunking**

- [ ] Is the relevant content in a single chunk?
- [ ] Is the chunk too small/large?
- [ ] Is important context preserved?
- [ ] Are there low-value chunks polluting results?

**3. Retrieval**

- [ ] Is the correct document retrieved at all?
- [ ] What rank is it at?
- [ ] What's the score compared to other results?
- [ ] Does query expansion help?

**4. Re-ranking**

- [ ] Does re-ranking improve or hurt the ranking?
- [ ] Is the re-ranker appropriate for this content type?
- [ ] Are there score calibration issues?

**5. Generation**

- [ ] Is the answer supported by retrieved content?
- [ ] Is there hallucination?
- [ ] Is relevant content being ignored?
- [ ] Is the prompt appropriate?

## Summary

Debugging RAG systems requires systematic investigation rather than random experimentation. Key principles:

1. **Isolate components**: Determine which stage is failing before attempting fixes
2. **Gather evidence**: Collect data that supports or refutes hypotheses
3. **Fix root causes**: Address underlying issues rather than symptoms
4. **Build test cases**: Create reproducible tests from failures
5. **Monitor continuously**: Detect issues before users report them

The most common failures trace back to data quality issues (silent document loss, extraction failures, chunking problems) rather than model limitations. Investing in robust data pipelines and monitoring pays dividends in system reliability.

## Further Reading

- [Chapter 1: Evaluation-First Development](chapter1.md) - Building evaluation frameworks
- [Appendix C: Benchmarking Your RAG System](appendix-benchmarks.md) - Systematic testing approaches
- [RAG Antipatterns Talk (Skylar Payne)](../talks/rag-antipatterns-skylar-payne.md) - Common failure patterns

## Navigation

- **Previous**: [Appendix C: Benchmarking Your RAG System](appendix-benchmarks.md)
- **Reference**: [Glossary](glossary.md) | [Quick Reference](quick-reference.md)
- **Book Index**: [Book Overview](index.md)
