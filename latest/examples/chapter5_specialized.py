"""
Chapter 5: Specialized Retrieval Systems - Code Examples

This module provides specialized retrieval infrastructure:
- RAPTOR implementation for long documents
- Metadata extraction pipelines
- Synthetic text generation for multimodal content
- Multimodal retrieval (images, tables)
- Page-aware chunking
- Contextual chunk rewriting

Usage:
    from chapter5_specialized import (
        build_raptor_index,
        raptor_retrieve,
        extract_metadata,
        generate_synthetic_chunk,
        chunk_by_pages,
        create_contextual_chunk,
    )

    # Build RAPTOR index for long documents
    index = await build_raptor_index(chunks)

    # Retrieve using RAPTOR
    results = await raptor_retrieve(query, index)

    # Extract metadata from documents
    metadata = await extract_metadata(document_text, FinancialStatement)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np
from pydantic import BaseModel
from sklearn.cluster import AgglomerativeClustering

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Data Models
# =============================================================================


class ContentType(str, Enum):
    """Types of content for specialized retrieval."""

    DOCUMENT = "document"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"
    AUDIO = "audio"
    VIDEO = "video"


class SyntheticChunk(BaseModel):
    """Synthetic text chunk optimized for retrieval.

    Attributes:
        title: Descriptive title for the chunk
        category: Content category
        summary: Searchable summary text
        entities: Key entities mentioned
        source_id: Pointer back to original content
        content_type: Type of original content
    """

    title: str
    category: str
    summary: str
    entities: list[str]
    source_id: str
    content_type: ContentType = ContentType.DOCUMENT


class FinancialStatement(BaseModel):
    """Structured representation of a financial statement document.

    Example metadata extraction target for financial documents.

    Attributes:
        company: Company name
        period_ending: End date of the reporting period
        revenue: Total revenue
        net_income: Net income
        earnings_per_share: EPS
        fiscal_year: Whether this is fiscal year (vs calendar year)
        sector: Business sector
        currency: Currency code
        restated: Whether statement has been restated
    """

    company: str
    period_ending: date
    revenue: float
    net_income: float
    earnings_per_share: float
    fiscal_year: bool = True
    sector: str | None = None
    currency: str = "USD"
    restated: bool = False


class ContractMetadata(BaseModel):
    """Metadata extracted from contracts.

    Attributes:
        parties: List of parties to the contract
        effective_date: When contract becomes effective
        expiration_date: When contract expires
        contract_type: Type of contract
        value: Contract value if specified
        signed: Whether contract is signed
        jurisdiction: Legal jurisdiction
    """

    parties: list[str]
    effective_date: date | None = None
    expiration_date: date | None = None
    contract_type: str | None = None
    value: float | None = None
    signed: bool = False
    jurisdiction: str | None = None


class ImageDescription(BaseModel):
    """Structured description of an image for retrieval.

    Attributes:
        description: Natural language description
        objects: Objects detected in image
        text_content: Any text visible in image
        colors: Dominant colors
        scene_type: Type of scene (indoor, outdoor, diagram, etc.)
        source_id: Original image identifier
    """

    description: str
    objects: list[str]
    text_content: str | None = None
    colors: list[str] = []
    scene_type: str | None = None
    source_id: str = ""


class TableDescription(BaseModel):
    """Structured description of a table for retrieval.

    Attributes:
        title: Table title or caption
        columns: Column names
        row_count: Number of rows
        summary: Natural language summary
        key_values: Important values extracted
        source_id: Original table identifier
    """

    title: str
    columns: list[str]
    row_count: int
    summary: str
    key_values: dict[str, Any] = {}
    source_id: str = ""


@dataclass
class RAPTORIndex:
    """RAPTOR hierarchical index structure.

    Attributes:
        leaf_chunks: Original document chunks
        clusters: Mapping of cluster labels to chunks
        summaries: Summaries for each cluster
        embeddings: Embeddings for leaf chunks
        tree_levels: Number of levels in the tree
    """

    leaf_chunks: list[str]
    clusters: dict[int, list[str]]
    summaries: dict[int, str]
    embeddings: np.ndarray
    tree_levels: int = 1


@dataclass
class Chunk:
    """A document chunk with metadata.

    Attributes:
        text: Chunk text content
        chunk_id: Unique identifier
        source_id: Source document identifier
        page_number: Page number if applicable
        section: Section name if applicable
        metadata: Additional metadata
    """

    text: str
    chunk_id: str
    source_id: str
    page_number: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Embedding Protocol
# =============================================================================


class EmbeddingModel:
    """Protocol for embedding models."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        raise NotImplementedError


class SimpleEmbeddingModel:
    """Simple embedding model using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =============================================================================
# RAPTOR Implementation
# =============================================================================


async def build_raptor_index(
    chunks: list[str],
    embedding_model: EmbeddingModel | None = None,
    distance_threshold: float = 0.8,
    summarize_fn: Any | None = None,
) -> RAPTORIndex:
    """
    Build RAPTOR hierarchical index from document chunks.

    RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
    creates a tree structure where leaf nodes are original chunks and
    internal nodes are summaries of semantically similar chunks.

    Args:
        chunks: List of document chunks (page-level or section-level)
        embedding_model: Model for generating embeddings
        distance_threshold: Threshold for hierarchical clustering
        summarize_fn: Function to generate summaries

    Returns:
        RAPTORIndex containing tree structure with summaries

    Examples:
        >>> chunks = ["Section 1 content...", "Section 2 content..."]
        >>> index = await build_raptor_index(chunks)
        >>> results = await raptor_retrieve("query", index)
    """
    if embedding_model is None:
        embedding_model = SimpleEmbeddingModel()

    logger.info(f"Building RAPTOR index for {len(chunks)} chunks...")

    # Step 1: Embed all chunks
    embeddings = embedding_model.encode(chunks)

    # Step 2: Cluster similar chunks using hierarchical clustering
    if len(chunks) > 1:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="ward",
        )
        cluster_labels = clustering.fit_predict(embeddings)
    else:
        cluster_labels = np.array([0])

    # Step 3: Group chunks by cluster
    clusters: dict[int, list[str]] = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunks[idx])

    logger.info(f"Created {len(clusters)} clusters")

    # Step 4: Generate summaries for each cluster
    cluster_summaries: dict[int, str] = {}
    for label, cluster_chunks in clusters.items():
        combined_text = "\n\n---\n\n".join(cluster_chunks)

        if summarize_fn:
            summary = await summarize_fn(combined_text)
        else:
            # Simple fallback: use first 500 chars of combined text
            summary = combined_text[:500] + "..." if len(combined_text) > 500 else combined_text

        cluster_summaries[label] = summary

    return RAPTORIndex(
        leaf_chunks=chunks,
        clusters=clusters,
        summaries=cluster_summaries,
        embeddings=embeddings,
        tree_levels=1,
    )


async def raptor_retrieve(
    query: str,
    raptor_index: RAPTORIndex,
    embedding_model: EmbeddingModel | None = None,
    top_k: int = 5,
    top_clusters: int = 3,
) -> list[str]:
    """
    Retrieve from RAPTOR index using multi-level search.

    First searches cluster summaries for broad matching, then
    searches within top clusters for specific chunks.

    Args:
        query: User query
        raptor_index: Built RAPTOR index
        embedding_model: Model for generating query embedding
        top_k: Number of final results to return
        top_clusters: Number of clusters to search within

    Returns:
        List of relevant chunks
    """
    if embedding_model is None:
        embedding_model = SimpleEmbeddingModel()

    query_embedding = embedding_model.encode([query])[0]

    # Search summaries first for broad matching
    summary_scores = []
    for label, summary in raptor_index.summaries.items():
        summary_embedding = embedding_model.encode([summary])[0]
        score = cosine_similarity(query_embedding, summary_embedding)
        summary_scores.append((label, score))

    # Get top clusters
    top_cluster_labels = sorted(summary_scores, key=lambda x: x[1], reverse=True)[
        :top_clusters
    ]

    # Search within top clusters for specific chunks
    candidate_chunks = []
    candidate_embeddings = []

    for label, _ in top_cluster_labels:
        for chunk in raptor_index.clusters[label]:
            candidate_chunks.append(chunk)
            # Find embedding for this chunk
            idx = raptor_index.leaf_chunks.index(chunk)
            candidate_embeddings.append(raptor_index.embeddings[idx])

    if not candidate_chunks:
        return []

    # Score candidates
    candidate_scores = []
    for chunk, emb in zip(candidate_chunks, candidate_embeddings):
        score = cosine_similarity(query_embedding, emb)
        candidate_scores.append((chunk, score))

    # Return top-k
    final_results = sorted(candidate_scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [chunk for chunk, _ in final_results]


async def build_multi_level_raptor(
    chunks: list[str],
    embedding_model: EmbeddingModel | None = None,
    max_levels: int = 3,
    min_clusters_for_next_level: int = 5,
) -> dict[int, RAPTORIndex]:
    """
    Build multi-level RAPTOR index for very long documents.

    Creates multiple levels of abstraction, with each level
    summarizing the level below it.

    Args:
        chunks: Original document chunks
        embedding_model: Embedding model
        max_levels: Maximum number of levels
        min_clusters_for_next_level: Minimum clusters to create another level

    Returns:
        Dict mapping level number to RAPTORIndex
    """
    if embedding_model is None:
        embedding_model = SimpleEmbeddingModel()

    levels: dict[int, RAPTORIndex] = {}
    current_chunks = chunks

    for level in range(max_levels):
        logger.info(f"Building RAPTOR level {level} with {len(current_chunks)} chunks")

        index = await build_raptor_index(
            current_chunks,
            embedding_model=embedding_model,
        )
        levels[level] = index

        # Check if we should create another level
        if len(index.summaries) < min_clusters_for_next_level:
            break

        # Use summaries as chunks for next level
        current_chunks = list(index.summaries.values())

    return levels


# =============================================================================
# Metadata Extraction
# =============================================================================


async def extract_metadata(
    document_text: str,
    schema: type[T],
    llm_client: Any = None,
    system_prompt: str | None = None,
) -> T:
    """
    Extract structured metadata from document text using LLM.

    Args:
        document_text: Raw text from document
        schema: Pydantic model defining expected structure
        llm_client: LLM client with structured output support
        system_prompt: Optional custom system prompt

    Returns:
        Structured object with extracted data

    Examples:
        >>> metadata = await extract_metadata(
        ...     document_text="Annual Report 2024...",
        ...     schema=FinancialStatement
        ... )
    """
    if system_prompt is None:
        field_descriptions = []
        for name, field in schema.model_fields.items():
            desc = field.description or name
            field_descriptions.append(f"- {name}: {desc}")

        system_prompt = f"""Extract the following information from the document:
{chr(10).join(field_descriptions)}

Format your response as a JSON object with these fields.
If a field cannot be determined from the document, use null.
"""

    if llm_client:
        # Use structured outputs for reliable extraction
        result = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": document_text[:10000]},  # Limit context
            ],
            response_model=schema,
        )
        return result
    else:
        # Return empty instance for demo
        return schema.model_construct()


async def extract_metadata_batch(
    documents: list[tuple[str, str]],  # (doc_id, text) pairs
    schema: type[T],
    llm_client: Any = None,
) -> dict[str, T]:
    """
    Extract metadata from multiple documents.

    Args:
        documents: List of (doc_id, text) tuples
        schema: Pydantic model for extraction
        llm_client: LLM client

    Returns:
        Dict mapping doc_id to extracted metadata
    """
    results = {}
    for doc_id, text in documents:
        try:
            metadata = await extract_metadata(text, schema, llm_client)
            results[doc_id] = metadata
        except Exception as e:
            logger.error(f"Failed to extract metadata from {doc_id}: {e}")
    return results


# =============================================================================
# Synthetic Text Generation
# =============================================================================


async def generate_synthetic_chunk(
    content: str,
    task_context: str,
    source_id: str,
    content_type: ContentType = ContentType.DOCUMENT,
    llm_client: Any = None,
) -> SyntheticChunk:
    """
    Generate summary optimized for specific retrieval tasks.

    Creates a synthetic chunk that acts as a better search target
    than the original content, while pointing back to the source.

    Args:
        content: Source content (text, image description, etc.)
        task_context: What users typically query for
        source_id: Identifier for original content
        content_type: Type of original content
        llm_client: LLM client for generation

    Returns:
        SyntheticChunk optimized for retrieval

    Examples:
        >>> chunk = await generate_synthetic_chunk(
        ...     content="Blueprint showing floor plan...",
        ...     task_context="room counts, dimensions, spatial relationships",
        ...     source_id="blueprint_001"
        ... )
    """
    prompt = f"""Create a searchable summary of this content optimized for these query types:
{task_context}

Include:
- Explicit counts of items users will search for
- Key dimensions and measurements
- Important dates and timelines
- Critical relationships between entities

The summary should match how users phrase their searches, not how
the content is written.

Content:
{content[:5000]}

Respond with:
TITLE: <descriptive title>
CATEGORY: <content category>
SUMMARY: <searchable summary>
ENTITIES: <comma-separated list of key entities>
"""

    if llm_client:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content

        # Parse response
        lines = text.strip().split("\n")
        title = category = summary = ""
        entities = []

        for line in lines:
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
            elif line.startswith("CATEGORY:"):
                category = line.replace("CATEGORY:", "").strip()
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("ENTITIES:"):
                entities = [e.strip() for e in line.replace("ENTITIES:", "").split(",")]

        return SyntheticChunk(
            title=title,
            category=category,
            summary=summary,
            entities=entities,
            source_id=source_id,
            content_type=content_type,
        )
    else:
        # Demo response
        return SyntheticChunk(
            title=f"Summary of {source_id}",
            category="document",
            summary=content[:200],
            entities=[],
            source_id=source_id,
            content_type=content_type,
        )


async def generate_image_description(
    image_path: str,
    vision_client: Any = None,
    task_context: str = "general image search",
) -> ImageDescription:
    """
    Generate searchable description from an image.

    Args:
        image_path: Path to image file
        vision_client: Vision-capable LLM client
        task_context: What users typically search for

    Returns:
        ImageDescription for retrieval
    """
    prompt = f"""Describe this image in detail for search purposes.
Focus on: {task_context}

Include:
- All visible objects and their relationships
- Any text visible in the image
- Colors and visual characteristics
- Scene type (indoor, outdoor, diagram, etc.)
"""

    if vision_client:
        # Use vision API
        response = await vision_client.describe_image(image_path, prompt)
        # Parse response into ImageDescription
        return ImageDescription(
            description=response,
            objects=[],  # Would be extracted from response
            source_id=image_path,
        )
    else:
        return ImageDescription(
            description=f"Image at {image_path}",
            objects=[],
            source_id=image_path,
        )


async def generate_table_description(
    table_data: list[list[str]],
    table_id: str,
    llm_client: Any = None,
) -> TableDescription:
    """
    Generate searchable description from a table.

    Args:
        table_data: Table as list of rows (first row is headers)
        table_id: Identifier for the table
        llm_client: LLM client for generation

    Returns:
        TableDescription for retrieval
    """
    if not table_data:
        return TableDescription(
            title="Empty table",
            columns=[],
            row_count=0,
            summary="",
            source_id=table_id,
        )

    headers = table_data[0]
    rows = table_data[1:]

    # Format table for LLM
    table_str = " | ".join(headers) + "\n"
    table_str += "-" * 50 + "\n"
    for row in rows[:10]:  # Limit rows for context
        table_str += " | ".join(str(cell) for cell in row) + "\n"

    prompt = f"""Analyze this table and provide:
1. A descriptive title
2. A natural language summary of what the table contains
3. Key values or insights from the data

Table:
{table_str}
"""

    if llm_client:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.choices[0].message.content
    else:
        summary = f"Table with {len(rows)} rows and columns: {', '.join(headers)}"

    return TableDescription(
        title=f"Table: {headers[0] if headers else 'Unknown'}",
        columns=headers,
        row_count=len(rows),
        summary=summary,
        source_id=table_id,
    )


# =============================================================================
# Page-Aware Chunking
# =============================================================================


def chunk_by_pages(
    document: str,
    page_separator: str = "\n\n--- PAGE BREAK ---\n\n",
    min_size: int = 200,
    max_size: int = 2000,
    overlap: int = 100,
) -> list[Chunk]:
    """
    Chunk document respecting page boundaries.

    Args:
        document: Full document text with page markers
        page_separator: String marking page breaks
        min_size: Minimum chunk size in characters
        max_size: Maximum chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of Chunk objects with page metadata
    """
    pages = document.split(page_separator)
    chunks = []

    for page_num, page_text in enumerate(pages, 1):
        page_text = page_text.strip()
        if not page_text:
            continue

        if len(page_text) <= max_size:
            # Page fits in one chunk
            chunks.append(
                Chunk(
                    text=page_text,
                    chunk_id=f"page_{page_num}_chunk_1",
                    source_id="document",
                    page_number=page_num,
                )
            )
        else:
            # Split page into multiple chunks
            start = 0
            chunk_num = 1
            while start < len(page_text):
                end = min(start + max_size, len(page_text))

                # Try to break at sentence boundary
                if end < len(page_text):
                    for sep in [". ", ".\n", "\n\n"]:
                        last_sep = page_text.rfind(sep, start, end)
                        if last_sep > start + min_size:
                            end = last_sep + len(sep)
                            break

                chunk_text = page_text[start:end].strip()
                if len(chunk_text) >= min_size:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id=f"page_{page_num}_chunk_{chunk_num}",
                            source_id="document",
                            page_number=page_num,
                        )
                    )
                    chunk_num += 1

                start = end - overlap if end < len(page_text) else end

    return chunks


def chunk_by_sections(
    document: str,
    section_pattern: str = r"^#{1,3}\s+.+$",
    min_size: int = 200,
    max_size: int = 2000,
) -> list[Chunk]:
    """
    Chunk document by section headers.

    Args:
        document: Document text with markdown headers
        section_pattern: Regex pattern for section headers
        min_size: Minimum chunk size
        max_size: Maximum chunk size

    Returns:
        List of Chunk objects with section metadata
    """
    import re

    sections = re.split(f"({section_pattern})", document, flags=re.MULTILINE)
    chunks = []

    current_section = "Introduction"
    chunk_num = 1

    for i, part in enumerate(sections):
        if re.match(section_pattern, part):
            current_section = part.strip("#").strip()
            continue

        part = part.strip()
        if not part or len(part) < min_size:
            continue

        if len(part) <= max_size:
            chunks.append(
                Chunk(
                    text=part,
                    chunk_id=f"section_{chunk_num}",
                    source_id="document",
                    section=current_section,
                )
            )
            chunk_num += 1
        else:
            # Split large sections
            start = 0
            while start < len(part):
                end = min(start + max_size, len(part))
                chunk_text = part[start:end].strip()
                if len(chunk_text) >= min_size:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id=f"section_{chunk_num}",
                            source_id="document",
                            section=current_section,
                        )
                    )
                    chunk_num += 1
                start = end

    return chunks


# =============================================================================
# Contextual Chunk Rewriting
# =============================================================================


async def create_contextual_chunk(
    chunk: str,
    document_title: str,
    section: str | None = None,
    llm_client: Any = None,
) -> str:
    """
    Rewrite chunk with document context for better retrieval.

    Chunks in isolation can be ambiguous. This function rewrites
    them to include necessary context.

    Args:
        chunk: Original chunk text
        document_title: Title of source document
        section: Section name if available
        llm_client: LLM client for rewriting

    Returns:
        Rewritten chunk with context

    Examples:
        >>> # Original: "Jason the doctor is unhappy with Patient X"
        >>> # Could mean many things without context
        >>> rewritten = await create_contextual_chunk(
        ...     chunk="Jason the doctor is unhappy with Patient X",
        ...     document_title="Hospital Staff Meeting Notes - March 2024",
        ...     section="Patient Care Discussion"
        ... )
        >>> # Now: "In the March 2024 hospital staff meeting, during the
        >>> # patient care discussion, Dr. Jason expressed concerns about
        >>> # the treatment plan for Patient X"
    """
    prompt = f"""Document context: {document_title}
{f"Section: {section}" if section else ""}

Original chunk: {chunk}

Rewrite this chunk to include necessary context so it can be understood
in isolation. Maintain all factual information but add context clues
that clarify ambiguous references.
"""

    if llm_client:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    else:
        # Simple fallback: prepend context
        context = f"[From: {document_title}"
        if section:
            context += f", Section: {section}"
        context += "] "
        return context + chunk


async def enrich_chunks_with_context(
    chunks: list[Chunk],
    document_title: str,
    llm_client: Any = None,
) -> list[Chunk]:
    """
    Enrich multiple chunks with contextual information.

    Args:
        chunks: List of Chunk objects
        document_title: Title of source document
        llm_client: LLM client for rewriting

    Returns:
        List of chunks with enriched text
    """
    enriched = []
    for chunk in chunks:
        enriched_text = await create_contextual_chunk(
            chunk.text,
            document_title,
            chunk.section,
            llm_client,
        )
        enriched.append(
            Chunk(
                text=enriched_text,
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                page_number=chunk.page_number,
                section=chunk.section,
                metadata={**chunk.metadata, "original_text": chunk.text},
            )
        )
    return enriched


# =============================================================================
# Specialized Index Building
# =============================================================================


@dataclass
class SpecializedIndex:
    """A specialized index for a specific content type.

    Attributes:
        name: Index name
        content_type: Type of content indexed
        chunks: Indexed chunks
        embeddings: Chunk embeddings
        metadata: Index metadata
    """

    name: str
    content_type: ContentType
    chunks: list[Chunk]
    embeddings: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


async def build_specialized_index(
    documents: list[dict[str, Any]],
    content_type: ContentType,
    index_name: str,
    embedding_model: EmbeddingModel | None = None,
    chunking_strategy: str = "pages",
) -> SpecializedIndex:
    """
    Build a specialized index for a specific content type.

    Args:
        documents: List of document dicts with 'id' and 'text'
        content_type: Type of content being indexed
        index_name: Name for the index
        embedding_model: Embedding model
        chunking_strategy: How to chunk documents ("pages", "sections", "fixed")

    Returns:
        SpecializedIndex ready for retrieval
    """
    if embedding_model is None:
        embedding_model = SimpleEmbeddingModel()

    all_chunks = []

    for doc in documents:
        doc_id = doc.get("id", "unknown")
        text = doc.get("text", "")

        if chunking_strategy == "pages":
            chunks = chunk_by_pages(text)
        elif chunking_strategy == "sections":
            chunks = chunk_by_sections(text)
        else:
            # Fixed-size chunking
            chunks = [
                Chunk(
                    text=text[i : i + 1000],
                    chunk_id=f"{doc_id}_chunk_{i // 1000}",
                    source_id=doc_id,
                )
                for i in range(0, len(text), 1000)
            ]

        # Update source_id
        for chunk in chunks:
            chunk.source_id = doc_id

        all_chunks.extend(chunks)

    # Generate embeddings
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedding_model.encode(texts)

    return SpecializedIndex(
        name=index_name,
        content_type=content_type,
        chunks=all_chunks,
        embeddings=embeddings,
    )


async def search_specialized_index(
    query: str,
    index: SpecializedIndex,
    embedding_model: EmbeddingModel | None = None,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    """
    Search a specialized index.

    Args:
        query: Search query
        index: SpecializedIndex to search
        embedding_model: Embedding model
        top_k: Number of results

    Returns:
        List of (chunk, score) tuples
    """
    if embedding_model is None:
        embedding_model = SimpleEmbeddingModel()

    query_embedding = embedding_model.encode([query])[0]

    # Calculate similarities
    scores = []
    for i, chunk in enumerate(index.chunks):
        score = cosine_similarity(query_embedding, index.embeddings[i])
        scores.append((chunk, score))

    # Sort and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# =============================================================================
# Two-Level Measurement Framework
# =============================================================================


@dataclass
class TwoLevelMetrics:
    """Metrics for the two-level measurement framework.

    P(finding data) = P(selecting retriever) x P(finding data | retriever)

    Attributes:
        retriever_selection_accuracy: P(selecting correct retriever)
        retriever_recall: P(finding data | correct retriever)
        overall_recall: Combined probability
        retriever_name: Name of the retriever being measured
    """

    retriever_selection_accuracy: float
    retriever_recall: float
    overall_recall: float
    retriever_name: str


def calculate_two_level_metrics(
    queries: list[dict[str, Any]],
    retriever_name: str,
) -> TwoLevelMetrics:
    """
    Calculate two-level metrics for a specialized retriever.

    Args:
        queries: List of query results with 'correct_retriever',
                 'selected_retriever', and 'found_relevant' fields
        retriever_name: Name of the retriever being measured

    Returns:
        TwoLevelMetrics object
    """
    # Filter to queries where this retriever should have been selected
    relevant_queries = [
        q for q in queries if q.get("correct_retriever") == retriever_name
    ]

    if not relevant_queries:
        return TwoLevelMetrics(
            retriever_selection_accuracy=0.0,
            retriever_recall=0.0,
            overall_recall=0.0,
            retriever_name=retriever_name,
        )

    # P(selecting retriever)
    correctly_selected = sum(
        1 for q in relevant_queries if q.get("selected_retriever") == retriever_name
    )
    selection_accuracy = correctly_selected / len(relevant_queries)

    # P(finding data | retriever)
    queries_with_correct_retriever = [
        q for q in relevant_queries if q.get("selected_retriever") == retriever_name
    ]
    if queries_with_correct_retriever:
        found_relevant = sum(
            1 for q in queries_with_correct_retriever if q.get("found_relevant", False)
        )
        retriever_recall = found_relevant / len(queries_with_correct_retriever)
    else:
        retriever_recall = 0.0

    # Overall
    overall_recall = selection_accuracy * retriever_recall

    return TwoLevelMetrics(
        retriever_selection_accuracy=selection_accuracy,
        retriever_recall=retriever_recall,
        overall_recall=overall_recall,
        retriever_name=retriever_name,
    )
