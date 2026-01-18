"""
Chapter 3: Feedback Systems and UX - Code Examples

This module provides feedback collection and UX infrastructure for RAG systems:
- Structured feedback collection with typed enums
- Implicit signal mining from user behavior
- Streaming response implementation
- Citation extraction and validation
- Chain of thought prompting
- Validation patterns for error catching
- Slack integration for enterprise feedback

Usage:
    from chapter3_feedback import (
        FeedbackType,
        UserFeedback,
        collect_feedback,
        mine_hard_negatives_from_session,
        stream_response,
        extract_citations,
        validate_citations,
    )

    # Collect explicit feedback
    feedback = await collect_feedback(query_id, FeedbackType.POSITIVE)

    # Mine implicit signals
    negatives = await mine_hard_negatives_from_session(session)

    # Stream a response
    async for event in stream_response(query, documents):
        print(event)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Protocol

from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Feedback Data Models
# =============================================================================


class FeedbackType(str, Enum):
    """Types of explicit user feedback."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    PARTIAL = "partial"


class NegativeFeedbackReason(str, Enum):
    """Specific reasons for negative feedback."""

    TOO_SLOW = "too_slow"
    WRONG_INFORMATION = "wrong_information"
    BAD_FORMAT = "bad_format"
    MISSING_INFORMATION = "missing_information"
    IRRELEVANT_SOURCES = "irrelevant_sources"
    HALLUCINATION = "hallucination"
    OFF_TOPIC = "off_topic"


class ImplicitSignalType(str, Enum):
    """Types of implicit user signals."""

    QUERY_REFINED = "query_refined"
    SESSION_ABANDONED = "session_abandoned"
    CITATION_CLICKED = "citation_clicked"
    CITATION_DELETED = "citation_deleted"
    COPY_PASTE = "copy_paste"
    REGENERATION_REQUESTED = "regeneration_requested"
    WORKFLOW_ACTIVATED = "workflow_activated"
    DWELL_TIME_SHORT = "dwell_time_short"
    DWELL_TIME_LONG = "dwell_time_long"


class UserFeedback(BaseModel):
    """Structured user feedback with full context.

    Attributes:
        query_id: Unique identifier for the query
        session_id: Session identifier for grouping interactions
        feedback_type: Type of feedback (positive, negative, partial)
        negative_reason: Specific reason if feedback is negative
        free_text: Optional free-text comment from user
        timestamp: When feedback was submitted
        user_id: Optional user identifier
        metadata: Additional context (tools used, entry point, etc.)
    """

    query_id: str
    session_id: str | None = None
    feedback_type: FeedbackType
    negative_reason: NegativeFeedbackReason | None = None
    free_text: str | None = None
    timestamp: datetime = datetime.now()
    user_id: str | None = None
    metadata: dict[str, Any] = {}


class ImplicitSignal(BaseModel):
    """Implicit signal derived from user behavior.

    Attributes:
        query_id: The query this signal relates to
        session_id: Session identifier
        signal_type: Type of implicit signal
        document_id: Document ID if signal is document-specific
        confidence: Confidence in the signal (0-1)
        timestamp: When signal was detected
        metadata: Additional context
    """

    query_id: str
    session_id: str
    signal_type: ImplicitSignalType
    document_id: str | None = None
    confidence: float = 1.0
    timestamp: datetime = datetime.now()
    metadata: dict[str, Any] = {}


class HardNegativeCandidate(BaseModel):
    """A candidate hard negative for training.

    Hard negatives are documents that appear relevant but are
    actually unhelpful - the most valuable training examples.

    Attributes:
        query: The original query
        document_id: ID of the document
        document_text: Text of the document
        signal_type: How we know it's a negative
        confidence: Confidence in the negative label
    """

    query: str
    document_id: str
    document_text: str | None = None
    signal_type: str
    confidence: float


# =============================================================================
# Session and Context Models
# =============================================================================


@dataclass
class QueryContext:
    """Full context for a query, used for feedback analysis.

    Attributes:
        query_id: Unique identifier
        query: The user's query text
        retrieved_docs: Documents retrieved for this query
        response: The generated response
        citations: Citations included in the response
        timestamp: When query was processed
    """

    query_id: str
    query: str
    retrieved_docs: list[dict[str, Any]]
    response: str
    citations: list[dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Session:
    """User session tracking for implicit signal mining.

    Attributes:
        session_id: Unique session identifier
        query: Original query
        initial_retrieved_docs: Documents from first retrieval
        final_cited_docs: Documents actually cited in final response
        deleted_citations: Citations user explicitly removed
        refined_queries: Subsequent query refinements
        regenerated: Whether user requested regeneration
        workflow_activated: Whether user activated a workflow
        dwell_time_seconds: Time spent on response
    """

    session_id: str
    query: str
    initial_retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    final_cited_docs: set[str] = field(default_factory=set)
    deleted_citations: list[dict[str, Any]] = field(default_factory=list)
    refined_queries: list[str] = field(default_factory=list)
    regenerated: bool = False
    workflow_activated: bool = False
    dwell_time_seconds: float | None = None


# =============================================================================
# Feedback Collection
# =============================================================================


class FeedbackStore(Protocol):
    """Protocol for feedback storage backends."""

    async def insert(self, feedback: UserFeedback) -> None:
        """Store feedback."""
        ...

    async def get_by_query_id(self, query_id: str) -> list[UserFeedback]:
        """Retrieve feedback for a query."""
        ...


class InMemoryFeedbackStore:
    """Simple in-memory feedback store for development.

    Attributes:
        feedback: List of stored feedback
    """

    def __init__(self):
        self.feedback: list[UserFeedback] = []

    async def insert(self, feedback: UserFeedback) -> None:
        """Store feedback in memory."""
        self.feedback.append(feedback)

    async def get_by_query_id(self, query_id: str) -> list[UserFeedback]:
        """Retrieve feedback for a query."""
        return [f for f in self.feedback if f.query_id == query_id]


# Global store (replace with actual database in production)
_feedback_store = InMemoryFeedbackStore()


async def collect_feedback(
    query_id: str,
    feedback_type: FeedbackType,
    negative_reason: NegativeFeedbackReason | None = None,
    free_text: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    slack_webhook_url: str | None = None,
) -> UserFeedback:
    """
    Collect structured feedback with optional follow-up.

    When feedback is negative, prompt for specific reason using
    checkboxes rather than free text for better analysis.

    Args:
        query_id: Unique identifier for the query
        feedback_type: Type of feedback (positive, negative, partial)
        negative_reason: Specific reason if feedback is negative
        free_text: Optional free-text comment
        user_id: Optional user identifier
        session_id: Optional session identifier
        metadata: Additional context
        slack_webhook_url: Optional Slack webhook for enterprise alerts

    Returns:
        The stored UserFeedback object

    Examples:
        >>> feedback = await collect_feedback(
        ...     query_id="q123",
        ...     feedback_type=FeedbackType.NEGATIVE,
        ...     negative_reason=NegativeFeedbackReason.WRONG_INFORMATION
        ... )
    """
    feedback = UserFeedback(
        query_id=query_id,
        session_id=session_id,
        feedback_type=feedback_type,
        negative_reason=negative_reason,
        free_text=free_text,
        user_id=user_id,
        metadata=metadata or {},
        timestamp=datetime.now(),
    )

    # Store for analysis and training
    await _feedback_store.insert(feedback)
    logger.info(f"Stored feedback for query {query_id}: {feedback_type.value}")

    # For enterprise: post to Slack for visibility
    if feedback_type == FeedbackType.NEGATIVE and slack_webhook_url:
        await post_feedback_to_slack(feedback, slack_webhook_url)

    return feedback


async def track_implicit_signal(
    query_id: str,
    session_id: str,
    signal_type: ImplicitSignalType,
    document_id: str | None = None,
    confidence: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> ImplicitSignal:
    """
    Track an implicit signal from user behavior.

    Implicit signals provide training data even when users don't
    provide explicit feedback.

    Args:
        query_id: The query this signal relates to
        session_id: Session identifier
        signal_type: Type of implicit signal
        document_id: Document ID if signal is document-specific
        confidence: Confidence in the signal (0-1)
        metadata: Additional context

    Returns:
        The tracked ImplicitSignal object
    """
    signal = ImplicitSignal(
        query_id=query_id,
        session_id=session_id,
        signal_type=signal_type,
        document_id=document_id,
        confidence=confidence,
        metadata=metadata or {},
        timestamp=datetime.now(),
    )

    logger.info(f"Tracked implicit signal: {signal_type.value} for query {query_id}")
    return signal


# =============================================================================
# Hard Negative Mining
# =============================================================================


async def mine_hard_negatives_from_session(
    session: Session,
) -> list[HardNegativeCandidate]:
    """
    Extract hard negative training examples from user behavior.

    Hard negatives are documents that appear relevant but are
    actually unhelpful - the most valuable training examples
    for improving retrieval quality.

    Args:
        session: Session object with user interaction data

    Returns:
        List of HardNegativeCandidate objects

    Examples:
        >>> session = Session(
        ...     session_id="s123",
        ...     query="password reset",
        ...     deleted_citations=[{"document_id": "doc1"}]
        ... )
        >>> negatives = await mine_hard_negatives_from_session(session)
    """
    candidates: list[HardNegativeCandidate] = []

    # Citation deletions are strong signals
    for deleted_citation in session.deleted_citations:
        candidates.append(
            HardNegativeCandidate(
                query=session.query,
                document_id=deleted_citation.get("document_id", ""),
                document_text=deleted_citation.get("text"),
                signal_type="citation_deleted",
                confidence=0.9,
            )
        )

    # Query refinements suggest retrieval failure
    if session.refined_queries:
        for doc in session.initial_retrieved_docs:
            doc_id = doc.get("id", "")
            if doc_id not in session.final_cited_docs:
                candidates.append(
                    HardNegativeCandidate(
                        query=session.query,
                        document_id=doc_id,
                        document_text=doc.get("text"),
                        signal_type="query_refined",
                        confidence=0.7,
                    )
                )

    # Regeneration requests indicate first response failed
    if session.regenerated:
        for doc in session.initial_retrieved_docs:
            doc_id = doc.get("id", "")
            if doc_id not in session.final_cited_docs:
                candidates.append(
                    HardNegativeCandidate(
                        query=session.query,
                        document_id=doc_id,
                        document_text=doc.get("text"),
                        signal_type="regenerated",
                        confidence=0.6,
                    )
                )

    # Short dwell time suggests unhelpful response
    if session.dwell_time_seconds and session.dwell_time_seconds < 5:
        for doc in session.initial_retrieved_docs:
            doc_id = doc.get("id", "")
            candidates.append(
                HardNegativeCandidate(
                    query=session.query,
                    document_id=doc_id,
                    document_text=doc.get("text"),
                    signal_type="short_dwell_time",
                    confidence=0.5,
                )
            )

    logger.info(f"Mined {len(candidates)} hard negative candidates from session")
    return candidates


# =============================================================================
# Slack Integration
# =============================================================================


async def post_feedback_to_slack(
    feedback: UserFeedback,
    webhook_url: str,
    query_context: QueryContext | None = None,
) -> None:
    """
    Post negative feedback to Slack for immediate visibility.

    This is the enterprise feedback pattern: create shared Slack
    channels with customer stakeholders and post negative feedback
    directly for discussion.

    Args:
        feedback: The feedback to post
        webhook_url: Slack webhook URL
        query_context: Optional query context for more detail
    """
    import httpx

    if feedback.feedback_type != FeedbackType.NEGATIVE:
        return

    message = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Negative Feedback Alert"},
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*User:* {feedback.user_id or 'Anonymous'}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Reason:* {feedback.negative_reason.value if feedback.negative_reason else 'Not specified'}",
                    },
                ],
            },
        ]
    }

    if query_context:
        message["blocks"].append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Query:* {query_context.query}"},
            }
        )

    if feedback.free_text:
        message["blocks"].append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Comment:* {feedback.free_text}"},
            }
        )

    message["blocks"].append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Full Context"},
                    "url": f"https://app.example.com/queries/{feedback.query_id}",
                }
            ],
        }
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=message)
            response.raise_for_status()
            logger.info(f"Posted feedback to Slack for query {feedback.query_id}")
    except Exception as e:
        logger.error(f"Failed to post to Slack: {e}")


# =============================================================================
# Streaming
# =============================================================================


@dataclass
class StreamEvent:
    """A single event in a streaming response.

    Attributes:
        event_type: Type of event (status, answer, citation, followup, done)
        data: Event data (varies by type)
    """

    event_type: str
    data: Any


async def stream_response(
    query: str,
    documents: list[dict[str, Any]],
    generate_fn: Callable[[str, list[dict]], AsyncIterator[str]] | None = None,
    interstitials: list[str] | None = None,
) -> AsyncIterator[StreamEvent]:
    """
    Stream a response with interstitials, answer, and citations.

    Streaming transforms the user experience from a binary
    "waiting/complete" pattern to a continuous flow.

    Args:
        query: The user's query
        documents: Retrieved documents
        generate_fn: Async generator function for answer tokens
        interstitials: Optional custom interstitial messages

    Yields:
        StreamEvent objects for each part of the response

    Examples:
        >>> async for event in stream_response("password reset", docs):
        ...     if event.event_type == "answer":
        ...         print(event.data, end="")
    """
    # Default interstitials
    if interstitials is None:
        interstitials = [
            "Searching documents...",
            f"Found {len(documents)} relevant sources",
            "Generating response...",
        ]

    # Stream interstitials during retrieval
    for message in interstitials:
        yield StreamEvent(event_type="status", data=message)
        await asyncio.sleep(0.1)

    # Stream the answer token by token
    if generate_fn:
        async for chunk in generate_fn(query, documents):
            yield StreamEvent(event_type="answer", data=chunk)
            await asyncio.sleep(0.01)
    else:
        # Simulate streaming for demo
        demo_response = "Based on the documents, here is the answer..."
        for word in demo_response.split():
            yield StreamEvent(event_type="answer", data=word + " ")
            await asyncio.sleep(0.05)

    # Stream citations after answer
    for i, doc in enumerate(documents[:5]):
        citation = {
            "id": f"cite_{i}",
            "document_id": doc.get("id", f"doc_{i}"),
            "title": doc.get("title", f"Document {i}"),
            "snippet": doc.get("text", "")[:200],
        }
        yield StreamEvent(event_type="citation", data=citation)

    # Stream follow-up questions
    followups = [
        "Would you like more details?",
        "Do you have any follow-up questions?",
    ]
    yield StreamEvent(event_type="followups", data=followups)

    yield StreamEvent(event_type="done", data=None)


def format_sse_event(event: StreamEvent) -> str:
    """
    Format a StreamEvent as a Server-Sent Event string.

    Args:
        event: The StreamEvent to format

    Returns:
        SSE-formatted string
    """
    data = json.dumps({"type": event.event_type, "data": event.data})
    return f"data: {data}\n\n"


def get_domain_interstitials(query_category: str) -> list[str]:
    """
    Return domain-specific interstitial messages.

    Meaningful interstitials build trust by showing users what
    is happening, and can make perceived wait times up to 40%
    shorter than actual wait times.

    Args:
        query_category: Category of the query (technical, legal, medical, etc.)

    Returns:
        List of interstitial messages
    """
    interstitials = {
        "technical": [
            "Scanning documentation and code repositories...",
            "Identifying relevant code examples and patterns...",
            "Analyzing technical specifications...",
        ],
        "legal": [
            "Searching legal databases and precedents...",
            "Reviewing relevant case law and statutes...",
            "Analyzing jurisdictional applicability...",
        ],
        "medical": [
            "Consulting medical literature and guidelines...",
            "Reviewing clinical studies and research papers...",
            "Analyzing treatment protocols...",
        ],
        "financial": [
            "Searching financial regulations and guidelines...",
            "Reviewing market data and reports...",
            "Analyzing compliance requirements...",
        ],
    }

    return interstitials.get(
        query_category,
        [
            "Processing your query...",
            "Searching for relevant information...",
            "Analyzing related documents...",
        ],
    )


# =============================================================================
# Citations
# =============================================================================


class Citation(BaseModel):
    """A citation linking response text to source document.

    Attributes:
        id: Unique citation identifier
        document_id: ID of the source document
        start_char: Start character position in source
        end_char: End character position in source
        cited_text: The text being cited
    """

    id: str
    document_id: str
    start_char: int
    end_char: int
    cited_text: str


def extract_citations(response: str) -> list[Citation]:
    """
    Extract citations from XML-tagged response.

    XML-based citations are the most reliable format:
    - Survives markdown parsing
    - Enables precise highlighting
    - Works well with fine-tuning
    - Handles abbreviations and technical language

    Args:
        response: Response text with XML citation tags

    Returns:
        List of Citation objects

    Examples:
        >>> text = 'The contract <cite id="doc123" start="450" end="467">requires 30 days notice</cite>.'
        >>> citations = extract_citations(text)
        >>> citations[0].cited_text
        'requires 30 days notice'
    """
    pattern = r'<cite id="([^"]+)" start="(\d+)" end="(\d+)">([^<]+)</cite>'
    citations = []

    for match in re.finditer(pattern, response):
        citations.append(
            Citation(
                id=match.group(1),
                document_id=match.group(1).split("_")[0],
                start_char=int(match.group(2)),
                end_char=int(match.group(3)),
                cited_text=match.group(4),
            )
        )

    return citations


async def validate_citations(
    citations: list[Citation],
    documents: dict[str, Any],
) -> list[Citation]:
    """
    Validate that cited text exists in source documents.

    Always validate citations against source documents before
    display. Hallucinated citations destroy trust.

    Args:
        citations: List of citations to validate
        documents: Dict mapping document_id to document content

    Returns:
        List of valid citations
    """
    valid_citations = []

    for citation in citations:
        doc = documents.get(citation.document_id)
        if doc:
            doc_content = doc.get("content", doc.get("text", ""))
            if citation.cited_text in doc_content:
                valid_citations.append(citation)
            else:
                logger.warning(
                    f"Citation {citation.id} text not found in document {citation.document_id}"
                )
        else:
            logger.warning(f"Document {citation.document_id} not found for citation")

    return valid_citations


def strip_citations(response: str) -> str:
    """
    Remove citation tags from response, keeping the cited text.

    Args:
        response: Response with XML citation tags

    Returns:
        Response with citations removed but text preserved
    """
    pattern = r'<cite id="[^"]+" start="\d+" end="\d+">([^<]+)</cite>'
    return re.sub(pattern, r"\1", response)


# =============================================================================
# Chain of Thought
# =============================================================================


def chain_of_thought_prompt(query: str, documents: list[dict[str, Any]]) -> str:
    """
    Create a prompt that encourages step-by-step reasoning.

    Chain-of-thought prompting typically provides a 10-15%
    performance improvement for classification and reasoning tasks.

    Args:
        query: The user's query
        documents: Retrieved documents

    Returns:
        Formatted prompt string
    """
    context = "\n\n".join(
        [f"DOCUMENT {i+1}: {doc.get('text', '')}" for i, doc in enumerate(documents)]
    )

    return f"""Answer the user's question based on the provided documents.
First, think step by step about how to answer using the documents.
Then provide your final answer.

Structure your response like this:
<thinking>
Your step-by-step reasoning process here...
</thinking>

<answer>
Your final answer here, with citations to specific documents...
</answer>

USER QUESTION: {query}

DOCUMENTS:
{context}
"""


def monologue_prompt(
    query: str, documents: list[dict[str, Any]], additional_context: str = ""
) -> str:
    """
    Create a prompt that encourages information reiteration.

    Monologuing - having the model explicitly reiterate key
    information before generating a response - improves reasoning
    without complex architectural changes.

    Args:
        query: The user's query
        documents: Retrieved documents
        additional_context: Additional context (e.g., pricing data)

    Returns:
        Formatted prompt string
    """
    context = "\n\n".join(
        [f"DOCUMENT {i+1}: {doc.get('text', '')}" for i, doc in enumerate(documents)]
    )

    return f"""Generate a response based on the documents and context provided.

First, reiterate the key variables and constraints from the documents.
Then, identify specific parts of the documents that relate to the question.
Next, determine which information is most relevant.
Finally, provide your response with justification.

QUESTION: {query}

DOCUMENTS:
{context}

{f"ADDITIONAL CONTEXT:{chr(10)}{additional_context}" if additional_context else ""}

MONOLOGUE AND ANSWER:
"""


def extract_thinking_and_answer(response: str) -> tuple[str, str]:
    """
    Extract thinking and answer sections from chain-of-thought response.

    Args:
        response: Response with <thinking> and <answer> tags

    Returns:
        Tuple of (thinking, answer) strings
    """
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

    thinking = thinking_match.group(1).strip() if thinking_match else ""
    answer = answer_match.group(1).strip() if answer_match else response

    return thinking, answer


# =============================================================================
# Validation Patterns
# =============================================================================


class ValidationResult(BaseModel):
    """Result of a validation check.

    Attributes:
        is_valid: Whether validation passed
        issues: List of issues found
        suggestions: Suggestions for fixing issues
    """

    is_valid: bool
    issues: list[str] = []
    suggestions: list[str] = []


async def validate_urls_in_response(
    response: str,
    allowed_domains: list[str] | None = None,
    check_reachability: bool = True,
) -> ValidationResult:
    """
    Validate that all URLs in response are valid and reachable.

    Args:
        response: Response text to validate
        allowed_domains: Optional list of allowed domains
        check_reachability: Whether to check if URLs are reachable

    Returns:
        ValidationResult with issues found
    """
    import httpx
    from urllib.parse import urlparse

    url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
    urls = re.findall(url_pattern, response)

    issues = []
    suggestions = []

    for url in urls:
        parsed = urlparse(url)

        # Check domain allowlist
        if allowed_domains and parsed.netloc not in allowed_domains:
            issues.append(f"URL {url} contains disallowed domain {parsed.netloc}")
            suggestions.append(f"Replace with URL from: {', '.join(allowed_domains)}")
            continue

        # Check reachability
        if check_reachability:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.head(url, timeout=3, follow_redirects=True)
                    if resp.status_code >= 400:
                        issues.append(f"URL {url} returned status {resp.status_code}")
                        suggestions.append("Remove or replace this URL")
            except Exception as e:
                issues.append(f"URL {url} failed to connect: {str(e)}")
                suggestions.append("Remove or replace this URL")

    return ValidationResult(
        is_valid=len(issues) == 0,
        issues=issues,
        suggestions=suggestions,
    )


async def validate_citations_exist(
    response: str,
    documents: dict[str, Any],
) -> ValidationResult:
    """
    Validate that all citations reference existing documents.

    Args:
        response: Response text with citations
        documents: Available documents

    Returns:
        ValidationResult with issues found
    """
    citations = extract_citations(response)
    valid_citations = await validate_citations(citations, documents)

    invalid_count = len(citations) - len(valid_citations)
    issues = []
    suggestions = []

    if invalid_count > 0:
        issues.append(f"{invalid_count} citations reference non-existent content")
        suggestions.append("Regenerate response with valid citations only")

    return ValidationResult(
        is_valid=invalid_count == 0,
        issues=issues,
        suggestions=suggestions,
    )


async def regenerate_if_invalid(
    query: str,
    initial_response: str,
    validators: list[Callable[[str], ValidationResult]],
    generate_fn: Callable[[str], str],
    max_retries: int = 2,
) -> str:
    """
    Validate response and regenerate if issues found.

    Validation both catches errors and creates training data.
    Each correction becomes a learning opportunity.

    Args:
        query: Original query
        initial_response: Initial response to validate
        validators: List of validation functions
        generate_fn: Function to regenerate response
        max_retries: Maximum regeneration attempts

    Returns:
        Valid response (or best attempt after max retries)
    """
    response = initial_response

    for attempt in range(max_retries + 1):
        all_issues = []

        for validator in validators:
            result = await validator(response)
            if not result.is_valid:
                all_issues.extend(result.issues)

        if not all_issues:
            return response

        if attempt < max_retries:
            issues_text = "\n".join(all_issues)
            regeneration_prompt = f"""
The previously generated response contained issues:
{issues_text}

Please regenerate, fixing these issues.

Original request: {query}
"""
            response = await generate_fn(regeneration_prompt)
            logger.info(f"Regenerated response (attempt {attempt + 2})")

    logger.warning(f"Max retries reached, returning best attempt")
    return response


# =============================================================================
# Strategic Rejection
# =============================================================================


@dataclass
class RejectionDecision:
    """Decision about whether to reject a query.

    Attributes:
        should_reject: Whether to reject the query
        reason: Human-readable reason for rejection
        confidence: Confidence in the decision
        alternative_suggestion: Suggested alternative action
    """

    should_reject: bool
    reason: str | None = None
    confidence: float = 1.0
    alternative_suggestion: str | None = None


async def should_reject_query(
    query: str,
    query_classifier: Callable[[str], str] | None = None,
    complexity_assessor: Callable[[str], float] | None = None,
    confidence_threshold: float = 0.85,
) -> RejectionDecision:
    """
    Determine if a query should be politely rejected.

    Strategic rejection builds trust - users prefer "I don't know"
    to confidently wrong answers.

    Args:
        query: The user's query
        query_classifier: Function to classify query type
        complexity_assessor: Function to assess query complexity
        confidence_threshold: Minimum confidence to proceed

    Returns:
        RejectionDecision with recommendation
    """
    # Default implementations for demo
    if query_classifier is None:

        def query_classifier(q):
            return "general"

    if complexity_assessor is None:

        def complexity_assessor(q):
            return 0.5

    query_category = query_classifier(query)
    query_complexity = complexity_assessor(query)

    # Simple heuristic: complex queries in certain categories may need rejection
    high_risk_categories = ["legal", "medical", "financial"]
    expected_confidence = 0.9 - (query_complexity * 0.3)

    if query_category in high_risk_categories:
        expected_confidence -= 0.1

    if expected_confidence < confidence_threshold:
        return RejectionDecision(
            should_reject=True,
            reason=(
                f"This appears to be a {query_category} question with "
                f"high complexity. Based on similar questions, "
                f"our confidence is {expected_confidence:.0%}, which is below "
                f"our threshold of {confidence_threshold:.0%}."
            ),
            confidence=1 - expected_confidence,
            alternative_suggestion=(
                "Would you like me to focus on a more specific aspect, "
                "or connect you with a specialist?"
            ),
        )

    return RejectionDecision(should_reject=False, confidence=expected_confidence)


# =============================================================================
# Slack Bot Patterns
# =============================================================================


async def handle_slack_message(
    client: Any,  # AsyncWebClient
    channel: str,
    thread_ts: str,
    query: str,
    process_fn: Callable[[str], str],
) -> None:
    """
    Handle a Slack message with progress indicators.

    Slack doesn't support true streaming, but you can create
    the illusion of progress through careful interaction design.

    Args:
        client: Slack AsyncWebClient
        channel: Channel ID
        thread_ts: Thread timestamp
        query: User's query
        process_fn: Function to process the query
    """
    # Acknowledge receipt immediately
    await client.reactions_add(channel=channel, timestamp=thread_ts, name="eyes")

    # Post progress update
    progress_msg = await client.chat_postMessage(
        channel=channel, thread_ts=thread_ts, text="Searching knowledge base..."
    )

    # Process query
    response = await process_fn(query)

    # Update with final response
    await client.chat_update(
        channel=channel, ts=progress_msg["ts"], text=response
    )

    # Mark as complete
    await client.reactions_add(
        channel=channel, timestamp=thread_ts, name="white_check_mark"
    )

    # Pre-fill feedback reactions (increases feedback by up to 5x)
    for emoji in ["thumbsup", "thumbsdown", "star"]:
        await client.reactions_add(
            channel=channel, timestamp=progress_msg["ts"], name=emoji
        )
