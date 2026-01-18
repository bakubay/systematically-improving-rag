---
title: "Chapter 3: Feedback Systems and UX"
description: "Build feedback flywheels that collect 5x more data, implement streaming for better perceived performance, and add quality-of-life improvements that transform RAG from occasionally useful to daily essential."
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - feedback
  - user experience
  - streaming
  - citations
  - chain of thought
  - validation
---

# Chapter 3: Feedback Systems and UX

## Chapter at a Glance

**Prerequisites**: Chapter 1 (evaluation framework), Chapter 2 (fine-tuning basics), basic web development

**What You Will Learn**:

- How to design feedback mechanisms that collect 5x more data
- Streaming techniques that reduce perceived latency by 40%
- Citation patterns that build trust and generate training data
- Chain-of-thought reasoning for 15-20% accuracy improvements
- Validation patterns that catch errors before users see them

**Case Study Reference**: Zapier (10 to 40 feedback submissions/day), Legal research team (50,000+ labeled examples from citations)

**Time to Complete**: 60-90 minutes

---

## Key Insight

**Good copy beats good UI—changing "How did we do?" to "Did we answer your question?" increases feedback rates by 5x.** The difference between 0.1% and 0.5% feedback is not just more data—it is the difference between flying blind and having a clear view of what works. Design every user interaction to potentially generate training data, stream everything to maintain engagement, and add validation layers that catch errors before they reach users.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Design high-visibility feedback mechanisms that increase collection rates from 0.1% to 0.5%
2. Implement streaming responses that make users perceive systems as 40% faster
3. Build interactive citation systems that generate training data from every interaction
4. Apply chain-of-thought reasoning to improve accuracy by 15-20%
5. Create validation patterns that catch errors before users see them
6. Know when to strategically reject work to build trust

---

## Introduction

In Chapter 1, we established evaluation frameworks with synthetic data. In Chapter 2, we learned how to convert that data into fine-tuned models. Now comes the critical question: how do you collect real user data to fuel the improvement flywheel?

Most RAG implementations focus exclusively on retrieval and generation while neglecting the infrastructure needed to collect and utilize user feedback. This is a mistake. Without robust feedback mechanisms, you are flying blind—unable to identify which aspects of your system perform well and which need enhancement.

This chapter covers three interconnected topics:

1. **Feedback Collection**: How to design mechanisms that collect 5x more data
2. **Streaming and Perceived Performance**: How to maintain engagement during processing
3. **Quality of Life Improvements**: Citations, chain of thought, and validation patterns

Each topic reinforces the others. Streaming keeps users engaged long enough to provide feedback. Citations create natural touchpoints for feedback collection. Validation ensures the feedback you collect reflects actual system quality, not random errors.

!!! tip "For Product Managers"
    This chapter establishes the user experience foundation for continuous improvement. Focus on the business impact of feedback collection rates, the ROI of streaming implementation, and how citations build trust. The technical implementation details matter less than understanding what each technique enables.

!!! tip "For Engineers"
    This chapter provides practical implementation patterns you will use daily. Pay close attention to the streaming code examples, citation formats, and validation patterns. These techniques directly impact both user experience and your ability to collect training data.

---

## Core Content

### Feedback Collection: Building Your Improvement Flywheel

The first principle of effective feedback collection is visibility. Your feedback mechanisms should be prominent and engaging, not hidden in dropdown menus or settings pages.

!!! tip "For Product Managers"
    **Why feedback collection matters**: Every piece of user feedback is potential training data. At 0.1% feedback rate, a system with 10,000 daily queries generates 10 labeled examples per day. At 0.5%, that same system generates 50 examples—enough to fine-tune models 5x faster.

    **Real numbers from production systems**:

    - Zapier increased feedback from 10 to 40+ submissions per day with better copy
    - 90% of follow-up emails accepted without edits when using structured feedback
    - 35% reduction in escalation rates when feedback gets specific
    - 5x more feedback with enterprise Slack integrations

    **The copy that works**:

    | Bad Copy | Good Copy |
    |----------|-----------|
    | "How did we do?" | "Did we answer your question?" |
    | "Rate your experience" | "Did this code solve your problem?" |
    | "Give feedback" | "Did we take the correct actions?" |

    The key is focusing on your core value proposition rather than generic satisfaction.

!!! tip "For Engineers"
    **Implementing high-visibility feedback**:

    ```python
    from pydantic import BaseModel
    from enum import Enum
    from typing import Optional
    from datetime import datetime

    class FeedbackType(str, Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        PARTIAL = "partial"

    class NegativeFeedbackReason(str, Enum):
        TOO_SLOW = "too_slow"
        WRONG_INFORMATION = "wrong_information"
        BAD_FORMAT = "bad_format"
        MISSING_INFORMATION = "missing_information"
        IRRELEVANT_SOURCES = "irrelevant_sources"

    class UserFeedback(BaseModel):
        query_id: str
        feedback_type: FeedbackType
        negative_reason: Optional[NegativeFeedbackReason] = None
        free_text: Optional[str] = None
        timestamp: datetime
        user_id: Optional[str] = None

    async def collect_feedback(
        query_id: str,
        feedback_type: FeedbackType,
        negative_reason: Optional[NegativeFeedbackReason] = None
    ) -> UserFeedback:
        """
        Collect structured feedback with optional follow-up.

        When feedback is negative, prompt for specific reason
        using checkboxes rather than free text.
        """
        feedback = UserFeedback(
            query_id=query_id,
            feedback_type=feedback_type,
            negative_reason=negative_reason,
            timestamp=datetime.now()
        )

        # Store for analysis and training
        await store_feedback(feedback)

        # For enterprise: post to Slack for visibility
        if feedback_type == FeedbackType.NEGATIVE:
            await post_to_slack_channel(feedback)

        return feedback
    ```

    **Key implementation details**:

    - Make buttons large and prominent (not hidden in corners)
    - Ask follow-up questions only after negative feedback
    - Use checkboxes for common issues rather than free text
    - Log the query, retrieved documents, and user response together

---

### Mining Implicit Feedback

While explicit feedback (ratings, comments) is valuable, users express opinions through their actions even when they do not provide direct feedback.

!!! tip "For Product Managers"
    **Implicit signals to track**:

    | Signal | What It Indicates | Training Value |
    |--------|-------------------|----------------|
    | Query refinements | Initial response was inadequate | Negative example |
    | Session abandonment | User gave up | Strong negative |
    | Citation clicks | User found source relevant | Positive signal |
    | Copy/paste actions | Response was useful | Strong positive |
    | Regeneration requests | First response failed | Negative example |
    | Workflow activation | System worked correctly | Strong positive |

    **The dating app insight**: Dating apps like Tinder and Hinge train excellent embedding models because they have high volume, clear binary signals (swipe right/left), and simple objectives (match prediction). Design your RAG interactions to generate training labels naturally in the same way.

!!! tip "For Engineers"
    **Mining hard negatives from user behavior**:

    ```python
    from typing import List
    from pydantic import BaseModel

    class HardNegativeCandidate(BaseModel):
        query: str
        document_id: str
        signal_type: str  # "citation_deleted", "query_refined", "regenerated"
        confidence: float

    async def mine_hard_negatives(
        session_id: str
    ) -> List[HardNegativeCandidate]:
        """
        Extract hard negative training examples from user behavior.

        Hard negatives are documents that appear relevant but
        are actually unhelpful—the most valuable training examples
        for improving retrieval quality.
        """
        session = await get_session(session_id)
        candidates = []

        # Citation deletions are strong signals
        for deleted_citation in session.deleted_citations:
            candidates.append(HardNegativeCandidate(
                query=session.query,
                document_id=deleted_citation.document_id,
                signal_type="citation_deleted",
                confidence=0.9
            ))

        # Query refinements suggest retrieval failure
        if session.refined_queries:
            original_docs = session.initial_retrieved_docs
            for doc in original_docs:
                if doc.id not in session.final_cited_docs:
                    candidates.append(HardNegativeCandidate(
                        query=session.query,
                        document_id=doc.id,
                        signal_type="query_refined",
                        confidence=0.7
                    ))

        return candidates
    ```

    **UI patterns for hard negative collection**:

    1. **Interactive citations**: Let users mark citations as irrelevant
    2. **Document filtering**: Show top documents, let users remove irrelevant ones
    3. **Regeneration after removal**: When users remove a citation and regenerate, that document becomes a hard negative

---

### Enterprise Feedback with Slack Integration

For B2B applications with dedicated customer success teams, Slack integration dramatically increases feedback collection.

!!! tip "For Product Managers"
    **The enterprise feedback pattern**:

    1. Create shared Slack channel with customer stakeholders
    2. Post negative feedback directly to the channel in real-time
    3. Allow your team to discuss issues and ask follow-up questions
    4. Document how feedback is addressed
    5. Report improvements during regular sync meetings

    This approach typically increases feedback by 5x compared to traditional forms while building trust through transparency.

!!! tip "For Engineers"
    **Slack webhook implementation**:

    ```python
    import httpx
    from typing import Optional

    async def post_feedback_to_slack(
        feedback: UserFeedback,
        webhook_url: str,
        channel: Optional[str] = None
    ):
        """
        Post negative feedback to Slack for immediate visibility.
        """
        if feedback.feedback_type != FeedbackType.NEGATIVE:
            return

        # Get context for the feedback
        query_context = await get_query_context(feedback.query_id)

        message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Negative Feedback Alert"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*User:* {feedback.user_id or 'Anonymous'}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Reason:* {feedback.negative_reason or 'Not specified'}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Query:* {query_context.query}"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "View Full Context"},
                            "url": f"https://app.example.com/queries/{feedback.query_id}"
                        }
                    ]
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=message)
    ```

---

### Streaming: The Ultimate Progress Indicator

Streaming transforms the user experience from a binary "waiting/complete" pattern to a continuous flow. Users can start reading while the system continues generating.

!!! tip "For Product Managers"
    **Why streaming matters**:

    - Users perceive animated progress bars as 11% faster even with identical wait times
    - Users will tolerate up to 8 seconds of waiting with visual feedback
    - Applications with engaging loading screens report higher satisfaction scores
    - Streaming increases feedback collection rates by 30-40%

    **The implementation timing decision**: If you are uncertain about implementing streaming, do it early. Migrating from non-streaming to streaming is significantly more complex than building with streaming from the start. Retrofitting can add weeks to your development cycle.

    **Only about 20% of companies implement streaming well**—but the ones that do see massive UX improvements.

!!! tip "For Engineers"
    **Basic streaming implementation**:

    ```python
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    import asyncio
    import json

    app = FastAPI()

    @app.post("/query/stream")
    async def stream_query_response(request: Request):
        """
        Stream a response with interstitials, answer, and citations.
        """
        data = await request.json()
        query = data.get("query")

        async def event_generator():
            # Stream interstitials during retrieval
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching documents...'})}\n\n"

            documents = await retrieve_documents(query)

            yield f"data: {json.dumps({'type': 'status', 'message': f'Found {len(documents)} relevant sources'})}\n\n"

            # Stream the answer token by token
            async for chunk in generate_answer_stream(query, documents):
                yield f"data: {json.dumps({'type': 'answer', 'content': chunk})}\n\n"
                await asyncio.sleep(0.01)

            # Stream citations after answer
            citations = extract_citations(documents)
            for citation in citations:
                yield f"data: {json.dumps({'type': 'citation', 'data': citation})}\n\n"

            # Stream follow-up questions
            followups = await generate_followups(query)
            yield f"data: {json.dumps({'type': 'followups', 'questions': followups})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    ```

    **What to stream**:

    - Interstitials explaining what is happening
    - Answer tokens as they generate
    - Citations as separate structured data
    - Follow-up questions
    - Function call arguments (for agentic systems)

---

### Meaningful Interstitials

Generic loading indicators waste an opportunity. Meaningful interstitials build trust by showing users what is happening.

!!! tip "For Product Managers"
    **Generic vs meaningful interstitials**:

    | Generic (Bad) | Meaningful (Good) |
    |---------------|-------------------|
    | "Loading..." | "Searching 382,549 documents in our knowledge base..." |
    | "Please wait" | "Finding relevant precedent cases from 2021-2022..." |
    | "Processing" | "Analyzing 3 legal frameworks that might apply..." |

    Meaningful interstitials can make perceived wait times up to 40% shorter than actual wait times.

!!! tip "For Engineers"
    **Domain-specific interstitials**:

    ```python
    def get_interstitials(query_category: str) -> list[str]:
        """
        Return domain-specific interstitial messages.
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
        }

        return interstitials.get(query_category, [
            "Processing your query...",
            "Searching for relevant information...",
            "Analyzing related documents..."
        ])
    ```

---

### Skeleton Screens

Skeleton screens are placeholder UI elements that mimic the structure of content while it loads. They create the impression that content is almost ready.

!!! tip "For Product Managers"
    **Facebook's research**: Skeleton screens significantly reduced perceived load times, resulting in better user retention and engagement. Users reported that the experience "felt faster" even when actual load times were identical to spinner-based approaches.

    Skeleton screens work because they:

    1. Set clear expectations about what content is loading
    2. Provide a sense of progress without requiring actual progress data
    3. Create the impression that the system is actively working
    4. Give users visual stimulation during the waiting period

!!! tip "For Engineers"
    For RAG applications, skeleton screens can show:

    - The structure of the answer before content loads
    - Citation placeholders that will be filled
    - Follow-up question button outlines
    - Tool usage summaries that will appear

---

### Platform-Specific Streaming: Slack Bots

Slack does not support true streaming, but you can create the illusion of progress through careful interaction design.

!!! tip "For Engineers"
    **Slack bot pattern**:

    1. React with eyes emoji immediately to acknowledge receipt
    2. Use threaded updates to show progress
    3. Mark completion with checkmark emoji
    4. Pre-fill feedback reactions (thumbs up, thumbs down, star)

    ```python
    from slack_sdk.web.async_client import AsyncWebClient

    async def handle_slack_message(
        client: AsyncWebClient,
        channel: str,
        thread_ts: str,
        query: str
    ):
        # Acknowledge receipt immediately
        await client.reactions_add(
            channel=channel,
            timestamp=thread_ts,
            name="eyes"
        )

        # Post progress update
        progress_msg = await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text="Searching knowledge base..."
        )

        # Process query
        response = await process_query(query)

        # Update with final response
        await client.chat_update(
            channel=channel,
            ts=progress_msg["ts"],
            text=response.answer
        )

        # Mark as complete
        await client.reactions_add(
            channel=channel,
            timestamp=thread_ts,
            name="white_check_mark"
        )

        # Pre-fill feedback reactions
        for emoji in ["thumbsup", "thumbsdown", "star"]:
            await client.reactions_add(
                channel=channel,
                timestamp=progress_msg["ts"],
                name=emoji
            )
    ```

    Pre-filling emoji reactions increases feedback collection by up to 5x compared to no reactions.

---

### Citations: Building Trust and Collecting Feedback

Citations serve multiple purposes: they build trust, provide transparency, and create opportunities for feedback collection.

!!! tip "For Product Managers"
    **Why citations matter**:

    - Users want to know where information comes from
    - Citations show what data is being used to generate responses
    - Interactive citations create opportunities for document-level relevance signals

    **Real results from a legal research team**:

    - 50,000+ labeled examples collected for fine-tuning
    - User satisfaction increased from 67% to 89%
    - Citation accuracy improved from 73% to 91% through feedback loops
    - Attorney trust scores increased by 45%

!!! tip "For Engineers"
    **XML-based citation pattern** (most reliable):

    ```xml
    According to the contract, <cite id="doc123" start="450" end="467">
    the termination clause requires 30 days notice</cite> and
    <cite id="doc124" start="122" end="134">includes a penalty
    fee of $10,000</cite>.
    ```

    **Benefits of XML citations**:

    - Survives markdown parsing
    - Enables precise highlighting
    - Works well with fine-tuning
    - Handles abbreviations and technical language

    **Implementation**:

    ```python
    from pydantic import BaseModel
    from typing import List
    import re

    class Citation(BaseModel):
        id: str
        document_id: str
        start_char: int
        end_char: int
        cited_text: str

    def extract_citations(response: str) -> List[Citation]:
        """
        Extract citations from XML-tagged response.
        """
        pattern = r'<cite id="([^"]+)" start="(\d+)" end="(\d+)">([^<]+)</cite>'
        citations = []

        for match in re.finditer(pattern, response):
            citations.append(Citation(
                id=match.group(1),
                document_id=match.group(1).split("_")[0],
                start_char=int(match.group(2)),
                end_char=int(match.group(3)),
                cited_text=match.group(4)
            ))

        return citations

    async def validate_citations(
        citations: List[Citation],
        documents: dict
    ) -> List[Citation]:
        """
        Validate that cited text exists in source documents.
        """
        valid_citations = []

        for citation in citations:
            doc = documents.get(citation.document_id)
            if doc and citation.cited_text in doc.content:
                valid_citations.append(citation)

        return valid_citations
    ```

    **Fine-tuning for citation accuracy**:

    - Train on 10,000+ examples of correct citations
    - Focus on common failure modes (wrong chunk, hallucinated citations)
    - Always validate citations against source documents before display

---

### Chain of Thought: Making Thinking Visible

Chain-of-thought prompting—asking the model to reason step by step before providing its final answer—typically provides a 10-15% performance improvement for classification and reasoning tasks.

!!! tip "For Product Managers"
    **Why chain of thought matters**:

    - Improves accuracy by 10-20% on complex reasoning tasks
    - Makes AI decision-making transparent to users
    - Creates natural loading interstitials during streaming
    - Builds trust by showing how conclusions were reached

    With models like Claude and GPT-4, chain of thought has become standard practice. Even without reasoning models like o1, implementing chain of thought in business-relevant ways is consistently one of the highest-impact changes.

!!! tip "For Engineers"
    **Chain of thought prompt structure**:

    ```python
    def chain_of_thought_prompt(query: str, documents: list) -> str:
        """
        Create a prompt that encourages step-by-step reasoning.
        """
        context = "\n\n".join([f"DOCUMENT: {doc.content}" for doc in documents])

        return f"""
        Answer the user's question based on the provided documents.
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
    ```

    **Streaming chain of thought as interstitial**:

    The thinking section can be streamed as a separate UI component, turning waiting time into a transparent window into how the system works through the problem.

---

### Monologues: Solving Context Management

When dealing with long contexts, language models often struggle with recall and processing all instructions. Monologuing—having the model explicitly reiterate key information before generating a response—improves reasoning without complex architectural changes.

!!! tip "For Product Managers"
    **When monologues help**:

    - Long documents where relevant information is scattered
    - Complex queries requiring synthesis from multiple sources
    - Tasks with many constraints or requirements

    **Case study: SaaS pricing quotes**

    A company needed to generate pricing quotes from sales call transcripts and a 15-page pricing document. Initial approach: provide both as context. Result: inconsistent quotes that missed key information.

    With monologue approach:

    1. Model first reiterates variables that determine pricing
    2. Then identifies relevant parts of transcript
    3. Then determines which pricing tiers apply
    4. Finally generates the quote

    Result: Quote accuracy improved from 62% to 94%. 90% of follow-up emails were accepted without edits.

!!! tip "For Engineers"
    **Monologue prompt structure**:

    ```python
    def monologue_prompt(query: str, documents: list, pricing_data: str) -> str:
        """
        Create a prompt that encourages information reiteration.
        """
        context = "\n\n".join([f"TRANSCRIPT: {doc.content}" for doc in documents])

        return f"""
        Generate a pricing quote based on the call transcript and pricing documentation.

        First, reiterate the key variables that determine pricing options.
        Then, identify specific parts of the transcript that relate to these variables.
        Next, determine which pricing options from the documentation are most relevant.
        Finally, provide a recommended pricing quote with justification.

        QUESTION: {query}

        TRANSCRIPT:
        {context}

        PRICING DOCUMENTATION:
        {pricing_data}

        MONOLOGUE AND ANSWER:
        """
    ```

    Monologues often replace complex agent architectures. Rather than building multi-stage processes, you can achieve similar results with a single well-constructed monologue prompt.

---

### Validation Patterns: Catching Errors Before Users

Validation patterns act as safety nets for your RAG system. For latency-insensitive applications, validators can significantly increase trust and satisfaction.

!!! tip "For Product Managers"
    **When to use validators**:

    - High-stakes domains where errors have significant consequences
    - Applications where users make important decisions based on output
    - Scenarios where specific constraints must be enforced
    - Cases where you need to increase user trust

    **Real example**: A marketing team built a system to generate personalized emails with links to case studies. About 4% of emails contained invalid URLs. After implementing URL validation with one retry, the error rate dropped to 0%. After fine-tuning on the corrections, the base error rate dropped to nearly zero—the model learned from its corrections.

!!! tip "For Engineers"
    **URL validation example**:

    ```python
    import re
    from urllib.parse import urlparse
    import httpx

    async def validate_urls_in_email(
        email_body: str,
        allowed_domains: list[str]
    ) -> tuple[bool, list[str]]:
        """
        Validate that all URLs are valid and from allowed domains.
        """
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        urls = re.findall(url_pattern, email_body)

        issues = []

        for url in urls:
            domain = urlparse(url).netloc
            if domain not in allowed_domains:
                issues.append(f"URL {url} contains disallowed domain {domain}")
                continue

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.head(url, timeout=3)
                    if response.status_code != 200:
                        issues.append(f"URL {url} returned status {response.status_code}")
            except Exception as e:
                issues.append(f"URL {url} failed to connect: {str(e)}")

        return len(issues) == 0, issues

    async def regenerate_if_invalid(
        query: str,
        initial_response: str,
        allowed_domains: list[str]
    ) -> str:
        """
        Validate and regenerate if URLs are problematic.
        """
        is_valid, issues = await validate_urls_in_email(
            initial_response, allowed_domains
        )

        if is_valid:
            return initial_response

        # Regenerate with specific guidance
        issues_text = "\n".join(issues)
        regeneration_prompt = f"""
        The previously generated response contained URL issues:
        {issues_text}

        Please regenerate, either:
        1. Removing problematic URLs entirely, or
        2. Replacing them with valid URLs from: {', '.join(allowed_domains)}

        Original request: {query}
        """

        return await generate_response(regeneration_prompt)
    ```

    **Key insight**: Validation both catches errors and creates training data. Each correction becomes a learning opportunity, gradually reducing the need for validation.

---

### Strategic Rejection of Work

One of the most overlooked strategies for improving reliability is knowing when to reject work. Rather than delaying deployment until all edge cases are solved, implement strategic rejection for scenarios where your system is not yet strong enough.

!!! tip "For Product Managers"
    **Why strategic rejection builds trust**:

    - Acknowledging limitations transparently builds confidence
    - Users prefer "I don't know" to confidently wrong answers
    - Rejection collects data about what users need
    - Allows deployment sooner while collecting data to improve

    **Example rejection message**:

    > "I notice you're asking about cross-jurisdictional implications of regulation X. Currently, I'm not confident in my ability to analyze multi-jurisdictional regulatory conflicts accurately. Would you like me to instead focus on the requirements within your primary jurisdiction, or connect you with a regulatory specialist?"

!!! tip "For Engineers"
    **Implementing strategic rejection**:

    ```python
    async def should_reject_query(
        query: str,
        confidence_threshold: float = 0.85
    ) -> tuple[bool, str | None]:
        """
        Determine if a query should be politely rejected.
        """
        query_category = await classify_query(query)
        query_complexity = await assess_complexity(query)
        expected_confidence = await predict_confidence(
            query, query_category, query_complexity
        )

        if expected_confidence < confidence_threshold:
            reason = (
                f"This appears to be a {query_category} question with "
                f"{query_complexity} complexity. Based on similar questions, "
                f"our confidence is {expected_confidence:.0%}, which is below "
                f"our threshold of {confidence_threshold:.0%}."
            )
            return True, reason

        return False, None
    ```

    Design rejection with precision-recall tradeoffs in mind—avoid rejecting questions you can actually answer well.

---

### Showcasing Capabilities

While RAG systems can theoretically answer a wide range of questions, most excel at particular types. Explicitly highlighting what your system does well guides users toward successful interactions.

!!! tip "For Product Managers"
    **Prompting the user, not just the model**:

    - Show suggested query types that leverage your strengths
    - Create UI elements that highlight special capabilities
    - Provide examples of successful interactions
    - Use white space to showcase specialized capabilities

    Perplexity provides a good example: their interface shows different capabilities (web search, academic papers, math equations) with specific UI elements, guiding users toward interactions that will be successful.

!!! tip "For Engineers"
    Implement capability showcasing through:

    - Dynamic suggestion generation based on system strengths
    - UI components that visually distinguish different capabilities
    - Example queries that demonstrate successful patterns
    - Clear labeling of experimental vs production-ready features

---

## Case Study Deep Dive

### Zapier Central: 4x Feedback Improvement

Zapier Central faced a common challenge: limited feedback despite active user engagement. Their feedback submission rates were around 10 per day, almost exclusively negative from frustrated users experiencing errors.

!!! tip "For Product Managers"
    **The change**: Instead of tiny, muted feedback buttons in the corner, they added a natural-looking chat message at the end of workflow tests asking: "Did this run do what you expected it to do?"

    **The results**:

    - Feedback submissions increased from 10 to 40 per day (4x improvement)
    - Started receiving substantial positive feedback (previously almost non-existent)
    - Built evaluation suite from 23 to 383 evaluations based on real interactions
    - Could make informed decisions about model upgrades

    **Why it worked**:

    1. Positioning: Request appeared as natural part of conversation
    2. Timing: Asked immediately after interaction while context was fresh
    3. Specificity: "Did this do what you expected?" is clearer than "How did we do?"
    4. Visibility: Larger buttons made the action obvious

!!! tip "For Engineers"
    **Implementation details**:

    - Built internal feedback triaging system where all submissions land
    - Implemented "labeling parties"—weekly team meetings to categorize feedback
    - Added extensive metadata (tools used, context, entry point)
    - Created tooling to easily convert feedback into formal evaluations

    **Mining implicit feedback**:

    - Workflow activation signals (user tests then activates = positive)
    - Tool call validation errors (likely LLM mistake = negative)
    - Follow-up message analysis (rephrasing = previous response inadequate)
    - Hallucination detection (pattern matching for common hallucination phrases)

---

### Legal Research Team: 50,000+ Labeled Examples

A legal research team implemented interactive citations for their in-house attorneys. Each response included citations linked to specific case law or statutes.

!!! tip "For Product Managers"
    **The approach**:

    - Attorneys could click citations to see full context
    - Could mark citations as relevant or irrelevant
    - When marked irrelevant, system would regenerate without that source

    **The results**:

    - 50,000+ labeled examples collected for fine-tuning
    - User satisfaction: 67% to 89% (+22 percentage points)
    - Citation accuracy: 73% to 91% through feedback loops
    - Attorney trust scores increased by 45%

!!! tip "For Engineers"
    **Technical implementation**:

    - XML-based citation format with chunk IDs and text spans
    - Validation layer verifying cited text exists in referenced chunks
    - Fine-tuning on citation-specific tasks reduced errors from 4% to 0.1%
    - Special handling for legal abbreviations and technical language

---

## Implementation Guide

### Quick Start for PMs

**Week 1: Audit Current Feedback**

1. Measure current feedback collection rate
2. Review feedback copy—is it specific to your value proposition?
3. Identify where feedback buttons are hidden
4. List implicit signals you could be tracking

**Week 2: Implement Quick Wins**

1. Change feedback copy to be specific ("Did we answer your question?")
2. Make feedback buttons larger and more prominent
3. Add follow-up questions for negative feedback
4. Set up basic logging of user interactions

**Week 3: Plan Streaming and Citations**

1. Assess current latency and user abandonment rates
2. Prioritize streaming implementation if not already in place
3. Design citation format appropriate for your domain
4. Plan validation patterns for high-stakes outputs

### Detailed Implementation for Engineers

**Phase 1: Feedback Infrastructure (1-2 weeks)**

```python
# 1. Define feedback schema
class FeedbackEvent(BaseModel):
    query_id: str
    session_id: str
    feedback_type: FeedbackType
    negative_reason: Optional[NegativeFeedbackReason]
    timestamp: datetime
    metadata: dict  # tools used, entry point, etc.

# 2. Set up storage
async def store_feedback(event: FeedbackEvent):
    # Store in database for analysis
    await db.feedback.insert(event.dict())

    # Post to Slack for enterprise customers
    if event.feedback_type == FeedbackType.NEGATIVE:
        await post_to_slack(event)

# 3. Implement implicit signal tracking
async def track_implicit_signals(session: Session):
    signals = []

    if session.query_refined:
        signals.append(("query_refined", session.original_query))

    if session.regenerated:
        signals.append(("regenerated", session.original_response))

    for deleted in session.deleted_citations:
        signals.append(("citation_deleted", deleted.document_id))

    return signals
```

**Phase 2: Streaming Implementation (2-3 weeks)**

```python
# 1. Backend streaming endpoint
@app.post("/query/stream")
async def stream_response(request: QueryRequest):
    async def generate():
        # Stream interstitials
        yield sse_event("status", "Searching documents...")

        docs = await retrieve(request.query)
        yield sse_event("status", f"Found {len(docs)} sources")

        # Stream answer
        async for chunk in generate_answer(request.query, docs):
            yield sse_event("answer", chunk)

        # Stream citations
        for citation in extract_citations(docs):
            yield sse_event("citation", citation)

        yield sse_event("done", None)

    return StreamingResponse(generate(), media_type="text/event-stream")

# 2. Frontend handling (React example)
# const eventSource = new EventSource('/query/stream');
# eventSource.onmessage = (event) => {
#     const data = JSON.parse(event.data);
#     switch(data.type) {
#         case 'status': setStatus(data.message); break;
#         case 'answer': setAnswer(prev => prev + data.content); break;
#         case 'citation': setCitations(prev => [...prev, data.data]); break;
#     }
# };
```

**Phase 3: Quality of Life (1-2 weeks)**

```python
# 1. Citation validation
async def validate_and_filter_citations(
    response: str,
    documents: dict
) -> str:
    citations = extract_citations(response)
    valid = await validate_citations(citations, documents)

    if len(valid) < len(citations):
        # Log invalid citations for analysis
        await log_invalid_citations(citations, valid)

    return response  # Or regenerate if too many invalid

# 2. Chain of thought wrapper
async def generate_with_cot(query: str, documents: list) -> Response:
    prompt = chain_of_thought_prompt(query, documents)
    raw_response = await generate(prompt)

    # Parse thinking and answer sections
    thinking = extract_section(raw_response, "thinking")
    answer = extract_section(raw_response, "answer")

    return Response(
        thinking=thinking,  # Can be shown as expandable section
        answer=answer,
        citations=extract_citations(answer)
    )

# 3. Validation layer
async def generate_with_validation(
    query: str,
    validators: list[Validator]
) -> str:
    response = await generate(query)

    for validator in validators:
        is_valid, issues = await validator.validate(response)
        if not is_valid:
            response = await regenerate_with_feedback(query, issues)

    return response
```

---

## Common Pitfalls

### PM Pitfalls

!!! warning "PM Pitfall: Generic Feedback Copy"
    **The mistake**: Using vague questions like "How did we do?" or "Rate your experience."

    **Why it fails**: Users do not know what aspect to evaluate. Responses are vague and uncorrelated with actual system performance.

    **The fix**: Use specific questions aligned with your value proposition. "Did we answer your question?" for Q&A systems. "Did we take the correct actions?" for agentic systems.

!!! warning "PM Pitfall: Hidden Feedback Mechanisms"
    **The mistake**: Placing feedback buttons in corners or dropdown menus.

    **Why it fails**: Users will not find them. You collect 0.1% feedback instead of 0.5%.

    **The fix**: Make feedback impossible to miss. Place it directly after responses. Use large, prominent buttons.

!!! warning "PM Pitfall: Ignoring Implicit Signals"
    **The mistake**: Only tracking explicit thumbs up/down feedback.

    **Why it fails**: You miss 90%+ of user signals. Query refinements, abandonment, and citation interactions are valuable data.

    **The fix**: Track all user behaviors that indicate satisfaction or dissatisfaction.

### Engineering Pitfalls

!!! warning "Engineering Pitfall: Retrofitting Streaming"
    **The mistake**: Building without streaming, planning to add it later.

    **Why it fails**: Migrating from non-streaming to streaming is significantly more complex than building with streaming from the start. Can add weeks to development.

    **The fix**: Implement streaming from day one, even if basic.

!!! warning "Engineering Pitfall: Unvalidated Citations"
    **The mistake**: Displaying citations without verifying they exist in source documents.

    **Why it fails**: Hallucinated citations destroy trust. Users will stop believing any citations.

    **The fix**: Always validate that cited text exists in referenced documents before display.

!!! warning "Engineering Pitfall: No Feedback Context"
    **The mistake**: Storing feedback without the query, retrieved documents, and response.

    **Why it fails**: You cannot analyze why feedback was negative or use it for training.

    **The fix**: Log complete context with every feedback event.

---

## Related Content

### Talks

**How Zapier 4x'd Their AI Feedback Collection (Vitor)**

Key insights:

- Positioning, visibility, and wording of feedback requests dramatically impacts response rates
- Mining implicit feedback from workflow activations, validation errors, and follow-up messages
- "Labeling parties" for team-wide feedback analysis

[See the full talk summary](../talks/zapier-vitor-evals.md)

**Why Your AI Is Failing in Production (Ben & Sidhant)**

Key insights:

- Traditional error monitoring does not work for AI—there is no exception when something goes wrong
- The Trellis framework for organizing AI outputs into controllable segments
- Implicit signals (user frustration, task failures) vs explicit signals (ratings, regenerations)

[See the full talk summary](../talks/online-evals-production-monitoring-ben-sidhant.md)

### Office Hours

**Cohort 2, Week 3**: Negative feedback handling, feedback lifecycle management, citation and UX best practices

**Cohort 3, Week 3**: Re-ranking models, user feedback integration, compute allocation strategies

---

## Action Items

### For Product Teams

1. **Audit current feedback collection** - Measure your current rate and identify quick wins
2. **Rewrite feedback copy** - Make it specific to your value proposition
3. **Plan enterprise feedback loops** - Consider Slack integration for B2B customers
4. **Define implicit signals to track** - Query refinements, abandonment, citation interactions
5. **Establish feedback-driven roadmap process** - Regular review cycles with engineering

### For Engineering Teams

1. **Implement streaming** - If not already in place, prioritize this
2. **Add meaningful interstitials** - Replace generic loading with domain-specific messages
3. **Build citation validation** - Never display unvalidated citations
4. **Set up feedback logging** - Store complete context with every feedback event
5. **Create hard negative mining pipeline** - Extract training data from user behavior

---

## Reflection Questions

1. What is your current feedback collection rate? What would 5x more feedback enable?

2. How visible are your feedback mechanisms? Could a new user find them in 5 seconds?

3. What implicit signals are you not currently tracking that could provide training data?

4. If you implemented streaming tomorrow, how would it change user experience?

5. What validation patterns would catch the most common errors in your system?

---

## Summary

### Key Takeaways for Product Managers

1. **Feedback copy matters more than UI** - "Did we answer your question?" beats "How did we do?" by 5x
2. **Streaming is table stakes** - Only 20% of companies do it well, but it dramatically improves UX
3. **Citations build trust and collect data** - Interactive citations can generate 50,000+ labeled examples
4. **Strategic rejection builds confidence** - "I don't know" is better than confidently wrong

### Key Takeaways for Engineers

1. **Implement streaming from day one** - Retrofitting adds weeks to development
2. **Track implicit signals** - Query refinements, citation deletions, and regenerations are valuable training data
3. **Validate citations before display** - Hallucinated citations destroy trust
4. **Use chain of thought for complex reasoning** - 10-20% accuracy improvement with minimal effort
5. **Build validation layers** - Catch errors before users see them, create training data from corrections

---

## Further Reading

1. Nielsen Norman Group, ["Progress Indicators Make a Slow System Less Insufferable"](https://www.nngroup.com/articles/progress-indicators/)

2. Facebook Engineering, ["Building Skeleton Screens"](https://engineering.fb.com/2016/06/30/web/shimmer-an-open-source-library-for-loading-content/)

3. OpenAI Documentation, ["Streaming API Best Practices"](https://platform.openai.com/docs/guides/chat/streaming)

4. Anthropic, ["Constitutional AI: Harmlessness from AI Feedback"](https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback)

---

## Navigation

- **Previous**: [Chapter 2: Training Data and Fine-Tuning](chapter2.md) - Converting evaluations into training data
- **Next**: [Chapter 4: Query Understanding and Prioritization](chapter4.md) - Finding patterns in user data
- **Reference**: [Glossary](glossary.md) | [Quick Reference](quick-reference.md)
- **Book Index**: [Book Overview](index.md)
