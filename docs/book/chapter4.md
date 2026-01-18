---
title: "Chapter 4: Query Understanding and Prioritization"
description: "Transform raw query data into actionable insights through clustering and segmentation, then use prioritization frameworks to focus engineering effort where it matters most."
authors:
  - Jason Liu
date: 2025-01-17
tags:
  - query clustering
  - topic modeling
  - prioritization
  - segmentation
  - economic analysis
  - roadmapping
---

# Chapter 4: Query Understanding and Prioritization

## Chapter at a Glance

**Prerequisites**: Chapter 1 (evaluation framework), Chapter 3 (feedback collection), basic understanding of clustering algorithms

**What You Will Learn**:

- How to segment queries into actionable clusters using K-means and embeddings
- The 2x2 prioritization matrix for identifying high-impact improvements
- Distinguishing inventory issues (missing data) from capability issues (missing features)
- The Expected Value formula for data-driven prioritization
- How to detect user adaptation patterns that mask system failures
- Building production classification systems for real-time query routing

**Case Study Reference**: Construction project management (35% retention improvement), Voice AI restaurant system ($2M revenue opportunity), Customer support RAG (28% support ticket reduction)

**Time to Complete**: 60-90 minutes

---

## Key Insight

**Not all query failures are equal—fixing 20% of segments can solve 80% of user problems.** Segmentation transforms vague complaints into actionable insights. Use the 2x2 matrix (volume vs satisfaction) to identify your danger zones: high-volume, low-satisfaction segments that are killing your product. The formula is simple: Expected Value = Impact x Volume % x Success Rate. When retrieval fails, ask: is the information missing (inventory) or can we not process it correctly (capability)? Knowing the difference saves months of misdirected effort.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Apply the 80/20 rule to RAG improvement by identifying how fixing 20% of query segments can solve 80% of user problems
2. Build query segmentation systems that transform user feedback into actionable segments using K-means clustering
3. Master the 2x2 prioritization matrix to identify danger zones (high volume, low satisfaction) requiring immediate attention
4. Implement the Expected Value formula (Impact x Volume % x Success Rate) for data-driven prioritization
5. Distinguish inventory issues from capability issues and apply the appropriate solution strategy
6. Detect user adaptation patterns that mask system failures and prevent misleading satisfaction metrics

---

## Introduction

In Chapter 3, you built feedback collection systems that generate thousands of queries, ratings, and user signals. Your manager asks "What should we improve next?" and suddenly the abundance of data becomes overwhelming. Which patterns matter? Which improvements will move the needle?

This is a common challenge. Organizations collect extensive feedback but lack systematic approaches for finding actionable patterns. That company with 30 evaluations from Chapter 1? They now have thousands of real queries. But without segmentation, they are drowning in data instead of surfacing insights.

**Where We Have Been:**

- **Chapter 1**: Built evaluation framework establishing baselines
- **Chapter 2**: Turned evaluations into training data for fine-tuning
- **Chapter 3**: Collected real user feedback at scale

**Now What?** Topic modeling and clustering transform raw feedback into actionable insights. Instead of reading thousands of queries individually, group similar patterns and identify the real problems worth fixing. Not all improvements matter equally—some query segments affect 80% of users, while others represent edge cases. Segmentation reveals which is which.

!!! tip "For Product Managers"
    This chapter establishes the analytical foundation for strategic roadmapping. Focus on the prioritization frameworks, the 2x2 matrix, and how to distinguish inventory from capability issues. These concepts will help you justify engineering investments and communicate priorities to stakeholders.

!!! tip "For Engineers"
    This chapter provides practical techniques for query analysis that directly inform what to build next. Pay attention to the clustering implementation, classification pipelines, and how to detect user adaptation patterns. These skills help you identify high-impact improvements rather than working on technically interesting but low-value problems.

---

## Core Content

### Why Segmentation Beats Random Improvements

Consider a marketing analogy that clarifies why segmentation matters. Imagine sales jump 80% in a quarter. Celebration is natural, but the question remains: why? Was it the Super Bowl advertisement? The packaging redesign? Seasonal trends? Pure luck?

Without segmentation, insights remain hidden. But with systematic analysis, patterns emerge. At Stitch Fix, when sales jumped 80%, segmentation revealed that 60% of the increase came specifically from women aged 30-45 in the Midwest. This insight was worth millions—it showed exactly where to double down marketing spend, which channels performed best for that demographic, and which product lines resonated most strongly.

!!! tip "For Product Managers"
    **The business case for segmentation**:

    Without segmentation: "70% satisfaction, we're doing okay."

    With segmentation:

    - Document search: 85% satisfaction (crushing it!)
    - Schedule queries: 35% satisfaction (yikes!)
    - Comparison queries: 60% satisfaction (fixable)

    Now you know where to focus. Remember from Chapter 2—systems at 70% can reach 85-90%. But you need to know which 70% to focus on first.

    **ROI of segmentation**:

    - Reduces wasted engineering effort by 40-60%
    - Identifies quick wins that would otherwise be missed
    - Provides data for stakeholder conversations
    - Enables accurate forecasting of improvement impact

!!! tip "For Engineers"
    **Technical rationale for segmentation**:

    Aggregate metrics hide important details. A system with 70% overall satisfaction might have:

    - 95% satisfaction on 30% of queries (already optimized)
    - 40% satisfaction on 45% of queries (high-impact opportunity)
    - 20% satisfaction on 15% of queries (potential quick win)
    - 85% satisfaction on 10% of queries (monitor only)

    Without segmentation, you might optimize the already-good 30% because those queries are easiest to understand. With segmentation, you focus on the 45% segment where improvements have 3x the impact.

---

### Query Clustering: From Raw Data to Insights

The process of query clustering is straightforward:

1. Embed all your queries
2. Use K-means clustering (start with 20 clusters)
3. Group similar queries together
4. Analyze patterns within each cluster

Do not overthink the clustering algorithm—simple K-means works fine. The insights come from manually reviewing the clusters, not from fancy algorithms.

!!! tip "For Product Managers"
    **What clustering reveals**:

    After clustering, you might discover:

    - "Password reset queries" (15% of queries, 90% satisfaction) → Monitor only
    - "Schedule lookup queries" (8% of queries, 25% satisfaction) → High priority
    - "Document search queries" (52% of queries, 70% satisfaction) → Moderate priority

    This systematic approach transforms raw query logs into actionable insights about where to invest development effort.

    **The 10-10 Rule**: For each cluster, manually review 10 queries with positive feedback and 10 queries with negative feedback. This tells you what is working and what is broken in that segment.

!!! tip "For Engineers"
    **Implementing query clustering**:

    ```python
    import numpy as np
    from sklearn.cluster import KMeans
    from openai import OpenAI
    from pydantic import BaseModel
    from typing import List

    class QueryCluster(BaseModel):
        cluster_id: int
        queries: List[str]
        centroid: List[float]
        size: int
        avg_satisfaction: float

    async def cluster_queries(
        queries: List[str],
        satisfaction_scores: List[float],
        n_clusters: int = 20
    ) -> List[QueryCluster]:
        """
        Cluster queries using embeddings and K-means.

        Start with 20 clusters and adjust based on results.
        More clusters = more specific segments.
        Fewer clusters = broader patterns.
        """
        client = OpenAI()

        # Generate embeddings
        embeddings = []
        for query in queries:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            embeddings.append(response.data[0].embedding)

        embeddings_array = np.array(embeddings)

        # Cluster with K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        # Build cluster objects
        clusters = []
        for i in range(n_clusters):
            mask = cluster_labels == i
            cluster_queries = [q for q, m in zip(queries, mask) if m]
            cluster_scores = [s for s, m in zip(satisfaction_scores, mask) if m]

            clusters.append(QueryCluster(
                cluster_id=i,
                queries=cluster_queries,
                centroid=kmeans.cluster_centers_[i].tolist(),
                size=len(cluster_queries),
                avg_satisfaction=np.mean(cluster_scores) if cluster_scores else 0.0
            ))

        return clusters
    ```

    **Advanced clustering with the Kura process**:

    For more sophisticated analysis of conversation history:

    1. **Summarize**: Create summaries of every conversation or query session
    2. **Extract**: Pull out languages, topics, tasks, requests, complaints, errors
    3. **Concatenate**: Combine extracted information into searchable text
    4. **Embed**: Create embeddings from concatenated text
    5. **Cluster**: Perform clustering on embeddings
    6. **Label**: Use LLMs to generate meaningful cluster names

    ```python
    async def label_clusters_with_llm(
        clusters: List[QueryCluster],
        samples_per_cluster: int = 10
    ) -> dict[int, str]:
        """
        Use an LLM to generate meaningful names for each cluster.
        """
        client = OpenAI()
        labels = {}

        for cluster in clusters:
            # Sample queries from cluster
            sample = cluster.queries[:samples_per_cluster]

            prompt = f"""Analyze these queries and provide:
            1. A short name (2-4 words) for this cluster
            2. A one-sentence description
            3. 3 good examples of what belongs in this cluster
            4. 2 examples of what does NOT belong

            Queries:
            {chr(10).join(f'- {q}' for q in sample)}
            """

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )

            labels[cluster.cluster_id] = response.choices[0].message.content

        return labels
    ```

---

### The 2x2 Prioritization Matrix

Once you have your segments, plot them on this matrix:

```
                    High Satisfaction
                          |
    Low Volume            |            High Volume
    High Satisfaction     |            High Satisfaction
    📢 PROMOTE FEATURES   |            ✅ MONITOR ONLY
                          |
    ----------------------+----------------------
                          |
    Low Volume            |            High Volume
    Low Satisfaction      |            Low Satisfaction
    🤔 COST-BENEFIT       |            🚨 DANGER ZONE
                          |
                    Low Satisfaction
```

!!! tip "For Product Managers"
    **What to do in each quadrant**:

    **High Volume + High Satisfaction (Monitor Only)**

    - You are doing great here
    - Set up alerts if performance drops
    - Use as examples of what works
    - Consider if you can break this down further

    **Low Volume + High Satisfaction (Promote Features)**

    - Users do not know you are good at this
    - Add UI hints showing these capabilities
    - Include in onboarding
    - Show example queries below search bar

    **High Volume + Low Satisfaction (DANGER ZONE)**

    - This is killing your product
    - Immediate priority for improvement
    - Conduct user research to understand why
    - Set sprint goals to fix this

    **Low Volume + Low Satisfaction (Cost-Benefit)**

    - Maybe you do not need to solve this
    - Could be out of scope
    - Consider explicitly saying "we don't do that"
    - Or find low-effort improvements

!!! tip "For Engineers"
    **Implementing the prioritization matrix**:

    ```python
    from enum import Enum
    from pydantic import BaseModel

    class Quadrant(str, Enum):
        MONITOR = "monitor"
        PROMOTE = "promote"
        DANGER_ZONE = "danger_zone"
        COST_BENEFIT = "cost_benefit"

    class SegmentPriority(BaseModel):
        cluster_id: int
        cluster_name: str
        volume_pct: float
        satisfaction: float
        quadrant: Quadrant
        priority_score: float
        recommended_action: str

    def classify_segment(
        volume_pct: float,
        satisfaction: float,
        volume_threshold: float = 0.10,  # 10% of queries
        satisfaction_threshold: float = 0.60  # 60% satisfaction
    ) -> Quadrant:
        """
        Classify a segment into the 2x2 matrix.
        """
        high_volume = volume_pct >= volume_threshold
        high_satisfaction = satisfaction >= satisfaction_threshold

        if high_volume and high_satisfaction:
            return Quadrant.MONITOR
        elif not high_volume and high_satisfaction:
            return Quadrant.PROMOTE
        elif high_volume and not high_satisfaction:
            return Quadrant.DANGER_ZONE
        else:
            return Quadrant.COST_BENEFIT

    def calculate_priority_score(
        impact: float,  # 1-10 scale
        volume_pct: float,
        success_rate: float,
        effort: float = 5.0,  # 1-10 scale
        risk: float = 2.0  # 1-5 scale
    ) -> float:
        """
        Calculate priority score using the Expected Value formula.

        Priority = (Impact × Volume %) / (Effort × Risk)

        Higher scores = higher priority.
        """
        return (impact * volume_pct * (1 - success_rate)) / (effort * risk)
    ```

---

### Topics vs Capabilities: Two Fundamental Dimensions

Most teams only segment by topic (what users ask about). That is a mistake. You need to segment by both topics AND capabilities (what users want the system to do).

**Topics** = What users ask about (account management, pricing, technical specs)

**Capabilities** = What they want the system to do (summarize, compare, explain step-by-step)

!!! tip "For Product Managers"
    **The healthcare example**:

    A healthcare company was categorizing everything by medical condition. Seemed logical, right? But when capability analysis was added:

    - **Common conditions** (diabetes, hypertension): Users mostly wanted comparisons between treatments
    - **Rare conditions**: Users needed comprehensive summaries of all options
    - **Emergency conditions**: Users needed step-by-step immediate actions

    Same topic dimension, completely different capability needs. This changed everything about what to build next.

    **Mapping topics to capabilities**:

    | Query | Topic | Capability |
    |-------|-------|------------|
    | "How do I reset my password?" | Account security | Step-by-step instructions |
    | "Compare the Pro and Basic plans" | Pricing | Comparison |
    | "Summarize the latest release notes" | Product updates | Summarization |
    | "What's the difference between 2022 and 2023 budgets?" | Financial data | Comparison + Temporal filtering |

    See how the same capability (like comparison) can apply to different topics? And the same topic can need different capabilities? That is why you need both dimensions.

!!! tip "For Engineers"
    **Detecting capabilities programmatically**:

    ```python
    from typing import List, Set
    from pydantic import BaseModel

    class CapabilityDetection(BaseModel):
        query: str
        detected_capabilities: Set[str]
        confidence: float

    CAPABILITY_PATTERNS = {
        "comparison": ["compare", "versus", "vs", "difference between", "better"],
        "summarization": ["summarize", "overview", "summary", "brief", "tldr"],
        "temporal_filtering": ["yesterday", "last week", "recent", "latest", "2023", "2024"],
        "aggregation": ["total", "sum", "average", "count", "how many"],
        "step_by_step": ["how to", "steps", "guide", "tutorial", "instructions"],
        "filtering": ["only", "filter by", "where", "that have", "with"],
    }

    def detect_capabilities(query: str) -> CapabilityDetection:
        """
        Detect required capabilities from query text.

        This is simple pattern matching—for production,
        consider using a trained classifier.
        """
        query_lower = query.lower()
        detected = set()

        for capability, patterns in CAPABILITY_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    detected.add(capability)
                    break

        return CapabilityDetection(
            query=query,
            detected_capabilities=detected,
            confidence=0.8 if detected else 0.5
        )
    ```

---

### Inventory vs Capability Issues: The Critical Distinction

When retrieval fails, ask: is the information missing (inventory) or can we not process it correctly (capability)? This distinction fundamentally changes how you approach improvements.

!!! tip "For Product Managers"
    **Inventory Issues: Missing Data**

    Think of inventory like a library. If someone asks for a book you do not have, that is an inventory problem. No amount of organization or search improvements will help—you need the book.

    **Real examples**:

    | Query | Issue | Solution |
    |-------|-------|----------|
    | "Spanish TV shows on Netflix" | No Spanish content indexed | Add Spanish content metadata |
    | "Greek restaurants near me" | No Greek restaurants in database | Onboard Greek restaurants |
    | "Q3 2024 financial results" | Data stops at Q2 2024 | Update data pipeline |
    | "Oscar-nominated movies" | No awards metadata | Pay IMDB for better metadata |

    **Capability Issues: Missing Features**

    Capability issues are like having all the books but no way to find them by publication date, or no ability to compare two books side-by-side.

    **Real examples**:

    | Query | Issue | Solution |
    |-------|-------|----------|
    | "Affordable shoes under 3-inch heels" | No heel height metadata | Extract and index heel heights |
    | "Compare 2022 vs 2023 revenue" | No comparison capability | Build comparison function |
    | "Documents modified yesterday" | No timestamp filtering | Add datetime metadata |
    | "Total spend by department" | No aggregation capability | Build SQL generation |

    **The decision framework**:

    1. **Ask: Does the information exist?**
       - Yes → Capability issue (add functionality)
       - No → Inventory issue (add data)

    2. **Ask: What is the business impact?**
       - High impact + simple solution = highest priority
       - High impact + complex solution = evaluate ROI
       - Low impact = deprioritize

!!! tip "For Engineers"
    **Detecting inventory vs capability issues**:

    ```python
    from enum import Enum
    from pydantic import BaseModel
    from typing import List, Optional

    class IssueType(str, Enum):
        INVENTORY = "inventory"
        CAPABILITY = "capability"
        BOTH = "both"
        UNKNOWN = "unknown"

    class IssueClassification(BaseModel):
        query: str
        issue_type: IssueType
        indicators: List[str]
        suggested_solution: str

    async def classify_issue(
        query: str,
        retrieval_results: List[dict],
        max_similarity: float
    ) -> IssueClassification:
        """
        Classify whether a query failure is inventory or capability.

        Inventory indicators:
        - Low cosine similarity (< 0.5)
        - Zero lexical search matches
        - No sources cited in response
        - LLM says "no information available"

        Capability indicators:
        - Data exists but can't be filtered correctly
        - Unable to perform requested operations
        - Missing metadata for filtering
        - Can't handle temporal queries
        """
        indicators = []

        # Check inventory indicators
        if max_similarity < 0.5:
            indicators.append("low_similarity")
        if len(retrieval_results) == 0:
            indicators.append("no_results")

        # Check capability indicators
        capabilities_needed = detect_capabilities(query)
        if "temporal_filtering" in capabilities_needed.detected_capabilities:
            if not any(r.get("has_timestamp") for r in retrieval_results):
                indicators.append("missing_temporal_metadata")
        if "comparison" in capabilities_needed.detected_capabilities:
            indicators.append("comparison_capability_needed")

        # Classify
        inventory_indicators = {"low_similarity", "no_results"}
        capability_indicators = {"missing_temporal_metadata", "comparison_capability_needed"}

        has_inventory = bool(set(indicators) & inventory_indicators)
        has_capability = bool(set(indicators) & capability_indicators)

        if has_inventory and has_capability:
            issue_type = IssueType.BOTH
        elif has_inventory:
            issue_type = IssueType.INVENTORY
        elif has_capability:
            issue_type = IssueType.CAPABILITY
        else:
            issue_type = IssueType.UNKNOWN

        return IssueClassification(
            query=query,
            issue_type=issue_type,
            indicators=indicators,
            suggested_solution=get_solution_suggestion(issue_type, indicators)
        )
    ```

    **Common capability gaps and solutions**:

    **Datetime Filtering**

    - Detection: Words like "yesterday", "last week", "recent", "latest"
    - Solution: Add timestamp metadata and range queries
    - Implementation: Use PostgreSQL with datetime indexes

    **Comparison**

    - Detection: "versus", "compare", "difference between"
    - Solution: Parallel retrieval + comparison prompt
    - Implementation: Retrieve for each entity, then synthesize

    **Aggregation**

    - Detection: "total", "sum", "average", "count"
    - Solution: SQL generation or structured extraction
    - Implementation: Text-to-SQL with validation

---

### User Adaptation: The Hidden Pattern

Users adapt to your system's limitations. High satisfaction in one area might be masking failures elsewhere. Always look at user journeys, not just aggregate metrics.

!!! tip "For Product Managers"
    **The construction company case study**:

    Initial data:

    - Document search: 52% of queries (70% satisfaction)
    - Scheduling: 8% of queries (25% satisfaction)
    - Cost lookup: 15% of queries (82% satisfaction)

    The product team thought scheduling was a minor issue—only 8% of queries. But when they looked at user cohorts over time:

    | User Cohort | Day 1 | Day 7 | Day 30 |
    |-------------|-------|-------|--------|
    | Scheduling queries | 90% | 60% | 20% |
    | Document search | 10% | 40% | 80% |

    **The hidden pattern**: Users were adapting to failures. They wanted scheduling but learned it did not work, so they switched to document search (which worked better).

    **The solution**: Fixed scheduling search by extracting date metadata, building a calendar index, and adding date filtering. Results:

    - Scheduling satisfaction: 25% → 78%
    - New user retention: +35%
    - Document search volume actually increased (users trusted the system more)

!!! warning "PM Pitfall"
    **User Adaptation Blindness**: Never assume low volume means low importance. Users vote with their behavior—if they stop asking certain questions, it might mean they have given up, not that they do not care.

!!! tip "For Engineers"
    **Detecting user adaptation patterns**:

    ```python
    from datetime import datetime, timedelta
    from typing import List, Dict
    from pydantic import BaseModel

    class UserJourney(BaseModel):
        user_id: str
        first_seen: datetime
        query_distribution_day1: Dict[str, float]
        query_distribution_day7: Dict[str, float]
        query_distribution_day30: Dict[str, float]

    async def analyze_user_adaptation(
        user_id: str,
        queries: List[dict]
    ) -> UserJourney:
        """
        Analyze how a user's query patterns change over time.

        Significant shifts in query distribution suggest
        adaptation to system limitations.
        """
        first_query = min(q["timestamp"] for q in queries)

        def get_distribution(start: datetime, end: datetime) -> Dict[str, float]:
            period_queries = [
                q for q in queries
                if start <= q["timestamp"] < end
            ]
            if not period_queries:
                return {}

            clusters = {}
            for q in period_queries:
                cluster = q.get("cluster", "unknown")
                clusters[cluster] = clusters.get(cluster, 0) + 1

            total = sum(clusters.values())
            return {k: v / total for k, v in clusters.items()}

        return UserJourney(
            user_id=user_id,
            first_seen=first_query,
            query_distribution_day1=get_distribution(
                first_query, first_query + timedelta(days=1)
            ),
            query_distribution_day7=get_distribution(
                first_query + timedelta(days=6), first_query + timedelta(days=8)
            ),
            query_distribution_day30=get_distribution(
                first_query + timedelta(days=29), first_query + timedelta(days=31)
            )
        )

    def detect_adaptation(journey: UserJourney, threshold: float = 0.3) -> List[str]:
        """
        Detect clusters where users have significantly reduced usage.

        A drop of more than 30% from day 1 to day 30 suggests
        users have adapted away from that query type.
        """
        adaptations = []

        for cluster, day1_pct in journey.query_distribution_day1.items():
            day30_pct = journey.query_distribution_day30.get(cluster, 0)
            if day1_pct - day30_pct > threshold:
                adaptations.append(cluster)

        return adaptations
    ```

---

### Building Your Classification Pipeline

Once you have identified your segments, build a production pipeline that classifies incoming queries in real-time.

!!! tip "For Product Managers"
    **Why real-time classification matters**:

    - Route queries to appropriate specialized systems
    - Track segment distributions over time
    - Detect drift when new patterns emerge
    - Alert when high-priority segments degrade

    **The "Other" category**: Always include an "other" category in your classification. When it grows above 10-15%, it is time to re-cluster. This is your early warning system for concept drift.

!!! tip "For Engineers"
    **Building a few-shot classifier**:

    ```python
    from typing import List, Tuple
    from pydantic import BaseModel

    class ClassificationResult(BaseModel):
        query: str
        predicted_cluster: str
        confidence: float
        capabilities: List[str]

    class FewShotClassifier:
        def __init__(self, examples_per_cluster: int = 5):
            self.examples: dict[str, List[str]] = {}
            self.examples_per_cluster = examples_per_cluster

        def add_examples(self, cluster_name: str, queries: List[str]):
            """Add representative examples for a cluster."""
            self.examples[cluster_name] = queries[:self.examples_per_cluster]

        async def classify(self, query: str) -> ClassificationResult:
            """
            Classify a query using few-shot prompting.
            """
            client = OpenAI()

            # Build few-shot prompt
            examples_text = ""
            for cluster, examples in self.examples.items():
                examples_text += f"\n{cluster}:\n"
                for ex in examples:
                    examples_text += f"  - {ex}\n"

            prompt = f"""Classify this query into one of the following categories.
            If it doesn't fit any category well, classify as "other".

            Categories and examples:
            {examples_text}

            Query to classify: {query}

            Respond with just the category name."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            predicted = response.choices[0].message.content.strip().lower()

            # Detect capabilities
            capabilities = detect_capabilities(query)

            return ClassificationResult(
                query=query,
                predicted_cluster=predicted,
                confidence=0.8,  # Could use logprobs for real confidence
                capabilities=list(capabilities.detected_capabilities)
            )
    ```

    **Monitoring dashboard essentials**:

    Track these metrics for each segment:

    - **Volume percentage**: What % of total queries
    - **Satisfaction score**: Average user satisfaction
    - **Retrieval quality**: Average cosine similarity
    - **Response time**: P50 and P95 latency
    - **Trend direction**: Improving or declining
    - **User retention**: Do users return after these queries
    - **Escalation rate**: How often users contact support

---

### The Expected Value Formula

Every improvement decision should be based on this formula:

**Expected Value = Impact x Volume % x (1 - Success Rate)**

Where:

- **Impact**: Business value on 1-10 scale (revenue, retention, strategic value)
- **Volume %**: Percentage of total queries in this segment
- **Success Rate**: Current satisfaction or success rate (lower = more opportunity)

!!! tip "For Product Managers"
    **Practical example: E-commerce search**

    | Segment | Impact | Volume % | Success % | Expected Value |
    |---------|--------|----------|-----------|----------------|
    | Product by SKU | $100/query | 30% | 95% | 1.5 |
    | "Affordable shoes" | $50/query | 45% | 40% | 13.5 |
    | "Gift ideas under $50" | $75/query | 15% | 20% | 9.0 |
    | Technical specs | $25/query | 10% | 85% | 0.4 |

    Even though "affordable shoes" has lower individual impact, its high volume and low success rate makes it the highest priority. This is how you make data-driven decisions.

    **The priority formula with effort and risk**:

    Priority Score = (Impact x Volume %) / (Effort x Risk)

    Where:

    - **Effort**: Implementation difficulty on 1-10 scale
    - **Risk**: Chance of breaking something on 1-5 scale

    Inventory issues typically have lower effort (3-4) since you are just adding data. Capability issues have higher effort (6-9) since you are building features.

!!! tip "For Engineers"
    **Implementing priority scoring**:

    ```python
    from pydantic import BaseModel
    from typing import List

    class ImprovementCandidate(BaseModel):
        segment_name: str
        issue_type: IssueType
        impact: float  # 1-10
        volume_pct: float  # 0-1
        current_success: float  # 0-1
        effort: float  # 1-10
        risk: float  # 1-5
        priority_score: float = 0.0

        def calculate_priority(self) -> float:
            """Calculate priority score."""
            opportunity = 1 - self.current_success
            self.priority_score = (
                self.impact * self.volume_pct * opportunity
            ) / (self.effort * self.risk)
            return self.priority_score

    def prioritize_improvements(
        candidates: List[ImprovementCandidate]
    ) -> List[ImprovementCandidate]:
        """
        Sort improvement candidates by priority score.
        """
        for candidate in candidates:
            candidate.calculate_priority()

        return sorted(candidates, key=lambda x: x.priority_score, reverse=True)
    ```

---

## Case Study Deep Dive

### Construction Project Management: 35% Retention Improvement

A construction project management company built a RAG system for contractors. The product team was convinced scheduling was a minor feature—only 8% of queries.

!!! tip "For Product Managers"
    **Initial analysis**:

    | Segment | Volume | Satisfaction |
    |---------|--------|--------------|
    | Document search | 52% | 70% |
    | Scheduling | 8% | 25% |
    | Cost lookup | 15% | 82% |
    | Compliance | 12% | 78% |
    | Other | 13% | 65% |

    **The hidden pattern**: New users asked 90% scheduling queries on day 1, but by day 30, only 20% were scheduling queries. Users were adapting to the system's failures.

    **The solution**: Fixed scheduling by extracting date metadata, building a calendar index, and adding date filtering capabilities.

    **Results**:

    - Scheduling satisfaction: 25% → 78%
    - New user retention: +35%
    - Document search volume increased (users trusted the system more)

!!! tip "For Engineers"
    **Technical implementation**:

    1. **Date metadata extraction**: Used LLM to extract dates from all documents
    2. **Calendar index**: Built specialized index for date-based queries
    3. **Date filtering**: Added explicit date range filtering to query processor
    4. **Router training**: Trained classifier to detect scheduling queries and route appropriately

    The key insight was that this was a capability issue, not an inventory issue. The scheduling information existed in documents—the system just could not process temporal queries correctly.

---

### Voice AI Restaurant System: $2M Revenue Opportunity

A voice AI company making calls for restaurants discovered a massive opportunity through query analysis.

!!! tip "For Product Managers"
    **The discovery**: Through data analysis, they found that when the AI attempted upselling, it generated 20% more revenue 50% of the time—a 10% overall increase. However, the agent only tried upselling in 9% of calls.

    **The solution**: Add a simple check ensuring the agent always asks if the customer wants anything else before ending the call.

    **Projected impact**: Increasing upselling attempts from 9% to 40% could generate an additional $2 million in revenue.

    **The insight**: The biggest business value came from analyzing usage patterns to identify a capability gap (upselling), not from improving core AI performance. A simple business rule delivered millions in value without touching the AI model.

!!! tip "For Engineers"
    **Implementation approach**:

    This was a capability issue solved with a simple rule:

    ```python
    async def handle_call_ending(call_state: CallState) -> str:
        """
        Before ending any call, check if upselling was attempted.
        """
        if not call_state.upsell_attempted:
            return "Before I let you go, would you like to add anything else to your order?"

        return call_state.closing_message
    ```

    No ML improvements needed—just a business rule based on data analysis.

---

### Customer Support RAG: 28% Ticket Reduction

A customer support system used prioritization to dramatically reduce support tickets.

!!! tip "For Product Managers"
    **Initial analysis**:

    | Segment | Volume | Satisfaction | Type |
    |---------|--------|--------------|------|
    | Password reset | 25% | 85% | Capability |
    | Billing questions | 20% | 45% | Inventory |
    | Feature requests | 15% | 30% | Capability |
    | Bug reports | 15% | 70% | Inventory |
    | How-to guides | 10% | 60% | Inventory |
    | Account deletion | 5% | 90% | Capability |
    | Integration help | 10% | 35% | Both |

    **Prioritization using the formula**:

    1. **Billing questions** (score: 85) - High volume + Low satisfaction + Low effort (inventory) = TOP PRIORITY
    2. **Integration help** (score: 72) - Medium volume + Very low satisfaction + Mixed issues = HIGH PRIORITY
    3. **Feature requests** (score: 58) - Medium volume + Very low satisfaction + High effort (capability) = MEDIUM PRIORITY

    **Results after 3 months**:

    - Billing questions: 45% → 82% satisfaction (+37%)
    - Integration help: 35% → 78% satisfaction (+43%)
    - Feature requests: 30% → 71% satisfaction (+41%)
    - Overall satisfaction: 58% → 76% (+18%)
    - Support ticket volume: -28%
    - Time to resolution: -45%

    ROI: The improvements paid for themselves in reduced support costs within 6 weeks.

!!! tip "For Engineers"
    **The roadmap that worked**:

    **Sprint 1 (Week 1-2): Quick Wins**

    - Add missing billing documentation (inventory)
    - Update integration guides with latest API changes (inventory)
    - Expected impact: +20% satisfaction for 30% of queries

    **Sprint 2 (Week 3-4): Capability Building**

    - Build feature request tracker/searcher (capability)
    - Add status filtering for bug reports (capability)
    - Expected impact: +30% satisfaction for 30% of queries

    **Quarter Goals (Month 2-3): Strategic Improvements**

    - Implement intelligent routing between documentation and support tickets
    - Build comparison tool for plan features
    - Add temporal filtering for "recent" queries

---

## Implementation Guide

### Quick Start for PMs

**Week 1: Initial Analysis**

1. Export query logs from your feedback system
2. Run basic clustering (K-means with 20 clusters)
3. Manually review 10 positive and 10 negative examples per cluster
4. Identify top 5 underperforming segments

**Week 2: Classification**

1. For each underperforming segment, classify as inventory or capability issue
2. Calculate Expected Value for each segment
3. Plot segments on the 2x2 matrix
4. Identify quick wins (high impact, low effort)

**Week 3: Roadmap**

1. Prioritize segments using the priority formula
2. Create 4-week improvement plan
3. Define success metrics for each improvement
4. Get stakeholder alignment on priorities

**Week 4: Execute and Measure**

1. Implement top priority improvements
2. Measure impact against baseline
3. Re-analyze query distribution
4. Adjust priorities based on results

### Detailed Implementation for Engineers

**Phase 1: Clustering Infrastructure (1 week)**

```python
# 1. Set up embedding pipeline
async def embed_queries(queries: List[str]) -> np.ndarray:
    client = OpenAI()
    embeddings = []

    # Batch for efficiency
    batch_size = 100
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings.extend([e.embedding for e in response.data])

    return np.array(embeddings)

# 2. Cluster and analyze
async def analyze_queries(
    queries: List[str],
    satisfaction_scores: List[float],
    n_clusters: int = 20
) -> dict:
    embeddings = await embed_queries(queries)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Build analysis
    analysis = {}
    for i in range(n_clusters):
        mask = labels == i
        cluster_queries = [q for q, m in zip(queries, mask) if m]
        cluster_scores = [s for s, m in zip(satisfaction_scores, mask) if m]

        analysis[i] = {
            "size": len(cluster_queries),
            "volume_pct": len(cluster_queries) / len(queries),
            "avg_satisfaction": np.mean(cluster_scores),
            "sample_queries": cluster_queries[:10],
        }

    return analysis
```

**Phase 2: Classification Pipeline (1 week)**

```python
# 1. Build few-shot classifier
classifier = FewShotClassifier(examples_per_cluster=5)

# Add examples from each cluster
for cluster_id, data in analysis.items():
    cluster_name = await get_cluster_name(cluster_id, data["sample_queries"])
    classifier.add_examples(cluster_name, data["sample_queries"])

# 2. Set up real-time classification
@app.post("/classify")
async def classify_query(request: QueryRequest) -> ClassificationResult:
    result = await classifier.classify(request.query)

    # Log for monitoring
    await log_classification(result)

    return result
```

**Phase 3: Monitoring Dashboard (1 week)**

```python
# 1. Track metrics per segment
async def update_segment_metrics(
    segment: str,
    query_id: str,
    satisfaction: Optional[float] = None,
    latency_ms: Optional[float] = None
):
    await db.segment_metrics.insert({
        "segment": segment,
        "query_id": query_id,
        "satisfaction": satisfaction,
        "latency_ms": latency_ms,
        "timestamp": datetime.now()
    })

# 2. Generate daily reports
async def generate_segment_report() -> dict:
    segments = await db.segment_metrics.aggregate([
        {"$match": {"timestamp": {"$gte": yesterday}}},
        {"$group": {
            "_id": "$segment",
            "count": {"$sum": 1},
            "avg_satisfaction": {"$avg": "$satisfaction"},
            "p50_latency": {"$percentile": ["$latency_ms", 0.5]},
            "p95_latency": {"$percentile": ["$latency_ms", 0.95]},
        }}
    ])

    return {s["_id"]: s for s in segments}
```

---

## Common Pitfalls

### PM Pitfalls

!!! warning "PM Pitfall: Analysis Paralysis"
    **Problem**: Spending months analyzing without implementing anything.

    **Solution**: Set hard deadlines. After 2 weeks of analysis, ship something. Perfect analysis paralysis kills more projects than imperfect action.

    **Timeline**:

    - Week 1-2: Analysis phase
    - Week 3-4: Implementation of top 3 segments
    - Week 5: Measure and iterate

!!! warning "PM Pitfall: Ignoring User Adaptation"
    **Problem**: Assuming low volume means low importance.

    **Solution**: Track behavior changes monthly. Compare query distributions between months. Look for drift > 20% in any segment. Users are smart—they will work around your limitations.

!!! warning "PM Pitfall: Over-Engineering Solutions"
    **Problem**: Building complex systems for simple problems.

    **Solution**: Start with the simplest solution that could work:

    1. Can better prompts fix this?
    2. Can metadata filtering help?
    3. Do we need a specialized index?
    4. Do we need a custom model?
    5. Do we need a complete rebuild?

    Always start at level 1. Most problems are solved by level 2-3.

### Engineering Pitfalls

!!! warning "Engineering Pitfall: Over-Segmentation"
    **Problem**: Having 100 micro-segments that are not actionable.

    **Solution**: Start with 10-20 clusters and refine from there. The goal is actionable insights, not perfect categorization.

!!! warning "Engineering Pitfall: Ignoring Cross-Segment Patterns"
    **Problem**: Missing that the same capability issue (like date filtering) affects multiple topic segments.

    **Solution**: After topic clustering, also cluster by capability. Look for patterns that span multiple topics.

!!! warning "Engineering Pitfall: Static Segmentation"
    **Problem**: Running clustering once and never updating.

    **Solution**: Re-run clustering monthly. Track drift in your "other" category—when it grows above 10%, it is time to re-cluster. User behavior evolves, and your segmentation should too.

!!! warning "Engineering Pitfall: Not Measuring Impact"
    **Problem**: Implementing improvements without tracking results.

    **Solution**: Define success metrics before implementation:

    - **Primary metric**: User satisfaction
    - **Secondary metrics**: Query success rate, time to answer
    - **Business metric**: Support ticket reduction
    - **Success threshold**: 15% improvement minimum

    If you cannot measure it, you cannot improve it.

---

## Related Content

### Transcript

- **Workshop Transcript**: `docs/workshops/chapter4-transcript.txt` - Full lecture content with examples and Q&A

### Talks

- **Query Routing (Anton, ChromaDB)**: `docs/talks/query-routing-anton.md`
    - Key insight: The "big pile of records" approach reduces recall due to filtering overhead. Consider one index per user per data source for better performance.
    - Relevant for: Understanding why segmentation improves retrieval quality

- **Domain Experts (Chris Lovejoy, Anterior)**: `docs/talks/chris-lovejoy-domain-expert-vertical-ai.md`
    - Key insight: Domain experts are essential for defining what "good" looks like in specialized industries. Build systematic review processes to capture their insights.
    - Relevant for: Understanding how to validate segment classifications

### Office Hours

- **Cohort 2 Week 4**: `docs/office-hours/cohort2/week4-summary.md`
    - Key insight: Segmentation helps figure out what new tools to build. The goal is external understanding of your data.
    - Topics: Customer segmentation, query pattern analysis, conversation-level analysis

- **Cohort 3 Week 4-1**: `docs/office-hours/cohort3/week-4-1.md`
    - Key insight: Model selection should be driven by business value, not technical specifications alone.
    - Topics: Evaluation data collection, model selection, visual elements in reports

- **Cohort 3 Week 4-2**: `docs/office-hours/cohort3/week-4-2.md`
    - Key insight: For unstructured feedback analysis, combine semantic search with hierarchical clustering for accurate counts.
    - Topics: Dynamic data visualization, customer feedback analysis, tool-based vs semantic search

---

## Action Items

### For Product Teams

**This Week**:

1. Export last 30 days of query logs
2. Run initial clustering analysis
3. Identify top 3 underperforming segments
4. Calculate Expected Value for each segment

**This Month**:

1. Build 2x2 prioritization matrix
2. Classify issues as inventory vs capability
3. Create prioritized roadmap
4. Define success metrics for top improvements

**This Quarter**:

1. Implement top priority improvements
2. Set up segment monitoring dashboard
3. Establish monthly re-clustering cadence
4. Build stakeholder reporting on segment performance

### For Engineering Teams

**This Week**:

1. Set up embedding pipeline for queries
2. Implement K-means clustering
3. Build basic classification endpoint
4. Create segment metrics logging

**This Month**:

1. Build few-shot classifier for segments
2. Implement capability detection
3. Create monitoring dashboard
4. Set up alerting for segment degradation

**This Quarter**:

1. Build specialized retrievers for high-priority segments
2. Implement routing based on classification
3. Create A/B testing framework for improvements
4. Automate monthly re-clustering

---

## Reflection Questions

1. **Strategic**: What percentage of your queries fall into your "danger zone" (high volume, low satisfaction)? What would be the business impact of improving that segment by 20%?

2. **Technical**: How would you detect if users are adapting to your system's limitations? What signals would indicate this pattern?

3. **Prioritization**: Using the Expected Value formula, which of your current improvement ideas has the highest priority score? Does this match your current roadmap?

4. **Classification**: For your top underperforming segment, is it an inventory issue or a capability issue? How would you validate your classification?

5. **Measurement**: What metrics would you track to know if your segmentation-based improvements are working? How would you distinguish signal from noise?

---

## Summary

### For Product Managers

- **Segmentation reveals hidden patterns**: Aggregate metrics hide important details. A 70% satisfaction rate might mask a 25% satisfaction rate in a critical segment.
- **Use the 2x2 matrix**: Volume vs satisfaction tells you what to prioritize. High volume + low satisfaction = danger zone.
- **Distinguish inventory from capability**: Missing data needs different solutions than missing features. Ask "does the information exist?" first.
- **Watch for user adaptation**: Users vote with their behavior. Low volume might mean they gave up, not that they do not care.
- **Use the Expected Value formula**: Impact x Volume % x (1 - Success Rate) / (Effort x Risk) makes prioritization objective.

### For Engineers

- **Start simple with K-means**: Do not overthink clustering. 20 clusters with manual review beats fancy algorithms.
- **Build classification pipelines**: Real-time query classification enables routing, monitoring, and drift detection.
- **Track the "other" category**: When it grows above 10%, re-cluster. This is your early warning system.
- **Detect capabilities, not just topics**: The same topic can need different capabilities. Build detection for both dimensions.
- **Measure everything**: Define success metrics before implementation. If you cannot measure it, you cannot improve it.

---

## Further Reading

### Academic Papers

- "Clustering by Compression" - Information-theoretic approach to clustering
- "Topic Models" - Latent Dirichlet Allocation and variations
- "Learning to Rank" - Foundations of prioritization algorithms

### Tools

- **BERTopic**: Topic modeling with transformers (https://maartengr.github.io/BERTopic/)
- **Kura**: Open-source conversation analysis (similar to Anthropic's Clio)
- **scikit-learn**: K-means and other clustering algorithms

### Industry Resources

- Anthropic's Clio paper on privacy-preserving conversation analysis
- ChromaDB documentation on query routing and data organization

---

## Navigation

**Previous**: [Chapter 3: Feedback Systems and UX](chapter3.md) - Building feedback collection infrastructure

**Next**: [Chapter 5: Specialized Retrieval Systems](chapter5.md) - Building targeted retrievers for high-priority segments

**Related**:

- [Chapter 6: Query Routing and Orchestration](chapter6.md) - Routing queries to appropriate specialized systems
- [Chapter 2: Training Data and Fine-Tuning](chapter2.md) - Creating training data for underperforming segments
