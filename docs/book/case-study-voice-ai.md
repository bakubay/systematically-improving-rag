---
title: "Case Study: Voice AI Restaurant System"
description: "Case study demonstrating how data analysis revealed a $2M revenue opportunity through a simple business rule change"
authors:
  - Jason Liu
date: 2024-01-15
tags:
  - case-study
  - voice-ai
  - restaurant
  - prioritization
  - business-rules
---

# Case Study: Voice AI Restaurant System

## Overview

This case study follows a voice AI company that makes automated calls for restaurants. Through systematic data analysis, they discovered a massive revenue opportunity that required no AI improvements—just a simple business rule change.

**Key Results**:

| Metric | Before | After | Impact |
|--------|--------|-------|--------|
| Upselling Attempts | 9% of calls | 40% of calls | +31 points |
| Revenue per Upsell | +20% | +20% | Unchanged |
| Success Rate | 50% | 50% | Unchanged |
| Projected Annual Revenue | Baseline | +$2M | Significant |

**The Core Insight**: The biggest business value came from analyzing usage patterns to identify a capability gap, not from improving core AI performance. A simple business rule delivered millions in value without touching the AI model.

---

## Chapter Connections

This case study demonstrates concepts from Chapter 4:

| Chapter | Concept Applied | Result |
|---------|-----------------|--------|
| Chapter 4 | Query segmentation | Identified upselling pattern |
| Chapter 4 | Capability vs inventory | Diagnosed as capability issue |
| Chapter 4 | Expected value formula | Quantified opportunity |
| Chapter 4 | Prioritization framework | Justified investment |

---

## The Business Problem

!!! tip "For Product Managers"
    **The scenario**: A voice AI company provides automated phone ordering for restaurants. The AI handles incoming calls, takes orders, and processes payments.

    **Initial focus**: The engineering team was working on improving speech recognition accuracy, reducing latency, and handling edge cases like complex menu modifications.

    **The question**: Where should the team focus to maximize business impact?

!!! tip "For Engineers"
    **The system architecture**:

    ```text
    Customer Call → Speech Recognition → Intent Classification →
    Order Processing → Payment → Confirmation
    ```

    **Technical metrics being tracked**:

    - Speech recognition accuracy: 94%
    - Intent classification accuracy: 89%
    - Order completion rate: 82%
    - Average call duration: 3.2 minutes

    The team was focused on improving these technical metrics, assuming better AI would drive better business outcomes.

---

## The Discovery

The breakthrough came from analyzing call transcripts, not from improving AI models.

!!! tip "For Product Managers"
    **The data analysis**:

    The team segmented calls by behavior patterns and discovered:

    | Behavior | Frequency | Revenue Impact |
    |----------|-----------|----------------|
    | Basic order only | 91% | Baseline |
    | Upselling attempted | 9% | +20% revenue (50% success) |

    **The math**:

    - When the AI attempted upselling, it generated 20% more revenue 50% of the time
    - This equals a 10% overall revenue increase per upselling attempt
    - But the agent only tried upselling in 9% of calls

    **The opportunity**:

    - Current: 9% of calls × 10% revenue increase = 0.9% total revenue lift
    - Potential: 40% of calls × 10% revenue increase = 4% total revenue lift
    - For a company processing $50M in orders: $2M additional revenue

!!! tip "For Engineers"
    **How the analysis was done**:

    ```python
    from dataclasses import dataclass

    @dataclass
    class CallAnalysis:
        call_id: str
        upsell_attempted: bool
        upsell_successful: bool
        order_total: float
        call_duration: float

    async def analyze_calls(calls: list[CallAnalysis]) -> dict:
        """Analyze call patterns to identify opportunities."""
        upsell_calls = [c for c in calls if c.upsell_attempted]
        no_upsell_calls = [c for c in calls if not c.upsell_attempted]

        upsell_rate = len(upsell_calls) / len(calls)
        upsell_success_rate = (
            sum(1 for c in upsell_calls if c.upsell_successful)
            / len(upsell_calls)
        )

        avg_order_with_upsell = sum(c.order_total for c in upsell_calls) / len(upsell_calls)
        avg_order_without = sum(c.order_total for c in no_upsell_calls) / len(no_upsell_calls)

        revenue_lift = (avg_order_with_upsell - avg_order_without) / avg_order_without

        return {
            "upsell_rate": upsell_rate,
            "upsell_success_rate": upsell_success_rate,
            "revenue_lift_per_upsell": revenue_lift,
            "total_revenue_lift": upsell_rate * upsell_success_rate * revenue_lift
        }
    ```

    **Results**:

    ```python
    {
        "upsell_rate": 0.09,           # Only 9% of calls
        "upsell_success_rate": 0.50,    # 50% success when attempted
        "revenue_lift_per_upsell": 0.20, # 20% more revenue
        "total_revenue_lift": 0.009     # 0.9% total lift currently
    }
    ```

---

## Diagnosis: Capability vs Inventory

Using the framework from Chapter 4, the team diagnosed this as a capability issue.

!!! tip "For Product Managers"
    **The diagnosis framework**:

    | Issue Type | Definition | This Case |
    |------------|------------|-----------|
    | Inventory | Missing data or content | No—the AI knew about upsell items |
    | Capability | Missing feature or behavior | Yes—the AI did not consistently attempt upselling |

    **Why it was a capability issue**:

    - The AI had access to upsell suggestions (drinks, sides, desserts)
    - The AI knew how to offer upsells when it did attempt them
    - The AI simply was not programmed to consistently attempt upselling

    **The solution**: Add a business rule, not improve AI capabilities.

!!! tip "For Engineers"
    **Root cause analysis**:

    The AI's conversation flow did not include a mandatory upselling step. It would sometimes offer upsells based on context, but there was no systematic check.

    **Before**:

    ```python
    async def handle_order_completion(order: Order) -> str:
        """Complete the order and provide confirmation."""
        # Sometimes offers upsell based on context
        if should_offer_upsell(order):  # Inconsistent logic
            upsell_response = await offer_upsell(order)
            if upsell_response:
                order.add_item(upsell_response)

        return f"Your total is ${order.total}. Is that correct?"
    ```

    **The problem**: `should_offer_upsell()` was based on complex heuristics that only triggered 9% of the time.

---

## The Solution

The fix was remarkably simple: ensure the AI always asks if the customer wants anything else before ending the call.

!!! tip "For Product Managers"
    **The business rule**:

    > Before completing any order, the AI must ask: "Would you like to add anything else to your order?"

    **Implementation timeline**:

    | Phase | Duration | Activity |
    |-------|----------|----------|
    | Analysis | 1 week | Identified opportunity |
    | Implementation | 2 days | Added business rule |
    | Testing | 3 days | A/B test validation |
    | Rollout | 1 week | Gradual deployment |

    **Total time to value**: 2.5 weeks

    **Comparison to AI improvements**:

    | Approach | Time | Expected Impact |
    |----------|------|-----------------|
    | Improve speech recognition 94% → 96% | 3 months | +1% order completion |
    | Add upselling rule | 2.5 weeks | +$2M revenue |

!!! tip "For Engineers"
    **The implementation**:

    ```python
    async def handle_call_ending(call_state: CallState) -> str:
        """
        Before ending any call, check if upselling was attempted.
        This is a mandatory step in the conversation flow.
        """
        if not call_state.upsell_attempted:
            call_state.upsell_attempted = True
            return "Before I let you go, would you like to add anything else to your order?"

        return call_state.closing_message
    ```

    **Key design decisions**:

    1. **Mandatory, not optional**: The rule always triggers, not based on heuristics
    2. **Natural phrasing**: "Before I let you go" sounds conversational
    3. **Single attempt**: Only ask once to avoid annoying customers
    4. **State tracking**: Record that upselling was attempted for analytics

    **A/B test setup**:

    ```python
    async def route_call(call_id: str) -> str:
        """Route calls to control or treatment group."""
        if hash(call_id) % 100 < 50:
            return "control"  # Original behavior
        return "treatment"    # New upselling rule
    ```

---

## Results

!!! tip "For Product Managers"
    **A/B test results** (2 weeks, 10,000 calls per group):

    | Metric | Control | Treatment | Change |
    |--------|---------|-----------|--------|
    | Upsell attempts | 9% | 42% | +33 points |
    | Upsell success rate | 50% | 48% | -2 points |
    | Revenue per call | $24.50 | $25.80 | +5.3% |
    | Call duration | 3.2 min | 3.4 min | +6% |
    | Customer satisfaction | 4.2/5 | 4.1/5 | -2% |

    **Analysis**:

    - Upsell success rate dropped slightly (50% → 48%) because more marginal opportunities were attempted
    - Revenue per call increased 5.3%, validating the opportunity
    - Call duration increased slightly but within acceptable range
    - Customer satisfaction dropped marginally but remained high

    **Projected annual impact**: $2.1M additional revenue

!!! tip "For Engineers"
    **Statistical validation**:

    ```python
    from scipy import stats

    def validate_results(control: list[float], treatment: list[float]) -> dict:
        """Validate A/B test results with statistical significance."""
        t_stat, p_value = stats.ttest_ind(control, treatment)

        control_mean = sum(control) / len(control)
        treatment_mean = sum(treatment) / len(treatment)
        lift = (treatment_mean - control_mean) / control_mean

        return {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "lift": lift,
            "p_value": p_value,
            "significant": p_value < 0.05
        }

    # Results
    # {
    #     "control_mean": 24.50,
    #     "treatment_mean": 25.80,
    #     "lift": 0.053,
    #     "p_value": 0.0003,
    #     "significant": True
    # }
    ```

    **Monitoring after rollout**:

    ```python
    async def monitor_upselling_metrics() -> dict:
        """Track upselling metrics in production."""
        calls_today = await get_calls_since(days=1)

        return {
            "upsell_attempt_rate": calculate_upsell_rate(calls_today),
            "upsell_success_rate": calculate_success_rate(calls_today),
            "revenue_per_call": calculate_avg_revenue(calls_today),
            "call_duration_avg": calculate_avg_duration(calls_today)
        }
    ```

---

## Key Lessons Learned

!!! tip "For Product Managers"
    **Strategic insights**:

    1. **Data analysis before AI improvement**: The biggest opportunity was found through analyzing usage patterns, not improving AI models.

    2. **Capability vs inventory matters**: Understanding that this was a capability issue (missing behavior) rather than an inventory issue (missing data) led directly to the solution.

    3. **Simple rules can be powerful**: A single business rule delivered $2M in value. No machine learning required.

    4. **Measure behavior, not just performance**: Technical metrics (speech recognition accuracy) were good, but behavior metrics (upselling rate) revealed the opportunity.

    5. **Time to value matters**: 2.5 weeks to $2M is better than 3 months to marginal improvement.

!!! tip "For Engineers"
    **Technical insights**:

    1. **Instrument everything**: The opportunity was only visible because call behavior was being tracked. Without data on upselling attempts, this would have been invisible.

    2. **Business rules complement AI**: The AI handled the complex parts (speech recognition, intent classification). The business rule handled the simple part (always ask about upsells).

    3. **A/B test before rollout**: The 2-week A/B test validated the opportunity and caught the slight decrease in customer satisfaction before full rollout.

    4. **Monitor after deployment**: Continued monitoring ensured the improvement persisted and caught any degradation.

---

## Applying This Pattern

This case study demonstrates a general pattern for finding high-value improvements:

!!! tip "For Product Managers"
    **The pattern**:

    1. **Segment by behavior**: Group interactions by what happened, not just outcomes
    2. **Calculate expected value**: For each behavior, calculate frequency × impact
    3. **Identify gaps**: Look for high-value behaviors that happen infrequently
    4. **Diagnose root cause**: Is it inventory (missing data) or capability (missing feature)?
    5. **Implement and measure**: Start with the simplest solution that could work

    **Questions to ask**:

    - What behaviors correlate with high-value outcomes?
    - How often do those behaviors occur?
    - Why don't they occur more often?
    - What's the simplest intervention that could increase frequency?

!!! tip "For Engineers"
    **Implementation checklist**:

    ```python
    # 1. Instrument behavior tracking
    async def track_call_behavior(call: Call) -> None:
        await db.insert("call_behaviors", {
            "call_id": call.id,
            "upsell_attempted": call.upsell_attempted,
            "upsell_successful": call.upsell_successful,
            "order_total": call.order_total,
            # Add other behaviors to track
        })

    # 2. Build analysis queries
    async def analyze_behavior_impact() -> dict:
        return await db.query("""
            SELECT
                behavior_name,
                COUNT(*) as frequency,
                AVG(order_total) as avg_value,
                AVG(CASE WHEN successful THEN 1 ELSE 0 END) as success_rate
            FROM call_behaviors
            GROUP BY behavior_name
        """)

    # 3. Implement intervention
    async def ensure_behavior(call_state: CallState, behavior: str) -> None:
        if not getattr(call_state, f"{behavior}_attempted", False):
            setattr(call_state, f"{behavior}_attempted", True)
            # Trigger the behavior

    # 4. A/B test the intervention
    # 5. Monitor after rollout
    ```

---

## Related Content

- [Chapter 4: Query Understanding and Prioritization](chapter4.md) - Segmentation and prioritization framework
- [Chapter 3: Feedback Systems and UX](chapter3.md) - Data collection patterns
- [Appendix D: Debugging RAG Systems](appendix-debugging.md) - Systematic analysis methodology

---

## Navigation

[Previous: WildChat Case Study](case-study-wildchat.md) | [Back to Case Studies Index](index.md)
