# Plan: Transform "Systematically Improving RAG Applications" into a Technical Reference Series

## Executive Summary

Transform the existing workshop materials into a comprehensive technical reference book. Each chapter is created fresh in `docs/book/` directory, synthesizing content from workshops, transcripts, talks, and office hours. Each chapter goes through three phases: Content Creation → Review → Editorial.

**Key Principles**:

- All new content created in `docs/book/` directory
- Synthesize from multiple sources (workshops, transcripts, talks, office hours)
- Each chapter follows standard template with PM and Engineer sections
- Use admonitions to mark audience-specific content
- Introduce foundational concepts before they're used

**Workflow Guidelines**:

- Commit often and in phases - make frequent commits as you work through each section or part
- Use graphite stacks for large sections of work - create a new graphite stack branch for each major stage/section/part
- Keep iterating on the same graphite stack branch for a given stage/part until that work is complete
- Commit changelog files along with your work - document decisions, blockers, and observations in `docs/book/changelog/{phase_name}_notes.md`
- Always commit the updated `docs/book/plan.md` when you check off a section or part

---

## Standard Chapter Template

Every chapter must include:

1. **Front Matter** (title, description, authors, date, tags)
2. **Chapter at a Glance** (prerequisites, outcomes, case study reference)
3. **Key Insight** (one-paragraph summary)
4. **Learning Objectives** (5-6 measurable outcomes)
5. **Introduction** (context, building on previous chapters)
6. **Core Content** (3-5 major sections)
   - **For Product Managers**: Business value, decision frameworks, ROI, success metrics
   - **For Engineers**: Implementation details, code examples, algorithms, tradeoffs
7. **Case Study Deep Dive** (dedicated section)
8. **Implementation Guide**
   - **Quick Start for PMs**: High-level steps, decision points
   - **Detailed Implementation for Engineers**: Full code, configuration
9. **Common Pitfalls**
   - **PM Pitfalls**: Strategic mistakes, resource misallocation
   - **Engineering Pitfalls**: Technical mistakes, implementation errors
10. **Related Content** (transcripts, talks, office hours with key insights)
11. **Action Items**
    - **For Product Teams**: Strategic planning, stakeholder alignment
    - **For Engineering Teams**: Technical implementation, testing
12. **Reflection Questions** (5 questions, mix of strategic and technical)
13. **Summary** (key takeaways, separate for PM vs Engineer)
14. **Further Reading** (academic papers, tools)
15. **Navigation** (previous, next, reference links)

**Admonition Usage**:

- `!!! tip "For Product Managers"` - PM-specific tips
- `!!! tip "For Engineers"` - Engineer-specific tips
- `!!! warning "PM Pitfall"` - Strategic mistakes
- `!!! warning "Engineering Pitfall"` - Technical mistakes
- `!!! info` - General information
- `!!! example` - Concrete examples
- `!!! success` - Success stories

---

## Chapter Organization

### Book 1: Foundations

**Chapter 0: Introduction - The Product Mindset for RAG**

**Chapter 1: Evaluation-First Development**

**Chapter 2: Training Data and Fine-Tuning**

### Book 2: User-Centric Design

**Chapter 3: Feedback Systems and UX**

**Chapter 4: Query Understanding and Prioritization**

### Book 3: Architecture and Production

**Chapter 5: Specialized Retrieval Systems**

**Chapter 6: Query Routing and Orchestration**

**Chapter 7: Production Operations**

### Book 4: Advanced Topics

**Chapter 8: Hybrid Search**

**Chapter 9: Context Window Management**

### Appendices

**Appendix A: Mathematical Foundations**

**Appendix B: Algorithms Reference**

**Appendix C: Benchmarking Your RAG System**

**Appendix D: Debugging RAG Systems**

**Appendix E: Graph vs SQL Decision Guide**

**Appendix F: Multi-Tenancy Patterns**

### Supporting Materials

**How to Use This Book**

**Glossary**

**Quick Reference**

**Case Studies**

---

## Chapter 0: Introduction - The Product Mindset for RAG

### Phase 1: Content Creation

**Goal**: Create comprehensive introduction chapter that establishes foundational concepts and sets up the book structure.

**Source Materials to Synthesize**:

- `docs/workshops/chapter0.md` - Workshop content
- `docs/workshops/chapter0-transcript.txt` - Actual lecture words
- `docs/talks/rag-antipatterns-skylar-payne.md` - RAG Antipatterns talk
- Office hours: Various sessions on product mindset

**Tasks**:

- [x] Create `docs/book/chapter0.md`
  - [x] Write front matter (title, description, authors, date, tags)
  - [x] Write "Chapter at a Glance" section
  - [x] Write "Key Insight" paragraph
  - [x] Write "Learning Objectives" (5-6 outcomes)
  - [x] Write "Introduction" section
    - [x] Build context for the book
    - [x] Explain the product mindset shift
    - [x] Introduce the improvement flywheel
  - [x] Write "Core Content" sections:
    - [x] **Foundational Concepts** (introduce before used elsewhere):
      - [x] Embeddings and vector representations
        - [x] PM section: Business value, when embeddings matter
        - [x] Engineer section: How embeddings work, vector space intuition
        - [x] Visual diagram: Vector space showing similar texts
      - [x] Vector databases
        - [x] PM section: Why vector DBs matter, cost considerations
        - [x] Engineer section: ANN search, tradeoffs, examples
      - [x] Semantic vs lexical search
        - [x] PM section: When to use each, business implications
        - [x] Engineer section: BM25 vs embeddings, hybrid approaches
      - [x] Chunking strategies
        - [x] PM section: Why chunking matters, size considerations
        - [x] Engineer section: Strategies (fixed, sentence, semantic, page-level)
      - [x] Cosine similarity
        - [x] Engineer section: What it measures, why it's used
      - [x] The alignment problem
        - [x] PM section: Why alignment matters, business impact
        - [x] Engineer section: Examples, how to detect misalignment
      - [x] Inventory vs capability problem
        - [x] PM section: Strategic distinction, different solutions
        - [x] Engineer section: How to diagnose each
    - [x] **The Product Mindset**:
      - [x] PM section: Why product thinking matters, ROI of systematic improvement
      - [x] Engineer section: How to apply product thinking to technical work
    - [x] **The Improvement Flywheel**:
      - [x] PM section: Business value of flywheel, examples
      - [x] Engineer section: Technical implementation of flywheel
    - [x] **Common Failure Patterns**:
      - [x] PM section: Strategic mistakes, how to avoid
      - [x] Engineer section: Technical mistakes, how to avoid
  - [x] Write "How to Use This Book" section:
    - [x] Reading paths (PM, Engineer, Full)
    - [x] Admonition types explanation with examples
    - [x] PM vs Engineer content markers explanation
    - [x] Navigation guide
  - [x] Write "Case Study Deep Dive" section:
    - [x] Legal tech company case study (from workshop)
    - [x] PM perspective: Business outcomes
    - [x] Engineer perspective: Technical implementation
  - [x] Write "Implementation Guide" section:
    - [x] Quick Start for PMs: How to apply product mindset
    - [x] Detailed Implementation for Engineers: Setting up evaluation infrastructure
  - [x] Write "Common Pitfalls" section:
    - [x] PM Pitfalls: Strategic mistakes
    - [x] Engineering Pitfalls: Technical mistakes
  - [x] Write "Related Content" section:
    - [x] Link to transcript: `docs/workshops/chapter0-transcript.txt`
    - [x] Link to talk: RAG Antipatterns (Skylar Payne)
    - [x] Extract key insights from each source
  - [x] Write "Action Items" section:
    - [x] For Product Teams: Strategic planning
    - [x] For Engineering Teams: Technical setup
  - [x] Write "Reflection Questions" (5 questions)
  - [x] Write "Summary" section (separate bullets for PM vs Engineer)
  - [x] Write "Further Reading" section
  - [x] Write "Navigation" section
  - [x] Use admonitions throughout to mark PM vs Engineer content
  - [x] Ensure all 15 template sections are present

**Acceptance Criteria**:

- All foundational concepts introduced
- PM and Engineer sections clearly separated
- Admonitions used appropriately
- All 15 template sections present
- Content synthesizes from multiple sources

### Phase 2: Review

**Goal**: Technical accuracy and peer review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify foundational concepts are correct
  - [x] Check that concepts are properly introduced
  - [x] Validate that concepts will be used correctly in later chapters
  - [x] Review cross-references
- [ ] Peer review
  - [ ] Distribute to 2+ reviewers (mix of PMs and engineers)
  - [ ] Collect feedback on clarity, completeness, accuracy
  - [ ] Incorporate feedback
- [x] Code example review (if any)
  - [x] Verify code examples work
  - [x] Check code formatting

**Acceptance Criteria**:

- All technical content verified accurate
- Peer feedback incorporated
- Code examples (if any) work correctly

### Phase 3: Editorial

**Goal**: Copy editing and professional polish

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent
  - [x] Image formatting consistent

**Acceptance Criteria**:

- No spelling or grammar errors
- Consistent style throughout
- All links functional
- Consistent formatting

---

## Chapter 1: Evaluation-First Development

### Phase 1: Content Creation

**Goal**: Create comprehensive evaluation chapter with technical depth and mathematical rigor.

**Source Materials to Synthesize**:

- `docs/workshops/chapter1.md` - Workshop content
- `docs/workshops/chapter1-transcript.txt` - Actual lecture words
- `docs/talks/embedding-performance-generative-evals-kelly-hong.md` - Generative Evals talk
- `docs/talks/zapier-vitor-evals.md` - Zapier Feedback talk
- `docs/office-hours/cohort2/week1-summary.md` - Office hours session
- `docs/office-hours/cohort3/week-1-1.md` - Office hours session

**Tasks**:

- [x] Create `docs/book/chapter1.md`
  - [x] Write front matter
  - [x] Write "Chapter at a Glance" section
  - [x] Write "Key Insight" paragraph
  - [x] Write "Learning Objectives"
  - [x] Write "Introduction" section
    - [x] Build on Chapter 0
    - [x] Explain why evaluation comes first
  - [x] Write "Core Content" sections:
    - [x] **Leading vs Lagging Metrics** (introduce early):
      - [x] PM section: Why leading metrics matter, examples
      - [x] Engineer section: How to measure leading metrics
    - [x] **Precision vs Recall** (intuitive explanation before formulas):
      - [x] PM section: Business implications of tradeoff
      - [x] Engineer section: Intuitive explanation before math
    - [x] **Evaluation Frameworks**:
      - [x] PM section: ROI of evaluation, when to invest
      - [x] Engineer section: Implementation details, code examples
    - [x] **Synthetic Data Generation**:
      - [x] PM section: Business value, when to use
      - [x] Engineer section: Implementation, code examples
    - [x] **Retrieval Metrics** (mathematical deep dive):
      - [x] Engineer section: Precision/recall/F1 derivations
      - [x] Engineer section: Statistical significance testing
      - [x] Engineer section: Confidence intervals
      - [x] Engineer section: Sample size calculations
    - [x] **Evaluation Infrastructure**:
      - [x] PM section: Resource requirements, ROI
      - [x] Engineer section: Code examples, architecture
  - [x] Write "Case Study Deep Dive" section:
    - [x] Consulting firm case study
    - [x] Blueprint search case study
    - [x] PM and Engineer perspectives
  - [x] Write "Implementation Guide" section:
    - [x] Quick Start for PMs: Setting up evaluation
    - [x] Detailed Implementation for Engineers: Full pipeline code
  - [x] Write "Common Pitfalls" section:
    - [x] PM Pitfalls: Strategic mistakes
    - [x] Engineering Pitfalls: Technical mistakes
  - [x] Write "Related Content" section:
    - [x] Transcript: `docs/workshops/chapter1-transcript.txt`
    - [x] Talk: Generative Evals (Kelly Hong)
    - [x] Talk: Zapier Feedback (Vitor)
    - [x] Office Hours: C2 Week 1, C3 Week 1.1
    - [x] Extract key insights from each
  - [x] Write "Action Items" section
  - [x] Write "Reflection Questions"
  - [x] Write "Summary" section
  - [x] Write "Further Reading" section
  - [x] Write "Navigation" section
  - [x] Use admonitions appropriately
  - [x] Ensure all 15 template sections present

**Acceptance Criteria**:

- Leading vs lagging metrics introduced early
- Precision vs recall explained intuitively before formulas
- Mathematical deep dives in Engineer sections
- All source materials synthesized
- All 15 template sections present

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify mathematical formulas are correct
  - [x] Test code examples
  - [x] Validate evaluation methodology
  - [x] Check statistical methods
- [ ] Peer review
  - [ ] Distribute to reviewers
  - [ ] Collect feedback
  - [ ] Incorporate feedback

**Acceptance Criteria**:

- All formulas verified correct
- All code examples work
- Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent
  - [x] Image formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---

## Chapter 2: Training Data and Fine-Tuning

### Phase 1: Content Creation

**Goal**: Create comprehensive fine-tuning chapter with technical depth.

**Source Materials to Synthesize**:

- `docs/workshops/chapter2.md` - Workshop content
- `docs/workshops/chapter2-transcript.txt` - Actual lecture words
- `docs/talks/glean-manav.md` - Glean talk
- `docs/talks/fine-tuning-rerankers-embeddings-ayush-lancedb.md` - LanceDB Re-rankers talk
- `docs/office-hours/cohort2/week2-summary.md` - Office hours session
- `docs/office-hours/cohort3/week-2-1.md` - Office hours session

**Tasks**:

- [x] Create `docs/book/chapter2.md`
  - [x] Write front matter
  - [x] Write "Chapter at a Glance" section
  - [x] Write "Key Insight" paragraph
  - [x] Write "Learning Objectives"
  - [x] Write "Introduction" section
    - [x] Build on Chapter 1
    - [x] Explain why fine-tuning matters
  - [x] Write "Core Content" sections:
    - [x] **Bi-Encoder vs Cross-Encoder** (introduce early):
      - [x] PM section: Cost/speed tradeoffs
      - [x] Engineer section: Architecture differences, when to use each
    - [x] **Contrastive Learning** (before loss functions):
      - [x] PM section: Why it works, business value
      - [x] Engineer section: Positive/negative pairs, intuition
    - [x] **Re-Ranking** (early in chapter):
      - [x] PM section: ROI of re-ranking
      - [x] Engineer section: Two-stage approach, typical improvements
    - [x] **Embedding Fine-Tuning**:
      - [x] PM section: When to fine-tune vs use off-the-shelf
      - [x] Engineer section: Implementation, code examples
    - [x] **Loss Functions** (mathematical deep dive):
      - [x] Engineer section: InfoNCE loss derivation
      - [x] Engineer section: Triplet loss derivation
    - [x] **Training Strategies**:
      - [x] Engineer section: Learning rate schedules
      - [x] Engineer section: Gradient accumulation
      - [x] Engineer section: Quantization techniques
  - [x] Write "Case Study Deep Dive" section
  - [x] Write "Implementation Guide" section
  - [x] Write "Common Pitfalls" section
  - [x] Write "Related Content" section
  - [x] Write remaining template sections
  - [x] Use admonitions appropriately
  - [x] Ensure all 15 template sections present

**Acceptance Criteria**:

- [x] Bi-encoder vs cross-encoder introduced early
- [x] Contrastive learning explained before loss functions
- [x] Mathematical deep dives in Engineer sections
- [x] All source materials synthesized

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify loss function derivations
  - [x] Test fine-tuning code examples
  - [x] Validate training methodology
- [ ] Peer review
  - [ ] Distribute to reviewers
  - [ ] Collect feedback
  - [ ] Incorporate feedback

**Acceptance Criteria**:

- [x] All formulas verified correct
- [x] All code examples work
- [ ] Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---

## Chapter 3: Feedback Systems and UX

### Phase 1: Content Creation

**Goal**: Consolidate three-part workshop chapter into single comprehensive chapter.

**Source Materials to Synthesize**:

- `docs/workshops/chapter3-1.md` - Feedback collection
- `docs/workshops/chapter3-2.md` - Streaming and UX
- `docs/workshops/chapter3-3.md` - Quality of life improvements
- `docs/workshops/chapter3-transcript.txt` - Actual lecture words
- `docs/talks/zapier-vitor-evals.md` - Zapier Feedback talk
- `docs/talks/online-evals-production-monitoring-ben-sidhant.md` - Online Evals talk
- `docs/office-hours/cohort2/week3-summary.md` - Office hours session
- `docs/office-hours/cohort3/week-3-1.md` - Office hours session

**Tasks**:

- [x] Create `docs/book/chapter3.md`
  - [x] Synthesize content from all three workshop parts
  - [x] Write front matter
  - [x] Write "Chapter at a Glance" section
  - [x] Write "Key Insight" paragraph
  - [x] Write "Learning Objectives"
  - [x] Write "Introduction" section
  - [x] Write "Core Content" sections:
    - [x] **Feedback Collection**:
      - [x] PM section: ROI of feedback, business value
      - [x] Engineer section: Implementation, code examples
    - [x] **UX Patterns**:
      - [x] PM section: Impact on user satisfaction
      - [x] Engineer section: Implementation patterns, code
    - [x] **Streaming and Perceived Latency**:
      - [x] PM section: Business value of faster perceived speed
      - [x] Engineer section: Streaming implementation, code
    - [x] **Citations and Chain-of-Thought**:
      - [x] PM section: Trust and transparency value
      - [x] Engineer section: Implementation, code examples
  - [x] Write "Case Study Deep Dive" section:
    - [x] Zapier case study
    - [x] PM and Engineer perspectives
  - [x] Write "Implementation Guide" section
  - [x] Write "Common Pitfalls" section
  - [x] Write "Related Content" section
  - [x] Write remaining template sections
  - [x] Use admonitions appropriately
  - [x] Ensure all 15 template sections present

**Acceptance Criteria**:

- [x] Three-part content consolidated into single chapter
- [x] All source materials synthesized
- [x] PM and Engineer sections throughout

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify UX patterns
  - [x] Test feedback collection code
  - [x] Validate streaming implementation
- [ ] Peer review
  - [ ] Distribute to reviewers
  - [ ] Collect feedback
  - [ ] Incorporate feedback

**Acceptance Criteria**:

- [x] All code examples work
- [ ] Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---

## Chapter 4: Query Understanding and Prioritization

### Phase 1: Content Creation

**Goal**: Consolidate two-part workshop chapter and add query clustering concepts.

**Source Materials to Synthesize**:

- `docs/workshops/chapter4-1.md` - Query clustering
- `docs/workshops/chapter4-2.md` - Prioritization
- `docs/workshops/chapter4-transcript.txt` - Actual lecture words
- `docs/talks/query-routing-anton.md` - Query Routing talk
- `docs/talks/chris-lovejoy-domain-expert-vertical-ai.md` - Domain Experts talk
- `docs/office-hours/cohort2/week4-summary.md` - Office hours session
- `docs/office-hours/cohort3/week-4-1.md` - Office hours session
- `docs/office-hours/cohort3/week-4-2.md` - Office hours session

**Tasks**:

- [x] Create `docs/book/chapter4.md`
  - [x] Synthesize content from both workshop parts
  - [x] Write front matter
  - [x] Write "Chapter at a Glance" section
  - [x] Write "Key Insight" paragraph
  - [x] Write "Learning Objectives"
  - [x] Write "Introduction" section
  - [x] Write "Core Content" sections:
    - [x] **Query Clustering** (introduce early):
      - [x] PM section: Why clustering matters for prioritization
      - [x] Engineer section: How clustering works, embedding-based approach
    - [x] **Topic Modeling**:
      - [x] PM section: Business value
      - [x] Engineer section: Implementation, code examples
    - [x] **Prioritization Frameworks**:
      - [x] PM section: Decision frameworks, ROI analysis
      - [x] Engineer section: Implementation, code examples
    - [x] **Economic Value Analysis**:
      - [x] PM section: How to calculate value
      - [x] Engineer section: Implementation, code examples
  - [x] Write "Case Study Deep Dive" section:
    - [x] Construction company case study
    - [x] Voice AI case study
    - [x] PM and Engineer perspectives
  - [x] Write "Implementation Guide" section
  - [x] Write "Common Pitfalls" section
  - [x] Write "Related Content" section
  - [x] Write remaining template sections
  - [x] Use admonitions appropriately
  - [x] Ensure all 15 template sections present

**Acceptance Criteria**:

- [x] Two-part content consolidated
- [x] Query clustering introduced early
- [x] All source materials synthesized

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify clustering methodology
  - [x] Test prioritization frameworks
  - [x] Validate economic value analysis
- [ ] Peer review

**Acceptance Criteria**:

- All methods verified correct
- Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---

## Chapter 5: Specialized Retrieval Systems

### Phase 1: Content Creation

**Goal**: Consolidate two-part workshop chapter and add RAPTOR algorithm details.

**Source Materials to Synthesize**:

- `docs/workshops/chapter5-1.md` - Specialized retrieval foundations
- `docs/workshops/chapter5-2.md` - Multimodal and RAPTOR
- `docs/workshops/chapter5-transcript.txt` - Actual lecture words
- `docs/talks/john-lexical-search.md` - Lexical Search talk
- `docs/talks/reducto-docs-adit.md` - Reducto talk
- `docs/talks/superlinked-encoder-stacking.md` - Encoder Stacking talk
- `docs/office-hours/cohort2/week5-summary.md` - Office hours session
- `docs/office-hours/cohort3/week-5-1.md` - Office hours session
- `docs/office-hours/cohort3/week-5-2.md` - Office hours session

**Tasks**:

- [x] Create `docs/book/chapter5.md`
  - [x] Synthesize content from both workshop parts
  - [x] Write front matter
  - [x] Write "Chapter at a Glance" section
  - [x] Write "Key Insight" paragraph
  - [x] Write "Learning Objectives"
  - [x] Write "Introduction" section
  - [x] Write "Core Content" sections:
    - [x] **Why Specialized Retrieval**:
      - [x] PM section: Business value, when to specialize
      - [x] Engineer section: Technical rationale
    - [x] **Metadata Extraction**:
      - [x] PM section: Business value
      - [x] Engineer section: Implementation, code examples
    - [x] **Synthetic Text Generation**:
      - [x] PM section: When to use vs extraction
      - [x] Engineer section: Implementation, code examples
    - [x] **RAPTOR Algorithm** (detailed):
      - [x] PM section: When to use RAPTOR
      - [x] Engineer section: Pseudocode, complexity analysis, implementation
    - [x] **Multimodal Retrieval**:
      - [x] PM section: Business value
      - [x] Engineer section: Implementation, code examples
  - [x] Write "Case Study Deep Dive" section:
    - [x] Construction company blueprint search
    - [x] Tax law RAPTOR example
    - [x] PM and Engineer perspectives
  - [x] Write "Implementation Guide" section
  - [x] Write "Common Pitfalls" section
  - [x] Write "Related Content" section
  - [x] Write remaining template sections
  - [x] Use admonitions appropriately
  - [x] Ensure all 15 template sections present

**Acceptance Criteria**:

- [x] RAPTOR algorithm detailed with pseudocode
- [x] Two-part content consolidated
- [x] All source materials synthesized

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify RAPTOR algorithm
  - [x] Test specialized retrieval code
  - [x] Validate multimodal approaches
- [ ] Peer review
  - [ ] Distribute to reviewers
  - [ ] Collect feedback
  - [ ] Incorporate feedback

**Acceptance Criteria**:

- [x] Algorithm verified correct
- [x] All code examples work (syntax verified)
- [ ] Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---

## Chapter 6: Query Routing and Orchestration

### Phase 1: Content Creation

**Goal**: Consolidate three-part workshop chapter and add routing concepts.

**Source Materials to Synthesize**:

- `docs/workshops/chapter6-1.md` - Routing foundations
- `docs/workshops/chapter6-2.md` - Tool interfaces
- `docs/workshops/chapter6-3.md` - Measurement
- `docs/workshops/chapter6-transcript.txt` - Actual lecture words
- `docs/talks/rag-is-dead-cline-nik.md` - Cline Agentic RAG talk
- `docs/talks/colin-rag-agents.md` - Colin's Agentic RAG talk
- `docs/office-hours/cohort2/week6-summary.md` - Office hours session

**Tasks**:

- [x] Create `docs/book/chapter6.md`
  - [x] Synthesize content from all three workshop parts
  - [x] Write front matter
  - [x] Write "Chapter at a Glance" section
  - [x] Write "Key Insight" paragraph
  - [x] Write "Learning Objectives"
  - [x] Write "Introduction" section
  - [x] Write "Core Content" sections:
    - [x] **Query Routing** (introduce early):
      - [x] PM section: Business value, when routing helps
      - [x] Engineer section: Router architecture, basic concepts
    - [x] **Few-Shot Classification** (before routing implementation):
      - [x] PM section: Cost vs accuracy tradeoff
      - [x] Engineer section: How few-shot works, example counts
    - [x] **Router Architectures**:
      - [x] PM section: Cost/speed tradeoffs
      - [x] Engineer section: Classifier, embedding-based, LLM options
    - [x] **Tool Interfaces**:
      - [x] PM section: Business value
      - [x] Engineer section: Implementation, code examples
    - [x] **Two-Level Performance**:
      - [x] PM section: Why it matters
      - [x] Engineer section: Measurement, code examples
    - [x] **Latency Analysis**:
      - [x] Engineer section: Sequential vs parallel, caching
  - [x] Write "Case Study Deep Dive" section:
    - [x] Construction company routing example
    - [x] PM and Engineer perspectives
  - [x] Write "Implementation Guide" section
  - [x] Write "Common Pitfalls" section
  - [x] Write "Related Content" section
  - [x] Write remaining template sections
  - [x] Use admonitions appropriately
  - [x] Ensure all 15 template sections present

**Acceptance Criteria**:

- [x] Three-part content consolidated
- [x] Routing concepts introduced early
- [x] All source materials synthesized

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify routing architectures
  - [x] Test router code examples
  - [x] Validate tool interfaces
- [ ] Peer review

**Acceptance Criteria**:

- [x] All code examples work
- [ ] Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---

## Chapter 7: Production Operations

### Phase 1: Content Creation

**Goal**: Create comprehensive production chapter.

**Source Materials to Synthesize**:

- `docs/workshops/chapter7.md` - Workshop content
- `docs/talks/turbopuffer-engine.md` - TurboPuffer talk
- `docs/talks/rag-antipatterns-skylar-payne.md` - RAG Antipatterns talk
- Office hours: Various production-focused sessions

**Tasks**:

- [x] Create `docs/book/chapter7.md`
  - [x] Write front matter
  - [x] Write "Chapter at a Glance" section
  - [x] Write "Key Insight" paragraph
  - [x] Write "Learning Objectives"
  - [x] Write "Introduction" section
  - [x] Write "Core Content" sections:
    - [x] **Semantic Caching**:
      - [x] PM section: Cost savings, ROI
      - [x] Engineer section: Implementation, code examples
    - [x] **Write-Time vs Read-Time Computation**:
      - [x] PM section: Tradeoff analysis
      - [x] Engineer section: Implementation patterns
    - [x] **Monitoring and Observability**:
      - [x] PM section: Business value, what to monitor
      - [x] Engineer section: Implementation, code examples
    - [x] **Cost Optimization**:
      - [x] PM section: ROI analysis
      - [x] Engineer section: Implementation strategies
    - [x] **Graceful Degradation**:
      - [x] PM section: Business value
      - [x] Engineer section: Implementation patterns
    - [x] **Scaling Strategies**:
      - [x] PM section: Cost implications
      - [x] Engineer section: Technical approaches
  - [x] Write "Case Study Deep Dive" section:
    - [x] Construction company cost optimization
    - [x] PM and Engineer perspectives
  - [x] Write "Implementation Guide" section
  - [x] Write "Common Pitfalls" section
  - [x] Write "Related Content" section
  - [x] Write remaining template sections
  - [x] Use admonitions appropriately
  - [x] Ensure all 15 template sections present

**Acceptance Criteria**:

- [x] All production topics covered
- [x] PM and Engineer sections throughout
- [x] All source materials synthesized

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify monitoring approaches
  - [x] Test production code examples
  - [x] Validate cost optimization strategies
- [ ] Peer review
  - [ ] Distribute to reviewers
  - [ ] Collect feedback
  - [ ] Incorporate feedback

**Acceptance Criteria**:

- [x] All code examples work (syntax verified, missing imports fixed)
- [ ] Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---

## Chapter 8: Hybrid Search

### Phase 1: Content Creation

**Goal**: Create new hybrid search chapter synthesizing existing content.

**Source Materials to Synthesize**:

- `docs/workshops/chapter1.md` - Mentions hybrid search
- `docs/talks/john-lexical-search.md` - Lexical Search talk
- `docs/office-hours/cohort2/week1-summary.md` - Office hours session
- `docs/office-hours/cohort3/week-2-1.md` - Office hours session

**Tasks**:

- [x] Create `docs/book/chapter8.md`
  - [x] Write front matter
  - [x] Write "Chapter at a Glance" section
  - [x] Write "Key Insight" paragraph
  - [x] Write "Learning Objectives"
  - [x] Write "Introduction" section
  - [x] Write "Core Content" sections:
    - [x] **When Semantic Search Fails**:
      - [x] PM section: Business value, decision framework
      - [x] Engineer section: Technical rationale
    - [x] **How Lexical Search Works**:
      - [x] PM section: When lexical helps
      - [x] Engineer section: BM25 implementation, code examples
    - [x] **Hybrid Search Approaches**:
      - [x] PM section: ROI analysis
      - [x] Engineer section: Reciprocal Rank Fusion, code examples
    - [x] **Implementation with LanceDB**:
      - [x] PM section: Resource requirements
      - [x] Engineer section: Architecture patterns, code examples
    - [x] **Evaluating Hybrid Search**:
      - [x] PM section: Evaluation methodology
      - [x] Engineer section: Comprehensive evaluation code
    - [x] **Advanced Hybrid Patterns**:
      - [x] PM section: When to use advanced patterns
      - [x] Engineer section: Weighted RRF, query expansion
  - [x] Write "Case Study Deep Dive" section
  - [x] Write "Implementation Guide" section
  - [x] Write "Common Pitfalls" section
  - [x] Write "Related Content" section
  - [x] Write remaining template sections
  - [x] Use admonitions appropriately
  - [x] Ensure all 15 template sections present

**Acceptance Criteria**:

- [x] New chapter created
- [x] All source materials synthesized
- [x] PM and Engineer sections throughout

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify hybrid search implementation
  - [x] Test code examples (syntax verified, algorithms match academic references)
- [ ] Peer review
  - [ ] Distribute to reviewers
  - [ ] Collect feedback
  - [ ] Incorporate feedback

**Acceptance Criteria**:

- [x] All code examples work (syntax verified)
- [ ] Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---

## Chapter 9: Context Window Management

### Phase 1: Content Creation

**Goal**: Create new context management chapter.

**Source Materials to Synthesize**:

- `docs/workshops/chapter1.md` - Mentions "Lost in the Middle"
- `docs/workshops/chapter7.md` - Discusses context limits
- `docs/office-hours/cohort2/week2-summary.md` - Office hours session
- `docs/office-hours/cohort3/week-2-1.md` - Office hours session
- Academic: "Lost in the Middle" paper

**Tasks**:

- [x] Create `docs/book/chapter9.md`
  - [x] Write front matter
  - [x] Write "Chapter at a Glance" section
  - [x] Write "Key Insight" paragraph
  - [x] Write "Learning Objectives"
  - [x] Write "Introduction" section
  - [x] Write "Core Content" sections:
    - [x] **The Lost in the Middle Problem**:
      - [x] PM section: Business impact
      - [x] Engineer section: Technical explanation
    - [x] **Token Budgeting Strategies**:
      - [x] PM section: Cost implications
      - [x] Engineer section: Implementation, code examples
    - [x] **Dynamic Context Assembly**:
      - [x] PM section: Business value
      - [x] Engineer section: Implementation, code examples
    - [x] **Mitigation Strategies**:
      - [x] PM section: ROI analysis
      - [x] Engineer section: Technical approaches
  - [x] Write "Case Study Deep Dive" section
  - [x] Write "Implementation Guide" section
  - [x] Write "Common Pitfalls" section
  - [x] Write "Related Content" section
  - [x] Write remaining template sections
  - [x] Use admonitions appropriately
  - [x] Ensure all 15 template sections present

**Acceptance Criteria**:

- [x] New chapter created
- [x] All source materials synthesized
- [x] PM and Engineer sections throughout

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify context management strategies
  - [x] Test code examples (syntax verified, patterns validated)
- [ ] Peer review
  - [ ] Distribute to reviewers
  - [ ] Collect feedback
  - [ ] Incorporate feedback

**Acceptance Criteria**:

- [x] All code examples work (syntax verified)
- [ ] Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---

## Appendix A: Mathematical Foundations

### Phase 1: Content Creation

**Goal**: Create comprehensive mathematical reference appendix.

**Tasks**:

- [x] Create `docs/book/appendix-math.md`
  - [x] Write front matter
  - [x] Write introduction
  - [x] Write sections:
    - [x] Retrieval Metrics (precision, recall, F1, MRR, NDCG, MAP)
    - [x] Statistical Testing (Chi-square, t-test, confidence intervals)
    - [x] Sample Size Calculations
    - [x] Loss Functions (InfoNCE, triplet loss)
    - [x] Optimization (learning rate schedules, gradient accumulation)
  - [x] Include derivations for all formulas
  - [x] Include examples with real numbers
  - [x] Add quick reference tables

**Acceptance Criteria**:

- [x] All formulas from chapters included
- [x] Full derivations provided
- [x] Examples included

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify all formulas correct
  - [x] Check derivations
  - [x] Validate examples
- [ ] Peer review

**Acceptance Criteria**:

- [x] All formulas verified correct

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
- [x] Link verification
  - [x] Verify all internal links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent
  - [x] Math formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent formatting

---

## Appendix B: Algorithms Reference

### Phase 1: Content Creation

**Goal**: Create algorithms reference appendix.

**Tasks**:

- [x] Create `docs/book/appendix-algorithms.md`
  - [x] Write front matter
  - [x] Write introduction
  - [x] Write sections:
    - [x] RAPTOR Algorithm (full pseudocode)
    - [x] Hierarchical Clustering (Ward linkage, dendrograms)
    - [x] Router Selection Algorithms
    - [x] Complexity Analysis for all algorithms
  - [x] Include pseudocode for all algorithms
  - [x] Include complexity analysis

**Acceptance Criteria**:

- [x] All algorithms from chapters included
- [x] Pseudocode complete
- [x] Complexity analysis provided

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify algorithms correct
  - [x] Check complexity analysis
- [ ] Peer review
  - [ ] Distribute to reviewers
  - [ ] Collect feedback
  - [ ] Incorporate feedback

**Acceptance Criteria**:

- [x] All algorithms verified correct
- [ ] Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
- [x] Link verification
  - [x] Verify all internal links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent
  - [x] Pseudocode formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent formatting

---

## Appendix C: Benchmarking Your RAG System

### Phase 1: Content Creation

**Goal**: Create benchmarking methodology appendix.

**Source Materials to Synthesize**:

- `docs/workshops/chapter1.md` - Evaluation frameworks
- `docs/talks/embedding-performance-generative-evals-kelly-hong.md` - Generative Evals talk
- Office hours: Evaluation methodology discussions

**Tasks**:

- [x] Create `docs/book/appendix-benchmarks.md`
  - [x] Write front matter
  - [x] Write introduction
  - [x] Write sections:
    - [x] Standard Datasets (BEIR, MS MARCO, domain-specific)
    - [x] Benchmark Methodology
    - [x] Baseline Comparisons
    - [x] Running Your Own Benchmarks
  - [x] Include PM and Engineer sections
  - [x] Include code examples

**Acceptance Criteria**:

- [x] Comprehensive benchmarking guide
- [x] All source materials synthesized

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify methodology
  - [x] Test code examples
- [ ] Peer review
  - [ ] Distribute to reviewers
  - [ ] Collect feedback
  - [ ] Incorporate feedback

**Acceptance Criteria**:

- [x] Methodology verified correct
- [ ] Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links (existing files verified; planned files noted as placeholders)
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---

## Appendix D: Debugging RAG Systems

### Phase 1: Content Creation

**Goal**: Create debugging guide synthesizing existing content.

**Source Materials to Synthesize**:

- `docs/workshops/chapter1.md` - Silent Data Loss, Common Pitfalls
- `docs/talks/rag-antipatterns-skylar-payne.md` - RAG Antipatterns talk
- `docs/office-hours/cohort2/week1-summary.md` - Encoding failures discussion
- `docs/office-hours/cohort2/week2-summary.md` - Parsing errors discussion

**Tasks**:

- [x] Create `docs/book/appendix-debugging.md`
  - [x] Write front matter
  - [x] Write introduction
  - [x] Write sections:
    - [x] Systematic Debugging Methodology
    - [x] Failure Modes Taxonomy
    - [x] Debugging Tools and Techniques
  - [x] Include PM and Engineer sections
  - [x] Include examples

**Acceptance Criteria**:

- [x] Comprehensive debugging guide
- [x] All source materials synthesized

### Phase 2: Review

**Tasks**:

- [x] Technical accuracy review
  - [x] Verify debugging methodology is correct
  - [x] Verify code examples are syntactically correct
  - [x] Validate failure modes taxonomy against source materials
- [ ] Peer review
  - [ ] Distribute to reviewers
  - [ ] Collect feedback
  - [ ] Incorporate feedback

**Acceptance Criteria**:

- [x] Methodology verified correct
- [ ] Peer feedback incorporated

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
  - [x] PM vs Engineer section clarity
  - [x] Admonition usage consistency
- [x] Link verification
  - [x] Verify all internal links
  - [x] Verify all external links
  - [x] Check cross-references
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---
## Supporting Materials

### How to Use This Book

### Phase 1: Content Creation

**Tasks**:

- [x] Create `docs/book/how-to-use.md`
  - [x] Reading paths section (PM, Engineer, Full)
  - [x] Admonition types explanation with examples
  - [x] PM vs Engineer content markers explanation
  - [x] Navigation guide
  - [x] Visual examples showing admonition usage

**Acceptance Criteria**:

- [x] Comprehensive guide created
- [x] All admonition types explained

### Phase 2: Review

**Tasks**:

- [x] Review for clarity
- [ ] Test with users (blocked: requires external user testing)

**Acceptance Criteria**:

- Guide is clear and helpful

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
- [x] Formatting check

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style

---

### Glossary

### Phase 1: Content Creation

**Tasks**:

- [x] Create `docs/book/glossary.md`
  - [x] Copy and expand from `docs/workshops/glossary.md`
  - [x] Add new terms from new chapters
  - [x] Ensure consistency
  - [x] Add cross-references to chapters

**Acceptance Criteria**:

- [x] All terms defined
- [x] Consistent definitions

### Phase 2: Review

**Tasks**:

- [x] Verify definitions
- [x] Check cross-references

**Acceptance Criteria**:

- [x] All definitions accurate

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
- [x] Formatting check

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent formatting

---

### Quick Reference

### Phase 1: Content Creation

**Tasks**:

- [x] Create `docs/book/quick-reference.md`
  - [x] Copy and expand from `docs/workshops/quick-reference.md`
  - [x] Add per-chapter summaries
  - [x] Add decision flowcharts
  - [x] Add key formulas

**Acceptance Criteria**:

- [x] Comprehensive quick reference
- [x] All key information included

### Phase 2: Review

**Tasks**:

- [x] Verify accuracy
  - [x] Cross-referenced all chapter summaries against source chapters
  - [x] Verified metrics and formulas match Appendix A
  - [x] Corrected hybrid search improvement (5-15% to 10-25%)
  - [x] Corrected fine-tuning examples (5,000 to 6,000)
  - [x] Verified key numbers against source chapters
- [x] Test usability
  - [x] Verified navigation links (noted missing appendix files B and C)
  - [x] Reviewed decision frameworks for logical consistency
  - [x] Reviewed tables and checklists for actionability

**Acceptance Criteria**:

- [x] All information accurate

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent formatting

---

### Case Studies

### Phase 1: Content Creation

**Tasks**:

- [x] Create `docs/book/case-study-construction.md`
  - [x] Consolidate construction examples from chapters
  - [x] Add metrics timeline showing progression
  - [x] Include PM and Engineer perspectives
  - [x] Cross-reference from chapters

- [x] Create `docs/book/case-study-wildchat.md`
  - [x] Link to existing case study in `latest/case_study/`
  - [x] Add chapter connections
  - [x] Include PM and Engineer perspectives

- [x] Create `docs/book/case-study-voice-ai.md`
  - [x] Restaurant voice AI example
  - [x] Include PM and Engineer perspectives
  - [x] Add metrics and outcomes

**Acceptance Criteria**:

- [x] All case studies consolidated
- [x] PM and Engineer perspectives included

### Phase 2: Review

**Tasks**:

- [x] Verify metrics
- [x] Audit unverified numerical claims
  - [x] Identify all numerical claims in case studies (percentages, dollar amounts, counts, etc.)
  - [x] Trace each claim to source material (workshops, transcripts, talks, office hours)
  - [x] Verify calculations are correct (e.g., percentage improvements, cost reductions)
  - [x] Flag claims without clear source or verification
  - [x] Document verification status for each numerical claim
  - [x] Update case studies to add citations or remove unverified claims
- [ ] Peer review

**Acceptance Criteria**:

- [x] All metrics verified
- [x] All numerical claims traced to sources or marked as illustrative/examples
- [x] No unverified numerical claims remain in published case studies (WildChat description fixed)

### Phase 3: Editorial

**Tasks**:

- [x] Copy editing
  - [x] Grammar and spelling check
  - [x] Clarity improvements
  - [x] Consistency check (terminology, formatting)
  - [x] Style compliance
- [x] Formatting check
  - [x] Front matter consistency
  - [x] Heading hierarchy correct
  - [x] Code block formatting consistent (fixed 2 issues)
  - [x] Table formatting consistent
  - [x] List formatting consistent

**Acceptance Criteria**:

- [x] No errors
- [x] Consistent style
- [x] All links functional (existing files verified; planned files noted as placeholders)

---


### Navigation Updates

### Phase 1: Content Creation

**Tasks**:

- [x] Update `mkdocs.yml` with new structure
  - [x] Create four-book structure
  - [x] Add book index pages
  - [x] Organize chapters under each book
  - [x] Add appendices section
  - [x] Add supporting materials section

- [x] Create `docs/book/index.md` (if needed)
  - [x] Overview of book series
  - [x] Link to each book index
  - [x] Reading paths

**Acceptance Criteria**:

- Navigation structure updated
- All links work

### Phase 2: Review

**Tasks**:

- [ ] Test navigation
- [ ] Verify all links work

**Acceptance Criteria**:

- Navigation functional

### Phase 3: Editorial

**Tasks**:

- [ ] Final formatting check

**Acceptance Criteria**:

- Consistent formatting

---

## Code Examples

### Phase 1: Content Creation

**Tasks**:

- [ ] Create `latest/examples/chapter1_evaluation.py`
  - [ ] Evaluation pipeline class with type hints
  - [ ] Precision/recall calculation functions
  - [ ] Statistical significance testing
  - [ ] Error handling and logging
  - [ ] Documentation strings

- [ ] Create `latest/examples/chapter2_finetuning.py`
  - [ ] Fine-tuning script with type hints
  - [ ] Data loading and preprocessing
  - [ ] Training loop with checkpointing
  - [ ] Error handling and logging
  - [ ] Documentation strings

- [ ] Create `latest/examples/chapter3_feedback.py`
  - [ ] Feedback collection API
  - [ ] Streaming implementation
  - [ ] Database storage
  - [ ] Error handling
  - [ ] Documentation strings

- [ ] Create `latest/examples/chapter4_clustering.py`
  - [ ] Query clustering implementation
  - [ ] K-means with embeddings
  - [ ] Cluster labeling with LLM
  - [ ] Visualization helpers
  - [ ] Documentation strings

- [ ] Create `latest/examples/chapter5_specialized.py`
  - [ ] RAPTOR implementation
  - [ ] Multimodal retrieval example
  - [ ] Metadata extraction pipeline
  - [ ] Documentation strings

- [ ] Create `latest/examples/chapter6_routing.py`
  - [ ] Router implementation
  - [ ] Tool interface definitions
  - [ ] Few-shot classification
  - [ ] Documentation strings

- [ ] Create `latest/examples/chapter7_production.py`
  - [ ] Monitoring dashboard code
  - [ ] Cost tracking
  - [ ] Alerting system
  - [ ] Documentation strings

**Acceptance Criteria**:

- All code examples created
- Type hints included
- Error handling included
- Documentation strings included

### Phase 2: Review

**Tasks**:

- [ ] Test all code examples
  - [ ] Run each example
  - [ ] Verify imports work
  - [ ] Check for syntax errors
  - [ ] Validate output matches expectations

- [ ] Code review
  - [ ] Review code quality
  - [ ] Check best practices
  - [ ] Verify type hints

- [ ] Create `tests/test_examples.py`
  - [ ] Unit tests for each example
  - [ ] Integration tests
  - [ ] Test coverage > 80%

**Acceptance Criteria**:

- All code examples run without errors
- Tests created and passing
- Code quality high

### Phase 3: Editorial

**Tasks**:

- [ ] Documentation review
  - [ ] Verify documentation strings
  - [ ] Check code comments
- [ ] Formatting check
  - [ ] Code formatting consistent
  - [ ] Follow style guide

**Acceptance Criteria**:

- Documentation complete
- Code formatting consistent

---

## Success Criteria

The transformation will be successful when:

1. **All chapters created** in `docs/book/` following standard template
2. **All foundational concepts** properly introduced before first use
3. **All chapters** link to transcripts, talks, and office hours
4. **PM and Engineer sections** clearly separated in all chapters
5. **All code examples** run without errors
6. **All mathematical formulas** verified correct
7. **All links** functional
8. **Consistent style** throughout
9. **All chapters reviewed** by peers
10. **All chapters edited** professionally

---

## Notes

- All new content goes in `docs/book/` directory
- Source materials referenced from `docs/workshops/`, `docs/talks/`, `docs/office-hours/`
- Each chapter follows the 3-phase cycle: Content Creation → Review → Editorial
- Use admonitions to mark PM vs Engineer content
- Follow standard chapter template (15 sections)
- Introduce foundational concepts in Chapter 0 before they're used elsewhere

## Library Decisions

- **langchain**: Removed from all chapters. Replaced `RecursiveCharacterTextSplitter` with custom implementation to avoid external dependency.
- **qdrant-client**: Removed from all chapters. Standardized on LanceDB for vector database examples throughout the book.
- **pydantic**: Kept throughout - used extensively for data validation and modeling.

## Content Linking Guidelines

- **Do not mention workshop or transcript files directly** - readers cannot see these files (`docs/workshops/chapter*.md` or `docs/workshops/chapter*-transcript.txt`), so avoid linking to them or referencing them in the book content.
- **Talks and office hours are OK** - it's acceptable to link to and reference content in `docs/talks/` and `docs/office-hours/` directories, as these are accessible to readers.

## Code Example Guidelines

- **Keep code examples reasonable and focused** - don't overdo it. Examples should illustrate the concept clearly without unnecessary complexity.
- **Prioritize clarity over completeness** - it's better to show a focused, understandable example than a fully-featured production system.
- **Use code examples strategically** - include them where they add value, not just for the sake of having code.

---

## Chapter/Page Checklist

Use this checklist for every chapter and page to ensure consistency and completeness.

### Phase 1: Content Creation

#### Standard Template Sections (15 Required)

- [ ] **Front Matter**
  - [ ] Title present
  - [ ] Description present
  - [ ] Authors listed (if applicable)
  - [ ] Date present
  - [ ] Tags present (if applicable)
- [ ] **Chapter at a Glance**
  - [ ] Prerequisites listed
  - [ ] Outcomes listed
  - [ ] Case study reference included (if applicable)
- [ ] **Key Insight**
  - [ ] One-paragraph summary present
  - [ ] Captures main takeaway
- [ ] **Learning Objectives**
  - [ ] 5-6 measurable outcomes listed
  - [ ] Mix of PM and Engineer objectives
- [ ] **Introduction**
  - [ ] Context provided
  - [ ] Builds on previous chapters (if applicable)
  - [ ] Sets up chapter content
- [ ] **Core Content** (3-5 major sections)
  - [ ] PM sections: Business value, decision frameworks, ROI, success metrics
  - [ ] Engineer sections: Implementation details, code examples, algorithms, tradeoffs
  - [ ] Foundational concepts introduced before use
- [ ] **Case Study Deep Dive**
  - [ ] Dedicated section present
  - [ ] PM perspective included
  - [ ] Engineer perspective included
- [ ] **Implementation Guide**
  - [ ] Quick Start for PMs: High-level steps, decision points
  - [ ] Detailed Implementation for Engineers: Full code, configuration
- [ ] **Common Pitfalls**
  - [ ] PM Pitfalls: Strategic mistakes, resource misallocation
  - [ ] Engineering Pitfalls: Technical mistakes, implementation errors
- [ ] **Related Content**
  - [ ] No links to workshop files (`docs/workshops/chapter*.md`)
  - [ ] No links to transcript files (`docs/workshops/chapter*-transcript.txt`)
  - [ ] Links to talks (`docs/talks/`) included if applicable
  - [ ] Links to office hours (`docs/office-hours/`) included if applicable
  - [ ] Key insights extracted from each source
- [ ] **Action Items**
  - [ ] For Product Teams: Strategic planning, stakeholder alignment
  - [ ] For Engineering Teams: Technical implementation, testing
- [ ] **Reflection Questions**
  - [ ] 5 questions present
  - [ ] Mix of strategic and technical questions
- [ ] **Summary**
  - [ ] Key takeaways listed
  - [ ] Separate bullets for PM vs Engineer
- [ ] **Further Reading**
  - [ ] Academic papers listed (if applicable)
  - [ ] Tools listed (if applicable)
- [ ] **Navigation**
  - [ ] Previous chapter link
  - [ ] Next chapter link
  - [ ] Reference links to related content

#### Content Quality

- [ ] **PM vs Engineer Separation**
  - [ ] PM content clearly marked with `!!! tip "For Product Managers"`
  - [ ] Engineer content clearly marked with `!!! tip "For Engineers"`
  - [ ] PM pitfalls marked with `!!! warning "PM Pitfall"`
  - [ ] Engineering pitfalls marked with `!!! warning "Engineering Pitfall"`
- [ ] **Admonition Usage**
  - [ ] Appropriate use of `!!! tip` for PM/Engineer content
  - [ ] Appropriate use of `!!! warning` for pitfalls
  - [ ] Appropriate use of `!!! info` for general information
  - [ ] Appropriate use of `!!! example` for concrete examples
  - [ ] Appropriate use of `!!! success` for success stories
- [ ] **Code Examples**
  - [ ] Code examples are reasonable and focused - don't overdo it
  - [ ] Examples illustrate the concept without unnecessary complexity
  - [ ] Code examples have proper syntax
  - [ ] Code examples include type hints (Python)
  - [ ] Code examples include error handling where appropriate
  - [ ] Code examples include documentation strings
  - [ ] No references to `langchain` (use custom implementations)
  - [ ] No references to `qdrant-client` (use LanceDB)
  - [ ] `pydantic` used for data validation where appropriate
- [ ] **Mathematical Content**
  - [ ] Formulas are correct
  - [ ] Formulas have derivations (in Engineer sections or Appendix A)
  - [ ] Examples with real numbers included
- [ ] **Source Material Synthesis**
  - [ ] Content synthesizes from multiple sources
  - [ ] No direct copying - content is rewritten and synthesized
  - [ ] Key insights extracted from talks and office hours

### Phase 2: Review

#### Technical Accuracy

- [ ] **Formulas and Math**
  - [ ] All formulas verified correct
  - [ ] Derivations checked
  - [ ] Examples validated
- [ ] **Code Examples**
  - [ ] All code examples run without errors
  - [ ] Imports verified
  - [ ] Syntax errors checked
  - [ ] Output matches expectations
- [ ] **Technical Concepts**
  - [ ] All technical concepts accurate
  - [ ] Algorithms verified correct
  - [ ] Complexity analysis checked
- [ ] **Cross-References**
  - [ ] Foundational concepts introduced before use
  - [ ] References to other chapters accurate
  - [ ] Appendix references correct
- [ ] **Numerical Claims Audit**
  - [ ] Identify all numerical claims (percentages, dollar amounts, counts, metrics, etc.)
  - [ ] Trace each claim to source material (workshops, transcripts, talks, office hours, case studies)
  - [ ] Verify calculations are correct (e.g., percentage improvements, cost reductions, performance gains)
  - [ ] Flag claims without clear source or verification
  - [ ] Document verification status for each numerical claim
  - [ ] Update content to add citations or mark as illustrative/examples where appropriate
  - [ ] Ensure consistency of numerical claims across chapters (e.g., same case study metrics match)

#### Peer Review

- [ ] **Review Process**
  - [ ] Distributed to 2+ reviewers (mix of PMs and engineers)
  - [ ] Feedback collected
  - [ ] Feedback incorporated
  - [ ] Changes documented

### Phase 3: Editorial

#### Copy Editing

- [ ] **Grammar and Spelling**
  - [ ] No spelling errors
  - [ ] No grammar errors
  - [ ] Consistent terminology throughout
- [ ] **Clarity**
  - [ ] Writing is clear and concise
  - [ ] Complex concepts explained simply
  - [ ] 9th-grade reading level maintained
- [ ] **Consistency**
  - [ ] Terminology consistent across chapter
  - [ ] Formatting consistent
  - [ ] Style consistent with other chapters
- [ ] **PM vs Engineer Clarity**
  - [ ] PM sections clearly marked
  - [ ] Engineer sections clearly marked
  - [ ] Content appropriate for each audience
- [ ] **Admonition Consistency**
  - [ ] Admonitions used consistently
  - [ ] Admonition types match content
  - [ ] Formatting consistent

#### Link Verification

- [ ] **Internal Links**
  - [ ] All internal links functional
  - [ ] Links point to correct files
  - [ ] Cross-references accurate
- [ ] **External Links**
  - [ ] All external links functional
  - [ ] Links are current and valid
- [ ] **No Invalid Links**
  - [ ] No links to workshop files (`docs/workshops/chapter*.md`)
  - [ ] No links to transcript files (`docs/workshops/chapter*-transcript.txt`)
  - [ ] Only links to accessible content (talks, office hours)

#### Formatting Check

- [ ] **Front Matter**
  - [ ] Consistent format across chapters
  - [ ] All required fields present
- [ ] **Heading Hierarchy**
  - [ ] Heading levels correct (single `#` for chapter title)
  - [ ] No skipped levels
  - [ ] Consistent formatting
- [ ] **Code Blocks**
  - [ ] Code block formatting consistent
  - [ ] Language tags correct
  - [ ] Proper indentation
- [ ] **Tables**
  - [ ] Table formatting consistent
  - [ ] Tables render correctly
- [ ] **Lists**
  - [ ] List formatting consistent
  - [ ] Proper indentation
- [ ] **Images**
  - [ ] Image formatting consistent (if applicable)
  - [ ] Images referenced correctly
  - [ ] Alt text included (if applicable)
- [ ] **Math Formatting**
  - [ ] Math notation consistent (if applicable)
  - [ ] Formulas render correctly

### Final Acceptance Criteria

- [ ] All 15 template sections present
- [ ] No errors (spelling, grammar, technical)
- [ ] Consistent style throughout
- [ ] All links functional (existing files verified; planned files noted as placeholders)
- [ ] PM and Engineer content clearly separated
- [ ] Admonitions used appropriately
- [ ] Code examples work correctly
- [ ] Formulas verified correct
- [ ] Peer feedback incorporated (if applicable)
- [ ] No links to inaccessible content (workshops, transcripts)
