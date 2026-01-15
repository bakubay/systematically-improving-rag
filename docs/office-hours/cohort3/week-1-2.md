---
title: Week 1 - Office Hour 2
date: "2024-05-22"
cohort: 3
week: 1
session: 2
type: Office Hour
transcript: ../GMT20250522-174339_Recording.transcript_(1).vtt
description: RAG implementation challenges, precision-recall tradeoffs, and practical business applications of AI systems
topics:
  - Precision in LLMs and RAG
  - Small Language Models in RAG Architecture
  - Multi-turn Conversation Quality
  - Business Value through Data Analysis
  - Balancing Technical and Business Outcomes
  - AI Development Best Practices
---

# Week 1 - Office Hour 2 (May 22)

Study Notes:

I hosted an office hours session focused on RAG implementation challenges, precision-recall tradeoffs, and practical business applications of AI systems. Here are my insights on evaluating RAG systems, understanding model sensitivity to context quality, and identifying high-value business opportunities through data analysis rather than just technical improvements.

---

---

## How should we understand precision in the context of LLMs and RAG?

Precision has become increasingly important as most people are comfortable adding substantial information to the context window. The key question is how sensitive these models are to ambiguous or irrelevant information.

Most current models are very good at recall because they're optimized for "needle in the haystack" tests. Due to the attention mechanism with sliding windows, they can typically catch important information even when it's buried among other content. However, the sensitivity to irrelevant information is less well-studied.

As we increase the amount of data in the context, we need to understand what precision looks like and how it correlates with factuality or answer quality. Earlier models like GPT-3.5 were quite sensitive to long context with low precision - if you gave them too much information, they would "overthink" because of the volume of content to process.

This is why it's valuable to experiment with different retrieval settings: "For the top 10 documents, this is my precision and recall; for my top 25 documents, this is my precision and recall. If my recall is not increasing as a function of K (the number of documents), that's where I decide to set a threshold."

Some people set thresholds based on re-ranker scores, but that can be dangerous since these scores aren't true probabilities. You can't just set 0.5 as a universal threshold - you need to understand the precision-recall tradeoffs for your specific application.

**_Key Takeaway:_** Models can be sensitive to low-precision context, where irrelevant information causes them to incorporate red herrings into answers. Testing different document count thresholds is more reliable than using arbitrary re-ranker score cutoffs.

## What role can small language models play in a RAG architecture?

Small language models can serve several valuable functions within a RAG system:

First, they can rewrite queries quickly and efficiently. For example, if a user asks "What is happening this week?", a small model could convert this into a structured query like a JSON object specifying "search all documents where the datetime is between today and today minus 7 days." This type of entity resolution and query parsing doesn't require the full knowledge of a large model but benefits from the lower latency of smaller models.

Second, small models can build better embedding spaces. Most current embedding models are relatively simple, but a small language model fine-tuned on your specific task could significantly improve embedding quality or re-ranking performance.

In this context, I think of "small" as meaning lower latency with less world knowledge - models that can perform specific tasks efficiently without needing to understand everything about the world.

## How can we measure quality in multi-turn conversations?

When evaluating multi-turn exchanges like simulated customer interactions or teacher training scenarios, there are several approaches to consider.

One approach is to model the interaction as a state machine or "LangDraft" type model where there are defined states that can be traversed. For example, in a customer support scenario, you might have an intake state, followed by various question states, triage states, and resolution states.

We've used this with "LLM therapists" where the system might say, "I can tell you're angry. Let me transition you to this sub-draft that deals with negative emotions." The user remains in that state until they've acknowledged something, then returns to the main flow.

Another approach is to use rubrics and data mining. We extract examples from historical transactions that match specific rubric criteria, then have experts score these examples. These scored examples become few-shot examples for future evaluations.

For instance, with venture capital funding requests, we might extract 200 examples of founders discussing how well they know each other, then have experts grade these as good, bad, or great. This creates a training set for evaluating future conversations.

The model we build with these rubrics typically extracts scores for 30+ criteria, with LLMs giving scores from 0-4, which then feed into a logistic regression model. This makes the evaluation somewhat explainable - if a candidate gets prioritized unexpectedly, we can see which features drove that decision.

**_Key Takeaway:_** For evaluating multi-turn conversations, combine state machines to enforce guardrails with human-labeled examples to create scoring rubrics. Using a simple model like logistic regression on top of LLM-generated feature scores maintains interpretability.

## How do you approach data analysis to find business value in AI applications?

My favorite content in the course is actually weeks 4 and 5, where I cover my process for data analysis and uncovering new capabilities needed in AI systems.

When analyzing data from AI applications, I look for two main types of issues:

1. Inventory issues: These occur when the system lacks the necessary data to fulfill user requests. For example, if users search for content that doesn't exist in your database, the solution isn't to improve the AI - it's to add the missing content. Many companies don't realize their inventory might be the problem rather than their AI.

1. Capabilities issues: These involve functionality gaps where the system can't perform certain types of queries or filters. For instance, you might need to add metadata filters or specialized search capabilities to handle specific user needs.

I've found tremendous business value by identifying these issues through data analysis. In one case with a restaurant voice AI system, we discovered that when the AI attempted upselling, it generated 20% more revenue 50% of the time - a 10% overall increase. However, the agent only tried upselling in 9% of calls.

The solution wasn't to improve the AI's core capabilities but to add a simple check ensuring the agent always asks if the customer wants anything else before ending the call. This small change could generate an additional $2 million in revenue by increasing upselling attempts from 9% to 40%.

For me, the most enjoyable work is identifying these business opportunities that don't necessarily require complex AI improvements. Software engineers often aren't trained to think this way, but my background in data science makes this approach natural.

**_Key Takeaway:_** The biggest business value often comes from analyzing usage patterns to identify inventory gaps or missing capabilities, rather than improving core AI performance. Simple changes like adding missing data or implementing basic business rules can deliver millions in value.

## How do you balance technical implementation with business outcomes?

I've worked with many companies where they think they want me to make their AI better, but my actual job is to make their business better. There's often substantial money to be captured by focusing on business outcomes rather than technical improvements.

For example, with a construction project, I spoke with contractors to understand their actual pain points. While they initially thought they needed better document search, the real issue was tracking delays and identifying who was causing them. This led us to implement contact search with metadata filters - a solution that addressed a $100,000/month problem.

Similarly, with Netflix, if users search for "Oscar-nominated" movies but get results about Oscar Wilde or actors named Oscar, the solution might not be more sophisticated AI. It could be as simple as paying IMDB for better awards metadata.

I'm constantly looking for these opportunities where a relatively simple technical solution can unlock significant business value. This approach has been much more impactful than pursuing technical sophistication for its own sake.

## What are your thoughts on the latest AI developments like Claude 3?

I'm currently reviewing the entire Instructor codebase to adapt it for Claude 3. The model is making about 15 pull requests for me, so we'll see how that goes.

Regarding the guest speakers we've had, I found the Chroma presentation particularly valuable for its hands-on, practical approach. While the Exa presentation was more high-level and story-focused, both offered valuable perspectives.

I try to balance technical depth with accessibility in these sessions. When Nils gave his talk, it quickly became very technical with neural network diagrams and mathematical equations, and I could see people leaving the call. It's challenging to find the right balance between technical content and storytelling.

## How should we approach building RAG applications for course materials?

If someone wanted to build a RAG application over all the course transcripts and office hours, I'd love to see that. However, this would quickly reveal the limitations of simple chunking approaches.

You'd discover that people have different capability requests - like wanting to know who asked specific questions or what was discussed in a particular week. This would require metadata filters for cohort numbers, transcript types (lectures vs. office hours vs. lightning lessons), and speaker identification.

You might also need to handle requests for information about guest speakers, like their LinkedIn profiles. All of these are inventory issues that could be solved by ensuring you have the right metadata alongside your content.

For a dataset as small as course transcripts, long-context models like Claude 3 might work well without complex RAG. It's really the enterprise use cases with massive document collections that need sophisticated RAG approaches.

## How do you handle UI/UX development for AI applications?

I try to write most things as command line tools - I'm a "filthy machine learning Python engineer" who finds any UI to be too much work. Even Streamlit feels excessive to me when a command line interface would suffice.

That said, Claude has demonstrated how well you can do with thoughtful UX patterns. In Week 3, I'll talk about UX patterns that make applications feel responsive - like how Claude shows progress counters as it's uploading and downloading tokens, ensuring something on the page is always moving to indicate work is happening.

For those who need to build UIs but lack JavaScript skills, LLMs are remarkably good at writing JavaScript. I've built many bespoke data labeling applications in just 5 hours by prompting models to convert JSON structures to PostgreSQL databases and build the corresponding UIs.

The software can be ephemeral enough that I don't worry about long-term maintenance. For more polished applications, I recommend checking out Lovable.dev - I've built about 20 apps with them that work quite well.

**_Key Takeaway:_** Focus on learning the concepts rather than specific implementation details. Modern LLMs can generate high-quality UI code, making it easier than ever to build functional applications without deep frontend expertise.

## What's been your most rewarding project in the AI space?

My background is in physics - I initially thought the universe would generate the most interesting datasets. Then I went to Facebook because I believed people-to-people interactions would be the most fascinating data. Now I'm focused on people-to-AI interactions, and in the future, it will likely be AI-to-AI interactions. I'm essentially chasing the most interesting datasets I can analyze.

The most rewarding projects have been those where data analysis revealed clear business opportunities. For instance, with the restaurant voice AI system, identifying the upselling opportunity was straightforward but incredibly valuable.

I enjoy working with teams that have access to subject matter experts who can help interpret the data. For the construction project, I spoke with contractors wearing hard hats on Zoom to understand why certain questions were valuable and what problems they were trying to solve.

This approach of combining data analysis with domain expertise has consistently led to high-impact solutions that address real business needs rather than just technical challenges.

## Final thoughts on balancing course workload

I recognize that the course material can be overwhelming, especially for those balancing it with full-time jobs. We'll have no notebooks in Week 3, which should provide a buffer, and you'll always have access to the Slack channel even after the 6 weeks are over.

For those feeling overwhelmed, remember that many people take multiple cohorts to fully absorb the material. The flexible structure is intentional - unlike more prescriptive courses, this approach allows you to focus on what's most relevant to your specific needs.

As one participant noted, they've found at least one "golden nugget" from each session so far, including the introduction where I presented the "sandwich view" of RAG systems. These conceptual frameworks can provide clarity when you're deep in implementation details.

Remember that the AI field is moving incredibly quickly, and none of us can absorb everything. The goal isn't to become an expert on everything but to get really good at leveraging AI to stay ahead of everyone else.

---

FAQs

## How can I balance this course with my day job?

Managing this course alongside your regular work can be challenging. Many students find success by aligning the course with existing work projects, allowing them to apply what they're learning directly to their professional tasks. If you don't have a relevant project, the course notebooks provide boilerplate code you can use as a starting point. Remember that Week 3 has no notebooks, which gives you a buffer to catch up if needed. The course is designed with some flexibility, so you can prioritize the most relevant content for your needs.

## What should I do if I don't have a specific project to apply the course material to?

You can start with the boilerplate code provided in the notebooks. These are designed to demonstrate key concepts even without a specific application in mind. Additionally, consider looking for datasets from colleagues or within your organization that might benefit from the techniques taught in the course. Many people have conversation data or other information they're not sure how to leverage effectively. The course materials are structured to help you experiment with these techniques regardless of whether you have a specific project.

## How are the course materials structured?

The course includes lecture videos, notebooks with code examples, office hours, and summary notes. Each set of notebooks focuses on a specific theme or concept, such as synthetic data generation or evaluation metrics. The notebooks are designed to be practical and applicable to real-world scenarios. Week 3 has no notebooks, providing a buffer period. Weeks 4-5 focus on data analysis processes and building specific tools based on identified needs. The course also includes guest lectures from industry experts to provide different perspectives.

## Where can I find the summary notes and FAQs?

Currently, summary notes are posted in Slack, but they will eventually be available in Notion or another website format. Many students find these notes helpful as they allow them to focus more on understanding the content rather than taking extensive notes during lectures.

## What's the instructor's approach to evaluating RAG applications?

The instructor emphasizes a data-driven approach to evaluations rather than relying on subjective assessments. This includes measuring precision and recall for different numbers of retrieved documents, understanding how models respond to ambiguous information, and using metrics to make informed decisions about system design. The instructor discourages using adjectives to describe performance and instead encourages teams to use numbers, plots, and quantifiable metrics to evaluate their systems.

## How can small language models be used in a RAG architecture?

Small language models can serve several purposes in a RAG architecture. They can be used to quickly rewrite queries, breaking them down into more structured formats. They can help build better embedding spaces or re-rankers that are fine-tuned for specific tasks. Small language models generally offer lower latency with less world knowledge, making them suitable for specific components of a RAG system where full context understanding isn't necessary.

## What are the most valuable insights from the course so far?

Many students highlight the "sandwich view" of RAG systems (where RAG is presented as a recommendation system between LLM layers) as particularly insightful. The course provides practical "golden nuggets" in each session, including frameworks for thinking about RAG applications, evaluation techniques, and implementation strategies. The balance between technical details and storytelling across different guest lectures has been valuable for understanding both theoretical concepts and practical applications.

## What's the instructor's perspective on building UI/UX for LLM applications?

The instructor suggests focusing on understanding concepts rather than specific UI technologies. Command-line tools can be highly effective for many applications, and modern LLMs are excellent at generating JavaScript and other frontend code when needed. Understanding server-sent events and streaming is particularly important for creating responsive LLM applications. The instructor emphasizes that streaming is essential for good user experience - applications without streaming capabilities are generally considered subpar in the current landscape.

## How does the instructor approach business value in AI projects?

The instructor focuses on identifying business value through data analysis rather than just improving AI capabilities. This involves analyzing user interactions, identifying patterns, and determining whether issues are related to inventory (missing data) or capabilities (features the system can't perform). Often, the most valuable insights come from discovering simple business improvements that don't require complex AI solutions. The instructor recommends working closely with subject matter experts to understand the real business needs behind technical requirements.

## Will there be opportunities to continue learning after the course ends?

Yes, students will still have access to Slack after the 6-week course concludes, and the instructor encourages continued questions. Additionally, students can join future cohorts of the course if they need more time to absorb the material. Many students find they benefit from going through the content multiple times as the field evolves.
