---
title: Week 3 - Office Hours Summary
date: "2024-01-29"
cohort: 2
week: 3
session: 1
type: Office Hour Summary
description: Negative feedback handling, RAG vs recommendations, impact measurement, and advanced evaluation strategies
topics:
  - LLM Labeler Reliability
  - Inventory vs Capabilities Analysis
  - RAG vs Recommendation Systems
  - Impact and Economic Value
  - Feedback Lifecycle Management
  - Knowledge Base Topology
  - Classical vs Agent-based RAG
  - Citation and UX Best Practices
  - Data Labeling Evolution
---

# RAG Office Hours Q&A Summary - Week 3

---

## When gathering negative feedback from documents not being found, how do we use and validate the reliability of an LLM labeler?

When it comes to getting negative feedback that documents were not found, I'd assume we're running into issues around low recall. This might manifest as low re-ranker scores or low cosine similarities with embedding models.

What I would do is identify whether the language model itself can identify if documents are irrelevant. We'll need some manual labeling step. With our clients, we generally find questions that emit flags - maybe you tell a language model to always say "we couldn't find relevant documents" when it can't find anything.

You can then label that as traffic is being processed. We might sample 1% of traffic, and some percentage of that might have that message. That's one level of detection.

The second level would be building a streamlit UI where we can manually label whether we agree with the irrelevancy assessment. The hard task is determining if any of 10 text chunks are relevant to a question. The easier task is determining if a single text chunk is relevant to a question. That's easy for a human to do and also pretty easy to prompt for.

This approach helps ensure the judgment we're using is aligned with human preferences. There's obviously a big difference between 60% alignment and 95% alignment, but this is a good start for figuring out whether low relevancy is causing the lack of documents.

## In the segmentation topic, we talked about inventories and capabilities. Is it realistic to do this automatically or is it something we have to do manually?

I would generally recommend doing this manually, because it's so important for what your business is trying to do that you need to actually think about these problems.

We've delegated so much thinking to language models. If we just think a bit harder about our problem, we often find very specific issues.

For example, with a client doing tax law resolution, the first 20 pages were massive articles, and then pages 30-40 were the exemptions to those articles. We spent maybe $20 of LLM calls to rewrite the documents so that the exemptions were close to the relevant articles. Now we have a single page/chunk covering an article and all its exemptions, with references to related articles.

We run that job once a week when new tax laws come in. Since we only have about 45 documents we really care about, I'd rather spend the money upfront to get the process right rather than waste customer time requerying data.

The real goal isn't to get a number right - it's to figure out what to do next. The AI can't tell us that. Your job isn't to automate this process; you're being paid to figure out what the next intervention should be.

## Can you elaborate on your view on RAG versus recommendations? How would you approach the use case of friend suggestions?

When you build a recommendation system, there are several steps:

1. **Sourcing** - What inventory can I show my customer? In the friends case, this would be all users on the platform.
1. **Query** - Either your user ID or a question embedding.
1. **Scoring** - For simple RAG, this is cosine distance of embeddings and maybe re-ranker distance. For friends, it might include mutual connections, location, etc.
1. **Filtering** - In RAG this might be top 10 results or embeddings greater than a threshold. For friends, filters might include having at least 3 mutual friends, same zip code, etc.
1. **Rendering** the results

When users take actions (adding/removing friends, opening files, deleting citations), you collect feedback to improve your system. When a language model sees 10 documents but only cites 3, those 3 are likely more relevant. You can use that signal to improve your re-ranker or embedding model.

If the user deletes one of those citations, you have a triplet: documents the model thinks are important, plus a negative example. When training, these need to be adjusted accordingly.

It's like different levels of signals in e-commerce: liking a product is a weaker signal than adding it to cart, which is weaker than buying it, which is different from buying and returning it. That's your portfolio of data collected over time.

## In the 4th lecture you mentioned the formula expected value as impact times the number of queries, times the probability of success. Can you explain more what you mean by impact?

Impact here is a general term that the Facebook folks like to use. I generally think of impact as economic value.

In the construction example I often mention, about 70% of questions were simple things like "Where do I show up today?" or "How thick is the drywall?" These weren't individually valuable.

But we also found a set of questions that were super valuable - around scheduling and figuring out if contracts were signed. These were only about 10% of queries but extremely important. When we asked our clients, they said that preventing one missed contract could save $60,000 in delays.

This told us these queries had high economic value, even though they were less frequent. So we invested resources in making sure we could query contracts and schedules to answer that segment.

Impact is about how valuable a problem is and how much it's worth, rather than just how frequently it occurs. Every metric you track should enable you to take follow-up action afterward - it's not just about knowing the number.

## What is the lifecycle of feedback? If we improve the UI, old labels might be out of date and new data will be labeled differently. What is good to keep versus letting go?

This depends on how much data we have and the blend of that data. If we had a million labels before changing the UI, I'd push hard to keep the new UI somewhat similar to ensure the data we collect remains comparable.

If we're really changing things dramatically, there are modeling techniques to control for this. You might pre-train on the old data for your embedding model, then use that as a foundation for training a newer model. You can also control for the source of data in your modeling.

You can have different evaluations to verify performance on the old data versus the new data, then choose how to weight those scores. Generally, I'd try to keep things as generic as possible - you don't want a dataset that's too specific and won't generalize.

For embedding models specifically, I'd typically include everything, as more data is generally better.

## Is it interesting to collect feedback not only as thumbs up or thumbs down, but let users explain in text what is wrong with the answer?

Yes and no. Thumbs up/down is super useful, and it would be hard to convince me not to use these binary labels. Going to a 5-star scale creates issues where you don't know if users consider 3 or 4 stars to be "average."

With free text feedback, you'll face two issues:

1. Probably less than 10% of users will give a text response. If only 1% of users leave feedback at all, and only 10% of those leave text, you get very little text data, and you don't know how biased that sample is.
1. You likely won't be able to read all the free text, so you'll build clustering models to analyze the feedback - in which case, you might as well just have 5 buttons for the most common issues (too slow, answer too long, format incorrect, etc.).

It's about maximizing data per label. Having buttons for common issues will get you more useful data than open text fields.

That said, free text can help you figure out what those buttons should be. For enterprise situations, we include the default buttons plus free text, and when users enter text, we post it to Slack where the team and customer can see it. This shows users their feedback is seen, making them more likely to provide it.

But think about how often you've thumbs-downed a ChatGPT response, let alone written why. Most users simply won't take the time.

## How do you handle recall when dealing with large knowledge bases with a messy topology (near-identical documents, overlapping content, hub pages, etc.)?

This is challenging, especially with something like a large software product knowledge base (44,000+ documents) where many people have been adding content, creating overlap and interstitial hub pages.

One approach is to build a system where if you retrieve a subset of pages, you can reference the connections. Similar to how e-commerce sites show "people who viewed this also viewed" suggestions.

As context windows get larger, you could implement a system where if you pull in a page that references other documents, you traverse one level and bring in those referenced documents too.

You could also do clustering and summarization. If your repository is very valuable, maybe it costs 10 cents to process a page, but with a budget of 50 cents per query, you could chunk everything, cluster similar content, and then summarize the clusters. This essentially rewrites the knowledge base in a less duplicated way.

The more fundamental question is about how you define relevance. Do you have a policy document on what makes a document relevant? Google has a detailed document on what makes a good search result. You need to establish and document your criteria so everyone has the same understanding.

## Have you compared the effectiveness of classical and agent-based RAG systems with capabilities offered by models like Gemini Flashlight for real projects?

I prefer not to think about systems as "classical" versus "agent-based" RAG systems. Most RAG systems are essentially function calling in a for-loop or while-loop.

The goal is to provide the language model with two things:

1. Good functions
1. Good indices for each function to query that are well-defined

You want to ensure each index has good recall, each function is useful for the system, and you have good prompts to help the model choose the right function.

For real projects, it's not just about question answering but also about tool rendering. Some tool calls define UX elements - like a fitness company chatbot that renders modals for booking calendar events and following up with payment links. This becomes the economically valuable work - not just answering questions but helping the company make money.

## What's the moat for companies building RAG systems when so much is being open-sourced?

I generally think the moat is your labeled data. There probably isn't much difference between various newsfeed algorithms, but the moat is the inventory - the content that's already out there.

If you have relationships in a specific sector like construction and can be the first to build connectors and bring in that data, that's an easy moat (though not as defensible).

After that, it's about analyzing that data to understand what questions people are asking and building specialized tools for those needs. This is software that LLMs won't replace anytime soon.

Then it's understanding what relevance actually means - fine-tuning re-ranking models, training custom embedding models. These are aspects that LLM companies won't compete against.

The moat becomes your data - both relevancy data and access to the content itself - plus your understanding of customer needs and workflows. The more you understand what customers are truly trying to do (beyond just answering questions about PDFs), the better your product will be.

## In the UX lectures, you mentioned that explicit copy instead of just thumbs up/down can impact whether people give feedback. Have you observed an impact based on what the copy actually says?

Absolutely. At Zapier, they asked "How did we do?" which was a very vague question that didn't get much feedback.

When we A/B tested copy, the version that got 5x more feedback was "Did we answer your question?" This was much more specific and focused on the core value proposition, not about latency or formatting. If users said no, we'd follow up with "Do you have any other feedback? Was it too slow? Was the formatting wrong?" since we knew those were common failure modes.

This not only got more feedback but also correlated better with customer satisfaction. The previous vague question made it hard to identify what was a good or bad answer - some might say we did poorly because we answered correctly but too slowly.

At Raycast, our copy now is "Did we take the correct actions?" since we're showing function calls like "Set a 1-hour lunch on my calendar and update my Slack status." We show users the sequence of function calls and ask if we're taking the correct actions.

The key is that every metric you track should lead to a follow-up action. It's not just about knowing the number.

## How can we extract value from template/pre-filled questions in chatbots?

For a situation like a lawn care subscription company's chatbot where 70% of conversations start with template questions, I'd be curious to understand what the follow-up questions look like. This helps determine if we could create complete guides for common paths.

If people start with a certain template question, do their follow-ups cluster in a specific domain? This can help you understand if your example questions are actually helpful or if you should be writing better content to answer these questions more comprehensively.

One approach is to use a language model to summarize conversations, identifying what topics come after the template questions. This gives you insight into actual user intents that might be hidden behind that initial templated interaction.

You should analyze which topics are economically important by looking at metrics like thumbs up/down data. For instance, we found that many negative ratings come from users who want to talk to a real person but can't easily figure out how to do that.

It's also valuable to analyze what products you should recommend based on question patterns. If you're seeing thumbs-down ratings, analyze whether it's because you don't have the right content in your knowledge base, or if there are capabilities you're missing. Often, the solution might be as simple as hiring someone to write targeted content for frequently asked questions.

## How do you handle business knowledge translation (like acronyms) in RAG?

When you have documents that spell everything out formally but users want to query using acronyms (like "What's the deal with ABC?"), I'd generally just put this translation knowledge in the prompt unless you have an enormous number of acronyms.

If you have fewer than 80 acronyms or terms that need translation, putting them directly in the prompt is the simplest and most effective approach. You only need to explore more complex approaches when you have evidence that this simple solution isn't working.

You can also create synthetic data to test how well your system handles these acronym queries, which is usually straightforward to generate.

## What are the best practices for chunking in RAG systems?

The general advice from companies like OpenAI and Anthropic is to start with around 800 tokens with 50% overlap using a sliding window approach. That should be enough to get you started.

After that initial setup, the real improvements come from understanding what kinds of questions are being asked and what the answers look like. If most questions can be answered by a single document, focus on improving document search and relevancy rather than chunking. If answers typically come from small paragraphs across many documents, then experiment more with chunking.

We've spent weeks doing chunking experiments and often haven't seen significant improvements. It's rarely the case that changing from 500 to 800 tokens suddenly makes everything work better - that would suggest most answers require just a few more sentences in the same document, which is usually not the issue.

What's been more helpful is looking at the questions and working backward: What are people trying to do, and what design assumptions can I make to better serve that? For instance, if users are searching for blueprints, maybe summarizing blueprints first would help, or perhaps including text above and below the blueprint, or even applying OCR and building a bounding box model to count rooms.

Solve specific problems where you can justify that "this is 20% of our questions" - if you make those 20% twice as good, you've improved overall performance by 8%, which is meaningful.

## Are XML tags still best practice for prompting models?

Yes, we've learned that even the GPT-4 models now perform better with XML formatting. We have internal evaluations from Zenbase showing that XML is good not just for Anthropic models but also for ChatGPT models.

The second thing we've found is that you generally want to have all the long context information at the beginning of the prompt - first the goal, then all the documents, with the actual questions at the bottom.

Claude's prompt rewriter has been very helpful for showing how to write better prompts. I almost always run my prompts through it first before setting up evaluation suites, as it's a free way to get useful feedback.

## How do you handle tokenization concerns with things like wallet addresses?

When dealing with data that contains wallet addresses (which are 52 characters of what looks like nonsense), I'd worry less about the tokenization itself and focus more on validation.

For example, in situations where we use UUIDs, we reference content with a UUID, and we tell the model to cite everything. We then have an allowlist of valid UUIDs from our data, and we check that any UUID the model outputs exists in that allowlist.

So if you have a use case where users ask about wallet IDs, focus on making sure the model can only reference valid wallet IDs from your dataset rather than worrying about how they're tokenized.

These days, models aren't typically off by a few characters - they'll either get it right or completely make up new identifiers. Having logical checks in your code is more important than the tokenization strategy.

You can also generate synthetic test data where you know which wallet addresses should appear in the answers and ensure there are no hallucinations.

## Should we transform content from narrative format to Q&A format for better retrieval?

Yes, massively. This can be very beneficial, especially for a question-answering chatbot.

It's already an assumption to think that everything is going to be in the form of a question. For some assistants, it might be more about conversations or past memories. If you know your use case is primarily Q&A, then extracting question-answer pairs from your documents is valuable.

You can build a system where when you embed a question, you retrieve the embedding of similar questions, but pull in both the question and its answer. This makes sense if your use cases are mostly Q&A-based rather than narrative requests like "tell me a story."

One of the big assumptions in RAG is that the embedding of a question is similar to the embedding of a relevant document, which is actually a massive assumption that doesn't always hold true.

To prevent retrieving too many similar question-answer pairs (which could be redundant when getting top-K results), consider doing clustering. You could extract 10 questions per document, then cluster similar questions together and rewrite them to create a more concise, focused knowledge base.

## Can you recommend any open source libraries or tools for streaming UIs and interstitials?

I can't necessarily recommend a specific library too strongly because most companies I've worked with have built these themselves. However, if you're in the Python world, using something like FastAPI and server-side events (SSE) API is probably the simplest approach. In the slides, we give an example of what this looks like - you're basically using the yield keyword from Python generators to emit events.

If you're using JavaScript and part of the Vercel/React ecosystem, I think Vercel's AI library does a great job of handling structured streaming. Other libraries like LangChain, LlamaIndex, and Instructor also support partial streaming where you can send incomplete JSON to a frontend, which can then rerender it.

For interstitials, I've been impressed with what Ankur from BrainTrust has done in their playground. I've reached out to him to ask about recommendations for this.

With these tools, the implementation is fairly straightforward. The bigger challenge is often designing a UX that communicates progress effectively. Notion's approach is a good example - when you enter a search query, it shows "making a search request," rewrites the request, then renders documents one by one, and finally shows steps like "carefully reading documents," "thinking," and "formulating an answer." This is really just buying time while showing progress, but it dramatically improves the perceived responsiveness.

## Why aren't data labeling companies a bigger focus in current AI discussions?

This is an interesting historical shift. Around 2018, data labeling was a huge focus because the biggest models were vision models that required massive amounts of labeled data. Vision models aren't very data-efficient - training ImageNet required labeling a million JPEGs. Companies like Scale AI won by excelling at tasks like self-driving car LiDAR labeling.

As we've moved to LLMs, two things have changed:

1. The big winners (like Scale AI) have already established themselves and now focus on large contracts. Smaller players either grew or struggled to find viable business models on smaller contracts.
1. LLMs are much more data-efficient.

The data efficiency of modern LLMs is remarkable. You're better off having 1,000 very high-quality labels to fine-tune a model than 10,000 mediocre labels. This means that instead of outsourcing labeling work, it often makes more sense to have subject matter experts do a one-month project to create the data you need.

We're so sample-efficient now that offshore labeling doesn't make economic sense for many use cases, especially when LLMs have been shown to match or exceed the quality of offshore labeling for many tasks. If you have specific legal workflows, you're better off asking the lawyer on your team to do the labeling.

The real challenge now is: how do you find people who are smarter than GPT-4 to label data to train the next generation of models? That hiring problem is different from the traditional labeling company approach.

## How do you see re-rankers evolving beyond just measuring relevancy?

Right now, most RAG systems only rank based on relevancy between a query and a document. But I think re-rankers will soon incorporate much more side information, similar to what we see in e-commerce recommendation systems.

In e-commerce, we have additional rankers for things like price sensitivity, seasonality, and product age to determine if customers prefer trendy or timeless items. This hasn't really happened in the RAG world yet.

As AI systems accumulate multiple years of memories about users, figuring out what information to put in context will become much more interesting. Re-rankers won't just measure string similarity between a question and document - they'll likely incorporate user features, environmental features, and contextual information to determine relevance.

For example:

- Security constraints (only searching documents you have access to)
- Time/recency components for memories
- Domain authority when sources disagree
- User preferences based on past interactions

Even systems like Deep Research might evolve to pull from sources you tend to agree with, or deliberately include sources that challenge your viewpoint. These personalized relevancy signals could dramatically improve RAG systems beyond simple semantic matching.

---

## Key Takeaways and Additional Resources

### Key Takeaways:

- Data quality is becoming more important than ever - good models make data quality the differentiator
- When collecting feedback, be specific with your questions to increase response rates
- Focus on economically valuable workflows, not just answering questions
- For messy knowledge bases, consider clustering and summarization approaches
- The moat for RAG companies is proprietary data and domain expertise, not algorithms
- Binary feedback (thumbs up/down) generally gets more responses than free text
- Always have a clear next action from any metric you collect
- Focus on impact (economic value) rather than just query volume

### Additional Resources:

- Google Search Relevancy document/policy is a good reference for defining relevance
- RAPTOR paper for document summarization approaches
- Week 3-4 content in the course covers more on these topics
- For prompt rewriting, Claude's prompt rewriter is highly recommended
- When dealing with streaming UIs and latencies, Notion's approach of showing steps visually is a good reference
- For friends example in recommendation systems, consider platforms like Facebook's friend recommendation system as reference implementations

_Note: I'll continue to add resources and notes from future office hours sessions_

---

