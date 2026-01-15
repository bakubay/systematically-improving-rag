---
title: Week 1 - Office Hours Summary
date: "2024-01-15"
cohort: 2
week: 1
session: 1
type: Office Hour Summary
description: Comprehensive Q&A covering DSpy, graph databases, ColBERT models, multimodal retrieval, and AI reasoning models
topics:
  - DSpy and Prompt Optimization
  - Graph Database Evaluation
  - ColBERT Models and Specialized Retrieval
  - Multimodal Retrieval
  - AI Reasoning Models
  - Medical RAG Applications
  - Vector Database Selection
  - Fine-tuning Embedding Models
  - Citation and Report Generation
---

# RAG Office Hours Q&A Summary - Week 1

---

## What is your take on DSpy? Should we use it?

Generally, I think DSpy allows you to do some kind of prompt optimization by synthetically creating a bunch of few-shot examples and then identifying which of these examples could improve the performance of your system.

Personally, I feel like most of the time you should be spending a lot of time actually just tweaking those prompts yourself. The most valuable part of looking at data, few-shots, and examples is you building an intuition of what customers are looking for and what mistakes the system is making.

Your product isn't just a prompt—it includes how you collect feedback, how you set expectations in the UI, how you think about data extraction, and how you represent chunks in the context. If you spend the time to look at how the model is making mistakes and what users are asking for, you'll make much more progress in improving the product as a whole.

DSpy is fine especially when you have very specific evaluations. For example, maybe you have a 35-class classification task where all you care about is accuracy. Then DSpy really works because you can figure out which of the 10 examples you need to maximize your accuracy.

But most of the time, that's not the case. If I'm building a model to extract sales insights from a transcript, I don't have a dataset of "here's all the sales insights." The real work might be extracting everything and hand-labeling some stuff. Because these tasks are very hard to hill-climb (when metrics aren't just classification accuracy), tools like DSpy don't work as well.

Another good use of DSpy is around using LLMs as judges. If you have a tonality or factuality evaluation you really care about, it makes sense to label a hundred examples yourself and then use prompt optimization tools to create your own judge that aligns with your grades.

## Is it useful to prompt language models with an understanding of structure and rationale for their actions?

Yes, absolutely. Understanding structure and rationale is critical because your product includes the ways you collect feedback, set expectations in the UI, perform data extraction, and represent chunks in the context.

It's not just about the prompt—it's a whole system. And if you can spend time looking at how the model makes mistakes and what users are asking for, you'll make much more progress in improving the product holistically.

When you build an intuition for what's happening, you can make smarter design decisions across the entire product experience.

## How do we introduce a concept of time and vector search to answer questions like "What's the latest news?" without needing to move to a graph database?

The answer is to use a SQL database. If you use something like Timescale or PostgreSQL, there are many ways of doing time filtering.

One specific thing to note is the difference between pgvector and pgvector-scale. Pgvector does not do exhaustive search, so there's a chance you don't recall all information because of how the database segments things. With pgvector-scale, it will exhaustively search every single row in your database if required. This small difference means a lot if you're trying to find very specific details.

The general idea is to use structured extraction to identify start and end dates, prompt your language model with an understanding of what those dates are, and then use filtering. You would do an embedding search plus a BETWEEN statement in your time query. This works pretty well.

## Is knowledge graph RAG production ready by now? Do you recommend it?

In my 10 years of doing data science and machine learning, I generally stay away from any kind of graph modeling. The reason is that every time I've seen a company go into this graph-based world, within 4-5 years they decide to move back to a PostgreSQL database.

There are several issues with graph databases:

1. They're really hard to learn - it's much easier to hire talent that knows PostgreSQL than graph databases.
1. Defining schemas in PostgreSQL and joins is well-defined, whereas in graph databases there's often too much debate and not enough best practices.
1. Most cases don't require more than one or two traversals of your graph.

When I was at Facebook, their graph was actually just a very large MySQL database. This makes me cautious about using graph databases unless you have expert users.

The only company I really believe could effectively use a graph database is LinkedIn, because they need to compute things like nearest neighbors up to three or five degrees away.

Even for cases like Microsoft's approach where you build a document graph with entities and relationships, I'd prefer to use fine-tuned embeddings. A graph can be defined as an adjacency matrix, and fine-tuning your embeddings can get you pretty close to the similarity definition that a graph could maintain.

I'd rather start with data and say, "There are certain kinds of queries that really need a graph structure" and let that justify the graph structure. Most technology needs to be justified by what the product needs to deliver rather than thinking about technology first.

## Would you recommend using Colbert models or other specialized retrieval approaches?

All of these models do similar things at their core. To decide what to use, we should start with a synthetic dataset to measure precision and recall. Then the real question becomes: do any of these interventions (graph RAG, Colbert models, embeddings, re-rankers) beat the baseline in terms of precision and recall?

It might be that graph for a certain problem is only 2% better, in which case it might not be worth the complexity. But if you found that, for parsing hospital records, graph RAG is 40% better on recall and precision, then it doesn't matter what I think—the data would speak for itself.

For Colbert specifically, it probably does very well for certain tasks. For example, statements like "I love coffee" and "I hate coffee" would be very similar in embedding space because embeddings don't fully understand negation. With a Colbert model, the cross-attention mechanism can figure out that these statements are different.

But you need to tell the model what's important in your context. Without enough tests to guide us, it's hard to know if these interventions work. Usually, it's hard to beat the baseline of embedding search with a good re-ranker. Colbert might do 4-5% better, but you need to justify that improvement against the added complexity.

## When working with legal documents that have multi-level outlines and reference sections from other documents, what approach would you recommend?

This could be done with a graph, but it could also be done with a simpler pointer system. When you load data, you can pull in other references. For example, in a construction project, whenever we pull up an image, we also pull up the paragraph above and below the image, augmenting the context.

We can do the same for legal documents—if it references another page or citation, we pull in that citation. Technically, this is a graph, but it's often easier to build this as a few LEFT JOINs in a PostgreSQL table.

When we pull in text chunks, if there are references, we just do another left join back to the original chunk. These systems tend to be much simpler to reason about than dealing with reference types in a graph. Usually, that level of complexity really needs to be earned when building bigger systems.

## Are we going to cover any fundamentals of how to systematically do generation?

In terms of generation, a lot comes down to prompting and using LLMs as judges, which we'll talk about in Week 3 when discussing product experience.

If you have specific aspects of generation you want to explore, it's mostly about ensuring formatting is correct and chain of thought is reasonable. The challenge is that you can't systematically improve generation primarily because generation evaluations are much more subjective.

If it's just formatting, that can be very explicit. But challenges with generation will mostly be addressed through LLM-as-judge approaches and different levels of regular expressions.

For example, we have an evaluation for summarization that simply measures what percentage shorter the summary is relative to the original input. These are very basic evaluations for summarization.

## What's your take on using RAG for report generation in response to requests for proposals?

The expert on report generation will talk in Week 4. Look out for a talk from Vantager, who does this for financial due diligence. Companies can give them existing reports, which they parse into a spec, and then when you upload new PDFs, it automatically generates a report for you.

There's a lot of economic value that can come from report generation, and it's probably more valuable than just doing generic question answering.

## What is your experience using reasoning models as the answer generator model?

Before there were specific reasoning models, I've been pushing everyone to at least have thinking tokens and a reasoning block in the output. This gives language models time to think and allows you to render in a way that minimizes perceived latency.

Now that O1 and DeepSeek are available, unless latency is a concern, I would try to use these reasoning models. O3 Mini is fairly affordable, and O1 is very affordable. You can render the product in a way that makes users feel it's faster.

DeepSeek's reasoning capability is one reason it stood out to people—they can actually see it think. For many practitioners, we've been asking language models to think step by step for quite a while.

## How do we set user expectations on the delay while using reasoning models?

The first UI tip is to stream out the thinking part of the model to the customer. Things will feel about 45% faster just because something is moving on the page.

The second approach, which DeepSeek does well, is to have a button called "Think harder" or "Reasoning." If users don't use it, they get the faster V3 model, but if they press reasoning, it switches to the R1 model. This both tells users you want the model to think (which they know will be slower) and, by rendering the thought tokens, improves the perceived latency.

## How should we handle multiple RAG sources with different levels of information?

When you have multiple RAG sources (like a calendar and a news site with more detailed event information), it can slow down the system when you want to use an LLM to act as a judge and provide a holistic answer.

One approach is to predict what types of questions are easy versus hard and route them effectively. Another approach is to improve the user experience by rendering sources before rendering the text. Show an animation like "I am thinking" and have document 1, 2, and 3 appear, then "I'm reading," and finally the answer.

Notion AI's UX does this well—it says "thinking about your question," "searching documents," animates the documents coming in, and then starts talking. The key is to keep the screen moving to make users believe something is happening.

Adding a loading screen that moves can make users feel the system is 30% faster, even if the actual processing time is the same.

## What strategies can help when there are negative consequences of "thinking too hard" with reasoning models?

One approach is to predict whether a question is easy or hard and decide when to turn on thinking. You could use a model like BERT to classify this.

If that's possible, you can make the decision to think on behalf of the user. The objective would be to maximize customer satisfaction while minimizing token costs.

Some companies like have their own proprietary model that tells you which is the best model to route to. You could have a model that's trained so that if you ask "what's 1+1," it sends that to a simpler model, but if you ask about reading a legal document, it routes to an R1 model.

For evaluation questions specifically, it really depends on the complexity. Some evaluations are simple yes/no decisions, while others involve complex reasoning like assigning the correct speaker to different comments in a transcript. You'll need to test with your specific use case.

## What advice would you give for introducing LLMs into a healthcare company that may not fully grasp their potential?

First, build a demo and let leadership see the results. Then, clearly identify what types of queries you won't attempt to answer, pre-loading all the risk discussions upfront.

Instead of saying "my model is 80% correct," say "I've identified the 20% of questions that don't work at all, but for the 80% of questions we can solve, the success rate is 99%."

Do the upfront work to know the failure modes and economically valuable opportunities, then present them clearly. Add guardrails to say what the LLM won't attempt to do. Much of this is about setting expectations for leadership.

## Are there open source re-ranking models that come close to Cohere's re-rankers in quality?

There are definitely good cross-encoders available, though some of the top models on leaderboards are 7 billion parameters, which may have high latency.

Modern BERT (a new BERT-based embedding model with about 8,000 token sequence length compared to the original 512) will likely lead to more powerful BERT-based re-rankers.

However, training your own re-ranker on your specific data will likely beat benchmark models. With just 6,000 examples from your own data, you can train a better embedding model and cross-encoder than what's publicly available, costing around $1.50 and 40 minutes on a laptop.

## Outside of personal experiments, what resources or mediums do you rely on to stay up to date on RAG?

Much of the content coming out is very hypey, and many research papers focus on public evaluations that don't mean as much as more fundamental work on data analysis, experimentation, and evaluation.

When reading papers, focus more on how they present results and think about experimentation rather than specific methodologies or implementations. The things that work well are often too maintenance-heavy or expensive for production use cases with millions of PDFs.

I like Anthropic's blog posts because they're fundamental—discussing how to think about error bars, clustering, and other approaches that everyone can use, not just researchers with 40,000 rows in a database.

Outside of that, a lot of information is in private Discords and Twitter. I'll have someone make a summary of the Discords with interesting "alpha" or insights.

## When working with documents with metadata, should search and retrieval methods change based on the level of metadata provided within the queries?

Yes, they should. For example, in a construction project, we found people really cared about who made the last edits on legal contracts or who sent particular information. The metadata was very important—queries like "which contracts did this person send us" become like SQL queries.

We learned that when answering questions about who's doing what, we should include their contact information. These are small details in improving a RAG system that create economic value.

Similarly, if you're building information that will be queried across time periods, you probably care about when documents were published and last crawled to determine relevance. A query like "what is the latest research in physics" might look at the past 6 months, while "what is new in AI" might only look at the past two weeks because it moves so quickly.

It comes down to analyzing the queries people are asking and figuring out what creates economic value.

## Do you know if Anthropic is working on an answer to O1 or R1 (reasoning models)?

Yes and no. If you use Claude's web app, it secretly has thinking tokens. Every time it says "pondering" or "thinking," it's actually outputting thinking tokens that you can't see.

If you ask Claude to replace the <antThinking> token with {anyThinking}, you'll start seeing those thinking tokens. You can request this token in the API as well.

The real question is whether Anthropic has thinking models that use RLHF, and I'm not fully sure about that. Their CTO has stated they don't do distillation, but there are mixed interpretations of what that means.

Claude 3.5 Sonnet is still impressive even without visible reasoning, including its vision capabilities. The bigger issue is that Anthropic is very concerned about safety and has questions about whether thinking tokens could lie to users or follow different policies.

## When working with unstructured data, mostly PDFs and drawings, how do you approach data labeling and what models do you use?

For unprocessed data, I look at companies like Llama Parse, Extend, and Reducto, which parse headers, bodies, tables, and figures so you can work with them separately.

For the most part, Claude Sonnet does a very good job—it's just a matter of how much data you need to process. For specific tasks like understanding figures, visual language models like Qwen via Ollama work well for single PDFs, though batch local processing is more challenging as tools like VLLM don't yet support these models.

## Why does this course favor LanceDB versus other vector databases?

The main reason is that I want everyone to experience running evaluations on not just embedding search but also full-text search. I want you to try hybrid search with or without a re-ranker.

With LanceDB, incorporating these approaches is just one extra line of code. You can do a search with different modes (lexical, vector, hybrid) and easily add a re-ranker. It's the simplest way to try all these combinations and discover what works best.

Additionally, LanceDB is backed by DuckDB, which means the same database that supports full-text search, semantic search, and re-rankers also supports SQL. If you want to analyze your queries with SQL, you can do that easily.

Another advantage is that LanceDB can be hosted on S3 and is easy to set up for large amounts of data.

## Which industry or application domain do you think is most difficult for LLMs?

It's hard to say definitively, but generally:

1. Tasks with complex images are difficult
1. Highly regulated industries like legal and healthcare contexts present challenges
1. Financial services, especially ratings agencies, face enormous regulatory hurdles

The fundamental challenge is that anything difficult for humans to collect data on will be hard for an LLM. It's about how much volume of data we have per industry and what kind of feedback loops exist.

If an LLM makes a decision that takes weeks to verify, it's going to be hard to improve. The timeline for regulatory approval in some industries (like ratings agencies) can be years, creating a massive barrier to implementing LLM-based solutions.

## Did you find a use case where re-rankers improve metrics?

Almost every case I've seen shows improvements with re-rankers, whether it's legal documents, question answering over books, or financial documents. A Cohere re-ranker typically improves performance by 6-12% while adding about 400-500ms of latency.

Companies like Cohere are building industry-specific rankers that support financial text, medical text, and code. They're working hard to beat OpenAI embeddings, and they generally succeed.

Re-rankers solve problems that embeddings miss, like distinguishing between "I love coffee" and "I hate coffee," which look similar in embedding space but are clearly different with cross-attention in a re-ranker.

## Can you share resources on how to create hybrid embeddings for PostgreSQL vector databases?

If you use a library called ParagraphDB, you can set up both sparse BM25 indices and dense embedding-based indices. This allows you to implement rank fusion.

Pinecone has good resources about this topic that I can share.

## For medical/healthcare administration, how can we get LLMs to be something that are trustworthy with serious decisions?

One approach is to use chain of thought models where we can read the reasoning to understand how the model arrived at a decision. Anthropic's concern may be that the chain of thought could be misleading.

There's likely a future where we can build UIs that let humans verify not only the decision but also the chain of thought behind it. Then we can train models so that even the reasoning aligns with user preferences. If a model gets the right answer but with faulty reasoning, that's where we'd provide feedback.

Another approach is to use ensembles—sample a suite of LLMs and use majority voting on decisions to establish confidence. I often train multiple smaller language models to grade things on a 0-1 scale, then use a classical ML model (like logistic regression) to make the final prediction. This helps with explainability because you can see which features influenced the prediction.

## For multimodal retrieval (text + images), what approaches work best?

For visual content like photographs, CLIP embeddings work well since they're inherently multimodal—they can represent both images and text in the same embedding space.

For instructional manuals with images, I'd pass the images to a language model and ask for a detailed summary of what the image shows, including all text in the image. Then embed that summary instead. This creates a text representation that points to the original image.

The approach has two steps:

1. Given an image, create a synthetic question that would retrieve it
1. Create a summary that would be retrieved for that question

For product marketing scenarios, CLIP embeddings can work well, but you need to define what "similar" means in your context. Does a red shirt match other red shirts, or just shirts of the same color? Should expensive silk shirts match inexpensive polyester versions?

This is why fine-tuning embedding models to understand your specific definition of similarity is important.

## How do you approach chunking very long documents (1,500-2,000 pages)?

If you have extremely long documents, I'd first try a page-level approach to determine if answers typically exist on a single page or span multiple pages.

One compelling approach is from a paper called RAPTOR. After chunking documents, they recluster the chunks. You embed every page, run a clustering model, and identify concepts that span multiple pages. Then summarize those clusters and use the summaries for retrieval—if the summary is retrieved, you can include all related pages in the context.

For metadata, look at your queries to determine what matters. If users frequently ask about publication dates or document authors, those should be included. The needs will become obvious as you analyze user queries—you'll realize what's important and what creates economic value.

Generally, if you can reorganize text chunks by clustering and bringing related information together, that's very valuable. For example, with tax law documents where laws are on pages 1-30 and exemptions on page 50, you could process the document once to place exemptions directly below the relevant laws. This preprocessing step might cost $10 of LLM calls per document, but for legal documents that might not change for years, it's worth the investment.

## Do you have a go-to approach for visual document image embeddings (like quarterly reports with tables, images, graphs)?

For visual documents like quarterly reports full of tables and images:

1. Dockling is a free library that works quite well, though it might take about 11 seconds per PDF
1. Claude Sonnet also works well for extraction
1. Reducto, Llama Parse, and other commercial tools can be worth the cost to save time
1. For multilingual content, VDR2B-Multi v1 handles multiple languages well

There's an ongoing discussion about using Gemini 2 (with its million-token context window) to convert documents to markdown and extract all the information. This approach is becoming more viable as models improve, potentially reducing the engineering needed for preprocessing.

Recent testing shows Reducto still has higher accuracy (0.9 ± 0.1) compared to Gemini (0.84 ± 0.16), but the gap is narrowing. The reason Reducto performs so well is that they have people manually labeling thousands of PDFs to train their models.

## Why at Meta did you prefer SQL databases over graph databases?

Graph databases are useful when you need complex traversals, like finding all of Jason's followers who follow a specific account, then finding what they like, and sorting by aggregated likes per product.

However, what we found is that most use cases are actually simpler—often just requiring 2-3 left joins in SQL rather than complex graph traversals. From a skills perspective, it's easier to hire people who know SQL well than to find graph database experts.

At scale, graphs are also hard to manage. Around 2017-2018, only LinkedIn had a true graph database because they needed to compute 3rd-degree friendships very quickly. For most companies, SQL databases offer better performance, easier maintenance, and more familiar tooling.

Over a 12-year career, we kept trying different technologies (Hadoop, Spark, etc.) but always ended up returning to SQL. The pattern is consistent across many organizations.

## What have you learned about prompt caching?

Prompt caching is a technique where language models can avoid reprocessing the beginning of prompts that are often identical.

Different providers handle this differently:

- Anthropic caches prompts for 5 minutes; if you make the same request within that time, the entire message is cached
- OpenAI figures out the optimal prefix to cache automatically

This is valuable because it can save significant processing time and costs, especially when you have many few-shot examples or large system prompts. If you have 50+ examples in your prompt, caching can dramatically improve performance.

For models like Claude on Bedrock, prompt caching wasn't available a few months ago but is likely coming soon. It's the kind of feature that rolls out gradually across providers.

## For visual document image processing, what's the state of the art?

There's a recent discussion on Hacker News about using Gemini 2 (with its million-token context window) to process documents and convert them to markdown, extracting tables, layout information, and text.

The engineering needed for document pre-processing is getting simpler as these models improve. Recent tests show Reducto still has higher accuracy (0.9 ± 0.1) compared to Gemini (0.84 ± 0.16), but the gap is narrowing.

Reducto's performance comes from having people manually label thousands of PDFs, then training models on that high-quality data. This reinforces the point that with 6,000-10,000 high-quality labels from your own data, you can train models that outperform even the biggest general models on your specific tasks.

## How does Brain Trust work with the notebooks in this course?

Brain Trust just saves the results that your laptop is running locally. It's not executing anything or using a better database—it's more like an observability tool (similar to Datadog).

When we run the notebooks, everything is running on your laptop in LanceDB. The only thing Brain Trust sees is row IDs and scores. Think of it as a powerful UI over a database that's saving your logs, not as a computation service.

## What's the difference between bi-encoders and cross-encoders?

A bi-encoder converts all documents into numbers (embeddings) first, and then the assumption is that when we compare those numbers, documents that look similar are similar. Because we pre-compute everything, we can search very quickly.

A cross-encoder doesn't compare numbers—it compares the actual sentences. This approach can't compare a million documents with a million other documents (too expensive), so instead it takes one question and 50 documents and compares each one individually. That's the "cross" part of cross-encoder.

The advantage of cross-encoders is that a language model can compare words like "love" and "hate" in "I love coffee" and "I hate coffee" and understand they're different, whereas bi-encoders just have lists of numbers that don't capture this nuance.

We'll cover this topic more deeply in Week 2, but the key takeaway is that bi-encoders are faster but less accurate, while cross-encoders are slower but better at understanding semantic distinctions.

## What's the process for fine-tuning embedding models?

In Week 2, we'll cover this topic extensively. The overall message is that:

1. It's probably a bad idea to train your own language model
1. It's a very good idea to train your own embedding model

Fine-tuning embedding models is much less resource-intensive—it typically costs around $1.50 and takes about 40 minutes on a laptop. With just 6,000 examples from your domain, you can train embedding models and cross-encoders that outperform general-purpose models on your specific tasks.

This is especially useful when you need embeddings to understand domain-specific concepts or when you're trying to define what "similar" means in your particular context (e.g., product recommendations where price range matters).

## How do you understand metrics like precision and recall in one-to-one answer scenarios?

For questions with exactly one correct answer, these metrics behave somewhat differently. Recall will be either 0% or 100% depending on whether K is large enough to include the correct answer.

For example, if we want to retrieve exactly one document and there's only one correct answer, precision could be either 0% or 100%, and the same for recall.

The metrics become more meaningful when:

1. There are multiple relevant documents
1. We're analyzing trends across many queries
1. We're comparing different retrieval methods

Even with one-to-one mappings, MRR (Mean Reciprocal Rank) is still useful to see where the correct answer appears in your results.

What really matters isn't the absolute number but whether we can move these metrics in a positive direction with our interventions. It's like weighing yourself—the absolute number may vary by scale, but if you've gained two pounds, you've definitely gained two pounds.

---

