---
title: Week 5 - Office Hour 2
date: "2024-06-19"
cohort: 3
week: 5
session: 2
type: Office Hour
transcript: ../RAG_Cohort_3_-_Office_Hour_Week_5_Day_2.txt
description: Specialized indices, data engineering for AI, and blending traditional ML with LLMs
topics:
  - Metadata extraction strategies
  - Processing cost considerations
  - Recommendation systems and embeddings
  - Synthetic data for cold-start
  - Blending ML with LLMs
  - Data engineering tools for LLMs
  - Specialized indices vs embeddings
  - Cost calculation methodologies
---

# Week 5, Office Hour 2 (June 19)

Study Notes:

In this office hours session, I addressed questions about specialized indices, data engineering for AI applications, and strategies for blending traditional ML with LLMs. The discussion covered practical approaches to metadata extraction, cost considerations for processing large datasets, and techniques for improving recommendation systems with AI.

---

---

## How should I approach dynamically generating and handling metadata for documents?

When dealing with the need to extract new metadata from existing documents, the architectural approach depends largely on your current infrastructure. Most companies I work with already have some existing setup, so we're rarely building from scratch.

In essence, this is just like any ETL (Extract, Transform, Load) job where a process creates a new database artifact. The key question is: what makes backfilling this data challenging in your specific context? Is it the cost of reprocessing millions of documents? Is it the unpredictability of expenses?

For cost estimation, I recommend calculating the token volume of your data. We had a task to summarize a million conversations, and we made sure to calculate the expected input and output tokens. This allowed us to make informed decisions about model selection - for instance, we discovered that using open source models was only 8 times cheaper than using OpenAI's API.

"I was really disappointed to realize that the open source models are only 8 times cheaper. We're putting all this effort to save $60. And that was for a million conversations - it cost $60 to summarize a million conversations. These models are just so cheap now."

For specialized extraction tasks, consider using smaller, purpose-built models. At Stitch Fix, we built a suite of small models doing specific extractions. For example, we realized we were selling belts with pants that had no belt loops, so we created a simple computer vision model to detect belt loops. This approach was efficient and solved a specific business problem worth millions of dollars.

**_Key Takeaway:_** Calculate token volumes and costs before deciding on your extraction approach. Sometimes the cost difference between APIs and self-hosted models is smaller than expected, making the engineering effort to switch questionable. For specialized extractions, consider purpose-built models that solve specific business problems rather than trying to do everything with one large model.

## What are the challenges with extracting multiple attributes in a single API call?

When extracting multiple attributes from documents, be aware that prompts for some attributes can affect the extraction of other attributes. We found this when processing transcripts - when we asked for shorter action items, the summaries would also get shorter.

To address this, we split our extraction into separate jobs: one for action items and another for summary and memo generation. This separation gave us better control over each component. We made this approach cost-effective by leveraging prompt caching - the transcript only needed to be processed once, with multiple outputs generated from that single input.

**_Key Takeaway:_** Be cautious about extracting too many attributes in a single API call, as they can influence each other in unexpected ways. Consider splitting extractions into separate jobs with specific focuses, and use techniques like prompt caching to maintain cost efficiency.

## How should I approach recommendation systems with LLMs?

For recommendation systems like predicting product purchases, I wouldn't use an LLM directly in the recommendation system. Companies like Stitch Fix and YouTube use LLMs primarily to create better embeddings, not for the core recommendation logic.

The approach I'd recommend is building item embeddings using historical data, where the inputs might include product images, descriptions, user comments, and checkout rates. Similarly, user embeddings would incorporate their feedback, fit comments, and other behavioral signals.

One valuable application of LLMs is creating synthetic users to run simulations, particularly for addressing cold-start problems. When a new item appears, there's no transaction or impression data to train on. An LLM can simulate transaction data and returns, helping predict success rates for the first orders.

"At Stitch Fix we needed about 400 shipments of a single SKU before we had a good embedding for it. So our only job was: how do we get to a world where we either can simulate the SKUs or need less data?"

We addressed this by building a "Tinder for clothes" where users could swipe left or right on clothing items. This generated 6,000 labels much faster than waiting for 400 actual shipments, as users would label 30 items a day versus receiving only 5 items a month.

**_Key Takeaway:_** Rather than using LLMs directly for recommendations, use them to generate better embeddings and synthetic data to address cold-start problems. Consider creative ways to gather user preferences at scale, as the velocity of data collection is often the limiting factor in recommendation quality.

## How can I blend traditional ML with unstructured data from LLMs?

The most promising approach I've seen is using LLMs for synthetic data generation and feature engineering. The challenge with many recommendation systems is the low velocity of data - unlike Spotify or Netflix where users consume content quickly, physical product recommendations might take weeks to validate through purchases and returns.

Our focus at Stitch Fix was making each sample more efficient. Instead of building general-purpose computer vision models, we created specialized models for specific attributes (like detecting belt loops). These targeted models were more data-efficient and could directly drive business decisions (like upselling belts with pants that have belt loops).

The workflow we found effective was:

1. Use smaller, data-efficient models for specific extractions
2. Use these models to generate simulations and synthetic data
3. Feed this expanded dataset into larger, more powerful models

"Can we use LLMs for feature engineering and then use traditional models because they're gonna absorb the data faster? And then, once those cap out, how can we use the traditional models to create more data for the larger models to take in more capacity?"

This approach recognizes that different models have different data efficiency profiles, and leveraging their strengths in combination yields better results than trying to solve everything with a single approach.

**_Key Takeaway:_** Blend traditional ML with LLMs by using LLMs for feature engineering and synthetic data generation. Build specialized, data-efficient models for specific attributes, then use these to feed larger models. This creates a virtuous cycle where each type of model enhances the capabilities of the others.

## Are there good tools for data engineering in the LLM ecosystem?

The data engineering landscape for LLMs is still developing, with most early-stage companies using relatively simple approaches like "data to JSON" pipelines. One company worth looking at is Tensor Lake, which provides sophisticated data processing for tensors.

A critical area that's often overlooked is managing evaluation datasets. Many companies have inconsistent approaches where individual team members export data in ad-hoc ways:

"Almost every company I work with has datasets for evals, but they're all kind of like one guy wrote a SQL query to export things, saved it as a CSV file on their laptop and started working with it. And then they wrote this to Brain Trust, and that's what they're working on. But the other guy on a different team is using a different dataset."

This creates problems when metrics improve - does anyone trust the results? Was the test data recent or old? Did it cover multiple organizations or just one customer? Proper data engineering for evaluation is a substantial undertaking that requires careful planning and coordination across teams.

At Facebook, defining a new table for newsfeed views would involve a data engineer interviewing 20 teams, designing columns to support various query patterns, and ensuring everyone could write consistent SQL queries against the database. This level of rigor is often missing in LLM evaluation setups.

**_Key Takeaway:_** The data engineering ecosystem for LLMs is still maturing. Pay special attention to how you organize evaluation datasets, as inconsistent approaches lead to unreliable metrics. Consider investing in proper data engineering for your evaluation pipeline, similar to how established companies handle critical data infrastructure.

## What's your approach to topic modeling and specialized indices?

For topic modeling and specialized indices, we've been developing tools like Kora, which helps with topic extraction from documents. This approach is becoming increasingly valuable as managing knowledge bases becomes more complex.

The fundamental issue is that embeddings alone aren't sufficient for many complex queries. If someone asks "Who is the best basketball player under 25 years old from Europe?", embeddings might not find a direct answer unless that exact information exists in a paragraph somewhere.

This is why we need to build a portfolio of tools rather than relying solely on embeddings. For the basketball player example, you might need:

1. A structured player database with extracted attributes
2. Specialized extractors that pull out statements about people
3. Tools that can perform semantic search combined with structured filtering

"It's not that the tools are one-to-one with the retriever. It's actually gonna be the case that we probably have multiple tools hitting the same index."

This is similar to how command-line tools interact with a file system - you have commands like "list directories" and "view files," but also more specialized commands like "list files sorted by last modified" or "list files by editor." A smart model can learn to use these various tools rather than trying to build one mega-search tool that works for all cases.

**_Key Takeaway:_** Don't rely solely on embeddings for complex information retrieval. Build a portfolio of specialized tools that can work with your data in different ways. This approach is gaining traction in code generation and will likely become standard across other domains as well.

## Will reasoning models eliminate the need for specialized indices?

Even with advanced reasoning models that can perform multi-step thinking, I don't believe they'll eliminate the need for specialized indices and tools. Instead, the focus should be on exposing a wide range of tools that these models can leverage.

The key insight is that tools aren't necessarily one-to-one with retrievers. You might have multiple tools hitting the same index, similar to how command-line tools interact with a file system. For example, you might have tools for listing directories, viewing files, sorting by modification date, or filtering by editor.

"A smart enough model might just be able to reason about how to use all five tools rather than trying to build a mega search tool that will work in all cases."

This is the direction that code generation tools are taking - they're finding that embedding your codebase isn't the right approach. Instead, they're building portfolios of tools, and I believe this pattern will spread to other domains as well.

**_Key Takeaway:_** Even with advanced reasoning capabilities, models benefit from having access to specialized tools rather than trying to do everything through a single approach. The future lies in building portfolios of tools that models can intelligently select and combine, not in creating a single universal solution.

## How do you approach cost calculations for AI processing?

When calculating costs for AI processing, focus on understanding your token volumes. For any extraction or processing task, calculate the expected input and output tokens to make informed decisions about model selection.

We had a surprising discovery when comparing OpenAI's API to open source models for summarizing a million conversations. The open source approach was only 8 times cheaper, saving just $60 total. While it was 26 times faster, the absolute cost was so low that it wasn't worth the engineering effort to switch.

"I was gonna write a blog post on how to use open source models to do the data extraction. I was like, 'Oh, it's not worth writing the blog post because 8 times cheaper for $60? Well, unless I'm doing this a hundred times, I don't need to save $50.'"

These calculations help you make rational decisions about where to invest your engineering time. Sometimes the cost difference between approaches is so small that it's not worth optimizing further, especially when the absolute costs are already low.

**_Key Takeaway:_** Calculate token volumes and costs before investing in optimization. Modern AI models are often surprisingly affordable at scale, making some optimizations unnecessary. Focus your engineering efforts where they'll have meaningful impact rather than chasing small percentage improvements.

## How should I approach dynamically generating and handling metadata for documents?

When building metadata extraction systems that need to evolve over time, consider treating each extraction as a separate ETL (Extract, Transform, Load) job. This approach allows you to add new extraction tasks without redoing everything. Before implementing, calculate the token volume to estimate costs - you might find that even with millions of records, the cost is surprisingly manageable (often just tens of dollars). For specialized extractions, consider using smaller, focused models rather than trying to extract everything in a single pass, as this can provide better control over individual attributes.

## Is it worth using open source models for data extraction tasks?

It depends on your specific needs. In many cases, the cost difference between using open source models versus API models like GPT-4 may be smaller than expected - sometimes only 8x cheaper. For a job that costs $60 with an API model, saving $50 might not justify the engineering effort required to implement an open source solution. Always calculate the token volume and expected costs before making this decision, and consider factors beyond cost such as latency and maintenance requirements.

## How can I estimate the cost of running extraction jobs on large datasets?

Create a table that tracks input token counts for your documents and calculate the expected costs based on current API pricing. This simple exercise can provide valuable insights that inform your architecture decisions. For many extraction tasks, you might find that using models like GPT-4 Mini or similar smaller models is cost-effective enough, especially for straightforward extraction tasks.

## Should I extract multiple attributes in a single API call or separate them?

It's often better to separate extraction tasks into multiple focused API calls rather than trying to extract everything at once. When multiple attributes are extracted in a single prompt, changes to one attribute's extraction can unintentionally affect others. For example, requesting shorter action items might inadvertently make summaries shorter as well. Breaking these into separate jobs gives you better control, and techniques like prompt caching can help manage costs by avoiding redundant processing of the same input text.

## How can I blend traditional ML with LLMs for recommendation systems?

Rather than using LLMs directly in recommendation systems, consider using them to:

1. Generate better embeddings for items and users
2. Create synthetic data to help with cold-start problems
3. Simulate user behavior for new items that lack transaction data
4. Extract structured attributes that can feed into traditional recommendation models

At companies like Stitch Fix, the approach has been to use a cascade of models (vision, text, feedback, factorization) that build different scores, then blend these scores into a final probability-of-sale model.

## What are effective strategies for specialized indices versus general embeddings?

For complex queries like "Who is the best European basketball player under 25 years old?", general embeddings often fall short. Instead, consider:

1. Building structured data extractors that pull out specific attributes (age, nationality, sport)
2. Creating a portfolio of specialized tools rather than relying on a single embedding approach
3. Using different representations for different types of data
4. Exposing multiple tools that might access the same index in different ways

The trend is moving toward having multiple specialized tools rather than trying to build a single "mega search tool" that works for all cases.

## How are companies handling data engineering for LLM applications?

Data engineering remains a significant challenge. Many companies are still figuring out best practices for:

1. Creating and maintaining evaluation datasets
2. Building extraction pipelines that can be easily updated
3. Managing backfills when new attributes need to be extracted
4. Ensuring consistency across teams using the same data

For companies exploring this space, tools like Tensor Lake might be worth investigating, as they're designed for tensor-based data processing at scale.

## Will better reasoning models eliminate the need for specialized indices?

Not entirely. Even as models improve at reasoning, having a portfolio of specialized tools remains valuable. The approach is shifting toward giving models access to multiple tools that can retrieve and process data in different ways, rather than expecting a single model to handle everything. For example, instead of one mega-search tool, you might have tools for listing directories, viewing files, filtering by metadata, semantic search, and full-text search - all potentially accessing the same underlying data but in different ways.

---

