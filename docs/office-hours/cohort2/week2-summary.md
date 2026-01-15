---
title: Week 2 - Office Hours Summary
date: "2024-01-22"
cohort: 2
week: 2
session: 1
type: Office Hour Summary
description: Advanced RAG evaluation strategies, fine-tuning decisions, synthetic data challenges, and multi-agent approaches
topics:
  - Document Parsing Evaluation
  - Fine-tuning vs Prompting
  - Synthetic Data Distribution Mismatch
  - Embedding vs Re-ranker Fine-tuning
  - Collaborative Filtering Cold Start
  - Multimodal Content Processing
  - Function Calling Workflows
  - Long Context vs RAG
  - Research Paper Evaluation
---

# RAG Office Hours Q&A Summary - Week 2

---

## How would you evaluate the effect of different parsing strategies in RAG, notably on documents with weird layouts, tables, and charts?

For documents with complex layouts like tables and charts, there are multiple levels of evaluation:

First, you need to evaluate OCR accuracy - checking whether text is being parsed correctly (e.g., is a 0 being parsed as an 8?). Then there's the bounding box detection problem - checking if tables are fully recognized as single bounding boxes using metrics like intersection over union.

It's generally safer to evaluate OCR/parsing and retrieval separately because parsing errors can be hard to trace back when they're part of the full RAG pipeline. If you parse an 8 as a 0 and generate synthetic data from that, you won't be able to capture that error in your evaluations.

I've leaned on parsing vendors because they're the most incentivized to have good and accurate labels. This lets me focus on retrieval, which is what will create the most value for my specific use case. While there are other businesses focused on PDF processing, no one will focus specifically on your ability to do retrieval well with your data.

## When does it make sense to create a targeted summary for an application's objective versus fine-tuning embedding models?

This depends on whether you have data to fine-tune and what your embedding should capture. If you're writing the summary yourself, you're essentially making an assumption about what the embedding should look like.

For example, with image embeddings, maybe the most common questions aren't just about what's in the photo but about the cinematic mood. In that case, it might make sense to have a language model create a summary describing the mood because I want to be able to search for "atmospheric" or "dark and gloomy" rather than just "trees in a forest."

However, if you have actual user interaction data, it's better to use that data to tell us what's similar rather than creating assumptions. For example, with blueprint images, an image model might just say "this is a blueprint," but what I specifically did was extract information like number of rooms, bathrooms, sizes, and addresses - information that would be harder for a CLIP embedding to capture.

In general, I'd much rather use the data my app generates than hard-code these summaries, but summaries can be useful when you need to extract specific structured information that embedding models might miss.

## What are the recommended approaches for evaluating RAG for single documents, like report generation for a proposal?

When working with a single PDF document that might be varying in length (from 10 to 400 pages), semantic chunking can be valuable to separate paragraphs based on their semantic meaning rather than just token-based chunking. This is especially important when requirements for different disciplines might be found in different sections (e.g., structural requirements mentioned within architectural requirements).

One approach is to generate synthetic questions per paragraph, asking "What are the requirements being mentioned in this paragraph?" rather than "What question can you ask from this paragraph?" This helps identify the key information.

For retrieval, you can also inject a summary of the page that a paragraph was extracted from and embed them together. This way, when retrieving, you have both the specific chunk and context about where it comes from, which can improve recall.

Whether adding summaries improves recall is an empirical question - if it increases recall by 1%, it might not be worth the extra LLM calls, but if it improves recall by 6-8%, it could be worth investigating further.

## Could you distill key reasons when someone should consider fine-tuning open source embedding models over proprietary models?

If you have 6,000-10,000 examples of question-document relevancy pairs, you can likely outperform closed-source models with a fine-tuned model. This is because your tasks can be much more specific than what general models are optimized for.

It can also be more valuable if you need to embed massive datasets at scale. By spinning up your own GPUs, you can process much more text per second at a lower cost. For example, embedding 20GB of text data might take only 15 minutes and cost around $20, whereas using OpenAI APIs would be more expensive and much slower.

The main downside is the need to maintain your inference server, which adds complexity. It's less about whether the model will perform well and more about whether you have the time and resources to maintain the infrastructure.

## Is there a reason to ever fine-tune the LLM rather than or in combination with fine-tuning the retriever model?

I'm pretty open to businesses fine-tuning their retrieval models because companies like OpenAI or Anthropic aren't primarily focused on making retrieval better - they're not launching new embedding models daily. Companies like Cohere, on the other hand, are actually thinking about retrieval.

If you spend effort fine-tuning an LLM, you need to consider inference, CUDA drivers, and whether your fine-tuned model will be competitive when the original model provider releases a new version in a few months.

It's generally very costly to fine-tune language models, and you often don't get much benefit. However, if there are specific reasons - like tonality, personalization, or access to proprietary data - it might make sense. But for a team of 4-5 people, it's probably not a good idea to spend effort maintaining that kind of infrastructure.

In contrast, fine-tuning embedding models can be done on a laptop, run on cloud instances, and be cost-effective. For most teams, the maintenance cost of running your own LLM is just too high to justify.

## One weakness of RAG is difficulty in detecting relationships between concepts because the retriever model isn't aware of how concepts relate to each other. Should we fine-tune the LLM for this?

Before considering fine-tuning the language model, I would ask: How much can we put into few-shot examples in the prompt? Can we come up with good chain-of-thought examples that describe these relationships? Can we provide a glossary?

The maintenance cost of running an LLM is so high that it's worth really trying to squeeze out as much as possible through prompt engineering, longer system prompts, more few-shot examples, and prompt caching before considering fine-tuning.

For example, Bloomberg spent millions on their own model, and within 5-6 months, GPT-4 was better. Instead of fine-tuning, consider using RAG to retrieve relationship information first, put that in the context, and then add the actual question. This is more maintainable and adaptable as new models are released.

## What is the main failure modes (like distribution mismatch or biases) that you've seen when relying on synthetic data for retrieval fine-tuning?

The biggest issue is mismatch between user questions in reality versus in the synthetic data. Once you have synthetic data for fine-tuning retrieval models, it's hard to imagine a case where creating more data for your use case would make the model worse.

What's more important is figuring out how to intelligently incorporate real-world examples from users into the few-shot examples for synthetic data generation, making it a more diverse process. You can check this by:

1. Looking at the variance of embeddings against each other to see if they're too similar
1. Checking general statistics like character count variance in questions
1. Ensuring the synthetic data matches user data characteristics

For example, if your customer questions typically have around 30 characters but your synthetic data averages 90 characters because the language model is too verbose, that's a simple distribution mismatch to fix.

## Can you share the intuition for the difference between a fine-tuned embedding model and a fine-tuned re-ranker?

The embedding model allows you to do search over a large number of documents - given an embedding model, you might retrieve the top 100 text chunks. The re-ranker model then takes these 100 chunks and finds the best 25.

We generally want to use both, and the dataset to train these models is actually the same dataset. If you can only afford to fine-tune one, you might choose based on where your bottleneck is:

1. Is recall at 100 already good (95%) but recall at 10 is poor (50%)? Then focus on the re-ranker.
1. Are you missing relevant documents even in your top 100 results? Then focus on the embedding model.

The key insight is that by having metrics on both stages, you can identify where to focus your improvement efforts.

## Do we need more data to fine-tune re-rankers than bi-encoders?

It depends on the model, but generally, Cohere has done a good job of being data-efficient for producing embedding models. The amount of data needed may vary by model and task.

## For collaborative filtering models, how do you address the cold start problem (new users/items the model hasn't seen) without retraining the model?

There are multiple approaches to this. Instead of using classical collaborative filtering models, many systems now build models with user embeddings and item embeddings. The question becomes: can we use some other model to predict the embeddings we would have trained using interaction data?

For example, in an e-commerce setting, if we trained our item embeddings using purchase data and a new item comes in, we could train a vision model to predict the embedding of the item based on its image and metadata. We can use that as an initial set of recommendations.

The core idea is using other available data to predict the embeddings that would have come from interaction data (like checkout data). This approach helps bridge the gap for new items or users.

## How do you handle RAG for multimodal content like PowerPoint presentations that have complex layouts?

For documents with complex layouts like PowerPoint presentations, the parsing and chunking processes are linked. You might want to evaluate them separately since parsing errors will be hard to detect in the full RAG pipeline.

One approach is to use general-purpose parsing tools like Dockling, Claude Sonnet, or commercial tools like Reducto, Llama Parse, and Extend. For multilingual content, models like VDR2B-Multi v1 handle multiple languages well.

Recent developments include using models like Gemini 2 (with its million-token context window) to convert documents to markdown and extract information, though specialized tools like Reducto still have higher accuracy (0.9 ± 0.1 vs. 0.84 ± 0.16 for Gemini). These gaps are narrowing as general models improve.

## Why did you prefer SQL databases over graph databases at Meta/Facebook?

Graph databases are useful when you need complex traversals, but most use cases only require 2-3 left joins in SQL rather than complex graph operations. From a skills perspective, it's easier to hire people who know SQL well than to find graph database experts.

At scale, graphs are also hard to manage. Around 2017-2018, only LinkedIn had a true graph database because they needed to compute 3rd-degree friendships very quickly. For most companies, SQL databases offer better performance, easier maintenance, and more familiar tooling.

Over a 12-year career, we kept trying different technologies (Hadoop, Spark, etc.) but always ended up returning to SQL. Most cases don't require more than one or two traversals of your graph, making SQL a more practical choice.

## What have you learned about prompt caching?

Prompt caching is a technique where language models can avoid reprocessing the beginning of prompts that are often identical:

- Anthropic caches prompts for 5 minutes; if you make the same request within that time, the entire message is cached
- OpenAI figures out the optimal prefix to cache automatically

This is valuable because it can save significant processing time and costs, especially when you have many few-shot examples or large system prompts. If you have 50+ examples, caching can dramatically improve performance.

For models like Claude on Bedrock, prompt caching wasn't available a few months ago but is likely coming soon. It's the kind of feature that rolls out gradually across providers.

## What's the difference between bi-encoders and cross-encoders?

A bi-encoder converts all documents into numbers (embeddings) first, and then compares those numbers. Because we pre-compute everything, we can search very quickly.

A cross-encoder doesn't compare numbers—it compares the actual sentences. This approach can't compare a million documents with a million other documents (too expensive), so instead it takes one question and 50 documents and compares each one individually.

The advantage of cross-encoders is that they can understand semantic distinctions like the difference between "I love coffee" and "I hate coffee," whereas bi-encoders just have numeric representations that might miss this nuance.

Bi-encoders are faster but less accurate, while cross-encoders are slower but better at understanding semantic distinctions.

## What's the process for fine-tuning embedding models?

It's probably a bad idea to train your own language model, but it's a very good idea to train your own embedding model.

Fine-tuning embedding models is much less resource-intensive—it typically costs around $1.50 and takes about 40 minutes on a laptop. With just 6,000 examples from your domain, you can train embedding models and cross-encoders that outperform general-purpose models on your specific tasks.

This is especially useful when you need embeddings to understand domain-specific concepts or when you're trying to define what "similar" means in your particular context (e.g., product recommendations where price range matters).

## What non-intuitive things have you learned about recommendation systems?

The big insight about recommendation systems is that inventory matters a lot more than the actual algorithm. While Tiktok's algorithm is good, what really allows it to produce great recommendations is the vast amount of content available. Without those videos, you can't do much - and the same applies to RAG.

The metadata you have and the inventory you have are much more important than the algorithm itself. For example, if recommendations for "Greek restaurants near me" are bad, the solution might be to add more Greek restaurants to your database, not to tweak the algorithm.

Similarly, if queries after 7 PM perform poorly, maybe you're missing information about whether restaurants are open. The solution is to collect that data rather than change your algorithms.

The hard work in recommendation systems is often: Do we have enough rows in the database? How much content do we have? And for that content, do we have the right metadata?

## When working with documents with metadata, should search and retrieval methods change based on the level of metadata provided within the queries?

Yes, they should. For example, in a construction project, we found people really cared about who made the last edits on legal contracts or who sent particular information. The metadata was very important for queries like "which contracts did this person send us," which function more like SQL queries.

Similarly, if you're building information that will be queried across time periods, you probably care about when documents were published and last crawled to determine relevance. A query like "what is the latest research in physics" might look at the past 6 months, while "what is new in AI" might only look at the past two weeks because it moves so quickly.

It comes down to analyzing the queries people are asking and figuring out what creates economic value.

## Do you have a go-to approach for visual document image embeddings (like quarterly reports with tables, images, graphs)?

For visual documents like quarterly reports full of tables and images:

1. Dockling is a free library that works quite well, though it might take about 11 seconds per PDF
1. Claude Sonnet also works well for extraction
1. Commercial tools like Reducto, Llama Parse, and others can be worth the cost to save time
1. For multilingual content, VDR2B-Multi v1 handles multiple languages well

Recent testing shows Reducto still has higher accuracy (0.9 ± 0.1) compared to Gemini (0.84 ± 0.16), but the gap is narrowing. Reducto performs well because they have people manually labeling thousands of PDFs to train their models.

## How do you handle multilingual RAG?

Cohere has put the most effort into multilingual models, with both multilingual local LLMs and embedding models.

I recommend figuring out which languages appear in your queries and ensuring your evaluation reflects that distribution. Check whether the models you're considering (Cohere, OpenAI) perform well on these languages.

While translation might seem like an option, if it worked well, companies like OpenAI and Cohere would already be using synthetic translation data to improve their language models. To evaluate performance across languages, create synthetic questions in multiple languages and verify whether recall rates differ between languages.

## How do you approach chunking very long documents (1,500-2,000 pages)?

If you have extremely long documents, start with a page-level approach to determine if answers typically exist on a single page or span multiple pages.

One compelling approach is from the RAPTOR paper. After chunking documents, they recluster the chunks by embedding every page, running a clustering model, and identifying concepts that span multiple pages. Then they summarize those clusters and use the summaries for retrieval—if a summary is retrieved, all related pages are included in the context.

For metadata, look at your queries to determine what matters. If users frequently ask about publication dates or document authors, those should be included. The needs will become obvious as you analyze user queries.

If you can reorganize text chunks by clustering and bringing related information together, that's very valuable. For example, with tax law documents where laws are on pages 1-30 and exemptions on page 50, you could process the document once to place exemptions directly below the relevant laws. This preprocessing step might cost $10 of LLM calls per document, but for legal documents that might not change for years, it's worth the investment.

## How do you understand metrics like precision and recall in one-to-one answer scenarios?

For questions with exactly one correct answer, these metrics behave somewhat differently. Recall will be either 0% or 100% depending on whether K is large enough to include the correct answer.

For example, if we want to retrieve exactly one document and there's only one correct answer, precision could be either 0% or 100%, and the same for recall.

The metrics become more meaningful when:

1. There are multiple relevant documents
1. We're analyzing trends across many queries
1. We're comparing different retrieval methods

Even with one-to-one mappings, MRR (Mean Reciprocal Rank) is still useful to see where the correct answer appears in your results.

What really matters isn't the absolute number but whether we can move these metrics in a positive direction with our interventions.

## How does a long context window affect RAG systems?

While having longer context windows allows for more content to be included, there are always tradeoffs with latency and cost. Just because we can fit more in context doesn't mean we should always do so.

Like how Amazon could theoretically score every product in their inventory for each user but chooses not to because each 100ms of latency costs them 1% in revenue, we still need to make choices about what to include in context.

The battery analogy is apt: iPhone batteries get more powerful every year, but battery life stays the same because we build more power-hungry apps. Similarly, as context windows grow, we'll find ways to use that additional capacity rather than making everything faster or cheaper.

There will always be cost, performance, and latency tradeoffs to consider. Having a longer context window doesn't eliminate the need for efficient retrieval - it just changes what problems we can solve.

## What tips do you have for making decisions about RAG system architecture without prototyping everything?

Start by asking for examples of 40-50 questions that customers might ask. Reading these helps build an intuition about what query mechanics need to exist.

For example, if questions include "what's the most recent news?", you'll need date filters. If queries ask "who do I talk to about fixing XYZ?", you need features for finding contacts.

This helps identify what metadata you need and whether you can access it. From there, building a demo with tools like LangChain or Llama Index should be quick. You may need to rewrite things later, but if the demo can answer generic questions, that's when you start thinking about synthetic data.

The key is getting the system in front of beta testers, collecting feedback, and analyzing what's working and what's not. This helps prioritize the next features to build. If 80% of questions are actually about image search, then that's clearly the next thing to build, regardless of what methodology is trending on Twitter.

## How do you optimally blend small pools of real data with large synthetic data sets?

Focus on whether blending improves your evaluation suite. If you have 500 real examples, put 250 in your training set and leave 250 for evaluation. Then experiment with different blends of synthetic and real data to see how they perform on your evaluation suite.

You might find that as you use more synthetic data, you perform worse on your real user data but better on synthetic data. You can weight these differently - perhaps real data success is worth 1.2 points while synthetic data success is worth 0.9 points - to create a single score for system health.

A lot of machine learning is empirical - you can't predict these things ahead of time. You need to run experiments and see what works.

## How do you approach function calling for complex workflows that require multiple function calls?

Instead of having the language model immediately execute functions one at a time, prompt it to show the entire plan to the user and potentially ask for confirmation. Have a separate function called "plan" where the model says "Based on this request, I think I'm going to use function 1, then 2, then 3. What do you think?"

When the user clicks yes or no, you've allowed human confirmation of the correct order. Since the plan already exists in the context, it's easier for the model to execute correctly.

The second benefit is that user requests, plans, and accept/reject decisions can be used as few-shot examples. You can embed these examples so that next time someone asks a similar question, you can say "Last time someone asked this, they approved calling functions 1, 2, 3 in this order."

This helps build a dataset of few-shot examples over plans, making the system more reliable.

## How do you constrain a RAG system that pulls from multiple data sources?

After conducting topic analysis of user questions, you can identify which types of questions you can answer well and which ones you struggle with. For low-percentage, low-performance questions, you might decide to simply decline those queries.

For example, if your system doesn't handle contact information well, you could add to your prompt: "If someone is asking about contact information, say no and tell them to message support." This saves face and avoids attempting questions you can't answer well.

Conversely, if there are questions you can answer very well (even if they're a small percentage), highlight these as sample questions in your UI to guide users toward queries you're confident in handling.

Much of the progress in making systems better comes from improving UI, better educating users about capabilities, and enhancing the quality of your inventory rather than just tweaking algorithms.

## Do you sometimes use differently tuned embeddings within the same query?

Unless you're at massive scale, having multiple embedding models for different content types (like product descriptions vs. comments) probably won't yield enough performance improvement to justify the maintenance cost.

There's evidence that having a single unified model trained on all your data performs better than specialized models. In machine translation, we used to train separate models for each language pair, but researchers found that a single model trained to translate all languages performed better than any individual model.

The unified model learns something about the underlying system that allows it to handle even rare cases better than specialized models would. The same principle likely applies to embedding models.

## How do you reason about papers and new research in RAG and LLMs?

Most papers published weekly aren't that important, and many are just reinventing ideas from decades ago. Instead of flooding yourself with information, focus on running experiments with your own data and solving specific problems.

The popular stuff on Twitter is generally reasonable to follow, but even then, much research is a distraction if you're building something. For example, a recent popular paper on "reasoning powered RAG" was essentially just using an LLM to judge relevancy pairs in a for loop - something basic that's been around for a while.

Rather than chasing the latest research, focus on building strong evaluation suites, analyzing your data, and solving specific problems in your implementation. These are the durable skills that will last throughout your career.

## Does a long context window make RAG obsolete?

No. Just like Amazon could theoretically score every product in their inventory for each user but chooses not to because each 100ms of latency costs them 1% in revenue, we still need to make choices about what to include in context.

The battery analogy is apt: iPhone batteries get more powerful every year, but battery life stays the same because we build more power-hungry apps. Similarly, as context windows grow, we'll find ways to use that additional capacity rather than making everything faster or cheaper.

There will always be cost, performance, and latency tradeoffs to consider. Having a longer context window doesn't eliminate the need for efficient retrieval - it just changes what problems we can solve.

## How do you handle the upkeep of documents that go in and out of scope or have frequent version changes?

For "evergreen" vs. "Rhodian" (frequently changing) documents, include published dates and make sure they're in the context so the language model is aware of them. For example, with HR holiday calendars for different years, include the dates so the model can reason about which is current.

Consider implementing both published dates and last modified dates, and be explicit in your function calling to filter on these attributes (e.g., only return documents published or updated in the past year).

The key question is how sensitive your model is to low precision, and whether that low precision is mainly happening because of your inability to expire outdated documents.

## How do you approach building voice AI for outbound calls or structured conversations?

Graph-based models or finite state machines have been very successful in the agent world. In this approach, you're in different states (introduction, data collection, etc.) with different system messages for each state.

For example, when collecting payment data to book a meeting, you have logical checks to ensure the date is correct and that you have necessary information like a phone number. The set of function calls available also changes based on the state.

Once you have fully successful conversations, you can summarize them and put them back in the system prompt to ensure transitions are more accurate. You can prompt the model with these few-shot examples to improve transition states: "When I have 5 few shots, my transitions are more accurate. When I have 20 few shots it gets too confused. So now I've picked 15 for now."

This finite state machine approach has been around for decades and is still very effective, with LLMs improving the transitions between states.

## When working with metadata, should you include it in the chunk or add it separately?

There are a few approaches:

1. Embed the string without metadata but add metadata when sending to the LLM
1. Embed the string with metadata included

This is something to test empirically - does including metadata in the embedding hurt retrieval? Cohere's embedding models (like Compass) can embed JSON quite well.

Including metadata in chunks is common practice as it allows answering questions like "who wrote this document" or "what's their contact information." This metadata can then be used for function calls, such as "Jason wrote the document 2 weeks ago, it has not been updated since. Here's Jason's email, click to write an email to Jason."

## How can you best apply synthetic data generation to agent workflows with multiple tools?

Instead of generating synthetic questions, you can generate synthetic queries that would trigger certain function calls. If you have 3-4 different functions, you can create synthetic queries that should call specific functions or combinations of functions.

If each individual function call is accurate, then the combined sequence should also be accurate. You can also use planning to improve data generation - create questions that would result in specific functions being called in sequence, then verify that with certain requests, these functions are indeed called by the model.

This approach helps ensure reliability across different types of function calling patterns.

## Key Takeaways

1. **Fine-tuning priorities**: Fine-tune embedding models, not LLMs. With just 6,000 examples, you can create embedding models that outperform general models on your specific tasks at minimal cost.

1. **Inventory matters more than algorithms**: Having the right documents and metadata is more important than the algorithm itself. Missing information can't be retrieved no matter how good your algorithm is.

1. **Evaluation is empirical**: Many decisions about chunking, including metadata, and blending synthetic data should be driven by empirical testing rather than theoretical assumptions.

1. **Parsing strategy**: For complex documents, consider evaluating parsing/OCR separately from retrieval performance since parsing errors will be difficult to trace in the full pipeline.

1. **Function calling with planning**: For complex agent workflows, have the model create a plan first and get user confirmation rather than executing functions immediately. This creates training data for future interactions.

1. **State machines still work**: Graph-based/finite state machine approaches remain effective for structured conversations, with LLMs improving the transitions between states.

1. **Metadata inclusion**: Include relevant metadata in chunks to answer questions about document properties like authorship, modification dates, and contact information.

1. **Long context doesn't eliminate RAG**: Despite larger context windows, there will always be latency, cost, and performance tradeoffs that make efficient retrieval necessary.

1. **Research pragmatism**: Focus on solving specific problems with your data rather than chasing the latest research papers, which often reinvent existing techniques.

1. **Cross-encoders vs. bi-encoders**: Cross-encoders (re-rankers) understand semantic distinctions better but are slower; bi-encoders (embedding models) are faster but less nuanced. Use both for optimal performance.

---

