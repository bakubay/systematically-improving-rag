---
title: Week 5 - Office Hour 1
date: "2024-06-17"
cohort: 3
week: 5
session: 1
type: Office Hour
transcript: ../RAG_Cohort_3_-_Office_Hour_Week_5_Day_1.txt
description: Fine-tuning citation accuracy, temporal reasoning, and specialized tool portfolios for RAG systems
topics:
  - Fine-tuning for citation accuracy
  - Citation source ordering
  - Tool portfolio design
  - Temporal reasoning in medical data
  - Multi-agent vs single-agent
  - Document summarization
  - Automated price quote generation
  - Data formatting best practices
  - Evaluation of complex RAG systems
---

# Week 5, Office Hour 1 (June 17)

I hosted an office hours session focused on fine-tuning models for citation accuracy and designing effective tool portfolios for RAG systems. Here are my insights on improving citation capabilities through fine-tuning, structuring data for temporal reasoning, and creating effective tool portfolios for specialized retrieval tasks.

---

---

## How effective is fine-tuning for improving citation accuracy?

When working with citation requirements, fine-tuning can dramatically reduce error rates. In one project, we used OpenAI's fine-tuning API with about 1,000 examples to improve our marketing content generation system.

The results were impressive - our error rates dropped from around 4% to essentially 0% on our test set of 200 examples. We didn't need complex frameworks like Fluoro since we weren't hosting local models, just using OpenAI's API directly.

The key was having evaluators validate our offline data, filtering out incorrectly formatted examples before using them in the fine-tuning process. This approach worked particularly well because we weren't trying to change the model's knowledge - just its formatting behavior.

When determining how much data you need, I recommend experimenting with different sample sizes:

"What I would often do is try to use a subset of my data for fine-tuning, then increase the sample size and figure out what that curve looks like. It's going to be performance versus volume."

Different models will have different learning curves - a 1.3 billion parameter model might flatten out at 10,000 data points, while larger models might show different patterns. Adjusting learning rates can also affect these curves.

**_Key Takeaway:_** Fine-tuning can be remarkably effective for formatting tasks like citation, often requiring less data than you might expect. Start with small batches, measure performance, and increase data volume until you reach your desired accuracy level.

## Should we shuffle citation sources during fine-tuning?

When fine-tuning models to cite sources correctly, shuffling the order of retrieved sources can be beneficial to prevent position bias. This approach makes the model invariant to the order of sources, which is particularly important if you're not sorting by relevance.

However, if you are sorting by relevance, maintaining the original order might actually be preferable: "Maybe it is important for the model to know that the first text chunk is the most relevant text chunk."

The need for shuffling may also depend on the context length of your model. With older, smaller context models (like 4K token models), position bias was more pronounced due to the "lost in the middle" effect. Newer models with better attention mechanisms have improved recall across their context window.

"If you look at the newer models, they just have way better lost-in-the-middle sensitivity in general, and I would expect that when you fine-tune these things, they also preserve some of that ability to attend over long contexts."

The real challenge that remains is reasoning over multiple "needles" of information scattered throughout a document - connecting facts from different sections remains difficult for most models.

**_Key Takeaway:_** Consider shuffling citation sources during fine-tuning if you want position-invariant citations, but if you're sorting by relevance, maintaining order may be beneficial. Newer models have better attention across their context window, reducing the need for this technique.

## How should we approach tool design for specialized retrieval tasks?

When designing tools for retrieval systems, focus on creating a portfolio of specialized tools rather than just distinguishing between semantic and structured data. The key question isn't "Am I searching semantic or structured data?" but rather "What is the portfolio of tools I want to expose to my system?"

For example, in a construction use case, we implemented several specialized tools:

- Generic document search that searches everything
- Contact search for finding people
- RFI (Request for Information) search that takes specific RFI codes
- Contract search that returns not just text chunks but also responsible parties

The implementation details (whether it's semantic search or structured data) matter less than how you present these tools to the language model. Your focus should be on making sure the model understands what tool to use and when.

For evaluating tool selection, I recommend having the model "write a plan of all the tools it might want to use" for a given query, then evaluating that plan first. You can even present this plan to users for approval before execution, which creates valuable training data based on acceptance rates.

"That gets you a pretty good dataset in terms of customer plan acceptance rates, and then you can look at the ones that are not accepted and figure out what you need to do afterwards."

The naming of tools significantly impacts how models use them. In coding agents, for example, providing a specific "grep" tool versus just mentioning grep in the command line instructions can change execution patterns by 2% in evaluations.

**_Key Takeaway:_** Design a portfolio of specialized tools based on specific use cases rather than general data types. Focus on clear tool descriptions and evaluate how well the model selects the appropriate tools for different queries.

## How can we handle temporal reasoning in medical data?

One of the most challenging aspects of working with medical data is reasoning about information across a timeline. When retrieving documents about medications, for example, you might get 20 documents all describing medications, but understanding what changed over time requires special handling.

"You might want to know what changed over time, or you have to always see it in the context of time. And you also need to find relationships like 'there's this medication and the patient became worse' or 'this medication went up' - that all needs to be conceived in the system."

For presenting temporal data effectively to models, I recommend structuring it as a markdown table whenever possible. In our testing, markdown tables performed 12% better than CSV, JSON, or YAML formats for complex lookup tasks across large datasets.

"We've done tests where I put like 6,000 rows, 50 columns as CSV, as markdown, as JSON, as YAML - and markdown tables is like 12% better in terms of identifying on row X where the value is Y, find me the row that's one above and one to the left."

The ordering of temporal data also matters significantly. You might get different results if you order events in ascending versus descending time. This affects how the model scans and reasons about cause and effect relationships.

For building better temporal reasoning capabilities, consider:

1. Ordering retrieved documents chronologically

2. Presenting data in markdown table format with clear timestamps

3. Having the model first extract and reorganize relevant information before reasoning about it

4. Mining reasoning chains from expert users to create training data

**_Key Takeaway:_** For temporal reasoning, structure data chronologically in markdown tables and implement a two-stage approach where the model first extracts and organizes relevant timeline information before reasoning about it.

## What's the difference between multi-agent and single-agent approaches?

The debate between multi-agent and single-agent systems often comes down to context coordination challenges. For coding tasks, Devin (from Cognition) chose a single-threaded approach because coordinating between agents modifying different parts of a codebase is extremely difficult.

"If one agent modifies one directory and another agent modifies another directory, that communication channel is sort of not well defined yet."

In contrast, Claude's Deep Research uses multiple agents, but they're all read-only - they don't need to coordinate changes because they're just retrieving information that will later be combined:

"In that multi-agent system, the agents are all read-only, so they don't need to manage that communication overhead because they're all going to be reduced. If I search about who I am, one agent searches childhood, one agent searches career, and once they bring all the information back, they can be reduced."

The primary benefit of multi-agent systems appears to be token efficiency - you can use more tokens across multiple agents than with a single agent. "The performance just increases with the amount of tokens each sub-agent is able to consume. If you have 10 sub-agents, you can use more tokens, and your research quality is better."

For medical data applications that are primarily read-only, a multi-agent approach might work, but the challenge remains in ensuring no context is missed when combining information from different agents.

**_Key Takeaway:_** Choose multi-agent approaches for read-only tasks where you need to process more tokens than a single context window allows. For tasks requiring coordination of changes, single-agent approaches remain more practical until better coordination mechanisms are developed.

## How can we use document summarization to improve retrieval?

Generating summaries during document ingestion can be a cost-effective approach to improving retrieval. Summaries function as a form of compression and can be particularly valuable when working with smaller context window models.

"In general, this is a good idea because that's almost in some ways just a more cost-effective way of doing this contextual retrieval stuff. Summary is just compression."

The key is designing your summarization prompt based on the specific tasks your system needs to perform. For example, with architectural blueprints, we knew users would ask about room counts and dimensions, so we created summaries that explicitly counted and listed these elements:

"Because we know that our tasks involve things like extracting the names of rooms and counting things, if our language model can have a summary that counts everything, then it becomes much easier to think about 'the place with 4 bedrooms and 2 bathrooms.'"

We implemented this as a separate document search tool that only hits the summaries. Through iteration and evaluation, we improved our summary generation from 16% recall to 85% recall in just a few days.

For implementation, you can:

1. Create a separate "search summaries" tool

2. Design summary prompts that extract the specific types of information users will query

3. Evaluate and iterate on summary quality using test queries

4. Use summaries as synthetic text chunks that supplement your existing text chunks

This approach works particularly well for documents like financial reports, where structured information can be extracted, or for multimedia content where describing images or videos in text makes them searchable.

**_Key Takeaway:_** Document summarization during ingestion creates valuable synthetic text chunks that can dramatically improve retrieval performance. Design summary prompts based on the specific information needs of your application and iterate based on evaluation results.

## How can we implement price quote generation using RAG?

One practical application we've built is an automated price quote system for sales teams. After multiple calls with a prospect, the system generates personalized pricing options and potential upsells.

The process works like this:

1. We have 16 pages of pricing information (per-seat pricing, volume discounts, prepayment options)

2. We have transcripts from 6 phone calls with the prospect

3. We ask the language model to:

- Read the transcripts and list all relevant variables
- Extract the values of those variables
- Reason about the pricing document
- Propose options and upsells
- Write an email to the prospect

"The email's like 'Great talking to you, Tim. It sounds like for a company your size, you can probably commit to 15 seats. This will get you a 20% discount. If you don't use it, we'll move it to next year, and if you pay upfront, we can give you another 20-25% discount because I know that's something your CTO really values.'"

Our evaluation method is simple but effective - we have salespeople review the generated emails before sending them, and we track whether they make edits. When edits are needed, we analyze what went wrong in the reasoning step.

This approach of extracting variables, reasoning about them, and then generating output could be applied to medical data as well. For example, if a patient shows drowsiness, the system could first extract all timeline information about drowsiness, then reason about potential causes.

**_Key Takeaway:_** For complex reasoning tasks, implement a multi-step process where the model first extracts and organizes relevant information, then reasons about it, and finally generates output. This structured approach makes the reasoning more transparent and easier to evaluate.

## What's the best way to format data for language models?

When presenting structured data to language models, markdown tables consistently outperform other formats like CSV, JSON, or YAML. In our testing, markdown tables were 12% more effective for complex lookup tasks.

"We've done tests where I put like 6,000 rows, 50 columns as CSV, as markdown, as JSON, as YAML - and markdown tables is like 12% better in terms of identifying on row X where the value is Y, find me the row that's one above and one to the left."

The formatting details matter significantly. For example, having spaces between tokens in markdown tables (like "| data |" instead of "|data|") affects how the model processes the information.

"If I search for the word Jason, the token is 'space Jason'. But if it's Jason in JSON, it's actually 'quote Jason' - those are different tokens. And so those things end up affecting the lookup a little bit."

These seemingly minor formatting choices can have meaningful impacts on model performance, especially for tasks requiring precise information retrieval or table navigation.

For temporal data specifically, presenting information in chronological order (either ascending or descending) can significantly affect how models reason about cause and effect. Testing both approaches is worthwhile, as one may work better than the other depending on your specific use case.

Markdown tables consistently outperform other data formats for structured information. Pay attention to spacing and formatting details, as they affect tokenization and retrieval performance. For temporal data, experiment with both chronological and reverse-chronological ordering.

## How should we approach end-to-end evaluation of complex RAG systems?

End-to-end evaluation of complex retrieval systems remains challenging, especially when there isn't a single correct answer or when the system needs to perform multi-step reasoning.

"The end-to-end evaluation of these kinds of things are still pretty challenging, unless it really is the case that there are just certain text chunks that we're trying to achieve or certain answers we already know ahead of time."

For tool selection, one effective approach is evaluating the system's planning capabilities:

1. Ask the model to write a plan of which tools it would use for a query

2. Evaluate the plan before executing it

3. Allow users to approve or reject the plan

4. Track plan acceptance rates and analyze rejected plans

For reasoning tasks, breaking evaluation into steps can be helpful:

1. Evaluate information extraction (did the system find the relevant information?)

2. Evaluate reasoning (given the correct information, did it reach valid conclusions?)

3. Evaluate output generation (was the final response clear and actionable?)

In some domains like coding, the evaluation metrics are clearer - does the code pass tests? In other domains like medical reasoning, evaluation may require expert review or comparison to known outcomes.

For systems like our price quote generator, we use a practical metric - do salespeople edit the generated emails before sending them? This real-world usage metric helps us identify where the system's reasoning falls short.

**_Key Takeaway:_** Break evaluation into component parts rather than relying solely on end-to-end metrics. Incorporate user feedback into your evaluation process, and track how often outputs require human editing or intervention.

## How does fine-tuning improve citation accuracy in LLMs?

Fine-tuning can dramatically reduce error rates when teaching models to properly cite sources. In one example, fine-tuning reduced citation errors from 4% to nearly 0% for marketing content generation. The process involves collecting properly formatted examples, validating them, filtering out incorrect formats, and using them in the fine-tuning process.

## How many examples are typically needed for effective fine-tuning?

Around 1,000 high-quality examples can be sufficient for format-related fine-tuning tasks. However, the exact number depends on your specific use case. It's recommended to experiment with increasing sample sizes to determine the optimal amount for your needs. Start with a smaller subset and gradually increase to identify where performance improvements begin to plateau.

## Should I shuffle the order of retrieved sources in my fine-tuning dataset?

Shuffling retrieved sources can be beneficial to make your model invariant to the order of information. This approach helps prevent the model from developing biases toward information presented first. However, if your retrieval system sorts by relevance, maintaining that order might be important as the first chunk would genuinely contain the most relevant information.

## How should I approach tool selection for my LLM application?

Focus on developing a portfolio of specialized tools rather than simply categorizing between semantic and structured data searches. Consider what specific capabilities would benefit your use case, such as date-range filtering, categorical filters, or metadata tag filtering. The implementation details (whether semantic or structured) matter less than ensuring your model understands when to use each tool.

## What's an effective way to evaluate tool selection by the model?

A practical approach is to have the model write a plan listing all tools it would use for a given query, then evaluate that plan before execution. You can present this plan to users for approval or rejection, which generates valuable feedback data. Analyzing rejected plans helps identify improvements needed in your tool selection and routing logic.

## How do coding agents approach tool integration?

Coding agents have made significant progress with tool integration. One key insight is that providing named tools for specific functions (rather than general capabilities) significantly changes how frequently these functions are used. For example, providing a dedicated "grep" tool versus expecting the model to remember to use grep through a general command line interface can improve performance by several percentage points.

## How should I organize timeline-based data for LLM processing?

For timeline data, consider presenting information in a markdown table format, which models tend to process effectively. Order the data chronologically (either ascending or descending) and include clear date markers. This organization helps the model understand temporal relationships and reason about cause and effect. Testing both ascending and descending time orders may yield different results depending on your use case.

## Why are markdown tables particularly effective for structured data?

Markdown tables have shown superior performance (approximately 12% better) compared to other formats like CSV, JSON, or YAML when models need to perform lookup tasks or understand relationships between data points. The spacing between tokens in markdown tables appears to be particularly well-suited to how models process information.

## How can I help models reason across complex information?

For complex reasoning tasks, consider implementing a two-step approach: first have the model extract and reorganize all relevant information from different sources, then reason about this reorganized information. This approach works well for tasks requiring synthesis across multiple data points, such as analyzing medical timelines or generating pricing quotes based on multiple conversations.

## Is it beneficial to generate summaries during data ingestion?

Creating summaries during data ingestion can be very effective, especially for longer documents. Summaries act as compressed versions of your data that can be more efficiently processed. For specific use cases like blueprints or financial documents, you can design summarization prompts that extract the most relevant information (like room counts or key financial figures) to make subsequent queries more efficient.

## How can I handle reasoning across multiple documents?

For reasoning across multiple documents, consider having the model first extract all relevant information related to the query, reorganize it (possibly chronologically or thematically), and then reason about the reorganized information. This approach helps manage context limitations and focuses the model's attention on the most pertinent details.

## What's the best way to handle long context windows?

Newer models with improved attention mechanisms handle long contexts better than older models. However, for complex reasoning tasks involving multiple "needles" of information spread throughout a document, consider using tools that first organize the relevant information before reasoning about it. This approach remains effective even with models that have long context windows.

---

