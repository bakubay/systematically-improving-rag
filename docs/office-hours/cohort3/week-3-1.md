---
title: Week 3 - Office Hour
date: "2024-06-05"
cohort: 3
week: 3
session: 1
type: Office Hour
transcript: ../GMT20250605-180115_Recording.transcript.vtt
description: Re-ranking models, embedding fine-tuning, training datasets, and compute allocation strategies for RAG systems
topics:
  - Open-source Re-ranking Models
  - Training Datasets for Embeddings vs Re-rankers
  - Negative Examples for Training
  - User Feedback Integration
  - Compute Allocation in Retrieval
  - Architecture Complexity Decisions
  - Synthetic Data Generation
  - Medical Application Considerations
---

# Week 3, Office Hour (June 5)

Study Notes:

I hosted a focused office hours session addressing questions about re-ranking models, embedding model fine-tuning, and data strategies for RAG systems. Here are my insights on selecting appropriate models, creating effective training datasets with hard negatives, and making strategic decisions about compute allocation in retrieval systems.

---

---

## What open-source re-ranking models work well for fine-tuning?

When selecting re-ranking models for fine-tuning, I believe the right approach depends on your specific constraints and data volume. For many scenarios, the BGE models from the Beijing Academy of Artificial Intelligence (BAAI) have proven quite stable and easy to fine-tune with datasets of around 100,000 examples.

The model selection process isn't about finding a single "best" model, but rather systematically testing different options against your specific requirements. I've found that BAAI models tend to have more predictable loss curves during training compared to some alternatives where parameters might need more careful tuning.

Your model selection should consider several factors:

- Latency requirements (5-10 seconds for total search and retrieval in your case)
- Hosting constraints (on-premises deployment for medical applications)
- The volume of training data available
- The trade-off between performance and computational cost

For on-premises medical applications requiring self-hosting, I'd recommend starting with the BGE models and systematically testing different configurations. The process is inherently experimental - you'll likely need to train numerous models with different parameters and dataset preparations before finding the optimal combination.

**_Key Takeaway:_** Don't get fixated on finding the "perfect" model architecture. Instead, create a systematic testing framework where you can evaluate multiple models against your specific constraints of latency, hosting requirements, and performance needs.

## How should I approach creating training datasets for embedding models versus re-rankers?

The fundamental difference between training embedding models and re-rankers lies in how they handle similarity scores. Embedding models typically work with triplets (this is similar to that, different from something else) with binary scores of 1 or -1. Re-rankers, however, can be more nuanced with scores like 0.8 or 0.9, allowing for finer distinctions between results.

I've found that focusing first on building a strong dataset for your embedding model is usually the most efficient approach. If you're currently only training on positive examples, incorporating negative examples will dramatically improve performance - we're talking about a 30% improvement rather than just 6%.

For creating effective negative examples, I recommend being strategic rather than random:

In a financial context I worked with recently, we were distinguishing between "fuel" (for employee vehicle reimbursements) and "equipment fuel" (for company vehicles like tractors). Simply using random negative examples wouldn't help the model learn this subtle distinction. Instead, we created hard negatives by:

1. Taking a transaction from one category

1. Finding another transaction in the same category as a positive example

1. Using embedding search to find the most similar transaction from a different category as the negative example

This approach forces the model to learn the meaningful boundaries between similar but distinct concepts. For your medical data with abbreviations that have different meanings in different contexts, you could apply a similar strategy - finding examples where the same abbreviation appears in different contexts to create hard negatives.

**_Key Takeaway:_** Including well-crafted negative examples in your training data is crucial for model performance. Focus on creating "hard negatives" that challenge the model to learn subtle distinctions rather than obvious differences.

## What are effective sources of negative examples for training data?

Some of the most valuable negative examples come from user interactions that indicate a mismatch between what the system thought was relevant and what the user actually found useful. I've implemented several approaches across different domains:

For citation systems:

When experts review citations and delete ones they find irrelevant, saying "regenerate without this document because it's misleading" - that's a perfect negative example. The question and the deleted chunk form a negative pair for training.

For recommendation systems:

- In content recommendation, when a salesperson deletes a suggested blog post from an automated email, that's a negative example
- In music services like Spotify, skipping a song is a weaker negative signal than deleting it from a playlist
- In e-commerce, items that are purchased together but later returned indicate a false positive that can be used as a negative example

For your medical context, you might consider:

- Tracking when users reject or ignore certain retrieved chunks
- Using expert feedback to identify misleading or irrelevant retrievals
- Creating synthetic data with language models to generate examples where abbreviations are used in different contexts

The key insight is that these high-signal negative examples often come from cases where the system initially thought it was right but was ultimately wrong - these boundary cases are extremely valuable for training.

**_Key Takeaway:_** The most valuable negative examples often come from user interactions that indicate a mismatch between system predictions and actual relevance. Design your system to capture these signals and incorporate them into your training data.

## How should I think about compute allocation in retrieval systems?

When designing retrieval systems, especially for complex documents like legal texts or medical records, I think about it as a fundamental trade-off: where do I want to allocate my compute resources? This is essentially a decision between investing compute at "write time" (indexing) versus "read time" (retrieval).

There are two main approaches to consider:

Contextual retrieval (compute-heavy at write time):

- Rewrite text chunks during indexing to include all necessary context
- For example, converting "He is unhappy with her" to "Jason the doctor is unhappy with Patient X"
- This makes retrieval simpler but requires more upfront processing
- Anthropic has published about this approach for their Claude assistant

Tool use and traversal (compute-heavy at read time):

- Store minimal context in each chunk
- Use additional compute during retrieval to navigate between related chunks
- Similar to how Cursor IDE navigates code by finding functions and then examining surrounding context
- This approach is more flexible but can feel slower to users

For your medical application where the data is self-contained (not requiring external information), and where you want to minimize user wait time, I'd lean toward investing more compute at indexing time. This is especially true since you can run indexing jobs overnight without affecting user experience.

The decision also relates to data normalization - do you want to denormalize data by including related information in each chunk (like adding phone numbers whenever a person is mentioned), or keep information separate and join it at retrieval time? The answer depends on your specific use case and resource constraints.

**_Key Takeaway:_** Frame your retrieval system design as a strategic decision about compute allocation. For medical applications with self-contained data and latency constraints, investing more compute at indexing time to create context-rich chunks will likely provide a better user experience.

## What determines the complexity of the architecture I should use?

I believe the volume and quality of your training data should be the primary factor determining architectural complexity. This is a principle I emphasize repeatedly: your dataset size dictates what approaches make sense.

As a general guideline:

- With ~100 examples: Use few-shot prompting
- With thousands of examples: Fine-tune embedding models
- With millions of examples: Fine-tune language models

The data volume determines what's feasible. If you told me you had a million examples, I'd probably just train a language model directly and worry about everything else later. With limited data, you need to be more strategic about targeting specific challenges like medical abbreviations with ambiguous meanings.

This is why I'm skeptical when I see engineers celebrating 98% accuracy on their first model - it usually means they've created a test set that's too easy. As your model improves, you should be making your test data harder by finding more challenging examples. If your retrieval dashboard is showing 95% accuracy, that's a sign you need to create harder test cases.

**_Key Takeaway:_** Let your data volume guide your architectural decisions. With limited data, focus on targeted improvements to specific challenges rather than complex architectures. As your model improves, continuously create harder test cases to drive further improvement.

## How can I improve my system when I don't yet have real user feedback?

Without real user feedback, you can still make significant progress through synthetic data generation and expert knowledge. For your medical abbreviation challenge, you could:

1. Identify known ambiguous abbreviations in medical contexts

1. Use language models like GPT-4 to generate synthetic examples showing these abbreviations in different contexts

1. Have medical experts validate these examples or create additional ones

1. Build a curated dataset of hard negatives focusing on these ambiguities

This approach lets you systematically address known challenges before deployment. Once you have real users, you can implement feedback mechanisms to capture when they reject or modify system outputs, creating a virtuous cycle of improvement.

Remember that as your system improves, you need to continuously create harder test cases. If you're scoring 95% accuracy, it's not because your AI is exceptional - it's because your test data isn't challenging enough. The goal is to build a dataset that pushes the boundaries of what your system can handle.

**_Key Takeaway:_** Before having real users, leverage synthetic data generation and expert knowledge to create challenging test cases. Design your system to capture user feedback from the start, as this will become your most valuable source of training data once deployed.

---

FAQs

## What open-source re-ranking models are recommended for fine-tuning?

The Beijing Academy of Artificial Intelligence (BAAI) models, such as BGE re-ranker v3, are often good choices for fine-tuning. These models have proven to be stable during the training process and work well with the sentence transformers library. When selecting a model, consider your specific data volume and performance requirements. Testing multiple models with your dataset is ultimately the best approach to finding the optimal solution for your specific use case.

## How should I approach model selection for my re-ranking needs?

Start by exploring models that perform well on MTAB benchmarks on Hugging Face. Consider your constraints around latency, hosting requirements, and data volume. With the right dataset, the process becomes more about searching the model space to find what works best for your specific scenario. For medical or specialized domains, you'll want to test various models against your specific data to determine which one provides the best performance-to-cost ratio.

## What are the different types of re-ranking models available?

Re-rankers can be embedding models, cross-encoder models, or even large language models (LLMs). Each has different characteristics: embedding models typically classify results as relevant or not relevant, cross-encoders can provide more nuanced scoring (like 0.8 vs 0.9 relevance), and LLM-based re-rankers (like those used by Exa) can provide sophisticated re-ranking but may have higher latency. Your choice depends on your specific requirements and constraints.

## Training Data for Embedding and Re-Ranking Models

How important are negative examples when training embedding models?

Negative examples are extremely valuable when training embedding models. Including hard negatives (examples that are similar but should be classified differently) can improve performance by 30% or more compared to training with only positive examples. These hard negatives help the model learn subtle distinctions that are crucial for accurate retrieval, especially in specialized domains like medicine or legal text.

## What's an effective way to generate hard negative examples?

One effective approach is to find examples that are semantically similar but belong to different categories. For instance, if you're working with medical abbreviations that have different meanings in different contexts, you could create examples showing the same abbreviation used in different medical specialties. Another method is to use embedding search to find the most similar items that should be classified differently, rather than just using random negative examples.

## How can user feedback be leveraged to improve retrieval models?

User interactions provide valuable signals for creating training data. For example, if users delete a citation or regenerate an answer without a specific document, that's a strong signal that the document was irrelevant (a negative example). Similarly, if users consistently skip or remove certain recommendations, those can be used as negative examples in your training data. These real-world signals often create the most valuable training examples.

## What factors should I consider when designing a retrieval system architecture?

Consider where to allocate your compute resources: at indexing/write time or at query/read time. If you invest more in preprocessing your data (like contextual retrieval where you rewrite chunks to include necessary context), your query-time processing can be simpler and faster. Alternatively, you can keep preprocessing minimal and implement more sophisticated query-time processing like document traversal. Your decision should balance user experience requirements, cost constraints, and the specific characteristics of your data.

## How do I handle context in long documents where paragraphs depend on previous content?

There are two main approaches: (1) Contextual retrieval, where you rewrite text chunks at indexing time to include all necessary context, making each chunk self-contained; or (2) Document traversal, where your system can navigate through the document at query time to gather needed context. The first approach frontloads the processing cost but enables faster query responses, while the second requires more complex query-time processing but minimizes preprocessing.

## What hosting considerations should I keep in mind for medical applications?

For medical applications, especially in European contexts like the Netherlands, you'll likely need to host models on your own hardware or on-premises at hospitals. This requires selecting models that can run efficiently on your available hardware while meeting your latency requirements. Consider models that can be fully self-hosted without external API dependencies, and ensure your architecture complies with relevant healthcare data regulations.
