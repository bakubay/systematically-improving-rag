---
title: Week 2 - Office Hour 1
date: "2024-05-27"
cohort: 3
week: 2
session: 1
type: Office Hour
transcript: ../GMT20250527-124233_Recording.transcript.vtt
description: Advanced RAG strategies for specialized domains, complex data structures, and economically valuable AI solutions
topics:
  - Medical RAG Systems
  - Citation Handling in RAG
  - Graph-based vs Traditional RAG
  - User Query Clustering
  - Documentation Chunking Strategies
  - Vector Database Trade-offs
  - Economic Value in AI Systems
  - Blueprint Analysis for Construction
---

# Week 2 - Office Hour 1 (May 27)

Study Notes:

I hosted an office hours session focused on advanced retrieval-augmented generation (RAG) strategies and implementation challenges. Here are my insights on building effective RAG systems for specialized domains, handling complex data structures, and designing economically valuable AI solutions that go beyond simple question-answering.

---

---

## How should I approach medical RAG systems with complex queries?

When dealing with specialized domains like medical records where users ask comprehensive questions (e.g., "Give a complete medical history of patient X"), the key is understanding that you can't just throw everything into a generic RAG system and expect good results.

I've found that building separate indices for different document categories is essential. For example, with an oil and gas client I'm working with, we're processing millions of PDFs but categorizing them into about five different types: drill logs, specifications, geospatial diagrams, etc.

Within each category, we extract specific data structures. For drill logs, we identify that some pages represent different days of logging, so we extract metadata like dates, drill IDs, and personnel information. This allows us to build specialized tools for querying each data type effectively.

For medical history queries, you'd want to create structured data that can be queried directly - essentially turning it into a "SELECT \* FROM medical_history WHERE client_id = X" type of operation rather than relying on semantic search across unstructured text.

**_Key Takeaway:_** Don't try to build one universal RAG system. Instead, identify the categories of documents in your domain, extract relevant structured data from each category, and build specialized tools to query that structured data effectively.

## What's the best approach to handling citations in RAG systems?

When your LLM isn't reliable for generating citations and semantic/fuzzy similarity doesn't work well (particularly in domains with many abbreviations like medicine), you need a more structured approach.

I recommend following Claude's citation approach, which uses XML tags to wrap citations. When you create statements, include XML that references the source ID. In your text chunks, you'll have chunk IDs and other metadata alongside the content.

To make this more precise, especially with longer contexts, include the first three words and last three words of the cited span. For example, if citing "similarity isn't reliable either for our use case," the citation would include both the chunk ID and "start is similarity isn't reliable, end is for our use case."

This approach works well with fine-tuning. We implemented something similar in Instructor, where an answer is structured as a list of facts, each with a substring quote, ensuring alignment between facts and quotes to minimize hallucinations.

**_Key Takeaway:_** Structure your citations with explicit references to chunk IDs and text spans rather than relying on similarity matching. This approach can be implemented through fine-tuning and provides much more reliable attribution.

## Should I use graph-based RAG approaches?

I'm generally skeptical about graph-based RAG systems. In my experience with data analysis over many years, graph databases often fall away in favor of embeddings and SQL databases.

The main challenge with graph RAG is that building out the taxonomy is often harder than you expect. You think you're avoiding the complexity of embedding models, but you're just substituting it with the problem of modeling out the taxonomy, which can be equally challenging.

For most use cases where you might consider a graph, you can achieve similar results with a few SQL joins. Unless you need to do complex traversals (like LinkedIn finding connections-of-connections), the overhead of learning graph query languages and modeling data as graphs usually isn't worth it.

Even Facebook, despite being fundamentally a social graph, uses a very large-scale MySQL instance rather than a dedicated graph database. If you only need one-way traversals, a standard relational database is typically sufficient.

**_Key Takeaway:_** Unless your use case requires complex multi-step graph traversals, you're likely better off using embeddings with SQL databases rather than implementing a graph-based RAG system. The taxonomy development often becomes more complex than the embedding approach you were trying to avoid.

## How do you recommend clustering and categorizing user queries?

For understanding what users are asking about, we've developed a library called Cura (similar to Anthropic's Clio) that performs population-level analysis of conversation history.

The process works like this:

1. We summarize every conversation
2. We extract key information: languages used, topics, tasks, requests, user complaints, and assistant errors
3. We concatenate everything and create embeddings
4. We perform clustering to identify patterns
5. We use a language model to group and label clusters

This approach gives you insights into what people are asking for, how big each cluster is, and metrics like error rates or user satisfaction for different types of queries. You can then identify which clusters are performing well and which need improvement, helping you decide where to invest in new tools or capabilities.

We're releasing a new version of Cura soon with better ergonomics and UI for exploration. This will be covered in more detail in Week 4 of the course.

**_Key Takeaway:_** Systematic analysis of user queries through summarization, extraction, embedding, and clustering helps identify patterns in how people use your system, allowing you to prioritize improvements where they'll have the most impact.

## What's your recommendation for chunking documentation?

When dealing with documentation PDFs containing tables, definitions, paragraphs, and figures, I challenge the conventional wisdom about chunking. For documentation, I believe the chunk should often be the size of the document page.

The right question to ask is "which page do I need to look on?" rather than trying to break documents into arbitrary chunks. Modern models are large enough to handle page-sized chunks, and documentation typically uses consistent terminology (unlike cases where semantic search helps bridge vocabulary differences).

By combining semantic and lexical search and focusing on page-level retrieval, you can often get better results than with smaller chunks. This approach also respects the semantic boundaries that document authors typically maintain - they rarely split headers from content across pages or break logical sections in awkward places.

**_Key Takeaway:_** For documentation, consider using page-level chunking rather than arbitrary token-based chunking. This respects the document's inherent structure and works well when combined with both semantic and lexical search approaches.

## What are the trade-offs between different vector database options?

I generally prefer using Postgres with pgvector because it allows me to join on different tables, which is extremely valuable. However, pgvector doesn't do exhaustive search by default, which can be a limitation with large datasets.

If you're dealing with very large vector collections, consider Timescale's pgvector_scale, which has better streaming methods for exhaustive search. Another advantage of the Postgres approach is that you can install pg_search from PostgresML to get BM25 implementation, giving you both vector search and lexical search in the same database.

This combination of vector search and lexical search in a single database that also supports filtering by metadata (like dates or access permissions) is powerful for real-world applications.

**_Key Takeaway:_** Postgres with pgvector provides a good balance of functionality for most RAG applications, especially when combined with pg_search for lexical search. For very large datasets, consider specialized extensions like pgvector_scale.

## How do you approach building economically valuable AI systems?

When building AI systems, I focus on economic value rather than just time savings. Time savings is bounded - you can only save as much time as someone currently spends. But economic value can be much larger.

For example, with construction blueprints, we realized that simply answering questions about window heights wasn't that valuable - it just saved a worker a few minutes. But by extracting structured data about room counts, building lines, and floor numbers, we could quickly locate the right blueprints when workers were on site, preventing costly delays.

In another case, we built voice agents that call car owners to schedule maintenance appointments. Rather than charging by call duration, the system charges a percentage of what the mechanic makes. This aligns incentives - the AI provider is motivated to resolve phone numbers correctly, ensure calendar synchronization works, and get customers to actually show up.

The most valuable systems help make better decisions, not just answer questions. If you're building a hiring assistant, don't just price based on tokens used - think about what a bad hire costs a company and how much value your system provides by helping them avoid that outcome.

**_Key Takeaway:_** Focus on building systems that drive economic value through better decision-making rather than just answering questions or saving time. Structure your pricing to align with the value you create, such as taking a percentage of revenue generated or costs avoided.

## How did you handle blueprint analysis for construction projects?

For a construction project involving blueprints, we realized through user query analysis that workers needed to find specific documents based on their location in a building. They'd say things like "I'm on the 40th floor in a room with two bedrooms and a bathroom on the north-facing side - find me the schemas for the windows."

We built a system that extracted structured data from blueprints: which building, which floor, which "line" (position in the building), room counts, and directional orientation. This required a combination of bounding box models and LLMs to identify and extract this information.

The challenge was proving that we needed to invest in these specialized models. By analyzing the types of questions being asked, we could justify building tools that could count rooms, identify directions, and extract other key metadata that made retrieval much more effective.

For specialized domains like blueprints, it's crucial to understand the specific queries users have and build structured data models that directly address those needs rather than relying on generic text embeddings.

**_Key Takeaway:_** For specialized visual content like blueprints, invest in extracting structured data that matches the way users think about and query the information. This often requires specialized models beyond general-purpose LLMs, but the investment pays off in much more effective retrieval.

## Final thoughts on building effective RAG systems

The most successful RAG implementations I've seen share a few common characteristics:

1. They don't try to build one universal system but instead create specialized tools for different document categories and query types
2. They extract structured data that matches the way users think about and query information
3. They combine multiple search approaches - semantic, lexical, and metadata filtering
4. They focus on delivering economic value, not just answering questions
5. They evolve based on systematic analysis of user queries and pain points

As we continue to develop these systems, I expect to see more specialized, domain-specific implementations that go beyond generic question-answering to provide decision support and drive measurable business outcomes. The future of these agents will be selling work and outcomes, not just time and tokens.

---

FAQs

## What approach should I take for medical RAG with complex queries?

For complex medical queries like "give a complete medical history of patient," a generic chunking approach isn't sufficient. Instead, build separate indices for different categories of documents and create specific data structures for each type. For example, with medical records, you might create distinct structures for doctor's visits, referral letters, and prescriptions. This allows you to develop targeted tools that can directly query these structures rather than relying on general semantic search across all documents.

## How should I handle citations in my LLM responses?

When implementing citations in LLM responses, consider using an XML-based approach similar to Claude's citation system. This involves wrapping citations with XML tags that reference the source chunk ID along with the first and last few words of the cited span. For fine-tuned models, you can train the model to output citations in this format, which provides more precise references than simple chunk IDs. This approach works well even when the model rephrases information from abbreviation-heavy medical texts.

## What are your thoughts on graph-based RAG versus traditional approaches?

While graph-based RAG sounds promising, it often substitutes one complex problem (embedding models) with another (taxonomy modeling). For most use cases, a well-structured SQL database with appropriate joins is more practical than implementing a graph database. Graph databases require learning new query languages and modeling approaches, which adds significant overhead. Unless you need complex multi-step traversals (like LinkedIn's connection finder), the benefits rarely outweigh the costs. Most "graph-like" relationships can be effectively modeled with standard SQL joins.

## How should I approach chunking for documentation-based RAG?

For documentation, consider using page-level chunking rather than arbitrary token-based chunks. This aligns with how documentation is naturally structured and how authors organize information. Combine semantic search with lexical search for better results, as documentation typically uses consistent terminology. Test this approach with evaluations to verify its effectiveness for your specific use case. Remember that document creators are usually aware of page-level semantics and rarely split important concepts across pages.

## How can I understand what my users are asking about?

To analyze user queries effectively, use a conversation analysis tool like Cura. This approach involves:

1. Summarizing each conversation
2. Extracting key information (language used, topics, tasks, requests, complaints)
3. Embedding this data
4. Clustering similar conversations
5. Using an LLM to label and group these clusters

This gives you insights into what users are asking, which features are performing well, and which need improvement. You can then develop targeted tools to address the most common or high-value query types.

## What's your experience with extracting data from construction blueprints?

When working with construction blueprints, focus on extracting structured data that answers specific questions users ask. For example, in a condominium project, we extracted data like floor numbers, room counts, directional orientation, and unit identifiers. This required developing specialized bounding box models to identify key elements in the blueprints. The approach was driven by analyzing actual user queries, which revealed they needed to quickly locate specific information like window dimensions or material specifications for particular rooms or floors.

## Should I use Postgres with pgvector for my RAG implementation?

Postgres with pgvector is a good choice for RAG implementations because it allows you to combine vector search with traditional SQL queries, enabling pre-filtering by metadata like dates or access permissions. For better performance, consider pgvector-scale, which provides more efficient exhaustive search capabilities for larger datasets. Adding pg_search from PostgreSQL gives you BM25 implementation, allowing you to combine vector search with lexical search in the same database. This approach gives you flexibility to switch between semantic and lexical search while maintaining the ability to join with other data tables.

## How do you determine which user questions are worth optimizing for?

Focus on identifying which questions deliver the most economic value, not just which are most common. For example, in a construction project, helping workers quickly locate specific blueprint details might save a few minutes, but identifying unsigned contracts that could cause project delays delivers much higher value. Analyze your user conversations to identify these high-value query patterns, then build specialized tools to address them. The goal isn't just to answer questions faster but to help users make better decisions that impact their bottom line.

## What's your recommendation on model selection for RAG applications?

There's no one-size-fits-all model recommendation for RAG. Start by getting your retrieval right, as reasoning over data you can't find is a bigger issue than reasoning capabilities. Then run evaluations to test different models against your specific use cases. Consider your budget constraints across three dimensions: cost, latency, and performance. Your choice will depend on the economic value of the application - a financial analysis tool might justify using GPT-4 at $4 per report if it's still cheaper than human analysis, while a nutritionist website chatbot might need a more cost-effective model.
