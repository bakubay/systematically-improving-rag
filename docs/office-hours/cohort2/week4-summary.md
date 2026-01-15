---
title: RAG Office Hours Q&A Summary - Week 4
date: "2024-02-05"
cohort: 2
week: 4
session: 1
type: Office Hour Summary
description: Advanced segmentation strategies for RAG systems, conversation analysis techniques, and user behavior pattern identification
topics:
  - Customer Segmentation
  - Query Pattern Analysis
  - Conversation-Level Analysis
  - Tool Building Strategy
  - Data-Driven Insights
  - Intent Discovery
---

# RAG Office Hours Q&A Summary - Week 4

---

## What can "segments" mean beyond query volume and satisfaction values in a RAG system?

Segmentation really depends on who your customers are and what they're trying to do. With a generic chatbot, it's hard to figure out what segmentation means. But if you think about intents of a specific application, you can uncover different patterns.

For example, with a nutrition company chatbot, you might discover segments within product search – different capabilities around understanding deliveries, rescheduling, recurring orders, etc. Data analysis helps figure out what's important to build for the customer.

In a construction context, we found segments around:

- Users inputting specific project IDs (e.g., "Tell me about RFC 1257")
- Questions about time windows ("What do I have due today?" or "What's happening this week?")
- Counting items in documents

The goal of segmentation is to help you figure out what new function tools to build and what workflows might be viable. Another example: for a product that takes screenshots of users' computers, we found 10% of customers asking "How much time did I spend in this application?" That's impossible to answer with just screenshots, but we realized we had a Postgres database of all screenshots with timestamps, so we built a specific tool to query, group, and sum that data to answer the question.

The key is to find external understanding of your data – what are you worried about, and if you discover certain properties, what can you do about it?

## How should we approach segmentation for chatbots where the output is a whole conversation rather than just a query response?

If you have the compute resources, do similar classification and segmentation over your conversations. You'll uncover different insights beyond just tools.

When analyzing queries alone, we're basically asking how well we can execute tools to answer in one generation. By analyzing conversations, we might find segments that tell us:

- Users think the chatbot talks too much or not enough
- Users are frustrated with responses
- Common patterns in how conversations progress

The general idea is to gain an external understanding of your data – what properties are you concerned about, and if you discover X% of conversations have a certain property, what action can you take?

For example, if you find many users asking the language model to rewrite answers in their own words, should that be part of your system prompt? Analysis might show only 10% want tone matching, while most users actually prefer the AI voice.

## What approaches do you recommend for topic clustering, and have you tried using thinking models to generate clusters?

I generally use what I call "old school" approaches – K-means and DBSCAN. I typically start with the default settings in BERTTopic, which has been very good. The topic modeling goal isn't to uncover topics for production use but to do data analysis that helps you understand your data better.

For example, I might take Ada 2 embeddings, use K-means to pick 10-30 clusters, and look at 100 questions per cluster. That might take 2-3 days but teaches you a lot about your data. It's rarely the case that you run topic models and can just use them directly in your business.

When working with thinking models for clustering, I still do the initial clustering first because I might have 20 million questions to analyze. I'll cluster that data, find good and bad examples across clusters, and put that into Claude 3.7 or similar models, asking them to:

- Name each cluster
- Provide a short description
- Give good examples of what belongs in the cluster
- Provide nuanced examples of what's not in the cluster

This produces a YAML file that I can then use for classification. The language model helps expand our understanding, especially when we can't easily enumerate all possibilities ourselves.

## What are your thoughts on chunk size and chunk overlap? Is it worth trying out different chunking strategies?

I generally use 800 tokens with 50% overlap, which is what OpenAI recommends in their blog posts. In my experience, chunking strategies rarely make a significant difference compared to other improvements.

There's only a small subset of questions where chunk size makes a difference – you would need a question that can only be answered by a paragraph where two concepts are exactly 500 tokens apart. Performance gains usually come from better re-ranking, contextual retrieval (where you rewrite text chunks given the entire document), or better filtering and metadata capabilities.

I've rarely seen chunk size be the 10% improvement win – it might be a 1-2% improvement, which could just be noise. I would focus more on contextual retrieval if you have the compute budget for it.

For semantic chunking (using an LLM to determine good chunking points), I'm actually pretty convinced that contextual retrieval is better than dynamically chunking. The real question is whether you need to cite things word-for-word (in which case you shouldn't rewrite chunks) or if you just need general question answering.

I'd always spend more compute upfront to improve data quality. For example, I worked with a company doing Brazilian tax law with 50 documents, each 600 pages long. I asked, "Why are you only spending 70 cents to process this PDF? Why not spend $30?" If you're processing billions of dollars through the system, you should invest in good ingestion.

## What strategies can improve experimentation speed when working with RAG systems?

If you feel like you're not running enough experiments, focus on improving your infrastructure:

1. **Write parallelized code**: Many teams are still doing all their tests using for loops. Spending 1-2 hours learning to write parallelized code can dramatically reduce your experimentation time, going from days to hours. Using tools like multiprocessing to hit multiple endpoints simultaneously is much better than having code break on iteration 2,000.

1. **Improve data access and understanding**: Document how to query your data effectively. It's a waste of time if you write a query to prepare data, and someone comes back a day later saying, "That's wrong, we actually need to include only last week's data."

1. **Build modular pipelines**: If your entire RAG application is a giant Python file, it will be hard to test. But if each search index is a separate POST request, you can test them individually. This allows you to focus on one component (like an image retriever system) and improve it from 30% to 80% accuracy in one afternoon before integrating it back into your router.

1. **Test locally when possible**: Create smaller synthetic datasets for quick iteration before running larger tests.

Being able to test components in isolation is crucial for rapid experimentation. A lot of this comes down to good software engineering practices and thoughtful system design.

## How do you handle multiple languages in a RAG system, especially when job titles may be similar but written differently across languages?

For multilingual challenges like job titles across different languages, I recommend two approaches:

1. **Metadata extraction and filtering**: Build classifiers to add more metadata to your ontology. For example, "software engineering recruiter" and "software engineer" go into two different classes, allowing you to filter for one and not the other. This improves search precision.

1. **Fine-tune embedding models with triplets**: Create a dataset with examples like "software engineer" (query), "python developer" (positive example), and "software engineering recruiter" (hard negative). This teaches your model to separate similar-looking job titles that have different meanings.

For handling multiple languages, run tests to see whether translation improves performance. For instance, does your classifier perform better if you translate everything to English first, or if you use the original languages? If translating provides only a 1-2% improvement but requires complex infrastructure to maintain, it might make sense to accept slightly lower performance.

If you lack training data for certain languages, consider using synthetic data creation. Use $2,000 of API credits to generate examples that cover edge cases in your domain, like distinguishing between "real estate developer" and "python developer" across languages.

## What are your thoughts on vision RAG, and what databases would you recommend for multimodal embeddings?

Vision RAG isn't talked about as much because it's more expensive and most of the important data is typically in text. That said, there are valuable use cases – like a company that does RAG over video clips to help movie producers find content, using Gemini Flash to describe what's happening in scenes.

For databases, I'd recommend looking at:

- ChromaDB
- LanceDB
- TurboBuffer (used by Notion and Cursor)
- PgVector with Scale (for relational data with many reads/writes)

However, I'm finding that pure multimodal embeddings aren't always the best approach anymore. Often it's better to generate a text summary of the image data. For example, when trying to embed images and text in the same space, CLIP embeddings often work worse than just doing image captioning and then embedding that text.

In week 5, I'll talk more about this – there are many things you can't do with multimodal embeddings. They're trained mostly with caption data, which limits their capabilities for certain tasks.

## What are your experiences with the Model Context Protocol (MCP) and how might it change RAG systems?

MCP is becoming increasingly important because it allows different systems to connect with each other. When you own all the code, you don't really need MCP since you can just use function calling. But the ability to connect different systems is very compelling.

Some interesting examples of MCP usage:

- Having an MCP server in Cursor to do image generation while building a video game
- Creating an MCP server to access network logs for debugging web applications
- Building MCP servers that connect to production databases so Cursor can understand your schema and write SQL
- Setting up an MCP server that writes conversation notes to Notion automatically

What makes MCP powerful is that it standardizes these integrations and reduces boilerplate code. The protocol founders explain that it's easy to integrate with other servers when building your own client or server. Instead of rebuilding connectors with databases or services, you can reuse patterns and implementations.

Claude 3.7 with Claude Code, for instance, has impressive agent functionality using MCP. It features better context management through commands like "/compact" which summarizes conversation history effectively without bloating the context window.

## How can we use synthetic data generation for summarization tasks?

There are many creative ways to generate synthetic data. For summarization, you can:

1. **Create reverse tasks**: For example, start with the outcomes you care about (like action items) and ask an LLM to generate a transcript that would produce those items. Then you can verify if your summarization system correctly extracts the original action items from this synthetic transcript.

1. **Use data augmentation techniques**: Look at techniques from other domains like speech detection, where researchers combine clean audio samples to create more complex scenarios (like overlapping speakers). You can apply similar principles to text.

1. **Apply transformations similar to image processing**: In computer vision, we've long used techniques like converting color photos to black and white, then training models to predict the original colors. Similarly, we convert high-resolution images to low-resolution and train models to predict the original resolution. We can apply similar transformations to text data.

The key is to think about ways to go from your desired output backward to input data, or to systematically transform existing data in ways that preserve the information you care about while changing other aspects.

## When using structured outputs with few-shot prompts, should the examples use the exact same JSON schema or can they be plain text?

I would almost always try to keep the JSON format consistent in your few-shot examples. This is somewhat superstitious, but I feel like the attention mechanism will always attend better to similar tokens.

The schema itself is probably not what's going to break things these days. More likely, problems will arise from unintended properties of your examples. For instance, if all your action items in the few-shot examples are very short (under 4 words), your outputs will tend to be very short too. The examples communicate that these properties are correlated.

I'd rather keep everything in JSON because there will be other random issues that come up. The only caution is to make sure you have checks in place so that when the language model has nothing in the context, it won't just automatically recite the few-shot examples.

For complex contexts (like insurance claims that require understanding policies and forms), if including the context for each few-shot example would make your context window explode, consider few-shotting the thinking more importantly. Show examples of the reasoning process: "I noticed the customer said they had 28 people, and our pricing page has different pricing for teams with less than 30 employees, so I'll use that pricing tier and mention they could get a better price with more employees..."

## How do you approach RAG when you have transcripts or unstructured text without clear paragraph markers?

For transcripts without clear paragraph markers, a few approaches work well:

1. **Use diarization models** to get speaker tags, which can serve as natural boundaries (each dialog line becomes a chunk)

1. **Detect silences** in the audio and chunk on those silences

1. **Consider the structure of your content** - for instance, if it's an interview format, you might know it's always question-answer pairs, so you can embed those pairs together

It ultimately depends on your specific use case. For a general conversation, chunking on silences or using diarization with a sliding window over dialog will work. For job interviews or expert interviews, understanding the structure (question followed by answer) lets you optimize your chunking strategy.

If you have mixed domains and raw transcripts without access to the original source, you might need to default to generic approaches like 800 tokens with 40% overlap, then rely more on contextual retrieval techniques.

## What are your recommendations for building slide presentations with AI tools?

I've been using AI tools to build academic-style slides with LaTeX and Beamer. My process is:

1. Load all relevant content into Cursor (in my case, all 6 hours of course transcripts)
1. Create an outline for the presentation
1. Use Claude to extract key case studies and insights from the transcripts
1. Have the LLM generate slides using LaTeX Beamer format
1. Use a simple auto-compiler (built with Watchdog) that recompiles the PDF whenever the file changes

The advantages of this approach:

- You can create both slides and a detailed document from the same source
- The LLM can generate diagrams using TikZ (a graphics creation library)
- Everything is vector-based so it looks clean at any resolution
- You can have the LLM add callouts, highlights, and formatting

This approach lets me essentially talk to my slides and have them update in real-time. I can say "make this section shorter" or "add an example about X" and see the changes immediately in the PDF preview.

For those who prefer different formats, you could also try reveal.js for web-based presentations. The key is finding a workflow that lets you focus on content while the AI handles formatting and details.

## How do AI coding tools compare (Claude Code, Aider, Cursor, Windsurf)?

There's been significant evolution in AI coding tools, with different strengths and approaches:

- **Claude Code** has impressive agent functionality with excellent context management. It features a "/compact" command that summarizes conversation history effectively without bloating the context window. Some users report it's more capable than Cursor for certain tasks, particularly with how it handles context and managing complexity.

- **Aider** is a CLI-based tool that gives very low-level control over the files you can edit. It's open source and allows granular control over which models you use at specific points. Some users have migrated from Cursor to Aider due to its flexibility, though it has a steeper learning curve.

- **Cursor** is widely used for its UI and integrations. It works well for incremental changes to code and has good MCP integrations, but some find its context management becomes less effective over time on complex projects.

- **Windsurf** is particularly good at handling projects with good requirements and system design. It excels at context management over time and keeping track of multiple files in a repository. It's especially valuable for staff engineers and system architects who start with clear system designs.

The key differentiation often comes down to context management - how well the tool maintains an understanding of your entire codebase and project requirements as you work. For complex projects, tools that help document the goals and requirements (like adding branch goals in comments) tend to perform better.

## How do you use Deep Research and other search tools effectively?

Different search tools serve different purposes depending on context:

- **Claude's Deep Research** works well for technical documentation, business-level competitive analysis, and generating comprehensive memos. Its tone is particularly well-suited for business communications that need minimal editing. Many users leverage it to materialize blog posts or analyses they want to read (e.g., "Write me a blog post on why someone should look at MCP versus just using the Open API spec").

- **Grok's Deep Search** has different strengths, with some users preferring it for timely news or quick research questions. Interestingly, usage patterns often split between mobile (Grok) and desktop (Claude/OpenAI) platforms based on when and where research is being done.

- **Perplexity** offers another approach to deep research, useful for generating product specs and learning resource reports, especially for colleagues without AI engineering backgrounds.

The quality of these tools has advanced to the point where they can effectively replace traditional research methods for many use cases, saving significant time for competitive analyses and technical investigations.

## What makes Lovable stand out for no-code app generation?

Lovable has emerged as a powerful tool for no-code app generation:

- It excels at creating fully functional applications with modern UIs from scratch, going beyond simple prototypes to production-ready systems
- Its deep integration with Supabase provides authentication, real-time features, and database capabilities out of the box
- Every code change gets pushed to GitHub, allowing developers to fix issues locally in tools like Cursor or Windsurf when needed
- Each commit creates a preview deployment on Cloudflare, streamlining the development and testing process
- The tool can implement complex features like row-level security, push notifications, and real-time commenting systems using websockets

Users report that Lovable outperforms alternatives like V0 and Bolt for creating complete applications, though it can be expensive ($200+ for complex projects). The tight integration with Supabase is particularly valuable, with many users becoming paid Supabase customers after using Lovable to build their applications.

## What emerging techniques are promising for handling long documents in RAG?

Handling long documents effectively is still evolving, with several promising approaches:

1. **Hierarchical retrieval**: Create summary or header-level embeddings for entire documents/chapters, then more granular embeddings for sections/paragraphs. This allows multi-stage retrieval that narrows down from document to specific passages.

1. **Graph-based approaches**: Build knowledge graphs connecting concepts across documents, enabling retrieval that follows conceptual relationships rather than just lexical similarity.

1. **Hybrid sparse-dense retrieval**: Combine embedding-based retrieval with keyword/BM25 approaches to capture both semantic and lexical matches, which is particularly valuable for documents with specialized terminology.

1. **Learning to rewrite**: Train models to rewrite retrieved chunks into more coherent contexts that preserve the key information while eliminating redundancy.

1. **Recursive summarization**: For extremely long documents, apply recursive summarization techniques that gradually compress information while maintaining key details.

Projects like LangChain's Document Transformer framework and repositories focusing on document processing show significant advances in these areas. The most effective systems often combine multiple approaches based on the specific characteristics of their document collections.

## How can I approach RAG for messy knowledge bases with duplicate documents?

When dealing with messy knowledge bases that contain duplicate or near-duplicate documents:

1. **Pre-processing pipeline**: Implement de-duplication strategies during ingestion. This could involve computing similarity scores between documents and merging or filtering based on a threshold.

1. **Metadata extraction and filtering**: Add more metadata to your ontology by building classifiers for different document types or topics. This allows you to filter for specific categories during retrieval.

1. **Query classification**: For ambiguous queries, implement both pre-retrieval and post-retrieval classification to identify query intent and determine when clarification is needed.

1. **Progressive disclosure**: Consider displaying intermediate results with summarized information about potential topics before generating a complete answer. This helps users navigate ambiguity, especially for queries that could refer to multiple topics.

1. **Dynamic presentation**: For high-latency requirements (e.g., responses needed in under 6 seconds), consider showing retrieved documents first while the full answer is being generated, allowing users to see some results immediately.

Remember that the goal isn't perfect retrieval but helping users find the information they need. Sometimes showing multiple possible interpretations of a query is more helpful than trying to guess the single "right" answer.

## When is it better to use DAGs versus agentic approaches?

For specific workflows with well-defined steps, DAGs (Directed Acyclic Graphs) often provide more reliable and predictable results than fully agentic approaches:

1. **Use DAGs when**:

   - The workflow has clear, sequential steps
   - You know the process is correct and just need to choose the right workflow
   - You're implementing established protocols (like therapy approaches or compliance processes)
   - Predictability and consistency are critical

1. **Use agentic approaches when**:
   - The problem space is exploratory
   - Tasks require adaptation to unpredictable user input
   - The workflow needs to evolve based on intermediate results
   - You need to handle a wide variety of open-ended requests

The distinction often comes down to control versus flexibility. DAGs provide more control over the exact process, while agentic approaches offer more flexibility but less predictability.

For example, in a therapeutic chatbot following an established CBT protocol, a DAG approach ensures the conversation follows the correct therapeutic sequence. However, for an open-ended research assistant, an agentic approach allows for more dynamic problem-solving.

## How do I create effective negative examples for training retrieval models?

Creating effective negative examples for training retrieval models involves several strategies:

1. **Hard negative mining**: Find examples that are semantically similar but actually irrelevant. For job listings, "software engineer recruiter" is a hard negative for "software engineer" queries - they look similar textually but represent different job categories.

1. **Top-K analysis**: Run retrieval with your current model, then have an LLM evaluate which results in the top K are actually irrelevant. These make excellent negative examples because they expose weaknesses in your current model.

1. **Controlled random sampling**: While pure random sampling provides some signal, it's often too easy for the model to distinguish. Instead, use controlled randomization that preserves some properties of the positive examples.

When working with triplet learning (query, positive example, negative example), the quality of your negative examples often has more impact on model performance than adding more positive examples. Focus on finding negative examples that are difficult to distinguish from positive ones.

For multimodal or multilingual applications, you may need to create synthetic data, especially for languages with limited training data. This can be done by using LLMs to generate examples that explore edge cases in your domain.

## What strategies can improve response time in RAG systems with tight latency requirements?

For applications requiring responses in just a few seconds:

1. **Progressive rendering**: Show retrieved documents first (which can be returned in 150-400ms) while the LLM generates the complete answer in the background. This gives users immediate results while they wait for the full response.

1. **Caching**: Implement aggressive caching for common queries. When a question-answer pair receives positive feedback (like being forwarded, shared, or rated highly), save it as a new document that can be quickly retrieved for similar questions.

1. **Response type classification**: Use a lightweight classifier to determine if a query needs full retrieval and generation or if it can be answered with a simpler approach.

1. **Contextual snippet generation**: During retrieval, generate quick summaries of each chunk that can be displayed alongside search results before the complete answer is ready.

1. **Parallel processing**: Run multiple retrieval strategies in parallel and combine the results, rather than using sequential processing that adds to the total latency.

The key insight is to avoid an all-or-nothing approach to response generation. By decomposing the process into steps that can be displayed incrementally, you can significantly improve perceived latency even when the complete answer takes longer to generate.

## What are your experiences with the Model Context Protocol (MCP)?

MCP (Model Context Protocol) is becoming increasingly important as it allows different AI systems to connect with each other:

1. **Key benefits**:

   - Standardizes integrations between AI systems
   - Reduces boilerplate code when connecting to different services
   - Allows models to access data and functionality they wouldn't normally have permission to use

1. **Practical examples**:

   - Image generation servers in Cursor for creating assets while building applications
   - Servers that connect to network logs for debugging web applications
   - Connectors to production databases that help models understand schemas and write SQL
   - Automation tools that write conversation notes directly to Notion or other note-taking systems

1. **Comparison to function calling**:
   - When you own all the code, function calling may be simpler
   - MCP becomes valuable when connecting separate systems with different permission models
   - Provides a standardized way to expose capabilities across different AI platforms

The protocol is still evolving but shows promise for creating more powerful AI systems by composing specialized components. Some implementations like Claude 3.7 with Claude Code demonstrate how MCP can enable better context management and more sophisticated agent capabilities.

## Key Takeaways and Additional Resources

### Key Takeaways:

- The goal of segmentation is to understand customer needs and determine what tools to build next
- Chunking strategy (800 tokens, 50% overlap) is rarely the bottleneck - focus on contextual retrieval instead
- For topic modeling, start with BERTTopic defaults and then use thinking models to better understand clusters
- Spend more compute upfront to improve data quality - particularly for high-value documents
- Write parallelized code to dramatically speed up experimentation
- For multilingual RAG, test whether translation improves performance enough to justify the added complexity
- Consider transforming image content to text summaries rather than using pure multimodal embeddings
- MCP is becoming increasingly important for connecting different AI systems together
- Use structured JSON consistently in few-shot examples rather than plain text
- For slide creation, AI tools can generate both content and formatting in vector-based formats
- For long documents, consider hierarchical retrieval, graph-based approaches, hybrid sparse-dense retrieval, learning to rewrite, and recursive summarization
- For messy knowledge bases, implement pre-processing pipeline, metadata extraction and filtering, query classification, progressive disclosure, and dynamic presentation
- For DAGs versus agentic approaches, use DAGs when the workflow has clear, sequential steps, and use agentic approaches when the problem space is exploratory
- For negative examples, use hard negative mining, top-K analysis, and controlled random sampling
- For response time, implement progressive rendering, caching, response type classification, contextual snippet generation, and parallel processing

### Additional Resources:

- BERTTopic: https://maartengr.github.io/BERTopic/index.html
- MCP Agent: https://github.com/lastmile-ai/mcp-agent
- Claude Code: https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview
- RepoPrompt: https://repoprompt.com/
- Aider CLI coding tool: https://aider.chat/
- Lovable for no-code app generation with Supabase integration
- Cursor and Windsurf for AI-assisted coding environments

---

