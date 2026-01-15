---
title: RAG Office Hours Q&A Summary - Week 5
date: "2024-02-12"
cohort: 2
week: 5
session: 1
type: Office Hour Summary
description: Complex Excel file handling, SQL generation techniques, and embedding fine-tuning with Linear Adapters
topics:
  - Excel File Processing
  - Multi-Table Data Handling
  - SQL Generation
  - Embedding Fine-Tuning
  - Linear Adapters
  - Data Validation
---

# RAG Office Hours Q&A Summary - Week 5

---

## How should we handle Excel files with multiple sheets and tables?

Handling Excel files with multiple sheets and tables is challenging, and few companies have solved this problem well. Experience from companies like Zapier shows that connecting Excel spreadsheets to automation tools requires many controls to work properly.

The recommended approach is implementing checks on uploads to ensure files meet certain criteria. These checks might verify if an Excel file contains a single table, stays within size limits, or contains a specific number of tables. For simpler Excel files with single tables, implementing validation and processing works well, but for more complex files with multiple sheets and scattered tables, exporting to PDF might allow for easier parsing.

It's important to segment data and build specific extractors based on the data's structure. Single-table files can go through a dedicated single-table pipeline, while multi-table files might work better through a PDF parsing pipeline.

The most practical advice is to focus on simpler problems first and reject more complex ones until better solutions are developed. Solving for the simpler 40% of use cases while avoiding the most complex scenarios can be an effective strategy. Exporting Excel files as CSVs might also provide better compatibility with processing tools in many situations.

## What tools are recommended for SQL generation?

[Claude Sonnet](https://www.anthropic.com/claude) has proven effective for generating SQL queries. Success depends heavily on whether the system can retrieve the correct tables and CREATE statements.

The key to successful SQL generation is having good descriptions and CREATE statements for the tables, as well as ensuring that embedding and search capabilities properly identify the right tables when needed.

A recommended approach from [Timescale](https://www.timescale.com/blog/enhancing-text-to-sql-with-synthetic-summaries) involves first retrieving relevant tables, then retrieving pre-existing, approved SQL snippets. When both the correct tables and appropriate SQL patterns are in context, the generation process becomes significantly more reliable.

The complexity increases with many tables and columns, but focusing on retrieving the correct tables first, then incorporating approved SQL snippets to guide the generation process creates a two-step approach that significantly reduces errors in SQL generation.

## What is the Linear Adapter for embeddings and how does it work?

Linear adapters provide a cost-effective way to fine-tune embeddings. An embedding model takes data and produces a vector, with the dot product of two vectors indicating how similar they are. A linear adapter learns how to "rotate" these vectors slightly to better align with specific queries.

The approach is very economical - if a vector has 500 dimensions, the linear adapter is just a 500 by 500 matrix that multiplies the original vector. This allows for significant improvements in embedding quality with minimal computational cost.

Linear adapters can be compared to [LoRA (Low-Rank Adaptation)](https://research.trychroma.com/embedding-adapters), but with key differences. LoRA works between many layers of a neural network, while a linear adapter works only at the end. Additionally, linear adapters can be applied to pre-trained embeddings like those from OpenAI without needing access to the original model weights.

This approach enables domain-specific adaptations - for example, creating different adapters for marketing versus sales questions, or specialized adapters for legal, marketing, or tax information. The cost benefit is significant - training a linear adapter typically costs around $12 and can be done quickly, making it much more accessible than full model fine-tuning.

Implementation uses the standard fine-tuning process with triplets (question, positive example, negative example), but specifically changes the embedding function for the query. This rotation of vectors into more effective alignments can significantly improve retrieval performance for domain-specific applications.

## How does partitioning work in retrieval systems?

Partitioning in retrieval systems refers to how data is organized and segmented, rather than being about individual users. In applications like Cursor, a "user" might represent a documentation page. When working with code libraries like Requests, there might be a dedicated Requests index that multiple users access, but the partition is organized around the library package, documentation URL, or codebase.

Similarly, in applications like Notion, a "user" isn't an individual email account but represents an entire workspace. This means a company's complete Notion workspace would exist in a single index.

An interesting approach is partition-specific fine-tuning, which involves using different models to embed questions versus text chunks. Standard fine-tuning uses one model for both the question and the text chunk, but it's possible to fine-tune only one side of that equation. This might involve using the same model to embed all text chunks but having a different model to embed the question.

This technique proves particularly valuable in e-commerce settings. One embedding model might identify products that are similar to each other (creating a "similar products" carousel), while a different embedding model could identify complementary products (for "frequently bought together" recommendations). Both embeddings operate on the same product data but serve different retrieval purposes.

## What is the Model Context Protocol (MCP) and how does it differ from regular APIs?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/concepts/prompts) functions like a USB-C connector for AI systems - it's about standardization rather than specific functionality. While the devices that connect may vary, the connection method itself is standardized.

The key advantage of MCP over traditional APIs is the separation of client from backend. With a REST API, developers must write specific code to interact with each API, and each application needs to implement these integrations. MCP creates a standardized way for different systems to connect without custom code for each integration.

Consider an enterprise example where different teams might focus on different functionality: one team might work on email search while another builds CRM search tools. Both teams can develop their MCP clients, and the chatbot developers can easily integrate both without writing extensive custom code for each system.

This standardization means tools built with MCP work across multiple platforms without requiring custom integration code. A tool that works in one MCP-compatible environment (like Cursor) will work in others (like Claude desktop app) with minimal additional effort.

Beyond just function execution, MCP also supports resources and prompts. The MCP developer can provide not just functionality but also the prompts needed to use that functionality effectively. This means client developers don't need to write their own prompts for common operations like summarization or action item extraction.

This approach significantly reduces the need for glue code and allows developers to build applications without having to own all the client code, making integrations between different AI systems much more seamless.

## What AI tools are recommended for daily work?

[Claude Code](https://claude.ai/code) stands out among AI coding tools, particularly for its high-quality prompts. What makes it exceptional is how it handles context and continuity - when asked to write blog posts, it will first analyze existing posts to create a style guide, then reference that guide for every new post it writes.

This attention to existing style and consistency makes it particularly effective for content creation tasks. Despite higher costs compared to some alternatives (potentially $100+ per weekend of heavy use), many users find the value justifies the expense, with some noting they would willingly pay $200 monthly for the service.

For report generation specifically, there's significant value in tools that provide templated outputs. Ideally, a report generation tool would allow users to standardize formats across different reports - ensuring all market analysis reports follow the same structure, or that candidate evaluation reports maintain consistent formatting rather than varying significantly in format and depth.

This points to a broader trend in AI tool development - the need for tools that not only generate content but do so in consistent, predictable formats that align with existing workflows and style guidelines.

## What approaches are being used for multimodal applications?

Multimodal applications combining vision and language models are expanding into specialized domains. One example shared during office hours involved food image analysis, where a system extracts structured data from food photographs.

These systems can identify cuisine type, restaurant information, nutritional content, and dietary characteristics like whether a food item is vegan. While acknowledging practical limitations ("there's a limit to what you can effectively do"), early experiments show promising results in extracting valuable information from visual food content.

This example demonstrates how multimodal AI applications are moving beyond basic image recognition to extract detailed, structured information from visual content. The integration of vision models with language models allows systems to interpret and categorize visual information in ways that support practical applications like dietary tracking or restaurant recommendations.

## What emerging trends should we be aware of?

Report generation emerges as a particularly important trend in AI applications. The ability to automatically generate structured reports from unstructured data represents significant economic value for organizations.

Structured outputs and templates are increasingly valuable, especially for business use cases where standardized formats are essential. The ideal scenario allows for consistency in outputs - ensuring all reports of a specific type follow the same structure rather than varying significantly in format and organization.

Several organizations are developing report generation capabilities for internal use, with teams requiring standardized reports on a regular basis. This trend spans multiple industries, with financial due diligence being one area where automated report generation from multiple PDF sources shows particular promise.

The growing importance of fine-tuning approaches for handling data from multiple teams or domains also represents a significant trend. As organizations deploy AI systems across different business units, finding ways to effectively fine-tune models while maintaining performance becomes crucial.

Report generation capabilities demonstrate how AI can move beyond simple question answering to create significant economic value through structured information synthesis - transforming unstructured data into formatted reports that follow organizational templates and standards.

## How should we handle retrieval across multiple queries in a conversation?

When handling retrieval across multiple queries within the same conversation, the simplest approach is often the most effective. Using function calls for retrieval is recommended, where each call retrieves text chunks and includes information in XML format specifying the context and the question.

A key consideration is whether to keep retrieved context from previous queries in the message history for subsequent queries. This requires careful balancing, as including all previous context can consume tokens quickly.

The recommended practice is to prompt the retrieval system to always generate fully specified queries. For example, if the first question is "Where does Jason live?" and the follow-up is "What's the population of that city?", the retrieval system should be prompted to expand the second query to "What is the population of New York City?" This approach can be implemented through few-shot examples that demonstrate how to handle conversational context.

This strategy works because the model has access to both the previous question and answer in its context, allowing it to formulate complete, self-contained queries even when the user's input is ambiguous or relies on previous context.

An additional benefit of this approach is the ability to generate follow-up questions based on retrieved content. For instance, if a text chunk mentions that "Jason lived in different places when he was younger versus when he was older," the system can suggest follow-up questions like "Where did Jason live when he was younger?" This not only improves information discovery but also demonstrates to users that there are more interesting questions they could ask about the topic.

## What innovations are happening in memory and context management for agents?

Recent innovations in memory and context management for agents focus on creating more dynamic and self-improving systems. Rather than simply saving memories at the end of a conversation, newer approaches incorporate real-time memory creation and utilization during interactions.

Frameworks like Letta are incorporating self-editing capabilities alongside memory management. This integration allows agents to refactor their understanding and approach during a conversation rather than only learning from interactions after they conclude.

Implementation of these advances requires significant infrastructure changes, as memory layers affect prompt construction and the overall flow of agent interactions. The approach involves creating memories as the conversation progresses, which in turn influences the prompts used in subsequent exchanges.

With newer models like Claude 3.7 Sonnet, there's a shift in how tools are used by agents. Similar to the adaptation period seen with Claude Opus or GPT-4, these models require different prompting patterns to effectively utilize tools. Studying how systems like Cloud Code implement their tool use can provide valuable insights for optimizing agent performance with newer models.

## What can we learn from examining Cloud Code's approach to prompts and tools?

Cloud Code's approach to prompts and tools offers valuable insights for designing effective AI systems. Analysis of the Cloud Code source (extracted from their minified code) reveals highly detailed and carefully structured prompts for various tools.

Cloud Code implements a robust workflow where tools are designed to work together cohesively. Each tool's prompt contains specific instructions about how to use other tools first, creating a well-defined sequence of operations. For example, tools may include instructions to check for uniqueness before proceeding or to use specific validation approaches.

The prompts are remarkably detailed, with extensive instructions for common operations. For instance, batch tools contain comprehensive guidelines just for creating pull requests. This level of specificity helps ensure consistent and reliable performance.

Another notable aspect is Cloud Code's implementation of specialized tools like a notebook tool, showing how diverse functionality can be incorporated into a unified system. The prompts demonstrate that Claude is capable of working with numerous tools simultaneously when the system is properly designed.

This examination highlights the importance of thoughtful prompt engineering and tool design in building effective AI systems. By providing clear, detailed instructions and establishing well-defined workflows between tools, systems can achieve more reliable and sophisticated functionality.

## How can we create automated evaluation reports and insights for RAG systems?

Automated evaluation reports can significantly enhance RAG system development by providing structured insights and clear next steps. A comprehensive approach involves building a pipeline that:

1. Takes validation datasets and runs them through the RAG system
1. Computes various metrics (correctness, citation accuracy, URL validity)
1. Generates visualizations segmented by topic, question type, and other dimensions
1. Uses LLMs to analyze metrics and provide insights
1. Creates recommendations for system improvements

The reports can include both detailed analyses (15+ pages) and condensed slides for easier consumption in meetings. Key components include:

- Methodology explanations for stakeholders who may not be familiar with technical details
- System architecture diagrams
- Performance visualizations broken down by different segments
- Statistical analysis of which topics or question types perform well or poorly
- LLM-generated insights that explain patterns in the data
- Specific recommendations tied to the codebase

This process can effectively "close the loop" in the flywheel of development by identifying specific areas for improvement. For example, the system might recommend improving schema handling, enhancing retrieval tools, or adding more data for underrepresented topics.

The insights generated by LLMs analyzing the metrics can often align well with developer intuitions, but having them formally documented provides better clarity for prioritization and communication with stakeholders. These insights can be directly translated into development tickets, creating a streamlined workflow from evaluation to implementation.

The ultimate goal is to use these insights to guide the next iteration of development, run the evaluation again, and continue improving through this structured feedback loop.

## What strategies help maintain context when working with AI coding assistants?

When working with AI coding assistants like Cursor, maintaining context throughout a development session can be challenging. Several effective strategies have emerged from practical experience:

Creating and maintaining to-do lists within project documentation provides an effective way to preserve context. By instructing the AI to update the to-do list after completing each task, you ensure the progress remains in context for subsequent interactions. This approach creates a record of what has been accomplished and what remains to be done.

Templates within documentation files help maintain consistent structure across generated content. For example, having templates for different documentation sections ensures the AI follows established patterns when creating new content. This approach allows you to simply prompt the AI to "make more concept templates that generally look like this," maintaining consistency through visual examples rather than complex rules.

For more structured workflows, some developers create development plans using models like Claude Opus, which provide a roadmap the AI can follow. This helps prevent the AI from getting lost during implementation or going down unproductive paths.

Many developers find that keeping AI agents away from certain code areas (particularly tests) helps maintain structure. This can be accomplished either through explicit instructions or by adding files to the cursor.ignore configuration, which prevents them from being indexed while still allowing the AI to run commands like pytest.

Follow-up prompts at the end of interactions help maintain momentum. By asking what else needs to be done or what the next steps are, you encourage the AI to reference the to-do list and continue working on remaining tasks, creating a more cohesive development experience.

---

