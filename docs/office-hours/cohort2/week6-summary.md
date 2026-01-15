---
title: RAG Office Hours Q&A Summary - Week 6
date: "2024-02-19"
cohort: 2
week: 6
session: 1
type: Office Hour Summary
description: Deep Research applications, long context vs RAG trade-offs, and the importance of human-labeled data for RAG systems
topics:
  - Deep Research
  - Long Context Models
  - Document-Level Retrieval
  - Human-Labeled Data
  - Report Generation
  - Model Fine-Tuning
---

# RAG Office Hours Q&A Summary - Week 6

---

## What is Deep Research and how does it relate to RAG?

Deep Research is essentially a model fine-tuned for tool use that leverages RAG and iteration in a loop to produce reports. It can be viewed as RAG with solid data sources and strong reasoning capabilities on top. Deep Research is distinct from standard RAG applications because it typically produces more comprehensive outputs like reports rather than just answering specific questions.

While Deep Research generates reports that might differ in structure between runs, more advanced approaches like those used by Vantage aim to create standardized, deterministic reports. The ideal approach is to define a specific structure for reports, particularly when you know exactly what questions need to be answered for your domain.

There's significant economic value in creating structured reports rather than just answering ad-hoc questions. For example, instead of building a system that allows recruiters to query interview transcripts individually, creating a standardized hiring report that distills key information from all interviews provides greater business value. This approach helps stakeholders make better decisions rather than just saving time on information retrieval.

The techniques taught in the RAG course are directly applicable to building Deep Research-style systems, particularly when focused on specific domains rather than general-purpose research.

## Should we use long context windows or RAG for complex questions?

Long context windows should be leveraged first when possible, as they generally produce better results than relying solely on chunk retrieval. The ideal approach is often to use document-level retrieval rather than chunk-level retrieval when working with long context models.

When faced with specific tasks that require processing lengthy documents (like generating pricing emails based on policy documents), consider creating dedicated tools that use the full context window rather than breaking documents into chunks. This can be implemented as a function that uses a very long prompt containing all relevant policy documents.

This approach simplifies the retrieval problem from needing good chunk retrieval to just needing good document retrieval, which can be accomplished with simpler techniques like full-text search. The decision becomes not about whether you have good chunk retrieval but rather if you have good document retrieval capabilities.

As models' context windows continue to expand, this approach becomes increasingly viable for more use cases, potentially reducing the complexity of some RAG implementations.

## How important is human-labeled data for RAG systems?

Human-labeled data remains essential for building high-quality RAG systems, though many teams underestimate its importance. Teams that are reluctant to invest in data labeling often struggle to achieve meaningful performance improvements.

From a consulting perspective, one effective approach is to demonstrate the impact of data quality through experimentation. Show how model performance improves with synthetic data, then demonstrate how it plateaus. This creates a data-driven argument that once synthetic data reaches diminishing returns, real human-labeled data becomes necessary for further improvement.

For high-value applications, the investment in human labeling is justified. Companies like Vantage, which produces due diligence reports for investment decisions, dedicate staff to labeling and evaluating the quality of question-answer pairs. This reflects the understanding that without at least one human producing high-quality data, systems will struggle to achieve meaningful differentiation in output quality.

The economic argument is compelling: if a model is helping make decisions that involve millions or billions of dollars (as in investment due diligence or hiring), the cost of high-quality human labeling is minimal compared to the value it creates.

## How do you handle model evaluation when generating reports rather than simple answers?

Evaluating report generation presents different challenges than evaluating direct question answering. While individual components can be measured with standard metrics, evaluating complete reports often requires human judgment against a defined rubric.

Language models can perform reasonably well as judges against a rubric, but they primarily assess whether all required elements are present rather than providing nuanced feedback on quality or analysis. Human evaluation remains important for assessing whether the analysis itself is valuable and meets business needs.

This challenge mirrors broader evaluation difficulties in the generative AI space, where outputs become more complex and subjective. The solution often involves creating clear rubrics for what constitutes a good report in your specific domain, then combining automated checks with strategic human review.

Teams should focus on defining what makes a report valuable to their specific users rather than pursuing generic quality metrics. This might involve understanding whether users need comprehensive information, specific recommendations, or particular formatting that helps with decision-making.

## What broader trends are emerging in AI consulting?

The AI consulting landscape is evolving rapidly, with several key trends emerging:

1. **Shift from implementation to experimentation**: More consulting work now involves helping teams design and run effective experiments rather than just implementing specific techniques. This includes teaching scientific methods, hypothesis formation, and systematic testing.

1. **Focus on data quality over algorithms**: Successful consultants emphasize improving data quality and data collection processes rather than just applying newer algorithms. Many organizations still lack basic data infrastructure for effective AI work.

1. **Organizational change management**: A significant portion of AI consulting now involves helping teams adapt to new workflows and develop the right skills. This includes teaching software engineers to approach problems more like data scientists.

1. **Economic value alignment**: The most successful AI implementations focus on creating decision-making value rather than just time savings. Products that help customers make better decisions (like hiring recommendations or investment analysis) can command higher prices than those that merely save time.

The role of consultants remains valuable even as AI tools become more accessible because they bring expertise in experiment design, data quality improvement, and aligning AI capabilities with business value.

## How will AI impact the consulting industry itself?

The consulting industry will continue to evolve alongside AI advancements, but consultants who adapt will remain valuable. The core value of consulting is increasingly about bringing expertise in scientific methods, data analysis, and business process transformation rather than simply implementing technology.

Several shifts are occurring in the consulting space:

1. **Distribution becomes more important**: Consultants who can effectively share their insights through content creation (blogs, videos, courses) will have advantages in attracting clients.

1. **Process expertise over pure technical knowledge**: As technical implementation becomes easier with AI tools, consultants who understand how to change organizational processes and workflows become more valuable.

1. **Organization and workflow design**: Consultants who can help structure workflows and processes that leverage AI effectively will remain in demand, even as some technical implementation work becomes automated.

1. **Connection to economic value**: Consultants who can clearly connect AI capabilities to business value and ROI will continue to thrive, focusing less on technology and more on business outcomes.

While AI will automate some aspects of consulting work, it simultaneously creates new opportunities for consultants who can help organizations navigate the complex landscape of AI implementation and business transformation.

## How should we handle training data contamination from AI-generated content?

As more content on the internet becomes AI-generated, concerns about training data contamination and potential "model collapse" are valid but may be overstated for several reasons:

1. **Unexplored modalities**: Even if text data becomes saturated with AI-generated content, there are many other modalities (video, computer interaction data, etc.) that remain largely untapped for training.

1. **Mode covering vs. mode collapse**: Advanced research at organizations like OpenAI focuses on developing models that can identify multiple solution modes rather than collapsing to the lowest-resistance path. Models that are "mode covering" can maintain diversity in their outputs even when trained on some low-quality data.

1. **Real-world data sources**: For many specialized applications, the most valuable data isn't from the public internet but from proprietary sources or human interaction with systems. This data remains largely uncontaminated.

1. **Post-training refinement**: Much of the current improvement in AI models comes from post-training techniques like RLHF rather than pre-training alone. This allows models to improve based on high-quality human feedback even if pre-training data becomes noisier.

OpenAI researchers reportedly maintain confidence that there's still significant high-quality data available, suggesting that concerns about running out of training data may be premature.

## What are emerging trends in AI tool development?

Several noteworthy trends are emerging in AI tool development:

1. **Advanced agents like Manus**: New tools like Manus are providing powerful capabilities by combining foundation models (like Claude Sonnet) with extensive tooling. While details are limited, these systems represent a new generation of AI assistants with enhanced capabilities.

1. **Cloud Code improvements**: Cloud Code has shown impressive performance for specific tasks, sometimes outperforming tools like Cursor for certain types of development work. However, success often depends on the user's expertise in the domain they're working in - users still need significant knowledge to effectively guide AI tools.

1. **Context management evolution**: Newer AI tools are improving how they manage context over time, creating better continuity between sessions and maintaining understanding of project requirements.

1. **Focus on expert augmentation**: The most successful AI tools are those that augment human expertise rather than trying to replace it entirely. Tools work best when users have clear goals and domain knowledge, with the AI handling implementation details.

Despite significant advances in AI capabilities, domain expertise remains crucial for effective use of these tools. The relationship between user expertise and AI capabilities creates a complex dynamic where both need to evolve together for optimal results.

## How will data collection evolve for AI applications?

Data collection for AI is shifting in several important ways:

1. **Purposeful logging**: Companies are moving beyond debugging-focused logging to capturing data specifically designed for model training. This requires engineers to think about what signals might be useful for future models rather than just for troubleshooting.

1. **Structured feedback collection**: More companies are implementing systematic ways to collect user feedback and interactions, recognizing these signals as valuable training data rather than just product metrics.

1. **Data quality over quantity**: There's growing recognition that having smaller amounts of high-quality, well-labeled data is often more valuable than vast amounts of noisy data.

1. **Economic value alignment**: Organizations are increasingly evaluating what data to collect based on economic value rather than technical feasibility alone. This means focusing data collection efforts on areas where improved model performance translates directly to business outcomes.

Many companies still struggle with basic data collection infrastructure, often lacking the systems needed to capture useful signals from user interactions. Building these foundations remains a critical first step before more advanced AI applications can be developed.

## How should we think about distribution and economic viability in AI products?

The most successful AI applications focus on creating decision-making value rather than just time savings. This fundamental shift in value proposition affects pricing, distribution, and product design:

1. **Value-based pricing**: Products that help customers make better decisions (like hiring recommendations or investment analysis) can command higher prices than those that merely save time. For example, recruiters charge 25% of a hire's salary not because they save time but because they help make better hiring decisions.

1. **Structured outputs**: There's increasing value in AI systems that produce standardized, structured outputs (like reports) rather than just answering ad-hoc questions. This creates more consistent value and makes the outputs more directly usable in business processes.

1. **Domain specialization**: Applications focused on specific domains with clear economic value (financial analysis, legal research, specialized technical fields) can support higher pricing than general-purpose AI tools.

1. **Content as marketing**: For many AI consultants and product builders, content creation (blog posts, courses, etc.) derived from their expertise serves as efficient marketing. This "sawdust" from their core work helps attract clients and build credibility.

The most economically viable AI products are those that align directly with high-value business decisions rather than just providing generalized capabilities or incremental efficiency improvements.

## What recommendations do you have for structuring the course and its content?

Several suggestions emerged for improving the course structure and content:

1. **Better content organization**: Ensure core videos and tutorials are prominently featured in the main menu rather than buried under multiple links. This would improve discoverability and help students stay on track.

1. **Standardized office hours format**: Implement a consistent format for office hours, with the first 10-20 minutes dedicated to setting context about the week's material before moving to questions. This helps orient participants who may be joining different sessions.

1. **Email reminders with direct links**: Send regular emails with direct links to the week's core videos and tutorials to ensure students know exactly what to watch and when.

1. **Calendar integration**: Consider adding placeholder calendar events for self-study time to help students schedule time to watch asynchronous content.

1. **Expanded coverage of enterprise tools**: While OpenAI tools were featured prominently for practical reasons, more coverage of enterprise platforms (Azure, AWS, Google Vertex) would be valuable for many students working in corporate environments.

1. **Open-source alternatives**: Include more examples using open-source tools alongside commercial offerings, especially for cases where data residency requirements make cloud services challenging.

The feedback emphasized that while the course content was valuable, improvements to structure and discoverability would help students manage the significant amount of material more effectively.

## How can we use Slack effectively after the course ends?

The Slack channel will remain available as an ongoing resource for students after the course concludes. Several recommendations for effective use include:

1. **Specific questions get better answers**: When asking questions in Slack, provide specific details about your use case, what you've already tried, and exactly what you're trying to accomplish. This allows for more targeted and helpful responses.

1. **Share real-world applications**: Sharing how you're applying concepts from the course to real projects provides valuable context for others and creates learning opportunities for everyone.

1. **Ongoing community learning**: The Slack channel offers an opportunity to continue learning from peers who are implementing RAG systems across different industries and use cases.

1. **Access to course materials**: All course materials will remain accessible through Maven, and the Slack community provides a way to discuss those materials as you continue to review them.

The instructor emphasized that students will get as much value from the community as they put in through specific, thoughtful questions and sharing their own experiences.

## What future trends do you anticipate in AI development?

Several key trends are likely to shape AI development in the near future:

1. **Structured output generation**: The ability to generate consistent, structured reports and analyses will become increasingly valuable, particularly for business applications where standardized formats are essential.

1. **Report generation workflows**: Building on the structured output trend, more sophisticated workflows for generating comprehensive reports from multiple data sources will become mainstream.

1. **Scientific approach to AI development**: Organizations that adopt rigorous experimentation, hypothesis testing, and data analysis will pull ahead of those that simply implement the latest techniques without careful evaluation.

1. **Economic alignment**: AI applications that directly support high-value decision making will see stronger adoption and commercial success compared to those that merely provide incremental efficiency improvements.

1. **Integration of multiple modalities**: While still evolving, the ability to reason across text, images, video, and interactive data will create new application possibilities, though many practical applications will still focus on extracting structured information from these inputs rather than general understanding.

The most successful organizations will be those that develop systematic processes for continuous improvement of their AI systems rather than chasing the latest models or techniques without a clear evaluation framework.

## How do you balance providing generic AI solutions versus domain-specific implementations?

The balance between generic AI solutions and domain-specific implementations depends on both economic factors and technical feasibility:

1. **Start with domain specificity**: Focusing on specific domains allows for more valuable outputs, better evaluation, and clearer value propositions. This approach makes it easier to create systems that provide significant value.

1. **Specialize by intent rather than content**: Even within a domain, segmenting by user intent (what they're trying to accomplish) rather than just content type creates more focused and effective solutions.

1. **Economic viability**: Domain-specific solutions can often command higher prices because they solve specific high-value problems rather than providing general capabilities. This makes them more economically viable despite smaller potential market size.

1. **Technical feasibility**: Creating effective general-purpose AI systems remains technically challenging, while domain-specific implementations can achieve high performance by narrowing the scope of what they need to handle.

For most organizations building AI applications, starting with a specific domain and set of well-defined use cases is likely to produce better results than attempting to build general-purpose systems. This focus allows for better data collection, more effective evaluation, and clearer alignment with business value.

## Key Takeaways and Additional Resources

### Key Takeaways:

- Deep Research can be understood as RAG with high-quality data sources and strong reasoning capabilities
- Structured reports often provide more business value than ad-hoc question answering
- Long context windows should be leveraged first when possible before falling back to chunking
- Human-labeled data remains essential for high-quality RAG systems, especially as systems reach the limits of improvement from synthetic data
- Evaluating report generation often requires human judgment against defined rubrics
- AI consulting is shifting toward experimental design and process transformation rather than just implementation
- Data collection is evolving to focus more on purposeful logging and structured feedback collection
- The most economically viable AI products align with high-value business decisions rather than just providing efficiency improvements
- Content organization and standardized formats for course materials can significantly improve the learning experience
- Domain-specific AI implementations typically provide better economic and technical outcomes than general-purpose solutions

### Additional Resources:

- [The Future of RAG](https://jliu.net/blog/future-of-rag) - Jason Liu's blog post on where RAG is heading
- [Deep Research](https://openai.com/index/introducing-deep-research/) - OpenAI's introduction to Deep Research
- [Vantage](https://www.vantage.co/) - Company mentioned as an example of advanced report generation
- [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) - Latest model referenced in discussions
- [Cloud Code](https://cloud.google.com/code) - AI coding tool discussed in the sessions
- [Manus](https://manus.ai/) - Emerging AI agent mentioned in the discussions

---

