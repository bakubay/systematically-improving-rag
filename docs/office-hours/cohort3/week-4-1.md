---
title: Week 4 - Office Hour 1
date: "2024-06-10"
cohort: 3
week: 4
session: 1
type: Office Hour
transcript: ../GMT20250610-160018_Recording.transcript.vtt
description: Practical RAG implementation, data management strategies, and AI business models and value capture
topics:
  - Evaluation Data Collection
  - Model Selection for RAG
  - Visual Elements in Reports
  - Managing AI Expectations
  - Open-source AI Tools Vision
  - Pricing and Value Capture
  - AI Platform Integration
  - Business Model Transformation
  - Future AI Development Data
---

# Week 4, Office Hour 1 (June 10)

Study Notes:

I hosted an office hours session focused on practical RAG implementation challenges, data management strategies, and the business aspects of AI consulting. Here are my insights on handling evaluation data, choosing the right tools, and thinking strategically about AI business models and value capture.

---

---

## How should we approach evaluation data collection for RAG systems?

One participant was struggling with exporting conversation data from Langsmith for evaluation purposes. This highlights a common challenge with many tracing tools - they're often better at collecting data than exporting it in useful formats.

When Langsmith or similar tools create export difficulties, I recommend two alternative approaches:

1. Direct database storage: Instead of relying on tracing software, consider saving queries directly to your database as the application runs.
   "This is something we do all the time - we just write the question, answer pairs, or chunks to Postgres. That way, we can build UI on top of that database rather than trying to export data out of tools like Langsmith."
1. Create a simple, wide database table that includes:
   - Session ID
   - User ID
   - Query text
   - Retrieved chunks
   - Generated answer

This approach gives you direct access to your data without depending on third-party export functionality, which can be unreliable. It's like building your own analytics system rather than trying to export from something like Data Dog for analysis.

**_Key Takeaway:_** While tracing tools like Langsmith and Log Fire are valuable for telemetry, consider implementing your own database storage for evaluation data to avoid export headaches and gain more control over your analysis process.

## Which models should we use for different RAG applications?

When choosing between models like GPT-4, GPT-4 Turbo, or GPT-3.5, I've observed different selection patterns based on the task's importance and time constraints:

For high-value applications where accuracy is critical (like financial due diligence with 44 data rooms generating reports for clients paying $200,000 annually), companies often default to GPT-4 because "if it is just 2% better, it'll be worth it."

For applications requiring speed, GPT-3.5 or GPT-4 are common choices.

Many developers are now using Gemini for RAG applications because its large context window allows for less precision in retrieval: "You can just be really frivolous with how much context you use."

The decision often comes down to the stakes involved rather than technical benchmarks. For example, when helping sales teams craft follow-up emails containing offers, we use GPT-4 because the potential revenue impact justifies the additional cost.

**_Key Takeaway:_** Model selection should be driven by business value rather than technical specifications alone. For high-stakes applications where even small improvements matter, use the most capable model available. For less critical applications, prioritize speed and cost-efficiency.

## How can we enhance report generation with visual elements?

One exciting development in RAG applications is the integration of visual elements into generated reports. I'm currently working with a company on two key improvements:

1. Supporting mermaid diagrams in reports to visualize relationships and processes
1. Intelligently adding relevant images to reports

For example, in a construction permitting application, this could mean automatically including screenshots of potential errors in blueprints with accompanying explanations: "If in a report of predicted potential errors that you should pay attention to on your project, it would actually take a screenshot of the error in the PDF of the blueprint, and then have a narrative around it."

This approach dramatically increases the value of generated reports by combining visual and textual information, making complex issues immediately understandable to users.

**_Key Takeaway:_** The next frontier in RAG applications involves intelligently incorporating visual elements like diagrams and contextual images to enhance understanding and provide more comprehensive analysis.

## How should we manage expectations around AI capabilities?

Managing expectations is one of the biggest challenges when implementing AI systems, especially with clients who have either unrealistic expectations or excessive skepticism.

For construction applications, one participant described their approach: "We try to explain to people that ultimately in our field, you're an architect or a structural engineer. It's your stamp on the docs. You're making the call. We're just here to provide suggestions and things to look out for."

This aligns with my experience working with large enterprises, where much of my consulting work involves "dealing with the personality of the CEO" who might want AI to be a major theme at the next sales conference without understanding the practical limitations.

The most effective approach is focusing on how AI can augment human decision-making rather than replace it. For example, having the LLM run simulations and help humans interpret the results is more realistic than promising fully autonomous systems.

**_Key Takeaway:_** Set clear boundaries around AI capabilities by positioning your system as a decision support tool rather than an autonomous decision-maker. Be explicit about where human judgment remains essential, especially in high-stakes domains like construction or finance.

## What's your vision for building open-source AI tools?

When asked about my vision for building AI tools, I explained that my approach differs from the typical venture-backed startup model:

"Before it was okay, consulting can drive revenue that allows us to do open source work. The open source projects don't need to raise venture capital or figure out how to monetize, which changes the nature of the code."

This model allows me to create a portfolio of small, useful tools without worrying about monetization. The course serves as a way to connect with practitioners across different industries and identify common challenges:

"The most I ever did was like 7 clients in a month, and that was kind of a hazy period of my life where I have no memory of what happened. Whereas with the course, I can do these office hours - 10 people show up, great. I can understand how this permitting thing goes, maybe some architectural things, some construction things, some supply chain stuff."

This broader exposure helps me identify patterns across industries, like the common need for better report generation or specialized table parsing, which informs both my consulting work and open-source development.

**_Key Takeaway:_** By funding open-source development through consulting and courses rather than venture capital, I can focus on building genuinely useful tools without the pressure to monetize every component, leading to more sustainable and practical solutions.

## How should we think about pricing and value capture for AI systems?

One of the most exciting developments I see in AI is the evolution of pricing models away from usage-based metrics toward outcome-based pricing:

"I'm personally curious about pricing the work that LLMs do. A lot of systems right now are being priced on usage. I'm really excited about what it would mean to have a system that has so much accountability that you can price on the outcome it delivers."

I shared an example of a company that uses voice AI to make calls to car owners on behalf of dealerships. Under a usage-based model, the calls that make the most money are often those that waste time with confusion and errors. But with an outcome-based model, the incentives change dramatically:

"If you change the model to say, 'We want to take 3% of the mechanic's cost,' then it becomes, 'What if we had systems that are intelligently doing upsells? What if we intelligently figure out the right time and try to load balance the mechanic?'"

This shift changes the fundamental question from "How much am I willing to pay to process one PDF file?" (maybe 30 cents) to "Under what circumstances would I be willing to pay $20 to process a PDF?" The answer depends on the business value created.

**_Key Takeaway:_** The future of AI pricing will likely move from usage-based models (tokens, API calls) to outcome-based models where vendors are compensated based on the business value they create. This will drive investment in higher-quality systems that optimize for results rather than minimizing usage.

## Will AI capabilities eventually be built into platforms or remain in applications?

When asked whether AI capabilities will eventually be absorbed into platforms rather than remaining in applications, I suggested it depends on the time horizon:

"On any reasonable time horizon, it will probably just be the applications. The limiting factor is that for any specific application, we don't actually have the training data to bake this back into the model."

I referenced the "bitter lesson" in AI, which shows that when you have enough data and compute, general approaches tend to outperform specialized ones. However, we first need applications to generate the necessary data:

"We still have to build these applications as sort of sensors to create this data. And then, once we do, we can kind of sidestep the next innovation."

This is similar to how speech recognition evolved from complex phoneme-based systems to end-to-end models, but only after platforms like YouTube created enough data to make this possible.

"We had to build YouTube to produce enough data to get to a world where now we can train the GPT-4 model. So we still have to build these applications as sensors to create this data."

**_Key Takeaway:_** While AI capabilities will eventually be absorbed into platforms, we first need to build applications that generate the necessary training data. This creates a cycle where applications serve as data collection mechanisms that eventually enable more general-purpose AI systems.

## How might AI transform business models and value chains?

I believe AI will fundamentally change how businesses capture value, potentially shifting from software-as-a-service models to more integrated approaches:

"Everything stops becoming SaaS budget and it's all headcount budget. If you absorb this entire 'dealership calls car owner to get them in the mechanic' thing, at some point you're just a sales guy."

This could lead to companies expanding vertically to capture more of the value chain:

"Why don't you just own the entire value chain? Because then you can really price on the outcome that you're trying to deliver rather than just tokens."

While this approach means taking on additional complexity (like owning car mechanics with "all the pros and cons"), it allows for capturing more of the value created. This is similar to how I view the difference between writers who charge by word versus those who are paid based on qualified leads that convert.

"If there was an agent that's like, 'We'll just take all your phone calls and turn them into blog posts, and we only get charged a commission of course sales,' I would probably be really happy with that."

**_Key Takeaway:_** AI may drive a shift from software companies selling tools to companies that own entire value chains and are compensated based on business outcomes. This will require building systems that connect previously separate data streams to create end-to-end accountability.

## What's the most valuable data for future AI development?

The most valuable data for AI development has evolved over time:

"When I started, it was physics. And then it's like, 'Well, we're running out of sensors, but the next sensor is going to cost us 10 billion dollars.' So I went to Facebook - what's every post and comment and Facebook group and the social graph?"

Now, I believe the most valuable data will be how humans interact with AI:

"How humans use AI will be the most interesting dataset. And then in 10 years, it'll be how AI talks to AI. Most of the data produced will just be AI talking to AI."

This is why I'm particularly interested in working with companies that have large proprietary datasets in specialized domains:

"Someone was like, 'Oh, we have the last 40 years of investment decisions.' I was like, 'What?' Now I'm willing to pay so much to process this. Let's actually think about what the schemas look like and how to design this system."

These unique datasets offer opportunities to create specialized tools that can extract insights that general models can't access without the proper context and structure.

**_Key Takeaway:_** The most valuable data is shifting from general internet content to human-AI interactions and eventually AI-to-AI interactions. Companies with large proprietary datasets in specialized domains are particularly well-positioned to create value with AI systems tailored to their unique information.

---

FAQs

## What tools are recommended for tracing and evaluations in AI applications?

While Langsmith is commonly used, some users experience technical issues with data exports. Alternative options include Brain Trust for evaluations and Log Fire for tracing. The choice often depends on your specific needs and existing partnerships. For simpler implementations, consider storing question-answer pairs directly in a database rather than relying on third-party tracing software, which gives you more control and easier access to your data.

## How should I approach data collection for evaluating my AI application?

Start by creating an evaluation dataset from real user interactions. This can be done by exporting traces from tools like Langsmith or by directly storing question-answer pairs in your database. Once you have real data, you can generate synthetic questions to expand your test set. Focus on collecting both the user queries and your system's responses, along with any relevant context like retrieved document chunks, to enable comprehensive evaluation.

## Which language models are best for RAG (Retrieval-Augmented Generation) applications?

The choice depends on your specific requirements. GPT-4 is commonly used for standard implementations, while GPT-3.5 may be sufficient for applications where speed is critical. Gemini is popular for RAG applications due to its large context window, allowing you to include more retrieved content without worrying about token limits. For high-stakes applications where accuracy is paramount, GPT-3.5 is sometimes preferred despite being older, as it can be more reliable for certain use cases.

## How should I approach improving my AI application's performance?

Focus on systematic evaluation before making changes. Create a representative dataset of real user queries, then establish metrics that align with your business goals. Prioritize experiments based on potential impact and resource constraints—you can only run a limited number of experiments in a given timeframe. Remember that improving AI performance is an iterative process requiring continuous testing and refinement rather than a one-time fix.

## What are effective ways to manage expectations when implementing AI solutions?

Be transparent about both capabilities and limitations. Help stakeholders understand that AI implementation is an iterative process requiring ongoing refinement rather than a one-time deployment. Clearly define the role of AI as a tool to assist humans rather than replace them completely. For specialized fields like architecture or engineering, emphasize that professionals still need to make the final decisions, with AI serving as a support system that provides suggestions and identifies potential issues.

## How can I integrate visuals and diagrams into AI-generated reports?

This is an emerging area with promising developments. Consider implementing systems that can intelligently select and incorporate relevant images from your existing resources. For technical applications like construction or engineering, the ability to include screenshots of blueprints with annotations highlighting specific areas of concern can significantly enhance the value of AI-generated reports. Libraries like Mermaid for diagram generation are becoming more widely supported and can be integrated into AI workflows.

## How should AI applications be priced to capture appropriate value?

Consider moving beyond usage-based pricing (like per-token or per-user) toward outcome-based models that align with the actual business value delivered. For example, charging per resolved customer support ticket rather than per API call creates better alignment between your pricing and the value customers receive. This shift requires building systems with sufficient accountability and measurement capabilities to track outcomes reliably. The most innovative pricing approaches treat AI capabilities as replacements for headcount rather than as traditional software tools.

## What's the relationship between data collection and future AI capabilities?

Every AI application serves as a sensor that generates valuable data. The applications built today create the datasets that will enable more advanced AI capabilities tomorrow. Proprietary datasets from specialized industries (like investment decisions, supply chain operations, or construction projects) are particularly valuable for building domain-specific AI capabilities. The most interesting future developments will likely come from analyzing how humans interact with AI systems, creating a feedback loop of continuous improvement.
