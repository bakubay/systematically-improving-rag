# Week 6 Assignment: Tool Routing System

## Learning Goal

Build a query router using OpenAI tool calling with few-shot examples and dynamic example selection.

## Setup

```bash
uv add openai sentence-transformers chromadb
```

## Dataset

Create a labeled routing dataset with 50-100 queries and the correct tool.

```python
queries = [
    {"query": "Find blueprints for the main building from 2020", "tool": "search_blueprints"},
    {"query": "What is the maintenance schedule for next week?", "tool": "search_schedule"},
    {"query": "Show documents about safety protocols", "tool": "search_documents"},
    {"query": "I am not sure what I need", "tool": "clarify_question"},
]
```

Split into train and test.

## Why This Works

- **Routing multiplies retrieval quality**: overall success is roughly \(routing\_accuracy \times tool\_success\).
- **Few-shot examples encode product knowledge**: your examples teach what each tool is for.
- **Dynamic example selection helps edge cases**: similar past queries reduce “weird” misroutes.

## Common Mistakes

- **Bad tool contracts**: if tool schemas are vague, the model will “guess” arguments.
- **Forgetting strict-mode rules**: with `strict: True`, optional fields must be nullable and still listed in `required`.
- **Training/test leakage**: if you reuse the same queries in both sets, accuracy looks fake.

## Requirements

### Part 1: Define Tool Schemas

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

# IMPORTANT: With strict: True, ALL properties must be in required.
# For optional params, use type: ["string", "null"] to allow null values.

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_blueprints",
            "description": "Search building plans and blueprints by description and date range",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "start_date": {"type": ["string", "null"], "description": "Optional start date"},
                    "end_date": {"type": ["string", "null"], "description": "Optional end date"},
                },
                "required": ["description", "start_date", "end_date"],  # All params required with strict
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_schedule",
            "description": "Search schedules, appointments, and calendar events",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "date_range": {"type": "string"},
                    "event_type": {
                        "type": ["string", "null"],
                        "enum": ["maintenance", "inspection", "meeting", "other", None],
                    },
                },
                "required": ["date_range", "event_type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search general documents and reports",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "document_type": {
                        "type": ["string", "null"],
                        "enum": ["report", "protocol", "manual", "other", None],
                    },
                },
                "required": ["query", "document_type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clarify_question",
            "description": "Ask a clarifying question when the query is ambiguous",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"clarification": {"type": "string"}},
                "required": ["clarification"],
                "additionalProperties": False,
            },
        },
    },
]
```

### Part 2: Few-Shot Router

```python
def build_system_prompt(examples: list[dict], examples_per_tool: int = 10) -> str:
    grouped = {}
    for item in examples:
        grouped.setdefault(item["tool"], []).append(item["query"])

    prompt = "You are a query router. Choose the correct tool for each query.\n\nExamples:\n"
    for tool_name, tool_examples in grouped.items():
        selected = tool_examples[:examples_per_tool]
        prompt += f"\n{tool_name}:\n"
        for ex in selected:
            prompt += f"- {ex}\n"
    return prompt

async def route_query(query: str, system_prompt: str) -> dict | None:
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        tools=tools,
        parallel_tool_calls=False,
        tool_choice="auto",
    )
    message = response.choices[0].message
    if not message.tool_calls:
        return None
    tool_call = message.tool_calls[0]
    return {"tool": tool_call.function.name, "arguments": tool_call.function.arguments}
```

### Part 3: Dynamic Example Selection

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def select_similar_examples(query: str, examples: list[dict], tool: str, n: int = 5) -> list[str]:
    candidates = [e["query"] for e in examples if e["tool"] == tool]
    if len(candidates) <= n:
        return candidates
    query_vec = model.encode([query])
    candidate_vecs = model.encode(candidates)
    scores = np.dot(candidate_vecs, query_vec.T).flatten()
    top_idx = np.argsort(scores)[::-1][:n]
    return [candidates[i] for i in top_idx]

def build_dynamic_prompt(query: str, examples: list[dict], n: int = 5) -> str:
    tools_list = ["search_blueprints", "search_schedule", "search_documents", "clarify_question"]
    prompt = "You are a query router. Choose the correct tool for each query.\n\nExamples:\n"
    for tool in tools_list:
        selected = select_similar_examples(query, examples, tool, n)
        prompt += f"\n{tool}:\n"
        for ex in selected:
            prompt += f"- {ex}\n"
    return prompt
```

### Part 4: Evaluate Routing Accuracy

```python
async def evaluate_accuracy(test_set: list[dict], train_set: list[dict], n: int, dynamic: bool) -> float:
    correct = 0
    for item in test_set:
        query = item["query"]
        expected = item["tool"]
        if dynamic:
            prompt = build_dynamic_prompt(query, train_set, n)
        else:
            prompt = build_system_prompt(train_set, n)
        result = await route_query(query, prompt)
        predicted = result["tool"] if result else None
        if predicted == expected:
            correct += 1
    return correct / len(test_set) if test_set else 0.0
```

Note: This evaluation can be parallelized with asyncio.gather if you want faster throughput.

## Deliverable

- Tool schemas with strict mode enabled
- Static and dynamic few-shot routing
- Accuracy comparison at 5, 10, 20, and 40 examples per tool
- Target: 88% at 10 examples and 95% at 40 examples
