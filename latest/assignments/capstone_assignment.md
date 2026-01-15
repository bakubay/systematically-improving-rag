# Capstone Assignment: End-to-End Improving RAG System

## Learning Goal

Build a full RAG system that demonstrates the improvement flywheel: measure, analyze, improve, and iterate.

## Setup

```bash
uv add chromadb sentence-transformers openai pandas matplotlib streamlit fastapi uvicorn redis datasets
```

## Success Criteria

- Baseline metrics established (precision@1, recall@1, MRR)
- At least one improvement cycle with 10%+ gain
- Routing accuracy > 85%
- Cost per successful query calculated
- Short report documenting the improvement flywheel

## Why This Works

- **It forces a real loop**: measure → analyze → fix one thing → re-measure.
- **You learn the multiplication effect**: weak routing or weak retrieval can cap the whole system.
- **You practice choosing the highest ROI fix**: not every improvement is worth the cost.

## Common Mistakes

- **Changing many things at once**: you won’t know what caused improvement or regression.
- **No held-out test set**: you’ll overfit to the examples you looked at during debugging.
- **Optimizing only “accuracy”**: in production, cost and latency matter too.

## Dataset

Choose one:

- ArXiv titles and abstracts: `sentence-transformers/arxiv-titles`
- MS MARCO passage ranking: `microsoft/ms_marco`
- Natural Questions: `sentence-transformers/natural-questions`

## Phase 1: Baseline System

```python
import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset

dataset = load_dataset("sentence-transformers/arxiv-titles", split="train[:1000]")
documents = [f"{t} {a}" for t, a in zip(dataset["title"], dataset["abstract"])]

client = chromadb.PersistentClient(path="./capstone_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection("baseline", embedding_function=embedding_fn)
collection.add(documents=documents, ids=[f"doc_{i}" for i in range(len(documents))])
```

## Phase 2: Evaluation

Use custom metrics from Week 1 (precision@k, recall@k, MRR, NDCG). Generate synthetic queries with OpenAI.

```python
import json
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def generate_questions(doc: str, n: int = 3) -> list[dict]:
    prompt = f"Generate {n} questions that can be answered by this text. Return JSON."
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": f"{prompt}\n\n{doc}"}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content).get("questions", [])
```

## Phase 3: Specialization

Build 2-3 specialized collections:

- Factual queries (titles, authors, dates)
- Conceptual queries (full abstracts)
- Temporal queries (year filters)

## Phase 4: Routing

Implement OpenAI tool calling with strict schemas. Use 20+ examples per tool and measure routing accuracy.

## Phase 5: Feedback Collection

Create a Streamlit UI with citation feedback:

- Show sources per answer
- Allow thumbs up and thumbs down
- Log feedback to SQLite

## Phase 6: Improvement

Analyze failures and apply one targeted fix:

- Add missing data (inventory issue)
- Improve retriever (capability issue)
- Add a new tool

Re-run evaluation on a held-out test set.

## Deliverable

Provide a short report that includes:

1. Baseline metrics (precision@5, recall@5, MRR)
2. Routing accuracy before and after changes
3. One improvement cycle with measured impact (+10-15% target)
4. Cost analysis (tokens and cost per successful query)
5. Architecture diagram of the system
