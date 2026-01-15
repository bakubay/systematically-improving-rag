# Week 5 Assignment: Multimodal Search System

## Learning Goal

Build specialized retrieval systems for tables and images. You will compare basic and enhanced approaches to show measurable gains.

## Setup

```bash
uv add chromadb sentence-transformers pandas openai pillow datasets
```

## Dataset Options

**Track A - Table Search**: Use WikiTableQuestions or create your own tables.

```python
from datasets import load_dataset
import pandas as pd

# Option 1: WikiTableQuestions
wtq = load_dataset("wikitablequestions", split="train[:100]")

# Option 2: Create sample tables (easier to start)
sales_table = pd.DataFrame({
    "product": ["Widget A", "Widget B", "Gadget X"],
    "price": [29.99, 39.99, 149.99],
    "units_sold": [1500, 1200, 450],
})
```

**Track B - Image Search**: Use COCO captions or local images.

```python
from datasets import load_dataset
from PIL import Image

# Option 1: COCO dataset
coco = load_dataset("HuggingFaceM4/COCO", split="train[:500]")

# Option 2: Local images (requires actual image files)
images = [Image.open(f"images/{i}.jpg") for i in range(10)]
```

## Success Criteria

- Track A: Table search retrieves correct table 80%+ of the time
- Track B: Rich descriptions improve recall by 20%+ over basic captions
- Evaluation on at least 20 test queries

## Why This Works

- **Tables and images need different retrieval**: plain chunking often breaks structure (tables) or loses meaning (images).
- **Rich descriptions improve recall**: you turn “pixels” into searchable text that matches user intent.
- **Two baselines make it measurable**: basic captions vs rich descriptions is a clean A/B comparison.

## Common Mistakes

- **Embedding raw tables without metadata**: you need column names and summaries, not just cells.
- **Using the same query set for both tracks**: tables and images need different test queries.
- **Evaluating without ground truth**: pick 20 queries where you know the right table/image, or you can’t measure improvement.

## Track A: Table Search

### Part 1: Convert Tables to Markdown

```python
import json
import pandas as pd
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def table_to_markdown(df: pd.DataFrame, table_name: str) -> str:
    markdown_table = df.to_markdown(index=False)
    prompt = f"""Analyze this table and return JSON with:
1. summary: 2-3 sentences describing the table
2. sample_queries: 5-7 example questions

Table name: {table_name}
Table preview:
{df.head(10).to_markdown()}
"""
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    metadata = json.loads(response.choices[0].message.content)
    return f"""# {table_name}

## Summary
{metadata["summary"]}

## Table Data
{markdown_table}

## Sample Queries
{chr(10).join(f"- {q}" for q in metadata["sample_queries"])}
"""
```

### Part 2: Index Tables in ChromaDB

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./table_search_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="tables",
    embedding_function=embedding_fn,
)
```

### Part 3: Dual Search Modes

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def search_tables(query: str, mode: str = "semantic") -> dict:
    if mode == "semantic":
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["documents", "metadatas"],
        )
        return {"tables": results["ids"][0], "docs": results["documents"][0]}

    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents"],
    )
    context = "\n\n".join(results["documents"][0][:3])
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[{
            "role": "user",
            "content": f"Answer the question using the table data.\n\nQuestion: {query}\n\n{context}",
        }],
    )
    return {"answer": response.choices[0].message.content, "sources": results["ids"][0][:3]}
```

## Track B: Image Search

### Part 1: Generate Rich Descriptions

```python
import base64
import io
from PIL import Image
from openai import AsyncOpenAI

client = AsyncOpenAI()

def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

async def generate_rich_description(image: Image.Image, caption: str | None = None) -> str:
    img_base64 = encode_image(image)
    prompt = (
        "Analyze this image and return:\n"
        "1. Scene description\n"
        "2. Objects and people\n"
        "3. Colors and lighting\n"
        "4. 5-7 search queries\n"
        "5. 10 tags\n"
    )
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ],
        }],
        max_tokens=500,
    )
    rich = response.choices[0].message.content
    return f"Caption: {caption}\n\nRich description:\n{rich}" if caption else rich
```

### Part 2: Compare Basic vs Enhanced Search

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./image_search_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection_basic = client.get_or_create_collection(
    name="images_basic",
    embedding_function=embedding_fn,
)
collection_enhanced = client.get_or_create_collection(
    name="images_enhanced",
    embedding_function=embedding_fn,
)
```

## Deliverable

Choose one track and submit:

- A working system with ChromaDB indexing
- Evaluation on a small test set (10-20 queries)
- Precision and recall comparison between basic and enhanced methods

## Bonus

Build a unified search that returns both tables and images for a single query.
