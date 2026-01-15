# Week 4 Assignment: Build a Query Clustering System

## Learning Goal

Analyze query patterns to identify improvement opportunities. You'll cluster queries, visualize satisfaction patterns, and build a prioritization framework.

## Setup

```bash
uv add chromadb sentence-transformers scikit-learn umap-learn matplotlib plotly pandas
```

## Success Criteria

- Cluster queries into 10-20 meaningful groups
- UMAP visualization shows clear cluster separation
- 2x2 matrix identifies at least 2 "danger zone" clusters
- Prioritized roadmap with actionable recommendations

## Why This Works

- **You stop treating all failures equally**: clustering finds repeating patterns, not one-off bugs.
- **Volume + satisfaction gives priorities**: fixing a high-volume, low-satisfaction cluster usually moves the product.
- **Segmentation prevents “random tweaking”**: you can pick one cluster, fix it, and measure impact.

## Common Mistakes

- **Using only silhouette score to choose \(k\)**: it’s a hint, not the truth. Always inspect sample queries.
- **Clustering on tiny datasets**: < 100 queries often produces noisy clusters.
- **Ignoring user adaptation**: users may rephrase to “work around” your system, making satisfaction look better than it is.

## Dataset

Use query logs from your Week 0 dashboard, or simulate with **HotpotQA**:

```python
from datasets import load_dataset

# HotpotQA has diverse question types
hotpot = load_dataset("hotpot_qa", "distractor", split="train[:1000]")

# Extract queries and satisfaction scores (simulate based on answer quality)
queries = []
satisfaction_scores = []

for item in hotpot:
    queries.append(item["question"])
    # Simulate satisfaction: 1.0 if answer found, 0.5 if partial, 0.0 if failed
    satisfaction_scores.append(1.0 if item.get("answer") else 0.5)
```

Or load from your SQLite logs:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("rag_metrics.db")
df = pd.read_sql("SELECT query, feedback FROM queries WHERE feedback IS NOT NULL", conn)
queries = df["query"].tolist()
satisfaction_scores = df["feedback"].apply(lambda x: 1.0 if x == 1 else 0.0).tolist()
```

## Requirements

### Part 1: Cluster Queries

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import numpy as np

# Embed queries
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(queries, show_progress_bar=True)

# Determine optimal k using elbow method
inertias = []
k_range = range(5, 25)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.savefig("elbow_curve.png")

# Choose k (e.g., k=15)
optimal_k = 15
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

# Reduce to 2D for visualization
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embedding_2d = reducer.fit_transform(embeddings)

# Plot clusters colored by satisfaction
plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    embedding_2d[:, 0], 
    embedding_2d[:, 1], 
    c=satisfaction_scores, 
    cmap='RdYlGn',
    s=50,
    alpha=0.6,
    edgecolors='black',
    linewidth=0.5
)
plt.colorbar(scatter, label='Satisfaction Score')
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("Query Clusters Colored by Satisfaction")
plt.savefig("query_clusters.png", dpi=300)
```

### Part 2: Analyze Each Cluster

```python
from openai import AsyncOpenAI
import json

client = AsyncOpenAI()

async def analyze_clusters() -> dict:
    cluster_analysis: dict[int, dict] = {}
    for cluster_id in range(optimal_k):
        cluster_queries = [q for i, q in enumerate(queries) if cluster_labels[i] == cluster_id]
        cluster_satisfaction = [
            satisfaction_scores[i] for i in range(len(queries)) if cluster_labels[i] == cluster_id
        ]

        avg_satisfaction = np.mean(cluster_satisfaction)
        volume = len(cluster_queries)

        # Find failure examples
        failure_indices = [
            i for i in range(len(queries))
            if cluster_labels[i] == cluster_id and satisfaction_scores[i] < 0.5
        ]
        failure_examples = [queries[i] for i in failure_indices[:10]]

        # Generate cluster label using LLM
        response = await client.chat.completions.create(
            model="gpt-5.2",
            messages=[{
                "role": "user",
                "content": f"""Analyze these queries and generate a descriptive label for this cluster:

Queries:
{json.dumps(cluster_queries[:20], indent=2)}

Return a short label (2-4 words) describing the common theme."""
            }],
        )
        cluster_label = response.choices[0].message.content.strip()

        cluster_analysis[cluster_id] = {
            "label": cluster_label,
            "volume": volume,
            "avg_satisfaction": avg_satisfaction,
            "failure_examples": failure_examples,
        }

        print(f"\n=== Cluster {cluster_id}: {cluster_label} ===")
        print(f"Volume: {volume} queries ({volume/len(queries)*100:.1f}%)")
        print(f"Avg Satisfaction: {avg_satisfaction:.2f}")
        print(f"Top Failures: {failure_examples[:3]}")

    return cluster_analysis

# In notebooks, you can run: cluster_analysis = await analyze_clusters()
```

### Part 3: Build 2x2 Prioritization Matrix

```python
import matplotlib.pyplot as plt

# Calculate volume percentage and satisfaction for each cluster
volumes = [cluster_analysis[c]["volume"] / len(queries) * 100 for c in range(optimal_k)]
satisfactions = [cluster_analysis[c]["avg_satisfaction"] for c in range(optimal_k)]
labels = [cluster_analysis[c]["label"] for c in range(optimal_k)]

# Create 2x2 matrix
fig, ax = plt.subplots(figsize=(12, 10))

# Color by quadrant
colors = []
for v, s in zip(volumes, satisfactions):
    if v > np.median(volumes) and s < np.median(satisfactions):
        colors.append("red")  # High volume, low satisfaction = DANGER ZONE
    elif v > np.median(volumes) and s >= np.median(satisfactions):
        colors.append("green")  # High volume, high satisfaction = GOOD
    elif v <= np.median(volumes) and s < np.median(satisfactions):
        colors.append("orange")  # Low volume, low satisfaction = Monitor
    else:
        colors.append("blue")  # Low volume, high satisfaction = Niche success

scatter = ax.scatter(volumes, satisfactions, c=colors, s=200, alpha=0.6, edgecolors='black')

# Add labels
for i, label in enumerate(labels):
    ax.annotate(label[:20], (volumes[i], satisfactions[i]), fontsize=8, ha='center')

# Add quadrant lines
ax.axvline(np.median(volumes), color='gray', linestyle='--', alpha=0.5)
ax.axhline(np.median(satisfactions), color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel("Volume (%)")
ax.set_ylabel("Average Satisfaction")
ax.set_title("Query Cluster Prioritization Matrix")
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='High Volume, Low Satisfaction (PRIORITY)'),
    Patch(facecolor='orange', label='Low Volume, Low Satisfaction'),
    Patch(facecolor='green', label='High Volume, High Satisfaction'),
    Patch(facecolor='blue', label='Low Volume, High Satisfaction')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig("prioritization_matrix.png", dpi=300)
```

### Part 4: Inventory vs Capability Classifier

```python
async def classify_issue_type(cluster_id: int, cluster_data: dict) -> str:
    """Classify as inventory (missing data) or capability (missing feature)."""
    
    # Analyze failure examples
    failure_examples = cluster_data["failure_examples"]
    
    response = await client.chat.completions.create(
        model="gpt-5.2",
        messages=[{
            "role": "user",
            "content": f"""Analyze these failed queries and determine if the issue is:

1. INVENTORY: The relevant data exists but wasn't retrieved (retrieval problem)
2. CAPABILITY: The data doesn't exist in our corpus (data gap)

Failed queries:
{json.dumps(failure_examples[:5], indent=2)}

Respond with only: INVENTORY or CAPABILITY"""
        }]
    )
    
    return response.choices[0].message.content.strip()

# Classify all clusters
for cluster_id in range(optimal_k):
    issue_type = await classify_issue_type(cluster_id, cluster_analysis[cluster_id])
    cluster_analysis[cluster_id]["issue_type"] = issue_type
    print(f"Cluster {cluster_id} ({cluster_analysis[cluster_id]['label']}): {issue_type}")
```

### Part 5: Generate Prioritized Roadmap

```python
def calculate_priority(cluster_data: dict) -> float:
    """Priority = (Impact × Volume%) / (Effort × Risk)"""
    impact = 1.0 - cluster_data["avg_satisfaction"]  # Higher impact if satisfaction is low
    volume_pct = cluster_data["volume"] / len(queries) * 100
    
    # Effort: 1.0 for inventory (easier to fix), 2.0 for capability (harder)
    effort = 1.0 if cluster_data["issue_type"] == "INVENTORY" else 2.0
    
    # Risk: 0.5 for low volume (less risky), 1.0 for high volume (more risky)
    risk = 1.0 if cluster_data["volume"] > np.median([c["volume"] for c in cluster_analysis.values()]) else 0.5
    
    return (impact * volume_pct) / (effort * risk)

# Calculate priorities
for cluster_id in range(optimal_k):
    cluster_analysis[cluster_id]["priority"] = calculate_priority(cluster_analysis[cluster_id])

# Sort by priority
sorted_clusters = sorted(
    cluster_analysis.items(),
    key=lambda x: x[1]["priority"],
    reverse=True
)

print("\n=== Prioritized Improvement Roadmap ===")
for rank, (cluster_id, data) in enumerate(sorted_clusters[:10], 1):
    print(f"\n{rank}. {data['label']} (Cluster {cluster_id})")
    print(f"   Priority Score: {data['priority']:.2f}")
    print(f"   Volume: {data['volume']} queries ({data['volume']/len(queries)*100:.1f}%)")
    print(f"   Satisfaction: {data['avg_satisfaction']:.2f}")
    print(f"   Issue Type: {data['issue_type']}")
    print(f"   Estimated Effort: {'Low' if data['issue_type'] == 'INVENTORY' else 'High'}")
```

## Deliverable

A Jupyter notebook with:

1. Query clustering with optimal k selection (elbow method)
2. UMAP visualization colored by satisfaction
3. Cluster analysis with LLM-generated labels
4. 2x2 prioritization matrix identifying "danger zones"
5. Inventory vs capability classification
6. Prioritized roadmap with top 10 improvement opportunities

## Bonus

Build an interactive dashboard with Plotly:

```python
import plotly.graph_objects as go
import plotly.express as px

# Interactive scatter plot
fig = px.scatter(
    x=volumes,
    y=satisfactions,
    color=colors,
    size=[v*10 for v in volumes],
    hover_data=[labels],
    labels={"x": "Volume %", "y": "Satisfaction"},
    title="Query Cluster Prioritization"
)
fig.show()
```
