"""
Week 5: Multimodal Search System

Implement table search with markdown conversion and image search
with rich descriptions.
"""

from dataclasses import dataclass

# Check dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("pandas not installed. Run: uv add pandas")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("chromadb not installed. Run: uv add chromadb")

OPENAI_AVAILABLE = False  # Not currently used, but available via: uv add openai


@dataclass 
class TableDocument:
    """A table converted to searchable document."""
    id: str
    name: str
    markdown: str
    summary: str
    sample_queries: list[str]
    schema: dict


@dataclass
class ImageDocument:
    """An image with rich description."""
    id: str
    path: str
    caption: str
    rich_description: str
    tags: list[str]
    search_queries: list[str]


def create_sample_tables() -> list[pd.DataFrame]:
    """Create sample tables for testing."""
    if not PANDAS_AVAILABLE:
        return []
    
    # Sales data
    sales = pd.DataFrame({
        "product": ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Tool Z"],
        "category": ["Widgets", "Widgets", "Gadgets", "Gadgets", "Tools"],
        "price": [29.99, 39.99, 149.99, 199.99, 79.99],
        "units_sold": [1500, 1200, 450, 380, 890],
        "revenue": [44985.0, 47988.0, 67495.5, 75996.2, 71191.1],
    })
    
    # Employee data
    employees = pd.DataFrame({
        "name": ["Alice Smith", "Bob Jones", "Carol White", "David Brown", "Eve Davis"],
        "department": ["Engineering", "Sales", "Engineering", "Marketing", "Sales"],
        "title": ["Senior Engineer", "Sales Rep", "Engineer", "Marketing Manager", "Sales Lead"],
        "salary": [120000, 65000, 95000, 85000, 75000],
        "start_date": ["2020-03-15", "2021-06-01", "2022-01-10", "2019-08-22", "2020-11-30"],
    })
    
    # Product catalog
    products = pd.DataFrame({
        "sku": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
        "name": ["Laptop Pro", "Wireless Mouse", "USB Hub", "Monitor 27\"", "Keyboard"],
        "category": ["Computers", "Accessories", "Accessories", "Displays", "Accessories"],
        "in_stock": [45, 230, 180, 28, 156],
        "reorder_point": [20, 100, 50, 15, 75],
    })
    
    return [
        ("sales_data", sales),
        ("employees", employees),
        ("product_catalog", products),
    ]


def table_to_markdown(df: pd.DataFrame, table_name: str) -> TableDocument:
    """Convert a DataFrame to a rich markdown document."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required")
    
    # Generate markdown
    markdown = df.to_markdown(index=False)
    
    # Extract schema
    schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": list(df.shape),
    }
    
    # Generate summary (mock - would use LLM in production)
    summary = f"Table '{table_name}' contains {len(df)} rows and {len(df.columns)} columns. "
    summary += f"Columns: {', '.join(df.columns)}. "
    
    if df.select_dtypes(include=['number']).columns.any():
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        summary += f"Numeric columns: {', '.join(numeric_cols)}."
    
    # Generate sample queries (mock)
    sample_queries = [
        f"Show all data from {table_name}",
        f"What are the columns in {table_name}?",
        f"How many rows in {table_name}?",
    ]
    
    # Add column-specific queries
    for col in df.columns[:3]:
        sample_queries.append(f"What is the {col} in {table_name}?")
    
    return TableDocument(
        id=f"table_{table_name}",
        name=table_name,
        markdown=markdown,
        summary=summary,
        sample_queries=sample_queries,
        schema=schema,
    )


def create_sample_images() -> list[dict]:
    """Create sample image metadata (without actual images)."""
    return [
        {
            "id": "img_001",
            "path": "images/product_laptop.jpg",
            "caption": "A laptop on a desk",
            "rich_description": """
            Scene: A modern silver laptop computer placed on a wooden desk in a bright office setting.
            Objects: Laptop (open, showing screen with code), wooden desk, coffee mug, notebook, pen.
            Colors: Silver laptop, warm brown wood, white coffee mug.
            Lighting: Natural daylight from window, soft shadows.
            Style: Professional product photography, clean composition.
            """,
            "tags": ["laptop", "computer", "office", "desk", "technology", "workspace"],
            "search_queries": [
                "laptop on desk",
                "office computer setup",
                "work from home laptop",
                "coding workspace",
            ],
        },
        {
            "id": "img_002",
            "path": "images/team_meeting.jpg",
            "caption": "People in a meeting",
            "rich_description": """
            Scene: A diverse group of professionals gathered around a conference table in a modern office.
            Objects: 5 people, conference table, laptops, presentation screen, glass walls.
            Colors: Blue shirts, white walls, natural wood table, green plants.
            Lighting: Bright overhead LED lighting, natural light through windows.
            Style: Corporate documentary style, candid moment.
            """,
            "tags": ["meeting", "team", "office", "business", "collaboration", "conference"],
            "search_queries": [
                "business meeting",
                "team collaboration",
                "office meeting room",
                "corporate teamwork",
            ],
        },
        {
            "id": "img_003",
            "path": "images/data_chart.jpg",
            "caption": "A chart showing data",
            "rich_description": """
            Scene: A colorful bar chart displayed on a computer monitor showing quarterly revenue.
            Objects: Monitor, bar chart, data labels, axis labels, legend.
            Colors: Blue and green bars, white background, black text.
            Lighting: Screen glow, office ambient light.
            Style: Business analytics visualization, clean data presentation.
            """,
            "tags": ["chart", "data", "analytics", "business", "visualization", "revenue"],
            "search_queries": [
                "revenue chart",
                "business analytics",
                "quarterly data",
                "bar chart visualization",
            ],
        },
    ]


class TableSearchSystem:
    """Search system for tables."""
    
    def __init__(self):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb required")
        
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.create_collection(
            name="tables",
            embedding_function=self.embedding_fn,
        )
        
        self.tables: dict[str, TableDocument] = {}
    
    def add_table(self, table_doc: TableDocument):
        """Add a table to the search index."""
        # Create searchable text
        searchable_text = f"""
        Table: {table_doc.name}
        Summary: {table_doc.summary}
        Columns: {', '.join(table_doc.schema['columns'])}
        Sample queries: {', '.join(table_doc.sample_queries)}
        Data preview:
        {table_doc.markdown[:500]}
        """
        
        self.collection.add(
            documents=[searchable_text],
            ids=[table_doc.id],
            metadatas=[{"name": table_doc.name, "type": "table"}],
        )
        
        self.tables[table_doc.id] = table_doc
    
    def search(self, query: str, k: int = 3) -> list[TableDocument]:
        """Search for relevant tables."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )
        
        found_tables = []
        for doc_id in results["ids"][0]:
            if doc_id in self.tables:
                found_tables.append(self.tables[doc_id])
        
        return found_tables


class ImageSearchSystem:
    """Search system for images with rich descriptions."""
    
    def __init__(self):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb required")
        
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Two collections: basic captions vs rich descriptions
        self.basic_collection = self.client.create_collection(
            name="images_basic",
            embedding_function=self.embedding_fn,
        )
        
        self.rich_collection = self.client.create_collection(
            name="images_rich",
            embedding_function=self.embedding_fn,
        )
        
        self.images: dict[str, ImageDocument] = {}
    
    def add_image(self, image: dict):
        """Add an image to both collections."""
        doc = ImageDocument(
            id=image["id"],
            path=image["path"],
            caption=image["caption"],
            rich_description=image["rich_description"],
            tags=image["tags"],
            search_queries=image["search_queries"],
        )
        
        # Add to basic collection (caption only)
        self.basic_collection.add(
            documents=[doc.caption],
            ids=[doc.id],
            metadatas=[{"path": doc.path}],
        )
        
        # Add to rich collection (full description + tags + queries)
        rich_text = f"""
        Caption: {doc.caption}
        Description: {doc.rich_description}
        Tags: {', '.join(doc.tags)}
        Related queries: {', '.join(doc.search_queries)}
        """
        
        self.rich_collection.add(
            documents=[rich_text],
            ids=[doc.id],
            metadatas=[{"path": doc.path}],
        )
        
        self.images[doc.id] = doc
    
    def search_basic(self, query: str, k: int = 3) -> list[str]:
        """Search using basic captions."""
        results = self.basic_collection.query(
            query_texts=[query],
            n_results=k,
        )
        return results["ids"][0]
    
    def search_rich(self, query: str, k: int = 3) -> list[str]:
        """Search using rich descriptions."""
        results = self.rich_collection.query(
            query_texts=[query],
            n_results=k,
        )
        return results["ids"][0]
    
    def compare_search(self, query: str, k: int = 3) -> dict:
        """Compare basic vs rich search results."""
        basic_results = self.search_basic(query, k)
        rich_results = self.search_rich(query, k)
        
        return {
            "query": query,
            "basic_results": basic_results,
            "rich_results": rich_results,
            "overlap": len(set(basic_results) & set(rich_results)),
        }


def main():
    """Demo the multimodal search systems."""
    print("=" * 60)
    print("WEEK 5: MULTIMODAL SEARCH SYSTEM")
    print("=" * 60)
    
    # === Table Search ===
    print("\n" + "-" * 60)
    print("TRACK A: TABLE SEARCH")
    print("-" * 60)
    
    # Create and index tables
    tables = create_sample_tables()
    table_search = TableSearchSystem()
    
    print("\nIndexing tables...")
    for name, df in tables:
        doc = table_to_markdown(df, name)
        table_search.add_table(doc)
        print(f"  Added: {name} ({len(df)} rows, {len(df.columns)} columns)")
    
    # Test queries
    table_queries = [
        "What products do we sell?",
        "Show employee salaries",
        "Which items are low on stock?",
        "Revenue by product",
    ]
    
    print("\nTable search results:")
    for query in table_queries:
        results = table_search.search(query, k=1)
        if results:
            print(f"\n  Query: {query}")
            print(f"  Found: {results[0].name}")
            print(f"  Summary: {results[0].summary[:100]}...")
    
    # === Image Search ===
    print("\n" + "-" * 60)
    print("TRACK B: IMAGE SEARCH")
    print("-" * 60)
    
    # Create and index images
    images = create_sample_images()
    image_search = ImageSearchSystem()
    
    print("\nIndexing images...")
    for img in images:
        image_search.add_image(img)
        print(f"  Added: {img['id']} - {img['caption']}")
    
    # Compare search methods
    image_queries = [
        "computer workspace",
        "business presentation",
        "revenue visualization",
        "team collaboration meeting",
    ]
    
    print("\nComparing basic vs rich search:")
    print(f"\n{'Query':<30} {'Basic':<20} {'Rich':<20}")
    print("-" * 70)
    
    for query in image_queries:
        comparison = image_search.compare_search(query, k=2)
        basic = ", ".join(comparison["basic_results"][:2])
        rich = ", ".join(comparison["rich_results"][:2])
        print(f"{query:<30} {basic:<20} {rich:<20}")
    
    # Evaluate improvement
    print("\n" + "-" * 60)
    print("SEARCH QUALITY COMPARISON")
    print("-" * 60)
    
    # Simulate ground truth evaluation
    ground_truth = {
        "laptop coding workspace": ["img_001"],
        "business meeting room": ["img_002"],
        "quarterly revenue data": ["img_003"],
    }
    
    basic_correct = 0
    rich_correct = 0
    
    for query, expected in ground_truth.items():
        basic_results = image_search.search_basic(query, k=1)
        rich_results = image_search.search_rich(query, k=1)
        
        if basic_results and basic_results[0] in expected:
            basic_correct += 1
        if rich_results and rich_results[0] in expected:
            rich_correct += 1
    
    total = len(ground_truth)
    print(f"\nBasic caption search accuracy: {basic_correct}/{total} ({basic_correct/total:.0%})")
    print(f"Rich description search accuracy: {rich_correct}/{total} ({rich_correct/total:.0%})")
    
    if rich_correct > basic_correct:
        improvement = (rich_correct - basic_correct) / basic_correct * 100 if basic_correct > 0 else 100
        print(f"Improvement: +{improvement:.0f}%")
    
    print("\n" + "=" * 60)
    print("Multimodal search demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
