"""
Week 6: Query Router with OpenAI Tool Calling

Implements few-shot routing using OpenAI's function calling API.
Includes dynamic example selection for improved accuracy.
"""

import json
from dataclasses import dataclass
from typing import Optional
import os

# Check for optional dependencies
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("openai not installed. Run: uv add openai")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("sentence-transformers not installed. Run: uv add sentence-transformers")


# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documentation",
            "description": "Search technical documentation, manuals, and guides",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for documentation"
                    },
                    "doc_type": {
                        "type": ["string", "null"],
                        "enum": ["api", "tutorial", "reference", "guide", None],
                        "description": "Type of documentation to search"
                    }
                },
                "required": ["query", "doc_type"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code_examples",
            "description": "Search for code snippets and implementation examples",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Description of the code example needed"
                    },
                    "language": {
                        "type": ["string", "null"],
                        "enum": ["python", "javascript", "typescript", "go", "rust", None],
                        "description": "Programming language"
                    }
                },
                "required": ["query", "language"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_troubleshooting",
            "description": "Search for error solutions and troubleshooting guides",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_message": {
                        "type": "string",
                        "description": "The error message or problem description"
                    },
                    "context": {
                        "type": ["string", "null"],
                        "description": "Additional context about when the error occurs"
                    }
                },
                "required": ["error_message", "context"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clarify_question",
            "description": "Ask for clarification when the query is ambiguous or unclear",
            "parameters": {
                "type": "object",
                "properties": {
                    "clarification_question": {
                        "type": "string",
                        "description": "Question to ask the user for clarification"
                    }
                },
                "required": ["clarification_question"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


# Training examples for few-shot learning
TRAINING_EXAMPLES = [
    # search_documentation
    {"query": "How do I authenticate with the API?", "tool": "search_documentation"},
    {"query": "What are the rate limits?", "tool": "search_documentation"},
    {"query": "Show me the API reference for embeddings", "tool": "search_documentation"},
    {"query": "Where can I find the quickstart guide?", "tool": "search_documentation"},
    {"query": "What parameters does the search endpoint accept?", "tool": "search_documentation"},
    {"query": "Explain the authentication flow", "tool": "search_documentation"},
    {"query": "What's the maximum request size?", "tool": "search_documentation"},
    {"query": "Documentation for batch processing", "tool": "search_documentation"},
    
    # search_code_examples
    {"query": "Show me how to make an API call in Python", "tool": "search_code_examples"},
    {"query": "Example of pagination implementation", "tool": "search_code_examples"},
    {"query": "Code for handling webhooks", "tool": "search_code_examples"},
    {"query": "How to implement retry logic?", "tool": "search_code_examples"},
    {"query": "Sample code for file upload", "tool": "search_code_examples"},
    {"query": "Example of async API calls", "tool": "search_code_examples"},
    {"query": "Show me a working example of streaming", "tool": "search_code_examples"},
    {"query": "Code snippet for error handling", "tool": "search_code_examples"},
    
    # search_troubleshooting
    {"query": "Getting 401 unauthorized error", "tool": "search_troubleshooting"},
    {"query": "Why is my request timing out?", "tool": "search_troubleshooting"},
    {"query": "Connection refused error", "tool": "search_troubleshooting"},
    {"query": "API returns 500 internal server error", "tool": "search_troubleshooting"},
    {"query": "Rate limit exceeded, what should I do?", "tool": "search_troubleshooting"},
    {"query": "SSL certificate error when connecting", "tool": "search_troubleshooting"},
    {"query": "Invalid JSON response error", "tool": "search_troubleshooting"},
    {"query": "Request failed with status 403", "tool": "search_troubleshooting"},
    
    # clarify_question
    {"query": "help", "tool": "clarify_question"},
    {"query": "how do I do this?", "tool": "clarify_question"},
    {"query": "it's not working", "tool": "clarify_question"},
    {"query": "can you help me?", "tool": "clarify_question"},
    {"query": "what's the best way?", "tool": "clarify_question"},
    {"query": "I need assistance", "tool": "clarify_question"},
]


# Test queries for evaluation
TEST_QUERIES = [
    {"query": "Show me the API documentation for creating users", "expected": "search_documentation"},
    {"query": "Python example for sending emails", "expected": "search_code_examples"},
    {"query": "Getting timeout errors on large requests", "expected": "search_troubleshooting"},
    {"query": "something is wrong", "expected": "clarify_question"},
    {"query": "What's the endpoint for search?", "expected": "search_documentation"},
    {"query": "Sample code for authentication", "expected": "search_code_examples"},
    {"query": "Error 429 too many requests", "expected": "search_troubleshooting"},
    {"query": "How do I implement caching?", "expected": "search_code_examples"},
    {"query": "List all available endpoints", "expected": "search_documentation"},
    {"query": "Why am I getting null responses?", "expected": "search_troubleshooting"},
]


@dataclass
class RoutingResult:
    """Result of routing a query."""
    query: str
    tool: str
    arguments: dict
    confidence: float = 1.0


class QueryRouter:
    """Route queries to appropriate tools using OpenAI function calling."""
    
    def __init__(self, model: str = "gpt-5.2"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai required. Run: uv add openai")
        
        self.client = OpenAI()
        self.model = model
        self.tools = TOOLS
        self.training_examples = TRAINING_EXAMPLES
        
        # For dynamic example selection
        self.embedding_model = None
        self.example_embeddings = None
        if ST_AVAILABLE:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._compute_example_embeddings()
    
    def _compute_example_embeddings(self):
        """Pre-compute embeddings for training examples."""
        if self.embedding_model is None:
            return
        
        texts = [ex["query"] for ex in self.training_examples]
        self.example_embeddings = self.embedding_model.encode(texts)
    
    def _build_system_prompt(self, n_examples: int = 5, query: Optional[str] = None) -> str:
        """Build system prompt with few-shot examples."""
        prompt = """You are a query router. Route user queries to the appropriate tool.

Available tools:
- search_documentation: For API references, guides, and documentation
- search_code_examples: For code snippets and implementation examples  
- search_troubleshooting: For errors, bugs, and problems
- clarify_question: For ambiguous or unclear queries

Examples:
"""
        # Select examples
        if query and self.embedding_model is not None:
            examples = self._select_similar_examples(query, n_examples)
        else:
            examples = self.training_examples[:n_examples * 4]  # n per tool
        
        # Group by tool
        by_tool = {}
        for ex in examples:
            tool = ex["tool"]
            if tool not in by_tool:
                by_tool[tool] = []
            by_tool[tool].append(ex["query"])
        
        for tool, queries in by_tool.items():
            prompt += f"\n{tool}:\n"
            for q in queries[:n_examples]:
                prompt += f"  - {q}\n"
        
        return prompt
    
    def _select_similar_examples(self, query: str, n_per_tool: int = 3) -> list[dict]:
        """Select most similar examples for each tool."""
        if self.embedding_model is None or self.example_embeddings is None:
            return self.training_examples
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = np.dot(self.example_embeddings, query_embedding)
        
        # Group by tool and select top-n for each
        selected = []
        tools = set(ex["tool"] for ex in self.training_examples)
        
        for tool in tools:
            tool_indices = [
                i for i, ex in enumerate(self.training_examples)
                if ex["tool"] == tool
            ]
            tool_similarities = [(i, similarities[i]) for i in tool_indices]
            tool_similarities.sort(key=lambda x: x[1], reverse=True)
            
            for idx, _ in tool_similarities[:n_per_tool]:
                selected.append(self.training_examples[idx])
        
        return selected
    
    def route(
        self,
        query: str,
        n_examples: int = 5,
        use_dynamic_examples: bool = True
    ) -> Optional[RoutingResult]:
        """Route a query to the appropriate tool."""
        system_prompt = self._build_system_prompt(
            n_examples=n_examples,
            query=query if use_dynamic_examples else None
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                return RoutingResult(
                    query=query,
                    tool=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments)
                )
            
            return None
            
        except Exception as e:
            print(f"Error routing query: {e}")
            return None
    
    def evaluate(
        self,
        test_queries: list[dict],
        n_examples: int = 5,
        use_dynamic_examples: bool = True
    ) -> dict:
        """Evaluate routing accuracy on test queries."""
        correct = 0
        results = []
        
        for test in test_queries:
            result = self.route(
                test["query"],
                n_examples=n_examples,
                use_dynamic_examples=use_dynamic_examples
            )
            
            predicted = result.tool if result else "none"
            expected = test["expected"]
            is_correct = predicted == expected
            
            if is_correct:
                correct += 1
            
            results.append({
                "query": test["query"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct
            })
        
        accuracy = correct / len(test_queries) if test_queries else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_queries),
            "results": results
        }


class MockRouter:
    """Mock router for testing without OpenAI API."""
    
    def __init__(self):
        self.keywords = {
            "search_documentation": ["documentation", "api", "reference", "guide", "endpoint", "parameter"],
            "search_code_examples": ["example", "code", "sample", "implement", "snippet", "python", "javascript"],
            "search_troubleshooting": ["error", "fail", "not working", "timeout", "403", "401", "500", "problem"],
            "clarify_question": ["help", "how do i", "what's the best", "can you", "assist"],
        }
    
    def route(self, query: str, **kwargs) -> RoutingResult:
        """Route based on keyword matching."""
        query_lower = query.lower()
        
        for tool, keywords in self.keywords.items():
            if any(kw in query_lower for kw in keywords):
                return RoutingResult(
                    query=query,
                    tool=tool,
                    arguments={"query": query}
                )
        
        return RoutingResult(
            query=query,
            tool="clarify_question",
            arguments={"clarification_question": "Could you please provide more details?"}
        )
    
    def evaluate(self, test_queries: list[dict], **kwargs) -> dict:
        """Evaluate mock routing accuracy."""
        correct = 0
        results = []
        
        for test in test_queries:
            result = self.route(test["query"])
            is_correct = result.tool == test["expected"]
            
            if is_correct:
                correct += 1
            
            results.append({
                "query": test["query"],
                "expected": test["expected"],
                "predicted": result.tool,
                "correct": is_correct
            })
        
        return {
            "accuracy": correct / len(test_queries) if test_queries else 0.0,
            "correct": correct,
            "total": len(test_queries),
            "results": results
        }


def main():
    """Run the router demo."""
    print("=" * 60)
    print("WEEK 6: QUERY ROUTER")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key and OPENAI_AVAILABLE:
        print("\nUsing OpenAI API for routing...")
        router = QueryRouter()
    else:
        print("\nNo OpenAI API key found. Using mock router...")
        print("Set OPENAI_API_KEY environment variable for full functionality.")
        router = MockRouter()
    
    # Test individual routing
    print("\n" + "-" * 60)
    print("ROUTING EXAMPLES")
    print("-" * 60)
    
    test_examples = [
        "How do I authenticate with the API?",
        "Show me a Python example for file upload",
        "Getting 401 error when calling the endpoint",
        "help me please",
    ]
    
    for query in test_examples:
        result = router.route(query)
        print(f"\nQuery: {query}")
        print(f"  -> Tool: {result.tool}")
        print(f"  -> Args: {result.arguments}")
    
    # Evaluate accuracy
    print("\n" + "-" * 60)
    print("EVALUATION RESULTS")
    print("-" * 60)
    
    eval_results = router.evaluate(TEST_QUERIES)
    
    print(f"\nAccuracy: {eval_results['accuracy']:.1%}")
    print(f"Correct: {eval_results['correct']}/{eval_results['total']}")
    
    # Show errors
    errors = [r for r in eval_results["results"] if not r["correct"]]
    if errors:
        print(f"\nMisclassified queries ({len(errors)}):")
        for err in errors:
            print(f"  Query: {err['query']}")
            print(f"    Expected: {err['expected']}, Got: {err['predicted']}")
    
    print("\n" + "=" * 60)
    
    return eval_results


if __name__ == "__main__":
    main()
