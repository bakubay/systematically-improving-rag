"""
Week 7: Production Caching System

Multi-level caching for RAG systems:
- Level 1: In-memory cache (fastest)
- Level 2: Redis cache (persistent)
- Level 3: Semantic cache (similarity-based)
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import OrderedDict

# Check for optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class CacheEntry:
    """A cached item with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Time to live in seconds
    hits: int = 0
    
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass
    
    @abstractmethod
    def stats(self) -> dict:
        pass


class InMemoryCache(CacheBackend):
    """Fast in-memory LRU cache."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.hits += 1
            self.hits += 1
            return entry.value
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        # Evict if at capacity
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = CacheEntry(key=key, value=value, ttl=ttl)
        self.cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "type": "in_memory",
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }


class RedisCache(CacheBackend):
    """Redis-backed cache for persistence."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "rag:"
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis required. Run: uv add redis")
        
        self.client = redis.Redis(host=host, port=port, db=db)
        self.prefix = prefix
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        try:
            data = self.client.get(self._make_key(key))
            if data:
                self.hits += 1
                return json.loads(data)
            self.misses += 1
            return None
        except Exception:
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        try:
            data = json.dumps(value)
            if ttl:
                self.client.setex(self._make_key(key), int(ttl), data)
            else:
                self.client.set(self._make_key(key), data)
        except Exception as e:
            print(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(self._make_key(key)))
        except Exception:
            return False
    
    def clear(self) -> None:
        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
        except Exception:
            pass
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> dict:
        total = self.hits + self.misses
        try:
            size = len(self.client.keys(f"{self.prefix}*"))
        except Exception:
            size = 0
        
        return {
            "type": "redis",
            "size": size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }


class SemanticCache:
    """Cache that retrieves similar queries."""
    
    def __init__(self, similarity_threshold: float = 0.95, max_size: int = 1000):
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required. Run: uv add numpy")
        
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.embeddings: list[np.ndarray] = []
        self.responses: list[Any] = []
        self.queries: list[str] = []
        self.hits = 0
        self.misses = 0
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def get(self, query_embedding: np.ndarray) -> Optional[Any]:
        """Find cached response for similar query."""
        if not self.embeddings:
            self.misses += 1
            return None
        
        # Calculate similarities
        similarities = [
            self._cosine_similarity(query_embedding, emb)
            for emb in self.embeddings
        ]
        
        max_sim = max(similarities)
        if max_sim >= self.threshold:
            idx = similarities.index(max_sim)
            self.hits += 1
            return self.responses[idx]
        
        self.misses += 1
        return None
    
    def set(self, query: str, query_embedding: np.ndarray, response: Any) -> None:
        """Cache a response with its embedding."""
        # Evict oldest if at capacity
        while len(self.embeddings) >= self.max_size:
            self.embeddings.pop(0)
            self.responses.pop(0)
            self.queries.pop(0)
        
        self.embeddings.append(query_embedding)
        self.responses.append(response)
        self.queries.append(query)
    
    def clear(self) -> None:
        self.embeddings.clear()
        self.responses.clear()
        self.queries.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "type": "semantic",
            "size": len(self.embeddings),
            "max_size": self.max_size,
            "threshold": self.threshold,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }


class MultiLevelCache:
    """Multi-level caching system combining memory, Redis, and semantic caches."""
    
    def __init__(
        self,
        use_memory: bool = True,
        use_redis: bool = False,
        use_semantic: bool = True,
        memory_size: int = 1000,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        semantic_threshold: float = 0.95,
    ):
        self.levels: list[tuple[str, CacheBackend]] = []
        
        if use_memory:
            self.memory_cache = InMemoryCache(max_size=memory_size)
            self.levels.append(("memory", self.memory_cache))
        else:
            self.memory_cache = None
        
        if use_redis and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(host=redis_host, port=redis_port)
                self.levels.append(("redis", self.redis_cache))
            except Exception as e:
                print(f"Redis not available: {e}")
                self.redis_cache = None
        else:
            self.redis_cache = None
        
        if use_semantic and NUMPY_AVAILABLE:
            self.semantic_cache = SemanticCache(similarity_threshold=semantic_threshold)
        else:
            self.semantic_cache = None
    
    @staticmethod
    def _make_key(text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Any]:
        """Get from cache, checking each level."""
        key = self._make_key(query)
        
        for level_name, cache in self.levels:
            result = cache.get(key)
            if result is not None:
                # Populate higher levels
                for prev_name, prev_cache in self.levels:
                    if prev_name == level_name:
                        break
                    prev_cache.set(key, result)
                return result
        
        return None
    
    def get_semantic(self, query_embedding) -> Optional[Any]:
        """Get from semantic cache."""
        if self.semantic_cache:
            return self.semantic_cache.get(query_embedding)
        return None
    
    def set(self, query: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set in all cache levels."""
        key = self._make_key(query)
        
        for _, cache in self.levels:
            cache.set(key, value, ttl)
    
    def set_semantic(self, query: str, query_embedding, response: Any) -> None:
        """Set in semantic cache."""
        if self.semantic_cache:
            self.semantic_cache.set(query, query_embedding, response)
    
    def delete(self, query: str) -> None:
        """Delete from all cache levels."""
        key = self._make_key(query)
        for _, cache in self.levels:
            cache.delete(key)
    
    def clear(self) -> None:
        """Clear all cache levels."""
        for _, cache in self.levels:
            cache.clear()
        if self.semantic_cache:
            self.semantic_cache.clear()
    
    def stats(self) -> dict:
        """Get statistics from all levels."""
        stats = {"levels": {}}
        
        for name, cache in self.levels:
            stats["levels"][name] = cache.stats()
        
        if self.semantic_cache:
            stats["levels"]["semantic"] = self.semantic_cache.stats()
        
        # Calculate overall hit rate
        total_hits = sum(s.get("hits", 0) for s in stats["levels"].values())
        total_misses = sum(s.get("misses", 0) for s in stats["levels"].values())
        total = total_hits + total_misses
        
        stats["total_hits"] = total_hits
        stats["total_misses"] = total_misses
        stats["overall_hit_rate"] = total_hits / total if total > 0 else 0.0
        
        return stats
    
    def print_stats(self) -> None:
        """Print formatted statistics."""
        stats = self.stats()
        
        print("\n" + "=" * 50)
        print("CACHE STATISTICS")
        print("=" * 50)
        
        for level, level_stats in stats["levels"].items():
            print(f"\n{level.upper()} Cache:")
            print(f"  Size: {level_stats.get('size', 'N/A')}")
            print(f"  Hits: {level_stats.get('hits', 0)}")
            print(f"  Misses: {level_stats.get('misses', 0)}")
            print(f"  Hit Rate: {level_stats.get('hit_rate', 0):.1%}")
        
        print("\nOVERALL:")
        print(f"  Total Hits: {stats['total_hits']}")
        print(f"  Total Misses: {stats['total_misses']}")
        print(f"  Hit Rate: {stats['overall_hit_rate']:.1%}")
        print("=" * 50)


class CostTracker:
    """Track costs per query."""
    
    # Pricing per 1M tokens (approximate)
    PRICING = {
        "gpt-5.2": {"input": 0.15, "output": 0.60},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    }

    # A rough “where the money goes” breakdown for typical RAG systems.
    # This is not universal. It’s a starting point for thinking.
    # - LLM generation usually dominates.
    # - Retrieval infra is often second.
    # - Embeddings are often a smaller slice after initial indexing.
    DEFAULT_COST_BREAKDOWN = {
        "embedding_generation": 0.08,
        "retrieval_infra": 0.15,
        "llm_generation": 0.70,
        "logging_monitoring": 0.07,
    }
    
    def __init__(self):
        self.queries: list[dict] = []
    
    def track(
        self,
        query: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cached: bool = False,
        latency_ms: float = 0.0,
    ) -> float:
        """Track a query and return its cost."""
        if cached:
            cost = 0.0
        else:
            pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.01})
            cost = (tokens_in / 1_000_000) * pricing["input"]
            cost += (tokens_out / 1_000_000) * pricing["output"]
        
        self.queries.append({
            "query": query[:100],
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": cost,
            "cached": cached,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        })
        
        return cost
    
    def total_cost(self) -> float:
        return sum(q["cost"] for q in self.queries)
    
    def avg_cost_per_query(self) -> float:
        if not self.queries:
            return 0.0
        return self.total_cost() / len(self.queries)
    
    def cache_savings(self) -> float:
        """Estimate savings from cached queries."""
        cached_queries = [q for q in self.queries if q["cached"]]
        if not cached_queries:
            return 0.0
        
        # Estimate what they would have cost
        estimated_cost = 0.0
        for q in cached_queries:
            pricing = self.PRICING.get(q["model"], {"input": 0.01, "output": 0.01})
            estimated_cost += (q["tokens_in"] / 1_000_000) * pricing["input"]
            estimated_cost += (q["tokens_out"] / 1_000_000) * pricing["output"]
        
        return estimated_cost
    
    def stats(self) -> dict:
        total = len(self.queries)
        cached = sum(1 for q in self.queries if q["cached"])
        
        return {
            "total_queries": total,
            "cached_queries": cached,
            "cache_rate": cached / total if total > 0 else 0.0,
            "total_cost": self.total_cost(),
            "avg_cost_per_query": self.avg_cost_per_query(),
            "estimated_savings": self.cache_savings(),
        }
    
    def print_report(self) -> None:
        stats = self.stats()
        
        print("\n" + "=" * 50)
        print("COST TRACKING REPORT")
        print("=" * 50)
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Cached Queries: {stats['cached_queries']} ({stats['cache_rate']:.1%})")
        print(f"Total Cost: ${stats['total_cost']:.4f}")
        print(f"Avg Cost/Query: ${stats['avg_cost_per_query']:.6f}")
        print(f"Estimated Savings: ${stats['estimated_savings']:.4f}")
        print("=" * 50)

        print("\n" + "-" * 50)
        print("TOKEN ECONOMICS (Quick Guide)")
        print("-" * 50)
        print("Rule of thumb: cost scales with tokens and with how often you call the model.")
        print("If you cut tokens by 30%, you usually cut cost by about 30%.")
        print("If caching avoids 30% of calls, you usually cut cost by about 30%.")

        breakdown = self.DEFAULT_COST_BREAKDOWN
        print("\nTypical cost breakdown (starting guess):")
        for name, frac in breakdown.items():
            print(f"  - {name}: {frac:.0%}")

    def estimate_monthly_cost(
        self,
        queries_per_day: int,
        avg_tokens_in: int,
        avg_tokens_out: int,
        model: str = "gpt-5.2",
        cache_hit_rate: float = 0.0,
    ) -> dict:
        """Estimate monthly cost from token counts.

        This is a simple calculator meant for planning.
        """
        cache_hit_rate = max(0.0, min(1.0, cache_hit_rate))
        paid_queries_per_day = int(round(queries_per_day * (1.0 - cache_hit_rate)))

        pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.01})
        cost_per_query = (avg_tokens_in / 1_000_000) * pricing["input"]
        cost_per_query += (avg_tokens_out / 1_000_000) * pricing["output"]

        monthly = paid_queries_per_day * 30 * cost_per_query
        return {
            "model": model,
            "queries_per_day": queries_per_day,
            "paid_queries_per_day": paid_queries_per_day,
            "cache_hit_rate": cache_hit_rate,
            "avg_tokens_in": avg_tokens_in,
            "avg_tokens_out": avg_tokens_out,
            "estimated_cost_per_query": cost_per_query,
            "estimated_monthly_cost": monthly,
        }


def main():
    """Demo the caching system."""
    print("=" * 60)
    print("WEEK 7: PRODUCTION CACHING SYSTEM")
    print("=" * 60)
    
    # Initialize multi-level cache (without Redis for demo)
    cache = MultiLevelCache(
        use_memory=True,
        use_redis=False,  # Set to True if Redis is available
        use_semantic=True,
        memory_size=100,
        semantic_threshold=0.95,
    )
    
    # Initialize cost tracker
    cost_tracker = CostTracker()
    
    # Simulate queries
    test_queries = [
        "What is RAG?",
        "How do embeddings work?",
        "What is RAG?",  # Duplicate - should hit cache
        "Explain vector databases",
        "How do embeddings work?",  # Duplicate
        "What is semantic search?",
        "What is RAG?",  # Duplicate
        "Explain retrieval augmented generation",  # Similar to "What is RAG?"
    ]
    
    print("\nSimulating queries...")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        
        # Check cache
        cached_response = cache.get(query)
        
        if cached_response:
            print(f"{i}. CACHE HIT: {query[:40]}...")
            cost_tracker.track(
                query=query,
                model="gpt-5.2",
                tokens_in=50,
                tokens_out=200,
                cached=True,
                latency_ms=(time.time() - start_time) * 1000
            )
        else:
            print(f"{i}. CACHE MISS: {query[:40]}...")
            # Simulate generating response
            response = f"Response to: {query}"
            cache.set(query, response, ttl=3600)
            
            cost_tracker.track(
                query=query,
                model="gpt-5.2",
                tokens_in=50,
                tokens_out=200,
                cached=False,
                latency_ms=(time.time() - start_time) * 1000 + 500  # Add simulated API latency
            )
    
    # Print statistics
    cache.print_stats()
    cost_tracker.print_report()
    
    # Test semantic cache separately
    if NUMPY_AVAILABLE:
        print("\n" + "-" * 60)
        print("SEMANTIC CACHE TEST")
        print("-" * 60)
        
        semantic = SemanticCache(similarity_threshold=0.9)
        
        # Simulate embeddings (random for demo)
        np.random.seed(42)
        
        # Cache a response
        emb1 = np.random.randn(384)
        emb1 = emb1 / np.linalg.norm(emb1)
        semantic.set("What is RAG?", emb1, "RAG is Retrieval Augmented Generation...")
        
        # Query with similar embedding
        emb2 = emb1 + np.random.randn(384) * 0.05  # Small perturbation
        emb2 = emb2 / np.linalg.norm(emb2)
        
        result = semantic.get(emb2)
        if result:
            print(f"Similar query hit: {result[:50]}...")
        else:
            print("No similar query found")
        
        # Query with different embedding
        emb3 = np.random.randn(384)
        emb3 = emb3 / np.linalg.norm(emb3)
        
        result = semantic.get(emb3)
        if result:
            print(f"Different query hit: {result[:50]}...")
        else:
            print("Different query: no match (expected)")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
