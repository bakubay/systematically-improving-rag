"""
Examples demonstrating the additive hooks system

This module shows how to use client-level and call-level hooks together
to achieve fine-grained control over API calls in the generation pipeline.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the hooks system
from core.hooks import (
    BaseHook, 
    HookContext, 
    HookPhase,
    LoggingHook,
    MetricsHook,
    RateLimitHook,
    RetryHook,
    create_standard_hooks,
    create_monitoring_hooks,
    create_reliability_hooks
)
from core.client import (
    InstructorHookableClient,
    ClientFactory,
    create_question_generation_hooks,
    create_summary_generation_hooks
)
from pipelines.generation import (
    generate_questions_pipeline,
    generate_summaries_pipeline
)


# Example 1: Custom hooks for specific use cases

class ConversationFilterHook(BaseHook):
    """Hook that filters conversations based on length or content"""
    
    def __init__(self, min_length: int = 100, max_length: int = 10000):
        super().__init__("conversation_filter", priority=150)
        self.min_length = min_length
        self.max_length = max_length
        self.filtered_count = 0
    
    async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
        if context.phase == HookPhase.PRE_REQUEST:
            # Check if this is a generation request with messages
            if "messages" in context.request_data.get("kwargs", {}):
                messages = context.request_data["kwargs"]["messages"]
                total_length = sum(len(msg.get("content", "")) for msg in messages)
                
                if total_length < self.min_length or total_length > self.max_length:
                    self.filtered_count += 1
                    context.metadata["filtered"] = True
                    context.metadata["filter_reason"] = f"Length {total_length} outside range [{self.min_length}, {self.max_length}]"
                    # Skip this request
                    return {"skip_remaining_hooks": True}
        
        return None


class ExperimentTrackingHook(BaseHook):
    """Hook that tracks experiment metadata and performance"""
    
    def __init__(self, experiment_id: str, track_tokens: bool = True):
        super().__init__(f"experiment_{experiment_id}", priority=80)
        self.experiment_id = experiment_id
        self.track_tokens = track_tokens
        self.call_data = []
    
    async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
        if context.phase == HookPhase.PRE_REQUEST:
            context.metadata["experiment_id"] = self.experiment_id
            context.metadata["call_start"] = context.metadata["timestamp"]
            
        elif context.phase == HookPhase.POST_REQUEST:
            call_duration = context.metadata["timestamp"] - context.metadata.get("call_start", 0)
            
            call_info = {
                "experiment_id": self.experiment_id,
                "duration": call_duration,
                "timestamp": context.metadata["timestamp"],
                "success": True
            }
            
            # Track token usage if available
            if self.track_tokens and context.response_data:
                result = context.response_data.get("result")
                if hasattr(result, "usage"):
                    call_info["tokens"] = result.usage
            
            self.call_data.append(call_info)
            
        elif context.phase == HookPhase.ON_ERROR:
            call_info = {
                "experiment_id": self.experiment_id,
                "duration": context.metadata["timestamp"] - context.metadata.get("call_start", 0),
                "timestamp": context.metadata["timestamp"],
                "success": False,
                "error": str(context.error)
            }
            self.call_data.append(call_info)
        
        return None
    
    def get_experiment_data(self) -> Dict[str, Any]:
        """Get collected experiment data"""
        successful_calls = [c for c in self.call_data if c["success"]]
        failed_calls = [c for c in self.call_data if not c["success"]]
        
        return {
            "experiment_id": self.experiment_id,
            "total_calls": len(self.call_data),
            "successful_calls": len(successful_calls),
            "failed_calls": len(failed_calls),
            "average_duration": sum(c["duration"] for c in successful_calls) / max(1, len(successful_calls)),
            "total_duration": sum(c["duration"] for c in self.call_data),
            "calls": self.call_data
        }


class ContentValidationHook(BaseHook):
    """Hook that validates generated content meets quality standards"""
    
    def __init__(self, min_queries: int = 3, max_queries: int = 8):
        super().__init__("content_validation", priority=60)
        self.min_queries = min_queries
        self.max_queries = max_queries
        self.validation_failures = 0
    
    async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
        if context.phase == HookPhase.POST_REQUEST:
            if context.response_data and "result" in context.response_data:
                result = context.response_data["result"]
                
                # Validate question generation
                if hasattr(result, "queries"):
                    query_count = len(result.queries)
                    if query_count < self.min_queries or query_count > self.max_queries:
                        self.validation_failures += 1
                        context.metadata["validation_failed"] = True
                        context.metadata["validation_reason"] = f"Query count {query_count} outside range [{self.min_queries}, {self.max_queries}]"
                
                # Validate summary generation
                elif hasattr(result, "summary"):
                    if len(result.summary.strip()) < 50:  # Minimum summary length
                        self.validation_failures += 1
                        context.metadata["validation_failed"] = True
                        context.metadata["validation_reason"] = "Summary too short"
        
        return None


# Example 2: Using client-level hooks for all calls

async def example_client_level_hooks():
    """Example showing client-level hooks that apply to all calls"""
    
    print("=== Example 2: Client-level hooks ===")
    
    # Create client-level hooks
    client_hooks = [
        LoggingHook(name="client_logging", log_level="INFO"),
        MetricsHook(name="client_metrics"),
        ExperimentTrackingHook("client_experiment_001"),
        RateLimitHook(calls_per_second=5.0)  # Apply rate limiting to all calls
    ]
    
    # Create client with hooks
    client = InstructorHookableClient(
        provider="openai/gpt-4.1-nano",
        async_client=True,
        client_hooks=client_hooks
    )
    
    print(f"Created client with {len(client_hooks)} client-level hooks")
    
    # All calls from this client will have these hooks applied
    # This would be used in the generation pipeline
    
    return client


# Example 3: Using call-level hooks for specific calls

async def example_call_level_hooks():
    """Example showing call-level hooks for specific operations"""
    
    print("=== Example 3: Call-level hooks ===")
    
    # Basic client without hooks
    client = InstructorHookableClient(
        provider="openai/gpt-4.1-nano",
        async_client=True
    )
    
    # Define call-specific hooks
    high_priority_hooks = [
        RetryHook(max_retries=5, backoff_factor=2.0),  # More aggressive retry
        ContentValidationHook(min_queries=5, max_queries=10)  # Stricter validation
    ]
    
    low_priority_hooks = [
        LoggingHook(name="low_priority_logging", log_level="WARNING"),  # Less verbose
        RetryHook(max_retries=1)  # Minimal retry
    ]
    
    print(f"Defined high-priority hooks: {[h.name for h in high_priority_hooks]}")
    print(f"Defined low-priority hooks: {[h.name for h in low_priority_hooks]}")
    
    return client, high_priority_hooks, low_priority_hooks


# Example 4: Additive hooks - combining client and call level

async def example_additive_hooks():
    """Example showing how client-level and call-level hooks combine additively"""
    
    print("=== Example 4: Additive hooks (Client + Call level) ===")
    
    # Client-level hooks (applied to ALL calls)
    client_hooks = [
        LoggingHook(name="global_logging", priority=100),
        MetricsHook(name="global_metrics", priority=90),
        ExperimentTrackingHook("additive_experiment_001", track_tokens=True)
    ]
    
    # Create client with client-level hooks
    client = InstructorHookableClient(
        provider="openai/gpt-4.1-nano",
        async_client=True,
        client_hooks=client_hooks
    )
    
    # Call-level hooks (applied to specific calls)
    special_call_hooks = [
        ConversationFilterHook(min_length=200, max_length=5000),  # Filter conversations
        ContentValidationHook(min_queries=4, max_queries=6),     # Validate output
        RateLimitHook(calls_per_second=2.0, priority=150)        # Override global rate limit
    ]
    
    print(f"Client has {len(client_hooks)} client-level hooks")
    print(f"Special calls will add {len(special_call_hooks)} call-level hooks")
    print(f"Total hooks for special calls: {len(client_hooks) + len(special_call_hooks)}")
    
    # When making a special call, BOTH sets of hooks will execute
    # The hooks will be sorted by priority and executed in order:
    # 1. RateLimitHook (priority 150) - call-level
    # 2. ConversationFilterHook (priority 150) - call-level  
    # 3. LoggingHook (priority 100) - client-level
    # 4. MetricsHook (priority 90) - client-level
    # 5. ExperimentTrackingHook (priority 80) - client-level
    # 6. ContentValidationHook (priority 60) - call-level
    
    return client, special_call_hooks


# Example 5: Using hooks in the generation pipeline

async def example_generation_pipeline_with_hooks():
    """Example showing how to use hooks in the actual generation pipeline"""
    
    print("=== Example 5: Generation pipeline with additive hooks ===")
    
    # Define client-level hooks for the entire pipeline
    pipeline_client_hooks = [
        LoggingHook(name="pipeline_logging"),
        MetricsHook(name="pipeline_metrics"),
        RateLimitHook(calls_per_second=8.0),  # Conservative rate limiting
        ExperimentTrackingHook("pipeline_experiment_v1")
    ]
    
    # Define call-level hooks for specific types of calls
    question_call_hooks = [
        ContentValidationHook(min_queries=4, max_queries=7),
        ConversationFilterHook(min_length=150)
    ]
    
    summary_call_hooks = [
        ContentValidationHook(),  # Default validation for summaries
        ConversationFilterHook(min_length=100, max_length=15000)  # Different limits for summaries
    ]
    
    # Example usage (this would be actual conversation hashes in practice)
    conversation_hashes = ["hash1", "hash2", "hash3"]
    db_path = Path("data/conversations.db")
    
    print("Question generation with additive hooks:")
    print(f"  Client hooks: {[h.name for h in pipeline_client_hooks]}")
    print(f"  Call hooks: {[h.name for h in question_call_hooks]}")
    
    # This call would use BOTH client-level and call-level hooks
    try:
        question_results = await generate_questions_pipeline(
            conversation_hashes=conversation_hashes,
            version="v3",
            db_path=db_path,
            experiment_id="additive_hooks_demo",
            client_hooks=pipeline_client_hooks,  # Applied to all calls
            call_hooks=question_call_hooks       # Applied additionally to each call
        )
        print(f"Question generation results: {question_results}")
    except Exception as e:
        print(f"Question generation failed (expected in demo): {e}")
    
    print("\nSummary generation with different call hooks:")
    print(f"  Client hooks: {[h.name for h in pipeline_client_hooks]} (same)")
    print(f"  Call hooks: {[h.name for h in summary_call_hooks]} (different)")
    
    # Same client hooks, different call hooks
    try:
        summary_results = await generate_summaries_pipeline(
            conversation_hashes=conversation_hashes,
            version="v2",
            db_path=db_path,
            experiment_id="additive_hooks_demo",
            client_hooks=pipeline_client_hooks,  # Same client hooks
            call_hooks=summary_call_hooks        # Different call hooks
        )
        print(f"Summary generation results: {summary_results}")
    except Exception as e:
        print(f"Summary generation failed (expected in demo): {e}")


# Example 6: Factory patterns for common hook combinations

def create_development_hooks() -> List[BaseHook]:
    """Create hooks suitable for development environment"""
    return [
        LoggingHook(name="dev_logging", log_level="DEBUG"),
        MetricsHook(name="dev_metrics"),
        RetryHook(max_retries=1),  # Fail fast in development
        ContentValidationHook()
    ]


def create_production_hooks(experiment_id: str) -> List[BaseHook]:
    """Create hooks suitable for production environment"""
    return [
        LoggingHook(name="prod_logging", log_level="WARNING"),  # Less verbose
        MetricsHook(name="prod_metrics"),
        RateLimitHook(calls_per_second=10.0),  # Reasonable rate limiting
        RetryHook(max_retries=3, backoff_factor=1.5),  # Robust retry
        ExperimentTrackingHook(experiment_id, track_tokens=True)
    ]


def create_research_hooks(experiment_id: str) -> List[BaseHook]:
    """Create hooks suitable for research experiments"""
    return [
        LoggingHook(name="research_logging", log_level="INFO"),
        MetricsHook(name="research_metrics"),
        ExperimentTrackingHook(experiment_id, track_tokens=True),
        ContentValidationHook(min_queries=5, max_queries=8),  # Strict validation
        ConversationFilterHook(min_length=200, max_length=8000)  # Quality filtering
    ]


async def example_factory_patterns():
    """Example showing factory patterns for different environments"""
    
    print("=== Example 6: Factory patterns for different environments ===")
    
    # Development environment
    dev_client = ClientFactory.create_basic_client()
    dev_client_hooks = create_development_hooks()
    for hook in dev_client_hooks:
        dev_client.add_hook(hook)
    print(f"Development client hooks: {[h.name for h in dev_client_hooks]}")
    
    # Production environment
    prod_client = ClientFactory.create_reliable_client(calls_per_second=10.0)
    prod_additional_hooks = create_production_hooks("prod_experiment_001")
    for hook in prod_additional_hooks:
        prod_client.add_hook(hook)
    print(f"Production client hooks: {[h.name for h in prod_additional_hooks]}")
    
    # Research environment
    research_client = ClientFactory.create_monitored_client()
    research_hooks = create_research_hooks("research_experiment_001")
    for hook in research_hooks:
        research_client.add_hook(hook)
    print(f"Research client hooks: {[h.name for h in research_hooks]}")


# Example 7: Dynamic hook management

async def example_dynamic_hooks():
    """Example showing dynamic addition/removal of hooks"""
    
    print("=== Example 7: Dynamic hook management ===")
    
    # Start with basic client
    client = ClientFactory.create_basic_client()
    print(f"Initial hooks: {len(client.hook_manager.client_hooks)}")
    
    # Add monitoring during runtime
    client.add_hook(LoggingHook("runtime_logging"))
    client.add_hook(MetricsHook("runtime_metrics"))
    print(f"After adding monitoring: {len(client.hook_manager.client_hooks)} hooks")
    
    # Add experiment tracking
    experiment_hook = ExperimentTrackingHook("dynamic_experiment_001")
    client.add_hook(experiment_hook)
    print(f"After adding experiment tracking: {len(client.hook_manager.client_hooks)} hooks")
    
    # Remove specific hook
    client.remove_hook("runtime_logging")
    print(f"After removing logging: {len(client.hook_manager.client_hooks)} hooks")
    
    # Show remaining hooks
    remaining_hooks = [h.name for h in client.hook_manager.client_hooks]
    print(f"Remaining hooks: {remaining_hooks}")


# Main demonstration function

async def main():
    """Run all examples to demonstrate the additive hooks system"""
    
    print("Additive Hooks System Demonstration")
    print("=" * 50)
    
    # Run all examples
    await example_client_level_hooks()
    print()
    
    await example_call_level_hooks()
    print()
    
    await example_additive_hooks()
    print()
    
    await example_generation_pipeline_with_hooks()
    print()
    
    await example_factory_patterns()
    print()
    
    await example_dynamic_hooks()
    print()
    
    print("=" * 50)
    print("Key Benefits of the Additive Hooks System:")
    print("1. Client-level hooks apply to ALL calls from a client")
    print("2. Call-level hooks apply to SPECIFIC calls")
    print("3. Both sets combine additively - you get the union of both")
    print("4. Hooks are executed in priority order (higher priority first)")
    print("5. Hooks can modify requests, responses, and control flow")
    print("6. Easy to add/remove hooks dynamically")
    print("7. Factory patterns for common configurations")
    print("8. Custom hooks for specific use cases")


if __name__ == "__main__":
    asyncio.run(main())