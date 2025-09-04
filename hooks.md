# Hooks System Documentation

This hooks system provides flexible control over API calls by allowing you to define hooks at both the **client level** (applied to all calls from a client) and **per-create level** (applied to specific calls). The system supports additive composition, meaning both types of hooks can work together to provide maximum flexibility for logging, metrics, debugging, and other cross-cutting concerns.

## Overview

The hooks system is built around two key concepts:

- **Client-level hooks**: Applied to ALL calls made by a client
- **Call-level hooks**: Applied to SPECIFIC calls  
- **Additive composition**: When both are present, they combine (you get the union of both sets)

### Key Components

- **`BaseHook`**: Base class for creating custom hooks with priority system
- **`HookContext`**: Contains request/response data and metadata
- **`InstructorHookableClient`**: Client that supports both global and call-specific hooks
- **Hook Phases**: Different execution points (PRE_REQUEST, POST_REQUEST, ON_ERROR, ON_RETRY)

## Basic Usage

### 1. Client-Level Hooks (Global)

Client-level hooks are executed for every API call made by that client:

```python
from core.client import InstructorHookableClient
from core.hooks import LoggingHook, MetricsHook, RateLimitHook

# Define client-level hooks (applied to ALL calls)
client_hooks = [
    LoggingHook(name="global_logging"),
    MetricsHook(name="global_metrics"), 
    RateLimitHook(calls_per_second=10.0)
]

# Create client with hooks
client = InstructorHookableClient(
    provider="openai/gpt-4.1-nano",
    async_client=True,
    client_hooks=client_hooks
)

# All calls will use these hooks
result1 = await client.create(messages=messages1)  # Uses client hooks
result2 = await client.create(messages=messages2)  # Uses client hooks
```

### 2. Call-Level Hooks (Per-Create)

Call-level hooks are applied to specific API calls:

```python
from core.hooks import RetryHook, ContentValidationHook

# Define call-specific hooks
call_hooks = [
    RetryHook(max_retries=5),
    ContentValidationHook(min_queries=4)
]

# Use in specific generation calls
result = await generate_questions_pipeline(
    conversation_hashes=hashes,
    version="v3",
    db_path=db_path,
    call_hooks=call_hooks  # Applied only to this pipeline
)
```

### 3. Additive Behavior (Combined Hooks)

When you have both client-level and call-level hooks, they combine additively:

```python
# Client has these hooks:
client_hooks = [
    LoggingHook(priority=100),
    MetricsHook(priority=90),
    RateLimitHook(priority=200)
]

# Call adds these hooks:  
call_hooks = [
    RetryHook(priority=50),
    ValidationHook(priority=60)
]

# Total hooks executed (in priority order):
# 1. RateLimitHook (priority 200) - client-level
# 2. LoggingHook (priority 100) - client-level  
# 3. MetricsHook (priority 90) - client-level
# 4. ValidationHook (priority 60) - call-level
# 5. RetryHook (priority 50) - call-level
```

## Hook Phases

Hooks can execute at different phases of an API call:

- **`PRE_REQUEST`**: Before the API call is made
- **`POST_REQUEST`**: After successful API response  
- **`ON_ERROR`**: When an error occurs
- **`ON_RETRY`**: When a retry is attempted

```python
class CustomHook(BaseHook):
    def __init__(self):
        super().__init__("custom_hook", priority=100)
    
    async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
        if context.phase == HookPhase.PRE_REQUEST:
            # Pre-processing logic
            pass
        elif context.phase == HookPhase.POST_REQUEST:
            # Post-processing logic
            pass
        elif context.phase == HookPhase.ON_ERROR:
            # Error handling logic
            pass
        return None
```

## Built-in Hooks

### LoggingHook
Logs API requests and responses at configurable levels:

```python
LoggingHook(name="my_logger", log_level="INFO", priority=100)
```

### MetricsHook
Collects metrics on API calls (count, duration, errors):

```python
metrics_hook = MetricsHook(name="my_metrics", priority=90)
# Later: metrics = metrics_hook.get_metrics()
```

### RateLimitHook
Implements rate limiting with configurable calls per second:

```python
RateLimitHook(calls_per_second=5.0, priority=200)
```

### RetryHook
Implements retry logic with exponential backoff:

```python
RetryHook(max_retries=3, backoff_factor=1.5, priority=50)
```

## Custom Hooks

Create custom hooks by extending `BaseHook`:

```python
from core.hooks import BaseHook, HookContext, HookPhase

class CustomValidationHook(BaseHook):
    def __init__(self, min_length: int = 100):
        super().__init__("custom_validation", priority=70)
        self.min_length = min_length
    
    async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
        if context.phase == HookPhase.POST_REQUEST:
            # Validate response
            if context.response_data:
                result = context.response_data.get("result")
                if hasattr(result, "queries"):
                    # Custom validation logic here
                    if len(result.queries) < self.min_length:
                        return {"should_retry": True}
        return None
```

## Integration with Generation Pipelines

The generation pipelines support both client-level and call-level hooks:

```python
# Question generation with both hook types
await generate_questions_pipeline(
    conversation_hashes=hashes,
    version="v3", 
    db_path=db_path,
    client_hooks=[LoggingHook(), MetricsHook()],     # Applied to all calls
    call_hooks=[ValidationHook(), FilterHook()]      # Applied to each call
)

# Summary generation with different call hooks
await generate_summaries_pipeline(
    conversation_hashes=hashes,
    version="v2",
    db_path=db_path, 
    client_hooks=[LoggingHook(), MetricsHook()],     # Same client hooks
    call_hooks=[DifferentValidationHook()]           # Different call hooks
)
```

## Integration with Braintrust-Style Evaluations

This hooks system works seamlessly with existing Braintrust evaluation patterns:

```python
# Your existing task function pattern
async def task(query, hooks):
    # Process the query
    result = await process_query(query)
    
    # Call hooks.meta() as before
    hooks.meta(input=query, output=result)
    
    return result

# Now you can also use the client pattern with additive hooks
client = InstructorHookableClient(
    provider="openai/gpt-4.1-nano",
    async_client=True,
    client_hooks=[LoggingHook(), MetricsHook()]  # Global hooks
)

# Add call-specific hooks for this evaluation
call_hooks = [RetryHook(max_retries=5)]

# Both client and call hooks will be executed
result = await client.create(query, hooks=call_hooks)
```

## Factory Patterns

Use factory functions for common hook combinations:

```python
from core.client import ClientFactory

# Pre-configured clients
basic_client = ClientFactory.create_basic_client()
monitored_client = ClientFactory.create_monitored_client() 
reliable_client = ClientFactory.create_reliable_client(calls_per_second=5.0)
full_client = ClientFactory.create_full_featured_client()
```

## Hook Priority System

Hooks execute in priority order (higher numbers execute first):

- **`200+`**: Critical hooks (rate limiting, authentication)
- **`100-199`**: Monitoring and logging hooks
- **`50-99`**: Business logic hooks
- **`0-49`**: Cleanup and finalization hooks

```python
# Example priority ordering
hooks = [
    RateLimitHook(priority=200),      # Executes first
    LoggingHook(priority=100),        # Executes second
    ValidationHook(priority=60),      # Executes third
    CleanupHook(priority=10)          # Executes last
]
```

## Advanced Features

### Hook Communication
Hooks can communicate via the context metadata:

```python
# Hook A sets metadata
context.metadata["custom_flag"] = True

# Hook B reads metadata  
if context.metadata.get("custom_flag"):
    # Do something special
    pass
```

### Conditional Execution
Control when hooks execute:

```python
def should_execute(self, context: HookContext) -> bool:
    # Only execute for specific request types
    return context.request_data.get("method") == "chat.completions.create"
```

### Flow Control
Hooks can control execution flow:

```python
return {
    "skip_remaining_hooks": True,  # Stop executing other hooks
    "should_retry": True,          # Trigger retry logic
    "request_data": {...}          # Modify request
}
```

## Best Practices

### 1. Use Appropriate Priorities
- Critical functionality (auth, rate limiting): 200+
- Monitoring and logging: 100-199  
- Business logic: 50-99
- Cleanup: 0-49

### 2. Client vs Call Level Hooks
- **Client-level**: Cross-cutting concerns (logging, metrics, rate limiting)
- **Call-level**: Specific requirements (validation, filtering, special retry logic)

### 3. Hook Naming
Use descriptive names that indicate purpose and scope:

```python
LoggingHook("experiment_001_logging")
MetricsHook("production_metrics") 
RetryHook("high_priority_retry")
```

### 4. Error Handling
Hooks should handle their own errors gracefully:

```python
async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
    try:
        # Hook logic here
        pass
    except Exception as e:
        logger.error(f"Hook {self.name} failed: {e}")
        # Don't re-raise unless critical
    return None
```

## Migration Guide

### From Basic Instructor Client

```python
# Before
client = instructor.from_provider("openai/gpt-4.1-nano", async_client=True)

# After  
client = InstructorHookableClient(
    provider="openai/gpt-4.1-nano",
    async_client=True,
    client_hooks=[LoggingHook(), MetricsHook()]
)
```

### Updating Generation Pipelines

```python
# Before
results = await generate_questions_pipeline(
    conversation_hashes=hashes,
    version="v3",
    db_path=db_path
)

# After
results = await generate_questions_pipeline(
    conversation_hashes=hashes, 
    version="v3",
    db_path=db_path,
    client_hooks=[LoggingHook(), MetricsHook()],  # Applied to all calls
    call_hooks=[ValidationHook()]                 # Applied to each call
)
```

## Example: Complete Usage

```python
from core.client import InstructorHookableClient
from core.hooks import LoggingHook, MetricsHook, RetryHook, RateLimitHook

# Create client with global hooks
client_hooks = [
    LoggingHook("global_logging", priority=100),
    MetricsHook("global_metrics", priority=90),
    RateLimitHook(calls_per_second=10.0, priority=200)
]

client = InstructorHookableClient(
    provider="openai/gpt-4.1-nano",
    async_client=True,
    client_hooks=client_hooks
)

# Create call-specific hooks for special cases
special_call_hooks = [
    RetryHook(max_retries=5, priority=50),
    CustomValidationHook(min_length=100, priority=60)
]

# Regular call - only client hooks execute
result1 = await client.create(messages=messages1)

# Special call - both client and call hooks execute
result2 = await client.create(messages=messages2, hooks=special_call_hooks)

# Pipeline usage with both hook types
results = await generate_questions_pipeline(
    conversation_hashes=hashes,
    version="v3",
    db_path=db_path,
    client_hooks=client_hooks,      # Applied to all calls in pipeline
    call_hooks=special_call_hooks   # Applied additionally to each call
)
```

## Benefits

1. **Flexible**: Support both global and per-call hooks
2. **Additive**: Combine different hook types seamlessly
3. **Priority-based**: Control execution order with priority system
4. **Phase-aware**: Execute hooks at different API call phases
5. **Compatible**: Works with existing Braintrust evaluation patterns
6. **Extensible**: Easy to add new hook types and functionality
7. **Composable**: Hooks can be combined and reused across different contexts

This additive hooks system provides the fine-grained control needed for complex evaluation and generation pipelines while maintaining clean separation of concerns between global and call-specific functionality.