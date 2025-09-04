# Additive Hooks System Documentation

This document describes the additive hooks system that provides fine-grained control over API calls in the generation pipeline. The system allows you to combine client-level hooks (applied to all calls from a client) with call-level hooks (applied to specific calls) in an additive fashion.

## Overview

The hooks system is designed around the principle of **additive composition**. This means:

- **Client-level hooks** are applied to ALL calls made by a client
- **Call-level hooks** are applied to SPECIFIC calls  
- When both are present, they combine additively (you get the union of both sets)
- Hooks are executed in priority order (higher priority numbers execute first)

## Core Concepts

### Hook Phases

Hooks can execute at different phases of an API call:

- `PRE_REQUEST`: Before the API call is made
- `POST_REQUEST`: After successful API response  
- `ON_ERROR`: When an error occurs
- `ON_RETRY`: When a retry is attempted

### Hook Context

All hooks receive a `HookContext` object containing:

- `phase`: Current execution phase
- `request_data`: Information about the API request
- `response_data`: Information about the API response (if available)
- `error`: Exception information (if in error phase)
- `metadata`: Additional metadata that hooks can read/write

### Hook Priority

Hooks have a priority system where higher numbers execute first:

- `200+`: Critical hooks (rate limiting, authentication)
- `100-199`: Monitoring and logging hooks
- `50-99`: Business logic hooks
- `0-49`: Cleanup and finalization hooks

## Basic Usage

### Creating a Client with Client-Level Hooks

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
```

### Using Call-Level Hooks

```python
from core.hooks import RetryHook, ContentValidationHook

# Define call-specific hooks
call_hooks = [
    RetryHook(max_retries=5),
    ContentValidationHook(min_queries=4)
]

# Use in generation pipeline
results = await generate_questions_pipeline(
    conversation_hashes=hashes,
    version="v3",
    db_path=db_path,
    client_hooks=client_hooks,  # Applied to all calls
    call_hooks=call_hooks       # Applied additionally to each call
)
```

### Additive Behavior

When you have both client-level and call-level hooks, they combine:

```python
# Client has these hooks:
client_hooks = [LoggingHook(), MetricsHook(), RateLimitHook()]

# Call adds these hooks:  
call_hooks = [RetryHook(), ValidationHook()]

# Total hooks executed (in priority order):
# 1. RateLimitHook (priority 200) - client-level
# 2. LoggingHook (priority 100) - client-level  
# 3. MetricsHook (priority 90) - client-level
# 4. RetryHook (priority 50) - call-level
# 5. ValidationHook (priority 60) - call-level
```

## Built-in Hooks

### LoggingHook
Logs API requests and responses at configurable levels.

```python
LoggingHook(name="my_logger", log_level="INFO")
```

### MetricsHook
Collects metrics on API calls (count, duration, errors).

```python
metrics_hook = MetricsHook(name="my_metrics")
# Later: metrics = metrics_hook.get_metrics()
```

### RateLimitHook
Implements rate limiting with configurable calls per second.

```python
RateLimitHook(calls_per_second=5.0)
```

### RetryHook
Implements retry logic with exponential backoff.

```python
RetryHook(max_retries=3, backoff_factor=1.5)
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
                    pass
        return None
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

## Generation Pipeline Integration

The generation pipelines support both client-level and call-level hooks:

```python
# Question generation with hooks
await generate_questions_pipeline(
    conversation_hashes=hashes,
    version="v3", 
    db_path=db_path,
    client_hooks=[LoggingHook(), MetricsHook()],     # All calls
    call_hooks=[ValidationHook(), FilterHook()]      # Each call
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

## Dynamic Hook Management

Add or remove hooks at runtime:

```python
client = ClientFactory.create_basic_client()

# Add hooks dynamically
client.add_hook(LoggingHook("runtime_logging"))
client.add_hook(MetricsHook("runtime_metrics"))

# Remove hooks by name
client.remove_hook("runtime_logging")

# Check current hooks
current_hooks = client.hook_manager.client_hooks
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

### 5. Environment-Specific Configurations

```python
def create_development_hooks():
    return [
        LoggingHook(log_level="DEBUG"),
        MetricsHook(),
        RetryHook(max_retries=1)  # Fail fast
    ]

def create_production_hooks():
    return [
        LoggingHook(log_level="WARNING"), 
        MetricsHook(),
        RateLimitHook(calls_per_second=10.0),
        RetryHook(max_retries=3)
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

## Examples

See `examples/hooks_usage.py` for comprehensive examples including:

- Client-level hooks for global concerns
- Call-level hooks for specific operations  
- Additive composition of both types
- Custom hooks for domain-specific logic
- Factory patterns for common configurations
- Dynamic hook management
- Integration with generation pipelines

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
    client_hooks=[LoggingHook(), MetricsHook()],
    call_hooks=[ValidationHook()]
)
```

This additive hooks system provides the fine-grained control you requested, allowing you to combine client-level and call-level hooks for maximum flexibility while maintaining clean separation of concerns.