# Hooks System Documentation

This hooks system allows you to define hooks at both the client level and per individual `create` call, providing maximum flexibility for logging, metrics, debugging, and other cross-cutting concerns.

## Overview

The hooks system consists of two main classes:

- **`Hooks`**: A container for hook functions that can be executed with metadata
- **`Client`**: A base client class that supports both global and call-specific hooks

## Basic Usage

### 1. Creating a Hooks Object

```python
from hooks import Hooks

# Create a hooks container
call_hooks = Hooks()

# Add hook functions
def my_hook(**kwargs):
    print(f"Hook called with: {kwargs}")

call_hooks.add_hook(my_hook)
```

### 2. Using Hooks with a Client

```python
from hooks import Client, Hooks

class MyClient(Client):
    def _create_impl(self, data, hooks=None, **kwargs):
        # Your implementation here
        result = f"processed: {data}"
        
        # Call hooks with metadata
        if hooks:
            hooks.meta(input=data, output=result)
        
        return result

# Create client and add global hooks
client = MyClient()
client.hooks.add_hook(lambda **kwargs: print(f"Global: {kwargs}"))

# Create call-specific hooks
call_hooks = Hooks()
call_hooks.add_hook(lambda **kwargs: print(f"Call: {kwargs}"))

# Use both types of hooks
result = client.create("some data", hooks=call_hooks)
```

## Key Features

### 1. Client-Level Hooks

Hooks defined at the client level are executed for every `create` call:

```python
client = MyClient()
client.hooks.add_hook(logging_hook)
client.hooks.add_hook(metrics_hook)

# These hooks will run for all create calls
result1 = client.create("query 1")
result2 = client.create("query 2")
```

### 2. Call-Specific Hooks

You can add hooks for individual `create` calls:

```python
call_hooks = Hooks()
call_hooks.add_hook(debug_hook)

# This debug hook only runs for this specific call
result = client.create("special query", hooks=call_hooks)
```

### 3. Combined Hooks

When both client-level and call-specific hooks are present, all hooks are executed:

```python
# Client has global logging hook
client.hooks.add_hook(logging_hook)

# Add call-specific debug hook
call_hooks = Hooks()
call_hooks.add_hook(debug_hook)

# Both logging_hook and debug_hook will be executed
result = client.create("query", hooks=call_hooks)
```

## Hook Function Signature

Hook functions should accept keyword arguments:

```python
def my_hook(**kwargs):
    # Access metadata
    input_data = kwargs.get('input')
    output_data = kwargs.get('output')
    client_name = kwargs.get('client')
    
    # Your hook logic here
    print(f"Processing {input_data} -> {output_data}")
```

## Common Hook Patterns

### Logging Hook

```python
def logging_hook(**kwargs):
    print(f"[LOG] {kwargs}")
```

### Metrics Hook

```python
def metrics_hook(**kwargs):
    input_len = len(str(kwargs.get('input', '')))
    output = kwargs.get('output', 'N/A')
    print(f"[METRICS] Input length: {input_len}, Output: {output}")
```

### Debug Hook

```python
def debug_hook(**kwargs):
    client = kwargs.get('client', 'unknown')
    input_data = kwargs.get('input')
    print(f"[DEBUG] Client '{client}' processed: {input_data}")
```

## Integration with Braintrust-Style Evaluations

This hooks system is designed to work seamlessly with Braintrust-style evaluation patterns:

```python
# Your existing task function pattern
async def task(query, hooks):
    # Process the query
    result = await process_query(query)
    
    # Call hooks.meta() as before
    hooks.meta(input=query, output=result)
    
    return result

# Now you can also use the client pattern
client = EvaluationClient()
client.hooks.add_hook(global_logging_hook)

call_hooks = Hooks()
call_hooks.add_hook(specific_debug_hook)

result = await client.create(query, hooks=call_hooks)
```

## Advanced Usage

### Custom Client Implementation

```python
class MyEvaluationClient(Client):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    async def _create_impl(self, query, hooks=None, **kwargs):
        # Your custom logic here
        result = await self.process_query(query)
        
        # Call hooks with custom metadata
        if hooks:
            hooks.meta(
                client=self.name,
                input=query,
                output=result,
                timestamp=time.time(),
                **kwargs
            )
        
        return result
    
    async def create(self, query, hooks=None, **kwargs):
        combined_hooks = self.hooks.combine_with(hooks)
        return await self._create_impl(query, hooks=combined_hooks, **kwargs)
```

### Hook Composition

```python
# Create reusable hook combinations
def create_standard_hooks():
    hooks = Hooks()
    hooks.add_hook(logging_hook)
    hooks.add_hook(metrics_hook)
    return hooks

# Use in different contexts
call_hooks = create_standard_hooks()
call_hooks.add_hook(custom_debug_hook)

result = client.create("query", hooks=call_hooks)
```

## Benefits

1. **Simple**: Easy to understand and implement
2. **Flexible**: Support both global and per-call hooks
3. **Composable**: Hooks can be combined and reused
4. **Compatible**: Works with existing Braintrust patterns
5. **Extensible**: Easy to add new hook types and metadata

## Example Output

When running with multiple hooks:

```
[LOG] {'client': 'my-client', 'input': 'test query', 'output': 'processed: test query'}
[METRICS] Input length: 10, Output: processed: test query
[DEBUG] Client 'my-client' processed input: test query
Result: processed: test query
```

This shows all three hooks (logging, metrics, debug) being executed in order with the same metadata.