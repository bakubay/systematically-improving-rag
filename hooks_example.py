"""
Example showing how to use the hooks system with a Braintrust-style evaluation pattern.
This demonstrates the requested functionality where hooks can be defined at the client level
and also passed to individual create calls.
"""

from hooks import Hooks, Client
import asyncio


class EvaluationClient(Client):
    """
    Example client that demonstrates hooks integration with evaluation tasks.
    This simulates how you might use hooks with Braintrust-style evaluations.
    """
    
    def __init__(self, name="evaluation-client"):
        super().__init__()
        self.name = name
    
    async def _create_impl(self, query, hooks=None, **kwargs):
        """
        Simulate an evaluation task that processes a query and calls hooks.
        This is similar to the task functions used with Braintrust EvalAsync.
        """
        # Simulate some processing
        result = f"processed: {query}"
        
        # Call hooks with metadata (similar to Braintrust hooks.meta())
        if hooks:
            hooks.meta(
                client=self.name,
                input=query,
                output=result,
                **kwargs
            )
        
        return result
    
    async def create(self, query, hooks=None, **kwargs):
        """Async version of create method."""
        # Combine client-level hooks with call-level hooks
        combined_hooks = self.hooks.combine_with(hooks)
        
        # Call the implementation with combined hooks
        return await self._create_impl(query, hooks=combined_hooks, **kwargs)


# Example hook functions
def logging_hook(**kwargs):
    """Hook that logs all metadata."""
    print(f"[LOG] {kwargs}")

def metrics_hook(**kwargs):
    """Hook that tracks metrics."""
    print(f"[METRICS] Input length: {len(str(kwargs.get('input', '')))}, "
          f"Output: {kwargs.get('output', 'N/A')}")

def debug_hook(**kwargs):
    """Hook that prints debug information."""
    client = kwargs.get('client', 'unknown')
    print(f"[DEBUG] Client '{client}' processed input: {kwargs.get('input')}")


async def main():
    """Demonstrate the hooks system with various scenarios."""
    
    print("Hooks System Example")
    print("=" * 60)
    
    # Scenario 1: Client with global hooks
    print("\n1. Client with global hooks only:")
    print("-" * 40)
    
    client = EvaluationClient("global-hooks-client")
    client.hooks.add_hook(logging_hook)
    client.hooks.add_hook(metrics_hook)
    
    result = await client.create("What is the weather today?")
    print(f"Result: {result}")
    
    # Scenario 2: Client with both global and call-specific hooks
    print("\n2. Client with both global and call-specific hooks:")
    print("-" * 40)
    
    call_hooks = Hooks()
    call_hooks.add_hook(debug_hook)
    
    result = await client.create(
        "How do I reset my password?", 
        hooks=call_hooks,
        metadata="password-reset-query"
    )
    print(f"Result: {result}")
    
    # Scenario 3: Multiple calls with different call-specific hooks
    print("\n3. Multiple calls with different call-specific hooks:")
    print("-" * 40)
    
    # First call with debug hook
    call_hooks1 = Hooks()
    call_hooks1.add_hook(debug_hook)
    
    result1 = await client.create("Book a flight to Paris", hooks=call_hooks1)
    print(f"Result 1: {result1}")
    
    # Second call with no additional hooks (only global ones)
    result2 = await client.create("Cancel my subscription")
    print(f"Result 2: {result2}")
    
    # Scenario 4: New client with only call-specific hooks
    print("\n4. New client with only call-specific hooks:")
    print("-" * 40)
    
    client2 = EvaluationClient("call-hooks-only-client")
    
    call_hooks2 = Hooks()
    call_hooks2.add_hook(logging_hook)
    call_hooks2.add_hook(debug_hook)
    
    result = await client2.create("What are my account settings?", hooks=call_hooks2)
    print(f"Result: {result}")
    
    # Scenario 5: Simulating the pattern from the user's request
    print("\n5. User's requested pattern:")
    print("-" * 40)
    
    # This demonstrates the exact pattern the user wanted:
    # call_hooks = Hooks()
    # call_hooks...
    # client.create(..., hooks=call_hooks)
    
    client3 = EvaluationClient("user-pattern-client")
    client3.hooks.add_hook(logging_hook)  # Global hook
    
    call_hooks = Hooks()
    call_hooks.add_hook(metrics_hook)
    call_hooks.add_hook(debug_hook)
    
    result = await client3.create(
        "Process this important query",
        hooks=call_hooks
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())