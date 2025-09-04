"""
Integration example showing how the hooks system works alongside existing Braintrust patterns.
This demonstrates both the traditional task function approach and the new client approach.
"""

from hooks import Hooks, Client
import asyncio


# Traditional Braintrust-style task function
async def traditional_task(query, hooks):
    """
    Traditional task function that receives hooks from Braintrust EvalAsync.
    This is how hooks currently work in the existing codebase.
    """
    # Simulate processing
    result = f"traditional_result: {query}"
    
    # Call hooks.meta() as done in existing code
    hooks.meta(
        input=query,
        output=result,
        approach="traditional"
    )
    
    return result


# New client-based approach
class EvalClient(Client):
    """Client that can be used alongside traditional Braintrust evaluation."""
    
    def __init__(self, name="eval-client"):
        super().__init__()
        self.name = name
    
    async def _create_impl(self, query, hooks=None, **kwargs):
        """Implementation that works with the hooks system."""
        # Simulate processing (same as traditional task)
        result = f"client_result: {query}"
        
        # Call hooks with metadata
        if hooks:
            hooks.meta(
                input=query,
                output=result,
                client=self.name,
                approach="client-based",
                **kwargs
            )
        
        return result
    
    async def create(self, query, hooks=None, **kwargs):
        """Async create method that combines hooks."""
        combined_hooks = self.hooks.combine_with(hooks)
        return await self._create_impl(query, hooks=combined_hooks, **kwargs)


# Hook functions for demonstration
def braintrust_style_hook(**kwargs):
    """Hook that mimics Braintrust-style logging."""
    approach = kwargs.get('approach', 'unknown')
    input_data = kwargs.get('input')
    output_data = kwargs.get('output')
    print(f"[BRAINTRUST-STYLE] {approach}: {input_data} -> {output_data}")


def custom_metrics_hook(**kwargs):
    """Custom metrics hook."""
    client = kwargs.get('client', 'N/A')
    input_len = len(str(kwargs.get('input', '')))
    print(f"[METRICS] Client: {client}, Input length: {input_len}")


async def demonstrate_integration():
    """
    Demonstrate how the new hooks system integrates with existing patterns.
    """
    
    print("Integration Example: Traditional vs Client-based Hooks")
    print("=" * 60)
    
    # Sample query
    query = "What is the weather in New York?"
    
    # 1. Traditional Braintrust-style approach
    print("\n1. Traditional Braintrust-style task function:")
    print("-" * 50)
    
    # Create hooks object (simulating what Braintrust would pass)
    traditional_hooks = Hooks()
    traditional_hooks.add_hook(braintrust_style_hook)
    
    result1 = await traditional_task(query, traditional_hooks)
    print(f"Result: {result1}")
    
    # 2. New client-based approach with global hooks
    print("\n2. New client-based approach with global hooks:")
    print("-" * 50)
    
    client = EvalClient("weather-client")
    client.hooks.add_hook(braintrust_style_hook)
    client.hooks.add_hook(custom_metrics_hook)
    
    result2 = await client.create(query)
    print(f"Result: {result2}")
    
    # 3. Client-based with both global and call-specific hooks
    print("\n3. Client with both global and call-specific hooks:")
    print("-" * 50)
    
    # Create call-specific hooks
    call_hooks = Hooks()
    call_hooks.add_hook(lambda **kwargs: print(f"[CALL-SPECIFIC] Processing: {kwargs.get('input')}"))
    
    result3 = await client.create(query, hooks=call_hooks)
    print(f"Result: {result3}")
    
    # 4. Demonstrate the exact pattern requested by the user
    print("\n4. User's requested pattern:")
    print("-" * 50)
    
    # This is exactly what the user wanted:
    call_hooks = Hooks()
    call_hooks.add_hook(lambda **kwargs: print(f"[USER-PATTERN] {kwargs}"))
    
    result4 = await client.create(query, hooks=call_hooks)
    print(f"Result: {result4}")
    
    # 5. Show how both approaches can coexist
    print("\n5. Both approaches working together:")
    print("-" * 50)
    
    # Traditional approach
    print("Traditional:")
    traditional_result = await traditional_task("traditional query", traditional_hooks)
    
    # Client approach
    print("Client-based:")
    client_result = await client.create("client query")
    
    print(f"Both completed successfully!")


# Simulate how this might work with Braintrust EvalAsync
async def simulate_braintrust_integration():
    """
    Simulate how the client approach could work alongside EvalAsync.
    """
    
    print("\n" + "=" * 60)
    print("Simulated Braintrust Integration")
    print("=" * 60)
    
    # Create a client with global hooks
    client = EvalClient("braintrust-integration-client")
    client.hooks.add_hook(braintrust_style_hook)
    
    # Simulate multiple evaluation queries
    queries = [
        "Book a flight to Paris",
        "Cancel my subscription", 
        "What are my account settings?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nEvaluation {i}:")
        print("-" * 30)
        
        # For some evaluations, add call-specific hooks
        if i == 2:  # Add special hook for second query
            call_hooks = Hooks()
            call_hooks.add_hook(lambda **kwargs: print(f"[SPECIAL] Query {i}: {kwargs.get('input')}"))
            result = await client.create(query, hooks=call_hooks)
        else:
            result = await client.create(query)
        
        print(f"Result {i}: {result}")


async def main():
    await demonstrate_integration()
    await simulate_braintrust_integration()


if __name__ == "__main__":
    asyncio.run(main())