"""
Simple hooks implementation that allows defining hooks at the client level
and passing additional hooks on individual create calls.
"""

class Hooks:
    """A simple hooks container that can store and execute hook functions."""
    
    def __init__(self):
        self._hooks = []
    
    def add_hook(self, hook_func):
        """Add a hook function to be executed."""
        self._hooks.append(hook_func)
    
    def meta(self, **kwargs):
        """Execute all hooks with the provided metadata."""
        for hook in self._hooks:
            # Call each hook function with the metadata
            hook(**kwargs)
    
    def combine_with(self, other_hooks):
        """Combine this hooks object with another hooks object."""
        combined = Hooks()
        combined._hooks = self._hooks.copy()
        if other_hooks:
            combined._hooks.extend(other_hooks._hooks)
        return combined


class Client:
    """A client that supports hooks at the client level and per-create call."""
    
    def __init__(self):
        self.hooks = Hooks()
    
    def create(self, *args, hooks=None, **kwargs):
        """
        Create method that combines client-level hooks with call-level hooks.
        
        Args:
            *args: Arguments to pass to the actual create implementation
            hooks: Optional Hooks object for this specific create call
            **kwargs: Keyword arguments to pass to the actual create implementation
            
        Returns:
            The result of the create operation
        """
        # Combine client-level hooks with call-level hooks
        combined_hooks = self.hooks.combine_with(hooks)
        
        # Replace the hooks parameter with the combined hooks
        kwargs['hooks'] = combined_hooks
        
        # Call the actual create implementation
        return self._create_impl(*args, **kwargs)
    
    def _create_impl(self, *args, **kwargs):
        """
        Override this method in subclasses to implement the actual create logic.
        The hooks will be available in kwargs['hooks'].
        """
        raise NotImplementedError("Subclasses must implement _create_impl")


# Example usage and testing
if __name__ == "__main__":
    # Example implementation
    class ExampleClient(Client):
        def _create_impl(self, data, hooks=None, **kwargs):
            print(f"Creating with data: {data}")
            
            # Execute hooks with some metadata
            if hooks:
                hooks.meta(
                    input=data,
                    output="created_successfully",
                    operation="create"
                )
            
            return "created_successfully"
    
    # Define some hook functions
    def client_hook(**kwargs):
        print(f"Client hook called with: {kwargs}")
    
    def call_hook(**kwargs):
        print(f"Call hook called with: {kwargs}")
    
    # Test the implementation
    print("Testing hooks implementation:")
    print("=" * 50)
    
    # Create client and add a client-level hook
    client = ExampleClient()
    client.hooks.add_hook(client_hook)
    
    # Create a hooks object for a single call
    call_hooks = Hooks()
    call_hooks.add_hook(call_hook)
    
    # Test 1: Create with both client and call hooks
    print("\nTest 1: Both client and call hooks")
    result = client.create({"key": "value"}, hooks=call_hooks)
    print(f"Result: {result}")
    
    # Test 2: Create with only client hooks
    print("\nTest 2: Only client hooks")
    result = client.create({"key": "value2"})
    print(f"Result: {result}")
    
    # Test 3: Create with only call hooks (no client hooks)
    print("\nTest 3: Only call hooks (clean client)")
    client2 = ExampleClient()
    result = client2.create({"key": "value3"}, hooks=call_hooks)
    print(f"Result: {result}")