"""
Additive hooks system for API calls

This module provides a flexible hooks system that allows combining client-level
and call-level hooks. Hooks are executed additively, meaning both client and
call hooks can be applied to the same API call.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class HookPhase(Enum):
    """Phases when hooks can be executed"""
    PRE_REQUEST = "pre_request"
    POST_REQUEST = "post_request" 
    ON_ERROR = "on_error"
    ON_RETRY = "on_retry"


@dataclass
class HookContext:
    """Context object passed to hooks containing request/response data"""
    phase: HookPhase
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata with default values"""
        if "timestamp" not in self.metadata:
            import time
            self.metadata["timestamp"] = time.time()


class BaseHook(ABC):
    """Base class for all hooks"""
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority  # Higher priority hooks run first
    
    @abstractmethod
    async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
        """
        Execute the hook with the given context
        
        Args:
            context: HookContext containing request/response data
            
        Returns:
            Optional modifications to be applied to the context
        """
        pass
    
    def should_execute(self, context: HookContext) -> bool:
        """
        Determine if this hook should execute for the given context
        
        Args:
            context: HookContext to evaluate
            
        Returns:
            True if hook should execute, False otherwise
        """
        return True
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', priority={self.priority})"


class LoggingHook(BaseHook):
    """Hook that logs API requests and responses"""
    
    def __init__(self, name: str = "logging", priority: int = 100, log_level: str = "INFO"):
        super().__init__(name, priority)
        self.log_level = getattr(logging, log_level.upper())
    
    async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
        if context.phase == HookPhase.PRE_REQUEST:
            logger.log(self.log_level, f"API Request: {context.request_data.get('method', 'UNKNOWN')}")
        elif context.phase == HookPhase.POST_REQUEST:
            logger.log(self.log_level, f"API Response: Success")
        elif context.phase == HookPhase.ON_ERROR:
            logger.log(self.log_level, f"API Error: {context.error}")
        return None


class MetricsHook(BaseHook):
    """Hook that collects metrics on API calls"""
    
    def __init__(self, name: str = "metrics", priority: int = 90):
        super().__init__(name, priority)
        self.call_count = 0
        self.error_count = 0
        self.total_time = 0.0
    
    async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
        if context.phase == HookPhase.PRE_REQUEST:
            self.call_count += 1
            context.metadata["start_time"] = context.metadata["timestamp"]
        elif context.phase == HookPhase.POST_REQUEST:
            if "start_time" in context.metadata:
                duration = context.metadata["timestamp"] - context.metadata["start_time"]
                self.total_time += duration
        elif context.phase == HookPhase.ON_ERROR:
            self.error_count += 1
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        avg_time = self.total_time / max(1, self.call_count - self.error_count)
        return {
            "total_calls": self.call_count,
            "error_count": self.error_count,
            "success_count": self.call_count - self.error_count,
            "average_response_time": avg_time,
            "total_time": self.total_time
        }


class RateLimitHook(BaseHook):
    """Hook that implements rate limiting"""
    
    def __init__(self, name: str = "rate_limit", priority: int = 200, 
                 calls_per_second: float = 10.0):
        super().__init__(name, priority)
        self.calls_per_second = calls_per_second
        self.last_call_time = 0.0
        self.min_interval = 1.0 / calls_per_second
    
    async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
        if context.phase == HookPhase.PRE_REQUEST:
            import time
            current_time = time.time()
            time_since_last = current_time - self.last_call_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_call_time = time.time()
        return None


class RetryHook(BaseHook):
    """Hook that implements retry logic"""
    
    def __init__(self, name: str = "retry", priority: int = 50,
                 max_retries: int = 3, backoff_factor: float = 1.5):
        super().__init__(name, priority)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
        if context.phase == HookPhase.ON_ERROR:
            retry_count = context.metadata.get("retry_count", 0)
            if retry_count < self.max_retries:
                delay = (self.backoff_factor ** retry_count)
                await asyncio.sleep(delay)
                context.metadata["retry_count"] = retry_count + 1
                return {"should_retry": True}
        return None


class HookManager:
    """
    Manages and executes hooks in an additive fashion
    
    This allows combining client-level hooks with call-level hooks,
    executing them in priority order.
    """
    
    def __init__(self, client_hooks: Optional[List[BaseHook]] = None):
        self.client_hooks = client_hooks or []
        self._sort_hooks()
    
    def _sort_hooks(self):
        """Sort hooks by priority (higher priority first)"""
        self.client_hooks.sort(key=lambda h: h.priority, reverse=True)
    
    def add_client_hook(self, hook: BaseHook):
        """Add a client-level hook"""
        self.client_hooks.append(hook)
        self._sort_hooks()
    
    def remove_client_hook(self, hook_name: str):
        """Remove a client-level hook by name"""
        self.client_hooks = [h for h in self.client_hooks if h.name != hook_name]
    
    def get_combined_hooks(self, call_hooks: Optional[List[BaseHook]] = None) -> List[BaseHook]:
        """
        Combine client-level and call-level hooks
        
        Args:
            call_hooks: Optional list of call-specific hooks
            
        Returns:
            Combined list of hooks sorted by priority
        """
        all_hooks = self.client_hooks.copy()
        if call_hooks:
            all_hooks.extend(call_hooks)
        
        # Sort by priority (higher first)
        all_hooks.sort(key=lambda h: h.priority, reverse=True)
        return all_hooks
    
    async def execute_hooks(self, 
                          phase: HookPhase,
                          context: HookContext,
                          call_hooks: Optional[List[BaseHook]] = None) -> HookContext:
        """
        Execute hooks for a specific phase
        
        Args:
            phase: The hook phase to execute
            context: Context object with request/response data
            call_hooks: Optional call-specific hooks
            
        Returns:
            Updated context after all hooks have executed
        """
        context.phase = phase
        combined_hooks = self.get_combined_hooks(call_hooks)
        
        for hook in combined_hooks:
            if hook.should_execute(context):
                try:
                    result = await hook.execute(context)
                    if result:
                        # Apply any modifications returned by the hook
                        if "request_data" in result:
                            context.request_data.update(result["request_data"])
                        if "response_data" in result:
                            if context.response_data:
                                context.response_data.update(result["response_data"])
                            else:
                                context.response_data = result["response_data"]
                        if "metadata" in result:
                            context.metadata.update(result["metadata"])
                        
                        # Handle special control flow
                        if result.get("should_retry"):
                            context.metadata["should_retry"] = True
                        if result.get("skip_remaining_hooks"):
                            break
                            
                except Exception as e:
                    logger.error(f"Error executing hook {hook.name}: {e}")
                    # Continue with other hooks even if one fails
        
        return context


class HookableClient:
    """
    A wrapper that adds hooks support to any client
    
    This class wraps an existing client and adds additive hooks functionality
    """
    
    def __init__(self, client: Any, hooks: Optional[List[BaseHook]] = None):
        self.client = client
        self.hook_manager = HookManager(hooks)
    
    def add_hook(self, hook: BaseHook):
        """Add a client-level hook"""
        self.hook_manager.add_client_hook(hook)
    
    def remove_hook(self, hook_name: str):
        """Remove a client-level hook"""
        self.hook_manager.remove_client_hook(hook_name)
    
    async def call_with_hooks(self, 
                            method: str,
                            *args,
                            call_hooks: Optional[List[BaseHook]] = None,
                            **kwargs) -> Any:
        """
        Call a client method with hooks support
        
        Args:
            method: Method name to call on the wrapped client
            *args: Positional arguments for the method
            call_hooks: Optional call-specific hooks
            **kwargs: Keyword arguments for the method
            
        Returns:
            Result from the method call
        """
        # Create initial context
        context = HookContext(
            phase=HookPhase.PRE_REQUEST,
            request_data={
                "method": method,
                "args": args,
                "kwargs": kwargs
            }
        )
        
        # Execute pre-request hooks
        context = await self.hook_manager.execute_hooks(
            HookPhase.PRE_REQUEST, context, call_hooks
        )
        
        max_retries = 1  # Default no retries
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Get the method from the wrapped client
                client_method = getattr(self.client, method)
                
                # Call the method (handle both sync and async)
                if asyncio.iscoroutinefunction(client_method):
                    result = await client_method(*args, **kwargs)
                else:
                    result = client_method(*args, **kwargs)
                
                # Update context with response
                context.response_data = {"result": result}
                
                # Execute post-request hooks
                context = await self.hook_manager.execute_hooks(
                    HookPhase.POST_REQUEST, context, call_hooks
                )
                
                return result
                
            except Exception as e:
                context.error = e
                
                # Execute error hooks
                context = await self.hook_manager.execute_hooks(
                    HookPhase.ON_ERROR, context, call_hooks
                )
                
                # Check if we should retry
                if context.metadata.get("should_retry") and retry_count < max_retries - 1:
                    retry_count += 1
                    
                    # Execute retry hooks
                    context = await self.hook_manager.execute_hooks(
                        HookPhase.ON_RETRY, context, call_hooks
                    )
                    
                    # Update max_retries if hooks modified it
                    if "max_retries" in context.metadata:
                        max_retries = context.metadata["max_retries"]
                    
                    continue
                else:
                    # Re-raise the exception if no retry or max retries exceeded
                    raise e
        
        # This should not be reached, but just in case
        raise RuntimeError("Unexpected end of retry loop")
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped client"""
        return getattr(self.client, name)


# Convenience functions for creating common hook combinations

def create_standard_hooks(
    enable_logging: bool = True,
    enable_metrics: bool = True, 
    enable_rate_limiting: bool = False,
    calls_per_second: float = 10.0,
    enable_retry: bool = False,
    max_retries: int = 3
) -> List[BaseHook]:
    """Create a standard set of hooks with common functionality"""
    hooks = []
    
    if enable_logging:
        hooks.append(LoggingHook())
    
    if enable_metrics:
        hooks.append(MetricsHook())
    
    if enable_rate_limiting:
        hooks.append(RateLimitHook(calls_per_second=calls_per_second))
    
    if enable_retry:
        hooks.append(RetryHook(max_retries=max_retries))
    
    return hooks


def create_monitoring_hooks() -> List[BaseHook]:
    """Create hooks focused on monitoring and observability"""
    return [
        LoggingHook(log_level="DEBUG"),
        MetricsHook(),
    ]


def create_reliability_hooks(
    calls_per_second: float = 5.0,
    max_retries: int = 3
) -> List[BaseHook]:
    """Create hooks focused on reliability and fault tolerance"""
    return [
        RateLimitHook(calls_per_second=calls_per_second),
        RetryHook(max_retries=max_retries),
    ]