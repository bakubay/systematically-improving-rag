"""
Enhanced client wrapper with additive hooks support

This module provides enhanced client wrappers that integrate with the existing
instructor-based architecture while adding comprehensive hooks support.
"""

import instructor
from typing import Any, Dict, List, Optional, Union, Callable
import asyncio
from pathlib import Path

from .hooks import (
    BaseHook, 
    HookManager, 
    HookableClient, 
    HookContext, 
    HookPhase,
    create_standard_hooks,
    create_monitoring_hooks,
    create_reliability_hooks
)


class InstructorHookableClient(HookableClient):
    """
    Enhanced instructor client with additive hooks support
    
    This client wraps an instructor client and provides hooks functionality
    while maintaining compatibility with existing instructor-based code.
    """
    
    def __init__(self, 
                 provider: str = "openai/gpt-4.1-nano",
                 async_client: bool = True,
                 client_hooks: Optional[List[BaseHook]] = None,
                 **instructor_kwargs):
        """
        Initialize the hookable instructor client
        
        Args:
            provider: Instructor provider string (e.g., "openai/gpt-4.1-nano")
            async_client: Whether to use async client
            client_hooks: List of client-level hooks to apply to all calls
            **instructor_kwargs: Additional arguments passed to instructor.from_provider
        """
        # Create the base instructor client
        base_client = instructor.from_provider(
            provider, 
            async_client=async_client, 
            **instructor_kwargs
        )
        
        # Initialize the hookable wrapper
        super().__init__(base_client, client_hooks)
        
        self.provider = provider
        self.async_client = async_client
    
    async def chat_completions_create_with_hooks(self,
                                               call_hooks: Optional[List[BaseHook]] = None,
                                               **kwargs) -> Any:
        """
        Create chat completion with hooks support
        
        Args:
            call_hooks: Optional call-specific hooks
            **kwargs: Arguments passed to chat.completions.create
            
        Returns:
            Response from the chat completion call
        """
        return await self.call_with_hooks(
            "chat.completions.create",
            call_hooks=call_hooks,
            **kwargs
        )
    
    # Convenience method that maintains backward compatibility
    async def create_completion(self, 
                              messages: List[Dict[str, Any]],
                              response_model: Optional[Any] = None,
                              call_hooks: Optional[List[BaseHook]] = None,
                              **kwargs) -> Any:
        """
        Create a completion with optional response model and hooks
        
        Args:
            messages: List of message dictionaries
            response_model: Optional Pydantic model for structured responses
            call_hooks: Optional call-specific hooks
            **kwargs: Additional arguments for the completion
            
        Returns:
            Completion response
        """
        completion_kwargs = {
            "messages": messages,
            **kwargs
        }
        
        if response_model:
            completion_kwargs["response_model"] = response_model
        
        return await self.chat_completions_create_with_hooks(
            call_hooks=call_hooks,
            **completion_kwargs
        )


class ClientFactory:
    """
    Factory for creating pre-configured hookable clients
    
    This factory provides convenient methods for creating clients with
    common hook configurations.
    """
    
    @staticmethod
    def create_basic_client(provider: str = "openai/gpt-4.1-nano",
                          async_client: bool = True) -> InstructorHookableClient:
        """Create a basic client without hooks"""
        return InstructorHookableClient(
            provider=provider,
            async_client=async_client
        )
    
    @staticmethod
    def create_monitored_client(provider: str = "openai/gpt-4.1-nano",
                              async_client: bool = True,
                              log_level: str = "INFO") -> InstructorHookableClient:
        """Create a client with monitoring hooks"""
        hooks = create_monitoring_hooks()
        return InstructorHookableClient(
            provider=provider,
            async_client=async_client,
            client_hooks=hooks
        )
    
    @staticmethod
    def create_reliable_client(provider: str = "openai/gpt-4.1-nano",
                             async_client: bool = True,
                             calls_per_second: float = 5.0,
                             max_retries: int = 3) -> InstructorHookableClient:
        """Create a client with reliability hooks (rate limiting + retry)"""
        hooks = create_reliability_hooks(
            calls_per_second=calls_per_second,
            max_retries=max_retries
        )
        return InstructorHookableClient(
            provider=provider,
            async_client=async_client,
            client_hooks=hooks
        )
    
    @staticmethod
    def create_full_featured_client(provider: str = "openai/gpt-4.1-nano",
                                  async_client: bool = True,
                                  calls_per_second: float = 10.0,
                                  max_retries: int = 3,
                                  enable_logging: bool = True,
                                  enable_metrics: bool = True) -> InstructorHookableClient:
        """Create a client with all standard hooks enabled"""
        hooks = create_standard_hooks(
            enable_logging=enable_logging,
            enable_metrics=enable_metrics,
            enable_rate_limiting=True,
            calls_per_second=calls_per_second,
            enable_retry=True,
            max_retries=max_retries
        )
        return InstructorHookableClient(
            provider=provider,
            async_client=async_client,
            client_hooks=hooks
        )


# Utility functions for working with hooks in generation pipelines

def create_generation_hooks(experiment_id: Optional[str] = None,
                          enable_caching: bool = True) -> List[BaseHook]:
    """
    Create hooks specifically designed for generation pipelines
    
    Args:
        experiment_id: Optional experiment ID for tracking
        enable_caching: Whether to enable caching hooks
        
    Returns:
        List of hooks suitable for generation pipelines
    """
    from .hooks import LoggingHook, MetricsHook
    
    hooks = [
        LoggingHook(name="generation_logging"),
        MetricsHook(name="generation_metrics"),
    ]
    
    # Add experiment-specific hooks if experiment_id is provided
    if experiment_id:
        class ExperimentHook(BaseHook):
            def __init__(self, exp_id: str):
                super().__init__(f"experiment_{exp_id}")
                self.experiment_id = exp_id
            
            async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
                if context.phase == HookPhase.PRE_REQUEST:
                    context.metadata["experiment_id"] = self.experiment_id
                return None
        
        hooks.append(ExperimentHook(experiment_id))
    
    return hooks


def create_question_generation_hooks(version: str) -> List[BaseHook]:
    """Create hooks specific to question generation"""
    from .hooks import LoggingHook, MetricsHook
    
    class QuestionGenerationHook(BaseHook):
        def __init__(self, gen_version: str):
            super().__init__(f"question_gen_{gen_version}")
            self.version = gen_version
        
        async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
            if context.phase == HookPhase.PRE_REQUEST:
                context.metadata["generation_version"] = self.version
                context.metadata["generation_type"] = "questions"
            elif context.phase == HookPhase.POST_REQUEST:
                # Log successful generation
                if context.response_data and "result" in context.response_data:
                    result = context.response_data["result"]
                    if hasattr(result, 'queries'):
                        context.metadata["generated_count"] = len(result.queries)
            return None
    
    return [
        LoggingHook(name=f"question_gen_{version}_logging"),
        MetricsHook(name=f"question_gen_{version}_metrics"), 
        QuestionGenerationHook(version)
    ]


def create_summary_generation_hooks(version: str) -> List[BaseHook]:
    """Create hooks specific to summary generation"""
    from .hooks import LoggingHook, MetricsHook
    
    class SummaryGenerationHook(BaseHook):
        def __init__(self, gen_version: str):
            super().__init__(f"summary_gen_{gen_version}")
            self.version = gen_version
        
        async def execute(self, context: HookContext) -> Optional[Dict[str, Any]]:
            if context.phase == HookPhase.PRE_REQUEST:
                context.metadata["generation_version"] = self.version
                context.metadata["generation_type"] = "summaries"
            elif context.phase == HookPhase.POST_REQUEST:
                # Log successful generation
                if context.response_data and "result" in context.response_data:
                    result = context.response_data["result"]
                    if hasattr(result, 'summary'):
                        context.metadata["summary_length"] = len(result.summary)
            return None
    
    return [
        LoggingHook(name=f"summary_gen_{version}_logging"),
        MetricsHook(name=f"summary_gen_{version}_metrics"),
        SummaryGenerationHook(version)
    ]


# Migration utilities for existing code

def wrap_existing_client(existing_client: Any, 
                        client_hooks: Optional[List[BaseHook]] = None) -> HookableClient:
    """
    Wrap an existing client with hooks functionality
    
    Args:
        existing_client: The existing client to wrap
        client_hooks: Optional client-level hooks
        
    Returns:
        HookableClient wrapper around the existing client
    """
    return HookableClient(existing_client, client_hooks)


def migrate_instructor_client(provider: str = "openai/gpt-4.1-nano",
                            async_client: bool = True,
                            client_hooks: Optional[List[BaseHook]] = None) -> InstructorHookableClient:
    """
    Create a drop-in replacement for instructor.from_provider with hooks support
    
    Args:
        provider: Instructor provider string
        async_client: Whether to use async client
        client_hooks: Optional client-level hooks
        
    Returns:
        InstructorHookableClient that can replace existing instructor clients
    """
    return InstructorHookableClient(
        provider=provider,
        async_client=async_client,
        client_hooks=client_hooks
    )