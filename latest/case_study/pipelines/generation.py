"""
Generation pipelines for questions and summaries with additive hooks support
"""

import asyncio
import json
import instructor
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, TaskID

from core.db import (
    get_conversations_by_hashes,
    save_questions_to_sqlite,
    save_summaries_to_sqlite,
    filter_unprocessed_hashes,
)
from core.synthetic_queries import (
    synthetic_question_generation_v1,
    synthetic_question_generation_v2,
    synthetic_question_generation_v3,
    synthetic_question_generation_v5,
)
from core.summarization import (
    conversation_summary_v1,
    conversation_summary_v2,
    conversation_summary_v3,
    conversation_summary_v4,
    conversation_summary_v5,
)
from core.cache import setup_cache, GenericCache
from core.client import (
    InstructorHookableClient,
    ClientFactory,
    create_question_generation_hooks,
    create_summary_generation_hooks,
)
from core.hooks import BaseHook

console = Console()

fn_query = {
    "v1": synthetic_question_generation_v1,
    "v2": synthetic_question_generation_v2,
    "v3": synthetic_question_generation_v3,
    "v5": synthetic_question_generation_v5,
}

fn_summary = {
    "v1": conversation_summary_v1,
    "v2": conversation_summary_v2,
    "v3": conversation_summary_v3,
    "v4": conversation_summary_v4,
    "v5": conversation_summary_v5,
}


async def call_generation_function_with_hooks(
    generation_fn,
    client,
    messages,
    call_hooks: Optional[List[BaseHook]] = None
):
    """
    Helper function to call generation functions with optional call-level hooks
    
    Args:
        generation_fn: The generation function to call
        client: The client (hookable or regular)
        messages: Messages to pass to the function
        call_hooks: Optional call-level hooks
        
    Returns:
        Result from the generation function
    """
    # If client is hookable and we have call hooks, we could enhance this further
    # For now, just call the function directly since the client already has hooks
    return await generation_fn(client, messages)


def parse_conversation_messages(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse conversation data into messages format expected by generation functions"""
    try:
        # Try to parse as JSON if it's a string
        if isinstance(conv.get("conversation_full"), str):
            conversation_data = json.loads(conv["conversation_full"])
        else:
            conversation_data = conv.get("conversation_full", conv.get("text", ""))

        # If it's already a list of messages, return it
        if isinstance(conversation_data, list):
            return conversation_data

        # Otherwise, create a simple message structure
        return [{"role": "user", "content": str(conversation_data)}]
    except (json.JSONDecodeError, TypeError):
        # Fallback to simple text format
        text = conv.get("text", conv.get("conversation_full", ""))
        return [{"role": "user", "content": str(text)}]


async def generate_questions_pipeline(
    conversation_hashes: List[str],
    version: str,
    db_path: Path,
    experiment_id: Optional[str] = None,
    concurrency: int = 10,
    use_cache: bool = True,
    client_hooks: Optional[List[BaseHook]] = None,
    call_hooks: Optional[List[BaseHook]] = None,
) -> Dict[str, int]:
    """
    Generate questions for conversations using specified techniques with additive hooks support

    Args:
        conversation_hashes: List of conversation hashes to process
        version: Question generation version
        db_path: Path to SQLite database
        experiment_id: Optional experiment ID for tracking
        concurrency: Max concurrent API requests
        use_cache: Whether to use caching
        client_hooks: Optional client-level hooks applied to all calls
        call_hooks: Optional call-level hooks applied to individual calls

    Returns:
        Dict with generated counts per technique
    """
    console.print(
        f"[bold green]Generating questions with version: {version}[/bold green]"
    )

    # Filter out already processed conversations
    unprocessed_hashes = filter_unprocessed_hashes(
        conversation_hashes, version, db_path, experiment_id
    )

    if len(unprocessed_hashes) < len(conversation_hashes):
        console.print(
            f"[yellow]Skipping {len(conversation_hashes) - len(unprocessed_hashes)} already processed conversations[/yellow]"
        )

    if not unprocessed_hashes:
        console.print("[green]All conversations already processed[/green]")
        return {version: 0}

    # Initialize hookable instructor client with additive hooks support
    # Combine client-level hooks with version-specific hooks
    combined_client_hooks = []
    if client_hooks:
        combined_client_hooks.extend(client_hooks)
    
    # Add version-specific hooks
    version_hooks = create_question_generation_hooks(version)
    combined_client_hooks.extend(version_hooks)
    
    client = InstructorHookableClient(
        provider="openai/gpt-4.1-nano",
        async_client=True,
        client_hooks=combined_client_hooks
    )

    # Set up cache
    cache_dir = db_path.parent / "cache" / "questions"
    cache = setup_cache(cache_dir) if use_cache else None

    # Load conversations
    conversations = get_conversations_by_hashes(unprocessed_hashes, db_path)
    console.print(f"Loaded {len(conversations)} conversations to process")

    results = {}

    questions = []
    semaphore = asyncio.Semaphore(concurrency)

    with Progress() as progress:
        task = progress.add_task(
            f"Generating {version} questions", total=len(conversations)
        )

        async def process_conversation(
            conv: Dict[str, Any],
        ) -> List[Dict[str, Any]] | None:
            async with semaphore:
                try:
                    conversation_hash = conv["conversation_hash"]

                    # Check cache first
                    if cache:
                        cache_key = GenericCache.make_conversation_key(
                            conversation_hash, version
                        )
                        cached_queries = cache.get(cache_key)
                        if cached_queries is not None:
                            # Convert cached queries to question data format
                            question_data_list = []
                            for query_idx, query in enumerate(cached_queries):
                                question_data = {
                                    "id": f"{conversation_hash}_{version}_{query_idx}",
                                    "conversation_hash": conversation_hash,
                                    "version": version,
                                    "question": query,
                                    "experiment_id": experiment_id,
                                }
                                question_data_list.append(question_data)
                            progress.advance(task)
                            return question_data_list

                    # Parse conversation messages
                    messages = parse_conversation_messages(conv)

                    # Generate questions using the version with additive hooks
                    # The client already has client-level hooks, and we can add call-level hooks
                    generated = await call_generation_function_with_hooks(
                        fn_query[version], client, messages, call_hooks
                    )

                    # Cache the result
                    if cache and generated.queries:
                        cache_key = GenericCache.make_conversation_key(
                            conversation_hash, version
                        )
                        cache.set(cache_key, generated.queries)

                    # Create question data for each generated query
                    question_data_list = []
                    for idx, query in enumerate(generated.queries):
                        question_data = {
                            "id": f"{conversation_hash}_{version}_{idx}",
                            "conversation_hash": conversation_hash,
                            "version": version,
                            "question": query,
                            "experiment_id": experiment_id,
                        }
                        question_data_list.append(question_data)

                    progress.advance(task)
                    return question_data_list
                except Exception as e:
                    console.print(
                        f"[red]Error processing {conv['conversation_hash']}: {e}[/red]"
                    )
                    progress.advance(task)
                    return None

        # Process all conversations concurrently
        tasks = [process_conversation(conv) for conv in conversations]
        question_lists = await asyncio.gather(*tasks)

        # Flatten the list of lists and filter out None results
        questions = []
        for q_list in question_lists:
            if q_list is not None:
                questions.extend(q_list)

        # Save to database
        saved_count = save_questions_to_sqlite(questions, db_path)
        results[version] = saved_count
        console.print(f"[green]Saved {saved_count} {version} questions[/green]")

    return results


async def generate_summaries_pipeline(
    conversation_hashes: List[str],
    version: str,
    db_path: Path,
    experiment_id: Optional[str] = None,
    concurrency: int = 10,
    use_cache: bool = True,
    show_progress: bool = True,
    client_hooks: Optional[List[BaseHook]] = None,
    call_hooks: Optional[List[BaseHook]] = None,
) -> Dict[str, int]:
    """
    Generate summaries for conversations using specified techniques
    """
    console.print(
        f"[bold green]Generating summaries with version: {version}[/bold green]"
    )

    # Filter out already processed conversations
    unprocessed_hashes = filter_unprocessed_hashes(
        conversation_hashes, version, db_path, experiment_id, is_summary=True
    )

    if len(unprocessed_hashes) < len(conversation_hashes):
        console.print(
            f"[yellow]Skipping {len(conversation_hashes) - len(unprocessed_hashes)} already processed conversations[/yellow]"
        )

    if not unprocessed_hashes:
        console.print("[green]All conversations already processed[/green]")
        return {version: 0}

    # Initialize hookable instructor client with additive hooks support
    # Combine client-level hooks with version-specific hooks
    combined_client_hooks = []
    if client_hooks:
        combined_client_hooks.extend(client_hooks)
    
    # Add version-specific hooks
    version_hooks = create_summary_generation_hooks(version)
    combined_client_hooks.extend(version_hooks)
    
    client = InstructorHookableClient(
        provider="openai/gpt-4.1-nano",
        async_client=True,
        client_hooks=combined_client_hooks
    )

    # Set up cache
    # cache_dir = db_path.parent / "cache" / "summaries"
    # cache = setup_cache(cache_dir) if use_cache else None  # Unused for now

    # Load conversations
    conversations = get_conversations_by_hashes(unprocessed_hashes, db_path)
    console.print(f"Loaded {len(conversations)} conversations to process")

    results = {}

    summaries = []
    semaphore = asyncio.Semaphore(concurrency)

    async def process_conversation(
        conv: Dict[str, Any],
        progress_task: Optional[TaskID] = None,
        progress_obj: Optional[Progress] = None,
    ) -> Dict[str, Any] | None:
        async with semaphore:
            try:
                # Parse conversation messages
                messages = parse_conversation_messages(conv)

                # Generate summary using the version with additive hooks
                generated = await call_generation_function_with_hooks(
                    fn_summary[version], client, messages, call_hooks
                )

                # Add metadata for saving
                summary_data = {
                    "id": f"{conv['conversation_hash']}_{version}",  # Keep unique ID for database
                    "conversation_hash": conv["conversation_hash"],
                    "technique": version,
                    "summary": generated.summary,
                    "experiment_id": experiment_id,
                }

                if progress_obj and progress_task is not None:
                    progress_obj.advance(progress_task)
                return summary_data
            except Exception as e:
                console.print(
                    f"[red]Error processing {conv['conversation_hash']}: {e}[/red]"
                )
                if progress_obj and progress_task is not None:
                    progress_obj.advance(progress_task)
                return None

    if show_progress:
        with Progress() as progress:
            task = progress.add_task(
                f"Generating {version} summaries", total=len(conversations)
            )
            # Process all conversations concurrently
            tasks = [
                process_conversation(conv, task, progress) for conv in conversations
            ]
            summaries = await asyncio.gather(*tasks)
    else:
        # Process without progress bar
        tasks = [process_conversation(conv) for conv in conversations]
        summaries = await asyncio.gather(*tasks)

    # Filter out None results
    summaries = [s for s in summaries if s is not None]

    # Save to database
    saved_count = save_summaries_to_sqlite(summaries, db_path)
    results[version] = saved_count
    console.print(f"[green]Saved {saved_count} {version} summaries[/green]")

    return results
