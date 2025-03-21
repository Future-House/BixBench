import ast
import asyncio
import base64
import json
import random
import re
import time
from asyncio import Lock, Semaphore
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import litellm
import nbformat
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

from bixbench import prompts

# Enable debug logging for litellm to troubleshoot API issues
logging.basicConfig(level=logging.INFO)
litellm.set_verbose = True

# Set up direct tracing of litellm calls with proxy pattern
original_acompletion = litellm.acompletion

async def traced_acompletion(*args, **kwargs):
    """Wrapper function to trace litellm.acompletion calls"""
    trace_id = f"LITELLM-{int(time.time() * 1000) % 10000}"
    model = kwargs.get('model', 'unknown')
    print(f"[{trace_id}] ⭐ ENTERING litellm.acompletion for model: {model}")
    try:
        result = await original_acompletion(*args, **kwargs)
        print(f"[{trace_id}] ✅ EXITING litellm.acompletion for model: {model} (Success)")
        return result
    except Exception as e:
        print(f"[{trace_id}] ❌ EXITING litellm.acompletion for model: {model} (Error: {type(e).__name__}: {str(e)})")
        raise

# Replace litellm.acompletion with our traced version
litellm.acompletion = traced_acompletion

# Check if API keys are properly set
import os
print("Checking API environment variables:")
if "OPENAI_API_KEY" in os.environ:
    print("OPENAI_API_KEY: Found (hidden for security)")
else:
    print("OPENAI_API_KEY: NOT FOUND - OpenAI models may fail")
    
if "ANTHROPIC_API_KEY" in os.environ:
    print("ANTHROPIC_API_KEY: Found (hidden for security)")
else:
    print("ANTHROPIC_API_KEY: NOT FOUND - Claude models may fail")

# Print litellm configuration information
print(f"LiteLLM configuration info:")
print(f"  - Default timeout: {getattr(litellm, 'request_timeout', 'Not set')}")
print(f"  - Default max retries: {getattr(litellm, 'num_retries', 'Not set')}")
print(f"  - Verbose logging: {litellm.set_verbose}")

# Token rate limit tracking for different models
@dataclass
class ModelRateLimits:
    """Track and enforce token rate limits for a specific model"""
    # Rate limits (per minute)
    requests_per_min: int
    input_tokens_per_min: int
    output_tokens_per_min: int
    
    # Current usage tracking
    request_count: int = 0
    input_token_count: int = 0
    output_token_count: int = 0
    
    # Time tracking
    last_reset: float = field(default_factory=time.time)
    
    # Mutex lock for tracking updates
    lock: Lock = field(default_factory=Lock)
    
    async def reset_if_needed(self):
        """Reset counters if a minute has passed"""
        async with self.lock:
            current_time = time.time()
            if current_time - self.last_reset >= 60:
                self.request_count = 0
                self.input_token_count = 0
                self.output_token_count = 0
                self.last_reset = current_time
                
    async def add_usage(self, input_tokens: int, output_tokens: int):
        """Add usage and return time to wait if over rate limit"""
        async with self.lock:
            await self.reset_if_needed()
            
            # Ensure we're using reasonable values
            input_tokens = max(0, input_tokens)
            output_tokens = max(0, output_tokens)
            
            self.input_token_count += input_tokens
            self.output_token_count += output_tokens
            
            # Check if we're over any limits
            request_ratio = self.request_count / self.requests_per_min
            input_ratio = self.input_token_count / self.input_tokens_per_min
            output_ratio = self.output_token_count / self.output_tokens_per_min
            
            max_ratio = max(request_ratio, input_ratio, output_ratio)
            
            # If we're over a limit, calculate delay
            if max_ratio > 0.95:  # Allow a small buffer
                seconds_since_reset = time.time() - self.last_reset
                seconds_to_wait = max(0, 60 - seconds_since_reset)
                # Add jitter to avoid thundering herd
                jitter = random.random() * 0.1 * seconds_to_wait
                wait_time = max(0.5, seconds_to_wait + jitter)
                print(f"Rate limit reached, recommending wait of {wait_time:.2f}s")
                return wait_time
            
            return 0
    
    async def wait_if_needed(self, input_tokens_estimate: int = 500, output_tokens_estimate: int = 300):
        """Wait if we would exceed rate limits with the estimated token usage"""
        async with self.lock:
            await self.reset_if_needed()
            
            # Project usage with this request
            projected_requests = self.request_count + 1
            projected_input = self.input_token_count + input_tokens_estimate
            projected_output = self.output_token_count + output_tokens_estimate
            
            # Check if we'd exceed limits
            request_ratio = projected_requests / self.requests_per_min
            input_ratio = projected_input / self.input_tokens_per_min
            output_ratio = projected_output / self.output_tokens_per_min
            
            max_ratio = max(request_ratio, input_ratio, output_ratio)
            
            # If we would exceed a limit, wait until next reset
            if max_ratio > 0.95:  # Allow a small buffer
                seconds_since_reset = time.time() - self.last_reset
                seconds_to_wait = max(0, 60 - seconds_since_reset)
                
                # Only wait if we actually need to (positive wait time)
                if seconds_to_wait > 0:
                    # Add jitter to avoid thundering herd
                    jitter = random.random() * 0.1 * seconds_to_wait
                    wait_time = max(0.5, seconds_to_wait + jitter)
                    
                    print(f"Rate limit approaching: {max_ratio:.2f} ratio, waiting {wait_time:.2f}s, " +
                          f"req: {projected_requests}/{self.requests_per_min}, " +
                          f"tokens: {projected_input}/{self.input_tokens_per_min}")
                    
                    await asyncio.sleep(wait_time)
                
            # IMPORTANT: We no longer increment counters here!
            # Token reservation happens separately in reserve_tokens()
    
    async def reserve_tokens(self, input_tokens_estimate: int = 500, output_tokens_estimate: int = 300):
        """Reserve token capacity for an upcoming request"""
        async with self.lock:
            await self.reset_if_needed()
            
            # Reserve capacity
            self.request_count += 1
            self.input_token_count += input_tokens_estimate
            self.output_token_count += output_tokens_estimate
            
            return {
                "request_id": self.request_count,
                "input_tokens": input_tokens_estimate,
                "output_tokens": output_tokens_estimate
            }

    async def release_tokens(self, reservation: dict):
        """Release tokens if an API call fails"""
        async with self.lock:
            # Don't decrement request count below 0
            self.request_count = max(0, self.request_count - 1)
            
            # Release the reserved tokens
            self.input_token_count = max(0, self.input_token_count - reservation["input_tokens"])
            self.output_token_count = max(0, self.output_token_count - reservation["output_tokens"])
            
            print(f"Released reservation: request_id={reservation['request_id']}, " +
                  f"input_tokens={reservation['input_tokens']}, " +
                  f"output_tokens={reservation['output_tokens']}")

    async def force_reset_counters(self):
        """Force reset all counters if they get out of sync"""
        async with self.lock:
            prev_request = self.request_count
            prev_input = self.input_token_count
            prev_output = self.output_token_count
            
            self.request_count = 0
            self.input_token_count = 0
            self.output_token_count = 0
            self.last_reset = time.time()
            
            print(f"[FORCE RESET] Counters reset from req:{prev_request}, " +
                  f"in:{prev_input}, out:{prev_output} to zeros")

# Diagnostic heartbeat class for tracking progress
class DiagnosticHeartbeat:
    """Provide periodic status updates even when no requests complete"""
    
    def __init__(self, interval_seconds: int = 30):
        self.interval = interval_seconds
        self.last_heartbeat = time.time()
        self.start_time = time.time()
        self.api_call_time = 0.0
        self.backoff_time = 0.0
        self.successful_calls = 0
        self.failed_calls = 0
        self.lock = asyncio.Lock()
        self._task = None
        
    async def start(self):
        """Start the heartbeat"""
        self._task = asyncio.create_task(self._heartbeat_loop())
        
    async def stop(self):
        """Stop the heartbeat"""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _heartbeat_loop(self):
        """Loop that prints periodic status updates"""
        try:
            while True:
                await asyncio.sleep(self.interval)
                await self.print_status()
        except asyncio.CancelledError:
            # Final status update before stopping
            await self.print_status()
            raise
    
    async def record_api_call_time(self, duration: float, success: bool):
        """Record time spent in API calls"""
        async with self.lock:
            self.api_call_time += duration
            if success:
                self.successful_calls += 1
            else:
                self.failed_calls += 1
    
    async def record_backoff_time(self, duration: float):
        """Record time spent in backoff"""
        async with self.lock:
            self.backoff_time += duration
    
    async def print_status(self):
        """Print current status"""
        async with self.lock:
            now = time.time()
            total_time = now - self.start_time
            time_since_last = now - self.last_heartbeat
            
            # Calculate percentages
            api_pct = (self.api_call_time / total_time) * 100 if total_time > 0 else 0
            backoff_pct = (self.backoff_time / total_time) * 100 if total_time > 0 else 0
            other_pct = 100 - api_pct - backoff_pct
            
            print(f"\n[HEARTBEAT] Status at {time.strftime('%H:%M:%S')} " +
                  f"({time_since_last:.1f}s since last update)")
            print(f"[HEARTBEAT] Runtime: {total_time:.1f}s, API calls: " +
                  f"{self.successful_calls} successful, {self.failed_calls} failed")
            print(f"[HEARTBEAT] Time distribution: API calls: {api_pct:.1f}%, " +
                  f"Backoff: {backoff_pct:.1f}%, Other: {other_pct:.1f}%")
            
            self.last_heartbeat = now

# Create a global heartbeat instance
DIAGNOSTIC_HEARTBEAT = DiagnosticHeartbeat(interval_seconds=30)

# Initialize rate limits for each model based on documented limits
MODEL_RATE_LIMITS = {
    "claude-3-5-sonnet-20241022": ModelRateLimits(
        requests_per_min=2000,
        input_tokens_per_min=160000,
        output_tokens_per_min=32000
    ),
    "claude-3-7-sonnet-20250219": ModelRateLimits(
        requests_per_min=2000,
        input_tokens_per_min=80000,
        output_tokens_per_min=32000
    ),
    "gpt-4o": ModelRateLimits(
        requests_per_min=50000,  # Updated based on new limits (50K RPM)
        input_tokens_per_min=150000000,  # 150M tokens per minute
        output_tokens_per_min=150000000  # 150M tokens per minute
    )
}


def load_dataframe_from_json_directory(path: str) -> pd.DataFrame:
    """Load a dataframe from a json directory."""
    data = []
    for file in list(Path(path).glob("**/*.json")):
        with open(file, encoding="utf-8") as f:
            data.append(json.load(f))
    return pd.DataFrame(data)


def flatten_list(nested_list: list[list[Any]]) -> list[Any]:
    """Flatten a nested list of items into a single list.

    Args:
        nested_list: A list containing sublists to flatten

    Returns:
        A flattened list containing all items from sublists
    """
    return [item for sublist in nested_list for item in sublist]


async def send_message_to_llm(
    message: list[dict[str, str]], model: str, sem: Optional[Semaphore]
) -> Any:
    """Send a message to a language model with rate limiting.

    Args:
        message: The message to send to the LLM
        model: The model identifier to use
        sem: Semaphore for rate limiting requests (can be None if semaphore is managed by caller)

    Returns:
        The response from the language model
    """
    # Set appropriate max_tokens based on model to avoid output token rate limits
    max_tokens = 4000 if "claude" in model.lower() else 8000
    
    # Get rate limiter for this model if available
    rate_limiter = MODEL_RATE_LIMITS.get(model)
    
    # Estimate message input tokens (rough approximation)
    input_tokens_estimate = sum(len(m.get("content", "")) // 3 for m in message)
    output_tokens_estimate = max_tokens // 2  # Estimate half of max will be used
    
    # Create a unique request ID for tracking this API call
    request_id = f"{model}-{int(time.time() * 1000) % 10000}"
    
    # Token reservation to track if we need to release on failure
    token_reservation = None
    
    # Semaphore handling
    cm = sem if sem is not None else nullcontext()
    
    # Record API call start time
    call_start_time = time.time()
    print(f"[API {request_id}] Starting API request to {model} at {time.strftime('%H:%M:%S')}")
    
    # Use the appropriate context manager
    try:
        async with cm:
            # If we have a rate limiter for this model, wait if needed
            if rate_limiter:
                await rate_limiter.wait_if_needed(
                    input_tokens_estimate=input_tokens_estimate,
                    output_tokens_estimate=output_tokens_estimate
                )
                
                # Reserve tokens AFTER waiting
                token_reservation = await rate_limiter.reserve_tokens(
                    input_tokens_estimate=input_tokens_estimate,
                    output_tokens_estimate=output_tokens_estimate
                )
            
            # Add a small random delay before making the API call to stagger requests
            await asyncio.sleep(0.1 + random.random() * 0.2)
            
            # Make the API call with a proper timeout that will actually cause it to fail
            print(f"[API {request_id}] Attempting litellm.acompletion with model {model}")
            try:
                # Add debug info about the request
                print(f"[API {request_id}] Request details:")
                print(f"[API {request_id}] - Model: {model}")
                print(f"[API {request_id}] - Message count: {len(message)}")
                if len(message) > 0:
                    print(f"[API {request_id}] - First message role: {message[0].get('role', 'unknown')}")
                    content = message[0].get('content', '')
                    content_preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"[API {request_id}] - Content preview: {content_preview}")
                
                # Important: Set a timeout value with asyncio.wait_for that's LESS than the litellm timeout
                # This ensures our timeout actually works before litellm's internal timeout
                timeout_seconds = 20  # Lower than default litellm timeout
                print(f"[API {request_id}] Setting timeout to {timeout_seconds} seconds")
                
                # Create a future that we can explicitly cancel if needed
                api_task = asyncio.create_task(
                    litellm.acompletion(
                        model=model, 
                        messages=message,
                        max_tokens=max_tokens,
                        request_timeout=15  # Even lower internal timeout
                    )
                )
                
                # Wait for the task with a timeout
                start_time = time.time()
                try:
                    response = await asyncio.wait_for(api_task, timeout=timeout_seconds)
                    elapsed = time.time() - start_time
                    print(f"[API {request_id}] litellm.acompletion completed in {elapsed:.2f}s!")
                except asyncio.TimeoutError:
                    # Important: Cancel the task explicitly when timeout occurs
                    if not api_task.done():
                        print(f"[API {request_id}] Cancelling API task after timeout")
                        api_task.cancel()
                    elapsed = time.time() - start_time
                    print(f"[API {request_id}] TIMEOUT: The API call timed out after {elapsed:.2f}s")
                    raise
            except asyncio.TimeoutError:
                print(f"[API {request_id}] TIMEOUT: The API call timed out after {timeout_seconds}s")
                raise
            except Exception as e:
                print(f"[API {request_id}] CRITICAL API ERROR: {type(e).__name__}: {str(e)}")
                raise
            
            # Track token usage if we have a rate limiter
            if rate_limiter and hasattr(response, 'usage') and response.usage:
                # Log actual token usage vs estimated
                print(f"[API {request_id}] Token usage - Estimated: {input_tokens_estimate} in, {output_tokens_estimate} out " +
                      f"Actual: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
                
                # Update with actual usage instead of estimate
                await rate_limiter.add_usage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
                
                # Since we're using actual values now, release our reservation
                if token_reservation:
                    await rate_limiter.release_tokens(token_reservation)
            
            # Record successful API call time
            call_duration = time.time() - call_start_time
            await DIAGNOSTIC_HEARTBEAT.record_api_call_time(call_duration, success=True)
            print(f"[API {request_id}] Request to {model} completed successfully in {call_duration:.2f}s")
            
            return response
    except Exception as e:
        # If we had a token reservation, release the tokens since the call failed
        if rate_limiter and token_reservation:
            await rate_limiter.release_tokens(token_reservation)
        
        # Record failed API call time
        call_duration = time.time() - call_start_time
        await DIAGNOSTIC_HEARTBEAT.record_api_call_time(call_duration, success=False)
        
        # Re-raise the exception
        raise


models = {
    "4o": "gpt-4o",
    "claude35": "claude-3-5-sonnet-20241022",
    "claude37": "claude-3-7-sonnet-20250219",
}


async def process_model_batch(
    eval_df: pd.DataFrame, model_key: str, model_name: str, max_concurrent: int
) -> tuple[str, list[Any]]:
    """Process batch for a single model.

    Args:
        eval_df: Dataframe containing evaluation data
        model_key: Key for the model (e.g., "4o", "claude")
        model_name: Full name of the model to use
        max_concurrent: Maximum number of concurrent requests allowed

    Returns:
        Tuple of (model_key, results) for updating the dataframe
    """
    # Start a timer for the entire model batch
    model_batch_start = time.time()
    print(f"[MODEL BATCH] Starting processing for {model_name} at {time.strftime('%H:%M:%S')}")
    
    # Get rows for this model
    batch = eval_df.loc[eval_df.run_name.str.contains(model_key), "content"].tolist()
    
    if not batch:
        print(f"[{model_key}] No prompts found")
        return model_key, []
    
    # Determine model-specific concurrency settings
    # Calculate appropriate concurrency based on rate limits
    if model_name in MODEL_RATE_LIMITS:
        limits = MODEL_RATE_LIMITS[model_name]
        
        # For Claude 3.7, allow more concurrency than before but still be conservative
        if model_name == "claude-3-7-sonnet-20250219":
            # Increase from 1 to 2 for Claude 3.7 to balance throughput and reliability
            safe_concurrency = 2
            print(f"[{model_key}] Setting concurrency to 2 for Claude 3.7 (was 1 before)")
            print(f"[{model_key}] Rate limits: {limits.requests_per_min} RPM, {limits.input_tokens_per_min} input TPM")
        
        # For Claude 3.5, also increase concurrency slightly
        elif model_name == "claude-3-5-sonnet-20241022":
            # Increase from 2 to 3 for Claude 3.5
            safe_concurrency = 3
            print(f"[{model_key}] Setting concurrency to 3 for Claude 3.5 (was 2 before)")
            print(f"[{model_key}] Rate limits: {limits.requests_per_min} RPM, {limits.input_tokens_per_min} input TPM")
            
        # For other models, use the passed max_concurrent but cap at reasonable limit
        else:
            # Ensure minimum concurrency of 1 for all models
            safe_concurrency = max(1, min(10, max_concurrent))
            print(f"[{model_key}] Using concurrency: {safe_concurrency}")
    else:
        # Default fallback if no rate limits defined
        safe_concurrency = max_concurrent
        print(f"[{model_key}] Using default concurrency: {safe_concurrency}")
    
    # Add a delay before starting Claude models to stagger processing
    if "claude" in model_name.lower():
        if model_name == "claude-3-7-sonnet-20250219":
            # Reduce delay for Claude 3.7
            delay = 1.0  # Was 2.0 before
            print(f"[{model_key}] Adding {delay}s startup delay for Claude 3.7")
            await asyncio.sleep(delay)
        elif model_name == "claude-3-5-sonnet-20241022":
            # Keep the same delay for Claude 3.5 
            delay = 1.0
            print(f"[{model_key}] Adding {delay}s startup delay for Claude 3.5")
            await asyncio.sleep(delay)
    
    # Process batch
    print(f"[{model_key}] Processing batch of {len(batch)} prompts")
    results = await process_batch(batch, model_name, max_concurrent=safe_concurrency)
    
    # Log completion of the entire model batch
    model_batch_duration = time.time() - model_batch_start
    success_count = sum(1 for r in results if r is not None)
    print(f"[MODEL BATCH] Completed processing for {model_name} in {model_batch_duration:.2f}s")
    print(f"[MODEL BATCH] Final success rate: {success_count}/{len(batch)} ({success_count/len(batch):.1%})")
    
    return model_key, results


async def run_eval_loop(
    eval_df: pd.DataFrame, max_concurrent: int = 100
) -> pd.DataFrame:
    """Process evaluation dataframe with multiple LLM models sequentially.

    Sends prompts from the dataframe to different LLM models based on the run_name
    and collects the results in the dataframe. Processes one model at a time
    to avoid issues with concurrent API access.

    Args:
        eval_df: Dataframe containing evaluation data with prompts
        max_concurrent: Maximum number of concurrent requests allowed

    Returns:
        Updated dataframe with model responses in the llm_answer column
    """
    # Start the diagnostic heartbeat
    await DIAGNOSTIC_HEARTBEAT.start()
    
    try:
        # Add a little color based on model type for easier visual tracking
        model_colors = {
            "4o": "\033[1;32m",  # Light Green for GPT-4o
            "claude35": "\033[38;2;210;154;134m",  # Light Orange for Claude 3.5
            "claude37": "\033[38;2;203;122;95m",  # Darker Orange for Claude 3.7
        }
        reset_color = "\033[0m"
        blue_color = "\033[0;34m"
        
        # Ensure llm_answer column is of object type to handle mixed data types
        if "llm_answer" not in eval_df.columns:
            eval_df["llm_answer"] = None
        eval_df["llm_answer"] = eval_df["llm_answer"].astype("object")

        # Models to process
        model_batches = []
        
        print(f"{blue_color}[Postprocessing]{reset_color} Identifying models in the dataset")
        
        for model_key, model_name in models.items():
            # Check if there are any rows for this model
            if eval_df.run_name.str.contains(model_key).any():
                model_color = model_colors.get(model_key, "")
                prompt_count = eval_df.run_name.str.contains(model_key).sum()
                model_batches.append((model_key, model_name))
                print(f"{model_color}[{model_key}]{reset_color} Found {prompt_count} prompts for model: {model_name}")
        
        # Print summary
        print(f"{blue_color}[Postprocessing]{reset_color} Processing {len(model_batches)} models SEQUENTIALLY")
        print(f"{blue_color}[Postprocessing]{reset_color} Important: Processing one model at a time for stability")
        
        if not model_batches:
            print(f"{blue_color}[Postprocessing]{reset_color} No matching model keys found in the dataframe")
            return eval_df

        # Process each model sequentially (one at a time)
        for model_key, model_name in model_batches:
            model_color = model_colors.get(model_key, "")
            print(f"{model_color}[{model_key}]{reset_color} Starting processing task for {model_name}")
            
            try:
                # Process this model
                result_key, model_results = await process_model_batch(eval_df, model_key, model_name, max_concurrent)
                
                # Update the dataframe with results
                mask = eval_df.run_name.str.contains(model_key)
                
                if mask.any() and model_results:
                    if len(model_results) == mask.sum():
                        eval_df.loc[mask, "llm_answer"] = model_results
                        print(f"{model_color}[{model_key}]{reset_color} Updated {len(model_results)} answers")
                    else:
                        print(f"{model_color}[{model_key}]{reset_color} Warning: Mismatch in result count. Expected {mask.sum()}, got {len(model_results)}")
                        
                        # Still try to update as many as possible (partial results are better than none)
                        if len(model_results) > 0:
                            count = min(len(model_results), mask.sum())
                            eval_df.loc[mask.idxmax()[:count], "llm_answer"] = model_results[:count]
                            print(f"{model_color}[{model_key}]{reset_color} Updated {count} answers with partial results")
                else:
                    print(f"{model_color}[{model_key}]{reset_color} No results to update")
                
                # Insert a short delay between models
                await asyncio.sleep(2)
                    
            except Exception as e:
                print(f"{blue_color}[Postprocessing]{reset_color} ERROR: Exception processing {model_name}: {type(e).__name__}: {e}")
                # Continue with next model

        success_count = (~eval_df["llm_answer"].isna()).sum()
        total_count = len(eval_df)
        print(f"{blue_color}[Postprocessing]{reset_color} Overall success: {success_count}/{total_count} prompts processed ({success_count/total_count:.1%})")
    
    finally:
        # Stop the heartbeat before exiting
        await DIAGNOSTIC_HEARTBEAT.stop()
    
    return eval_df


async def process_single(prompt: str, model: str, sem: Optional[Semaphore]) -> Optional[str]:
    """Process a single prompt with a language model with retry logic.

    Makes up to 3 attempts (reduced from 7) to get a response from the model.
    Uses exponential backoff with jitter to avoid bursty retries.

    Args:
        prompt: The prompt to send to the model
        model: The model identifier to use
        sem: Semaphore for rate limiting requests (can be None if semaphore is managed by caller)

    Returns:
        The model's response content as string or None if all attempts fail
    """
    # Start a timer to track API call time
    call_start_time = time.time()
    
    # Create a unique request ID for tracking this API call
    request_id = f"{model}-{int(call_start_time * 1000) % 10000}"
    print(f"[API {request_id}] Starting API request to {model} at {time.strftime('%H:%M:%S')}")
    
    messages = [
        {"role": "user", "content": prompt},
    ]

    # IMPORTANT: Balancing retries and timeouts
    MAX_RETRIES = 2  # 3 attempts total (1 initial + 2 retries)
    overall_timeout = 75  # Max time in seconds for all retries combined
    
    # Track the total duration to implement an overall timeout
    for attempt in range(MAX_RETRIES + 1):  # MAX_RETRIES + 1 attempts total
        # Check if we've exceeded the overall timeout
        elapsed_total = time.time() - call_start_time
        if elapsed_total > overall_timeout:
            print(f"[API {request_id}] OVERALL TIMEOUT: Exceeded {overall_timeout}s total execution time")
            return None
            
        # Log the attempt number
        print(f"[API {request_id}] Attempt {attempt + 1}/{MAX_RETRIES + 1}")
        
        # Add a randomized startup delay on retries to avoid multiple retries bursting simultaneously
        # This delay gets longer with each retry (exponential) and includes significant jitter
        if attempt > 0:
            # Add exponential backoff with jitter to avoid retry storms
            base_delay = min(4.0 * (2 ** (attempt - 1)), 30)  # Cap at 30 seconds
            jitter = random.uniform(0.5, 1.5)  # 50% below to 50% above base delay
            actual_delay = base_delay * jitter
            print(f"[API {request_id}] Exponential backoff: waiting {actual_delay:.2f}s before retry {attempt+1}")
            await asyncio.sleep(actual_delay)
        
        try:
            # DIRECT CALL: Use litellm directly with optimized parameters
            
            # Set max tokens based on model
            max_tokens = 3500 if "claude" in model.lower() else 4000
            
            # Add a small jitter on initial request to desynchronize requests
            initial_jitter = random.uniform(0.1, 0.5)  
            await asyncio.sleep(initial_jitter)
            
            # Use a strict API timeout to fail fast
            api_timeout = 20
            print(f"[API {request_id}] Calling litellm.acompletion with timeout={api_timeout}s, max_tokens={max_tokens}")
            
            # Directly use litellm with wait_for for stricter timeout enforcement
            try:
                resp_future = litellm.acompletion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    request_timeout=api_timeout,
                    temperature=0.0  # Use deterministic outputs to improve caching
                )
                
                # Use our own timeout (slightly longer than API timeout)
                res = await asyncio.wait_for(resp_future, timeout=api_timeout + 5)
                
                # Log successful API call completion
                call_duration = time.time() - call_start_time
                print(f"[API {request_id}] Request to {model} completed successfully in {call_duration:.2f}s")
                return res.choices[0].message.content
                
            except asyncio.TimeoutError:
                print(f"[API {request_id}] Timeout after {api_timeout + 5}s waiting for API response")
                raise  # Re-raise to be caught by the outer try/except
                
        except asyncio.TimeoutError as e:
            # Record for diagnostics
            elapsed = time.time() - call_start_time
            print(f"[API {request_id}] TIMEOUT ERROR on attempt {attempt + 1}: {elapsed:.2f}s elapsed")
            
            # Continue to next attempt (will have exponential backoff applied)
            continue
            
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Enhanced rate limit error detection
            rate_limit_terms = [
                "rate limit", "ratelimit", "too many requests", "429", 
                "capacity", "quota", "exceeded", "throttl", "tps limit", 
                "token rate", "too fast", "server busy", "overloaded"
            ]
            is_rate_limit = any(term in error_str for term in rate_limit_terms)
            
            # Only reset rate limiters once per process to avoid constant resets
            if is_rate_limit and attempt == 0:
                rate_limiter = MODEL_RATE_LIMITS.get(model)
                if rate_limiter:
                    print(f"[API {request_id}] Force resetting token counters after rate limit error")
                    await rate_limiter.force_reset_counters()
            
            # Log error and continue to next attempt with exponential backoff
            print(f"[API {request_id}] ERROR on attempt {attempt + 1}: {error_type}: {e}")
            
            # If this was the last attempt, give up
            if attempt >= MAX_RETRIES:
                print(f"[API {request_id}] FAILED: All {MAX_RETRIES + 1} attempts failed. Last error: {error_type}")
                total_duration = time.time() - call_start_time
                print(f"[API {request_id}] Total time spent: {total_duration:.2f}s before giving up")
                return None
            
            continue
            
    # Should never reach here, but just in case
    return None


async def process_with_progress(
    prompt: str, model: str, sem: Semaphore, pbar: tqdm
) -> Optional[str]:
    """Process a single prompt and update progress bar.

    Callback function that processes a prompt and ensures the progress bar
    is updated even if an exception occurs.

    Args:
        prompt: The prompt to process
        model: The model identifier to use
        sem: Semaphore for rate limiting requests
        pbar: tqdm progress bar to update

    Returns:
        The result from processing the prompt or None if processing fails
    """
    # CRITICAL FIX: Use a separate try/except/finally to ensure semaphore release
    # This ensures we don't have multiple concurrent tasks even with asyncio
    try:
        # Acquire semaphore here and pass None to process_single
        async with sem:
            # Explicitly wait a random short amount to avoid exact simultaneous API calls
            await asyncio.sleep(0.05 + random.random() * 0.2)
            # Process without semaphore (we already acquired it here)
            result = await process_single(prompt, model, None)
            return result
    except Exception as e:
        print(f"Error in process_with_progress: {type(e).__name__}: {e}")
        return None
    finally:
        pbar.update(1)


async def process_batch(
    prompts: list[str], model: str, max_concurrent: int = 5
) -> list[Optional[str]]:
    """Process a batch of prompts concurrently with rate limiting and progress tracking.
    
    Sets appropriate concurrency based on the model being used, with defaults
    that respect API rate limits. Uses a token bucket approach for Claude models.

    Args:
        prompts: List of prompts to process
        model: The model identifier to use
        max_concurrent: Maximum number of concurrent requests allowed

    Returns:
        List of results from processing each prompt (strings or None values)
    """
    # Start a timer to track overall batch processing time
    batch_start_time = time.time()
    print(f"[DIAGNOSTIC] Starting batch processing for {model} at {time.strftime('%H:%M:%S')}")
    
    # IMPORTANT: Much more conservative concurrency limits
    if "claude-3-7-sonnet" in model:
        # Only allow 1 concurrent for Claude 3.7 (most restrictive)
        effective_max = 1
        print(f"[RATE LIMIT] Setting concurrency to {effective_max} for Claude 3.7 (very conservative)")
    elif "claude-3-5-sonnet" in model:
        # Only allow 2 concurrent for Claude 3.5
        effective_max = 2
        print(f"[RATE LIMIT] Setting concurrency to {effective_max} for Claude 3.5 (conservative)")
    else:
        # For other models like GPT-4o, reduce concurrency too
        effective_max = min(max_concurrent, 5)
        print(f"[RATE LIMIT] Setting concurrency to {effective_max} for {model} (reduced)")
    
    print(f"Processing batch with {effective_max} concurrent requests for model {model}")
    sem = Semaphore(effective_max)

    # Setup progress bar
    pbar = tqdm(total=len(prompts), desc=f"Processing {model}")
    
    # Process in smaller chunks to better manage rate limits and avoid overwhelming the API
    # Using EVEN smaller chunk sizes to prevent timeouts
    if "claude-3-7-sonnet" in model:
        # Just 1 at a time for Claude 3.7
        chunk_size = 1
        print(f"[DEBUG] Using single-item chunk_size={chunk_size} for Claude 3.7")
    elif "claude-3-5-sonnet" in model:
        # Smaller chunks (2) for Claude 3.5
        chunk_size = 2
        print(f"[DEBUG] Using reduced chunk_size={chunk_size} for Claude 3.5")
    else:
        # Smaller chunks for other models too
        chunk_size = 4
        print(f"[DEBUG] Using reduced chunk_size={chunk_size} for {model}")
    
    # Process prompts sequentially if needed
    if "claude" in model.lower():
        print(f"[DEBUG] Processing Claude model with sequential approach")
        results = []
        
        # For Claude models, process one at a time to avoid any rate limit issues
        for i, prompt in enumerate(prompts):
            print(f"Processing item {i+1}/{len(prompts)} for {model}")
            
            try:
                # Process directly (no concurrency for Claude)
                result = await process_single(prompt, model, None)
                results.append(result)
                # Update progress bar
                pbar.update(1)
                
                # Add a small cooldown between requests
                if i < len(prompts) - 1:  # Skip cooldown after last item
                    cooldown = 1.0 + (random.random() * 1.0)
                    print(f"Cooldown for {cooldown:.2f}s before next request")
                    await asyncio.sleep(cooldown)
                
            except Exception as e:
                print(f"Error processing item {i+1} with {model}: {type(e).__name__}: {e}")
                results.append(None)
                pbar.update(1)
    else:
        # For non-Claude models (like GPT-4o), we can use chunks
        results = []
        
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i:i+chunk_size]
            chunk_num = i//chunk_size + 1
            total_chunks = (len(prompts) + chunk_size - 1)//chunk_size
            
            # Log start of chunk processing with timestamp
            chunk_start_time = time.time()
            print(f"Processing chunk {chunk_num}/{total_chunks} for {model}")
            print(f"[DIAGNOSTIC] Starting chunk {chunk_num}/{total_chunks} at {time.strftime('%H:%M:%S')}")
            
            # Create tasks for this chunk
            tasks = [process_with_progress(prompt, model, sem, pbar) for prompt in chunk]
            
            try:
                # Process chunk with gather
                chunk_results = await asyncio.gather(*tasks)
                results.extend(chunk_results)
            except Exception as e:
                print(f"Error processing chunk with {model}: {type(e).__name__}: {e}")
                # Handle the failure by adding None values for all tasks in this chunk
                results.extend([None] * len(chunk))
            
            # Log end of chunk processing
            chunk_duration = time.time() - chunk_start_time
            chunk_success = sum(1 for r in chunk_results if r is not None) if 'chunk_results' in locals() else 0
            print(f"[DIAGNOSTIC] Chunk {chunk_num}/{total_chunks} completed in {chunk_duration:.2f} seconds")
            print(f"[DIAGNOSTIC] Chunk success rate: {chunk_success}/{len(chunk)} ({chunk_success/len(chunk):.1%})")
            
            # Insert a cooldown between chunks
            if i + chunk_size < len(prompts):
                cooldown = 1.0 + (random.random() * 1.0)
                print(f"Brief cooldown for {cooldown:.2f}s before next chunk")
                print(f"[DIAGNOSTIC] Starting cooldown at {time.strftime('%H:%M:%S')}")
                await asyncio.sleep(cooldown)
    
    # Log results
    success_count = len([r for r in results if r is not None])
    print(f"Batch completed with {success_count}/{len(prompts)} successful responses ({success_count/len(prompts):.1%})")
    
    # Close the progress bar
    pbar.close()
    
    # Log batch completion time for debugging
    batch_duration = time.time() - batch_start_time
    success_count = sum(1 for r in results if r is not None)
    print(f"[DIAGNOSTIC] Batch processing for {model} completed in {batch_duration:.2f} seconds")
    print(f"[DIAGNOSTIC] Success rate: {success_count}/{len(results)} ({success_count/len(results):.1%})")
    
    return results


def encode_image_to_base64(image: str) -> str:
    """Encode an already base64-encoded image string to base64 again.

    This function is used when the image data needs to be standardized
    to ensure consistent handling.

    Args:
        image: Base64 encoded image data

    Returns:
        Re-encoded base64 string
    """
    decoded_image = base64.b64decode(image)
    return base64.b64encode(decoded_image).decode("utf-8")


def load_notebook(notebook: str | dict[str, Any]) -> dict[str, Any]:
    """Parse a notebook into nbformat.

    Attempts to parse a notebook into a dictionary format using nbformat.

    Args:
        notebook: The notebook to parse, which could be a string or a dictionary

    Returns:
        Dictionary representation of the notebook or empty dict if parsing fails
    """
    if isinstance(notebook, str):
        return nbformat.reads(json.dumps(ast.literal_eval(notebook)), as_version=4)
    return nbformat.from_dict(notebook)


def load_answer(answer: str | dict[str, Any]) -> dict[str, Any]:
    """Parse an answer into a dictionary format.

    Attempts multiple parsing methods: direct dict access, ast.literal_eval,
    and json.loads to handle different input formats.

    Args:
        answer: The answer to parse, which could be a string, dict, or other format

    Returns:
        Dictionary representation of the answer or empty dict if parsing fails
    """
    if not answer:
        return {}
    if isinstance(answer, dict):
        return answer
    try:
        # Try literal eval first
        return ast.literal_eval(answer)
    except (ValueError, SyntaxError):
        try:
            # Fallback to json loads
            return json.loads(answer)
        except (ValueError, TypeError, json.JSONDecodeError):
            # Return empty dict if parsing fails
            return {}


def create_eval_df(data: list[dict[str, Any]]) -> pd.DataFrame:
    """Creates a dataframe for evaluation with one row per question.

    Uses vectorized operations for better performance.

    Args:
        data: List of dictionaries containing problem data

    Returns:
        DataFrame with one row per question, including formatted questions and prompts
    """
    # First, apply load_answer to all relevant columns at once
    evaluation_data = data.copy()

    # Handle list type agent answers
    for col in ["agent_answer", "mcq_question", "mcq_options"]:
        mask = evaluation_data[col].apply(lambda x: isinstance(x, list))
        evaluation_data.loc[mask, col] = evaluation_data.loc[mask, col].apply(
            lambda x: {f"q{i + 1}": v for i, v in enumerate(x)}
        )

    # Filter out rows without agent answers
    evaluation_data = evaluation_data[evaluation_data["agent_answer"].apply(bool)]

    # Now prepare for explosion
    # Create a column with question numbers from ideal_answer keys
    evaluation_data["question_keys"] = evaluation_data["ideal_answer"].apply(
        lambda x: list(x.keys())
    )
    # Explode the dataframe to create one row per question
    exploded = evaluation_data.explode("question_keys")

    # Now create the final dataframe in a vectorized way
    result = pd.DataFrame({
        "uuid": exploded["problem_id"] + "_" + exploded["question_keys"].astype(str),
        "problem_id": exploded["problem_id"],
        "question": exploded.apply(
            lambda row: row["mcq_question"].get(row["question_keys"], None), axis=1
        ),
        "question_num": exploded["question_keys"],
        "agent_answer": exploded.apply(
            lambda row: row["agent_answer"].get(row["question_keys"], None), axis=1
        ),
        "ideal_answer": exploded.apply(
            lambda row: row["ideal_answer"].get(row["question_keys"], None), axis=1
        ),
        "run_name": exploded["run_name"],
        "md_notebook": exploded["md_notebook"],
        "md_images": exploded["md_images"],
        "mcq_options": exploded.apply(
            lambda row: row["mcq_options"].get(row["question_keys"], None), axis=1
        ),
        "refusal_option": exploded.get("refusal_option", None),
        "question_format": exploded.get("question_format", None),
        "model": exploded.get("model", None),
    })

    # Drop rows with no question or no format
    result = result.dropna(
        subset=["question", "question_format"], how="any"
    ).reset_index(drop=True)

    # Drop MCQ questions with any NaN values
    mcq_mask = result["question_format"] == "mcq"
    result = result[~(mcq_mask & result.isna().any(axis=1))]

    # Apply MCQ formatting only to MCQ questions
    mcq_rows = result[result["question_format"] == "mcq"].index
    if len(mcq_rows) > 0:
        result.loc[
            mcq_rows, ["formatted_question", "correct_letter", "refusal_letter"]
        ] = result.loc[mcq_rows].apply(
            lambda row: pd.Series(
                questions_to_mcq(
                    row["question"],
                    row["mcq_options"],
                    refusal_option=row["refusal_option"],
                ),
                index=["formatted_question", "correct_letter", "refusal_letter"],
            ),
            axis=1,
        )
    result["prompt"] = result.apply(create_prompt, axis=1)
    result["content"] = result.apply(create_llm_message_content, axis=1)

    return result


def questions_to_mcq(
    question: str, options: list[str | dict[str, Any]], refusal_option: bool = True
) -> tuple[str, str, Optional[str]]:
    """Format a question and options into an MCQ format.

    Creates a formatted multiple-choice question with lettered options,
    randomly shuffles the options, and tracks the correct answer letter
    and optional refusal option letter.

    Args:
        question: The question text
        options: List of answer options with correct answer as first element
        refusal_option: Whether to include an "Insufficient information" option

    Returns:
        Tuple of (formatted question string, correct answer letter, refusal option letter)
    """
    options = options.copy()
    # Get all answer options
    correct_answer = options[0]
    if refusal_option:
        options.append("Insufficient information to answer the question")

    # Randomly shuffle options
    random.shuffle(options)

    # Find the index of the ideal answer to determine its letter
    correct_letter = chr(65 + options.index(correct_answer))
    if refusal_option:
        refusal_letter = chr(
            65 + options.index("Insufficient information to answer the question")
        )
    else:
        refusal_letter = None

    # Format the question with lettered options
    formatted_question = f"{question}\n"
    for j, opt in enumerate(options):
        formatted_question += f"{chr(65 + j)}. {opt}\n"

    # Join all questions with newlines
    return formatted_question, correct_letter, refusal_letter


def create_llm_message_content(row: pd.Series) -> list[dict[str, Any]]:
    """Create a message content structure for LLM API requests.

    Formats text and images from a dataframe row into the format expected
    by multimodal LLM APIs.

    Args:
        row: Dataframe row containing prompt and possibly images

    Returns:
        List of content elements (text and images) for the LLM API
    """
    content = [{"type": "text", "text": row.prompt}]

    if row.md_images:
        for img_data in row.md_images:
            try:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"{img_data}",
                    },
                })
            except Exception as e:
                print(f"Error adding image to content: {e}")

    return content


def create_prompt(row: pd.Series) -> str | float:
    """Create an appropriate prompt based on the question format.

    Selects either open-ended or MCQ prompt template and formats it
    with the data from the input row.

    Args:
        row: Dataframe row containing question and answer data

    Returns:
        Formatted prompt string or np.nan if no matching format
    """
    question_format = row.get("question_format", None)

    if question_format == "open":
        return prompts.OPEN_ENDED_EVAL_PROMPT.format(
            question=row.question,
            correct_answer=row.ideal_answer,
            proposed_answer=row.agent_answer,
        )
    if question_format == "mcq":
        return (
            prompts.MCQ_EVAL_PROMPT.replace("{{notebook}}", row.md_notebook)
            .replace("{{question}}", row.formatted_question)
            .replace("{{proposed_answer}}", str(row.agent_answer))
        )
    return np.nan


def xml_extract(text: str) -> str:
    """Extract an answer letter from XML tags in text.

    Looks for a pattern like <answer>A</answer> and extracts the letter.

    Args:
        text: The text to search for the answer pattern

    Returns:
        The extracted answer letter or 'Z' if no match is found
    """
    if text is None:
        return "Z"  # Return default for None inputs
        
    try:
        match = re.search(r"<answer>([A-Z])</answer>", text)
        if match:
            return match.group(1)
    except TypeError:
        # Handle any other type errors that might occur
        pass
        
    return "Z"


def majority_vote(row: pd.Series, k: int = 10) -> Optional[str]:
    """Apply majority voting to a series of predictions.

    Randomly samples k predictions from the input and returns the most common value.

    Args:
        row: Series of predictions
        k: Number of predictions to sample

    Returns:
        The most common prediction or None if none can be determined
    """
    # Get all predictions excluding the 'answer' column
    predictions = row[:-1]
    # Randomly sample k predictions without replacement
    rng = np.random.default_rng()
    sampled_votes = rng.choice(
        predictions, size=min(k, len(predictions)), replace=False
    )
    # Get mode (most common value) of sampled votes
    # Check if all votes are integers
    if not all(isinstance(vote, int | float | str) for vote in sampled_votes):
        return None
    unique_values, counts = np.unique(sampled_votes, return_counts=True)

    if unique_values.size == 0:
        return None
    return unique_values[np.argmax(counts)]


def run_majority_voting(
    grouped_df: pd.DataFrame, k_values: list[int], n_trials: int
) -> tuple[list[int], list[float], list[float]]:
    """Run majority voting experiments with different k values.

    Applies majority voting with various k values over multiple trials
    and collects accuracy statistics.

    Args:
        grouped_df: Dataframe with predictions grouped by question
        k_values: List of k values to test for majority voting
        n_trials: Number of trials to run for each k value

    Returns:
        Tuple of (k_values, mean accuracies, standard deviations)
    """
    # Check if we have enough data for meaningful majority voting
    if len(grouped_df) < 2:
        print("Insufficient data for majority voting (less than 2 questions)")
        return [], [], []
        
    # Check if we have enough model answers for voting
    if all(len(answers) < 2 for answers in grouped_df["llm_answer"]):
        print("Insufficient model answers for majority voting (less than 2 answers per question)")
        return [], [], []
    
    # Check if we have any valid k values that are less than the number of answers
    max_answers = max(len(answers) for answers in grouped_df["llm_answer"])
    valid_k_values = [k for k in k_values if k <= max_answers]
    
    if not valid_k_values:
        print(f"No valid k values for majority voting (max answers: {max_answers})")
        return [], [], []
    
    # Calculate majority predictions for the maximum valid k value
    majority_predictions = grouped_df["llm_answer"].apply(
        lambda x: majority_vote(x, k=min(max_answers, max(valid_k_values, default=1)))
    )

    # Calculate and display overall accuracy
    accuracy = (majority_predictions == grouped_df["correct_letter"]).mean()
    print(f"Majority voting accuracy: {accuracy:.2%}")

    # Run multiple trials for different k values
    accuracies = {k: [] for k in valid_k_values}

    for k in valid_k_values:
        for _ in range(n_trials):
            # Apply majority voting with current k to each row
            predictions = grouped_df["llm_answer"].apply(
                lambda x, k_value=k: majority_vote(x, k=k_value)
            )

            # Calculate and store accuracy
            acc = (predictions == grouped_df["correct_letter"]).mean()
            accuracies[k].append(acc)

    # Calculate means and standard deviations
    means = [np.mean(accuracies[k]) for k in valid_k_values]
    stds = [np.std(accuracies[k]) for k in valid_k_values]
    return valid_k_values, means, stds


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Calculate Wilson confidence interval for a proportion.

    The Wilson score interval is used to calculate confidence intervals for
    binomial proportions, especially when sample sizes are small.

    Args:
        p: Observed proportion
        n: Sample size
        z: Z-score for desired confidence level (default 1.96 for 95% CI)

    Returns:
        Tuple of (lower bound, upper bound) of the confidence interval
    """
    # Return simple estimate if n is too small
    if n < 5:
        return p, p
        
    # Ensure inputs are valid to avoid NaN results
    p = max(0.0, min(1.0, p))  # Ensure p is between 0 and 1
    n = max(1, n)              # Ensure n is at least 1
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    
    # Handle edge cases to avoid sqrt of negative numbers
    sqrt_term = max(0, p * (1 - p) / n + z**2 / (4 * n**2))
    spread = z * np.sqrt(sqrt_term) / denominator
    
    # Return bounds, ensuring they stay in [0,1]
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    
    # Add debug message for unusual CI behavior
    if lower > p or upper < p:
        print(f"Warning: Wilson CI calculation produced bounds that don't contain the proportion: p={p}, n={n}, CI=[{lower}, {upper}]")
        
    return lower, upper


def calculate_results(
    df: pd.DataFrame, total_questions_per_run: Optional[int] = None
) -> dict[str, dict[str, float]]:
    """
    Calculate means and confidence intervals for each model and format.

    Args:
        df: DataFrame containing model evaluation results
        total_questions_per_run: Total number of questions to normalize
            by as some runs may have failed and were not included in the eval_df

    Returns:
        Dictionary mapping run names to statistics including mean score and confidence intervals
    """
    results = {}
    for run in df["run_name"].unique():
        mask = df["run_name"].str.contains(run)
        scores = df[mask]["correct"]
        if len(scores) > 0:
            mean = (
                scores.sum() / total_questions_per_run
                if total_questions_per_run is not None
                else scores.mean()
            )
            n = (
                total_questions_per_run
                if total_questions_per_run is not None
                else len(scores)
            )
            
            # Handle the case where mean is 0 or very small with no correct answers
            if mean <= 0 or np.isnan(mean):
                ci_low, ci_high = 0.0, 0.0
                print(f"Warning: Zero or NaN mean for {run}, setting CI to [0,0]")
            else:
                ci_low, ci_high = wilson_ci(mean, n)
                
            results[run] = {
                "mean": float(mean),  # Convert numpy float64 to Python float
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
    return results