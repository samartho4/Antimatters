"""Utility functions for agent operations.

Includes exponential backoff retry logic for handling Gemini 503 overload errors.
"""

import time
import random
import logging
from functools import wraps
from typing import Callable, Any
from google.api_core import exceptions as google_exceptions

from .config import RETRY

logger = logging.getLogger(__name__)


def exponential_backoff_retry(func: Callable) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Specifically designed to handle Gemini API 503 overload errors.

    Usage:
        @exponential_backoff_retry
        def run_agent(...):
            # Agent logic that may encounter 503
            pass

    Args:
        func: Function to retry

    Returns:
        Wrapped function with retry logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        last_exception = None

        for attempt in range(RETRY.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                # Check if error is retryable
                is_retryable = False

                # Check for Google API errors with status codes
                if hasattr(e, 'code'):
                    if e.code in RETRY.retryable_codes:
                        is_retryable = True

                # Check error message for retryable patterns
                error_msg = str(e).lower()
                for pattern in RETRY.retryable_messages:
                    if pattern in error_msg:
                        is_retryable = True
                        break

                # If not retryable or final attempt, raise
                if not is_retryable or attempt == RETRY.max_retries:
                    logger.error(
                        f"Function {func.__name__} failed after {attempt + 1} attempts: {e}"
                    )
                    raise

                # Calculate backoff delay
                delay = min(
                    RETRY.initial_delay * (RETRY.exponential_base ** attempt),
                    RETRY.max_delay
                )

                # Add jitter to prevent thundering herd
                if RETRY.jitter:
                    delay = delay * (0.5 + random.random())  # 50-150% of base delay

                logger.warning(
                    f"Attempt {attempt + 1}/{RETRY.max_retries + 1} failed for {func.__name__}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

                time.sleep(delay)
                last_exception = e

        # Should never reach here due to raise in loop, but for type safety
        raise last_exception

    return wrapper


def is_transient_error(exception: Exception) -> bool:
    """Check if an exception is a transient error that should be retried.

    Args:
        exception: Exception to check

    Returns:
        True if error is transient, False otherwise
    """
    # Check status codes
    if hasattr(exception, 'code'):
        if exception.code in RETRY.retryable_codes:
            return True

    # Check error messages
    error_msg = str(exception).lower()
    for pattern in RETRY.retryable_messages:
        if pattern in error_msg:
            return True

    return False


def log_retry_attempt(attempt: int, max_attempts: int, error: Exception, delay: float):
    """Log a retry attempt with consistent formatting.

    Args:
        attempt: Current attempt number (1-indexed)
        max_attempts: Maximum number of attempts
        error: Exception that triggered retry
        delay: Delay before next retry in seconds
    """
    logger.warning(
        f"[Retry {attempt}/{max_attempts}] Error: {error.__class__.__name__}: {error}. "
        f"Waiting {delay:.1f}s before retry..."
    )


async def async_generator_with_retry(
    async_gen_func: Callable,
    *args,
    max_retries: int = None,
    **kwargs
):
    """Wrapper for async generators that implements retry logic.

    If the async generator fails mid-stream with a retryable error,
    this will restart the entire generator from the beginning.

    Args:
        async_gen_func: Async generator function to retry
        *args: Positional arguments for the function
        max_retries: Max retry attempts (defaults to RETRY.max_retries)
        **kwargs: Keyword arguments for the function

    Yields:
        Items from the async generator

    Raises:
        Exception: If max retries exceeded or non-retryable error
    """
    max_retries = max_retries or RETRY.max_retries

    for attempt in range(max_retries + 1):
        try:
            async for item in async_gen_func(*args, **kwargs):
                yield item
            # Generator completed successfully
            return

        except Exception as e:
            # Check if error is retryable
            is_retryable = False

            # Check for status codes
            if hasattr(e, 'code'):
                if e.code in RETRY.retryable_codes:
                    is_retryable = True

            # Check error message
            error_msg = str(e).lower()
            for pattern in RETRY.retryable_messages:
                if pattern in error_msg:
                    is_retryable = True
                    break

            # If not retryable or final attempt, raise
            if not is_retryable or attempt == max_retries:
                logger.error(
                    f"Async generator failed after {attempt + 1} attempts: {e}"
                )
                raise

            # Calculate backoff delay
            delay = min(
                RETRY.initial_delay * (RETRY.exponential_base ** attempt),
                RETRY.max_delay
            )

            # Add jitter
            if RETRY.jitter:
                delay = delay * (0.5 + random.random())

            logger.warning(
                f"[Retry {attempt + 1}/{max_retries + 1}] Async generator failed: {e}. "
                f"Restarting from beginning in {delay:.1f}s..."
            )

            time.sleep(delay)
