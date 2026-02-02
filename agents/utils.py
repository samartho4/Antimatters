"""Utility functions for agent operations.

Includes exponential backoff retry logic for handling Gemini 503 overload errors
and dynamic model fallback for sustained overload periods.
"""

import time
import random
import logging
import asyncio
from functools import wraps
from typing import Callable, Any, Dict, List, Optional
from google.api_core import exceptions as google_exceptions

from .config import RETRY, FALLBACKS

logger = logging.getLogger(__name__)


# =============================================================================
# Model Switcher for Dynamic Fallback
# =============================================================================

class ModelSwitcher:
    """Tracks model failures and provides fallback models.

    When a model returns 503 UNAVAILABLE repeatedly, this class
    will provide the next model in the fallback chain.

    Usage:
        switcher = ModelSwitcher()
        model = switcher.get_model("research")  # Returns primary

        # After 503 error:
        switcher.record_failure("research")
        model = switcher.get_model("research")  # Returns fallback

        # After success:
        switcher.record_success("research")  # Resets to primary
    """

    def __init__(self):
        self._failure_counts: Dict[str, int] = {}
        self._current_indices: Dict[str, int] = {}
        self._cooldown_until: Dict[str, float] = {}  # Model -> timestamp

    def get_model(self, agent_type: str) -> str:
        """Get the current model for an agent type.

        Args:
            agent_type: One of "research", "engineering", "evolution", "coordinator"

        Returns:
            Model name to use
        """
        chain = getattr(FALLBACKS, agent_type, None)
        if not chain:
            # Return default from MODELS
            from .config import MODELS
            return getattr(MODELS, agent_type, "gemini-2.0-flash")

        idx = self._current_indices.get(agent_type, 0)
        idx = min(idx, len(chain) - 1)  # Clamp to valid range
        return chain[idx]

    def record_failure(self, agent_type: str) -> Optional[str]:
        """Record a 503 failure and possibly switch to fallback model.

        Args:
            agent_type: The agent type that failed

        Returns:
            New model to use, or None if no more fallbacks
        """
        # Increment failure count
        self._failure_counts[agent_type] = self._failure_counts.get(agent_type, 0) + 1
        failures = self._failure_counts[agent_type]

        # Check if we should switch models
        if RETRY.model_rotation_enabled and failures >= RETRY.model_switch_after_retries:
            chain = getattr(FALLBACKS, agent_type, None)
            if chain:
                current_idx = self._current_indices.get(agent_type, 0)
                next_idx = current_idx + 1

                if next_idx < len(chain):
                    self._current_indices[agent_type] = next_idx
                    self._failure_counts[agent_type] = 0  # Reset failure count
                    new_model = chain[next_idx]
                    logger.warning(
                        f"[ModelSwitcher] {agent_type} switching to fallback model: {new_model} "
                        f"(was: {chain[current_idx]})"
                    )
                    print(f"⚠️  [ModelSwitcher] {agent_type} → {new_model} (after {failures} failures)", flush=True)
                    return new_model
                else:
                    logger.error(f"[ModelSwitcher] {agent_type} exhausted all fallback models")
                    print(f"❌ [ModelSwitcher] {agent_type} exhausted all fallbacks", flush=True)
                    return None

        return self.get_model(agent_type)

    def record_success(self, agent_type: str):
        """Record a successful API call, resetting failure count.

        Args:
            agent_type: The agent type that succeeded
        """
        self._failure_counts[agent_type] = 0
        # Note: We don't reset to primary immediately to avoid flip-flopping
        # The model will reset to primary on next server restart

    def reset(self, agent_type: str = None):
        """Reset to primary models.

        Args:
            agent_type: Specific agent to reset, or None for all
        """
        if agent_type:
            self._failure_counts.pop(agent_type, None)
            self._current_indices.pop(agent_type, None)
        else:
            self._failure_counts.clear()
            self._current_indices.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all model assignments."""
        status = {}
        for agent_type in ["research", "engineering", "evolution", "coordinator"]:
            chain = getattr(FALLBACKS, agent_type, [])
            idx = self._current_indices.get(agent_type, 0)
            status[agent_type] = {
                "current_model": chain[idx] if chain else "unknown",
                "fallback_index": idx,
                "total_fallbacks": len(chain),
                "consecutive_failures": self._failure_counts.get(agent_type, 0),
            }
        return status


# Global model switcher instance
model_switcher = ModelSwitcher()


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


# =============================================================================
# Agent Factory with Model Fallback
# =============================================================================

def create_coordinator_with_fallback():
    """Create root_agent with current fallback models.

    This function creates a fresh agent hierarchy using the models
    from ModelSwitcher. Call this after recording failures to get
    agents configured with fallback models.

    Returns:
        root_agent configured with current fallback models
    """
    from google.adk.agents import LlmAgent, SequentialAgent

    # Get current models from switcher
    research_model = model_switcher.get_model("research")
    engineering_model = model_switcher.get_model("engineering")
    evolution_model = model_switcher.get_model("evolution")
    coordinator_model = model_switcher.get_model("coordinator")

    print(f"🔧 [AgentFactory] Creating agents with models:", flush=True)
    print(f"   Research: {research_model}", flush=True)
    print(f"   Engineering: {engineering_model}", flush=True)
    print(f"   Evolution: {evolution_model}", flush=True)
    print(f"   Coordinator: {coordinator_model}", flush=True)

    # Import agent configurations (tools, instructions, etc.)
    from core.agents.antimatters._subagents.research.agent import (
        research_agent as _research_template
    )
    from core.agents.antimatters._subagents.engineering.agent import (
        engineer_coordinator as _engineering_template
    )
    from core.agents.antimatters._subagents.evolution.agent import (
        evolution_agent as _evolution_template
    )

    # Create new agents with fallback models
    # Note: We copy the configuration but use the fallback model
    research_agent = LlmAgent(
        name=_research_template.name,
        model=research_model,
        description=_research_template.description,
        instruction=_research_template.instruction,
        tools=_research_template.tools,
        output_key=getattr(_research_template, 'output_key', None),
    )

    # For engineering, it's a custom BaseAgent - use as-is with model override
    # The engineering agent handles its own model via config
    engineer_coordinator = _engineering_template

    evolution_agent = LlmAgent(
        name=_evolution_template.name,
        model=evolution_model,
        description=_evolution_template.description,
        instruction=_evolution_template.instruction,
        tools=_evolution_template.tools,
        output_key=getattr(_evolution_template, 'output_key', None),
    )

    # Create sequential workflow
    docking_workflow = SequentialAgent(
        name="docking_workflow",
        description="Complete docking workflow with fallback models",
        sub_agents=[research_agent, engineer_coordinator, evolution_agent],
    )

    # Import planning agent (uses evolution model)
    from core.agents.antimatters.coordinator import planning_agent as _planning_template

    planning_agent = LlmAgent(
        name=_planning_template.name,
        model=evolution_model,
        description=_planning_template.description,
        instruction=_planning_template.instruction,
        tools=_planning_template.tools,
        output_key=getattr(_planning_template, 'output_key', None),
    )

    # Create root coordinator
    root_agent = LlmAgent(
        name="antimatters_agent",
        model=coordinator_model,
        description="Antimatters: IDP docking platform with fallback models",
        instruction="""You coordinate IDP docking with TWO MODES.

**MODE DETECTION:**

1. **ANALYSIS MODE** (full workflow with docking):
   Keywords: "analyze", "dock", "experiment", "run workflow", "binding energy", "validate"
   → Transfer to **docking_workflow**

2. **PLANNING MODE** (fast generation, no docking):
   Keywords: "plan", "design", "generate molecules", "fast generation", "use KG"
   → Transfer to **planning_agent**""",
        tools=[],
        sub_agents=[docking_workflow, planning_agent],
        output_key="result",
    )

    return root_agent


async def run_with_model_fallback(runner_factory, user_id: str, session_id: str, message):
    """Run agent with automatic model fallback on 503 errors.

    This is an async generator that wraps the ADK runner.run_async()
    and handles 503 errors by switching to fallback models.

    Args:
        runner_factory: Callable that returns (runner, root_agent) with current models
        user_id: User ID for session
        session_id: Session ID
        message: ADK Message to send

    Yields:
        Events from the agent run
    """
    max_model_switches = 3  # Maximum times to switch models

    for switch_attempt in range(max_model_switches + 1):
        runner, root_agent = runner_factory()

        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message
            ):
                yield event

            # Success - record it and return
            model_switcher.record_success("coordinator")
            return

        except Exception as e:
            error_msg = str(e).lower()
            is_503 = "503" in error_msg or "unavailable" in error_msg or "overloaded" in error_msg

            if is_503 and switch_attempt < max_model_switches:
                # Record failure and get new model
                model_switcher.record_failure("coordinator")
                model_switcher.record_failure("research")
                model_switcher.record_failure("engineering")
                model_switcher.record_failure("evolution")

                delay = RETRY.initial_delay * (RETRY.exponential_base ** switch_attempt)
                if RETRY.jitter:
                    delay = delay * (0.5 + random.random())

                print(f"⏳ [ModelFallback] Waiting {delay:.1f}s before retry with new models...", flush=True)
                await asyncio.sleep(delay)
                continue
            else:
                # Non-503 error or exhausted fallbacks
                raise
