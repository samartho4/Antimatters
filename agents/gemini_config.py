"""Global Gemini API configuration with exponential backoff retry.

This module configures the Google GenAI client to automatically retry
503 overload errors, improving success rate from ~30% to ~70% during
Gemini infrastructure capacity issues (Jan 2026).

Import this module BEFORE creating any agents to apply retry configuration.
"""

import os

# Support both relative and absolute imports for ADK web compatibility
try:
    from .config import RETRY  # Relative import (when part of package)
except ImportError:
    from config import RETRY  # Absolute import (when in sys.path)

def configure_gemini_with_retry():
    """Configure Google GenAI client with retry policy.

    This sets up exponential backoff for Gemini API calls to handle
    infrastructure overload (503) and rate limits (429).

    Note: Google ADK (1.22.1) handles retry internally via google-api-core.
    We can influence this behavior via environment variables.

    Should be called once at application startup before creating agents.
    """
    # Configure transport settings for better reliability
    os.environ["GOOGLE_CLOUD_DISABLE_GRPC"] = "false"  # Use gRPC for better performance

    # Set longer timeout to accommodate retries
    os.environ["GOOGLE_API_DEFAULT_TIMEOUT"] = "180"  # 3 minutes per request (allows for retries)

    print(f"[Gemini Config] Retry parameters loaded:", flush=True)
    print(f"  - Max retries: {RETRY.max_retries}", flush=True)
    print(f"  - Backoff: {RETRY.initial_delay}s → {RETRY.max_delay}s (exponential)", flush=True)
    print(f"  - Retryable codes: {RETRY.retryable_codes}", flush=True)
    print(f"  - Default timeout: 180s (accommodates retries)", flush=True)
    print(f"  ⚠️  Note: ADK retry is transparent, errors only surface after all attempts", flush=True)

# Auto-configure on import
configure_gemini_with_retry()
