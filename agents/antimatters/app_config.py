"""
ADK App Configuration for Token Optimization

Implements:
- Context caching (90% token reduction on cached content)
- Context compaction (60-80% reduction on multi-turn)
- Static vs turn instruction pattern

References:
- https://google.github.io/adk-docs/context/caching/
- https://google.github.io/adk-docs/context/compaction/
- https://medium.com/google-cloud/the-adk-prompting-pattern-static-vs-turn-instructions-7a1e5b25eeef
"""

from google.adk.app import App
from google.adk.context import ContextCacheConfig, ContextCompactionConfig
from .coordinator import root_agent

# Context Caching Configuration
# Minimum: 1024 tokens for Gemini 2.5 Flash, 4096 for Pro
cache_config = ContextCacheConfig(
    min_tokens=2048,  # Only cache if prompt >= 2K tokens (saves on small requests)
    ttl_seconds=3600,  # Cache for 1 hour (good for testing workflow)
    cache_intervals=10,  # Reuse cache for up to 10 invocations
)

# Context Compaction Configuration
# Summarizes old conversation history to prevent token explosion
compaction_config = ContextCompactionConfig(
    compaction_interval=3,  # Compact every 3 agent turns
    overlap_size=1,  # Keep 1 turn of overlap for continuity
    enabled=True,
)

# Create ADK App with optimization
app = App(
    agent=root_agent,
    context_cache_config=cache_config,
    context_compaction_config=compaction_config,
)

__all__ = ["app"]
