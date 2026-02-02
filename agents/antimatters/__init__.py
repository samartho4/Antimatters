"""Antimatters Agent Module - Main entry point."""
import sys
from pathlib import Path

# Add project root to path so 'core' module can be imported
# This is needed because subagents use 'from core.mcp_servers...' imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from . import agent

__all__ = ["agent"]
