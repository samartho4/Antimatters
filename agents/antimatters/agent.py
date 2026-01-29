"""Antimatters Root Agent - Multi-agent coordinator for IDP docking."""
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Import the root agent from local coordinator
from .coordinator import root_agent

# Export for ADK web
__all__ = ["root_agent"]
