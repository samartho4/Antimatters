"""BioContext KB MCP toolset - Literature search tools.

Provides access to research paper databases:
- EuropePMC: Literature search with full text
- bioRxiv/medRxiv: Preprint search
- Google Scholar: Academic search

Uses tool_filter to expose only literature tools (avoids schema issues with Gemini).
"""
import shutil
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from core.agents.config import TIMEOUTS

# Find uvx dynamically
UVX_PATH = shutil.which("uvx") or "/Users/sam/Library/Python/3.9/bin/uvx"

# Literature + visualization tools (avoids Gemini schema validation errors)
LITERATURE_TOOLS = [
    "bc_get_europepmc_articles",           # Literature search
    "bc_get_europepmc_fulltext",           # Full text + figure refs
    "bc_get_recent_biorxiv_preprints",     # Recent preprints
    "bc_get_biorxiv_preprint_details",     # Preprint details
    "bc_search_google_scholar_publications", # Academic search
    "bc_get_string_network_image",         # Protein network visualization
]

biocontext_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=UVX_PATH,
            args=["biocontext_kb@latest"],
            env={"MCP_ENVIRONMENT": "DEVELOPMENT", "UV_PYTHON": "3.12"},
        ),
        timeout=TIMEOUTS.biocontext,
    ),
    tool_filter=LITERATURE_TOOLS,
)
