"""Docking MCP toolset (prepare, cluster, dock, analyze)."""
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from core.agents.config import MCP_SERVERS, TIMEOUTS

# Docking needs conda Python (has vina installed)
DOCKING_PYTHON = "/opt/homebrew/Caskroom/miniforge/base/bin/python"


def create_docking_tools() -> McpToolset:
    """Factory to create fresh McpToolset instance for parallel workers."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=DOCKING_PYTHON,
                args=[str(MCP_SERVERS["docking"])],
            ),
            timeout=TIMEOUTS.docking,
        ),
    )


# Shared instance for sequential use
docking_tools = create_docking_tools()
