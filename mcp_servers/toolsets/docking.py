"""Docking MCP toolset (prepare, cluster, dock, analyze)."""
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from core.agents.config import PYTHON_CMD, MCP_SERVERS, TIMEOUTS


def create_docking_tools() -> McpToolset:
    """Factory to create fresh McpToolset instance.

    Use this for ParallelAgent workers to avoid shared session conflicts.
    Per GitHub issues #2196, #1267: shared MCP sessions in parallel tasks
    cause 'cancel scope in different task' errors during cleanup.
    """
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=PYTHON_CMD,
                args=[str(MCP_SERVERS["docking"])],
            ),
            timeout=TIMEOUTS.docking,
        ),
    )


# Shared instance for sequential use (backward compatibility)
docking_tools = create_docking_tools()
