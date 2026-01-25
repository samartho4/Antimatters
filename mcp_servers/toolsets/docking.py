"""Docking MCP toolset (prepare, cluster, dock, analyze)."""
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from core.agents.config import PYTHON_CMD, DOCKING_SERVER, DOCKING_TIMEOUT

docking_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=PYTHON_CMD,
            args=[str(DOCKING_SERVER)],
        ),
        timeout=DOCKING_TIMEOUT,
    ),
)
