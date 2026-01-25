"""PED (Protein Ensemble Database) MCP toolset."""
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from core.agents.config import PYTHON_CMD, PED_SERVER, PED_TIMEOUT

ped_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=PYTHON_CMD,
            args=[str(PED_SERVER)],
        ),
        timeout=PED_TIMEOUT,
    ),
)
