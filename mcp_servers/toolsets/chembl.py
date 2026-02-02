"""ChEMBL MCP toolset (22 drug discovery tools)."""
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from core.agents.config import MCP_SERVERS, TIMEOUTS

chembl_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="node",
            args=[str(MCP_SERVERS["chembl"])],
        ),
        timeout=TIMEOUTS.chembl,
    ),
)
