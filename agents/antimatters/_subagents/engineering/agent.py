"""
Engineer Agent: Parallel docking with Experiment Matrix artifact.

Inspired by Schrödinger LiveDesign LiveReport:
- Real-time matrix of ligands × properties
- Parallel processing with independent status tracking
- Each ligand runs asynchronously

Uses ADK ParallelAgent for concurrent execution.
"""

import secrets
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, AsyncGenerator

# Configure logging for docking progress
DOCKING_LOG_DIR = Path("/tmp/mcp_docking_workspace/logs")
DOCKING_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from google.adk.agents import LlmAgent, ParallelAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models.google_llm import Gemini
from google.genai import types

# Retry config for workers - handles transient 429 errors
# Per: https://google.github.io/adk-docs/agents/models/#error-code-429-resource_exhausted
WORKER_RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,           # Retry up to 5 times
    initial_delay=2,      # Start with 2s delay
    exp_base=2,           # Exponential backoff: 2s, 4s, 8s, 16s, 32s
    http_status_codes=[429, 500, 503, 504],  # Retry on rate limits and server errors
)
from core.mcp_servers.toolsets.docking import docking_tools, create_docking_tools
from core.agents.config import (
    MODELS,
    THRESHOLDS,
    PROMPTS,
    LIGANDS,
    PROTEINS,
    GeminiModel,
    get_ligand_by_name,
    get_protein_by_ped_id,
)
from google.adk.tools import ToolContext
from core.agents.base import (
    create_artifact,
    read_artifact,
    update_artifact,
    ArtifactType,
    create_experiment_matrix,
    update_experiment_clustering,
    update_ligand_status,
)


# =============================================================================
# State Writing Tool for Workers
# =============================================================================

def save_docking_result(
    run_id: str,
    ligand_name: str,
    status: str,
    best_energy: float = None,
    best_cluster: int = None,
    best_residue: int = None,
    interactions: list = None,
    aromatic_rings: int = None,
    hbond_donors: int = None,
    charged_atoms: int = None,
    cluster_population: float = None,
    error_message: str = None,
    tool_context: ToolContext = None
) -> dict:
    """
    Save docking result to session state for coordinator to collect.

    Workers MUST call this tool after docking to save results.
    Coordinator reads from state key: docking:{run_id}:result:{ligand_name}
    """
    result = {
        "status": status,
        "ligand_name": ligand_name,
        "best_energy": best_energy,
        "best_cluster": best_cluster,
        "best_residue": best_residue,
        "interactions": interactions or [],
        "ligand_properties": {
            "aromatic_rings": aromatic_rings,
            "hbond_donors": hbond_donors,
            "charged_atoms": charged_atoms
        },
        "cluster_population": cluster_population,
        "error_message": error_message
    }

    state_key = f"docking:{run_id}:result:{ligand_name}"

    if tool_context:
        tool_context.state[state_key] = result
        return {"success": True, "saved_to": state_key}

    return {"success": False, "error": "No tool_context provided"}


# =============================================================================
# Experiment Matrix Tool Functions
# =============================================================================

def create_experiment(
    name: str,
    protein_pdb_path: str,
    ligands: List[Dict[str, str]],
    n_clusters: int = 3,
    tool_context: Any = None
) -> dict:
    """
    Create an Experiment Matrix artifact for parallel docking.

    This creates a LiveReport-style matrix where:
    - Rows = Ligands (each with independent status)
    - Columns = Properties (energy, cluster, residue, interactions)

    Args:
        name: Experiment name
        protein_pdb_path: Path to protein PDB file
        ligands: List of {name, smiles} dicts
        n_clusters: Number of clusters for ensemble
        tool_context: ToolContext for session state sharing

    Returns:
        Experiment Matrix artifact with all ligands queued
    """
    artifact = create_experiment_matrix(
        experiment_name=name,
        protein_pdb_path=protein_pdb_path,
        ligands=ligands,
        n_clusters=n_clusters,
        tool_context=tool_context
    )

    return {
        "success": True,
        "artifact_id": artifact["id"],
        "experiment_id": artifact["content"]["experiment_id"],
        "n_ligands": len(ligands),
        "message": f"Created experiment with {len(ligands)} ligands queued",
    }


def update_ligand_result(
    artifact_id: str,
    ligand_name: str,
    status: str,
    best_energy: float = None,
    best_cluster: int = None,
    best_residue: int = None,
    interactions: List[str] = None,
    ligand_properties_json: str = None,
    cluster_population: float = None
) -> dict:
    """
    Update a ligand's status in the Experiment Matrix.

    This enables real-time LiveReport-style updates as each
    parallel docking job completes.

    IMPORTANT: Pass ligand_properties_json and cluster_population for SAR discovery.
    Evolution agent uses these to correlate ligand features → binding outcomes.

    Args:
        artifact_id: Experiment Matrix artifact ID
        ligand_name: Name of the ligand to update
        status: New status (queued, running, completed, failed)
        best_energy: Best binding energy (kcal/mol)
        best_cluster: Cluster ID with best energy
        best_residue: Residue with best interaction
        interactions: List of interaction types found
        ligand_properties_json: JSON string from prepare_ligand (aromatic_rings, h_bond_donors, charged_atoms)
        cluster_population: Population weight of best_cluster from cluster_conformations

    Returns:
        Updated artifact with new ligand status
    """
    # Parse JSON string to dict
    import json
    ligand_properties = None
    if ligand_properties_json:
        try:
            ligand_properties = json.loads(ligand_properties_json)
        except:
            ligand_properties = None

    artifact = update_ligand_status(
        artifact_id=artifact_id,
        ligand_name=ligand_name,
        status=status,
        energy=best_energy,
        cluster=best_cluster,
        residue=best_residue,
        interactions=interactions,
        ligand_properties=ligand_properties,
        cluster_population=cluster_population
    )

    return {
        "success": bool(artifact),
        "ligand_name": ligand_name,
        "status": status,
        "best_energy": best_energy,
        "ligand_properties": ligand_properties,
        "cluster_population": cluster_population,
    }


def get_experiment(artifact_id: str) -> dict:
    """Read an Experiment Matrix artifact."""
    artifact = read_artifact(artifact_id)
    if artifact:
        content = artifact["content"]
        # Calculate progress
        results = content.get("ligand_results", [])
        completed = sum(1 for r in results if r.get("status") == "completed")
        return {
            "success": True,
            "artifact": artifact,
            "progress": f"{completed}/{len(results)} ligands completed",
            "best_energy": content.get("best_overall_energy"),
            "best_ligand": content.get("best_ligand"),
        }
    return {"success": False, "error": "Experiment not found"}


def get_experiment_table(artifact_id: str) -> str:
    """Get LiveReport-style markdown table from Experiment Matrix."""
    artifact = read_artifact(artifact_id)
    if not artifact:
        return "Experiment not found"

    content = artifact["content"]
    results = content.get("ligand_results", [])

    # Build markdown table
    lines = [
        "| Ligand | Status | Energy (kcal/mol) | Cluster | Residue | Interactions |",
        "|--------|--------|-------------------|---------|---------|--------------|"
    ]

    status_icons = {"queued": "🕐", "running": "⏳", "completed": "✅", "failed": "❌"}

    for r in results:
        icon = status_icons.get(r.get("status", "queued"), "?")
        energy = f"{r['best_energy']:.1f}" if r.get("best_energy") else "--"
        cluster = str(r.get("best_cluster", "--"))
        residue = f"Y{r['best_residue']}" if r.get("best_residue") else "--"
        interactions = ", ".join(r.get("interaction_types", [])) or "--"
        lines.append(f"| {r['ligand_name']} | {icon} | {energy} | {cluster} | {residue} | {interactions} |")

    return "\n".join(lines)


def find_latest_protocol(tool_context: Any = None) -> dict:
    """
    Find the latest Protocol artifact using session state.

    This enables engineer_agent to discover protocol from research_agent
    without requiring explicit artifact_id passing.

    Follows ADK pattern: use tool_context.state for cross-agent discovery.

    Args:
        tool_context: ToolContext containing session state

    Returns:
        dict with success, artifact_id, protein_pdb_path, and protocol details
    """
    if tool_context is None or not hasattr(tool_context, 'state'):
        return {"success": False, "error": "No tool context available"}

    # Get latest protocol ID from session state
    protocol_id = tool_context.state.get("latest_protocol")

    if not protocol_id:
        return {"success": False, "error": "No protocol found in session state"}

    # Read the protocol artifact
    protocol = read_artifact(protocol_id, tool_context)

    if not protocol:
        return {"success": False, "error": f"Protocol {protocol_id} not found"}

    content = protocol.get("content", {})

    return {
        "success": True,
        "artifact_id": protocol_id,
        "protein_pdb_path": content.get("protein_pdb_path"),
        "ped_id": content.get("ped_id"),
        "protein_name": content.get("protein_name"),
        "ensemble_id": content.get("ensemble_id", "e001"),
        "n_conformations": content.get("n_conformations"),
        "binding_site_residues": content.get("binding_site_residues", []),
        "ligands": content.get("ligands", []),
    }


# =============================================================================
# Build Instruction from Templates
# =============================================================================

def build_engineer_instruction() -> str:
    """Build engineer agent instruction using templates."""

    role = PROMPTS.role_template.format(
        agent_name="Docking Engineer",
        specialty="Molecular docking, ensemble analysis, and interaction mapping"
    )

    constraints = PROMPTS.constraints_template.format(
        current_date=datetime.now().strftime("%Y-%m-%d"),
        additional_constraints=f"Drug-like threshold: {THRESHOLDS.drug_like} kcal/mol"
    )

    reasoning = PROMPTS.reasoning_template

    # Few-shot example
    example = PROMPTS.example_template.format(
        input_example="Dock 3 ligands to alpha-synuclein ensemble",
        reasoning_example="1) Create experiment matrix, 2) Cluster ensemble once, 3) Dock each ligand in parallel, 4) Update matrix after each completion",
        action_example="create_experiment() → cluster_conformations() → [parallel: prepare_ligand + dock_ensemble for each] → update_ligand_result()",
        output_example="Experiment Matrix with all 3 ligands completed, ranked by energy"
    )

    # Known ligands context
    ligands_context = "\n".join([
        f"- {k}: {v.name} (SMILES: {v.smiles[:30]}...)"
        for k, v in LIGANDS.items()
    ])

    tools_doc = """
<tools>
**Protocol Discovery:**
- find_latest_protocol() → protocol details including protein_pdb_path, ligands, binding sites
  Use this to discover the protocol created by research_agent via session state.

**Docking Tools (MCP):**
- prepare_ligand(smiles) → PDBQT file path, properties
- cluster_conformations(pdb_path, n_clusters) → representative_frames
- dock_ensemble(protein_pdb, ligand_pdbqt, representative_frames) → energies
- analyze_interactions(protein_pdb, ligand_pdbqt, frames) → contacts

**Experiment Matrix (LiveReport):**
- create_experiment(name, protein_pdb_path, ligands, n_clusters) → artifact
- update_ligand_result(artifact_id, ligand_name, status, ...) → update row
- get_experiment(artifact_id) → full matrix with progress
- get_experiment_table(artifact_id) → markdown table view
</tools>"""

    workflow = """
<workflow>
For parallel docking of N ligands:

1. CREATE EXPERIMENT
   - Call create_experiment() with all ligands queued
   - Save artifact_id for updates

2. CLUSTER PROTEIN (once)
   - Call cluster_conformations(pdb_path)
   - Save representative_frames AND cluster_populations for all docking jobs
   - IMPORTANT: cluster_populations needed for Evolution agent SAR analysis

3. DOCK EACH LIGAND (MUST COMPLETE ALL BEFORE FINISHING)
   For each ligand:
   a) update_ligand_result(status="running")
   b) prepare_ligand(smiles) → SAVE aromatic_rings, h_bond_donors, charged_atoms
   c) dock_ensemble(protein_pdb, ligand_pdbqt, representative_frames) → Get best_cluster + energies
   d) analyze_interactions(protein_pdb, ligand_pdbqt, best_frames) → Get interaction types
   e) update_ligand_result(status="completed",
      best_energy=...,
      best_cluster=...,
      ligand_properties={"aromatic_rings": X, "h_bond_donors": Y, "charged_atoms": Z},
      interactions=[...],
      cluster_population=populations[best_cluster])

   CRITICAL:
   - You MUST dock ALL ligands before finishing your turn
   - Each ligand MUST have status="completed" with all properties
   - Pass ligand_properties and cluster_population to enable SAR discovery
   - Evolution agent will fail if any ligand is not completed!

4. SUMMARIZE
   - Call get_experiment_table() for final LiveReport view
   - Report best ligand and ranking
   - Confirm ALL ligands are completed
   - Data is now ready for Evolution agent SAR analysis
</workflow>"""

    parallel_note = """
<parallel_execution>
The Experiment Matrix supports asynchronous updates:
- Each ligand has independent status (queued → running → completed)
- Update after EACH ligand completes (don't wait for all)
- User sees real-time progress in matrix view
- Like Schrödinger LiveDesign LiveReport
</parallel_execution>"""

    return f"""
{role}

{constraints}

<context>
Known ligands:
{ligands_context}
</context>

{reasoning}

{example}

{tools_doc}

{workflow}

{parallel_note}

<output_format>
DO NOT finish until ALL ligands have status="completed"!

After ALL ligands are docked and completed, provide:
1. Final Experiment Matrix table (get_experiment_table) - must show ALL ligands as "completed"
2. Ranking by binding energy
3. Best interactions found for each ligand
4. Confirmation that all docking is complete

If any ligand is still "queued" or "running", continue docking it before finishing.
</output_format>
"""


# =============================================================================
# Single Ligand Docking Agent (for parallel execution)
# =============================================================================

def create_ligand_agent(ligand_name: str, smiles: str) -> LlmAgent:
    """
    Create a specialized agent for docking a single ligand.

    Used by ParallelAgent for concurrent execution.
    """
    return LlmAgent(
        name=f"dock_{ligand_name.lower().replace('-', '_')}",
        model=MODELS.engineering,
        description=f"Docks {ligand_name} to protein ensemble",
        instruction=f"""<task>
Dock {ligand_name} (SMILES: {smiles}) to the protein ensemble.

Steps:
1. Read protein_pdb and representative_frames from state
2. Call prepare_ligand("{smiles}")
3. Call dock_ensemble with the prepared ligand
4. Call analyze_interactions for key contacts
5. Report: best_energy, best_cluster, key_residue, interaction_types
</task>

<output>
Return JSON:
{{
  "ligand_name": "{ligand_name}",
  "best_energy": <float>,
  "best_cluster": <int>,
  "best_residue": <int>,
  "interaction_types": ["h_bond", "aromatic", ...]
}}
</output>""",
        tools=[docking_tools],
        output_key=f"dock_{ligand_name.lower()}_result",
    )


def create_parallel_docking_agent(ligands: List[Dict[str, str]]) -> ParallelAgent:
    """
    Create a ParallelAgent that docks multiple ligands concurrently.

    Like Antigravity's agent manager - spawns independent experiments
    that user can observe separately.

    Args:
        ligands: List of {name, smiles} dicts

    Returns:
        ParallelAgent with sub-agents for each ligand
    """
    sub_agents = [
        create_ligand_agent(lig["name"], lig["smiles"])
        for lig in ligands
    ]

    return ParallelAgent(
        name="parallel_docking",
        description=f"Parallel docking of {len(ligands)} ligands",
        sub_agents=sub_agents,
    )


# =============================================================================
# Engineer Coordinator (Custom BaseAgent with Dynamic ParallelAgent)
# =============================================================================

class EngineerCoordinator(BaseAgent):
    """
    Custom BaseAgent that dynamically creates ParallelAgent for ligand docking.

    Guarantees ALL ligands complete docking before finishing.

    Phases:
    1. Setup: Get protocol, create experiment matrix, cluster protein (sequential)
    2. Parallel Docking: Dynamically create ParallelAgent with one worker per ligand
    3. Gather: Collect results from session.state and update matrix

    State Management:
    - Uses prefixed keys to avoid collisions: `docking:{run_id}:*`
    - Each worker writes to: `docking:{run_id}:result:{ligand_name}`
    - Shared data: `docking:{run_id}:protein_pdb`, `docking:{run_id}:frames`, etc.
    """

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Execute complete docking workflow with guaranteed parallel completion."""
        run_id = secrets.token_hex(4)

        try:
            # ============================================================
            # PHASE 1: SETUP (Sequential)
            # ============================================================

            # Get protocol from session state
            protocol_id = ctx.session.state.get("latest_protocol")
            if not protocol_id:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role=self.name,
                        parts=[types.Part(text="Error: No protocol found in session state. Research agent must run first.")]
                    )
                )
                return

            # Read protocol artifact
            protocol = read_artifact(protocol_id)
            if not protocol:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role=self.name,
                        parts=[types.Part(text=f"Error: Protocol {protocol_id} not found.")]
                    )
                )
                return

            protocol_content = protocol.get("content", {})
            protein_pdb_path = protocol_content.get("protein_pdb_path")
            ligands = protocol_content.get("ligands", [])
            protein_name = protocol_content.get("protein_name", "Unknown")

            if not protein_pdb_path or not ligands:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role=self.name,
                        parts=[types.Part(text="Error: Protocol missing protein_pdb_path or ligands.")]
                    )
                )
                return

            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"Starting docking workflow for {protein_name} with {len(ligands)} ligands...")]
                )
            )

            # Create experiment matrix
            # base.py now handles both ToolContext and InvocationContext
            artifact = create_experiment_matrix(
                experiment_name=f"{protein_name} Docking",
                protein_pdb_path=protein_pdb_path,
                ligands=ligands,
                n_clusters=3,
                tool_context=ctx  # InvocationContext - will use .session.state
            )
            artifact_id = artifact["id"]

            # Store shared data in session.state with run_id prefix
            ctx.session.state[f"docking:{run_id}:artifact_id"] = artifact_id
            ctx.session.state[f"docking:{run_id}:protein_pdb"] = protein_pdb_path
            ctx.session.state[f"docking:{run_id}:ligands"] = ligands

            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"Created experiment matrix: {artifact_id}")]
                )
            )

            # ============================================================
            # PHASE 1.5: CLUSTER CONFORMATIONS (ONCE, BEFORE WORKERS)
            # ============================================================
            # CRITICAL: Call MCP tool DIRECTLY without LlmAgent
            # Per GitHub #1738: LlmAgent loses context and loops infinitely
            # Solution: Use MCP Python SDK to call tool directly from BaseAgent
            # Ref: https://github.com/modelcontextprotocol/python-sdk

            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"Clustering protein conformations from {protein_pdb_path}...")]
                )
            )

            # Call MCP tool directly using Python SDK (no LLM needed)
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client
            from mcp import StdioServerParameters as MCPStdioServerParameters
            from core.agents.config import PYTHON_CMD, MCP_SERVERS

            representative_frames = []
            cluster_populations = []

            try:
                server_params = MCPStdioServerParameters(
                    command=PYTHON_CMD,
                    args=[str(MCP_SERVERS["docking"])],
                )

                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()

                        # Call cluster_conformations directly - ONE call, no LLM
                        result = await session.call_tool(
                            "cluster_conformations",
                            arguments={
                                "pdb_path": protein_pdb_path,
                                "n_clusters": 2
                            }
                        )

                        # Parse result
                        if result.content and len(result.content) > 0:
                            content = result.content[0]
                            if hasattr(content, 'text'):
                                data = json.loads(content.text)
                                if data.get('success'):
                                    representative_frames = data.get('representative_frames', [])
                                    cluster_populations = data.get('cluster_populations', [])

            except Exception as e:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role=self.name,
                        parts=[types.Part(text=f"MCP clustering error: {str(e)[:100]}")]
                    )
                )

            # Fallback if clustering failed
            if not representative_frames:
                representative_frames = [0, 1]  # Use first 2 frames
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role=self.name,
                        parts=[types.Part(text="Warning: Clustering failed, using fallback frames [0, 1]")]
                    )
                )
            else:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role=self.name,
                        parts=[types.Part(text=f"Clustering complete: {len(representative_frames)} representative frames: {representative_frames}")]
                    )
                )

            # Store in state for workers and downstream agents
            ctx.session.state[f"docking:{run_id}:frames"] = representative_frames
            ctx.session.state[f"docking:{run_id}:populations"] = cluster_populations

            # Format for worker instructions
            frames_str = str(representative_frames)

            # ============================================================
            # PHASE 2: PARALLEL DOCKING (DIRECT MCP - NO LLM)
            # ============================================================
            # Per GitHub #1738: LlmAgent workers loop infinitely
            # Solution: Call MCP tools directly for deterministic workflow
            # Each worker runs: prepare_ligand → dock_ensemble → analyze → save

            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"Starting parallel docking for {len(ligands)} ligands (direct MCP)...")]
                )
            )

            async def dock_ligand_direct(ligand_name: str, smiles: str) -> dict:
                """
                Dock a single ligand using direct MCP calls (no LLM).
                Deterministic 4-step workflow that cannot loop.
                """
                # Create log file to capture docking server stderr
                log_file_path = DOCKING_LOG_DIR / f"docking_{ligand_name}_{run_id[:8]}.log"

                try:
                    server_params = MCPStdioServerParameters(
                        command=PYTHON_CMD,
                        args=[str(MCP_SERVERS["docking"])],
                    )

                    logger.info(f"[{ligand_name}] Starting docking workflow")

                    # Open log file and pass to stdio_client for stderr capture
                    with open(log_file_path, 'w') as errlog:
                        async with stdio_client(server_params, errlog=errlog) as (read, write):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                logger.info(f"[{ligand_name}] MCP session initialized")

                                # Step 1: Prepare ligand
                                logger.info(f"[{ligand_name}] Step 1/4: Preparing ligand from SMILES")
                                prep_result = await session.call_tool(
                                    "prepare_ligand",
                                    arguments={"smiles": smiles}
                                )
                                prep_data = json.loads(prep_result.content[0].text)
                                if not prep_data.get('success'):
                                    logger.error(f"[{ligand_name}] Prepare failed: {prep_data.get('error')}")
                                    return {"status": "error", "ligand": ligand_name, "error": prep_data.get('error', 'prepare failed')}

                                pdbqt_path = prep_data['pdbqt_path']
                                ligand_props = {
                                    "aromatic_rings": prep_data.get('aromatic_rings', 0),
                                    "hbond_donors": prep_data.get('h_bond_donors', 0),
                                    "charged_atoms": prep_data.get('charged_atoms', 0),
                                }
                                logger.info(f"[{ligand_name}] Ligand prepared: {prep_data.get('aromatic_rings', 0)} aromatic rings")

                                # Step 2: Dock ensemble
                                # Per Robustelli et al. 2025 (J. Chem. Inf. Model.): α-synuclein binds at
                                # C-terminal fragment (residues 121-140). Docking all 140 residues is 7x slower.
                                C_TERMINAL_RESIDUES = list(range(121, 141))  # residues 121-140 inclusive
                                logger.info(f"[{ligand_name}] Step 2/4: Docking ensemble (C-term residues 121-140 x {len(representative_frames)} clusters)")
                                logger.info(f"[{ligand_name}] Docking logs: {log_file_path}")
                                dock_result = await session.call_tool(
                                    "dock_ensemble",
                                    arguments={
                                        "protein_pdb": protein_pdb_path,
                                        "ligand_pdbqt": pdbqt_path,
                                        "representative_frames": representative_frames,
                                        "cluster_populations": cluster_populations or None,
                                        "residue_range": C_TERMINAL_RESIDUES
                                    }
                                )
                                dock_data = json.loads(dock_result.content[0].text)
                                if not dock_data.get('success'):
                                    logger.error(f"[{ligand_name}] Docking failed: {dock_data.get('error')}")
                                    return {"status": "error", "ligand": ligand_name, "error": dock_data.get('error', 'docking failed')}

                                # Extract best energy from cluster_results
                                # Server returns: cluster_results=[{cluster_id, best_energy, best_residue}, ...]
                                cluster_results = dock_data.get('cluster_results', [])
                                best_energy = None
                                best_cluster = None
                                best_residue = None
                                for cr in cluster_results:
                                    energy = cr.get('best_energy')
                                    if energy is not None:
                                        if best_energy is None or energy < best_energy:
                                            best_energy = energy
                                            best_cluster = cr.get('cluster_id')
                                            best_residue = cr.get('best_residue')
                                logger.info(f"[{ligand_name}] Docking complete: best energy {best_energy} kcal/mol, cluster {best_cluster}, residue {best_residue}")

                                # Step 3: Analyze interactions
                                logger.info(f"[{ligand_name}] Step 3/4: Analyzing interactions")
                                analyze_result = await session.call_tool(
                                    "analyze_interactions",
                                    arguments={
                                        "protein_pdb": protein_pdb_path,
                                        "ligand_pdbqt": pdbqt_path,
                                        "frame_index": representative_frames[best_cluster] if best_cluster is not None else 0
                                    }
                                )
                                analyze_data = json.loads(analyze_result.content[0].text)
                                interactions = analyze_data.get('interactions', []) if analyze_data.get('success') else []
                                logger.info(f"[{ligand_name}] Found {len(interactions)} interactions")

                                # Step 4: Save result to state
                                logger.info(f"[{ligand_name}] Step 4/4: Saving results to state")
                                result_data = {
                                    "status": "success",
                                    "best_energy": best_energy,
                                    "best_cluster": best_cluster,
                                    "interactions": interactions,
                                    "ligand_properties": ligand_props,
                                }
                                ctx.session.state[f"docking:{run_id}:result:{ligand_name}"] = result_data

                                logger.info(f"[{ligand_name}] ✓ Workflow complete")
                                return {"status": "success", "ligand": ligand_name, "energy": best_energy}

                except Exception as e:
                    logger.error(f"[{ligand_name}] Error: {str(e)[:200]}")
                    return {"status": "error", "ligand": ligand_name, "error": str(e)[:200]}

            # Run all ligands in parallel with staggered starts
            STAGGER_DELAY = 1.0  # seconds between starts (lower since no LLM calls)

            tasks = []
            for i, lig in enumerate(ligands):
                if i > 0:
                    await asyncio.sleep(STAGGER_DELAY)
                tasks.append(asyncio.create_task(dock_ligand_direct(lig["name"], lig["smiles"])))

            # Gather results
            worker_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            worker_errors = []
            for i, result in enumerate(worker_results):
                ligand_name = ligands[i]["name"]
                if isinstance(result, Exception):
                    worker_errors.append(f"{ligand_name}: {str(result)[:100]}")
                elif isinstance(result, dict):
                    if result.get("status") == "error":
                        worker_errors.append(f"{ligand_name}: {result.get('error', 'unknown')}")
                    elif result.get("status") == "success":
                        yield Event(
                            author=self.name,
                            content=types.Content(
                                role=self.name,
                                parts=[types.Part(text=f"Docked {ligand_name}: {result.get('energy', 'N/A')} kcal/mol")]
                            )
                        )

            if worker_errors:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role=self.name,
                        parts=[types.Part(text=f"Some workers had errors: {'; '.join(worker_errors[:3])}")]
                    )
                )

            # ============================================================
            # PHASE 3: GATHER RESULTS
            # ============================================================

            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="Gathering results from workers...")]
                )
            )

            # Collect results from session.state and update matrix
            completed_count = 0
            for lig in ligands:
                result_key = f"docking:{run_id}:result:{lig['name']}"
                result = ctx.session.state.get(result_key)

                if result and isinstance(result, dict):
                    if result.get("status") == "error":
                        # Handle error case
                        update_ligand_status(
                            artifact_id=artifact_id,
                            ligand_name=lig["name"],
                            status="failed",
                            energy=None,
                            cluster=None,
                            residue=None,
                            interactions=[],
                            ligand_properties=None,
                            cluster_population=None
                        )
                    else:
                        # Success case
                        update_ligand_status(
                            artifact_id=artifact_id,
                            ligand_name=lig["name"],
                            status="completed",
                            energy=result.get("best_energy"),
                            cluster=result.get("best_cluster"),
                            residue=result.get("best_residue"),
                            interactions=result.get("interactions", []),
                            ligand_properties=result.get("ligand_properties"),
                            cluster_population=result.get("cluster_population")
                        )
                        completed_count += 1
                else:
                    # No result found - mark as failed
                    update_ligand_status(
                        artifact_id=artifact_id,
                        ligand_name=lig["name"],
                        status="failed"
                    )

            # Final summary
            error_note = f"\nNote: {len(worker_errors)} worker(s) had errors" if worker_errors else ""
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"""Docking workflow complete.

Completed: {completed_count}/{len(ligands)} ligands
Experiment Matrix: {artifact_id}{error_note}

Results saved to experiment matrix and ready for Evolution agent analysis.""")]
                ),
                actions=EventActions(
                    state_delta={
                        "latest_experiment_matrix": artifact_id,
                        "docking_completed": True,
                        "docking_run_id": run_id
                    }
                )
            )

        except Exception as e:
            # Error handling
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"Critical error in docking workflow: {str(e)}")]
                )
            )

    def _create_ligand_worker(self, ligand_name: str, smiles: str, run_id: str, protein_pdb: str,
                               representative_frames: str, model: str = None) -> LlmAgent:
        """
        Create a fresh worker agent for a single ligand.

        Worker uses save_docking_result tool to write results to session.state.
        Model can be specified to distribute workers across quota buckets.
        """
        worker_model = model or GeminiModel.FLASH_2_0.value
        return LlmAgent(
            name=f"dock_{ligand_name.lower().replace('-', '_').replace(' ', '_')}",
            # Use Gemini wrapper with retry options for 429 handling
            model=Gemini(model=worker_model, retry_options=WORKER_RETRY_CONFIG),
            description=f"Docks {ligand_name} to protein ensemble",
            instruction=f"""You are a docking worker for ligand: {ligand_name}

GIVEN DATA (already computed - DO NOT recompute):
- SMILES: {smiles}
- PROTEIN: {protein_pdb}
- REPRESENTATIVE_FRAMES: {representative_frames}
- RUN_ID: {run_id}

YOUR 4 STEPS (execute in order, ONCE each):

1. prepare_ligand("{smiles}")
   → Get pdbqt_path

2. dock_ensemble("{protein_pdb}", pdbqt_path, {representative_frames})
   → Get best_energy, best_cluster

3. analyze_interactions("{protein_pdb}", pdbqt_path, best_frame)
   → Get interactions

4. save_docking_result(run_id="{run_id}", ligand_name="{ligand_name}", status="success",
   best_energy=..., best_cluster=..., interactions=..., ...)

STOP after step 4. Do not repeat any step.
""",
            # Fresh MCP instance per worker to avoid cancel scope conflicts
            # See GitHub issues google/adk-python #2196, #1267
            tools=[create_docking_tools(), save_docking_result],
            output_key=f"dock_{ligand_name.lower().replace('-', '_')}_result"
        )


# =============================================================================
# Main Engineer Agent Instances
# =============================================================================

# NEW: Custom BaseAgent with guaranteed parallel completion
engineer_coordinator = EngineerCoordinator(name="engineer_coordinator")

# OLD: LlmAgent (kept for backward compatibility, but not used in main workflow)
engineer_agent = LlmAgent(
    name="engineer_agent",
    model=MODELS.engineering,
    description="Docking engineer with parallel execution and Experiment Matrix",
    instruction=build_engineer_instruction(),
    tools=[
        docking_tools,
        create_experiment,
        update_ligand_result,
        get_experiment,
        get_experiment_table,
        find_latest_protocol,
    ],
    output_key="docking_results",
)


# =============================================================================
# Factory for Dynamic Parallel Agent
# =============================================================================

def create_docking_experiment(
    experiment_name: str,
    protein_pdb_path: str,
    ligand_names: List[str] = None,
    custom_ligands: List[Dict[str, str]] = None,
    n_clusters: int = 3,
) -> Dict[str, Any]:
    """
    Create a complete docking experiment with parallel agent.

    This is the main entry point for the frontend to configure experiments.

    Args:
        experiment_name: Name for the experiment
        protein_pdb_path: Path to protein PDB
        ligand_names: List of ligand keys from config (e.g., ["fasudil", "ligand_47"])
        custom_ligands: List of custom {name, smiles} dicts
        n_clusters: Number of clusters

    Returns:
        Dict with experiment artifact and parallel agent
    """
    # Collect ligands
    ligands = []

    if ligand_names:
        for name in ligand_names:
            config = get_ligand_by_name(name)
            if config:
                ligands.append({"name": config.name, "smiles": config.smiles})

    if custom_ligands:
        ligands.extend(custom_ligands)

    if not ligands:
        return {"success": False, "error": "No ligands provided"}

    # Create experiment artifact
    artifact = create_experiment_matrix(
        experiment_name=experiment_name,
        protein_pdb_path=protein_pdb_path,
        ligands=ligands,
        n_clusters=n_clusters
    )

    # Create parallel agent
    parallel_agent = create_parallel_docking_agent(ligands)

    return {
        "success": True,
        "artifact_id": artifact["id"],
        "experiment_id": artifact["content"]["experiment_id"],
        "parallel_agent": parallel_agent,
        "n_ligands": len(ligands),
        "ligands": [l["name"] for l in ligands],
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main agents
    "engineer_agent",  # OLD: LlmAgent (backward compat)
    "engineer_coordinator",  # NEW: BaseAgent with guaranteed completion
    "EngineerCoordinator",  # Class export

    # Parallel execution
    "create_parallel_docking_agent",
    "create_ligand_agent",
    "create_docking_experiment",

    # Artifact tools
    "create_experiment",
    "update_ligand_result",
    "get_experiment",
    "get_experiment_table",
]
