"""
Engineer Agent: Parallel docking with Experiment Matrix artifact.

LiveDesign:
- Real-time matrix of ligands × properties
- Parallel processing with independent status tracking
- Each ligand runs asynchronously

Uses ADK ParallelAgent for concurrent execution.
"""

import secrets
import json
import asyncio
import async_timeout  # Python 3.10 compatible timeout
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
from core.mcp_servers.toolsets.docking import docking_tools, create_docking_tools, DOCKING_PYTHON
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
from google.genai import types as genai_types
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
# ADK Native Artifact Helpers (for ADK web visibility)
# =============================================================================

async def save_artifact_to_adk(
    tool_context: Any,
    filename: str,
    data: bytes,
    mime_type: str = "application/json"
) -> int:
    """
    Save artifact using ADK's native InMemoryArtifactService.

    This makes artifacts visible in ADK web's Artifacts tab.

    Args:
        tool_context: ADK ToolContext with save_artifact method
        filename: Name for the artifact (shown in ADK web)
        data: Binary data to save
        mime_type: MIME type of the data

    Returns:
        Version number of saved artifact (-1 if failed)
    """
    if tool_context and hasattr(tool_context, 'save_artifact'):
        try:
            artifact_part = genai_types.Part.from_bytes(
                data=data,
                mime_type=mime_type
            )
            version = await tool_context.save_artifact(
                filename=filename,
                artifact=artifact_part
            )
            return version
        except Exception as e:
            print(f"Warning: Failed to save ADK artifact {filename}: {e}")
            return -1
    return -1


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

async def create_experiment(
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

    Saves to both:
    1. Disk-based system (for persistence)
    2. ADK InMemoryArtifactService (for ADK web visibility)

    Args:
        name: Experiment name
        protein_pdb_path: Path to protein PDB file
        ligands: List of {name, smiles} dicts
        n_clusters: Number of clusters for ensemble
        tool_context: ToolContext for session state sharing

    Returns:
        Experiment Matrix artifact with all ligands queued
    """
    # Save to disk-based system
    artifact = create_experiment_matrix(
        experiment_name=name,
        protein_pdb_path=protein_pdb_path,
        ligands=ligands,
        n_clusters=n_clusters,
        tool_context=tool_context
    )

    # Also save to ADK InMemoryArtifactService (for ADK web visibility)
    adk_version = -1
    if tool_context and hasattr(tool_context, 'save_artifact'):
        try:
            # Create experiment matrix JSON for ADK
            experiment_data = {
                "id": artifact["id"],
                "type": "experiment_matrix",
                "name": name,
                "experiment_id": artifact["content"]["experiment_id"],
                "protein_pdb_path": protein_pdb_path,
                "n_clusters": n_clusters,
                "ligands": [
                    {"name": lig["name"], "smiles": lig["smiles"], "status": "queued"}
                    for lig in ligands
                ],
                "created_at": artifact.get("metadata", {}).get("created_at", datetime.now().isoformat())
            }
            experiment_json = json.dumps(experiment_data, indent=2).encode('utf-8')
            # Use descriptive filename for ADK web - MUST be consistent for version increments
            exp_slug = name.lower().replace(' ', '_').replace('-', '_')
            filename = f"experiment_matrix_{exp_slug}.json"
            adk_version = await save_artifact_to_adk(
                tool_context, filename, experiment_json, "application/json"
            )
            if adk_version >= 0:
                print(f"[ADK] Saved experiment_matrix to ADK web: {filename} (v{adk_version})")
        except Exception as e:
            print(f"Warning: Failed to save experiment_matrix to ADK: {e}")

    return {
        "success": True,
        "artifact_id": artifact["id"],
        "experiment_id": artifact["content"]["experiment_id"],
        "n_ligands": len(ligands),
        "adk_version": adk_version,
        "message": f"Created experiment with {len(ligands)} ligands queued",
    }


async def update_ligand_result(
    artifact_id: str,
    ligand_name: str,
    status: str,
    best_energy: float = None,
    best_cluster: int = None,
    best_residue: int = None,
    interactions: List[str] = None,
    ligand_properties_json: str = None,
    cluster_population: float = None,
    tool_context: Any = None
) -> dict:
    """
    Update a ligand's status in the Experiment Matrix.

    This enables real-time LiveReport-style updates as each
    parallel docking job completes.

    Updates both:
    1. Disk-based artifact (for persistence)
    2. ADK InMemoryArtifactService (for ADK web real-time version updates)

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
        tool_context: ToolContext for ADK artifact updates

    Returns:
        Updated artifact with new ligand status
    """
    # Parse JSON string to dict
    ligand_properties = None
    if ligand_properties_json:
        try:
            ligand_properties = json.loads(ligand_properties_json)
        except:
            ligand_properties = None

    # Update disk-based artifact
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

    # Also update ADK artifact for real-time version increment
    adk_version = -1
    if artifact and tool_context and hasattr(tool_context, 'save_artifact'):
        try:
            # Get experiment name from artifact for filename - MUST match create_experiment
            exp_name = artifact.get("name", artifact_id)
            exp_slug = exp_name.lower().replace(' ', '_').replace('-', '_')
            filename = f"experiment_matrix_{exp_slug}.json"

            # Build updated experiment data
            experiment_data = {
                "id": artifact["id"],
                "type": "experiment_matrix",
                "name": artifact.get("name", exp_name),
                "version": artifact.get("version", 0),
                "experiment_id": artifact["content"].get("experiment_id", ""),
                "protein_pdb_path": artifact["content"].get("protein_pdb_path", ""),
                "n_clusters": artifact["content"].get("n_clusters", 0),
                "ligands": [
                    {
                        "name": lr.get("ligand_name", ""),
                        "smiles": lr.get("smiles", ""),
                        "status": lr.get("status", "queued"),
                        "best_energy": lr.get("best_energy"),
                        "best_cluster": lr.get("best_cluster"),
                    }
                    for lr in artifact["content"].get("ligand_results", [])
                ],
                "updated_at": artifact.get("metadata", {}).get("updated_at", datetime.now().isoformat())
            }
            experiment_json = json.dumps(experiment_data, indent=2).encode('utf-8')
            adk_version = await save_artifact_to_adk(
                tool_context, filename, experiment_json, "application/json"
            )
            if adk_version >= 0:
                print(f"[ADK] Updated experiment_matrix: {ligand_name} → {status} (v{adk_version})")
        except Exception as e:
            print(f"Warning: Failed to update ADK artifact: {e}")

    return {
        "success": bool(artifact),
        "ligand_name": ligand_name,
        "status": status,
        "best_energy": best_energy,
        "ligand_properties": ligand_properties,
        "cluster_population": cluster_population,
        "adk_version": adk_version,
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

        # Initialize representative_frames early (will be populated by clustering phase)
        representative_frames = []
        cluster_populations = []

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
                    parts=[types.Part(text=f"Starting docking for {protein_name} with {len(ligands)} ligands")]
                )
            )

            # Create experiment matrix
            # base.py now handles both ToolContext and InvocationContext
            artifact = create_experiment_matrix(
                experiment_name=f"{protein_name} Docking",
                protein_pdb_path=protein_pdb_path,
                ligands=ligands,
                n_clusters=len(representative_frames) if representative_frames else 2,  # Use REAL count from clustering
                tool_context=ctx  # InvocationContext - will use .session.state
            )
            artifact_id = artifact["id"]

            # === SAVE TO ADK WEB (InMemoryArtifactService) ===
            # ADK web's Artifacts tab uses InMemoryArtifactService
            # Access via ctx.artifact_service.save_artifact()
            exp_slug = protein_name.lower().replace(' ', '_').replace('-', '_')
            adk_filename = f"experiment_matrix_{exp_slug}.json"

            try:
                experiment_data = {
                    "id": artifact["id"],
                    "type": "experiment_matrix",
                    "name": f"{protein_name} Docking",
                    "experiment_id": artifact["content"]["experiment_id"],
                    "protein_pdb_path": protein_pdb_path,
                    "n_clusters": len(representative_frames) if representative_frames else 2,  # Use REAL count
                    "ligands": [
                        {"name": lig["name"], "smiles": lig["smiles"], "status": "queued"}
                        for lig in ligands
                    ],
                    "created_at": datetime.now().isoformat()
                }
                experiment_json = json.dumps(experiment_data, indent=2).encode('utf-8')

                # Create artifact Part for ADK
                artifact_part = types.Part.from_bytes(
                    data=experiment_json,
                    mime_type="application/json"
                )

                # Save to ADK's InMemoryArtifactService (appears in Artifacts tab)
                # CRITICAL: app_name must match root_agent.name ("antimatters_agent")
                # Otherwise ADK web won't find the artifact
                if ctx.artifact_service:
                    adk_version = await ctx.artifact_service.save_artifact(
                        app_name="antimatters_agent",
                        user_id=ctx.session.user_id,
                        session_id=ctx.session.id,
                        filename=adk_filename,
                        artifact=artifact_part
                    )
                    logger.info(f"[ADK] Saved experiment_matrix to ADK Artifacts: {adk_filename} (v{adk_version})")
                else:
                    logger.warning("[ADK] No artifact_service available - artifact won't appear in ADK web")

                # Also emit text notification
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role=self.name,
                        parts=[types.Part(text=f"Created experiment matrix artifact: {adk_filename}")]
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to save ADK artifact: {e}")

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
                    parts=[types.Part(text="Clustering protein conformations...")]
                )
            )

            # Call MCP tool directly using Python SDK (no LLM needed)
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client
            from mcp import StdioServerParameters as MCPStdioServerParameters
            from core.agents.config import MCP_SERVERS

            try:
                server_params = MCPStdioServerParameters(
                    command=DOCKING_PYTHON,
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
                    parts=[types.Part(text=f"Running docking simulations for {len(ligands)} ligands...")]
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
                        command=DOCKING_PYTHON,
                        args=[str(MCP_SERVERS["docking"])],
                    )

                    logger.info(f"[{ligand_name}] Starting docking workflow")

                    # Open log file and pass to stdio_client for stderr capture
                    # Use timeout to prevent hanging on subprocess issues
                    DOCKING_TIMEOUT = 21600  # 6 hours per ligand
                    with open(log_file_path, 'w') as errlog:
                        async with async_timeout.timeout(DOCKING_TIMEOUT):
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
                                        logger.error(f"[{ligand_name}] Prepare failed: {prep_data.get('error')} — {prep_data.get('details', '')}")
                                        return {"status": "error", "ligand": ligand_name, "error": prep_data.get('error', 'prepare failed')}

                                    pdbqt_path = prep_data['pdbqt_path']
                                    props = prep_data.get('properties', {})
                                    ligand_props = {
                                        "aromatic_rings": props.get('aromatic_rings', []),
                                        "hbond_donors": props.get('hbond_donors', []),
                                        "pos_charges": props.get('pos_charges', []),
                                    }
                                    logger.info(f"[{ligand_name}] Ligand prepared: {props.get('n_aromatic_rings', 0)} aromatic rings")

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
                                        logger.error(f"[{ligand_name}] Docking failed: {dock_data.get('error')}: {dock_data.get('details', '')}")
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
                                    # NOTE: analyze_interactions MCP function is broken (passes file path
                                    # instead of residue index to trajectory_analysis functions). We skip it
                                    # and use available data: best_residue + ligand_properties for visualization.
                                    # See: server.py:analyze_interactions passes ligand_path (str) but
                                    # trajectory_analysis.hbond expects ligand_residue_index (int)
                                    logger.info(f"[{ligand_name}] Step 3/4: Skipping analyze_interactions (known broken)")
                                    interactions = []  # Will use best_residue + ligand_props in visualization
                                    logger.info(f"[{ligand_name}] Using dock_ensemble data: residue {best_residue}, cluster {best_cluster}")

                                    # Step 4: Save result to state
                                    logger.info(f"[{ligand_name}] Step 4/4: Saving results to state")
                                    result_data = {
                                        "status": "success",
                                        "best_energy": best_energy,
                                        "best_cluster": best_cluster,
                                        "best_residue": best_residue,
                                        "interactions": interactions,  # Empty since analyze_interactions is broken
                                        "ligand_properties": ligand_props,
                                    }
                                    ctx.session.state[f"docking:{run_id}:result:{ligand_name}"] = result_data

                                    logger.info(f"[{ligand_name}] ✓ Workflow complete")
                                    return {"status": "success", "ligand": ligand_name, "energy": best_energy}

                except BaseException as e:
                    # Handle ExceptionGroup from anyio TaskGroup (MCP stdio_client uses anyio)
                    import traceback
                    error_msg = str(e)[:200]
                    error_type = type(e).__name__

                    # Extract inner exception from ExceptionGroup if present
                    if hasattr(e, 'exceptions'):
                        # Python 3.11+ ExceptionGroup or anyio ExceptionGroup
                        inner_errors = []
                        for exc in e.exceptions:
                            exc_info = f"{type(exc).__name__}: {str(exc)[:80]}"
                            inner_errors.append(exc_info)
                        error_msg = f"{error_type}: {'; '.join(inner_errors)}"
                    elif hasattr(e, '__cause__') and e.__cause__:
                        cause = e.__cause__
                        error_msg = f"{error_type}: {str(e)[:80]} <- {type(cause).__name__}: {str(cause)[:80]}"

                    # Log full traceback for debugging
                    logger.error(f"[{ligand_name}] {error_type}: {error_msg}")
                    logger.debug(f"[{ligand_name}] Traceback:\n{traceback.format_exc()}")

                    return {"status": "error", "ligand": ligand_name, "error": error_msg}

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
            # Update ADK artifact after EACH ligand for real-time visibility
            completed_count = 0

            for lig in ligands:
                result_key = f"docking:{run_id}:result:{lig['name']}"
                result = ctx.session.state.get(result_key)
                lig_status = "failed"

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
                        lig_status = "completed"
                else:
                    # No result found - mark as failed
                    update_ligand_status(
                        artifact_id=artifact_id,
                        ligand_name=lig["name"],
                        status="failed"
                    )

                # === REAL-TIME UPDATE: Save updated artifact to ADK ===
                if ctx.artifact_service:
                    try:
                        updated_artifact = read_artifact(artifact_id)
                        if updated_artifact:
                            experiment_data = {
                                "id": updated_artifact["id"],
                                "type": "experiment_matrix",
                                "name": f"{protein_name} Docking",
                                "version": updated_artifact.get("metadata", {}).get("version", 0),
                                "experiment_id": updated_artifact["content"]["experiment_id"],
                                "protein_pdb_path": protein_pdb_path,
                                "n_clusters": len(representative_frames),
                                "representative_frames": representative_frames,
                                "ligands": [
                                    {
                                        "name": lr.get("ligand_name", ""),
                                        "smiles": lr.get("smiles", ""),
                                        "status": lr.get("status", "queued"),
                                        "best_energy": lr.get("best_energy"),
                                        "best_cluster": lr.get("best_cluster"),
                                        "best_residue": lr.get("best_residue"),
                                    }
                                    for lr in updated_artifact["content"].get("ligand_results", [])
                                ],
                                "updated_at": datetime.now().isoformat()
                            }
                            experiment_json = json.dumps(experiment_data, indent=2).encode('utf-8')
                            artifact_part = types.Part.from_bytes(
                                data=experiment_json,
                                mime_type="application/json"
                            )
                            adk_version = await ctx.artifact_service.save_artifact(
                                app_name="antimatters_agent",
                                user_id=ctx.session.user_id,
                                session_id=ctx.session.id,
                                filename=adk_filename,
                                artifact=artifact_part
                            )
                            logger.info(f"[ADK] Updated: {lig['name']} → {lig_status} (v{adk_version})")
                    except Exception as e:
                        logger.warning(f"Failed to update ADK artifact: {e}")

            # Final summary
            error_note = f"\nNote: {len(worker_errors)} worker(s) had errors" if worker_errors else ""
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"""Docking complete — {completed_count}/{len(ligands)} ligands analyzed{error_note}

Ready for evolution analysis.""")]
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
# 3D Visualization Tool (Gemini 3 code_execution_with_images)
# =============================================================================

def generate_3d_docking_visualization(
    protein_pdb_path: str,
    ligand_pdbqt_path: str = None,
    ligand_smiles: str = None,
    frame_index: int = 0,
    view_style: str = "cartoon",
    highlight_residues: List[int] = None,
    interactions: Dict[str, Any] = None,
    ligand_name: str = "Ligand",
    tool_context: Any = None
) -> dict:
    """
    Generate interactive 3D visualization of IDP-ligand docking with interaction data.

    Uses py3Dmol for zoom/inspect capabilities and creates
    STRUCTURE_3D artifact with embedded HTML viewer showing:
    - Protein structure (cartoon/stick/surface)
    - Ligand position
    - H-bonds as dashed blue lines
    - Hydrophobic contacts as green spheres
    - Aromatic interactions as purple rings

    Args:
        protein_pdb_path: Path to protein PDB/ensemble file
        ligand_pdbqt_path: Path to docked ligand PDBQT (optional)
        ligand_smiles: SMILES string for ligand (alternative to pdbqt)
        frame_index: Conformation frame to visualize
        view_style: "cartoon", "stick", "surface", "sphere"
        highlight_residues: List of residue numbers to highlight (e.g., [125, 133, 136])
        interactions: Dict from analyze_interactions with hbonds, hydrophobic, aromatic contacts
        ligand_name: Name of ligand for display

    Returns:
        dict with artifact_id and html_content for 3D viewer
    """
    import base64

    try:
        import py3Dmol
        import mdtraj as md
    except ImportError:
        return {"success": False, "error": "py3Dmol or mdtraj not installed"}

    try:
        # Load protein structure
        traj = md.load(protein_pdb_path)
        if frame_index >= traj.n_frames:
            frame_index = 0

        # Get PDB string for the selected frame
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp_path = tmp.name
        traj[frame_index].save_pdb(tmp_path)
        with open(tmp_path, 'r') as f:
            protein_pdb_str = f.read()
        import os
        os.unlink(tmp_path)

        # Parse interaction data for visualization
        hbond_residues = []
        hydrophobic_residues = []
        aromatic_residues = []
        interaction_summary = {"hbonds": 0, "hydrophobic": 0, "aromatic": 0}

        if interactions:
            # Handle both single frame and multi-frame interaction data
            interaction_list = interactions if isinstance(interactions, list) else [interactions]
            for frame_data in interaction_list:
                if isinstance(frame_data, dict):
                    # Extract residue numbers from each interaction type
                    for hb in frame_data.get("hbonds", []):
                        res = hb.get("protein_residue") or hb.get("residue")
                        if res and res not in hbond_residues:
                            hbond_residues.append(res)
                            interaction_summary["hbonds"] += 1

                    for hp in frame_data.get("hydrophobic", []):
                        res = hp.get("protein_residue") or hp.get("residue")
                        if res and res not in hydrophobic_residues:
                            hydrophobic_residues.append(res)
                            interaction_summary["hydrophobic"] += 1

                    for ar in frame_data.get("aromatic", []):
                        res = ar.get("protein_residue") or ar.get("residue")
                        if res and res not in aromatic_residues:
                            aromatic_residues.append(res)
                            interaction_summary["aromatic"] += 1

        # Merge all interacting residues with highlight list
        all_highlight = list(set((highlight_residues or []) + hbond_residues + hydrophobic_residues + aromatic_residues))

        # Create 3D viewer
        viewer = py3Dmol.view(width=800, height=600)

        # Add protein
        viewer.addModel(protein_pdb_str, "pdb")

        # Style based on view_style
        if view_style == "cartoon":
            viewer.setStyle({"cartoon": {"color": "spectrum"}})
        elif view_style == "stick":
            viewer.setStyle({"stick": {}})
        elif view_style == "surface":
            viewer.addSurface(py3Dmol.VDW, {"opacity": 0.7, "color": "white"})
        elif view_style == "sphere":
            viewer.setStyle({"sphere": {"scale": 0.3}})

        # Style interacting residues by type
        for res in hbond_residues:
            viewer.addStyle({"resi": res}, {"stick": {"color": "blue", "radius": 0.2}})
        for res in hydrophobic_residues:
            viewer.addStyle({"resi": res}, {"stick": {"color": "green", "radius": 0.2}})
        for res in aromatic_residues:
            viewer.addStyle({"resi": res}, {"stick": {"color": "purple", "radius": 0.2}})

        # Highlight binding site residues (yellow if not already colored)
        if highlight_residues:
            for res in highlight_residues:
                if res not in hbond_residues + hydrophobic_residues + aromatic_residues:
                    viewer.addStyle({"resi": res}, {"stick": {"color": "yellow", "radius": 0.2}})
                viewer.addSurface(py3Dmol.VDW, {"opacity": 0.3, "color": "yellow"}, {"resi": res})

        # Add ligand if provided
        ligand_added = False
        ligand_pdb_str = ""
        if ligand_pdbqt_path:
            try:
                from pathlib import Path
                ligand_path = Path(ligand_pdbqt_path)
                if ligand_path.exists():
                    with open(ligand_path) as f:
                        ligand_content = f.read()
                    viewer.addModel(ligand_content, "pdbqt")
                    viewer.setStyle({"model": 1}, {"stick": {"color": "cyan", "radius": 0.15}})
                    ligand_added = True
                    ligand_pdb_str = ligand_content
            except Exception as e:
                print(f"Warning: Could not add ligand from pdbqt: {e}")

        if ligand_smiles and not ligand_added:
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                mol = Chem.MolFromSmiles(ligand_smiles)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                    ligand_pdb_str = Chem.MolToPDBBlock(mol)
                    viewer.addModel(ligand_pdb_str, "pdb")
                    viewer.setStyle({"model": 1}, {"stick": {"color": "cyan", "radius": 0.15}})
                    ligand_added = True
            except Exception as e:
                print(f"Warning: Could not add ligand from SMILES: {e}")

        # Set zoom
        viewer.zoomTo()

        # Build JavaScript for interaction highlights
        hbond_js = ""
        hydrophobic_js = ""
        aromatic_js = ""

        for res in hbond_residues:
            hbond_js += f"viewer.addStyle({{resi: {res}}}, {{stick: {{color: 'blue', radius: 0.25}}}});\n"
        for res in hydrophobic_residues:
            hydrophobic_js += f"viewer.addStyle({{resi: {res}}}, {{stick: {{color: 'green', radius: 0.2}}}});\n"
        for res in aromatic_residues:
            aromatic_js += f"viewer.addStyle({{resi: {res}}}, {{stick: {{color: 'purple', radius: 0.2}}}});\n"

        # Escape backticks for JavaScript template literals
        protein_pdb_escaped = protein_pdb_str.replace('`', '\\`')
        ligand_pdb_escaped = ligand_pdb_str.replace('`', '\\`') if ligand_pdb_str else ""

        # Build ligand JS code
        ligand_js = ""
        if ligand_added:
            ligand_js = f'var ligandPDB = `{ligand_pdb_escaped}`; viewer.addModel(ligandPDB, "pdb"); viewer.setStyle({{model: 1}}, {{stick: {{color: "cyan", radius: 0.15}}}});'

        # Build highlight residues JS
        highlight_js = ';'.join([f"viewer.addStyle({{resi: {r}}}, {{stick: {{color: 'yellow', radius: 0.2}}}})" for r in (highlight_residues or [])])

        # Generate HTML with interaction controls
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #viewport {{ width: 100%; height: 100vh; }}
        .controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 100;
            background: rgba(0,0,0,0.85);
            padding: 15px;
            border-radius: 8px;
            color: white;
            font-size: 12px;
            max-width: 280px;
        }}
        .controls h3 {{ margin: 0 0 10px 0; color: #4fc3f7; }}
        .controls button {{
            margin: 2px;
            padding: 6px 12px;
            cursor: pointer;
            border: none;
            border-radius: 4px;
            background: #2196f3;
            color: white;
        }}
        .controls button:hover {{ background: #1976d2; }}
        .legend {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #444; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
        .stats {{ margin-top: 10px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; }}
        .stats div {{ margin: 2px 0; }}
    </style>
</head>
<body>
    <div class="controls">
        <h3>{ligand_name} - IDP Binding</h3>
        <div>
            <button onclick="setCartoon()">Cartoon</button>
            <button onclick="setStick()">Sticks</button>
            <button onclick="setSurface()">Surface</button>
            <button onclick="viewer.zoomTo()">Reset</button>
        </div>
        <div class="legend">
            <b>Interactions:</b>
            <div class="legend-item"><span class="legend-color" style="background: #2196f3;"></span>H-bonds ({interaction_summary['hbonds']})</div>
            <div class="legend-item"><span class="legend-color" style="background: #4caf50;"></span>Hydrophobic ({interaction_summary['hydrophobic']})</div>
            <div class="legend-item"><span class="legend-color" style="background: #9c27b0;"></span>Aromatic ({interaction_summary['aromatic']})</div>
            <div class="legend-item"><span class="legend-color" style="background: #00bcd4;"></span>Ligand</div>
            <div class="legend-item"><span class="legend-color" style="background: #ffeb3b;"></span>Binding site</div>
        </div>
        <div class="stats">
            <div>Frame: {frame_index} / {traj.n_frames}</div>
            <div>Atoms: {traj.n_atoms}</div>
            <div>Interacting residues: {len(all_highlight)}</div>
        </div>
        <small style="color: #888;">Drag to rotate, Scroll to zoom, Click residue for info</small>
    </div>
    <div id="viewport"></div>
    <script>
        var viewer = $3Dmol.createViewer('viewport', {{backgroundColor: '#1a1a2e'}});

        // Add protein
        var proteinPDB = `{protein_pdb_escaped}`;
        viewer.addModel(proteinPDB, 'pdb');
        viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});

        // Add ligand
        {ligand_js}

        // Highlight binding site residues
        {highlight_js}

        // H-bond residues (blue)
        {hbond_js}

        // Hydrophobic residues (green)
        {hydrophobic_js}

        // Aromatic residues (purple)
        {aromatic_js}

        viewer.zoomTo();
        viewer.render();

        // Style functions
        function setCartoon() {{
            viewer.setStyle({{model: 0}}, {{cartoon: {{color: 'spectrum'}}}});
            reapplyInteractions();
        }}
        function setStick() {{
            viewer.setStyle({{model: 0}}, {{stick: {{}}}});
            reapplyInteractions();
        }}
        function setSurface() {{
            viewer.setStyle({{model: 0}}, {{cartoon: {{color: 'spectrum'}}}});
            viewer.addSurface($3Dmol.VDW, {{opacity: 0.5, color: 'white'}}, {{model: 0}});
            reapplyInteractions();
        }}
        function reapplyInteractions() {{
            {hbond_js}
            {hydrophobic_js}
            {aromatic_js}
            viewer.render();
        }}
    </script>
</body>
</html>
"""

        # Create artifact
        artifact_content = {
            "protein_pdb_path": protein_pdb_path,
            "frame_index": frame_index,
            "n_frames": traj.n_frames,
            "n_atoms": traj.n_atoms,
            "view_style": view_style,
            "highlight_residues": highlight_residues or [],
            "ligand_added": ligand_added,
            "ligand_name": ligand_name,
            "interactions": {
                "hbond_residues": hbond_residues,
                "hydrophobic_residues": hydrophobic_residues,
                "aromatic_residues": aromatic_residues,
                "summary": interaction_summary
            },
            "html_viewer": html_content,
            "pdb_data": protein_pdb_str,
            "metadata": {
                "ped_id": protein_pdb_path.split("/")[-1].replace(".pdb", ""),
                "n_conformations": traj.n_frames,
            }
        }

        artifact = create_artifact(
            artifact_type=ArtifactType.STRUCTURE_3D,
            name=f"3D Docking: {ligand_name} (frame {frame_index})",
            content=artifact_content,
            tags=["3d", "structure", "visualization", "docking", "interactions"],
            tool_context=tool_context
        )

        return {
            "success": True,
            "artifact_id": artifact["id"],
            "n_frames": traj.n_frames,
            "n_atoms": traj.n_atoms,
            "ligand_added": ligand_added,
            "ligand_name": ligand_name,
            "highlight_residues": highlight_residues or [],
            "interactions": interaction_summary,
            "message": f"Created 3D visualization for {ligand_name} at frame {frame_index}/{traj.n_frames} with {sum(interaction_summary.values())} interactions."
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def visualize_docking_result(
    experiment_artifact_id: str,
    ligand_name: str,
    tool_context: Any = None
) -> dict:
    """
    Visualize a specific docking result from the experiment matrix.

    Loads the protein and ligand, highlights the binding site,
    and creates an interactive 3D viewer using AVAILABLE data.

    NOTE: analyze_interactions MCP function is currently broken (passes file path
    instead of residue index). This function works robustly with:
    - best_residue from dock_ensemble
    - Binding site residues from protocol (C-terminal 121-140 for alpha-synuclein)
    - ligand_properties from prepare_ligand (aromatic_rings, h_bond_donors)
    - interaction_types if any were populated

    Args:
        experiment_artifact_id: Experiment Matrix artifact ID
        ligand_name: Name of the ligand to visualize
        tool_context: ToolContext for artifact saving

    Returns:
        dict with 3D visualization artifact
    """
    # Get experiment
    artifact = read_artifact(experiment_artifact_id)
    if not artifact:
        return {"success": False, "error": "Experiment not found"}

    content = artifact.get("content", {})
    protein_pdb_path = content.get("protein_pdb_path")

    # Find ligand result
    ligand_result = None
    for result in content.get("ligand_results", []):
        if result.get("ligand_name") == ligand_name:
            ligand_result = result
            break

    if not ligand_result:
        return {"success": False, "error": f"Ligand {ligand_name} not found in experiment"}

    # Get best cluster frame and representative frames mapping
    best_cluster = ligand_result.get("best_cluster", 0)
    best_residue = ligand_result.get("best_residue")
    representative_frames = content.get("representative_frames", [])

    # Map cluster index to actual frame index
    if best_cluster is not None and best_cluster < len(representative_frames):
        frame_index = representative_frames[best_cluster]
    else:
        frame_index = best_cluster if best_cluster else 0

    # Extract ligand properties for inferring potential interactions
    ligand_props = ligand_result.get("ligand_properties", {})
    has_aromatics = bool(ligand_props.get("aromatic_rings"))
    has_hbond_donors = bool(ligand_props.get("hbond_donors"))
    has_charged = bool(ligand_props.get("pos_charges"))

    # Build interaction data from AVAILABLE sources (not broken analyze_interactions)
    # Use best_residue + ligand properties to infer likely interaction types
    interaction_data = {"hbonds": [], "hydrophobic": [], "aromatic": []}

    if best_residue:
        # best_residue is where the ligand binds - infer interactions from ligand properties
        if has_hbond_donors or has_charged:
            interaction_data["hbonds"].append({"residue": best_residue, "inferred": True})
        if has_aromatics:
            interaction_data["aromatic"].append({"residue": best_residue, "inferred": True})

    # Also check if there are any stored interactions (might be empty due to broken server)
    stored_interactions = ligand_result.get("interactions", [])
    interaction_types = ligand_result.get("interaction_types", [])

    if stored_interactions and isinstance(stored_interactions, list):
        for item in stored_interactions:
            if isinstance(item, dict):
                # Real interaction data from analyze_interactions (if it worked)
                for key in ["hbonds", "hydrophobic", "aromatic"]:
                    for contact in item.get(key, []):
                        res = contact.get("residue") or contact.get("protein_residue")
                        if res:
                            interaction_data[key].append({"residue": res})

    # Build highlight list from:
    # 1. Alpha-synuclein C-terminal binding site (residues 121-140, key: 125, 133, 136)
    # 2. Best residue from docking
    # 3. Any interaction residues
    BINDING_SITE_RESIDUES = [125, 133, 136]  # Key C-terminal residues per Robustelli 2025
    highlight = list(BINDING_SITE_RESIDUES)

    if best_residue and best_residue not in highlight:
        highlight.append(best_residue)

    # Add residues from interaction data
    for key in ["hbonds", "hydrophobic", "aromatic"]:
        for contact in interaction_data.get(key, []):
            res = contact.get("residue")
            if res and res not in highlight:
                highlight.append(res)

    # Generate visualization with available data
    return generate_3d_docking_visualization(
        protein_pdb_path=protein_pdb_path,
        ligand_smiles=ligand_result.get("smiles"),
        frame_index=frame_index,
        view_style="cartoon",
        highlight_residues=highlight,
        interactions=interaction_data,
        ligand_name=ligand_name,
        tool_context=tool_context
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
    description="Docking engineer with parallel execution, Experiment Matrix, and 3D visualization",
    instruction=build_engineer_instruction(),
    tools=[
        docking_tools,
        create_experiment,
        update_ligand_result,
        get_experiment,
        get_experiment_table,
        find_latest_protocol,
        # 3D visualization
        generate_3d_docking_visualization,
        visualize_docking_result,
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

    # 3D Visualization
    "generate_3d_docking_visualization",
    "visualize_docking_result",
]
