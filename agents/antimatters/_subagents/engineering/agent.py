"""
Engineer Agent: Parallel docking with Experiment Matrix artifact.

Inspired by Schrödinger LiveDesign LiveReport:
- Real-time matrix of ligands × properties
- Parallel processing with independent status tracking
- Each ligand runs asynchronously

Uses ADK ParallelAgent for concurrent execution.
"""

from datetime import datetime
from typing import Dict, List, Any
from google.adk.agents import LlmAgent, ParallelAgent
from core.mcp_servers.toolsets.docking import docking_tools
from core.agents.config import (
    MODELS,
    THRESHOLDS,
    PROMPTS,
    LIGANDS,
    PROTEINS,
    get_ligand_by_name,
    get_protein_by_ped_id,
)
from core.agents.base import (
    create_artifact,
    read_artifact,
    update_artifact,
    ArtifactType,
    create_experiment_matrix,
    update_ligand_status,
)


# =============================================================================
# Experiment Matrix Tool Functions
# =============================================================================

def create_experiment(
    name: str,
    protein_pdb_path: str,
    ligands: List[Dict[str, str]],
    n_clusters: int = 20,
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

3. DOCK EACH LIGAND (parallel updates)
   For each ligand:
   a) update_ligand_result(status="running")
   b) prepare_ligand(smiles) → SAVE aromatic_rings, h_bond_donors, charged_atoms
   c) dock_ensemble(...) → Get best_cluster energy
   d) analyze_interactions(...) → Get interaction types
   e) update_ligand_result(status="completed", energy=...,
      ligand_properties={aromatic_rings, h_bond_donors, charged_atoms},
      cluster_population=populations[best_cluster])

   CRITICAL: Pass ligand_properties and cluster_population to enable SAR discovery!

4. SUMMARIZE
   - Call get_experiment_table() for final LiveReport view
   - Report best ligand and ranking
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
After completion, provide:
1. Final Experiment Matrix table (get_experiment_table)
2. Ranking by binding energy
3. Best interactions found
4. Recommendation for next steps
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
# Main Engineer Agent
# =============================================================================

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
    n_clusters: int = 20,
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
    # Main agent
    "engineer_agent",

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
