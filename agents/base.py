"""
Antimatters Agent Base Components
=================================

Artifact System inspired by:
- Antigravity: Task List, Implementation Plan, Walkthrough
- Schrödinger LiveDesign: LiveReport (molecule × property matrix)
- Benchling: Recipe, Workflow, Results

Antimatters Artifacts:
1. PROTOCOL - Research blueprint (PED ID, literature, compound properties)
2. EXPERIMENT_MATRIX - LiveReport-style ligand-conformation docking matrix
3. DISCOVERY_REPORT - Final analysis with 3D visualizations and insights

These artifacts enable:
- Transparency at every step
- Multimodal validation (figures, structures, interactions)
- Async user feedback (Google Docs-style comments)
- Knowledge graph integration for evolution
"""

import os
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum


# =============================================================================
# Configuration
# =============================================================================

CORE_ROOT = Path(__file__).parent.parent
# Use ADK standard location for web interface artifact visibility
ARTIFACTS_DIR = Path(__file__).parent / ".adk" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class ArtifactType(Enum):
    """Antimatters artifact types aligned with scientific workflow."""

    # Research Phase
    PROTOCOL = "protocol"  # Research blueprint with literature validation

    # Engineering Phase
    EXPERIMENT_MATRIX = "experiment_matrix"  # LiveReport-style docking results
    INTERACTION_MAP = "interaction_map"  # Protein-ligand contacts
    STRUCTURE_3D = "structure_3d"  # 3D molecular visualization

    # Evolution Phase
    DISCOVERY_REPORT = "discovery_report"  # Final analysis with insights
    EVOLUTION_TRACE = "evolution_trace"  # Optimization history

    # System
    TASK_LIST = "task_list"  # Sequence of steps
    VALIDATION_RESULT = "validation_result"  # Cross-validation report


@dataclass
class ArtifactMetadata:
    """Metadata for tracking artifact lifecycle."""
    id: str
    type: ArtifactType
    name: str
    created_at: str
    updated_at: str
    version: int = 1
    parent_id: Optional[str] = None  # For linked artifacts
    tags: List[str] = field(default_factory=list)
    comments: List[Dict] = field(default_factory=list)  # Google Docs-style


# =============================================================================
# Protocol Artifact (Research Agent)
# =============================================================================

@dataclass
class ProtocolArtifact:
    """
    Research Protocol - Blueprint for IDP ensemble docking.

    Inspired by Benchling's Recipe concept:
    - Defines inputs (PED ID, ligands, binding site)
    - Tracks literature validation
    - Links to experimental evidence (figures, papers)

    Example:
    ```
    Protocol: Alpha-Synuclein Y125/Y133/Y136 Docking
    ├── Target: PED00006e001 (576 conformations)
    ├── Binding Site: C-terminus tyrosine cluster
    ├── Ligands: [Fasudil, Ligand-47, Ligand-23]
    ├── Literature: Dhar et al. 2025 (PMC12345678)
    └── Validation: ✅ Figure 3 confirms Y136 interaction
    ```
    """
    # Target Information
    ped_id: str
    protein_name: str
    n_conformations: int
    binding_site_residues: List[int]

    # Ligand Panel
    ligands: List[Dict[str, str]]  # [{name, smiles, source}]

    # Literature Validation
    primary_citation: Optional[str] = None
    pmcid: Optional[str] = None
    figure_urls: List[str] = field(default_factory=list)
    validation_status: str = "pending"  # pending, validated, failed
    validation_notes: str = ""
    confidence_score: float = 0.0

    # Metadata
    metadata: Optional[ArtifactMetadata] = None


# =============================================================================
# Experiment Matrix Artifact (Engineer Agent)
# =============================================================================

@dataclass
class LigandResult:
    """Single ligand docking result for matrix."""
    ligand_name: str
    smiles: str
    pdbqt_path: str
    status: str  # queued, running, completed, failed
    best_energy: Optional[float] = None
    best_cluster: Optional[int] = None
    best_residue: Optional[int] = None
    interaction_types: List[str] = field(default_factory=list)
    completion_time: Optional[str] = None


@dataclass
class ExperimentMatrixArtifact:
    """
    Experiment Matrix - LiveDesign-inspired docking results.

    Inspired by Schrödinger LiveReport:
    - Rows = Ligands
    - Columns = Properties (energy, cluster, interactions)
    - Real-time updates as docking progresses

    Example:
    ```
    ┌─────────────┬──────────┬─────────┬──────────┬────────────┐
    │ Ligand      │ Status   │ Energy  │ Cluster  │ Key Residue│
    ├─────────────┼──────────┼─────────┼──────────┼────────────┤
    │ Fasudil     │ ✅ Done  │ -7.2    │ 3        │ Y136       │
    │ Ligand-47   │ ⏳ Run   │ -8.1    │ 5        │ Y125       │
    │ Ligand-23   │ 🕐 Queue │ --      │ --       │ --         │
    └─────────────┴──────────┴─────────┴──────────┴────────────┘
    ```

    Supports parallel execution with independent progress tracking.
    """
    # Experiment Configuration
    experiment_id: str
    protein_pdb_path: str
    n_clusters: int
    representative_frames: List[int]
    cluster_populations: List[float]

    # Results Matrix (LiveReport)
    ligand_results: List[LigandResult] = field(default_factory=list)

    # Aggregate Statistics
    best_overall_energy: Optional[float] = None
    best_ligand: Optional[str] = None
    ensemble_weighted_avg: Optional[float] = None

    # Visualization Links
    cluster_plot_path: Optional[str] = None
    energy_heatmap_path: Optional[str] = None

    # Metadata
    metadata: Optional[ArtifactMetadata] = None

    def get_status_summary(self) -> Dict[str, int]:
        """Get count of ligands by status."""
        summary = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
        for result in self.ligand_results:
            summary[result.status] = summary.get(result.status, 0) + 1
        return summary

    def to_markdown_table(self) -> str:
        """Generate LiveReport-style markdown table."""
        rows = ["| Ligand | Status | Energy (kcal/mol) | Best Cluster | Key Residue |",
                "|--------|--------|-------------------|--------------|-------------|"]

        status_icons = {
            "queued": "🕐",
            "running": "⏳",
            "completed": "✅",
            "failed": "❌"
        }

        for r in self.ligand_results:
            icon = status_icons.get(r.status, "?")
            energy = f"{r.best_energy:.1f}" if r.best_energy else "--"
            cluster = str(r.best_cluster) if r.best_cluster is not None else "--"
            residue = f"Y{r.best_residue}" if r.best_residue else "--"
            rows.append(f"| {r.ligand_name} | {icon} {r.status} | {energy} | {cluster} | {residue} |")

        return "\n".join(rows)


# =============================================================================
# Discovery Report Artifact (Evolution Agent)
# =============================================================================

@dataclass
class DiscoveryReportArtifact:
    """
    Discovery Report - Final analysis with multimodal insights.

    Inspired by Antigravity's Walkthrough:
    - Summary of completed work
    - Key discoveries and rankings
    - 3D visualizations for validation
    - Links to knowledge graph entities

    Example:
    ```
    ## Discovery Report: Alpha-Synuclein Ligand Screen

    ### Executive Summary
    Screened 3 ligands against 576 conformations.
    Ligand-47 shows strongest binding (-8.4 kcal/mol) at Y125.

    ### Ranking
    1. Ligand-47: -8.4 kcal/mol (Drug-like ✅)
    2. Fasudil: -7.2 kcal/mol (Drug-like ✅)
    3. Ligand-23: -5.1 kcal/mol (Weak ⚠️)

    ### Key Interactions
    - Y125: H-bond with Ligand-47 (occupancy: 85%)
    - Y136: Aromatic stacking with Fasudil

    ### 3D Visualization
    [Interactive Mol* viewer embedded]

    ### Knowledge Graph Entities
    - Protein: AlphaSynuclein (→ links to UniProt)
    - Ligand: Fasudil (→ links to ChEMBL)
    - Interaction: Y136-Aromatic (→ links to BindingDB)
    ```
    """
    # Summary
    title: str
    executive_summary: str
    n_ligands_screened: int
    n_conformations: int

    # Rankings
    ligand_rankings: List[Dict[str, Any]]  # [{name, energy, rank, drug_like}]

    # Key Interactions
    key_interactions: List[Dict[str, Any]]  # [{residue, type, ligand, occupancy}]

    # 3D Visualization
    visualization_3d: Optional[Dict[str, str]] = None  # {pdb_path, ligand_path, viewer_config}

    # Evolution Trace
    optimization_history: List[Dict] = field(default_factory=list)

    # Knowledge Graph Links
    kg_entities: List[Dict[str, str]] = field(default_factory=list)  # [{type, name, external_id}]

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    metadata: Optional[ArtifactMetadata] = None


# =============================================================================
# Artifact CRUD Operations
# =============================================================================

def create_artifact(
    artifact_type: ArtifactType,
    name: str,
    content: Any,
    tags: List[str] = None,
    parent_id: str = None,
    tool_context: Any = None
) -> Dict[str, Any]:
    """
    Create a new artifact with metadata.

    Args:
        artifact_type: Type of artifact (PROTOCOL, EXPERIMENT_MATRIX, etc.)
        name: Human-readable name
        content: Artifact content (dataclass or dict)
        tags: Optional tags for search/filter
        parent_id: Optional parent artifact ID for linking
        tool_context: Optional ToolContext for session state sharing (ADK pattern)

    Returns:
        Complete artifact with ID and metadata
    """
    artifact_id = f"art_{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()

    metadata = ArtifactMetadata(
        id=artifact_id,
        type=artifact_type,
        name=name,
        created_at=now,
        updated_at=now,
        tags=tags or [],
        parent_id=parent_id
    )

    # Convert dataclass to dict if needed
    if hasattr(content, '__dataclass_fields__'):
        content_dict = asdict(content)
    else:
        content_dict = content

    artifact_data = {
        "id": artifact_id,
        "type": artifact_type.value,
        "name": name,
        "metadata": asdict(metadata),
        "content": content_dict
    }

    # Save to disk (for backward compatibility)
    artifact_path = ARTIFACTS_DIR / f"{artifact_id}.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact_data, f, indent=2, default=str)

    # Persist to database for UI visibility
    try:
        from core.api.services import ArtifactService, init_db
        init_db()  # Ensure schema exists
        artifact_service = ArtifactService()

        # Extract conversation_id and workspace_id from tool_context if available
        conversation_id = None
        workspace_id = "ws_core"  # Default workspace
        if tool_context:
            state_dict = None
            if hasattr(tool_context, 'state') and tool_context.state:
                state_dict = tool_context.state
            elif hasattr(tool_context, 'session') and hasattr(tool_context.session, 'state'):
                state_dict = tool_context.session.state

            if state_dict:
                conversation_id = state_dict.get('conversation_id')
                workspace_id = state_dict.get('workspace_id', 'ws_core')

        artifact_service.create(
            artifact_type=artifact_type.value,
            content=content_dict,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            title=name,
            metadata={"artifact_id": artifact_id, "tags": tags or []}
        )
    except Exception as e:
        # Database insert failed - artifact still saved to disk
        pass

    # Store in session state if tool_context provided (ADK session sharing)
    if tool_context is not None:
        # Handle both ToolContext (.state) and InvocationContext (.session.state)
        state_dict = None
        if hasattr(tool_context, 'state'):
            # ToolContext (from LlmAgent tools)
            if tool_context.state is None:
                tool_context.state = {}
            state_dict = tool_context.state
        elif hasattr(tool_context, 'session') and hasattr(tool_context.session, 'state'):
            # InvocationContext (from BaseAgent._run_async_impl)
            state_dict = tool_context.session.state

        if state_dict is not None:
            if "artifacts" not in state_dict:
                state_dict["artifacts"] = {}

            state_dict["artifacts"][artifact_id] = {
                "type": artifact_type.value,
                "name": name,
                "path": str(artifact_path),
                "created_at": now
            }

            # Store latest by type for easy discovery
            type_key = f"latest_{artifact_type.value}"
            state_dict[type_key] = artifact_id

    return artifact_data


def read_artifact(artifact_id: str, tool_context: Any = None) -> Optional[Dict[str, Any]]:
    """
    Read an artifact by ID.

    Args:
        artifact_id: The artifact ID
        tool_context: Optional ToolContext/InvocationContext for session state lookup (ADK pattern)

    Returns:
        Artifact data or None if not found
    """
    # Try session state first if tool_context provided
    if tool_context is not None:
        state_dict = None
        if hasattr(tool_context, 'state') and tool_context.state:
            state_dict = tool_context.state
        elif hasattr(tool_context, 'session') and hasattr(tool_context.session, 'state'):
            state_dict = tool_context.session.state

        if state_dict:
            artifacts = state_dict.get("artifacts", {})
            if artifact_id in artifacts:
                artifact_path = Path(artifacts[artifact_id]["path"])
                if artifact_path.exists():
                    with open(artifact_path, "r") as f:
                        return json.load(f)

    # Fallback to file system
    artifact_path = ARTIFACTS_DIR / f"{artifact_id}.json"

    if artifact_path.exists():
        with open(artifact_path, "r") as f:
            return json.load(f)

    # Search by partial ID
    for path in ARTIFACTS_DIR.glob("*.json"):
        if artifact_id in path.stem:
            with open(path, "r") as f:
                return json.load(f)

    return None


def update_artifact(artifact_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Update an existing artifact.

    Args:
        artifact_id: The artifact ID
        updates: Dictionary of updates to apply

    Returns:
        Updated artifact or None if not found
    """
    artifact = read_artifact(artifact_id)
    if not artifact:
        return None

    # Apply updates to content
    artifact["content"].update(updates)
    artifact["metadata"]["updated_at"] = datetime.now().isoformat()
    artifact["metadata"]["version"] = artifact["metadata"].get("version", 1) + 1

    # Save back
    artifact_path = ARTIFACTS_DIR / f"{artifact_id}.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    return artifact


def add_artifact_comment(
    artifact_id: str,
    comment: str,
    user_id: str = "scientist",
    selection: str = None
) -> Optional[Dict[str, Any]]:
    """
    Add a Google Docs-style comment to an artifact.

    Args:
        artifact_id: The artifact ID
        comment: Comment text
        user_id: User who made the comment
        selection: Optional text selection the comment refers to

    Returns:
        Updated artifact with new comment
    """
    artifact = read_artifact(artifact_id)
    if not artifact:
        return None

    new_comment = {
        "id": f"cmt_{uuid.uuid4().hex[:6]}",
        "user_id": user_id,
        "comment": comment,
        "selection": selection,
        "created_at": datetime.now().isoformat(),
        "resolved": False
    }

    if "comments" not in artifact["metadata"]:
        artifact["metadata"]["comments"] = []

    artifact["metadata"]["comments"].append(new_comment)
    artifact["metadata"]["updated_at"] = datetime.now().isoformat()

    # Save back
    artifact_path = ARTIFACTS_DIR / f"{artifact_id}.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    return artifact


# =============================================================================
# Legacy Support Functions (for backward compatibility)
# =============================================================================

def create_markdown_artifact(name: str, content: str, artifact_type: str = "task_list") -> Dict[str, Any]:
    """
    Create a markdown artifact (legacy function).

    Maps to new artifact types:
    - task_list → TASK_LIST
    - implementation_plan → PROTOCOL
    - walkthrough → DISCOVERY_REPORT
    - research_data → PROTOCOL
    - validation_result → VALIDATION_RESULT
    """
    type_mapping = {
        "task_list": ArtifactType.TASK_LIST,
        "implementation_plan": ArtifactType.PROTOCOL,
        "walkthrough": ArtifactType.DISCOVERY_REPORT,
        "research_data": ArtifactType.PROTOCOL,
        "validation_result": ArtifactType.VALIDATION_RESULT,
        "experiment_matrix": ArtifactType.EXPERIMENT_MATRIX,
    }

    mapped_type = type_mapping.get(artifact_type, ArtifactType.TASK_LIST)

    return create_artifact(
        artifact_type=mapped_type,
        name=name,
        content={"markdown": content}
    )


def create_validation_artifact(
    name: str,
    claims: List[Dict],
    confidence_score: float,
    warnings: List[str] = None
) -> Dict[str, Any]:
    """Create a validation result artifact."""
    return create_artifact(
        artifact_type=ArtifactType.VALIDATION_RESULT,
        name=name,
        content={
            "claims": claims,
            "confidence_score": confidence_score,
            "warnings": warnings or [],
            "validated_at": datetime.now().isoformat()
        }
    )


def create_research_summary_artifact(
    name: str,
    ped_id: str,
    ligands: List[Dict],
    literature_refs: List[Dict],
    validation_status: str,
    confidence: float
) -> Dict[str, Any]:
    """Create a research summary artifact (Protocol)."""
    protocol = ProtocolArtifact(
        ped_id=ped_id,
        protein_name=name,
        n_conformations=0,  # Will be updated
        binding_site_residues=[125, 133, 136],  # Default for alpha-synuclein
        ligands=ligands,
        validation_status=validation_status,
        confidence_score=confidence
    )

    return create_artifact(
        artifact_type=ArtifactType.PROTOCOL,
        name=name,
        content=asdict(protocol)
    )


# =============================================================================
# Experiment Matrix Operations (for parallel processing)
# =============================================================================

def create_experiment_matrix(
    experiment_name: str,
    protein_pdb_path: str,
    ligands: List[Dict[str, str]],
    n_clusters: int = 20,
    tool_context: Any = None
) -> Dict[str, Any]:
    """
    Create an experiment matrix for parallel ligand docking.

    Args:
        experiment_name: Name of the experiment
        protein_pdb_path: Path to protein PDB file
        ligands: List of {name, smiles} dicts
        n_clusters: Number of clusters for ensemble
        tool_context: ToolContext for session state sharing

    Returns:
        Experiment matrix artifact with all ligands queued
    """
    matrix = ExperimentMatrixArtifact(
        experiment_id=f"exp_{uuid.uuid4().hex[:8]}",
        protein_pdb_path=protein_pdb_path,
        n_clusters=n_clusters,
        representative_frames=[],  # Will be populated during clustering
        cluster_populations=[],
        ligand_results=[
            LigandResult(
                ligand_name=lig["name"],
                smiles=lig["smiles"],
                pdbqt_path="",
                status="queued"
            )
            for lig in ligands
        ]
    )

    return create_artifact(
        artifact_type=ArtifactType.EXPERIMENT_MATRIX,
        name=experiment_name,
        content=asdict(matrix),
        tags=["parallel", "docking", "experiment"],
        tool_context=tool_context
    )


def update_experiment_clustering(
    artifact_id: str,
    representative_frames: List[int],
    cluster_populations: List[float]
) -> Optional[Dict[str, Any]]:
    """
    Update experiment matrix with clustering results.

    Called after cluster_conformations() to populate the matrix with
    representative frames and population weights for all docking jobs.

    Args:
        artifact_id: Experiment Matrix artifact ID
        representative_frames: List of frame indices for each cluster
        cluster_populations: Population weight of each cluster (for SAR analysis)

    Returns:
        Updated artifact or None if not found
    """
    artifact = read_artifact(artifact_id)
    if not artifact:
        return None

    content = artifact["content"]
    content["representative_frames"] = representative_frames
    content["cluster_populations"] = cluster_populations

    # Update artifact
    return update_artifact(artifact_id, {"representative_frames": representative_frames, "cluster_populations": cluster_populations})


def update_ligand_status(
    artifact_id: str,
    ligand_name: str,
    status: str,
    energy: float = None,
    cluster: int = None,
    residue: int = None,
    interactions: List[str] = None,
    ligand_properties: Dict[str, Any] = None,
    cluster_population: float = None
) -> Optional[Dict[str, Any]]:
    """
    Update a single ligand's status in the experiment matrix.

    This enables real-time LiveReport-style updates as parallel
    docking jobs complete.

    IMPORTANT: Pass ligand_properties and cluster_population for SAR discovery.
    - ligand_properties: From prepare_ligand (aromatic_rings, h_bond_donors, charged_atoms)
    - cluster_population: From cluster_conformations for weighted analysis
    """
    artifact = read_artifact(artifact_id)
    if not artifact:
        return None

    content = artifact["content"]
    for result in content["ligand_results"]:
        if result["ligand_name"] == ligand_name:
            result["status"] = status
            if energy is not None:
                result["best_energy"] = energy
            if cluster is not None:
                result["best_cluster"] = cluster
            if residue is not None:
                result["best_residue"] = residue
            if interactions:
                result["interaction_types"] = interactions
            if ligand_properties:
                result["ligand_properties"] = ligand_properties
            if cluster_population is not None:
                result["cluster_population"] = cluster_population
            if status == "completed":
                result["completion_time"] = datetime.now().isoformat()
            break

    # Update aggregate stats
    completed = [r for r in content["ligand_results"] if r["status"] == "completed" and r.get("best_energy")]
    if completed:
        energies = [r["best_energy"] for r in completed]
        content["best_overall_energy"] = min(energies)
        best_result = min(completed, key=lambda r: r["best_energy"])
        content["best_ligand"] = best_result["ligand_name"]

    return update_artifact(artifact_id, content)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ArtifactType",

    # Dataclasses
    "ArtifactMetadata",
    "ProtocolArtifact",
    "LigandResult",
    "ExperimentMatrixArtifact",
    "DiscoveryReportArtifact",

    # CRUD Operations
    "create_artifact",
    "read_artifact",
    "update_artifact",
    "add_artifact_comment",

    # Experiment Matrix Operations
    "create_experiment_matrix",
    "update_experiment_clustering",
    "update_ligand_status",

    # Legacy Functions
    "create_markdown_artifact",
    "create_validation_artifact",
    "create_research_summary_artifact",

    # Experiment Matrix
    "create_experiment_matrix",
    "update_ligand_status",
]
