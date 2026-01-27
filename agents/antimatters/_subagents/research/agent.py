"""
Research Agent: Self-validating research specialist with Protocol artifact.

Uses configuration-driven approach and Gemini prompting best practices:
- XML-structured prompts
- Few-shot examples
- 3-dimension reasoning framework
- No hardcoded values
"""

from datetime import datetime
from typing import Any
from google.adk.agents import LlmAgent
from core.mcp_servers.toolsets.ped import ped_tools
from core.mcp_servers.toolsets.chembl import chembl_tools
from core.mcp_servers.toolsets.biocontext import biocontext_tools
from core.agents.config import (
    MODELS,
    THRESHOLDS,
    PROMPTS,
    PROTEINS,
    get_protein_by_ped_id,
)
from core.agents.base import (
    create_artifact,
    read_artifact,
    update_artifact,
    ArtifactType,
)


# =============================================================================
# Tool Functions (Config-driven)
# =============================================================================

def create_protocol(
    ped_id: str,
    protein_name: str,
    n_conformations: int,
    binding_site_residues: list,
    ligands: list,
    protein_pdb_path: str = None,
    ensemble_id: str = "e001",
    n_residues: int = None,
    primary_citation: str = None,
    pmcid: str = None,
    figure_urls: list = None,
    tool_context: Any = None
) -> dict:
    """
    Create a Protocol artifact for research documentation.

    Args:
        ped_id: PED database ID
        protein_name: Name of the protein
        n_conformations: Number of conformations in ensemble
        binding_site_residues: List of residue numbers for binding site
        ligands: List of ligand names to test
        protein_pdb_path: Path to PDB file (CRITICAL for engineer agent)
        ensemble_id: Ensemble ID (default: e001)
        n_residues: Total residues in protein
        primary_citation: Citation for protein structure
        pmcid: PubMed Central ID
        figure_urls: URLs to figures
        tool_context: ToolContext for session state sharing
    """
    # Try to get config for known proteins
    protein_config = get_protein_by_ped_id(ped_id)
    if protein_config and not binding_site_residues:
        binding_site_residues = protein_config.binding_site_residues

    content = {
        "ped_id": ped_id,
        "protein_name": protein_name,
        "protein_pdb_path": protein_pdb_path,
        "ensemble_id": ensemble_id,
        "n_conformations": n_conformations,
        "n_residues": n_residues,
        "binding_site_residues": binding_site_residues,
        "ligands": ligands,
        "primary_citation": primary_citation,
        "pmcid": pmcid,
        "figure_urls": figure_urls or [],
        "validation_status": "pending",
        "confidence_score": 0.0,
        "created_at": datetime.now().isoformat(),
    }

    artifact = create_artifact(
        artifact_type=ArtifactType.PROTOCOL,
        name=f"Protocol: {protein_name}",
        content=content,
        tags=["research", "protocol", ped_id],
        tool_context=tool_context
    )

    return {
        "success": True,
        "artifact_id": artifact["id"],
        "message": f"Created Protocol for {protein_name}",
    }


def validate_protocol(
    artifact_id: str,
    validation_notes: str,
    confidence_score: float,
) -> dict:
    """Update Protocol with validation results."""
    threshold = THRESHOLDS.min_confidence_for_handoff
    status = "validated" if confidence_score >= threshold else "pending"
    if confidence_score < 5.0:
        status = "failed"

    updates = {
        "validation_status": status,
        "validation_notes": validation_notes,
        "confidence_score": confidence_score,
        "validated_at": datetime.now().isoformat(),
    }

    artifact = update_artifact(artifact_id, updates)
    return {
        "success": bool(artifact),
        "validation_status": status,
        "ready_for_engineering": confidence_score >= threshold,
    }


def get_protocol(artifact_id: str) -> dict:
    """Read a Protocol artifact."""
    artifact = read_artifact(artifact_id)
    return {"success": bool(artifact), "artifact": artifact}


# =============================================================================
# Build Instruction from Templates
# =============================================================================

def build_research_instruction() -> str:
    """Build research agent instruction using templates."""

    role = PROMPTS.role_template.format(
        agent_name="Research Specialist",
        specialty="Literature validation and Protocol creation for IDP docking"
    )

    constraints = PROMPTS.constraints_template.format(
        current_date=datetime.now().strftime("%Y-%m-%d"),
        additional_constraints="Minimum confidence for handoff: " + str(THRESHOLDS.min_confidence_for_handoff)
    )

    reasoning = PROMPTS.reasoning_template

    # Few-shot example
    example = PROMPTS.example_template.format(
        input_example="Dock Fasudil to alpha-synuclein PED00006e001",
        reasoning_example="1) Need ensemble data from PED, 2) Need ligand SMILES, 3) Should validate against literature",
        action_example="fetch_ped_ensemble('PED00006e001'), then search_compounds('fasudil'), then bc_get_europepmc_articles('alpha-synuclein fasudil binding')",
        output_example="Protocol artifact with confidence 8.5/10"
    )

    # Known proteins context
    proteins_context = "\n".join([
        f"- {k}: {v.name} (PED: {v.ped_id}, binding site: {v.binding_site_residues})"
        for k, v in PROTEINS.items() if v.ped_id
    ])

    tools_doc = """
<tools>
**Data Acquisition:**
- fetch_ped_ensemble(ped_id) → Ensemble with n_conformations
- search_compounds(query) → ChEMBL compound data
- assess_drug_likeness(smiles) → Lipinski check

**Literature:**
- bc_get_europepmc_articles(query) → Paper metadata
- bc_get_europepmc_fulltext(pmcid) → Full text + figures
- bc_search_google_scholar_publications(query) → Academic search

**Artifacts:**
- create_protocol(...) → Create Protocol artifact
- validate_protocol(artifact_id, notes, score) → Update validation
- get_protocol(artifact_id) → Read Protocol
</tools>"""

    workflow = """
<workflow>
1. ACQUIRE: Fetch PED ensemble, search compounds
2. CREATE: Call create_protocol() with gathered data
3. VALIDATE: Search literature, check figures, score confidence
4. UPDATE: Call validate_protocol() with findings
5. HANDOFF: If confidence >= {threshold}, ready for Engineer
</workflow>""".format(threshold=THRESHOLDS.min_confidence_for_handoff)

    return f"""
{role}

{constraints}

<context>
Known proteins:
{proteins_context}
</context>

{reasoning}

{example}

{tools_doc}

{workflow}

<output_format>
After validation, provide:
- Artifact ID
- Validation status (validated/pending/failed)
- Confidence score
- Key evidence (papers, figures)
- Ready for engineering: YES/NO
</output_format>
"""


# =============================================================================
# Agent Definition
# =============================================================================

research_agent = LlmAgent(
    name="research_agent",
    model=MODELS.research,
    description="Creates validated Protocol artifacts for IDP docking experiments",
    instruction=build_research_instruction(),
    tools=[
        ped_tools,
        chembl_tools,
        biocontext_tools,
        create_protocol,
        validate_protocol,
        get_protocol,
    ],
    output_key="research_results",
)

__all__ = ["research_agent", "create_protocol", "validate_protocol", "get_protocol"]
