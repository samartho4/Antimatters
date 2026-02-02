"""
Research Agent: Self-validating research specialist with Protocol artifact.

Inspired by:
- Antigravity: Progressive disclosure with task lists
- Schrödinger Maestro: Protein/ligand preparation workflow
- Real labs: Visual validation with structure images

Features:
- Task list for research workflow tracking
- Visual URLs for PED ensemble and ligand structures
- Literature validation with EuropePMC and Google Search
- Multimodal artifact with embedded visuals
"""

from datetime import datetime
from typing import Any, List, Dict
import json
import logging
import google.genai.types as genai_types
from google.adk.agents import LlmAgent
from core.mcp_servers.toolsets.ped import ped_tools

logger = logging.getLogger(__name__)
from core.mcp_servers.toolsets.chembl import chembl_tools
from core.mcp_servers.toolsets.biocontext import biocontext_tools
from core.agents.config import (
    MODELS,
    THRESHOLDS,
    PROMPTS,
    PROTEINS,
    LIGANDS,
    get_protein_by_ped_id,
    get_ligand_by_name,
)
from core.agents.base import (
    create_artifact,
    read_artifact,
    update_artifact,
    ArtifactType,
)


# =============================================================================
# Visual URL Generators
# =============================================================================

def get_ped_viewer_url(ped_id: str) -> str:
    """Get PED web viewer URL for ensemble visualization."""
    return f"https://proteinensemble.org/entries/{ped_id}"


def get_pubchem_image_url(cid: int) -> str:
    """Get PubChem 2D structure image URL."""
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"


def get_chembl_image_url(chembl_id: str) -> str:
    """Get ChEMBL structure image URL."""
    return f"https://www.ebi.ac.uk/chembl/api/data/image/{chembl_id}.svg"


def generate_ligand_image_base64(smiles: str) -> str:
    """Generate ligand 2D structure as base64 PNG using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import base64
        from io import BytesIO

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(200, 200))
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
    except Exception:
        pass
    return None


# =============================================================================
# Task List Functions (Antigravity-style)
# =============================================================================

async def create_research_task_list(
    target_protein: str,
    ligand_names: List[str],
    tool_context: Any = None
) -> dict:
    """
    Create an Antigravity-style task list for research workflow.

    Tasks are progressively completed as research proceeds.
    Saves via ADK InMemoryArtifactService for ADK web visibility.
    """
    tasks = [
        {"id": 1, "step": "Search PED database", "status": "pending", "details": f"Find ensemble for {target_protein}"},
        {"id": 2, "step": "Fetch ensemble", "status": "pending", "details": "Download PDB, get n_conformations"},
        {"id": 3, "step": "Prepare ligands", "status": "pending", "details": f"Validate SMILES for {len(ligand_names)} ligands"},
        {"id": 4, "step": "Search literature", "status": "pending", "details": "EuropePMC + Google Scholar"},
        {"id": 5, "step": "Visual validation", "status": "pending", "details": "Find structure figures"},
        {"id": 6, "step": "Create protocol", "status": "pending", "details": "Bundle data with visuals"},
        {"id": 7, "step": "Validate protocol", "status": "pending", "details": "Score confidence"},
    ]

    content = {
        "target_protein": target_protein,
        "ligand_names": ligand_names,
        "tasks": tasks,
        "created_at": datetime.now().isoformat(),
    }

    # Save to custom disk-based system
    artifact = create_artifact(
        artifact_type=ArtifactType.TASK_LIST,
        name=f"Research Tasks: {target_protein}",
        content=content,
        tags=["research", "tasks", target_protein.lower().replace(" ", "_")],
        tool_context=tool_context
    )

    # Save to ADK InMemoryArtifactService
    adk_artifact_saved = None
    if tool_context and hasattr(tool_context, 'save_artifact'):
        try:
            task_json = json.dumps(content, indent=2).encode('utf-8')
            task_filename = f"task_list_{target_protein.lower().replace(' ', '_')}.json"
            version = await save_artifact_to_adk(
                tool_context, task_filename, task_json, "application/json"
            )
            if version >= 0:
                adk_artifact_saved = task_filename
        except Exception as e:
            print(f"Warning: Failed to save ADK task list artifact: {e}")

    # Store in state for tracking
    if tool_context and hasattr(tool_context, 'state'):
        tool_context.state["research_task_list"] = artifact["id"]

    return {
        "success": True,
        "artifact_id": artifact["id"],
        "n_tasks": len(tasks),
        "message": f"Created task list with {len(tasks)} steps",
        "adk_artifact": adk_artifact_saved,
    }


async def update_task_status(
    artifact_id: str,
    task_id: int,
    status: str,
    result: str = None,
    tool_context: Any = None
) -> dict:
    """
    Update a task's status in the task list.

    CRITICAL: Re-saves to ADK with the SAME filename to increment version.
    Per ADK docs: "Each time you save an artifact with the same filename, a new version is created."
    """
    artifact = read_artifact(artifact_id)
    if not artifact:
        return {"success": False, "error": "Task list not found"}

    content = artifact.get("content", {})
    tasks = content.get("tasks", [])

    for task in tasks:
        if task["id"] == task_id:
            task["status"] = status
            if result:
                task["result"] = result
            task["updated_at"] = datetime.now().isoformat()
            break

    # Update disk-based artifact
    update_artifact(artifact_id, {"tasks": tasks})

    # CRITICAL: Also update ADK artifact with SAME filename to increment version
    adk_version = -1
    if tool_context and hasattr(tool_context, 'save_artifact'):
        try:
            # Get the target protein from content to reconstruct the filename
            target_protein = content.get("target_protein", "unknown")
            task_filename = f"task_list_{target_protein.lower().replace(' ', '_').replace('-', '_')}.json"

            # Rebuild the full task list content for ADK
            adk_content = {
                "id": artifact_id,
                "type": "task_list",
                "target_protein": target_protein,
                "ligand_names": content.get("ligand_names", []),
                "tasks": tasks,
                "updated_at": datetime.now().isoformat(),
            }
            task_json = json.dumps(adk_content, indent=2).encode('utf-8')

            # Save with SAME filename = version increment!
            adk_version = await save_artifact_to_adk(
                tool_context, task_filename, task_json, "application/json"
            )
            if adk_version >= 0:
                print(f"[ADK] Task {task_id} → {status} (v{adk_version})")
        except Exception as e:
            print(f"Warning: Failed to update ADK task list: {e}")

    return {
        "success": True,
        "task_id": task_id,
        "status": status,
        "adk_version": adk_version,
    }


# =============================================================================
# ADK Native Artifact Helpers
# =============================================================================

async def save_artifact_to_adk(
    tool_context: Any,
    filename: str,
    data: bytes,
    mime_type: str = "application/json"
) -> int:
    """
    Save artifact using ADK's native InMemoryArtifactService.

    Args:
        tool_context: ADK ToolContext with save_artifact method
        filename: Name for the artifact
        data: Binary data to save
        mime_type: MIME type of the data

    Returns:
        Version number of saved artifact
    """
    if tool_context and hasattr(tool_context, 'save_artifact'):
        artifact_part = genai_types.Part.from_bytes(
            data=data,
            mime_type=mime_type
        )
        version = await tool_context.save_artifact(
            filename=filename,
            artifact=artifact_part
        )
        return version
    return -1


async def save_image_artifact(
    tool_context: Any,
    image_bytes: bytes,
    filename: str
) -> int:
    """Save an image as an ADK artifact."""
    return await save_artifact_to_adk(
        tool_context=tool_context,
        filename=filename,
        data=image_bytes,
        mime_type="image/png"
    )


# =============================================================================
# Protocol Functions with Visuals
# =============================================================================

async def create_protocol(
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
    literature_results: list = None,
    tool_context: Any = None
) -> dict:
    """
    Create a Protocol artifact with embedded visuals.

    Saves artifacts via:
    1. ADK InMemoryArtifactService (for ADK web visibility)
    2. Custom disk-based system (for persistence)

    Args:
        ped_id: PED database ID
        protein_name: Name of the protein
        n_conformations: Number of conformations in ensemble
        binding_site_residues: List of residue numbers for binding site
        ligands: List of ligand dicts [{name, smiles}] or names
        protein_pdb_path: Path to PDB file (CRITICAL for engineer agent)
        ensemble_id: Ensemble ID (default: e001)
        n_residues: Total residues in protein
        primary_citation: Citation for protein structure
        pmcid: PubMed Central ID
        figure_urls: URLs to validation figures
        literature_results: Papers from EuropePMC
        tool_context: ToolContext for session state sharing
    """
    import re
    import base64

    # Try to get config for known proteins
    protein_config = get_protein_by_ped_id(ped_id)
    if protein_config and not binding_site_residues:
        binding_site_residues = protein_config.binding_site_residues

    # Parse ligands and generate images
    parsed_ligands = []
    ligand_images = []  # Store (name, image_bytes) for ADK artifacts

    for lig in ligands:
        if isinstance(lig, dict) and "name" in lig and "smiles" in lig:
            ligand_data = lig.copy()
            name = lig["name"]
            smiles = lig["smiles"]
        elif isinstance(lig, str):
            # Parse "Name (SMILES: ...)" format
            match = re.match(r"(.+?)\s*\(SMILES:\s*(.+)\)", lig)
            if match:
                name, smiles = match.group(1).strip(), match.group(2).strip()
                ligand_data = {"name": name, "smiles": smiles}
            else:
                # Try config lookup if just a name
                config = get_ligand_by_name(lig)
                if config:
                    name = config.name
                    smiles = config.smiles
                    ligand_data = {
                        "name": name,
                        "smiles": smiles,
                        "chembl_id": config.chembl_id,
                        "image_url": get_chembl_image_url(config.chembl_id) if config.chembl_id else None,
                    }
                else:
                    continue
        else:
            continue

        # Generate 2D structure image
        b64_img = generate_ligand_image_base64(smiles)
        if b64_img:
            ligand_data["image_base64"] = b64_img
            # Decode to bytes for ADK artifact
            ligand_images.append((name, base64.b64decode(b64_img)))

        parsed_ligands.append(ligand_data)

    # Build visual URLs
    visuals = {
        "ped_viewer_url": get_ped_viewer_url(ped_id),
        "ped_api_image": f"https://proteinensemble.org/api/v1/entries/{ped_id}/image",
    }

    content = {
        "ped_id": ped_id,
        "protein_name": protein_name,
        "protein_pdb_path": protein_pdb_path,
        "ensemble_id": ensemble_id,
        "n_conformations": n_conformations,
        "n_residues": n_residues,
        "binding_site_residues": binding_site_residues,
        "ligands": parsed_ligands,
        "visuals": visuals,
        "primary_citation": primary_citation,
        "pmcid": pmcid,
        "figure_urls": figure_urls or [],
        "literature_results": literature_results or [],
        "validation_status": "pending",
        "confidence_score": 0.0,
        "created_at": datetime.now().isoformat(),
    }

    # Save to custom disk-based system (for persistence)
    artifact = create_artifact(
        artifact_type=ArtifactType.PROTOCOL,
        name=f"Protocol: {protein_name}",
        content=content,
        tags=["research", "protocol", ped_id],
        tool_context=tool_context
    )

    # Save to ADK InMemoryArtifactService (for ADK web visibility)
    adk_artifacts_saved = []
    if tool_context and hasattr(tool_context, 'save_artifact'):
        try:
            # Save protocol as JSON
            protocol_json = json.dumps(content, indent=2).encode('utf-8')
            protocol_filename = f"protocol_{ped_id}.json"
            version = await save_artifact_to_adk(
                tool_context, protocol_filename, protocol_json, "application/json"
            )
            if version >= 0:
                adk_artifacts_saved.append(protocol_filename)

            # Save ligand images as separate artifacts
            for name, img_bytes in ligand_images:
                img_filename = f"ligand_{name.lower().replace(' ', '_')}.png"
                version = await save_image_artifact(tool_context, img_bytes, img_filename)
                if version >= 0:
                    adk_artifacts_saved.append(img_filename)

        except Exception as e:
            print(f"Warning: Failed to save ADK artifacts: {e}")

    # Store in session state for Engineer agent
    if tool_context and hasattr(tool_context, 'state'):
        tool_context.state["latest_protocol"] = artifact["id"]
        tool_context.state["protocol_content"] = content

    return {
        "success": True,
        "artifact_id": artifact["id"],
        "message": f"Created Protocol for {protein_name}",
        "visuals": visuals,
        "n_ligands": len(parsed_ligands),
        "adk_artifacts": adk_artifacts_saved,
    }


async def validate_protocol(
    artifact_id: str,
    validation_notes: str,
    confidence_score: float,
    literature_evidence: list = None,
    figure_validation: str = None,
    tool_context: Any = None,
) -> dict:
    """
    Update Protocol with validation results.

    CRITICAL: Re-saves to ADK with SAME filename to increment version.
    """
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

    if literature_evidence:
        updates["literature_evidence"] = literature_evidence
    if figure_validation:
        updates["figure_validation"] = figure_validation

    # Update disk-based artifact
    artifact = update_artifact(artifact_id, updates)

    # CRITICAL: Re-save to ADK with SAME filename to increment version
    adk_version = -1
    if artifact and tool_context and hasattr(tool_context, 'save_artifact'):
        try:
            content = artifact.get("content", {})
            ped_id = content.get("ped_id", "unknown")
            protocol_filename = f"protocol_{ped_id}.json"

            # Merge updates into content for ADK
            adk_content = {**content, **updates}
            protocol_json = json.dumps(adk_content, indent=2).encode('utf-8')

            # Save with SAME filename = version increment!
            adk_version = await save_artifact_to_adk(
                tool_context, protocol_filename, protocol_json, "application/json"
            )
            if adk_version >= 0:
                print(f"[ADK] Protocol validated: {status} (v{adk_version})")
        except Exception as e:
            print(f"Warning: Failed to update ADK protocol: {e}")

    return {
        "success": bool(artifact),
        "validation_status": status,
        "ready_for_engineering": confidence_score >= threshold,
        "adk_version": adk_version,
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
        input_example="Dock Fasudil to alpha-synuclein",
        reasoning_example="1) Create task list, 2) Search PED, 3) Fetch ensemble, 4) Get ligand SMILES, 5) Search literature, 6) Create protocol with visuals",
        action_example="create_research_task_list() → search_ped_protein() → fetch_ped_ensemble() → search_compounds() → bc_get_europepmc_articles() → bc_search_google_scholar_publications('[protein] binding') → create_protocol(literature_results=[...])",
        output_example="Protocol artifact with embedded ligand images and PED viewer URL"
    )

    # Known proteins context
    proteins_context = "\n".join([
        f"- {k}: {v.name} (PED: {v.ped_id}, binding site: {v.binding_site_residues})"
        for k, v in PROTEINS.items() if v.ped_id
    ])

    # Known ligands context
    ligands_context = "\n".join([
        f"- {k}: {v.name} (SMILES: {v.smiles[:30]}...)"
        for k, v in list(LIGANDS.items())[:5]
    ])

    tools_doc = """
<tools>
**Task Tracking (Antigravity-style):**
- create_research_task_list(target_protein, ligand_names) → Task list artifact
- update_task_status(artifact_id, task_id, status, result) → Update task

**PED Ensemble (3 tools):**
- search_ped_protein(protein_name) → Find PED IDs
- get_ped_entry_details(ped_id) → Ensemble info
- fetch_ped_ensemble(ped_id, ensemble_id) → Download PDB, get pdb_path
  CRITICAL: Must call this to get pdb_path for Engineer agent!

**ChEMBL Compounds (27 tools available):**
- search_compounds(query) → Find compounds by name
- get_compound_info(chembl_id) → Get SMILES, properties
- If ChEMBL fails, user-provided SMILES can be used directly

**Literature (EuropePMC + Google Scholar):**
- bc_get_europepmc_articles(query) → Paper metadata with PMCIDs
- bc_get_europepmc_fulltext(pmcid) → Full text for figure extraction
- bc_search_google_scholar_publications(query) → Google Scholar papers
- bc_get_biorxiv_preprint_details(doi) → bioRxiv preprint info

**Protocol Artifacts:**
- create_protocol(ped_id, protein_name, n_conformations, binding_site_residues,
    ligands, protein_pdb_path, figure_urls, literature_results) → Protocol
  - Ligands must be [{name, smiles}] format
  - Include figure_urls from Google Search for visual validation
  - Ligand 2D images are auto-generated from SMILES
- validate_protocol(artifact_id, notes, score, literature_evidence) → Update
- get_protocol(artifact_id) → Read Protocol
</tools>"""

    workflow = """
<workflow>
**Follow these steps for each research request:**

1. PLAN: create_research_task_list(target_protein, ligand_names)
   - Creates Antigravity-style task list for tracking

2. SEARCH PED: search_ped_protein(protein_name) or use known PED ID
   - Update task 1 status to "completed"

3. FETCH ENSEMBLE: fetch_ped_ensemble(ped_id, ensemble_id)
   - Get pdb_path, n_conformations
   - CRITICAL: Store pdb_path for Engineer agent!
   - Update task 2 status

4. PREPARE LIGANDS: search_compounds(name) or use provided SMILES
   - Verify SMILES are valid
   - Update task 3 status

5. LITERATURE: bc_get_europepmc_articles("[protein] [ligand] binding")
   - Get PMCIDs for key papers
   - Update task 4 status

6. VISUAL VALIDATION (OPTIONAL - skip if 503 error):
   - bc_get_string_network_image is OPTIONAL - skip if it fails
   - bc_get_europepmc_fulltext(pmcid) → Extract figure references
   - If any tool returns 503/overloaded, SKIP and continue
   - Update task 5 status (mark complete even if skipped)

7. CREATE PROTOCOL: create_protocol(... figure_urls=[...])
   - Include all gathered data
   - Ligand 2D images auto-embedded from SMILES (always works)
   - PED viewer URL auto-added
   - figure_urls can be empty [] if visual tools failed
   - Update task 6 status

8. VALIDATE: validate_protocol(artifact_id, notes, confidence_score)
   - Score based on literature + visual evidence
   - If score >= {threshold}: Ready for Engineer
   - Update task 7 status

</workflow>""".format(threshold=THRESHOLDS.min_confidence_for_handoff)

    return f"""
{role}

{constraints}

<context>
Known proteins:
{proteins_context}

Known ligands:
{ligands_context}
</context>

{reasoning}

{example}

{tools_doc}

{workflow}

<visual_urls>
Protocol automatically includes:
- PED viewer: https://proteinensemble.org/entries/[ped_id]
- Ligand 2D images: Auto-generated from SMILES (base64 embedded) ← ALWAYS WORKS
- ChEMBL images: https://www.ebi.ac.uk/chembl/api/data/image/[chembl_id].svg

OPTIONAL (skip if 503 error):
- bc_get_string_network_image → Skip if overloaded, not critical
- These are nice-to-have, NOT required for protocol creation
</visual_urls>

<output_format>
After validation, provide:
- Task List artifact ID (shows progress)
- Protocol artifact ID
- Validation status (validated/pending/failed)
- Confidence score
- Key visuals found (figure URLs)
- Ready for engineering: YES/NO
</output_format>
"""


# =============================================================================
# Agent Definition
# =============================================================================

research_agent = LlmAgent(
    name="research_agent",
    model=MODELS.research,
    description="Creates validated Protocol artifacts with embedded visuals for IDP docking experiments",
    instruction=build_research_instruction(),
    tools=[
        # Task tracking
        create_research_task_list,
        update_task_status,
        # Data acquisition
        ped_tools,
        chembl_tools,
        biocontext_tools,
        # Protocol management
        create_protocol,
        validate_protocol,
        get_protocol,
    ],
    output_key="research_results",
)

__all__ = [
    "research_agent",
    "create_protocol",
    "validate_protocol",
    "get_protocol",
    "create_research_task_list",
    "update_task_status",
]
