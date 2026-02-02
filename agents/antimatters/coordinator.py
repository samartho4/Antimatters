"""Antimatters - Multi-agent Scientific Research Platform

TWO OPERATING MODES:
====================
1. ANALYSIS MODE (full workflow): Research → Engineer → Evolution
   - Runs complete docking simulation with Vina
   - Gets real binding energies, SAR insights, UCB rankings
   - Creates: Protocol, ExperimentMatrix, DiscoveryReport, KG in Neo4j
   - Use for: NEW experiments, VALIDATING generated molecules

2. PLANNING MODE (fast LLM generation): Evolution agent only
   - REQUIRES prior Analysis (must have KG + SAR data)
   - Uses Knowledge Graph + SAR insights to guide Gemini 3
   - Generates 3D molecules (validated with RDKit)
   - Use for: Rapid iteration AFTER Analysis
    has run

Routing (from ADK best practices):
- "analyze", "dock", "experiment", "run workflow" → ANALYSIS (docking_workflow)
- "plan", "design", "generate molecules", "fast generation" → PLANNING (planning_agent)

THREE specialized agents:
1. Research Agent → Protocol artifact (literature validation)
2. Engineer Agent → ExperimentMatrix artifact (parallel docking)
3. Evolution Agent → DiscoveryReport + molecule generation

MCP Tools:
- docking: prepare_ligand, cluster_conformations, dock_ensemble, analyze_interactions
- ped: fetch_ped_ensemble
- chembl: search_compounds, similarity_search
- biocontext: literature search (EuropePMC, Google Scholar, bioRxiv)

References:
- ADK Multi-Agent: https://google.github.io/adk-docs/agents/multi-agents/
- Chem3DLLM: https://arxiv.org/html/2508.10696 (RCMT format)
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent 'agents' directory to path for shared imports
# This is standard ADK practice per: https://github.com/google/adk-python/discussions/3117
agents_dir = Path(__file__).parent.parent
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

# Load environment first
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Configure Gemini with retry BEFORE importing agents
# This applies exponential backoff to all Gemini API calls
import gemini_config  # noqa: F401

from google.adk.agents import LlmAgent, SequentialAgent  # SequentialAgent for guaranteed multi-step execution
from config import MODELS  # Import model configuration

from ._subagents.research import research_agent
from ._subagents.engineering import engineer_coordinator  # NEW: Custom BaseAgent
from ._subagents.evolution import evolution_agent
from ._subagents.evolution.agent import (
    generate_molecules_direct,  # Direct generation - NO approval needed (PLANNING MODE)
    propose_molecule_generation,  # HITL version for ANALYSIS MODE
    approve_molecule_generation,
    prepare_suggestions_for_docking,
    analyze_experiment_results,
    # Visual Analysis (Gemini 3 code_execution_with_images)
    generate_publication_figure,
    annotate_interaction_image,
    analyze_artifact_visualization,  # Agentic Vision for existing artifacts
    # De novo 3D visualization (NO prior docking needed)
    visualize_ligand_3d,
)

# Docking workflow: Guaranteed sequential execution (research → engineer → evolution)
docking_workflow = SequentialAgent(
    name="docking_workflow",
    description="Complete docking workflow: research → engineer_coordinator → evolution (sequential, guaranteed). Use for NEW docking experiments only.",
    sub_agents=[research_agent, engineer_coordinator, evolution_agent],
)

# Planning agent - PLANNING MODE for fast molecule generation (no docking)
# Dual-mode: SMILES (reliable) or RCMT (experimental, precise 3D)
# ADK best practice: Clear description for routing
from ._subagents.evolution.agent import get_sar_from_graph, get_binding_hotspots
# 3D Interactive Visualization (py3Dmol with zoom/inspect) - for planning_agent
from ._subagents.engineering.agent import visualize_docking_result

planning_agent = LlmAgent(
    name="planning_agent",
    model=MODELS.evolution,
    description="PLANNING MODE: Fast molecule generation using KG + SAR + Gemini. Also provides 3D visualization (py3Dmol) and visual analysis/annotation. Use for: 'plan', 'design', 'generate', 'show 3D', 'visualize', 'analyze image', 'annotate'. No docking.",
    instruction="""You are PLANNING MODE - fast molecule generation without docking simulation.

**WHEN TO USE:**
User says: "plan", "design molecules", "generate", "fast generation", "use KG", "planning mode"

**TWO GENERATION MODES:**
1. SMILES mode (default): Gemini → SMILES → RDKit 3D (more reliable)
   - generation_mode="smiles"
   - Output shows: generation_method="smiles_rdkit"
2. RCMT mode (experimental): Gemini → direct 3D coords (may have bond errors)
   - generation_mode="rcmt"
   - Output shows: generation_method="rcmt_direct"

**WORKFLOW (NO APPROVAL NEEDED):**
1. Get SAR context: Call get_sar_from_graph() and get_binding_hotspots() if KG exists
2. Generate DIRECTLY: Call generate_molecules_direct() with:
   - seed_smiles: Best binder SMILES (from state or user provided)
   - modification_type: "optimize" | "scaffold_hop" | "explore"
   - n_suggestions: 3-5
   - generation_mode: "smiles" (default) or "rcmt"
3. Results include: SMILES, 3D coordinates, drug-likeness, PDB blocks
4. Output CLEARLY shows:
   - generation_method: "smiles_rdkit" or "rcmt_direct"
   - generation_method_explanation: How 3D was created
5. For real energies: Tell user to run Analysis mode

**IMPORTANT - NO HITL:**
- Use generate_molecules_direct() - NO approval step
- Results are immediate
- Output clearly indicates generation method

**STATE (from prior Analysis if available):**
Check session state for these keys to see if Analysis has run:
- discovery_report_id
- latest_experiment_matrix
- docking_completed

**TOOLS:**
- generate_molecules_direct: DIRECT generation (NO approval) - use this!
- get_sar_from_graph: Query Neo4j for SAR patterns
- get_binding_hotspots: Find key residues from KG
- analyze_experiment_results: Get SAR from experiment artifact
- prepare_suggestions_for_docking: Format for Analysis mode

**3D INTERACTIVE VISUALIZATION (py3Dmol):**

TWO OPTIONS based on what data you have:

1. **visualize_ligand_3d(ligand_name, protein_name)** - NO PRIOR DOCKING NEEDED:
   - Use for: Config ligands (fasudil, ligand_47) or generated SMILES
   - Example: visualize_ligand_3d(ligand_name="fasudil")
   - Example: visualize_ligand_3d(ligand_smiles="CCO", protein_name="alpha_synuclein")
   - Creates: Interactive 3D viewer with protein + ligand
   - Shows: Binding site residues highlighted (yellow)
   - Returns: HTML artifact you can open in browser

2. **visualize_docking_result(experiment_artifact_id, ligand_name)** - AFTER DOCKING:
   - Use for: Viewing actual docked poses with interaction data
   - Requires: Run "analyze" workflow first to have experiment_artifact_id
   - Shows: H-bonds (blue), hydrophobic (green), aromatic (purple)

**VISUAL ANALYSIS (Gemini 3 Agentic Vision):**
Use these tools to analyze/annotate existing artifact images:
- analyze_artifact_visualization(artifact_filename, analysis_prompt, zoom_region):
  - Analyzes artifact images using Gemini 3's Think-Act-Observe loop
  - Can zoom, crop, annotate, and inspect molecular visualizations
  - Examples: "cluster_overlay_alpha_synuclein.png", "ligand_fasudil.png"
- generate_publication_figure(experiment_artifact_id, figure_type):
  - Creates publication-ready figures (sar_summary, energy_landscape, interaction_heatmap, ucb_ranking)
- annotate_interaction_image(image_path, interaction_data_json):
  - Adds arrows, boxes, labels to highlight binding sites and interactions
""",
    tools=[
        generate_molecules_direct,  # Direct - NO approval needed
        get_sar_from_graph,
        get_binding_hotspots,
        analyze_experiment_results,
        prepare_suggestions_for_docking,
        # 3D Interactive Visualization (py3Dmol)
        visualize_ligand_3d,  # De novo - NO prior docking needed
        visualize_docking_result,  # After docking - shows interactions
        # Visual Analysis (Gemini 3 code_execution_with_images)
        analyze_artifact_visualization,
        generate_publication_figure,
        annotate_interaction_image,
    ],
    output_key="planning_results",
)

root_agent = LlmAgent(
    name="antimatters_agent",
    model=MODELS.coordinator,  # Use configured Gemini 3 Pro
    description="Antimatters: IDP docking platform with Analysis and Planning modes",
    instruction="""You coordinate IDP docking with TWO MODES.

**MODE DETECTION:**

1. **ANALYSIS MODE** (full workflow with docking):
   Keywords: "analyze", "dock", "experiment", "run workflow", "binding energy", "validate"
   → Transfer to **docking_workflow**
   → Runs: Research → Engineer → Evolution
   → Gets: Real binding energies, SAR insights, KG

2. **PLANNING MODE** (fast generation, no docking):
   Keywords: "plan", "design", "generate molecules", "fast generation", "use KG"
   → Transfer to **planning_agent**
   → Uses: KG + SAR + Gemini for generation
   → Gets: Validated SMILES, 3D coords, drug-likeness (NO real energies)

**ROUTING LOGIC:**
- User wants real binding data → ANALYSIS (docking_workflow)
- User wants fast molecule ideas → PLANNING (planning_agent)
- User says "validate" generated molecules → ANALYSIS with those molecules

**STATE AWARENESS:**
Check session state for: docking_completed, discovery_report_id
These indicate if Analysis has run previously.

If Planning requested but no prior Analysis:
- Still works (de novo generation)
- Recommend running Analysis first for better SAR context

**ARTIFACTS:**
- Protocol (research): Target validation
- ExperimentMatrix (engineer): Docking results
- DiscoveryReport (evolution): SAR + Knowledge Graph
- Generated molecules (planning): SMILES + 3D coords""",
    tools=[],
    sub_agents=[docking_workflow, planning_agent],
    output_key="result",
)
