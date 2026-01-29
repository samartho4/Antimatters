"""
Evolution Agent: Scientific Knowledge Graph + SAR Discovery + Self-Improvement

Research Foundation:
=====================
1. MedGraphRAG (ACL 2025) - Triple-linked structure:
   - User Documents → Credible Sources → Controlled Vocabularies
   - U-Retrieval technique for hierarchical graph navigation
   - GitHub: SuperMedIntel/Medical-Graph-RAG

2. IRDiff (ICML 2024) - Retrieval-Augmented Molecular Generation:
   - PMINet for protein-molecule interaction embeddings
   - Retrieval of high-affinity ligand references
   - GitHub: YangLing0818/IRDiff

3. Drug Repurposing Knowledge Graph (DRKG):
   - 97,000 biomedical entities, 4.4M relationships
   - TransE embeddings for knowledge graph completion
   - LLM integration for explainable predictions

4. Neo4j GraphRAG (from KG_ADK patterns):
   - Entity extraction with propose → approve workflow
   - Jaro-Winkler similarity for entity resolution
   - CORRESPONDS_TO relationships linking graphs

Integration Points:
==================
- Backend: knowledge_service (types: insight, procedure, result, feedback, reference)
- Frontend: Evolution tab in Sidebar displaying knowledge items
- Database: SQLite via core/api/services.py
"""

import json
import uuid
import math
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext
from neo4j import GraphDatabase

# Neo4j connection setup
def get_neo4j_driver():
    """Get Neo4j driver with credentials from environment."""
    uri = os.getenv("NEO4J_URI", "neo4j+s://ccd5c5c1.databases.neo4j.io")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not password:
        raise ValueError("NEO4J_PASSWORD not set in environment")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    return driver, database

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
# Knowledge Types (aligned with KnowledgeService)
# =============================================================================

class KnowledgeType:
    """Knowledge types from services.py KnowledgeService."""
    INSIGHT = "insight"         # Extracted insight from analysis
    PROCEDURE = "procedure"     # Learned procedure or workflow
    RESULT = "result"           # Important result worth remembering
    FEEDBACK = "feedback"       # User feedback or correction
    REFERENCE = "reference"     # External reference (ChEMBL, PED, literature)


# =============================================================================
# Scientific Graph Entities (MedGraphRAG-style)
# =============================================================================

@dataclass
class ScientificEntity:
    """
    Scientific graph entity following MedGraphRAG triple-linked structure.

    Level 1: Experimental Data (docking results, conformations)
    Level 2: Reference Data (ChEMBL compounds, PED ensembles)
    Level 3: Controlled Vocabularies (UniProt, PubChem IDs)
    """
    entity_id: str
    entity_type: str  # Protein, Ligand, Residue, Conformation, Interaction
    level: int  # 1=experimental, 2=reference, 3=vocabulary
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    external_ids: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScientificRelationship:
    """
    Scientific graph relationship (triple structure).

    Examples:
    - (Ligand)-[DOCKS_TO]->(Protein)
    - (Ligand)-[INTERACTS_WITH]->(Residue)
    - (Experimental)-[CORRESPONDS_TO]->(Reference)
    """
    relationship_id: str
    subject_id: str
    predicate: str  # DOCKS_TO, INTERACTS_WITH, CORRESPONDS_TO, HAS_PROPERTY
    object_id: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SARInsight:
    """
    Structure-Activity Relationship insight.

    Discovered patterns that can be persisted as knowledge items.
    """
    insight_id: str
    pattern_type: str  # interaction_preference, residue_hotspot, conformational_selection
    title: str
    description: str
    evidence_ligands: List[str]
    confidence: float
    recommendation: str
    tags: List[str] = field(default_factory=list)


# =============================================================================
# Entity Resolution (Jaro-Winkler from KG_ADK)
# =============================================================================

def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaro-Winkler similarity between two strings.

    From kg_construction_2.md: apoc.text.jaroWinklerDistance
    Returns value in [0, 1] where 1 is exact match.
    """
    if s1 == s2:
        return 1.0

    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()

    if s1_lower == s2_lower:
        return 1.0

    len1, len2 = len(s1_lower), len(s2_lower)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            if s2_matches[j] or s1_lower[i] != s2_lower[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1_lower[i] != s2_lower[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions / 2) / matches) / 3

    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1_lower[i] == s2_lower[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


def resolve_to_reference(
    entity_name: str,
    entity_type: str,
    similarity_threshold: float = 0.85
) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Resolve experimental entity to reference database.

    Returns (reference_key, external_ids) if match found.
    Creates CORRESPONDS_TO relationship in the graph.
    """
    if entity_type == "Protein":
        for key, config in PROTEINS.items():
            similarity = jaro_winkler_similarity(entity_name, config.name)
            if similarity >= similarity_threshold:
                return key, {
                    "uniprot": config.uniprot_id,
                    "ped": config.ped_id,
                } if config.uniprot_id else {"ped": config.ped_id}

    elif entity_type == "Ligand":
        for key, config in LIGANDS.items():
            similarity = jaro_winkler_similarity(entity_name, config.name)
            if similarity >= similarity_threshold:
                return key, {
                    "chembl": config.chembl_id,
                    "smiles": config.smiles,
                } if config.chembl_id else {"smiles": config.smiles}

    return None, {}


# =============================================================================
# UCB Prioritization (rUCB from Chen et al. 2021)
# =============================================================================

def calculate_ucb_ranking(
    ligand_results: List[Dict],
    exploration_constant: float = 1.4
) -> List[Dict]:
    """
    Calculate Upper Confidence Bound rankings for ligand prioritization.

    Based on rUCB algorithm (Chen et al. 2021, PCCP):
    - UCB balances exploitation (best known) vs exploration (uncertainty)
    - For minimization: UCB = -mean + c * sqrt(log(N)/n)
    """
    rankings = []
    completed = [r for r in ligand_results if r.get("status") == "completed" and r.get("best_energy")]

    if not completed:
        return rankings

    total_samples = sum(len(r.get("cluster_results", [])) or 1 for r in completed)

    for result in completed:
        energy = result.get("best_energy", 0)
        n_samples = len(result.get("cluster_results", [])) or 1

        exploitation = -energy  # Negate so lower energy = higher score
        exploration = exploration_constant * math.sqrt(math.log(max(total_samples, 1)) / n_samples)
        ucb = exploitation + exploration

        rankings.append({
            "ligand_name": result.get("ligand_name"),
            "best_energy": energy,
            "n_samples": n_samples,
            "exploration_bonus": round(exploration, 3),
            "ucb_score": round(ucb, 3),
            "drug_like": energy <= THRESHOLDS.drug_like,
        })

    rankings.sort(key=lambda x: x["ucb_score"], reverse=True)
    return rankings


# =============================================================================
# SAR Pattern Discovery
# =============================================================================

def discover_sar_insights(ligand_results: List[Dict]) -> List[SARInsight]:
    """
    Discover Structure-Activity Relationship insights from docking results.

    These become knowledge items of type "insight" in the Evolution tab.

    Connects MCP tool outputs:
    - prepare_ligand: aromatic_rings, h_bond_donors, charged_atoms
    - analyze_interactions: h_bonds, hydrophobic, aromatic contacts
    """
    insights = []
    completed = [r for r in ligand_results if r.get("status") == "completed" and r.get("best_energy")]

    if len(completed) < 2:
        return insights

    # Sort by energy
    sorted_results = sorted(completed, key=lambda x: x.get("best_energy", 0))
    mid = len(sorted_results) // 2
    best_half = sorted_results[:mid] if mid > 0 else sorted_results
    worst_half = sorted_results[mid:] if mid > 0 else []

    # Insight 1: Interaction type preference
    best_interactions = set()
    for r in best_half:
        best_interactions.update(r.get("interaction_types", []))

    worst_interactions = set()
    for r in worst_half:
        worst_interactions.update(r.get("interaction_types", []))

    beneficial = best_interactions - worst_interactions
    if beneficial:
        avg_best = sum(r["best_energy"] for r in best_half) / len(best_half)
        insights.append(SARInsight(
            insight_id=str(uuid.uuid4())[:8],
            pattern_type="interaction_preference",
            title=f"Beneficial Interactions: {', '.join(beneficial)}",
            description=f"Top binders (avg {avg_best:.1f} kcal/mol) show {', '.join(beneficial)} interactions not seen in weak binders. Source: analyze_interactions MCP tool.",
            evidence_ligands=[r["ligand_name"] for r in best_half],
            confidence=min(0.9, len(best_half) / 5),
            recommendation=f"Design ligands that form {', '.join(beneficial)} interactions",
            tags=["sar", "interaction", "optimization", "mcp:analyze_interactions"]
        ))

    # Insight 2: Residue hotspot
    residue_scores = {}
    for r in completed:
        residue = r.get("best_residue")
        if residue:
            if residue not in residue_scores:
                residue_scores[residue] = []
            residue_scores[residue].append((r["ligand_name"], r["best_energy"]))

    for residue, ligand_energies in residue_scores.items():
        if len(ligand_energies) >= 2:
            avg = sum(e for _, e in ligand_energies) / len(ligand_energies)
            if avg <= THRESHOLDS.drug_like:
                insights.append(SARInsight(
                    insight_id=str(uuid.uuid4())[:8],
                    pattern_type="residue_hotspot",
                    title=f"Binding Hotspot: Y{residue}",
                    description=f"Tyrosine {residue} is a key binding determinant with avg energy {avg:.1f} kcal/mol. Source: dock_ensemble + analyze_interactions MCP tools.",
                    evidence_ligands=[name for name, _ in ligand_energies],
                    confidence=min(0.85, len(ligand_energies) / 5),
                    recommendation=f"Optimize interactions with Y{residue} as primary target",
                    tags=["sar", "residue", "hotspot", "mcp:dock_ensemble"]
                ))

    # Insight 3: Ligand property → interaction correlation (NEW)
    # Connect prepare_ligand outputs to analyze_interactions outcomes
    aromatic_binders = []
    hbond_binders = []

    for r in completed:
        ligand_props = r.get("ligand_properties", {})
        interactions = r.get("interaction_types", [])
        energy = r.get("best_energy", 0)

        # Check aromatic → aromatic stacking correlation
        # Handle both list format (from MCP) and count format
        aromatic_rings = ligand_props.get("aromatic_rings", [])
        n_aromatic = ligand_props.get("n_aromatic_rings", len(aromatic_rings) if isinstance(aromatic_rings, list) else 0)
        if n_aromatic > 0:
            aromatic_binders.append({
                "name": r["ligand_name"],
                "n_rings": n_aromatic,
                "has_aromatic_interaction": any("aromatic" in str(i).lower() or "pi" in str(i).lower() for i in interactions),
                "energy": energy
            })

        # Check H-bond donors → H-bond correlation
        # Handle both list format (from MCP) and count format
        hbond_donors = ligand_props.get("hbond_donors", ligand_props.get("h_bond_donors", []))
        n_hbond = ligand_props.get("n_hbond_donors", len(hbond_donors) if isinstance(hbond_donors, list) else 0)
        if n_hbond > 0:
            hbond_binders.append({
                "name": r["ligand_name"],
                "n_donors": n_hbond,
                "has_hbond_interaction": any("h-bond" in str(i).lower() or "h_bond" in str(i).lower() for i in interactions),
                "energy": energy
            })

    # Aromatic SAR insight
    if len(aromatic_binders) >= 2:
        with_aromatic = [b for b in aromatic_binders if b["has_aromatic_interaction"]]
        if len(with_aromatic) >= 2:
            avg_aromatic = sum(b["energy"] for b in with_aromatic) / len(with_aromatic)
            if avg_aromatic <= THRESHOLDS.drug_like:
                insights.append(SARInsight(
                    insight_id=str(uuid.uuid4())[:8],
                    pattern_type="functional_group_correlation",
                    title="Aromatic Groups Enable π-Stacking",
                    description=f"Ligands with aromatic rings (from prepare_ligand) that form π-stacking (from analyze_interactions) show avg energy {avg_aromatic:.1f} kcal/mol. Source: prepare_ligand → analyze_interactions correlation.",
                    evidence_ligands=[b["name"] for b in with_aromatic],
                    confidence=min(0.85, len(with_aromatic) / 4),
                    recommendation="Preserve or add aromatic moieties to maintain π-stacking capability",
                    tags=["sar", "aromatic", "functional_group", "mcp:prepare_ligand", "mcp:analyze_interactions"]
                ))

    # H-bond SAR insight
    if len(hbond_binders) >= 2:
        with_hbond = [b for b in hbond_binders if b["has_hbond_interaction"]]
        if len(with_hbond) >= 2:
            avg_hbond = sum(b["energy"] for b in with_hbond) / len(with_hbond)
            if avg_hbond <= THRESHOLDS.drug_like:
                insights.append(SARInsight(
                    insight_id=str(uuid.uuid4())[:8],
                    pattern_type="functional_group_correlation",
                    title="H-bond Donors Critical for Binding",
                    description=f"Ligands with H-bond donors (N-H, O-H from prepare_ligand) that form H-bonds (from analyze_interactions) show avg energy {avg_hbond:.1f} kcal/mol. Source: prepare_ligand → analyze_interactions correlation.",
                    evidence_ligands=[b["name"] for b in with_hbond],
                    confidence=min(0.85, len(with_hbond) / 4),
                    recommendation="Ensure ligand has H-bond donor groups (amines, hydroxyls) positioned for Y125/Y133/Y136 interaction",
                    tags=["sar", "hbond", "functional_group", "mcp:prepare_ligand", "mcp:analyze_interactions"]
                ))

    # Insight 4: Cluster population weighted insight (NEW)
    # Connect cluster_conformations population weights
    cluster_energies = {}
    for r in completed:
        cluster = r.get("best_cluster")
        if cluster is not None:
            if cluster not in cluster_energies:
                cluster_energies[cluster] = []
            cluster_energies[cluster].append({
                "ligand": r["ligand_name"],
                "energy": r.get("best_energy", 0),
                "population": r.get("cluster_population", 0.0)
            })

    for cluster_id, results in cluster_energies.items():
        if len(results) >= 2:
            avg_energy = sum(r["energy"] for r in results) / len(results)
            avg_pop = sum(r.get("population", 0) for r in results) / len(results) if results else 0
            if avg_energy <= THRESHOLDS.drug_like and avg_pop > 0.1:
                insights.append(SARInsight(
                    insight_id=str(uuid.uuid4())[:8],
                    pattern_type="conformational_selection",
                    title=f"Cluster {cluster_id}: Preferred Binding Conformation",
                    description=f"Cluster {cluster_id} (population {avg_pop:.1%}) shows consistent drug-like binding (avg {avg_energy:.1f} kcal/mol). This conformation is both common and favorable. Source: cluster_conformations + dock_ensemble MCP tools.",
                    evidence_ligands=[r["ligand"] for r in results],
                    confidence=min(0.8, avg_pop * 2),
                    recommendation=f"Focus docking on cluster {cluster_id} conformations for lead optimization",
                    tags=["sar", "conformation", "ensemble", "mcp:cluster_conformations", "mcp:dock_ensemble"]
                ))

    return insights


# =============================================================================
# Scientific Graph Construction
# =============================================================================

def build_scientific_graph(
    experiment_artifact: Dict,
    protein_name: str = "Alpha-Synuclein"
) -> Tuple[List[ScientificEntity], List[ScientificRelationship]]:
    """
    Build three-level scientific graph from experiment results.

    Following MedGraphRAG triple-linked structure:
    - Level 1: Experimental data (our docking results)
    - Level 2: Reference data (ChEMBL compounds, PED ensembles)
    - Level 3: Controlled vocabularies (UniProt, PubChem IDs)
    """
    entities = []
    relationships = []
    content = experiment_artifact.get("content", {})

    # Level 1: Experimental Protein Entity
    protein_ref_key, protein_external = resolve_to_reference(protein_name, "Protein")

    exp_protein = ScientificEntity(
        entity_id=f"exp_protein_{uuid.uuid4().hex[:8]}",
        entity_type="Protein",
        level=1,
        name=protein_name,
        properties={
            "pdb_path": content.get("protein_pdb_path"),
            "n_clusters": content.get("n_clusters"),
            "experiment_id": content.get("experiment_id"),
        }
    )
    entities.append(exp_protein)

    # Level 2: Reference Protein (if resolved)
    if protein_ref_key:
        ref_protein = ScientificEntity(
            entity_id=f"ref_protein_{protein_ref_key}",
            entity_type="Protein",
            level=2,
            name=protein_name,
            external_ids=protein_external
        )
        entities.append(ref_protein)

        # CORRESPONDS_TO relationship
        relationships.append(ScientificRelationship(
            relationship_id=f"corr_{uuid.uuid4().hex[:8]}",
            subject_id=exp_protein.entity_id,
            predicate="CORRESPONDS_TO",
            object_id=ref_protein.entity_id,
            properties={"resolution_method": "jaro_winkler"}
        ))

    # Process ligand results
    for result in content.get("ligand_results", []):
        ligand_name = result.get("ligand_name", "Unknown")

        # Level 1: Experimental Ligand
        exp_ligand = ScientificEntity(
            entity_id=f"exp_ligand_{uuid.uuid4().hex[:8]}",
            entity_type="Ligand",
            level=1,
            name=ligand_name,
            properties={
                "smiles": result.get("smiles"),
                "best_energy": result.get("best_energy"),
                "best_cluster": result.get("best_cluster"),
                "status": result.get("status"),
            }
        )
        entities.append(exp_ligand)

        # Level 2: Reference Ligand (if resolved)
        ligand_ref_key, ligand_external = resolve_to_reference(ligand_name, "Ligand")
        if ligand_ref_key:
            ref_ligand = ScientificEntity(
                entity_id=f"ref_ligand_{ligand_ref_key}",
                entity_type="Ligand",
                level=2,
                name=ligand_name,
                external_ids=ligand_external
            )
            entities.append(ref_ligand)

            relationships.append(ScientificRelationship(
                relationship_id=f"corr_{uuid.uuid4().hex[:8]}",
                subject_id=exp_ligand.entity_id,
                predicate="CORRESPONDS_TO",
                object_id=ref_ligand.entity_id,
                properties={"resolution_method": "jaro_winkler"}
            ))

        # DOCKS_TO relationship (if completed)
        if result.get("status") == "completed":
            relationships.append(ScientificRelationship(
                relationship_id=f"docks_{uuid.uuid4().hex[:8]}",
                subject_id=exp_ligand.entity_id,
                predicate="DOCKS_TO",
                object_id=exp_protein.entity_id,
                properties={
                    "best_energy": result.get("best_energy"),
                    "best_cluster": result.get("best_cluster"),
                }
            ))

        # Residue interaction
        residue = result.get("best_residue")
        if residue:
            residue_entity = ScientificEntity(
                entity_id=f"residue_Y{residue}",
                entity_type="Residue",
                level=1,
                name=f"Y{residue}",
                properties={"residue_number": residue, "residue_type": "TYR"}
            )
            entities.append(residue_entity)

            for itype in result.get("interaction_types", []):
                relationships.append(ScientificRelationship(
                    relationship_id=f"int_{uuid.uuid4().hex[:8]}",
                    subject_id=exp_ligand.entity_id,
                    predicate="INTERACTS_WITH",
                    object_id=residue_entity.entity_id,
                    properties={"interaction_type": itype}
                ))

    return entities, relationships


# =============================================================================
# Tool Functions (using ToolContext for state - KG_ADK pattern)
# =============================================================================

def analyze_experiment_results(
    experiment_artifact_id: str,
    protein_name: str = "Alpha-Synuclein",
    tool_context: ToolContext = None
) -> dict:
    """
    Analyze experiment results: UCB rankings + SAR insights.

    Sets results in tool_context.state for propose/approve workflow.
    """
    artifact = read_artifact(experiment_artifact_id)
    if not artifact:
        return {"success": False, "error": "Experiment not found"}

    content = artifact.get("content", {})
    ligand_results = content.get("ligand_results", [])

    # Calculate UCB rankings
    ucb_rankings = calculate_ucb_ranking(ligand_results)

    # Discover SAR insights
    sar_insights = discover_sar_insights(ligand_results)

    # Energy statistics
    completed = [r for r in ligand_results if r.get("status") == "completed" and r.get("best_energy")]
    if completed:
        energies = [r["best_energy"] for r in completed]
        stats = {
            "n_completed": len(completed),
            "best_energy": min(energies),
            "worst_energy": max(energies),
            "mean_energy": sum(energies) / len(energies),
            "n_drug_like": sum(1 for e in energies if e <= THRESHOLDS.drug_like),
        }
    else:
        stats = {"n_completed": 0}

    analysis = {
        "experiment_id": content.get("experiment_id"),
        "protein_name": protein_name,
        "energy_stats": stats,
        "ucb_rankings": ucb_rankings,
        "sar_insights": [asdict(i) for i in sar_insights],
        "n_insights": len(sar_insights),
    }

    # Store in state for propose/approve (KG_ADK pattern)
    if tool_context:
        tool_context.state["proposed_analysis"] = analysis
        tool_context.state["proposed_insights"] = [asdict(i) for i in sar_insights]

    return {"success": True, **analysis}


def propose_knowledge_items(
    tool_context: ToolContext
) -> dict:
    """
    Propose knowledge items from analysis for user approval.

    Following KG_ADK propose/approve pattern from user_intent.md.
    """
    analysis = tool_context.state.get("proposed_analysis")
    insights = tool_context.state.get("proposed_insights", [])

    if not analysis:
        return {"success": False, "error": "No analysis found. Run analyze_experiment_results first."}

    proposed_items = []

    # Propose SAR insights as knowledge items
    for insight in insights:
        proposed_items.append({
            "type": KnowledgeType.INSIGHT,
            "title": insight["title"],
            "content": insight["description"],
            "tags": insight.get("tags", []),
            "recommendation": insight.get("recommendation"),
        })

    # Propose best result as knowledge item
    stats = analysis.get("energy_stats", {})
    if stats.get("n_drug_like", 0) > 0:
        rankings = analysis.get("ucb_rankings", [])
        best = rankings[0] if rankings else None
        if best:
            proposed_items.append({
                "type": KnowledgeType.RESULT,
                "title": f"Drug-like binding: {best['ligand_name']}",
                "content": f"{best['ligand_name']} shows drug-like binding ({best['best_energy']:.1f} kcal/mol) to {analysis['protein_name']}. UCB score: {best['ucb_score']}",
                "tags": ["result", "drug_like", analysis["protein_name"].lower().replace(" ", "_")],
            })

    # Propose procedure (successful workflow)
    if stats.get("n_completed", 0) > 0:
        proposed_items.append({
            "type": KnowledgeType.PROCEDURE,
            "title": f"Ensemble docking: {analysis['protein_name']}",
            "content": f"Successfully docked {stats['n_completed']} ligands to {analysis['protein_name']} ensemble. Best: {stats.get('best_energy', 'N/A'):.1f} kcal/mol, Drug-like: {stats.get('n_drug_like', 0)}",
            "tags": ["procedure", "docking", "ensemble"],
        })

    tool_context.state["proposed_knowledge"] = proposed_items

    return {
        "success": True,
        "n_proposed": len(proposed_items),
        "proposed_items": proposed_items,
        "message": f"Proposed {len(proposed_items)} knowledge items. Review and call approve_knowledge_items() to persist."
    }


def approve_knowledge_items(
    workspace_id: str = "ws_core",
    conversation_id: str = None,
    tool_context: ToolContext = None
) -> dict:
    """
    Approve and persist proposed knowledge items to the database.

    Uses knowledge_service from core/api/services.py.
    Items appear in the Evolution tab of the frontend.
    """
    if not tool_context:
        return {"success": False, "error": "ToolContext required"}

    proposed = tool_context.state.get("proposed_knowledge", [])
    if not proposed:
        return {"success": False, "error": "No proposed knowledge. Call propose_knowledge_items first."}

    # Import knowledge_service (lazy to avoid circular imports)
    from core.api.services import knowledge_service

    created_items = []
    for item in proposed:
        try:
            knowledge = knowledge_service.create(
                workspace_id=workspace_id,
                knowledge_type=item["type"],
                title=item["title"],
                content=item["content"],
                source_conversation_id=conversation_id,
                tags=item.get("tags", [])
            )
            created_items.append(knowledge)
        except Exception as e:
            print(f"Warning: Failed to create knowledge item: {e}")

    # Clear proposed state
    tool_context.state["proposed_knowledge"] = []
    tool_context.state["approved_knowledge"] = created_items

    return {
        "success": True,
        "n_created": len(created_items),
        "created_items": created_items,
        "message": f"Created {len(created_items)} knowledge items in Evolution tab."
    }


def build_knowledge_graph(
    experiment_artifact_id: str,
    protein_name: str = "Alpha-Synuclein",
    tool_context: ToolContext = None
) -> dict:
    """
    Build scientific knowledge graph from experiment.

    Creates three-level graph following MedGraphRAG pattern.
    """
    artifact = read_artifact(experiment_artifact_id)
    if not artifact:
        return {"success": False, "error": "Experiment not found"}

    entities, relationships = build_scientific_graph(artifact, protein_name)

    # Create graph artifact
    graph_content = {
        "entities": [asdict(e) for e in entities],
        "relationships": [asdict(r) for r in relationships],
        "n_entities": len(entities),
        "n_relationships": len(relationships),
        "protein_name": protein_name,
        "levels": {
            "experimental": len([e for e in entities if e.level == 1]),
            "reference": len([e for e in entities if e.level == 2]),
            "vocabulary": len([e for e in entities if e.level == 3]),
        },
        "created_at": datetime.now().isoformat(),
    }

    graph_artifact = create_artifact(
        artifact_type=ArtifactType.EVOLUTION_TRACE,
        name=f"Scientific Graph: {protein_name}",
        content=graph_content,
        tags=["knowledge_graph", "scientific", protein_name.lower().replace(" ", "_")],
        tool_context=tool_context
    )

    # Persist to Neo4j for real graph queries
    neo4j_status = {"success": False, "message": "Neo4j not attempted"}
    try:
        driver, database = get_neo4j_driver()
        with driver.session(database=database) as session:
            # Create nodes for each entity
            for entity in entities:
                session.run("""
                    MERGE (n {id: $id})
                    SET n:Entity,
                        n.name = $name,
                        n.type = $type,
                        n.level = $level,
                        n.properties = $properties,
                        n.external_ids = $external_ids,
                        n.experiment_id = $experiment_id
                    RETURN n
                """, {
                    "id": entity.entity_id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "level": entity.level,
                    "properties": json.dumps(entity.properties),
                    "external_ids": json.dumps(entity.external_ids),
                    "experiment_id": experiment_artifact_id
                })

            # Create relationships
            for rel in relationships:
                session.run("""
                    MATCH (a {id: $subject_id})
                    MATCH (b {id: $object_id})
                    MERGE (a)-[r:RELATIONSHIP {id: $rel_id}]->(b)
                    SET r.type = $predicate,
                        r.properties = $properties,
                        r.experiment_id = $experiment_id
                    RETURN r
                """, {
                    "subject_id": rel.subject_id,
                    "object_id": rel.object_id,
                    "rel_id": rel.relationship_id,
                    "predicate": rel.predicate,
                    "properties": json.dumps(rel.properties),
                    "experiment_id": experiment_artifact_id
                })

        driver.close()
        neo4j_status = {"success": True, "message": f"Persisted {len(entities)} nodes and {len(relationships)} edges to Neo4j"}
    except Exception as e:
        neo4j_status = {"success": False, "message": f"Neo4j error: {str(e)}"}

    # Store in state
    if tool_context:
        tool_context.state["knowledge_graph"] = graph_content
        tool_context.state["knowledge_graph_id"] = graph_artifact["id"]
        tool_context.state["neo4j_status"] = neo4j_status

    return {
        "success": True,
        "artifact_id": graph_artifact["id"],
        "n_entities": len(entities),
        "n_relationships": len(relationships),
        "entity_types": list(set(e.entity_type for e in entities)),
        "relationship_types": list(set(r.predicate for r in relationships)),
        "external_links": [
            {"entity": e.name, "ids": e.external_ids}
            for e in entities if e.external_ids
        ],
        "neo4j_status": neo4j_status
    }


# =============================================================================
# Neo4j Query Tools (NEW - for graph retrieval)
# =============================================================================

def query_knowledge_graph(
    entity_type: str = None,
    property_filter: Dict[str, Any] = None,
    limit: int = 20,
    tool_context: ToolContext = None
) -> dict:
    """
    Query Neo4j knowledge graph for entities.

    Args:
        entity_type: Filter by type (Protein, Ligand, Residue)
        property_filter: Dict of property name → value to match
        limit: Max results to return

    Returns:
        List of matching entities with their properties.
    """
    try:
        driver, database = get_neo4j_driver()
        with driver.session(database=database) as session:
            # Build query
            where_clauses = []
            params = {"limit": limit}

            if entity_type:
                where_clauses.append("n.type = $entity_type")
                params["entity_type"] = entity_type

            if property_filter:
                for key, value in property_filter.items():
                    param_name = f"prop_{key}"
                    where_clauses.append(f"n.properties CONTAINS ${param_name}")
                    params[param_name] = f'"{key}": {json.dumps(value)}'

            where_str = " AND ".join(where_clauses) if where_clauses else "TRUE"

            result = session.run(f"""
                MATCH (n:Entity)
                WHERE {where_str}
                RETURN n.id AS id, n.name AS name, n.type AS type,
                       n.level AS level, n.properties AS properties,
                       n.external_ids AS external_ids
                LIMIT $limit
            """, params)

            entities = []
            for record in result:
                entities.append({
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "level": record["level"],
                    "properties": json.loads(record["properties"]) if record["properties"] else {},
                    "external_ids": json.loads(record["external_ids"]) if record["external_ids"] else {},
                })

        driver.close()
        return {
            "success": True,
            "n_results": len(entities),
            "entities": entities,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def find_related_entities(
    entity_id: str,
    relationship_type: str = None,
    direction: str = "both",
    depth: int = 1,
    tool_context: ToolContext = None
) -> dict:
    """
    Find entities related to a given entity via graph traversal.

    Args:
        entity_id: Source entity ID
        relationship_type: Filter by relationship (DOCKS_TO, INTERACTS_WITH, CORRESPONDS_TO)
        direction: "outgoing", "incoming", or "both"
        depth: Max relationship hops (1-3)

    Returns:
        Related entities with relationship details.
    """
    try:
        driver, database = get_neo4j_driver()
        with driver.session(database=database) as session:
            depth = min(max(depth, 1), 3)  # Clamp to 1-3

            if direction == "outgoing":
                pattern = f"(source)-[r*1..{depth}]->(target)"
            elif direction == "incoming":
                pattern = f"(source)<-[r*1..{depth}]-(target)"
            else:
                pattern = f"(source)-[r*1..{depth}]-(target)"

            rel_filter = f"ALL(rel IN r WHERE rel.type = $rel_type)" if relationship_type else "TRUE"

            result = session.run(f"""
                MATCH {pattern}
                WHERE source.id = $entity_id AND {rel_filter}
                RETURN target.id AS id, target.name AS name, target.type AS type,
                       target.properties AS properties,
                       [rel IN r | {{type: rel.type, properties: rel.properties}}] AS path
                LIMIT 50
            """, {"entity_id": entity_id, "rel_type": relationship_type})

            related = []
            for record in result:
                related.append({
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "properties": json.loads(record["properties"]) if record["properties"] else {},
                    "relationship_path": record["path"],
                })

        driver.close()
        return {
            "success": True,
            "source_id": entity_id,
            "n_related": len(related),
            "related_entities": related,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_binding_hotspots(
    protein_name: str = "Alpha-Synuclein",
    energy_threshold: float = -5.0,
    tool_context: ToolContext = None
) -> dict:
    """
    Get binding hotspots from knowledge graph - residues with best binding.

    Uses INTERACTS_WITH relationships and DOCKS_TO energies.
    """
    try:
        driver, database = get_neo4j_driver()
        with driver.session(database=database) as session:
            result = session.run("""
                MATCH (ligand:Entity {type: 'Ligand'})-[dock:RELATIONSHIP {type: 'DOCKS_TO'}]->(protein:Entity {type: 'Protein'})
                MATCH (ligand)-[interact:RELATIONSHIP {type: 'INTERACTS_WITH'}]->(residue:Entity {type: 'Residue'})
                WHERE protein.name = $protein_name
                WITH residue.name AS residue_name, residue.id AS residue_id,
                     collect(DISTINCT ligand.name) AS ligands,
                     collect(DISTINCT dock.properties) AS dock_props
                RETURN residue_name, residue_id, ligands, dock_props,
                       size(ligands) AS n_ligands
                ORDER BY n_ligands DESC
                LIMIT 10
            """, {"protein_name": protein_name})

            hotspots = []
            for record in result:
                # Parse energies from dock properties
                energies = []
                for prop_str in record["dock_props"]:
                    if prop_str:
                        props = json.loads(prop_str) if isinstance(prop_str, str) else prop_str
                        if props.get("best_energy"):
                            energies.append(props["best_energy"])

                avg_energy = sum(energies) / len(energies) if energies else None
                best_energy = min(energies) if energies else None

                hotspots.append({
                    "residue": record["residue_name"],
                    "residue_id": record["residue_id"],
                    "n_ligands": record["n_ligands"],
                    "ligands": record["ligands"],
                    "avg_energy": round(avg_energy, 2) if avg_energy else None,
                    "best_energy": round(best_energy, 2) if best_energy else None,
                    "is_hotspot": best_energy and best_energy <= energy_threshold,
                })

        driver.close()
        return {
            "success": True,
            "protein": protein_name,
            "energy_threshold": energy_threshold,
            "n_hotspots": len([h for h in hotspots if h["is_hotspot"]]),
            "hotspots": hotspots,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Gemini 3 Molecule Generation (SAR-informed)
# =============================================================================

def propose_molecule_generation(
    seed_smiles: str = None,
    modification_type: str = "optimize",
    n_suggestions: int = 3,
    tool_context: ToolContext = None
) -> dict:
    """
    PROPOSE molecule generation - requires user approval before running Gemini 3.

    This tool shows the user what will be generated and asks for approval.
    Call approve_molecule_generation() after user confirms.

    Args:
        seed_smiles: Starting molecule SMILES (optional - can suggest de novo)
        modification_type: "optimize" (improve binding), "explore" (diverse), "scaffold_hop"
        n_suggestions: Number of molecules to suggest (1-5)
    """
    # Get SAR context from state
    sar_insights = []
    ucb_rankings = []

    if tool_context:
        sar_insights = tool_context.state.get("proposed_insights", [])
        analysis = tool_context.state.get("proposed_analysis", {})
        ucb_rankings = analysis.get("ucb_rankings", [])

    # Build proposal
    proposal = {
        "seed_smiles": seed_smiles,
        "modification_type": modification_type,
        "n_suggestions": n_suggestions,
        "generate_3d": True,
        "sar_context": {
            "n_insights": len(sar_insights),
            "n_rankings": len(ucb_rankings),
            "top_insights": [i.get("title", "") for i in sar_insights[:3]],
        }
    }

    if tool_context:
        tool_context.state["molecule_generation_proposal"] = proposal

    return {
        "success": True,
        "status": "AWAITING_APPROVAL",
        "proposal": proposal,
        "message": f"""🧪 **Molecule Generation Proposal**

**Model:** Gemini 3 Pro (with SAR context)
**Strategy:** {modification_type}
**Seed:** {seed_smiles or 'De novo generation'}
**Count:** {n_suggestions} suggestions

**SAR Context:**
- {len(sar_insights)} insights will inform generation
- {len(ucb_rankings)} ranked ligands for reference

**Will generate:**
1. SMILES strings (validated with RDKit)
2. 3D conformations (MMFF optimized)
3. Drug-likeness assessment
4. PDB blocks for visualization

⚠️ **User approval required.** Call approve_molecule_generation() to proceed."""
    }


def approve_molecule_generation(
    tool_context: ToolContext = None
) -> dict:
    """
    Execute approved molecule generation with Gemini 3.

    Must be called after propose_molecule_generation() and user approval.
    """
    if not tool_context:
        return {"success": False, "error": "ToolContext required"}

    proposal = tool_context.state.get("molecule_generation_proposal")
    if not proposal:
        return {"success": False, "error": "No proposal found. Call propose_molecule_generation first."}

    # Execute generation
    return _execute_molecule_generation(
        seed_smiles=proposal.get("seed_smiles"),
        modification_type=proposal.get("modification_type", "optimize"),
        n_suggestions=proposal.get("n_suggestions", 3),
        generate_3d=proposal.get("generate_3d", True),
        tool_context=tool_context
    )


def _execute_molecule_generation(
    seed_smiles: str = None,
    modification_type: str = "optimize",
    n_suggestions: int = 3,
    generate_3d: bool = True,
    tool_context: ToolContext = None
) -> dict:
    """
    Internal: Execute molecule generation using Gemini 3 + SAR insights.

    Uses SAR insights from tool_context.state to guide generation.
    Optionally generates 3D conformations via RDKit.

    Research basis:
    - Gemini 3 projected 82% validity in de novo design (SparkCo analysis)
    - SMILES → 3D via RDKit AllChem.EmbedMolecule
    - SAR-informed generation per IRDiff (ICML 2024)
    """
    # Get SAR context from state
    sar_insights = []
    ucb_rankings = []
    kg_patterns = []

    if tool_context:
        sar_insights = tool_context.state.get("proposed_insights", [])
        analysis = tool_context.state.get("proposed_analysis", {})
        ucb_rankings = analysis.get("ucb_rankings", [])
        kg_patterns = tool_context.state.get("sar_from_graph", [])

    # Build SAR context for prompt
    sar_context = []
    if sar_insights:
        sar_context.append("**SAR Insights from docking experiments:**")
        for insight in sar_insights[:5]:
            sar_context.append(f"- {insight.get('title', '')}: {insight.get('recommendation', '')}")

    if ucb_rankings:
        sar_context.append("\n**Top performing ligands (UCB ranking):**")
        for r in ucb_rankings[:3]:
            sar_context.append(f"- {r['ligand_name']}: {r['best_energy']:.1f} kcal/mol")

    if kg_patterns:
        sar_context.append("\n**Patterns from knowledge graph:**")
        for p in kg_patterns[:3]:
            sar_context.append(f"- {p.get('title', '')}")

    sar_text = "\n".join(sar_context) if sar_context else "No prior SAR data available."

    # Build generation prompt
    if modification_type == "optimize":
        strategy = """Optimize binding affinity by:
1. Preserving key pharmacophore features
2. Adding/modifying groups that enhance identified beneficial interactions
3. Maintaining drug-likeness (Lipinski's Rule of 5)"""
    elif modification_type == "explore":
        strategy = """Explore chemical space by:
1. Trying different scaffolds while maintaining key binding features
2. Varying ring systems and linkers
3. Testing different heteroatom positions"""
    else:  # scaffold_hop
        strategy = """Perform scaffold hopping:
1. Replace core scaffold while preserving pharmacophore
2. Maintain similar 3D shape and electrostatic properties
3. Explore bioisosteric replacements"""

    seed_context = f"Starting molecule: {seed_smiles}" if seed_smiles else "Generate de novo (no seed molecule)"

    prompt = f"""You are a medicinal chemist AI designing molecules for alpha-synuclein binding.

{seed_context}

{sar_text}

**Generation Strategy:** {modification_type}
{strategy}

Generate exactly {n_suggestions} molecule suggestions.

For EACH suggestion, provide:
1. **name**: A descriptive name (e.g., "Fasudil-hydroxyl-analog")
2. **smiles**: Valid SMILES string
3. **rationale**: How this addresses the SAR insights (1-2 sentences)
4. **predicted_features**: List of expected interaction types

Return as valid JSON array:
[{{"name": "...", "smiles": "...", "rationale": "...", "predicted_features": ["h_bond", "aromatic"]}}]

IMPORTANT:
- Only output valid, synthesizable molecules
- SMILES must be chemically valid
- Consider drug-likeness (MW < 500, LogP < 5, HBD ≤ 5, HBA ≤ 10)
"""

    try:
        # Use Gemini 3 for generation (new google.genai SDK)
        from google import genai

        # Create client with API key
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        # Try gemini-3-pro-preview first, fallback to 2.0
        model_name = "gemini-3-pro-preview"
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
        except Exception:
            model_name = "gemini-2.0-flash"
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

        # Parse response
        response_text = response.text.strip()

        # Extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            suggestions = json.loads(json_match.group())
        else:
            return {"success": False, "error": "Failed to parse Gemini response as JSON"}

        # Validate and optionally generate 3D
        valid_suggestions = []
        for sugg in suggestions[:n_suggestions]:
            smiles = sugg.get("smiles", "")

            # Validate SMILES with RDKit
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem, Descriptors

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    sugg["valid"] = False
                    sugg["error"] = "Invalid SMILES"
                    valid_suggestions.append(sugg)
                    continue

                # Calculate properties
                sugg["valid"] = True
                sugg["molecular_weight"] = round(Descriptors.MolWt(mol), 1)
                sugg["logp"] = round(Descriptors.MolLogP(mol), 2)
                sugg["hbd"] = Descriptors.NumHDonors(mol)
                sugg["hba"] = Descriptors.NumHAcceptors(mol)
                sugg["rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)

                # Drug-likeness check
                sugg["drug_like"] = (
                    sugg["molecular_weight"] <= 500 and
                    sugg["logp"] <= 5 and
                    sugg["hbd"] <= 5 and
                    sugg["hba"] <= 10
                )

                # Generate 3D conformation
                if generate_3d:
                    mol_3d = Chem.AddHs(mol)
                    embed_result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)

                    if embed_result == 0:  # Success
                        AllChem.MMFFOptimizeMolecule(mol_3d)
                        conf = mol_3d.GetConformer()

                        # Extract 3D coordinates
                        coords = []
                        for i in range(mol_3d.GetNumAtoms()):
                            pos = conf.GetAtomPosition(i)
                            atom = mol_3d.GetAtomWithIdx(i)
                            coords.append({
                                "atom": atom.GetSymbol(),
                                "x": round(pos.x, 3),
                                "y": round(pos.y, 3),
                                "z": round(pos.z, 3)
                            })
                        sugg["coordinates_3d"] = coords
                        sugg["n_atoms"] = len(coords)

                        # Generate PDB block for visualization
                        sugg["pdb_block"] = Chem.MolToPDBBlock(mol_3d)
                    else:
                        sugg["coordinates_3d"] = None
                        sugg["error_3d"] = "Failed to generate 3D conformation"

            except ImportError:
                sugg["valid"] = None
                sugg["error"] = "RDKit not available for validation"
            except Exception as e:
                sugg["valid"] = False
                sugg["error"] = str(e)

            valid_suggestions.append(sugg)

        # Store in state for downstream use
        if tool_context:
            tool_context.state["molecule_suggestions"] = valid_suggestions
            tool_context.state["generation_model"] = model_name

        n_valid = sum(1 for s in valid_suggestions if s.get("valid"))
        n_drug_like = sum(1 for s in valid_suggestions if s.get("drug_like"))
        n_with_3d = sum(1 for s in valid_suggestions if s.get("coordinates_3d"))

        return {
            "success": True,
            "model_used": model_name,
            "n_suggestions": len(valid_suggestions),
            "n_valid": n_valid,
            "n_drug_like": n_drug_like,
            "n_with_3d": n_with_3d,
            "modification_type": modification_type,
            "seed_smiles": seed_smiles,
            "sar_context_used": len(sar_insights) + len(ucb_rankings) + len(kg_patterns),
            "suggestions": valid_suggestions,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def prepare_suggestions_for_docking(
    tool_context: ToolContext = None
) -> dict:
    """
    Prepare generated molecule suggestions for docking pipeline.

    Takes suggestions from tool_context.state["molecule_suggestions"]
    and formats them for the Engineering agent's docking workflow.

    Returns ligand configs compatible with dock_ensemble MCP tool.
    """
    if not tool_context:
        return {"success": False, "error": "ToolContext required"}

    suggestions = tool_context.state.get("molecule_suggestions", [])
    if not suggestions:
        return {"success": False, "error": "No molecule suggestions. Call generate_molecule_suggestions first."}

    # Filter to valid, drug-like molecules
    docking_candidates = []
    for sugg in suggestions:
        if not sugg.get("valid") or not sugg.get("drug_like"):
            continue

        candidate = {
            "name": sugg.get("name", f"GenMol_{len(docking_candidates)+1}"),
            "smiles": sugg["smiles"],
            "source": "gemini_3_generation",
            "rationale": sugg.get("rationale", ""),
            "predicted_features": sugg.get("predicted_features", []),
            "molecular_weight": sugg.get("molecular_weight"),
            "logp": sugg.get("logp"),
        }

        # Include 3D if available
        if sugg.get("pdb_block"):
            candidate["pdb_block"] = sugg["pdb_block"]
            candidate["has_3d"] = True

        docking_candidates.append(candidate)

    tool_context.state["docking_candidates"] = docking_candidates

    return {
        "success": True,
        "n_candidates": len(docking_candidates),
        "candidates": [
            {"name": c["name"], "smiles": c["smiles"], "has_3d": c.get("has_3d", False)}
            for c in docking_candidates
        ],
        "message": f"Prepared {len(docking_candidates)} candidates for docking. Pass to Engineering agent."
    }


def get_sar_from_graph(
    min_ligands: int = 2,
    tool_context: ToolContext = None
) -> dict:
    """
    Extract SAR patterns from knowledge graph using graph queries.

    Finds:
    - Interaction types shared by top binders
    - Residues with multiple good-binding ligands
    - Ligand property → binding correlations
    """
    try:
        driver, database = get_neo4j_driver()
        patterns = []

        with driver.session(database=database) as session:
            # Pattern 1: Interaction types in drug-like bindings
            result = session.run("""
                MATCH (ligand:Entity {type: 'Ligand'})-[dock:RELATIONSHIP {type: 'DOCKS_TO'}]->(protein:Entity)
                MATCH (ligand)-[interact:RELATIONSHIP {type: 'INTERACTS_WITH'}]->(residue:Entity)
                WITH ligand, interact.properties AS int_props, dock.properties AS dock_props
                WHERE dock_props IS NOT NULL
                WITH ligand.name AS ligand_name, int_props, dock_props
                RETURN ligand_name, int_props, dock_props
                LIMIT 100
            """)

            interactions_by_energy = {"good": [], "poor": []}
            for record in result:
                dock_props = json.loads(record["dock_props"]) if isinstance(record["dock_props"], str) else record["dock_props"]
                int_props = json.loads(record["int_props"]) if isinstance(record["int_props"], str) else record["int_props"]

                energy = dock_props.get("best_energy") if dock_props else None
                int_type = int_props.get("interaction_type") if int_props else None

                if energy and int_type:
                    if energy <= -5.0:
                        interactions_by_energy["good"].append(int_type)
                    else:
                        interactions_by_energy["poor"].append(int_type)

            good_set = set(interactions_by_energy["good"])
            poor_set = set(interactions_by_energy["poor"])
            beneficial = good_set - poor_set

            if beneficial:
                patterns.append({
                    "pattern_type": "interaction_preference",
                    "title": f"Beneficial interactions: {', '.join(beneficial)}",
                    "description": f"Good binders show {', '.join(beneficial)} not seen in poor binders",
                    "evidence_count": len(interactions_by_energy["good"]),
                    "source": "neo4j_graph_query",
                })

            # Pattern 2: Residue hotspots
            result = session.run("""
                MATCH (ligand:Entity {type: 'Ligand'})-[interact:RELATIONSHIP {type: 'INTERACTS_WITH'}]->(residue:Entity {type: 'Residue'})
                WITH residue.name AS residue, count(DISTINCT ligand) AS n_ligands,
                     collect(DISTINCT ligand.name) AS ligands
                WHERE n_ligands >= $min_ligands
                RETURN residue, n_ligands, ligands
                ORDER BY n_ligands DESC
            """, {"min_ligands": min_ligands})

            for record in result:
                patterns.append({
                    "pattern_type": "residue_hotspot",
                    "title": f"Hotspot: {record['residue']}",
                    "description": f"{record['residue']} interacts with {record['n_ligands']} ligands: {', '.join(record['ligands'])}",
                    "evidence_count": record["n_ligands"],
                    "source": "neo4j_graph_query",
                })

        driver.close()
        return {
            "success": True,
            "n_patterns": len(patterns),
            "sar_patterns": patterns,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_optimization_recommendations(
    experiment_artifact_id: str,
    tool_context: ToolContext = None
) -> dict:
    """
    Generate recommendations for next optimization cycle.

    Combines UCB rankings and SAR insights into actionable recommendations.
    """
    artifact = read_artifact(experiment_artifact_id)
    if not artifact:
        return {"success": False, "error": "Experiment not found"}

    content = artifact.get("content", {})
    ligand_results = content.get("ligand_results", [])

    ucb_rankings = calculate_ucb_ranking(ligand_results)
    sar_insights = discover_sar_insights(ligand_results)

    recommendations = []

    # SAR-based recommendations
    for insight in sar_insights:
        recommendations.append({
            "type": "sar_insight",
            "priority": "high" if insight.confidence > 0.7 else "medium",
            "recommendation": insight.recommendation,
            "evidence": insight.description,
            "confidence": insight.confidence,
        })

    # Exploration recommendation (UCB-based)
    if ucb_rankings:
        top = ucb_rankings[0]
        if top["exploration_bonus"] > 2.0:
            recommendations.append({
                "type": "exploration",
                "priority": "medium",
                "recommendation": f"Under-explored: {top['ligand_name']} has high uncertainty. Consider more conformations.",
                "evidence": f"UCB: {top['ucb_score']}, only {top['n_samples']} samples",
                "confidence": 0.6,
            })

    # Exploitation recommendation
    drug_like = [r for r in ucb_rankings if r["drug_like"]]
    if drug_like:
        recommendations.append({
            "type": "exploitation",
            "priority": "high",
            "recommendation": f"Focus on analogs of {drug_like[0]['ligand_name']} for lead optimization",
            "evidence": f"Best: {drug_like[0]['best_energy']:.1f} kcal/mol",
            "confidence": 0.8,
        })

    if tool_context:
        tool_context.state["recommendations"] = recommendations

    return {
        "success": True,
        "n_recommendations": len(recommendations),
        "recommendations": recommendations,
    }


def create_discovery_report(
    experiment_artifact_id: str,
    protein_name: str = "Alpha-Synuclein",
    workspace_id: str = "ws_core",
    tool_context: ToolContext = None
) -> dict:
    """
    Create comprehensive Discovery Report.

    Combines:
    - Analysis results
    - SAR insights
    - Knowledge graph
    - Recommendations
    """
    # Get analysis
    analysis_result = analyze_experiment_results(experiment_artifact_id, protein_name, tool_context)
    if not analysis_result.get("success"):
        return analysis_result

    # Build knowledge graph
    kg_result = build_knowledge_graph(experiment_artifact_id, protein_name, tool_context)

    # Generate recommendations
    rec_result = generate_optimization_recommendations(experiment_artifact_id, tool_context)

    stats = analysis_result.get("energy_stats", {})
    rankings = analysis_result.get("ucb_rankings", [])
    insights = analysis_result.get("sar_insights", [])

    # Build summary
    n_completed = stats.get("n_completed", 0)
    n_drug_like = stats.get("n_drug_like", 0)
    best_energy = stats.get("best_energy")
    top_ligand = rankings[0]["ligand_name"] if rankings else "N/A"

    # Format best_energy safely
    energy_str = f"{best_energy:.1f}" if isinstance(best_energy, (int, float)) else "N/A"

    summary = f"""Analyzed {n_completed} ligands against {protein_name} ensemble.
Top performer: {top_ligand} ({energy_str} kcal/mol).
{n_drug_like} compounds meet drug-like threshold (≤{THRESHOLDS.drug_like} kcal/mol).
Discovered {len(insights)} SAR patterns.
Built knowledge graph with {kg_result.get('n_entities', 0)} entities."""

    # Create report artifact
    report_content = {
        "title": f"Discovery Report: {protein_name}",
        "executive_summary": summary,
        "analysis_date": datetime.now().isoformat(),
        "energy_stats": stats,
        "ucb_rankings": rankings,
        "sar_insights": insights,
        "knowledge_graph_id": kg_result.get("artifact_id"),
        "recommendations": rec_result.get("recommendations", []),
        "methodology": {
            "ranking": "rUCB (Chen et al. 2021)",
            "sar": "Statistical pattern discovery",
            "graph": "MedGraphRAG triple-linked structure",
            "resolution": "Jaro-Winkler similarity",
        }
    }

    report_artifact = create_artifact(
        artifact_type=ArtifactType.DISCOVERY_REPORT,
        name=f"Discovery Report: {protein_name}",
        content=report_content,
        tags=["discovery", "report", protein_name.lower().replace(" ", "_")],
        parent_id=experiment_artifact_id
    )

    return {
        "success": True,
        "artifact_id": report_artifact["id"],
        "title": report_content["title"],
        "executive_summary": summary,
        "n_rankings": len(rankings),
        "n_insights": len(insights),
        "n_recommendations": len(rec_result.get("recommendations", [])),
        "knowledge_graph_id": kg_result.get("artifact_id"),
    }


# =============================================================================
# Build Instruction
# =============================================================================

def build_evolution_instruction() -> str:
    """Build evolution agent instruction using templates."""

    role = PROMPTS.role_template.format(
        agent_name="Evolution Scientist",
        specialty="Knowledge graph construction, SAR discovery, and self-improvement through learning"
    )

    constraints = PROMPTS.constraints_template.format(
        current_date=datetime.now().strftime("%Y-%m-%d"),
        additional_constraints=f"Drug-like threshold: {THRESHOLDS.drug_like} kcal/mol. Always use propose/approve for knowledge items."
    )

    reasoning = PROMPTS.reasoning_template

    example = PROMPTS.example_template.format(
        input_example="Analyze docking results and extract learnings",
        reasoning_example="1) Analyze experiment for UCB rankings and SAR, 2) Propose knowledge items for review, 3) Approve to persist to Evolution tab, 4) Build knowledge graph, 5) Generate recommendations",
        action_example="analyze_experiment_results() → propose_knowledge_items() → [user approves] → approve_knowledge_items() → build_knowledge_graph() → create_discovery_report()",
        output_example="Knowledge items in Evolution tab + Discovery Report artifact"
    )

    methodology = """
<methodology>
Based on:

1. MedGraphRAG (ACL 2025):
   - Triple-linked structure: Experimental → Reference → Vocabulary
   - CORRESPONDS_TO relationships for entity resolution

2. rUCB Algorithm (Chen et al. 2021):
   - Multi-armed bandit for ligand prioritization
   - UCB = -mean_energy + c * sqrt(log(N)/n)

3. KG_ADK Patterns:
   - Propose → Approve workflow for knowledge items
   - ToolContext.state for passing data between steps
   - Entity resolution using Jaro-Winkler similarity
</methodology>"""

    mcp_integration = """
<mcp_tool_integration>
**CRITICAL: Connect MCP tool outputs for deep SAR discovery**

The 4 MCP docking tools provide rich data. Your job is to find PATTERNS:

1. **prepare_ligand outputs** → Ligand Properties
   - aromatic_rings: Count and positions
   - h_bond_donors: N-H, O-H pairs detected
   - charged_atoms: Positive/negative centers

   SAR Question: Do binders share functional groups?

2. **cluster_conformations outputs** → Conformational Relevance
   - cluster_populations: Weight of each conformation
   - representative_frames: Which frames matter

   SAR Question: Which conformations favor binding?

3. **dock_ensemble outputs** → Energy Landscape
   - energies: Per-cluster binding energies
   - weighted_average: Population-adjusted score

   SAR Question: Which clusters show consistent binding?

4. **analyze_interactions outputs** → Molecular Contacts
   - h_bonds: Donor-acceptor pairs
   - hydrophobic: Van der Waals contacts
   - aromatic: Pi-stacking interactions
   - charge: Salt bridges

   SAR Question: Which interaction types predict potency?

**Pattern Discovery Logic:**
```
IF prepare_ligand detects aromatic_rings > 0
AND analyze_interactions finds aromatic contacts
AND energy < drug_like_threshold
THEN → SAR Insight: "Aromatic groups essential for binding"
```
</mcp_tool_integration>"""

    knowledge_types = """
<knowledge_types>
Types for Evolution tab (from KnowledgeService):
- insight: SAR patterns, binding preferences, functional group requirements
- procedure: Successful workflows, protocols, docking parameters
- result: Important binding results, drug-like hits
- feedback: User corrections (via feedback)
- reference: External database links (ChEMBL, PED)
</knowledge_types>"""

    tools_doc = """
<tools>
**Analysis:**
- analyze_experiment_results(artifact_id, protein_name) → UCB + SAR

**Knowledge (Propose/Approve):**
- propose_knowledge_items() → Propose items for review
- approve_knowledge_items(workspace_id) → Persist to Evolution tab

**Graph Construction:**
- build_knowledge_graph(artifact_id, protein_name) → Build and persist graph to Neo4j

**Graph Queries (retrieve from Neo4j):**
- query_knowledge_graph(entity_type, property_filter, limit) → Find entities by type/properties
- find_related_entities(entity_id, relationship_type, direction, depth) → Graph traversal
- get_binding_hotspots(protein_name, energy_threshold) → Residues with best binding
- get_sar_from_graph(min_ligands) → Extract SAR patterns from graph structure

**Molecule Generation (OPTIONAL - Requires User Approval):**
- propose_molecule_generation(seed_smiles, modification_type, n_suggestions) → PROPOSE Gemini 3 generation
- approve_molecule_generation() → EXECUTE after user approval
- prepare_suggestions_for_docking() → Format for Engineering agent

**Recommendations:**
- generate_optimization_recommendations(artifact_id) → Next steps

**Report:**
- create_discovery_report(artifact_id, protein_name, workspace_id) → Full report
</tools>"""

    workflow = """
<workflow>
1. ANALYZE: Call analyze_experiment_results() for rankings and insights
   - UCB prioritizes ligands balancing best energy vs uncertainty
   - SAR discovers patterns from ligand properties → interactions

2. PROPOSE: Call propose_knowledge_items() to prepare knowledge items
   - Present proposed items to user for review
   - Include: SAR insights, drug-like results, successful procedures
   - Ask: "These insights will be saved to Evolution. Approve?"

3. APPROVE: After user confirmation, call approve_knowledge_items()
   - Items persist to database
   - Appear in Evolution tab

4. BUILD GRAPH: Call build_knowledge_graph() for entity relationships
   - Experimental entities link to ChEMBL/PED references
   - CORRESPONDS_TO relationships via Jaro-Winkler similarity

5. RECOMMEND: Call generate_optimization_recommendations() for next steps
   - Based on SAR: "Add H-bond donor at position X"
   - Based on UCB: "Under-explored ligand Y needs more samples"
   - Based on interactions: "Target Y136 shows highest occupancy"

6. REPORT: Call create_discovery_report() for final artifact
</workflow>"""

    critical_thinking = """
<critical_thinking>
**What should you think critically about?**

1. **Correlation vs Causation**
   - Just because top binders have aromatic rings doesn't mean aromatics cause binding
   - Check: Do ALL aromatics bind well, or just specific positions?

2. **Conformational Bias**
   - cluster_populations tell you which conformations are common
   - Weight your SAR insights by population - rare conformations may be artifacts

3. **Interaction Quality**
   - Not all H-bonds are equal - check distances and angles
   - Strong interactions (< 2.5 Å) more predictive than weak

4. **Learning Loop**
   - Your recommendations should feed back to Research agent
   - If you find "Y136 is hotspot", Research should search for Y136 binders in literature

5. **External Validation**
   - Cross-reference SAR insights with ChEMBL bioactivity data
   - Does known potent compound share your predicted features?
</critical_thinking>"""

    return f"""
{role}

{constraints}

{methodology}

{mcp_integration}

{knowledge_types}

{reasoning}

{example}

{tools_doc}

{workflow}

{critical_thinking}

<output>
1. Executive Summary (2-3 sentences)
2. Top 3 ligands with UCB scores
3. Key SAR insights (persisted to Evolution)
   - Include which MCP tool outputs support each insight
4. Recommendations for next experiment
   - Concrete molecular modifications or new ligands to test
5. Report artifact ID
</output>
"""


# =============================================================================
# Agent Definition
# =============================================================================

evolution_agent = LlmAgent(
    name="evolution_agent",
    model=MODELS.evolution,
    description="Analyzes results, discovers SAR patterns, builds knowledge graphs, generates 3D molecules with Gemini 3 (on approval), and persists learnings",
    instruction=build_evolution_instruction(),
    tools=[
        # Analysis
        analyze_experiment_results,
        # Knowledge management
        propose_knowledge_items,
        approve_knowledge_items,
        # Graph construction
        build_knowledge_graph,
        # Neo4j query tools
        query_knowledge_graph,
        find_related_entities,
        get_binding_hotspots,
        get_sar_from_graph,
        # Gemini 3 molecule generation (requires user approval)
        propose_molecule_generation,
        approve_molecule_generation,
        prepare_suggestions_for_docking,
        # Recommendations
        generate_optimization_recommendations,
        create_discovery_report,
    ],
    output_key="evolution_results",
)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main agent
    "evolution_agent",

    # Analysis
    "analyze_experiment_results",
    "calculate_ucb_ranking",
    "discover_sar_insights",

    # Knowledge management
    "propose_knowledge_items",
    "approve_knowledge_items",

    # Graph construction
    "build_knowledge_graph",
    "build_scientific_graph",

    # Neo4j query tools
    "query_knowledge_graph",
    "find_related_entities",
    "get_binding_hotspots",
    "get_sar_from_graph",

    # Gemini 3 molecule generation
    "propose_molecule_generation",
    "approve_molecule_generation",
    "prepare_suggestions_for_docking",

    # Recommendations
    "generate_optimization_recommendations",
    "create_discovery_report",

    # Entity resolution
    "jaro_winkler_similarity",
    "resolve_to_reference",

    # Types
    "KnowledgeType",
    "ScientificEntity",
    "ScientificRelationship",
    "SARInsight",
]
