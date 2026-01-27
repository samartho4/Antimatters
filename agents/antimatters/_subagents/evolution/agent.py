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
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

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
        tags=["knowledge_graph", "scientific", protein_name.lower().replace(" ", "_")]
    )

    # Store in state
    if tool_context:
        tool_context.state["knowledge_graph"] = graph_content
        tool_context.state["knowledge_graph_id"] = graph_artifact["id"]

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
        ]
    }


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
    best_energy = stats.get("best_energy", "N/A")
    top_ligand = rankings[0]["ligand_name"] if rankings else "N/A"

    summary = f"""Analyzed {n_completed} ligands against {protein_name} ensemble.
Top performer: {top_ligand} ({best_energy:.1f} kcal/mol).
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

**Graph:**
- build_knowledge_graph(artifact_id, protein_name) → Scientific graph

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
    description="Analyzes results, discovers SAR patterns, builds knowledge graphs, and persists learnings to Evolution tab",
    instruction=build_evolution_instruction(),
    tools=[
        analyze_experiment_results,
        propose_knowledge_items,
        approve_knowledge_items,
        build_knowledge_graph,
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

    # Graph
    "build_knowledge_graph",
    "build_scientific_graph",

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
