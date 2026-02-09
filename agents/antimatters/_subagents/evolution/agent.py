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
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from google.adk.agents import LlmAgent

# Configure logger for verification logging
logger = logging.getLogger(__name__)
from google.adk.models.google_llm import Gemini
from google.genai import types as genai_retry_types
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
from google.genai import types as genai_types
from core.agents.base import (
    create_artifact,
    read_artifact,
    update_artifact,
    ArtifactType,
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
    # ===== VERIFICATION LOGGING: Processing ligand_results =====
    ligand_results = content.get("ligand_results", [])
    logger.info(f"[build_scientific_graph] ===== VERIFY: Processing {len(ligand_results)} ligand results =====")
    for i, lr in enumerate(ligand_results):
        logger.info(f"[build_scientific_graph] [VERIFY] ligand_results[{i}]: name={lr.get('ligand_name')}, status={lr.get('status')}, interaction_types={lr.get('interaction_types', [])}")
    logger.info(f"[build_scientific_graph] ===== END ligand_results preview =====")

    for result in ligand_results:
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
        interaction_types = result.get("interaction_types", [])

        # ===== VERIFICATION LOGGING: INTERACTS_WITH creation =====
        logger.info(f"[build_scientific_graph] [{ligand_name}] ===== VERIFY: INTERACTS_WITH CREATION =====")
        logger.info(f"[build_scientific_graph] [{ligand_name}] [VERIFY] best_residue: {residue}")
        logger.info(f"[build_scientific_graph] [{ligand_name}] [VERIFY] interaction_types from result: {interaction_types}")
        logger.info(f"[build_scientific_graph] [{ligand_name}] [VERIFY] interaction_types count: {len(interaction_types)}")

        if residue:
            residue_entity = ScientificEntity(
                entity_id=f"residue_Y{residue}",
                entity_type="Residue",
                level=1,
                name=f"Y{residue}",
                properties={"residue_number": residue, "residue_type": "TYR"}
            )
            entities.append(residue_entity)

            if interaction_types:
                for itype in interaction_types:
                    rel_id = f"int_{uuid.uuid4().hex[:8]}"
                    # CRITICAL: This creates the INTERACTS_WITH relationship that get_binding_hotspots queries
                    print(f"[build_scientific_graph] [{ligand_name}] [VERIFY] ★★★ CREATING INTERACTS_WITH: {exp_ligand.entity_id} --[{itype}]--> {residue_entity.entity_id} ★★★", flush=True)
                    logger.info(f"[build_scientific_graph] [{ligand_name}] [VERIFY] ✓ CREATING INTERACTS_WITH: {exp_ligand.entity_id} --[{itype}]--> {residue_entity.entity_id}")
                    relationships.append(ScientificRelationship(
                        relationship_id=rel_id,
                        subject_id=exp_ligand.entity_id,
                        predicate="INTERACTS_WITH",
                        object_id=residue_entity.entity_id,
                        properties={"interaction_type": itype}
                    ))
            else:
                print(f"[build_scientific_graph] [{ligand_name}] [VERIFY] ⚠ WARNING: No interaction_types! No INTERACTS_WITH will be created!", flush=True)
                logger.warning(f"[build_scientific_graph] [{ligand_name}] [VERIFY] ⚠ WARNING: No interaction_types! No INTERACTS_WITH will be created!")
        else:
            logger.warning(f"[build_scientific_graph] [{ligand_name}] [VERIFY] ⚠ WARNING: No best_residue! Cannot create Residue entity or INTERACTS_WITH!")
        logger.info(f"[build_scientific_graph] [{ligand_name}] ===== END INTERACTS_WITH CREATION =====")

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
    tool_context: ToolContext,
    auto_approve: bool = True
) -> dict:
    """
    Propose knowledge items from analysis.

    In Analysis mode (workflow), auto_approve=True saves items immediately.
    In interactive mode, set auto_approve=False to review first.
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

    # Auto-approve in Analysis mode (workflow) - no user interaction needed
    if auto_approve:
        # Automatically call approve_knowledge_items
        approve_result = approve_knowledge_items(
            workspace_id="ws_core",
            tool_context=tool_context
        )
        return {
            "success": True,
            "n_proposed": len(proposed_items),
            "n_saved": approve_result.get("n_saved", 0),
            "proposed_items": proposed_items,
            "auto_approved": True,
            "message": f"Proposed and AUTO-SAVED {len(proposed_items)} knowledge items to Evolution tab."
        }

    return {
        "success": True,
        "n_proposed": len(proposed_items),
        "proposed_items": proposed_items,
        "auto_approved": False,
        "message": f"Proposed {len(proposed_items)} knowledge items. Call approve_knowledge_items() to persist."
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
        logger.error(f"[build_knowledge_graph] [VERIFY] Experiment artifact NOT FOUND: {experiment_artifact_id}")
        return {"success": False, "error": "Experiment not found"}

    # ===== VERIFICATION LOGGING: build_knowledge_graph START =====
    logger.info(f"[build_knowledge_graph] ===== VERIFY: build_knowledge_graph CALLED =====")
    logger.info(f"[build_knowledge_graph] [VERIFY] experiment_artifact_id: {experiment_artifact_id}")
    logger.info(f"[build_knowledge_graph] [VERIFY] protein_name: {protein_name}")
    logger.info(f"[build_knowledge_graph] [VERIFY] artifact keys: {list(artifact.keys())}")
    content = artifact.get("content", {})
    logger.info(f"[build_knowledge_graph] [VERIFY] content keys: {list(content.keys())}")
    logger.info(f"[build_knowledge_graph] [VERIFY] ligand_results count: {len(content.get('ligand_results', []))}")
    logger.info(f"[build_knowledge_graph] ===== END build_knowledge_graph START =====")

    entities, relationships = build_scientific_graph(artifact, protein_name)

    # ===== VERIFICATION LOGGING: build_scientific_graph RESULT =====
    logger.info(f"[build_knowledge_graph] ===== VERIFY: build_scientific_graph RESULT =====")
    logger.info(f"[build_knowledge_graph] [VERIFY] Total entities: {len(entities)}")
    logger.info(f"[build_knowledge_graph] [VERIFY] Total relationships: {len(relationships)}")

    # Count relationship types
    rel_counts = {}
    for rel in relationships:
        predicate = rel.predicate
        rel_counts[predicate] = rel_counts.get(predicate, 0) + 1
    logger.info(f"[build_knowledge_graph] [VERIFY] Relationship breakdown: {rel_counts}")

    interacts_with_count = rel_counts.get("INTERACTS_WITH", 0)
    if interacts_with_count > 0:
        logger.info(f"[build_knowledge_graph] [VERIFY] ✓ GOOD: {interacts_with_count} INTERACTS_WITH relationships will be persisted to Neo4j")
    else:
        logger.warning(f"[build_knowledge_graph] [VERIFY] ⚠ WARNING: NO INTERACTS_WITH relationships! KG will have no spatial interaction data!")
    logger.info(f"[build_knowledge_graph] ===== END build_scientific_graph RESULT =====")

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

    # ===== VERIFICATION LOGGING: Neo4j PERSISTENCE =====
    logger.info(f"[build_knowledge_graph] ===== VERIFY: Neo4j PERSISTENCE =====")

    try:
        driver, database = get_neo4j_driver()
        logger.info(f"[build_knowledge_graph] [VERIFY] Connected to Neo4j database: {database}")

        with driver.session(database=database) as session:
            # Create nodes for each entity
            logger.info(f"[build_knowledge_graph] [VERIFY] Creating {len(entities)} entity nodes...")
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
            logger.info(f"[build_knowledge_graph] [VERIFY] Creating {len(relationships)} relationships...")
            interacts_count = 0
            for rel in relationships:
                if rel.predicate == "INTERACTS_WITH":
                    interacts_count += 1
                    logger.info(f"[build_knowledge_graph] [VERIFY] ✓ PERSISTING INTERACTS_WITH: {rel.subject_id} --[{rel.properties.get('interaction_type')}]--> {rel.object_id}")

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

            logger.info(f"[build_knowledge_graph] [VERIFY] Neo4j persist complete: {len(entities)} nodes, {len(relationships)} relationships ({interacts_count} INTERACTS_WITH)")

        driver.close()
        neo4j_status = {"success": True, "message": f"Persisted {len(entities)} nodes and {len(relationships)} edges to Neo4j"}
        logger.info(f"[build_knowledge_graph] [VERIFY] ✓ Neo4j SUCCESS: {neo4j_status['message']}")
    except Exception as e:
        neo4j_status = {"success": False, "message": f"Neo4j error: {str(e)}"}
        logger.error(f"[build_knowledge_graph] [VERIFY] ⚠ Neo4j FAILED: {str(e)}")
        import traceback
        logger.error(f"[build_knowledge_graph] [VERIFY] Neo4j traceback: {traceback.format_exc()}")
    logger.info(f"[build_knowledge_graph] ===== END Neo4j PERSISTENCE =====")

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
    property_filter_json: str = None,
    limit: int = 20,
    tool_context: ToolContext = None
) -> dict:
    """
    Query Neo4j knowledge graph for entities.

    Args:
        entity_type: Filter by type (Protein, Ligand, Residue)
        property_filter_json: JSON string of property filters, e.g. '{"binding_energy": -8.5}'
        limit: Max results to return

    Returns:
        List of matching entities with their properties.
    """
    # ===== VERIFICATION LOGGING: query_knowledge_graph CALLED =====
    logger.info(f"[query_knowledge_graph] ===== VERIFY: PLANNING AGENT KG QUERY =====")
    logger.info(f"[query_knowledge_graph] [VERIFY] entity_type: {entity_type}")
    logger.info(f"[query_knowledge_graph] [VERIFY] property_filter_json: {property_filter_json}")
    logger.info(f"[query_knowledge_graph] [VERIFY] limit: {limit}")

    try:
        # Parse property filter from JSON string
        property_filter = None
        if property_filter_json:
            try:
                property_filter = json.loads(property_filter_json)
            except json.JSONDecodeError:
                return {"success": False, "error": f"Invalid JSON in property_filter_json: {property_filter_json}"}

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

        # Also fetch relationships between these entities
        entity_ids = [e["id"] for e in entities]
        relationships = []

        if entity_ids:
            with driver.session(database=database) as session:
                rel_result = session.run("""
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE a.id IN $entity_ids AND b.id IN $entity_ids
                    RETURN a.id AS subject_id, type(r) AS predicate, b.id AS object_id,
                           properties(r) AS properties, id(r) AS rel_id
                    LIMIT 500
                """, {"entity_ids": entity_ids})

                for record in rel_result:
                    relationships.append({
                        "relationship_id": f"rel_{record['rel_id']}",
                        "subject_id": record["subject_id"],
                        "predicate": record["predicate"],
                        "object_id": record["object_id"],
                        "properties": record["properties"] or {},
                    })

        driver.close()

        # ===== VERIFICATION LOGGING: query_knowledge_graph RESULT =====
        logger.info(f"[query_knowledge_graph] [VERIFY] SUCCESS: Found {len(entities)} entities, {len(relationships)} relationships")
        if entities:
            for i, ent in enumerate(entities[:5]):  # First 5
                logger.info(f"[query_knowledge_graph] [VERIFY] entity[{i}]: type={ent.get('type')}, name={ent.get('name')}, id={ent.get('id')}")
        else:
            logger.warning(f"[query_knowledge_graph] [VERIFY] ⚠ NO ENTITIES FOUND! KG may be empty or query filters too strict.")

        # Check for INTERACTS_WITH relationships
        interacts_count = sum(1 for r in relationships if r.get('predicate') == 'INTERACTS_WITH')
        logger.info(f"[query_knowledge_graph] [VERIFY] INTERACTS_WITH relationships in result: {interacts_count}")
        if interacts_count == 0:
            logger.warning(f"[query_knowledge_graph] [VERIFY] ⚠ NO INTERACTS_WITH found! Planning agent has no spatial data to report.")
        logger.info(f"[query_knowledge_graph] ===== END PLANNING AGENT KG QUERY =====")

        return {
            "success": True,
            "n_results": len(entities),
            "entities": entities,
            "relationships": relationships,
        }

    except Exception as e:
        logger.error(f"[query_knowledge_graph] [VERIFY] ERROR: {str(e)}")
        return {"success": False, "error": str(e), "relationships": []}


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
    # ===== VERIFICATION LOGGING: get_binding_hotspots CALLED =====
    logger.info(f"[get_binding_hotspots] ===== VERIFY: PLANNING AGENT HOTSPOTS QUERY =====")
    logger.info(f"[get_binding_hotspots] [VERIFY] protein_name: {protein_name}")
    logger.info(f"[get_binding_hotspots] [VERIFY] energy_threshold: {energy_threshold}")
    logger.info(f"[get_binding_hotspots] [VERIFY] This query requires INTERACTS_WITH relationships in Neo4j!")

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

        # ===== VERIFICATION LOGGING: get_binding_hotspots RESULT =====
        n_hotspots = len([h for h in hotspots if h["is_hotspot"]])
        # Use print() for guaranteed visibility
        print(f"[get_binding_hotspots] [VERIFY] ★★★ Found {len(hotspots)} residues, {n_hotspots} are hotspots ★★★", flush=True)
        logger.info(f"[get_binding_hotspots] [VERIFY] Found {len(hotspots)} residues, {n_hotspots} are hotspots (energy <= {energy_threshold})")

        if hotspots:
            for i, hs in enumerate(hotspots[:5]):  # First 5
                print(f"[get_binding_hotspots] [VERIFY] hotspot[{i}]: residue={hs.get('residue')}, n_ligands={hs.get('n_ligands')}, best_energy={hs.get('best_energy')}", flush=True)
                logger.info(f"[get_binding_hotspots] [VERIFY] hotspot[{i}]: residue={hs.get('residue')}, n_ligands={hs.get('n_ligands')}, best_energy={hs.get('best_energy')}, is_hotspot={hs.get('is_hotspot')}")
        else:
            print(f"[get_binding_hotspots] [VERIFY] ⚠ NO HOTSPOTS FOUND! No INTERACTS_WITH in Neo4j for {protein_name}", flush=True)
            logger.warning(f"[get_binding_hotspots] [VERIFY] ⚠ NO HOTSPOTS FOUND!")
            logger.warning(f"[get_binding_hotspots] [VERIFY] This means NO INTERACTS_WITH relationships exist in Neo4j for {protein_name}.")
            logger.warning(f"[get_binding_hotspots] [VERIFY] Root cause: analyze_interactions returned empty interaction_types, OR build_knowledge_graph was never called.")
        logger.info(f"[get_binding_hotspots] ===== END PLANNING AGENT HOTSPOTS QUERY =====")

        return {
            "success": True,
            "protein": protein_name,
            "energy_threshold": energy_threshold,
            "n_hotspots": n_hotspots,
            "hotspots": hotspots,
        }

    except Exception as e:
        logger.error(f"[get_binding_hotspots] [VERIFY] ERROR: {str(e)}")
        return {"success": False, "error": str(e)}


def create_interaction_relationship(
    ligand_id: str,
    residue_id: str,
    interaction_type: str,
    properties_json: str = "{}",
    experiment_id: str = None,
    tool_context: ToolContext = None
) -> dict:
    """
    Create INTERACTS_WITH relationship between ligand and residue in Neo4j.

    Use after spatial analysis identifies specific interactions (H-bonds, pi-stacking, etc).

    Args:
        ligand_id: Entity ID or name (e.g., "exp_ligand_abc123" or "Ligand-47")
        residue_id: Entity ID or name (e.g., "residue_Y129" or "Y125")
        interaction_type: h_bond | pi_stacking | hydrophobic | ionic | vdw
        properties_json: JSON with distance, angle, etc. (e.g., '{"distance_angstrom": 2.8}')
        experiment_id: Optional experiment artifact ID

    Returns:
        Success status with relationship ID
    """
    try:
        properties = json.loads(properties_json) if properties_json else {}
        properties["interaction_type"] = interaction_type
        relationship_id = f"int_{uuid.uuid4().hex[:8]}"

        driver, database = get_neo4j_driver()
        with driver.session(database=database) as session:
            # FIX: Match by EITHER id OR name (case-insensitive for names)
            # This allows Gemini to pass "Ligand-47" or "Y125" instead of exact IDs
            result = session.run("""
                MATCH (ligand:Entity)
                WHERE ligand.id = $ligand_id
                   OR toLower(ligand.name) = toLower($ligand_id)
                   OR toLower(replace(ligand.name, '-', '_')) = toLower(replace($ligand_id, '-', '_'))
                WITH ligand ORDER BY
                    CASE WHEN ligand.id = $ligand_id THEN 0 ELSE 1 END
                LIMIT 1
                MATCH (residue:Entity)
                WHERE residue.id = $residue_id
                   OR toLower(residue.name) = toLower($residue_id)
                   OR residue.id = 'residue_' + $residue_id
                WITH ligand, residue ORDER BY
                    CASE WHEN residue.id = $residue_id THEN 0 ELSE 1 END
                LIMIT 1
                MERGE (ligand)-[r:RELATIONSHIP {id: $rel_id}]->(residue)
                SET r.type = 'INTERACTS_WITH',
                    r.properties = $properties,
                    r.experiment_id = $experiment_id,
                    r.created_at = datetime()
                RETURN ligand.id AS matched_ligand, residue.id AS matched_residue, r
            """, {
                "ligand_id": ligand_id,
                "residue_id": residue_id,
                "rel_id": relationship_id,
                "properties": json.dumps(properties),
                "experiment_id": experiment_id
            })

            record = result.single()
            if not record:
                return {"success": False, "error": f"Entities not found - ligand: {ligand_id}, residue: {residue_id}"}

        matched_ligand = record["matched_ligand"]
        matched_residue = record["matched_residue"]
        driver.close()
        return {
            "success": True,
            "relationship_id": relationship_id,
            "matched_ligand_id": matched_ligand,
            "matched_residue_id": matched_residue,
            "message": f"Created INTERACTS_WITH: {matched_ligand} → {matched_residue} ({interaction_type})"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Gemini 3 Molecule Generation (SAR-informed)
# =============================================================================

def generate_molecules_direct(
    seed_smiles: str = None,
    modification_type: str = "optimize",
    n_suggestions: int = 3,
    generation_mode: str = "smiles",
    tool_context: ToolContext = None
) -> dict:
    """
    DIRECT molecule generation - NO approval required. Use in PLANNING MODE.

    Args:
        seed_smiles: Starting molecule SMILES (optional - can suggest de novo)
        modification_type: "optimize" (improve binding), "explore" (diverse), "scaffold_hop"
        n_suggestions: Number of molecules to suggest (1-5)
        generation_mode: "smiles" (Gemini→SMILES→RDKit 3D) or "rcmt" (experimental, direct 3D)

    Returns:
        Generated molecules with clear indication of generation method:
        - generation_method: "smiles_rdkit" or "rcmt_direct"
        - For smiles_rdkit: Gemini generates SMILES, RDKit validates and creates 3D
        - For rcmt_direct: Gemini generates 3D coords directly (experimental, may have bond errors)
    """
    return _execute_molecule_generation(
        seed_smiles=seed_smiles,
        modification_type=modification_type,
        n_suggestions=n_suggestions,
        generate_3d=True,
        generation_mode=generation_mode,
        tool_context=tool_context
    )


def propose_molecule_generation(
    seed_smiles: str = None,
    modification_type: str = "optimize",
    n_suggestions: int = 3,
    tool_context: ToolContext = None
) -> dict:
    """
    PROPOSE molecule generation - requires user approval before running Gemini 3.

    NOTE: For PLANNING MODE (no docking), use generate_molecules_direct() instead.
    This function is for ANALYSIS MODE where user oversight is desired.

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
    generation_mode: str = "smiles",
    tool_context: ToolContext = None
) -> dict:
    """
    Internal: Execute molecule generation using Gemini 3 + SAR insights.

    Uses SAR insights from tool_context.state to guide generation.

    Args:
        generation_mode: "smiles" or "rcmt"
            - smiles: Gemini generates SMILES → RDKit validates → RDKit generates 3D
            - rcmt: Gemini generates RCMT format with 3D coords (experimental, may have bond errors)

    Output includes:
        - generation_method: "smiles_rdkit" or "rcmt_direct" (clear indication of method)
        - For smiles_rdkit: 3D is from RDKit MMFF optimization
        - For rcmt_direct: 3D is directly from Gemini (experimental)

    Research basis:
    - Gemini 3 projected 82% validity in de novo design (SparkCo analysis)
    - SMILES → 3D via RDKit AllChem.EmbedMolecule
    - RCMT from Chem3DLLM: ATOM@x,y,z format for lossless 3D
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

    # RCMT mode is experimental - warn and fall back to SMILES if requested
    # Testing showed Gemini makes bond errors with RCMT generation
    rcmt_warning = None
    if generation_mode == "rcmt":
        rcmt_warning = "⚠️ RCMT mode is experimental. Gemini may make bond errors. Using SMILES mode with RDKit 3D instead."
        # Fall back to SMILES mode for reliable results
        # Future: implement RCMT with post-processing to fix bond errors
        generation_mode = "smiles"

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

        # Determine generation method for clear output
        generation_method = "smiles_rdkit"  # Currently always SMILES (RCMT falls back)

        # Store in state for downstream use
        if tool_context:
            tool_context.state["molecule_suggestions"] = valid_suggestions
            tool_context.state["generation_model"] = model_name
            tool_context.state["generation_method"] = generation_method

        n_valid = sum(1 for s in valid_suggestions if s.get("valid"))
        n_drug_like = sum(1 for s in valid_suggestions if s.get("drug_like"))
        n_with_3d = sum(1 for s in valid_suggestions if s.get("coordinates_3d"))

        result = {
            "success": True,
            "model_used": model_name,
            "generation_method": generation_method,
            "generation_method_explanation": "Gemini generated SMILES → RDKit validated → RDKit created 3D (MMFF optimized)",
            "n_suggestions": len(valid_suggestions),
            "n_valid": n_valid,
            "n_drug_like": n_drug_like,
            "n_with_3d": n_with_3d,
            "modification_type": modification_type,
            "seed_smiles": seed_smiles,
            "sar_context_used": len(sar_insights) + len(ucb_rankings) + len(kg_patterns),
            "suggestions": valid_suggestions,
        }

        # Add RCMT fallback warning if applicable
        if rcmt_warning:
            result["rcmt_fallback_warning"] = rcmt_warning

        return result

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
    # ===== VERIFICATION LOGGING: get_sar_from_graph CALLED =====
    logger.info(f"[get_sar_from_graph] ===== VERIFY: PLANNING AGENT SAR QUERY =====")
    logger.info(f"[get_sar_from_graph] [VERIFY] min_ligands: {min_ligands}")
    logger.info(f"[get_sar_from_graph] [VERIFY] This query requires INTERACTS_WITH relationships in Neo4j!")

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

        # ===== VERIFICATION LOGGING: get_sar_from_graph RESULT =====
        logger.info(f"[get_sar_from_graph] [VERIFY] Found {len(patterns)} SAR patterns")
        if patterns:
            for i, pat in enumerate(patterns):
                logger.info(f"[get_sar_from_graph] [VERIFY] pattern[{i}]: type={pat.get('pattern_type')}, title={pat.get('title')}")
        else:
            logger.warning(f"[get_sar_from_graph] [VERIFY] ⚠ NO SAR PATTERNS FOUND!")
            logger.warning(f"[get_sar_from_graph] [VERIFY] This means NO INTERACTS_WITH relationships exist in Neo4j.")
            logger.warning(f"[get_sar_from_graph] [VERIFY] Root cause: analyze_interactions returned empty, OR build_knowledge_graph was never called.")
        logger.info(f"[get_sar_from_graph] ===== END PLANNING AGENT SAR QUERY =====")

        return {
            "success": True,
            "n_patterns": len(patterns),
            "sar_patterns": patterns,
        }

    except Exception as e:
        logger.error(f"[get_sar_from_graph] [VERIFY] ERROR: {str(e)}")
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


async def create_discovery_report(
    experiment_artifact_id: str,
    protein_name: str = "Alpha-Synuclein",
    workspace_id: str = "ws_core",
    tool_context: ToolContext = None
) -> dict:
    """
    Create comprehensive Discovery Report with 3D visualizations.

    Combines:
    - Analysis results
    - SAR insights
    - Knowledge graph
    - Recommendations
    - 3D structure visualization
    - Cluster conformation overlay

    Saves to both:
    1. Disk-based system (for persistence)
    2. ADK InMemoryArtifactService (for ADK web visibility)
    """
    # Import visualization functions
    from core.agents.antimatters._subagents.callbacks.visualization import (
        generate_3d_structure_view,
        generate_cluster_overlay,
        generate_docking_plot,
    )

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

    # Generate 3D visualizations
    visualization_artifacts = []
    experiment_artifact = read_artifact(experiment_artifact_id)
    if experiment_artifact:
        content = experiment_artifact.get("content", {})
        protein_pdb_path = content.get("protein_pdb_path", "")
        representative_frames = content.get("representative_frames", [])
        cluster_populations = content.get("cluster_populations", [])

        # Generate 3D structure view for top ligand
        if protein_pdb_path and rankings:
            top_ligand_data = rankings[0] if rankings else {}
            structure_img = generate_3d_structure_view(
                protein_pdb_path=protein_pdb_path,
                ligand_name=top_ligand_data.get("ligand_name", "Top Ligand"),
                title=f"Docked Complex: {protein_name}"
            )
            if structure_img and tool_context and hasattr(tool_context, 'save_artifact'):
                try:
                    protein_slug = protein_name.lower().replace(" ", "_").replace("-", "_")
                    filename = f"3d_structure_{protein_slug}.png"
                    version = await save_artifact_to_adk(
                        tool_context, filename, structure_img, "image/png"
                    )
                    if version >= 0:
                        visualization_artifacts.append(filename)
                        print(f"[ADK] Saved 3D structure to ADK web: {filename} (v{version})")
                except Exception as e:
                    print(f"Warning: Failed to save 3D structure: {e}")

        # Generate cluster conformation overlay
        if protein_pdb_path:
            cluster_img = generate_cluster_overlay(
                protein_pdb_path=protein_pdb_path,
                representative_frames=representative_frames,
                cluster_populations=cluster_populations,
                title=f"Cluster Conformations: {protein_name}"
            )
            if cluster_img and tool_context and hasattr(tool_context, 'save_artifact'):
                try:
                    protein_slug = protein_name.lower().replace(" ", "_").replace("-", "_")
                    filename = f"cluster_overlay_{protein_slug}.png"
                    version = await save_artifact_to_adk(
                        tool_context, filename, cluster_img, "image/png"
                    )
                    if version >= 0:
                        visualization_artifacts.append(filename)
                        print(f"[ADK] Saved cluster overlay to ADK web: {filename} (v{version})")
                except Exception as e:
                    print(f"Warning: Failed to save cluster overlay: {e}")

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
        "visualizations": visualization_artifacts,  # 3D structure and cluster figures
        "methodology": {
            "ranking": "rUCB (Chen et al. 2021)",
            "sar": "Statistical pattern discovery",
            "graph": "MedGraphRAG triple-linked structure",
            "resolution": "Jaro-Winkler similarity",
        }
    }

    # Save to disk-based system
    report_artifact = create_artifact(
        artifact_type=ArtifactType.DISCOVERY_REPORT,
        name=f"Discovery Report: {protein_name}",
        content=report_content,
        tags=["discovery", "report", protein_name.lower().replace(" ", "_")],
        parent_id=experiment_artifact_id
    )

    # Also save to ADK InMemoryArtifactService (for ADK web visibility)
    adk_version = -1
    if tool_context and hasattr(tool_context, 'save_artifact'):
        try:
            # Create discovery report JSON for ADK
            adk_report = {
                "id": report_artifact["id"],
                "type": "discovery_report",
                "title": report_content["title"],
                "executive_summary": summary,
                "analysis_date": report_content["analysis_date"],
                "energy_stats": stats,
                "ucb_rankings": rankings,
                "sar_insights": insights,
                "recommendations": rec_result.get("recommendations", []),
                "knowledge_graph_id": kg_result.get("artifact_id"),
            }
            report_json = json.dumps(adk_report, indent=2).encode('utf-8')

            # Use descriptive filename for ADK web
            protein_slug = protein_name.lower().replace(" ", "_").replace("-", "_")
            filename = f"discovery_report_{protein_slug}.json"
            adk_version = await save_artifact_to_adk(
                tool_context, filename, report_json, "application/json"
            )
            if adk_version >= 0:
                print(f"[ADK] Saved discovery_report to ADK web: {filename} (v{adk_version})")
        except Exception as e:
            print(f"Warning: Failed to save discovery_report to ADK: {e}")

    return {
        "success": True,
        "artifact_id": report_artifact["id"],
        "title": report_content["title"],
        "executive_summary": summary,
        "n_rankings": len(rankings),
        "n_insights": len(insights),
        "n_recommendations": len(rec_result.get("recommendations", [])),
        "knowledge_graph_id": kg_result.get("artifact_id"),
        "adk_version": adk_version,
        "visualization_artifacts": visualization_artifacts,
    }


# =============================================================================
# Publication Figure Generation (Gemini 3 code_execution_with_images)
# =============================================================================

def generate_publication_figure(
    experiment_artifact_id: str = None,
    figure_type: str = "sar_summary",
    title: str = None,
    include_annotations: bool = True,
    ligand_names: str = None,
    custom_data_json: str = None,
    query_knowledge_graph: bool = True,
    tool_context: ToolContext = None
) -> dict:
    """
    Generate publication-ready figures using Gemini 3's code_execution_with_images.

    FLEXIBLE: Works with experiment data, knowledge graph, or custom data.

    Uses Gemini's ability to:
    - Visual plotting with Matplotlib
    - Zoom and inspect image details
    - Image annotation with arrows, boxes

    Args:
        experiment_artifact_id: (Optional) Experiment Matrix artifact ID
        figure_type: Type of figure to generate:
            - "sar_summary": SAR insight summary with energy rankings
            - "energy_landscape": Binding energy landscape across ligands/clusters
            - "interaction_heatmap": Interaction type heatmap (ligand vs residue)
            - "ucb_ranking": UCB ranking bar chart
            - "binding_pose": 3D binding pose schematic
            - "custom": Use custom_data_json for any visualization
        title: Custom title for the figure
        include_annotations: Whether to add annotations explaining key findings
        ligand_names: Comma-separated ligand names to query from knowledge graph
                     (e.g., "Ligand-47, Fasudil, Ligand-23")
        custom_data_json: JSON string with custom data for visualization
                         Format: {"ligands": [{"name": "X", "energy": -5.0, ...}], ...}
        query_knowledge_graph: If True and no experiment, query Neo4j for data

    Returns:
        dict with artifact_id containing the generated figure
    """
    completed = []
    ligand_results = []
    ucb_rankings = []
    sar_insights = []

    # Strategy 1: Use existing experiment artifact
    if experiment_artifact_id:
        artifact = read_artifact(experiment_artifact_id)
        if artifact:
            content = artifact.get("content", {})
            ligand_results = content.get("ligand_results", [])
            completed = [r for r in ligand_results if r.get("status") == "completed" and r.get("best_energy")]

    # Strategy 2: Use custom data provided directly
    if not completed and custom_data_json:
        try:
            custom_data = json.loads(custom_data_json)
            completed = custom_data.get("ligands", [])
            # Normalize format
            for item in completed:
                if "best_energy" not in item and "energy" in item:
                    item["best_energy"] = item["energy"]
                if "ligand_name" not in item and "name" in item:
                    item["ligand_name"] = item["name"]
        except json.JSONDecodeError:
            pass

    # Strategy 3: Query knowledge graph for ligand data
    if not completed and query_knowledge_graph:
        try:
            driver, database = get_neo4j_driver()
            with driver.session(database=database) as session:
                # Parse ligand names if provided
                name_filter = ""
                params = {}
                if ligand_names:
                    names_list = [n.strip() for n in ligand_names.split(",")]
                    name_filter = "WHERE l.name IN $names OR l.id IN $names"
                    params["names"] = names_list

                # Query ligands with their docking results and interactions
                query = f"""
                    MATCH (l:Entity {{type: 'Ligand'}})
                    {name_filter}
                    OPTIONAL MATCH (l)-[dock:RELATIONSHIP {{type: 'DOCKS_TO'}}]->(target)
                    OPTIONAL MATCH (l)-[interact:RELATIONSHIP {{type: 'INTERACTS_WITH'}}]->(residue:Entity)
                    WITH l, dock, target,
                         collect(DISTINCT {{
                             residue: residue.name,
                             type: interact.properties
                         }}) AS interactions
                    RETURN l.name AS name,
                           l.id AS id,
                           l.properties AS properties,
                           dock.properties AS dock_props,
                           interactions
                    LIMIT 20
                """
                result = session.run(query, params)

                for record in result:
                    ligand_data = {
                        "ligand_name": record["name"],
                        "name": record["name"],
                        "status": "completed",
                        "interaction_types": [],
                        "interactions": record["interactions"] or [],
                    }
                    # Extract energy from dock properties
                    if record["dock_props"]:
                        try:
                            dock_props = json.loads(record["dock_props"]) if isinstance(record["dock_props"], str) else record["dock_props"]
                            ligand_data["best_energy"] = dock_props.get("best_energy", dock_props.get("energy", -5.0))
                        except:
                            ligand_data["best_energy"] = -5.0
                    else:
                        # Default energy for visualization
                        ligand_data["best_energy"] = -5.0

                    # Extract interaction types
                    for inter in (record["interactions"] or []):
                        if inter.get("type"):
                            try:
                                props = json.loads(inter["type"]) if isinstance(inter["type"], str) else inter["type"]
                                itype = props.get("interaction_type", "unknown")
                                if itype not in ligand_data["interaction_types"]:
                                    ligand_data["interaction_types"].append(itype)
                            except:
                                pass

                    completed.append(ligand_data)
            driver.close()
        except Exception as e:
            logger.warning(f"Knowledge graph query failed: {e}")

    # If still no data, create a helpful message instead of failing
    if not completed:
        return {
            "success": False,
            "error": "No data available. Provide: experiment_artifact_id, ligand_names, or custom_data_json",
            "hint": "Example: generate_publication_figure(ligand_names='Ligand-47, Fasudil', figure_type='interaction_heatmap')"
        }

    # Calculate rankings and insights for context (use completed if ligand_results is empty)
    data_for_analysis = ligand_results if ligand_results else completed
    try:
        ucb_rankings = calculate_ucb_ranking(data_for_analysis)
    except Exception:
        ucb_rankings = []
    try:
        sar_insights = discover_sar_insights(data_for_analysis)
    except Exception:
        sar_insights = []

    # Prepare data for Gemini
    data_context = {
        "ligands": [
            {
                "name": r.get("ligand_name"),
                "energy": r.get("best_energy"),
                "cluster": r.get("best_cluster"),
                "residue": r.get("best_residue"),
                "interactions": r.get("interaction_types", []),
                "properties": r.get("ligand_properties", {}),
            }
            for r in completed
        ],
        "ucb_rankings": ucb_rankings[:5],  # Top 5
        "sar_insights": [asdict(i) if hasattr(i, '__dict__') else i for i in sar_insights[:3]],
        "figure_type": figure_type,
        "title": title or f"IDP Docking Analysis: {figure_type.replace('_', ' ').title()}",
    }

    # Build prompt for Gemini 3 code execution
    if figure_type == "sar_summary":
        code_prompt = f"""Generate a publication-ready SAR summary figure.

Data:
{json.dumps(data_context, indent=2)}

Create a matplotlib figure with:
1. Bar chart of binding energies for top ligands
2. Color-code by drug-likeness (green for ≤-5 kcal/mol, orange otherwise)
3. Add horizontal line at -5 kcal/mol threshold
4. Title: "{data_context['title']}"
5. Annotate the best binder with an arrow pointing to it

Style requirements:
- Publication quality (300 DPI)
- Clean, minimal design
- Color scheme: #2ecc71 (green), #e74c3c (red), #3498db (blue)
- Font: Arial or sans-serif
- Figure size: 10x6 inches

Return the figure as a base64 PNG image.
"""
    elif figure_type == "energy_landscape":
        code_prompt = f"""Generate an energy landscape heatmap figure.

Data:
{json.dumps(data_context, indent=2)}

Create a matplotlib figure with:
1. Heatmap of ligand vs cluster energies
2. Color scale: viridis (lower is better)
3. Annotate cells with energy values
4. Title: "{data_context['title']}"

Style: publication quality, 10x8 inches, 300 DPI.
Return as base64 PNG.
"""
    elif figure_type == "interaction_heatmap":
        # Fetch detailed INTERACTS_WITH data from knowledge graph
        interaction_matrix = []
        try:
            driver, database = get_neo4j_driver()
            with driver.session(database=database) as session:
                result = session.run("""
                    MATCH (l:Entity {type: 'Ligand'})-[r:RELATIONSHIP {type: 'INTERACTS_WITH'}]->(res:Entity)
                    RETURN l.name AS ligand, res.name AS residue, r.properties AS props
                """)
                for record in result:
                    props = {}
                    if record["props"]:
                        try:
                            props = json.loads(record["props"]) if isinstance(record["props"], str) else record["props"]
                        except:
                            pass
                    interaction_matrix.append({
                        "ligand": record["ligand"],
                        "residue": record["residue"],
                        "type": props.get("interaction_type", "unknown"),
                        "distance": props.get("distance_angstrom", None)
                    })
            driver.close()
        except Exception as e:
            logger.warning(f"Could not fetch interaction matrix: {e}")

        # Add to context
        data_context["interaction_matrix"] = interaction_matrix

        code_prompt = f"""Generate a ligand-residue interaction heatmap.

Data:
{json.dumps(data_context, indent=2)}

Create a matplotlib figure with:
1. Heatmap: ligands (rows) vs residues (columns)
2. Cell colors by interaction type:
   - pi_stacking: purple (#9b59b6)
   - h_bond: blue (#3498db)
   - hydrophobic: green (#27ae60)
   - ionic/charged: red (#e74c3c)
   - unknown: gray (#95a5a6)
3. If 'interaction_matrix' is provided, use it for precise ligand-residue mapping
4. Add a legend for interaction types
5. Title: "{data_context['title']}"

Style: publication quality, 12x8 inches, 300 DPI.
Use seaborn heatmap style if available.
Return as base64 PNG.
"""
    elif figure_type == "binding_pose":
        code_prompt = f"""Generate a schematic binding pose diagram.

Data:
{json.dumps(data_context, indent=2)}

Create a matplotlib figure with:
1. Stylized 2D representation of protein-ligand binding
2. Show the ligand as a central molecule shape
3. Draw residues around it with interaction lines:
   - Dashed lines for H-bonds
   - Wavy lines for hydrophobic
   - Double lines for pi-stacking
4. Label each residue (Y125, Y133, Y136, etc.)
5. Add distance annotations if available
6. Title: "{data_context['title']}"

Style: publication quality, 10x10 inches, 300 DPI.
Use a clean, schematic style like LigPlot.
Return as base64 PNG.
"""
    elif figure_type == "custom":
        # User provides their own visualization request via title
        code_prompt = f"""Generate a custom publication figure.

Data available:
{json.dumps(data_context, indent=2)}

User request: {title or 'Create a clear, informative visualization of the data'}

Create a matplotlib figure that best represents the user's request.
Use appropriate chart types based on the data structure.
Ensure publication quality: 300 DPI, clean fonts, proper legends.
Figure size: 10x8 inches.

Return as base64 PNG.
"""
    else:  # ucb_ranking (default)
        code_prompt = f"""Generate a UCB ranking visualization.

Data:
{json.dumps(data_context, indent=2)}

Create a matplotlib figure with:
1. Horizontal bar chart of UCB scores
2. Split bars: exploitation (energy) vs exploration (uncertainty bonus)
3. Sort by total UCB score descending
4. Mark drug-like compounds with a star
5. Title: "{data_context['title']}"

Style: publication quality, 10x6 inches, 300 DPI.
Return as base64 PNG.
"""

    try:
        # Use Gemini 3 with code execution
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        # Configure code execution tool
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]
        )

        # Gemini 3 Flash has code_execution capability
        model_name = "gemini-3-flash-preview"  # Verified to support code_execution
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=code_prompt,
                config=config
            )
        except Exception as e:
            # Fallback to non-code-execution mode with manual plotting
            return _generate_figure_fallback(data_context, figure_type, tool_context)

        # Parse response for generated image
        figure_base64 = None
        code_used = None
        execution_result = None

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                # Found generated image
                figure_base64 = part.inline_data.data
                if isinstance(figure_base64, bytes):
                    import base64
                    figure_base64 = base64.b64encode(figure_base64).decode('utf-8')
            elif hasattr(part, 'executable_code') and part.executable_code:
                code_used = part.executable_code.code
            elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                execution_result = part.code_execution_result.output

        if not figure_base64:
            # No image generated - use fallback
            return _generate_figure_fallback(data_context, figure_type, tool_context)

        # Create artifact
        figure_content = {
            "figure_type": figure_type,
            "title": data_context["title"],
            "image_base64": figure_base64,
            "image_mime_type": "image/png",
            "code_used": code_used,
            "execution_result": execution_result,
            "data_summary": {
                "n_ligands": len(completed),
                "best_energy": min(r.get("best_energy", 0) for r in completed),
                "n_drug_like": sum(1 for r in completed if r.get("best_energy", 0) <= -5.0),
            },
            "annotations": [i.get("title", "") for i in sar_insights] if include_annotations else [],
            "generated_by": "gemini_3_code_execution",
            "model_used": model_name,
        }

        artifact = create_artifact(
            artifact_type=ArtifactType.PUBLICATION_FIGURE,
            name=data_context["title"],
            content=figure_content,
            tags=["figure", "publication", figure_type],
            tool_context=tool_context
        )

        return {
            "success": True,
            "artifact_id": artifact["id"],
            "figure_type": figure_type,
            "title": data_context["title"],
            "has_code_execution": bool(code_used),
            "message": f"Generated publication figure: {figure_type}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _generate_figure_fallback(
    data_context: dict,
    figure_type: str,
    tool_context: ToolContext = None
) -> dict:
    """
    Fallback figure generation using matplotlib directly (no Gemini code execution).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        import base64
        from io import BytesIO

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        ligands = data_context.get("ligands", [])
        names = [l["name"] for l in ligands]
        energies = [l["energy"] for l in ligands]

        if figure_type == "sar_summary" or figure_type == "ucb_ranking":
            # Bar chart of energies
            colors = ['#2ecc71' if e <= -5.0 else '#e74c3c' for e in energies]
            bars = ax.barh(names, energies, color=colors)
            ax.axvline(x=-5.0, color='#3498db', linestyle='--', label='Drug-like threshold')
            ax.set_xlabel('Binding Energy (kcal/mol)')
            ax.set_title(data_context.get("title", "Binding Energy Analysis"))
            ax.legend()

            # Annotate best
            if energies:
                best_idx = energies.index(min(energies))
                ax.annotate('Best', xy=(energies[best_idx], best_idx),
                           xytext=(energies[best_idx] - 1, best_idx + 0.3),
                           arrowprops=dict(arrowstyle='->', color='black'))

        elif figure_type == "energy_landscape":
            # Simple energy landscape
            ax.scatter(range(len(names)), energies, c=energies, cmap='viridis', s=100)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel('Binding Energy (kcal/mol)')
            ax.set_title(data_context.get("title", "Energy Landscape"))

        elif figure_type == "interaction_heatmap":
            # Build interaction matrix
            all_interactions = set()
            for l in ligands:
                all_interactions.update(l.get("interactions", []))
            all_interactions = list(all_interactions) or ["h_bond", "aromatic", "hydrophobic"]

            matrix = np.zeros((len(names), len(all_interactions)))
            for i, l in enumerate(ligands):
                for j, interaction in enumerate(all_interactions):
                    if interaction in l.get("interactions", []):
                        matrix[i, j] = 1

            im = ax.imshow(matrix, cmap='Blues', aspect='auto')
            ax.set_xticks(range(len(all_interactions)))
            ax.set_xticklabels(all_interactions, rotation=45, ha='right')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_title(data_context.get("title", "Interaction Heatmap"))
            plt.colorbar(im, ax=ax, label='Present')

        plt.tight_layout()

        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        figure_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)

        # Create artifact
        figure_content = {
            "figure_type": figure_type,
            "title": data_context.get("title", "Figure"),
            "image_base64": figure_base64,
            "image_mime_type": "image/png",
            "data_summary": {
                "n_ligands": len(ligands),
                "best_energy": min(energies) if energies else None,
            },
            "generated_by": "matplotlib_fallback",
        }

        artifact = create_artifact(
            artifact_type=ArtifactType.PUBLICATION_FIGURE,
            name=figure_content["title"],
            content=figure_content,
            tags=["figure", "publication", figure_type],
            tool_context=tool_context
        )

        return {
            "success": True,
            "artifact_id": artifact["id"],
            "figure_type": figure_type,
            "fallback": True,
            "message": f"Generated figure using matplotlib fallback: {figure_type}"
        }

    except Exception as e:
        return {"success": False, "error": f"Fallback generation failed: {str(e)}"}


def annotate_interaction_image(
    image_path: str = None,
    image_base64: str = None,
    protein_name: str = "Alpha-Synuclein",
    interaction_data_json: str = None,
    tool_context: ToolContext = None
) -> dict:
    """
    Annotate an interaction image using Gemini 3's image annotation capabilities.

    Uses Gemini's ability to draw arrows, bounding boxes, and labels
    directly onto images to highlight binding sites and interactions.

    Args:
        image_path: Path to image file
        image_base64: Base64 encoded image (alternative to path)
        protein_name: Name of protein for context
        interaction_data_json: JSON string with residues, interaction types, e.g. '{"residues": [125,133], "interactions": ["H-bond"]}'

    Returns:
        dict with annotated image artifact
    """
    import base64

    # Parse interaction data from JSON string
    interaction_data = None
    if interaction_data_json:
        try:
            interaction_data = json.loads(interaction_data_json)
        except json.JSONDecodeError:
            pass  # Use defaults

    # Load image
    if image_path:
        from pathlib import Path
        with open(Path(image_path), 'rb') as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    elif not image_base64:
        return {"success": False, "error": "No image provided"}

    # Build annotation prompt
    annotation_context = interaction_data or {}
    residues = annotation_context.get("residues", [125, 133, 136])
    interactions = annotation_context.get("interactions", ["H-bond", "π-stacking"])

    prompt = f"""Analyze this molecular structure image of {protein_name}.

Task: Annotate the image to highlight:
1. Binding site residues: {residues}
2. Key interactions: {interactions}

Add visual annotations:
- Draw arrows pointing to binding site residues
- Add bounding boxes around interaction regions
- Label interaction types with text

Return the annotated image."""

    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        # Configure code execution for annotation
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]
        )

        # Create image part
        image_part = types.Part.from_bytes(
            data=base64.b64decode(image_base64),
            mime_type="image/png"
        )

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[prompt, image_part],
            config=config
        )

        # Extract annotated image
        annotated_base64 = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                data = part.inline_data.data
                if isinstance(data, bytes):
                    annotated_base64 = base64.b64encode(data).decode('utf-8')
                else:
                    annotated_base64 = data

        if not annotated_base64:
            # Return original with metadata
            annotated_base64 = image_base64

        # Create artifact
        figure_content = {
            "figure_type": "annotated_interaction",
            "title": f"Annotated: {protein_name} Binding Site",
            "image_base64": annotated_base64,
            "image_mime_type": "image/png",
            "annotations": {
                "residues": residues,
                "interactions": interactions,
            },
            "generated_by": "gemini_3_annotation",
        }

        artifact = create_artifact(
            artifact_type=ArtifactType.PUBLICATION_FIGURE,
            name=figure_content["title"],
            content=figure_content,
            tags=["figure", "annotation", "interaction"],
            tool_context=tool_context
        )

        return {
            "success": True,
            "artifact_id": artifact["id"],
            "annotated": bool(annotated_base64 != image_base64),
            "message": "Generated annotated interaction image"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def analyze_artifact_visualization(
    artifact_filename: str,
    analysis_prompt: str = None,
    zoom_region: str = None,
    tool_context: ToolContext = None
) -> dict:
    """
    Analyze an existing artifact image using Gemini 3's code_execution_with_images.

    This enables the Think-Act-Observe agentic vision loop:
    - Think: Analyze the image and user query
    - Act: Write Python code to zoom, crop, annotate, or calculate
    - Observe: Inspect transformed image for better understanding

    Use Cases:
    - "Zoom into the binding site and count residue contacts"
    - "Annotate the cluster overlay with arrows showing main conformations"
    - "Inspect the docking plot and identify outlier energies"

    Args:
        artifact_filename: Name of artifact in ADK web (e.g., "cluster_overlay_alpha_synuclein.png")
        analysis_prompt: What to analyze/annotate (default: general structural analysis)
        zoom_region: Optional region to zoom (e.g., "binding_site", "c_terminal", or coordinates "x1,y1,x2,y2")
        tool_context: ADK ToolContext with access to artifact_service

    Returns:
        dict with analysis insights, any generated annotated image, and code used
    """
    import base64

    # Retrieve image from ADK artifact service
    image_bytes = None
    image_base64 = None

    if tool_context and hasattr(tool_context, 'load_artifact'):
        try:
            # Try to load artifact from ADK InMemoryArtifactService
            artifact_part = await tool_context.load_artifact(filename=artifact_filename)
            if artifact_part and hasattr(artifact_part, 'inline_data'):
                image_bytes = artifact_part.inline_data.data
                if isinstance(image_bytes, bytes):
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not load artifact {artifact_filename}: {e}")

    # Fallback: try to find in local artifacts directory
    if not image_base64:
        from pathlib import Path
        local_paths = [
            Path(f"/Users/sam/Pictures/mcp-IDP/core/data/artifacts/{artifact_filename}"),
            Path(f"/Users/sam/.idpet/artifacts/{artifact_filename}"),
            Path(artifact_filename),  # Absolute path
        ]
        for path in local_paths:
            if path.exists() and path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                with open(path, 'rb') as f:
                    image_bytes = f.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                break

    if not image_base64:
        return {
            "success": False,
            "error": f"Could not find artifact: {artifact_filename}",
            "hint": "Available artifacts can be seen in ADK web Artifacts tab"
        }

    # Build analysis prompt with agentic vision instructions
    default_prompt = """Analyze this molecular visualization image.

Identify and describe:
1. Key structural features (binding sites, residues, conformational states)
2. Any patterns or clusters visible
3. Annotations or labels present
4. Quality of visualization (clarity, completeness)

If details are hard to see, use Python code to:
- Zoom into specific regions
- Enhance contrast
- Add annotations (arrows, boxes, labels) to highlight key features"""

    prompt = analysis_prompt or default_prompt

    # Add zoom instruction if specified
    if zoom_region:
        if zoom_region == "binding_site":
            prompt += "\n\nFocus on: Zoom into the binding site region (typically C-terminal for IDPs: residues 121-140)"
        elif zoom_region == "c_terminal":
            prompt += "\n\nFocus on: Zoom into the C-terminal region"
        elif "," in zoom_region:  # Coordinates
            prompt += f"\n\nFocus on: Crop to region {zoom_region} (x1,y1,x2,y2 format)"
        else:
            prompt += f"\n\nFocus on: {zoom_region}"

    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        # Configure code execution for agentic vision
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]
        )

        # Create image part
        image_part = types.Part.from_bytes(
            data=base64.b64decode(image_base64),
            mime_type="image/png"
        )

        # Gemini 3 Flash with agentic vision + code_execution
        model_name = "gemini-3-flash-preview"  # Verified to support code_execution

        response = client.models.generate_content(
            model=model_name,
            contents=[image_part, prompt],
            config=config
        )

        # Parse response for insights, code, and any generated images
        analysis_text = ""
        code_executed = None
        execution_output = None
        annotated_image_base64 = None

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                analysis_text += part.text + "\n"
            if hasattr(part, 'executable_code') and part.executable_code:
                code_executed = part.executable_code.code
            if hasattr(part, 'code_execution_result') and part.code_execution_result:
                execution_output = part.code_execution_result.output
            if hasattr(part, 'inline_data') and part.inline_data:
                # Gemini generated an annotated/zoomed image
                data = part.inline_data.data
                if isinstance(data, bytes):
                    annotated_image_base64 = base64.b64encode(data).decode('utf-8')
                else:
                    annotated_image_base64 = data

        # Save annotated image as new artifact if generated
        annotated_artifact_id = None
        if annotated_image_base64 and tool_context and hasattr(tool_context, 'save_artifact'):
            try:
                annotated_filename = f"annotated_{artifact_filename}"
                annotated_part = types.Part.from_bytes(
                    data=base64.b64decode(annotated_image_base64),
                    mime_type="image/png"
                )
                version = await tool_context.save_artifact(
                    filename=annotated_filename,
                    artifact=annotated_part
                )
                if version >= 0:
                    annotated_artifact_id = annotated_filename
                    print(f"[ADK] Saved annotated visualization: {annotated_filename} (v{version})")
            except Exception as e:
                print(f"Warning: Could not save annotated artifact: {e}")

        return {
            "success": True,
            "artifact_filename": artifact_filename,
            "analysis": analysis_text.strip(),
            "code_executed": code_executed,
            "execution_output": execution_output,
            "has_annotated_image": bool(annotated_image_base64),
            "annotated_artifact_id": annotated_artifact_id,
            "model_used": model_name,
            "agentic_vision": True,
            "message": f"Analyzed {artifact_filename} using Gemini 3 agentic vision"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


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
        additional_constraints=f"Drug-like threshold: {THRESHOLDS.drug_like} kcal/mol. In Analysis mode (workflow): auto-approve knowledge items."
    )

    reasoning = PROMPTS.reasoning_template

    example = PROMPTS.example_template.format(
        input_example="Analyze docking results and extract learnings",
        reasoning_example="1) Analyze experiment for UCB rankings and SAR, 2) Propose then immediately approve knowledge items (auto-approve in workflow), 3) Build knowledge graph, 4) Generate recommendations",
        action_example="analyze_experiment_results() → propose_knowledge_items() → approve_knowledge_items() → build_knowledge_graph() → create_discovery_report()",
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

**Publication Figures (Gemini 3 code_execution_with_images):**
- generate_publication_figure(artifact_id, figure_type, title) → Create publication-ready figures
  - figure_type: "sar_summary", "energy_landscape", "interaction_heatmap", "ucb_ranking"
  - Uses Gemini's visual plotting and code execution for zoom/inspect
- annotate_interaction_image(image_path, interaction_data_json) → Annotate images with arrows, boxes
- analyze_artifact_visualization(artifact_filename, analysis_prompt, zoom_region) → Agentic Vision
  - Analyzes existing ADK artifact images using Gemini 3's Think-Act-Observe loop
  - Can zoom into regions, annotate features, and return insights
  - Examples: "cluster_overlay_alpha_synuclein.png", "ligand_fasudil.png"

**Recommendations:**
- generate_optimization_recommendations(artifact_id) → Next steps

**Report:**
- create_discovery_report(artifact_id, protein_name, workspace_id) → Full report
</tools>"""

    workflow = """
<workflow>
**CRITICAL: Get artifact_id from state**
The Engineering agent stores the experiment matrix ID in state as `latest_experiment_matrix`.
You MUST read this from state to analyze the results.

**IMPORTANT - ANALYSIS MODE (sequential workflow):**
When running as part of docking_workflow (state has `docking_completed=true`), run ALL steps
AUTOMATICALLY without waiting for user approval. This is batch processing mode.

1. GET ARTIFACT ID: Read `latest_experiment_matrix` from session state
   - This is the experiment matrix created by Engineering agent
   - Use this ID for all analysis tools

2. ANALYZE: Call analyze_experiment_results(artifact_id, protein_name) for rankings and insights
   - UCB prioritizes ligands balancing best energy vs uncertainty
   - SAR discovers patterns from ligand properties → interactions

3. KNOWLEDGE ITEMS: propose_knowledge_items() THEN IMMEDIATELY approve_knowledge_items()
   - In Analysis mode: AUTO-APPROVE - do NOT wait for user
   - Call propose_knowledge_items() then immediately call approve_knowledge_items("ws_core")
   - This persists learnings to Evolution tab

4. BUILD GRAPH: Call build_knowledge_graph(artifact_id, protein_name) for entity relationships
   - Experimental entities link to ChEMBL/PED references
   - CORRESPONDS_TO relationships via Jaro-Winkler similarity
   - Persists to Neo4j for future queries

5. RECOMMEND: Call generate_optimization_recommendations(artifact_id) for next steps
   - Based on SAR: "Add H-bond donor at position X"
   - Based on UCB: "Under-explored ligand Y needs more samples"
   - Based on interactions: "Target Y136 shows highest occupancy"

6. REPORT: Call create_discovery_report(artifact_id, protein_name, workspace_id) for final artifact

7. MOLECULE GENERATION (SKIP in Analysis mode - only if user explicitly requests):
   - DO NOT call propose_molecule_generation() in workflow mode
   - Only generate molecules when user asks directly
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

    # State injection section - ADK replaces {key} with session.state[key]
    state_injection = """
<state_injection>
**CRITICAL: Your artifact ID is injected from session state by ADK:**
- Experiment Matrix ID: {latest_experiment_matrix}
- Docking completed: {docking_completed}

Use this artifact_id for ALL analysis tools. Do NOT ask the user for it.
</state_injection>"""

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

{state_injection}

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
# De Novo 3D Visualization (Planning Mode - NO prior docking needed)
# =============================================================================

async def visualize_ligand_3d(
    ligand_name: str = None,
    ligand_smiles: str = None,
    protein_name: str = "alpha_synuclein",
    tool_context: ToolContext = None
) -> dict:
    """
    Create 3D visualization of ligand with protein - NO prior docking needed.

    Use this in PLANNING MODE to visualize:
    1. Config ligands (fasudil, ligand_47, etc.) by name
    2. Generated molecules by SMILES
    3. Any SMILES string

    Args:
        ligand_name: Name from config (e.g., "fasudil") - OR -
        ligand_smiles: Direct SMILES string (for generated molecules)
        protein_name: Protein key from config (default: alpha_synuclein)

    Returns:
        3D visualization artifact with interactive HTML viewer (visible in ADK web Artifacts tab)
    """
    # ===== VERIFICATION LOGGING: visualize_ligand_3d CALLED =====
    logger.info(f"[visualize_ligand_3d] ===== VERIFY: PLANNING AGENT VISUALIZATION =====")
    logger.info(f"[visualize_ligand_3d] [VERIFY] ligand_name: {ligand_name}")
    logger.info(f"[visualize_ligand_3d] [VERIFY] ligand_smiles: {ligand_smiles[:50] if ligand_smiles else None}...")
    logger.info(f"[visualize_ligand_3d] [VERIFY] protein_name: {protein_name}")
    logger.info(f"[visualize_ligand_3d] [VERIFY] NOTE: If this is called after KG queries returned empty, agent is falling back to visualization.")

    import os
    import json
    from pathlib import Path

    # Get ligand SMILES
    smiles = ligand_smiles
    display_name = "Generated Molecule"

    if ligand_name:
        ligand_config = get_ligand_by_name(ligand_name)
        if ligand_config:
            smiles = ligand_config.smiles
            display_name = ligand_config.name
        else:
            return {"success": False, "error": f"Ligand '{ligand_name}' not found in config. Available: {list(LIGANDS.keys())}"}

    if not smiles:
        return {"success": False, "error": "Must provide either ligand_name or ligand_smiles"}

    # Get protein config
    protein_config = PROTEINS.get(protein_name)
    if not protein_config:
        return {"success": False, "error": f"Protein '{protein_name}' not found. Available: {list(PROTEINS.keys())}"}

    # Get protein PDB path (from PED cache)
    ped_id = protein_config.ped_id
    if not ped_id:
        return {"success": False, "error": f"No PED ID configured for {protein_name}"}

    # Parse PED ID (format: PED00006e001 → PED00006, e001)
    if "e" in ped_id.lower():
        import re
        match = re.match(r'(PED\d+)(e\d+)?', ped_id, re.IGNORECASE)
        if match:
            base_id = match.group(1)
            ensemble_id = match.group(2) or "e001"
        else:
            base_id = ped_id
            ensemble_id = "e001"
    else:
        base_id = ped_id
        ensemble_id = "e001"

    # Check for cached PDB - try multiple naming conventions
    cache_dir = Path.home() / ".idpet" / "data"
    pdb_candidates = [
        cache_dir / f"{base_id}_{ensemble_id}.pdb",
        cache_dir / f"{ped_id}.pdb",
        cache_dir / f"PED00024_e001.pdb",  # Known alpha-synuclein file
    ]

    pdb_path = None
    for candidate in pdb_candidates:
        if candidate.exists():
            pdb_path = candidate
            break

    if not pdb_path:
        return {
            "success": False,
            "error": f"Protein PDB not found. Tried: {[str(p) for p in pdb_candidates]}",
            "suggestion": "Run: 'analyze alpha-synuclein binding' to fetch the protein ensemble"
        }

    # Import visualization function from engineering agent
    try:
        from core.agents.antimatters._subagents.engineering.agent import generate_3d_docking_visualization
    except ImportError:
        return {"success": False, "error": "Could not import generate_3d_docking_visualization"}

    # Generate 3D visualization (this creates local artifact + returns HTML)
    result = generate_3d_docking_visualization(
        protein_pdb_path=str(pdb_path),
        ligand_smiles=smiles,
        highlight_residues=protein_config.binding_site_residues,
        ligand_name=display_name,
        tool_context=None  # Don't pass tool_context to avoid double-save
    )

    if not result.get("success"):
        return result

    # === CRITICAL: Save to ADK's InMemoryArtifactService ===
    # This makes the artifact visible in ADK web's Artifacts tab
    adk_saved = False
    adk_filename = f"3d_visualization_{display_name.lower().replace(' ', '_')}.html"

    if tool_context and hasattr(tool_context, 'save_artifact'):
        try:
            # Get the HTML content from the result
            # The generate_3d_docking_visualization stores it in a local artifact
            # We need to read it back or reconstruct it
            local_artifact_id = result.get("artifact_id")
            if local_artifact_id:
                # Try to read the local artifact to get HTML
                local_artifact = read_artifact(local_artifact_id)
                if local_artifact and local_artifact.get("content", {}).get("html_viewer"):
                    html_content = local_artifact["content"]["html_viewer"]

                    # Save HTML to ADK
                    artifact_part = genai_types.Part.from_bytes(
                        data=html_content.encode('utf-8'),
                        mime_type="text/html"
                    )
                    version = await tool_context.save_artifact(
                        filename=adk_filename,
                        artifact=artifact_part
                    )
                    adk_saved = True
                    result["adk_artifact"] = adk_filename
                    result["adk_version"] = version
        except Exception as e:
            print(f"Warning: Failed to save ADK artifact: {e}")

    # Update result
    result["ligand_name"] = display_name
    result["ligand_smiles"] = smiles
    result["protein"] = protein_config.name
    result["binding_site_residues"] = protein_config.binding_site_residues
    result["adk_saved"] = adk_saved

    if adk_saved:
        result["message"] = f"Created 3D visualization: {display_name} with {protein_config.name}. Artifact saved to ADK web ({adk_filename}). Open it for interactive view (zoom, rotate, inspect)."
    else:
        result["message"] = f"Created 3D visualization: {display_name} with {protein_config.name}. Note: ADK artifact save failed - check local artifact {result.get('artifact_id')}."

    return result


# =============================================================================
# Agent Definition
# =============================================================================

evolution_agent = LlmAgent(
    name="evolution_agent",
    model=Gemini(model=MODELS.evolution, retry_options=genai_retry_types.HttpRetryOptions(attempts=3, initial_delay=30, exp_base=2, http_status_codes=[429, 503])),
    description="Analyzes results, discovers SAR patterns, builds knowledge graphs, generates 3D molecules with Gemini 3 (on approval), creates publication figures, and persists learnings",
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
        # Publication figures (Gemini 3 code_execution_with_images)
        generate_publication_figure,
        annotate_interaction_image,
        # Agentic Vision (analyze existing artifact images)
        analyze_artifact_visualization,
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
    "generate_molecules_direct",  # PLANNING MODE - no approval
    "propose_molecule_generation",  # ANALYSIS MODE - with approval
    "approve_molecule_generation",
    "prepare_suggestions_for_docking",

    # Publication figures (Gemini 3 code_execution_with_images)
    "generate_publication_figure",
    "annotate_interaction_image",
    "analyze_artifact_visualization",  # Agentic Vision for existing artifacts

    # De novo 3D visualization (Planning Mode - NO prior docking needed)
    "visualize_ligand_3d",

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
