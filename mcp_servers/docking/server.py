#!/usr/bin/env python3
"""
MCP Server for Ensemble-Based Molecular Docking
"""

from __future__ import annotations  # Enable string annotations for forward references
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import tempfile
import subprocess
import shutil
import pathlib
import asyncio
import json
from dataclasses import dataclass, asdict

# MCP Framework
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing_extensions import Annotated

# Type checking imports (not loaded at runtime)
if TYPE_CHECKING:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    import mdtraj as md
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from vina import Vina

# Lazy imports for heavy dependencies (loaded on first use)
# This speeds up server startup significantly

# Lazy import flags
_RDKIT_LOADED = False
_MDTRAJ_LOADED = False
_SKLEARN_LOADED = False
_VINA_LOADED = False
_ANALYSIS_LOADED = False

def _load_rdkit():
    global _RDKIT_LOADED, Chem, AllChem, Descriptors
    if not _RDKIT_LOADED:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
        _RDKIT_LOADED = True
    return Chem, AllChem, Descriptors

def _load_mdtraj():
    global _MDTRAJ_LOADED, md
    if not _MDTRAJ_LOADED:
        import mdtraj as md
        _MDTRAJ_LOADED = True
    return md

def _load_sklearn():
    global _SKLEARN_LOADED, TSNE, KMeans, np
    if not _SKLEARN_LOADED:
        import numpy as np
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        _SKLEARN_LOADED = True
    return TSNE, KMeans, np

def _load_vina():
    global _VINA_LOADED, Vina
    if not _VINA_LOADED:
        from vina import Vina
        _VINA_LOADED = True
    return Vina

def _load_analysis():
    global _ANALYSIS_LOADED, hbond, hphob_contacts, aro_contacts, charge_contacts, dual_contact, ANALYSIS_AVAILABLE
    if not _ANALYSIS_LOADED:
        try:
            from trajectory_analysis import (
                hbond, hphob_contacts, aro_contacts, charge_contacts,
                dual_contact
            )
            ANALYSIS_AVAILABLE = True
        except ImportError:
            ANALYSIS_AVAILABLE = False
        _ANALYSIS_LOADED = True
    return ANALYSIS_AVAILABLE

import logging
import sys

# Send all logging to stderr, not stdout (to avoid corrupting MCP JSON-RPC)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("ensemble-docking-server")

WORKSPACE_DIR = pathlib.Path("/tmp/mcp_docking_workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.pdb', '.mol2', '.sdf', '.pdbqt', '.xtc'}

@dataclass
class LigandData:
    ligand_id: str
    smiles: str
    mol2_path: str
    pdbqt_path: str
    num_atoms: int
    num_rotatable_bonds: int
    molecular_weight: float

@dataclass
class ClusterData:
    n_clusters: int
    cluster_assignments: List[int]
    cluster_populations: List[float]
    representative_frames: List[int]
    tsne_coordinates: List[Tuple[float, float]]

class DockingSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.temp_dir = tempfile.mkdtemp(
            prefix=f"mcp_dock_{session_id}_",
            dir=WORKSPACE_DIR
        )
        self.prepared_ligands: Dict[str, LigandData] = {}
        self.clustered_proteins: Dict[str, ClusterData] = {}
        
    def cleanup(self):
        if pathlib.Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

_sessions: Dict[str, DockingSession] = {}

def validate_file_path(filepath: str) -> pathlib.Path:
    path = pathlib.Path(filepath).resolve()
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Invalid file type: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path

def get_residue_com(frame, residue: int) -> Tuple[float, float, float]:
    """Get center of mass for a residue. frame is md.Trajectory"""
    md = _load_mdtraj()
    com = 10 * md.compute_center_of_mass(frame, select=f'resSeq {residue}')
    return com[0, 0].item(), com[0, 1].item(), com[0, 2].item()

def extract_vina_energy(pdbqt_path: str) -> float:
    with open(pdbqt_path, 'r') as f:
        for line in f:
            if line.startswith('REMARK VINA RESULT:'):
                parts = line.split()
                return float(parts[3])
    raise ValueError("Could not extract energy from PDBQT file")


def calculate_box_size(ligand_pdbqt: str) -> float:
    """Calculate box size from ligand radius + padding (eBoxSize equivalent).

    Original research: run_autodockvina.py get_boxsize() line 28-32
    Uses ligand diameter + 10 Å padding (Vina standard).
    """
    np = _load_sklearn()[2]

    coords = []
    with open(ligand_pdbqt, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

    if not coords:
        return 20.0  # Fallback

    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    max_radius = distances.max()

    # Ligand diameter + 10 Å padding (Vina standard)
    box_size = (max_radius * 2) + 10.0
    return max(box_size, 15.0)  # Minimum 15 Å


def apply_cluster_weights(
    cluster_results: List[dict],
    cluster_populations: List[float]
) -> dict:
    """Apply population-weighted averaging to docking results.

    Original research: plotting.py line 124
    con = np.sum(run_data * weights[:, np.newaxis, np.newaxis], axis=0)
    """
    np = _load_sklearn()[2]

    weights = np.array(cluster_populations)
    weights = weights / weights.sum()  # Normalize

    energies = np.array([r["best_energy"] for r in cluster_results if r["best_energy"] is not None])

    if len(energies) == 0:
        return {"weighted_average_energy": None, "unweighted_average_energy": None}

    # Match weights to available energies
    valid_weights = weights[:len(energies)]
    valid_weights = valid_weights / valid_weights.sum()

    weighted_avg = float(np.sum(energies * valid_weights))
    unweighted_avg = float(energies.mean())

    return {
        "weighted_average_energy": weighted_avg,
        "unweighted_average_energy": unweighted_avg,
        "weights_used": valid_weights.tolist(),
    }


def detect_ligand_properties(mol) -> dict:
    """Auto-detect aromatic rings, H-bond donors, charged atoms for interaction analysis.

    These are needed by trajectory_analysis.py functions:
    - aro_contacts(traj, ligand_rings=[[atom_indices]])
    - hbond(traj, ligand_idx, lig_hbond_donors=[[donor, H]])
    - charge_contacts(traj, Ligand_Pos_Charges=[indices])
    """
    # Aromatic rings
    ring_info = mol.GetRingInfo()
    aromatic_rings = []
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            aromatic_rings.append(list(ring))

    # H-bond donors (N-H, O-H pairs)
    hbond_donors = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ['N', 'O']:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    hbond_donors.append([atom.GetIdx(), neighbor.GetIdx()])

    # Charged atoms (formal charges from SMILES)
    pos_charges = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() > 0]
    neg_charges = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() < 0]

    return {
        'aromatic_rings': aromatic_rings,
        'hbond_donors': hbond_donors,
        'pos_charges': pos_charges,
        'neg_charges': neg_charges,
        'n_aromatic_rings': len(aromatic_rings),
        'n_hbond_donors': len(hbond_donors),
        'has_positive_charge': len(pos_charges) > 0,
        'has_negative_charge': len(neg_charges) > 0,
    }

@mcp.tool()
async def prepare_ligand(
    smiles: Annotated[str, Field(description="SMILES string of the ligand molecule")],
    optimize: Annotated[bool, Field(description="Perform UFF energy minimization")] = True,
    session_id: Annotated[str, Field(description="Session identifier")] = "default"
) -> dict:
    """Convert SMILES to 3D mol2 and PDBQT formats."""
    try:
        # Lazy load RDKit
        Chem, AllChem, Descriptors = _load_rdkit()
        
        if session_id not in _sessions:
            _sessions[session_id] = DockingSession(session_id)
        session = _sessions[session_id]
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        mol = Chem.AddHs(mol)
        
        logger.info("Generating 3D conformer with ETKDG...")
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        
        if result == -1:
            raise RuntimeError("Failed to generate 3D conformer")
        
        if optimize:
            AllChem.UFFOptimizeMolecule(mol)
        
        Chem.AssignAtomChiralTagsFromStructure(mol)
        
        ligand_id = f"lig_{abs(hash(smiles)) % 10**8}"
        
        pdb_path = f"{session.temp_dir}/{ligand_id}.pdb"
        Chem.MolToPDBFile(mol, pdb_path)
        
        mol2_path = f"{session.temp_dir}/{ligand_id}.mol2"
        subprocess.run(
            ['obabel', '-ipdb', pdb_path, '-omol2', '-O', mol2_path],
            check=True,
            capture_output=True,
            text=True
        )
        
        pdbqt_path = f"{session.temp_dir}/{ligand_id}.pdbqt"
        subprocess.run([
            'mk_prepare_ligand.py',
            '-i', mol2_path,
            '-o', pdbqt_path,
            '--merge_these_atom_types'
        ], check=True, capture_output=True, text=True)
        
        num_atoms = mol.GetNumAtoms()
        num_rotatable = Descriptors.NumRotatableBonds(mol)
        mol_weight = Descriptors.MolWt(mol)

        # Detect ligand properties for interaction analysis
        # Original research: trajectory_analysis.py needs aromatic rings, H-bond donors, charges
        ligand_properties = detect_ligand_properties(mol)

        ligand_data = LigandData(
            ligand_id=ligand_id,
            smiles=smiles,
            mol2_path=mol2_path,
            pdbqt_path=pdbqt_path,
            num_atoms=num_atoms,
            num_rotatable_bonds=num_rotatable,
            molecular_weight=mol_weight
        )
        session.prepared_ligands[ligand_id] = ligand_data

        logger.info(f"Prepared ligand {ligand_id}: {num_atoms} atoms, {ligand_properties['n_aromatic_rings']} aromatic rings")

        return {
            "success": True,
            "ligand_id": ligand_id,
            "smiles": smiles,
            "mol2_path": mol2_path,
            "pdbqt_path": pdbqt_path,
            "properties": {
                "num_atoms": num_atoms,
                "num_rotatable_bonds": num_rotatable,
                "molecular_weight": mol_weight,
                **ligand_properties,  # Include aromatic_rings, hbond_donors, pos/neg_charges
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": type(e).__name__,
            "details": str(e)
        }

@mcp.tool()
async def cluster_conformations(
    pdb_path: Annotated[str, Field(description="Path to multi-model PDB file")],
    n_clusters: Annotated[int, Field(description="Number of clusters")] = 20,
    perplexity: Annotated[int, Field(description="t-SNE perplexity")] = 30,
    session_id: Annotated[str, Field(description="Session identifier")] = "default"
) -> dict:
    """Cluster protein conformations using t-SNE on pairwise RMSD."""
    try:
        # Validate n_clusters range (temporarily lowered to 2 for testing)
        if not (2 <= n_clusters <= 50):
            raise ValueError(f"n_clusters must be between 2 and 50, got {n_clusters}")

        # Lazy load dependencies
        md = _load_mdtraj()
        TSNE, KMeans, np = _load_sklearn()

        pdb_file = validate_file_path(pdb_path)
        
        if session_id not in _sessions:
            _sessions[session_id] = DockingSession(session_id)
        session = _sessions[session_id]
        
        logger.info(f"Loading ensemble from {pdb_file}...")
        traj = md.load(str(pdb_file))
        protein_traj = traj.atom_slice(traj.top.select('protein'))
        n_frames = protein_traj.n_frames
        
        logger.info(f"Loaded {n_frames} conformations")
        
        if n_frames < n_clusters:
            raise ValueError(f"Need at least {n_clusters} conformations")
        
        logger.info("Computing pairwise RMSD matrix...")
        rmsd_matrix = np.zeros((n_frames, n_frames))
        for i in range(n_frames):
            rmsd_matrix[i] = md.rmsd(protein_traj, protein_traj, frame=i)
        
        logger.info("Running t-SNE clustering...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, n_frames - 1),
            random_state=42,
            n_jobs=-1
        )
        tsne_coords = tsne.fit_transform(rmsd_matrix)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tsne_coords)
        
        cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)
        cluster_populations = cluster_counts / n_frames
        
        representatives = []
        for i in range(n_clusters):
            cluster_mask = (cluster_labels == i)
            cluster_frames = np.where(cluster_mask)[0]
            
            if len(cluster_frames) == 0:
                continue
            
            centroid = tsne_coords[cluster_mask].mean(axis=0)
            distances = np.linalg.norm(
                tsne_coords[cluster_mask] - centroid,
                axis=1
            )
            rep_idx = cluster_frames[distances.argmin()]
            representatives.append(int(rep_idx))
        
        cluster_data = ClusterData(
            n_clusters=n_clusters,
            cluster_assignments=cluster_labels.tolist(),
            cluster_populations=cluster_populations.tolist(),
            representative_frames=representatives,
            tsne_coordinates=tsne_coords.tolist()
        )
        session.clustered_proteins[str(pdb_file)] = cluster_data
        
        logger.info(f"Selected {len(representatives)} representatives")

        return {
            "success": True,
            "cluster_data_key": str(pdb_file),  # Use this key for dock_ensemble
            "n_clusters": n_clusters,
            "n_conformations": n_frames,
            "cluster_populations": cluster_populations.tolist(),
            "representative_frames": representatives
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": type(e).__name__,
            "details": str(e)
        }

@mcp.tool()
async def dock_ensemble(
    protein_pdb: Annotated[str, Field(description="Protein PDB file path")],
    ligand_pdbqt: Annotated[str, Field(description="Path to ligand PDBQT file (from prepare_ligand output)")],
    representative_frames: Annotated[List[int], Field(description="List of representative frame indices from cluster_conformations")],
    residue_range: Annotated[Optional[List[int]], Field(description="Residues to dock (e.g., [125, 133, 136])")] = None,
    cluster_populations: Annotated[Optional[List[float]], Field(description="Cluster populations for weighted averaging (from cluster_conformations)")] = None,
    exhaustiveness: Annotated[int, Field(description="Vina exhaustiveness (8-64)")] = 32,
    box_size: Annotated[Optional[float], Field(description="Docking box size in Angstroms (auto-calculated if None)")] = None,
    receptor_prep_method: Annotated[str, Field(description="'obabel' (fast) or 'adfr' (publication quality)")] = "obabel"
) -> dict:
    """Perform ensemble docking using AutoDock Vina across representative conformations.

    This tool docks a ligand to multiple protein conformations (from clustering) and
    returns binding energies. More negative energy = stronger binding.

    REQUIRED INPUTS (from previous tools):
    - ligand_pdbqt: The pdbqt_path from prepare_ligand output
    - representative_frames: The representative_frames list from cluster_conformations output

    NEW FEATURES (aligned with original research):
    - cluster_populations: For population-weighted energy averaging
    - box_size=None: Auto-calculates from ligand radius + padding
    - receptor_prep_method: 'obabel' for speed, 'adfr' for publication quality
    - Returns XTC trajectory and NPY scores for downstream analysis
    """
    try:
        # Lazy load dependencies
        md = _load_mdtraj()
        Vina = _load_vina()
        np = _load_sklearn()[2]  # Just need numpy

        # Validate ligand file exists
        ligand_path = pathlib.Path(ligand_pdbqt)
        if not ligand_path.exists():
            raise FileNotFoundError(f"Ligand PDBQT not found: {ligand_pdbqt}")

        # Auto-calculate box size if not provided
        # Original research: run_autodockvina.py get_boxsize() line 28-32
        if box_size is None:
            box_size = calculate_box_size(str(ligand_path))
            logger.info(f"Auto-calculated box size: {box_size:.1f} Å")

        # Load protein trajectory
        traj = md.load(protein_pdb)
        protein_traj = traj.atom_slice(traj.top.select('protein'))

        # Determine residues to dock - dock ALL residues by default
        # Original research: run_autodockvina.py line 164 docks all residues
        residues_all = [res.resSeq for res in protein_traj.topology.residues]
        residues = residue_range if residue_range else residues_all

        # Create temp directory for this docking run
        temp_dir = tempfile.mkdtemp(prefix="mcp_dock_", dir=WORKSPACE_DIR)

        logger.info(f"Docking on {len(representative_frames)} clusters")

        all_results = []

        for cluster_id, frame_idx in enumerate(representative_frames):
            frame = protein_traj[frame_idx]

            frame_pdb = f"{temp_dir}/cluster_{cluster_id}_frame.pdb"
            frame.save_pdb(frame_pdb)

            frame_pdbqt = f"{temp_dir}/cluster_{cluster_id}_receptor.pdbqt"
            # Receptor preparation: obabel (fast) or ADFR (publication quality)
            # Original research: run_autodockvina.py protein_pdbqt_ADFR() line 41-44
            if receptor_prep_method == "adfr":
                subprocess.run([
                    'prepare_receptor', '-r', frame_pdb, '-A', 'hydrogens', '-o', frame_pdbqt
                ], check=True, capture_output=True, text=True)
            else:
                subprocess.run([
                    'obabel', '-ipdb', frame_pdb,
                    '-opdbqt', '-O', frame_pdbqt,
                    '-xr'  # -xr for rigid receptor (no rotatable bonds)
                ], check=True, capture_output=True, text=True)

            residue_energies = {}
            total_residues = len(residues)

            for idx, residue in enumerate(residues):
                try:
                    # Progress logging every 10 residues
                    if idx % 10 == 0:
                        logger.info(f"Cluster {cluster_id}: Progress {idx}/{total_residues} ({100*idx//total_residues}%)")

                    comx, comy, comz = get_residue_com(frame, residue)

                    # Vina's verbosity=0 suppresses all output
                    # REMOVED: os.dup2 pattern was NOT parallel-safe and caused
                    # BrokenPipeError when multiple docking tasks ran concurrently
                    # Ref: https://github.com/ccsb-scripps/AutoDock-Vina/issues/137
                    # Ref: https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html
                    v = Vina(sf_name='vina', verbosity=0)  # verbosity=0 suppresses all output
                    v.set_receptor(frame_pdbqt)
                    v.set_ligand_from_file(str(ligand_path))
                    v.compute_vina_maps(
                        center=[comx, comy, comz],
                        box_size=[box_size, box_size, box_size]
                    )
                    # Generate 20 poses (explore local minima) but keep only best
                    # Original research: run_autodockvina.py line 170-171
                    v.dock(exhaustiveness=exhaustiveness, n_poses=20)

                    pose_path = f"{temp_dir}/c{cluster_id}_r{residue}.pdbqt"
                    v.write_poses(pose_path, n_poses=1, overwrite=True)  # Keep only best pose

                    energy = extract_vina_energy(pose_path)
                    residue_energies[residue] = energy

                except Exception as e:
                    logger.warning(f"Docking failed for residue {residue}: {e}")
                    continue

            logger.info(f"Cluster {cluster_id}: Completed all {total_residues} residues")

            if residue_energies:
                best_residue = min(residue_energies, key=residue_energies.get)
                best_energy = residue_energies[best_residue]
            else:
                best_residue, best_energy = None, None

            all_results.append({
                "cluster_id": cluster_id,
                "frame_idx": frame_idx,
                "residue_energies": residue_energies,
                "best_residue": best_residue,
                "best_energy": best_energy
            })

            if best_energy is not None:
                logger.info(f"Cluster {cluster_id}: Best {best_residue} @ {best_energy:.2f}")

        # Build score array and save docked trajectory
        # Original research: run_autodockvina.py dock_all_frames() line 227-243
        score_array = np.zeros((len(all_results), 2))  # [energy, residue]
        docked_ligand_pdbs = []

        for i, result in enumerate(all_results):
            if result["best_energy"] is not None:
                score_array[i, 0] = result["best_energy"]
                score_array[i, 1] = result["best_residue"]

                # Convert best pose to PDB for trajectory
                best_pose_pdbqt = f"{temp_dir}/c{result['cluster_id']}_r{result['best_residue']}.pdbqt"
                best_pose_pdb = f"{temp_dir}/c{result['cluster_id']}_ligand.pdb"
                try:
                    subprocess.run([
                        'obabel', '-ipdbqt', best_pose_pdbqt, '-opdb', '-O', best_pose_pdb
                    ], check=True, capture_output=True, text=True)
                    docked_ligand_pdbs.append(best_pose_pdb)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Could not convert pose {best_pose_pdbqt}: {e}")

        # Save scores
        scores_path = f"{temp_dir}/docking_scores.npy"
        np.save(scores_path, score_array)

        # Save trajectory if we have results
        trajectory_xtc = None
        trajectory_pdb = None
        if docked_ligand_pdbs:
            try:
                lig_traj = md.load(docked_ligand_pdbs)
                trajectory_pdb = f"{temp_dir}/docked_trajectory.pdb"
                trajectory_xtc = f"{temp_dir}/docked_trajectory.xtc"
                lig_traj.save_xtc(trajectory_xtc)
                lig_traj[0].save_pdb(trajectory_pdb)
                logger.info(f"Saved trajectory: {len(docked_ligand_pdbs)} frames")
            except Exception as e:
                logger.warning(f"Could not save trajectory: {e}")

        # Compute weighted summary if populations provided
        # Original research: plotting.py line 124
        weighted_summary = None
        if cluster_populations:
            weighted_summary = apply_cluster_weights(all_results, cluster_populations)

        return {
            "success": True,
            "n_clusters": len(all_results),
            "cluster_results": all_results,
            "ensemble_summary": {
                "average_best_energy": float(np.mean([
                    r["best_energy"] for r in all_results if r["best_energy"]
                ])) if any(r["best_energy"] for r in all_results) else None,
                "std_best_energy": float(np.std([
                    r["best_energy"] for r in all_results if r["best_energy"]
                ])) if any(r["best_energy"] for r in all_results) else None
            },
            "weighted_summary": weighted_summary,
            "scores_npy": scores_path,
            "trajectory_xtc": trajectory_xtc,
            "trajectory_pdb": trajectory_pdb,
            "box_size_used": box_size,
            "receptor_prep_method": receptor_prep_method,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": type(e).__name__,
            "details": str(e)
        }

@mcp.tool()
async def analyze_interactions(
    protein_pdb: Annotated[str, Field(description="Protein PDB file path")],
    ligand_pdbqt: Annotated[str, Field(description="Ligand PDBQT file path")],
    frame_indices: Annotated[Optional[List[int]], Field(description="Specific frames to analyze")] = None,
    session_id: Annotated[str, Field(description="Session identifier")] = "default"
) -> dict:
    """Analyze protein-ligand interactions (H-bonds, hydrophobic, aromatic contacts)."""
    try:
        # Lazy load dependencies
        md = _load_mdtraj()
        ANALYSIS_AVAILABLE = _load_analysis()
        
        if not ANALYSIS_AVAILABLE:
            return {
                "success": False,
                "error": "AnalysisNotAvailable",
                "details": "trajectory_analysis.py module not found"
            }
        
        # Import analysis functions
        from trajectory_analysis import hbond, hphob_contacts, aro_contacts, charge_contacts, dual_contact
        
        if session_id not in _sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = _sessions[session_id]
        
        # Validate files
        protein_path = validate_file_path(protein_pdb)
        ligand_path = validate_file_path(ligand_pdbqt)
        
        # Load protein trajectory
        traj = md.load(str(protein_path))
        
        # Select frames to analyze
        if frame_indices is None:
            frame_indices = list(range(min(10, traj.n_frames)))  # Analyze first 10 frames by default
        
        # Analyze each frame
        all_interactions = []
        
        for frame_idx in frame_indices:
            if frame_idx >= traj.n_frames:
                continue
            
            frame = traj[frame_idx]
            
            # Compute interactions
            try:
                hbonds = hbond(frame, ligand_path)
                hydrophobic = hphob_contacts(frame, ligand_path)
                aromatic = aro_contacts(frame, ligand_path)
                charged = charge_contacts(frame, ligand_path)
                dual = dual_contact(frame, ligand_path)
                
                all_interactions.append({
                    "frame": frame_idx,
                    "hbonds": hbonds,
                    "hydrophobic": hydrophobic,
                    "aromatic": aromatic,
                    "charged": charged,
                    "dual": dual
                })
            except Exception as e:
                logger.warning(f"Analysis failed for frame {frame_idx}: {e}")
                continue
        
        # Summarize interactions across frames
        summary = {
            "n_frames_analyzed": len(all_interactions),
            "avg_hbonds": sum(len(i["hbonds"]) for i in all_interactions) / len(all_interactions) if all_interactions else 0,
            "avg_hydrophobic": sum(len(i["hydrophobic"]) for i in all_interactions) / len(all_interactions) if all_interactions else 0,
            "avg_aromatic": sum(len(i["aromatic"]) for i in all_interactions) / len(all_interactions) if all_interactions else 0,
        }
        
        return {
            "success": True,
            "interactions": all_interactions,
            "summary": summary
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": type(e).__name__,
            "details": str(e)
        }

if __name__ == "__main__":
    import sys
    
    # Check for optional external tools (warn but don't fail)
    required = ['obabel', 'mk_prepare_ligand.py', 'mk_prepare_receptor.py']
    missing = [cmd for cmd in required if shutil.which(cmd) is None]

    if missing:
        logger.warning(f"Missing optional commands: {', '.join(missing)}")
        logger.warning("Some tools may not work. Install meeko: pip install meeko gemmi")
    
    logger.info("Starting MCP Server...")
    mcp.run()
