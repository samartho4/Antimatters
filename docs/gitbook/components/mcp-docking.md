# MCP Docking Server

**File**: `mcp_servers/docking/server.py`

FastMCP server exposing 4 tools for molecular docking.

## Tools

### prepare_ligand

Converts SMILES to 3D structures.

```python
@mcp.tool
def prepare_ligand(smiles: str, optimize: bool = True, session_id: str = "default"):
    # 1. RDKit: SMILES → 3D mol (EmbedMolecule)
    # 2. UFF energy minimization (if optimize=True)
    # 3. Write mol2 and PDBQT files
    return {
        "ligand_id": ...,
        "mol2_path": ...,
        "pdbqt_path": ...,
        "num_atoms": ...,
        "num_rotatable_bonds": ...,
        "molecular_weight": ...
    }
```

### cluster_conformations

Clusters protein conformations using t-SNE on pairwise RMSD.

```python
@mcp.tool
def cluster_conformations(pdb_path: str, n_clusters: int = 20, perplexity: int = 30):
    # 1. Load multi-model PDB (MDTraj)
    # 2. Compute pairwise RMSD
    # 3. t-SNE dimensionality reduction
    # 4. k-medoids clustering
    return {
        "n_clusters": ...,
        "cluster_assignments": [...],
        "cluster_populations": [...],  # Fractional populations
        "representative_frames": [...],
        "tsne_coordinates": [...]
    }
```

### dock_ensemble

AutoDock Vina docking across cluster representatives.

```python
@mcp.tool
def dock_ensemble(
    ligand_pdbqt: str,
    representative_frames: List[int],
    protein_pdb: str,
    binding_site: List[int],       # Residue indices
    exhaustiveness: int = 32,
    box_size: Optional[float] = None,
    receptor_prep_method: str = "obabel"  # or "adfr"
):
    # Per representative frame:
    # 1. Extract frame from ensemble
    # 2. Prepare receptor PDBQT
    # 3. Calculate box from binding site COM
    # 4. Run Vina
    return {
        "binding_scores": [...],  # Per cluster
        "ensemble_summary": {
            "weighted_average": ...,  # Population-weighted
            "std_dev": ...,
            "best_score": ...
        }
    }
```

### analyze_interactions

Protein-ligand interaction analysis.

```python
@mcp.tool
def analyze_interactions(protein_pdb: str, ligand_pdbqt: str, frame_indices: List[int] = None):
    # Uses trajectory_analysis.py functions:
    # - aro_contacts() — π-stacking
    # - hbond() — H-bonds
    # - charge_contacts() — salt bridges
    return {
        "h_bonds": [...],
        "hydrophobic": [...],
        "aromatic": [...],
        "charge": [...]
    }
```

## Dependencies

- RDKit (SMILES processing)
- MDTraj (trajectory handling)
- scikit-learn (t-SNE, clustering)
- AutoDock Vina (docking)
- Open Babel (PDBQT conversion)

## Lazy Loading

Heavy imports deferred until first use:

```python
_RDKIT_LOADED = False
def _load_rdkit():
    global _RDKIT_LOADED, Chem, AllChem
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _RDKIT_LOADED = True
```
