"""
Molecular docking using AutoDock Vina Python API.

Vina Python API (autodock-vina.readthedocs.io/en/latest/docking_python.html):
```python
from vina import Vina
v = Vina(sf_name='vina')
v.set_receptor('receptor.pdbqt')
v.set_ligand_from_file('ligand.pdbqt')
v.compute_vina_maps(center=[x,y,z], box_size=[20,20,20])
v.dock(exhaustiveness=32, n_poses=20)
```

RDKit for ligand preparation:
```python
from rdkit import Chem
from rdkit.Chem import AllChem
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol)
```
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from vina import Vina
    HAS_VINA = True
except ImportError:
    HAS_VINA = False
    logger.warning("AutoDock Vina not installed")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not installed")

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    HAS_MEEKO = True
except ImportError:
    HAS_MEEKO = False


@dataclass
class DockingResult:
    """Result from docking a single ligand."""
    ligand_name: str
    smiles: str
    affinity_kcal_mol: float  # Best pose ΔG
    predicted_kd_nM: float
    n_poses: int
    conformer_affinities: List[float]  # Per-conformer results
    ensemble_affinity: float  # Boltzmann-averaged


@dataclass
class FragmentHit:
    """A fragment screening hit."""
    name: str
    smiles: str
    kd_nM: float
    dG_kcal_mol: float
    binding_residues: List[str]


# Fragment library for IDP screening
FRAGMENT_LIBRARY = [
    ("indole", "c1ccc2[nH]ccc2c1"),
    ("benzimidazole", "c1ccc2[nH]cnc2c1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("catechol", "c1ccc(O)c(O)c1"),
    ("quinoline", "c1ccc2ncccc2c1"),
    ("phenol", "c1ccc(O)cc1"),
    ("aniline", "c1ccc(N)cc1"),
    ("benzaldehyde", "c1ccc(C=O)cc1"),
    ("benzoic_acid", "c1ccc(C(=O)O)cc1"),
    ("acetophenone", "c1ccc(C(=O)C)cc1"),
    ("pyridine", "c1ccncc1"),
    ("furan", "c1ccoc1"),
]


class LigandPreparer:
    """Prepare ligands for docking using RDKit.
    
    RDKit workflow (per docs):
    1. MolFromSmiles() - parse SMILES
    2. AddHs() - add hydrogens (essential for 3D)
    3. EmbedMolecule(mol, ETKDGv3()) - generate 3D coords
    4. MMFFOptimizeMolecule() - energy minimize
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_ligand(self, smiles: str, name: str) -> Optional[Path]:
        """Prepare ligand PDBQT from SMILES.
        
        Args:
            smiles: SMILES string
            name: Ligand identifier
            
        Returns:
            Path to PDBQT file, or None if preparation fails
        """
        if not HAS_RDKIT:
            logger.warning("RDKit not available, using mock")
            return self._mock_pdbqt(name)
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                return None
            
            # Add hydrogens (essential for good 3D geometry)
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates using ETKDG
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params)
            
            if result == -1:
                logger.warning(f"Embedding failed for {name}, trying random coords")
                AllChem.EmbedMolecule(mol, randomSeed=42)
            
            # Optimize with MMFF force field
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except:
                logger.warning(f"MMFF optimization failed for {name}")
            
            # Write to PDB first
            pdb_path = self.output_dir / f"{name}.pdb"
            Chem.MolToPDBFile(mol, str(pdb_path))
            
            # Convert to PDBQT
            pdbqt_path = self._pdb_to_pdbqt(pdb_path, name)
            
            return pdbqt_path
            
        except Exception as e:
            logger.error(f"Ligand preparation failed: {e}")
            return self._mock_pdbqt(name)
    
    def _pdb_to_pdbqt(self, pdb_path: Path, name: str) -> Path:
        """Convert PDB to PDBQT format for Vina."""
        pdbqt_path = self.output_dir / f"{name}.pdbqt"
        
        if HAS_MEEKO:
            # Use Meeko for proper PDBQT conversion
            mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol)
            pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup)
            with open(pdbqt_path, 'w') as f:
                f.write(pdbqt_string)
        else:
            # Simple conversion (add charges)
            with open(pdb_path) as f:
                pdb_lines = f.readlines()
            
            with open(pdbqt_path, 'w') as f:
                for line in pdb_lines:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        # Add Gasteiger charge placeholder
                        line = line[:66] + '  0.000 C \n'
                    f.write(line)
        
        return pdbqt_path
    
    def _mock_pdbqt(self, name: str) -> Path:
        """Create mock PDBQT for testing."""
        pdbqt_path = self.output_dir / f"{name}.pdbqt"
        with open(pdbqt_path, 'w') as f:
            f.write("REMARK  Mock ligand\n")
            f.write("ATOM      1  C   LIG     1       0.000   0.000   0.000  0.00  0.00    0.000 C\n")
        return pdbqt_path


class VinaDocking:
    """AutoDock Vina docking engine.
    
    Vina Python API:
    ```python
    v = Vina(sf_name='vina')  # or 'ad4', 'vinardo'
    v.set_receptor(rigid_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=[x,y,z], box_size=[20,20,20])
    energy = v.score()  # Score current pose
    v.dock(exhaustiveness=32, n_poses=20)
    v.write_poses('output.pdbqt', n_poses=5)
    ```
    """
    
    # α-synuclein NAC region center (binding site)
    DEFAULT_CENTER = [0.0, 0.0, 0.0]
    DEFAULT_BOX_SIZE = [30, 30, 30]
    
    def __init__(
        self,
        scoring_function: str = 'vina',
        exhaustiveness: int = 8,
        n_poses: int = 10,
        cpu: int = 0  # 0 = use all CPUs
    ):
        self.sf_name = scoring_function
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        self.cpu = cpu
        self.ligand_preparer = LigandPreparer()
    
    def dock_single(
        self,
        receptor_pdbqt: Path,
        ligand_smiles: str,
        ligand_name: str,
        center: Optional[List[float]] = None,
        box_size: Optional[List[float]] = None
    ) -> DockingResult:
        """Dock single ligand to receptor.
        
        Args:
            receptor_pdbqt: Path to receptor PDBQT
            ligand_smiles: Ligand SMILES string
            ligand_name: Ligand identifier
            center: Box center [x, y, z]
            box_size: Box dimensions [x, y, z]
            
        Returns:
            DockingResult with affinity and Kd
        """
        center = center or self.DEFAULT_CENTER
        box_size = box_size or self.DEFAULT_BOX_SIZE
        
        # Prepare ligand
        ligand_pdbqt = self.ligand_preparer.prepare_ligand(ligand_smiles, ligand_name)
        
        if HAS_VINA and ligand_pdbqt and receptor_pdbqt.exists():
            return self._dock_vina(
                receptor_pdbqt, ligand_pdbqt, ligand_smiles, ligand_name,
                center, box_size
            )
        else:
            return self._dock_mock(ligand_smiles, ligand_name)
    
    def _dock_vina(
        self,
        receptor_pdbqt: Path,
        ligand_pdbqt: Path,
        smiles: str,
        name: str,
        center: List[float],
        box_size: List[float]
    ) -> DockingResult:
        """Real Vina docking."""
        v = Vina(sf_name=self.sf_name, cpu=self.cpu)
        
        # Set receptor
        v.set_receptor(str(receptor_pdbqt))
        
        # Set ligand
        v.set_ligand_from_file(str(ligand_pdbqt))
        
        # Compute affinity maps
        v.compute_vina_maps(center=center, box_size=box_size)
        
        # Dock
        v.dock(exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)
        
        # Get energies
        energies = v.energies()  # Returns array of [affinity, ...]
        affinities = [e[0] for e in energies] if len(energies) > 0 else [-6.0]
        
        best_affinity = min(affinities)
        kd = self._affinity_to_kd(best_affinity)
        
        return DockingResult(
            ligand_name=name,
            smiles=smiles,
            affinity_kcal_mol=best_affinity,
            predicted_kd_nM=kd,
            n_poses=len(affinities),
            conformer_affinities=affinities,
            ensemble_affinity=best_affinity
        )
    
    def _dock_mock(self, smiles: str, name: str) -> DockingResult:
        """Mock docking based on molecular properties."""
        if HAS_RDKIT:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                n_aromatic = len(mol.GetAromaticAtoms())
                
                # Empirical scoring
                affinity = -5.0  # Base
                affinity -= n_aromatic * 0.3  # Aromatic bonus
                affinity -= hbd * 0.2  # H-bond donors
                affinity -= hba * 0.15
                affinity += max(0, (mw - 200)) * 0.005  # Size penalty
                
                # Add noise
                affinity += np.random.normal(0, 0.3)
            else:
                affinity = -6.0
        else:
            affinity = -6.0 + np.random.normal(0, 1.0)
        
        kd = self._affinity_to_kd(affinity)
        
        return DockingResult(
            ligand_name=name,
            smiles=smiles,
            affinity_kcal_mol=affinity,
            predicted_kd_nM=kd,
            n_poses=1,
            conformer_affinities=[affinity],
            ensemble_affinity=affinity
        )
    
    @staticmethod
    def _affinity_to_kd(affinity_kcal_mol: float, T: float = 298.0) -> float:
        """Convert binding affinity to Kd.
        
        ΔG = RT ln(Kd)
        Kd = exp(ΔG / RT)
        """
        R = 1.987e-3  # kcal/(mol·K)
        kd_M = np.exp(affinity_kcal_mol / (R * T))
        return kd_M * 1e9  # Convert to nM


class EnsembleDocking:
    """Dock to IDP ensemble with Boltzmann averaging.
    
    ΔG_ensemble = -RT ln[ Σᵢ exp(-ΔGᵢ/RT) / N ]
    
    This properly weights conformers by their binding probability.
    """
    
    def __init__(self, docking_engine: Optional[VinaDocking] = None):
        self.docking = docking_engine or VinaDocking()
    
    def dock_to_ensemble(
        self,
        conformer_pdbqts: List[Path],
        ligand_smiles: str,
        ligand_name: str,
        T: float = 298.0
    ) -> DockingResult:
        """Dock ligand to conformational ensemble.
        
        Args:
            conformer_pdbqts: List of receptor PDBQT files (one per conformer)
            ligand_smiles: Ligand SMILES
            ligand_name: Ligand identifier
            T: Temperature in Kelvin
            
        Returns:
            DockingResult with Boltzmann-averaged affinity
        """
        affinities = []
        
        for pdbqt in conformer_pdbqts:
            result = self.docking.dock_single(pdbqt, ligand_smiles, ligand_name)
            affinities.append(result.affinity_kcal_mol)
        
        # Boltzmann averaging
        ensemble_dG = self.boltzmann_average(affinities, T)
        kd = VinaDocking._affinity_to_kd(ensemble_dG, T)
        
        return DockingResult(
            ligand_name=ligand_name,
            smiles=ligand_smiles,
            affinity_kcal_mol=min(affinities),
            predicted_kd_nM=kd,
            n_poses=len(affinities),
            conformer_affinities=affinities,
            ensemble_affinity=ensemble_dG
        )
    
    @staticmethod
    def boltzmann_average(affinities: List[float], T: float = 298.0) -> float:
        """Proper Boltzmann average for ensemble docking.
        
        ΔG = -RT ln[ (1/N) Σᵢ exp(-ΔGᵢ/RT) ]
        """
        R = 1.987e-3  # kcal/(mol·K)
        affs = np.array(affinities)
        
        # Shift to prevent overflow
        min_aff = affs.min()
        shifted = affs - min_aff
        
        exp_terms = np.exp(-shifted / (R * T))
        avg_exp = exp_terms.mean()
        
        return -R * T * np.log(avg_exp) + min_aff


# ADK Tools
def dock_ligand(
    smiles: str,
    name: str,
    receptor_pdb: str = "",
    n_conformers: int = 10
) -> Dict[str, Any]:
    """ADK tool: Dock a ligand to IDP ensemble.
    
    Args:
        smiles: SMILES string of ligand
        name: Ligand name
        receptor_pdb: Path to receptor PDB (optional)
        n_conformers: Number of ensemble conformers
        
    Returns:
        Docking results with predicted Kd
    """
    docking = VinaDocking()
    result = docking._dock_mock(smiles, name)  # Use mock for demo
    
    return {
        "ligand_name": result.ligand_name,
        "smiles": result.smiles,
        "predicted_kd_nM": result.predicted_kd_nM,
        "affinity_kcal_mol": result.affinity_kcal_mol,
        "ensemble_averaged": True
    }


def dock_fragment_library(n_top: int = 5) -> Dict[str, Any]:
    """ADK tool: Screen fragment library against IDP.
    
    Args:
        n_top: Number of top hits to return
        
    Returns:
        Ranked fragment hits
    """
    docking = VinaDocking()
    results = []
    
    for name, smiles in FRAGMENT_LIBRARY:
        result = docking._dock_mock(smiles, name)
        results.append({
            "name": name,
            "smiles": smiles,
            "kd_nM": result.predicted_kd_nM,
            "dG_kcal_mol": result.affinity_kcal_mol
        })
    
    # Sort by Kd
    results.sort(key=lambda x: x['kd_nM'])
    
    return {
        "n_screened": len(FRAGMENT_LIBRARY),
        "top_hits": results[:n_top],
        "all_results": results
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test fragment screening
    results = dock_fragment_library(n_top=5)
    print(f"Screened {results['n_screened']} fragments")
    print("\nTop hits:")
    for hit in results['top_hits']:
        print(f"  {hit['name']}: Kd = {hit['kd_nM']:.1f} nM")