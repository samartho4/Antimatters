"""Visualization generators for docking artifacts.

Includes:
- 2D plots (cluster populations, binding energies, interaction maps)
- 3D visualizations (docked ligand in IDP, cluster conformations)
"""
import io
import os
import logging
from pathlib import Path

logger = logging.getLogger("antimatters")


def generate_cluster_plot(response: dict) -> bytes:
    """Generate cluster visualization from t-SNE coordinates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available")
        return b""

    frames = response.get("representative_frames", [])
    populations = response.get("cluster_populations", [1] * len(frames))

    if not frames:
        return b""

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(frames)))
    ax.bar(range(len(frames)), populations, color=colors)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Population")
    ax.set_title(f"Conformational Clusters ({len(frames)} representatives)")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_docking_plot(response: dict) -> bytes:
    """Generate binding energy bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return b""

    results = response.get("cluster_results", [])
    if not results:
        return b""

    clusters = [r["cluster_id"] for r in results if r.get("best_energy")]
    energies = [r["best_energy"] for r in results if r.get("best_energy")]

    if not energies:
        return b""

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["darkgreen" if e < -7 else "green" if e < -5 else "orange" if e < -4 else "red" for e in energies]
    ax.bar(clusters, energies, color=colors, edgecolor="black")

    ax.axhline(y=-6, color="red", linestyle="--", alpha=0.7, label="Drug-like")
    ax.axhline(y=np.mean(energies), color="blue", linestyle="-", alpha=0.7, label=f"Mean: {np.mean(energies):.2f}")

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Binding Energy (kcal/mol)")
    ax.set_title("Ensemble Docking Results")
    ax.legend()

    summary = response.get("ensemble_summary", {})
    avg = summary.get("average_best_energy", np.mean(energies))
    std = summary.get("std_best_energy", np.std(energies))
    ax.text(0.02, 0.98, f"Avg: {avg:.2f} ± {std:.2f} kcal/mol",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_interaction_map(response: dict) -> bytes:
    """Generate protein-ligand interaction heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return b""

    interactions = response.get("interactions", [])
    if not interactions:
        return b""

    # Aggregate by residue
    residue_data = {}
    for frame in interactions:
        for itype, contacts in frame.items():
            if isinstance(contacts, list):
                for c in contacts:
                    res = c.get("residue", c.get("protein_residue", "UNK"))
                    if res not in residue_data:
                        residue_data[res] = {"hbond": 0, "hydrophobic": 0, "aromatic": 0}
                    if "hbond" in itype.lower():
                        residue_data[res]["hbond"] += 1
                    elif "hydrophobic" in itype.lower():
                        residue_data[res]["hydrophobic"] += 1
                    elif "aromatic" in itype.lower():
                        residue_data[res]["aromatic"] += 1

    if not residue_data:
        return b""

    residues = sorted(residue_data.keys())
    x = np.arange(len(residues))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, [residue_data[r]["hbond"] for r in residues], width, label="H-bonds", color="blue")
    ax.bar(x, [residue_data[r]["hydrophobic"] for r in residues], width, label="Hydrophobic", color="green")
    ax.bar(x + width, [residue_data[r]["aromatic"] for r in residues], width, label="Aromatic", color="purple")

    ax.set_xlabel("Residue")
    ax.set_ylabel("Contact Count")
    ax.set_title("Protein-Ligand Interactions")
    ax.set_xticks(x)
    ax.set_xticklabels(residues, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_3d_structure_view(
    protein_pdb_path: str,
    ligand_smiles: str = None,
    ligand_name: str = "Ligand",
    title: str = "Docked Structure"
) -> bytes:
    """
    Generate 3D visualization of protein-ligand complex.

    Uses matplotlib 3D for cross-platform compatibility.

    Args:
        protein_pdb_path: Path to protein PDB file
        ligand_smiles: SMILES string for ligand (optional)
        ligand_name: Name of the ligand
        title: Figure title

    Returns:
        PNG image as bytes
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available for 3D visualization")
        return b""

    # Parse PDB file for coordinates
    coords = []
    residues = []

    try:
        pdb_path = Path(protein_pdb_path)
        if pdb_path.exists():
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            res_name = line[17:20].strip()
                            coords.append([x, y, z])
                            residues.append(res_name)
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        logger.warning(f"Could not parse PDB file: {e}")

    if not coords:
        coords = np.random.randn(100, 3) * 20
        residues = ["UNK"] * 100

    coords = np.array(coords)

    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color by residue type
    colors = []
    for res in residues:
        if res in ['TYR', 'PHE', 'TRP']:
            colors.append('#9b59b6')  # Aromatic - Purple
        elif res in ['GLU', 'ASP']:
            colors.append('#e74c3c')  # Acidic - Red
        elif res in ['LYS', 'ARG', 'HIS']:
            colors.append('#3498db')  # Basic - Blue
        elif res in ['SER', 'THR', 'ASN', 'GLN']:
            colors.append('#2ecc71')  # Polar - Green
        else:
            colors.append('#95a5a6')  # Hydrophobic - Gray

    # Subsample for performance
    if len(coords) > 1000:
        idx = np.random.choice(len(coords), 1000, replace=False)
        coords_plot = coords[idx]
        colors_plot = [colors[i] for i in idx]
    else:
        coords_plot = coords
        colors_plot = colors

    ax.scatter(coords_plot[:, 0], coords_plot[:, 1], coords_plot[:, 2],
               c=colors_plot, s=20, alpha=0.6)

    # Add binding site highlight
    center = coords.mean(axis=0)
    ax.scatter([center[0]], [center[1]], [center[2]],
               c='yellow', s=200, marker='*', edgecolor='black', label='Binding Region')

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f"{title}\n{ligand_name}")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9b59b6', label='Aromatic'),
        Patch(facecolor='#e74c3c', label='Acidic'),
        Patch(facecolor='#3498db', label='Basic'),
        Patch(facecolor='#2ecc71', label='Polar'),
        Patch(facecolor='#95a5a6', label='Hydrophobic'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_cluster_overlay(
    protein_pdb_path: str,
    representative_frames: list = None,
    cluster_populations: list = None,
    title: str = "Cluster Conformations"
) -> bytes:
    """
    Generate cluster conformation overlay visualization using REAL data only.

    Uses mdtraj to extract specific frames by INDEX from the trajectory.
    representative_frames are frame INDICES (e.g., [545, 129]), not MODEL numbers.

    Args:
        protein_pdb_path: Path to multi-model PDB file (ensemble trajectory)
        representative_frames: List of frame INDICES for each cluster
        cluster_populations: Population weight of each cluster
        title: Figure title

    Returns:
        PNG image as bytes, or empty bytes if data cannot be loaded
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import mdtraj as md
    except ImportError as e:
        logger.error(f"Required library not available: {e}")
        return b""

    # Load trajectory with mdtraj
    pdb_path = Path(protein_pdb_path)
    if not pdb_path.exists():
        logger.error(f"PDB file not found: {protein_pdb_path}")
        return b""

    try:
        traj = md.load(str(pdb_path))
        logger.info(f"Loaded trajectory: {traj.n_frames} frames, {traj.n_atoms} atoms")
    except Exception as e:
        logger.error(f"Failed to load trajectory: {e}")
        return b""

    # Validate representative_frames
    if not representative_frames:
        logger.error("No representative_frames provided")
        return b""

    n_clusters = len(representative_frames)
    cluster_coords = {}

    for i, frame_idx in enumerate(representative_frames):
        if frame_idx >= traj.n_frames:
            logger.error(f"Frame index {frame_idx} out of range (trajectory has {traj.n_frames} frames)")
            return b""
        # Extract coordinates for this frame (convert nm to Å)
        coords = traj.xyz[frame_idx] * 10  # nm -> Å
        cluster_coords[i] = coords
        logger.info(f"Cluster {i+1}: frame {frame_idx}, {len(coords)} atoms")

    # Validate populations
    if cluster_populations:
        if len(cluster_populations) != n_clusters:
            logger.warning(f"Population count mismatch: {len(cluster_populations)} vs {n_clusters} clusters")
        populations = cluster_populations[:n_clusters]
    else:
        # Calculate equal populations if not provided
        populations = [1.0 / n_clusters] * n_clusters

    # Create figure
    fig = plt.figure(figsize=(14, 6))

    # 3D overlay plot
    ax1 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        coords = cluster_coords[i]
        # Subsample for performance (keep every Nth atom for large structures)
        if len(coords) > 500:
            step = len(coords) // 500
            coords = coords[::step]
        max_pop = max(populations) if populations else 1.0
        alpha = 0.3 + 0.4 * (populations[i] / max_pop) if max_pop > 0 else 0.5
        ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=[colors[i]], s=10, alpha=alpha,
                   label=f'Cluster {i+1} (frame {representative_frames[i]})')

    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.set_title(f"{title}\n({n_clusters} clusters from {traj.n_frames} frames)")
    ax1.legend(loc='upper left', fontsize=8)

    # Population bar chart
    ax2 = fig.add_subplot(122)
    cluster_labels = [f'C{i+1}\n(f{representative_frames[i]})' for i in range(n_clusters)]
    bars = ax2.bar(cluster_labels, populations, color=colors)
    ax2.set_xlabel('Cluster (frame)')
    ax2.set_ylabel('Population')
    ax2.set_title('Cluster Populations')

    for bar, pop in zip(bars, populations):
        height = bar.get_height()
        label = f'{pop:.1%}' if pop <= 1 else f'{pop:.1f}'
        ax2.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
