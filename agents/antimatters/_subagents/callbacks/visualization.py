"""Visualization generators for docking artifacts."""
import io
import logging

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
