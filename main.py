#!/usr/bin/env python3
"""
IDoIDR - Agentic IDP Drug Discovery Platform
Google Gemini Hackathon 2026
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import logging

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *args, **kwargs):
            text = str(args[0]) if args else ""
            # Strip rich markup
            import re
            text = re.sub(r'\[.*?\]', '', text)
            print(text)
    console = Console()
logging.basicConfig(level=logging.INFO, format='%(message)s')

BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║        IDoIDR - Agentic IDP Drug Discovery                    ║
║        Target: α-Synuclein (Parkinson's Disease)              ║
║        Google Gemini Hackathon 2026                           ║
╚═══════════════════════════════════════════════════════════════╝
"""

def show_banner():
    print(BANNER)

def run_demo():
    """Run complete discovery demo"""
    from agents.discovery import StandaloneAgent
    
    show_banner()
    console.print("\n[bold green]Starting autonomous discovery workflow...[/]\n")
    
    agent = StandaloneAgent("./data")
    
    # Step 1: Validate
    print("  Validating ensemble...")
    validation = agent.tools.validate_ensemble_tool()
    status = "✓ PASSED" if validation['passed'] else "✗ FAILED"
    print(f"  Ensemble validation: {status} (χ² = {validation['chi2']:.2f})")
    
    # Step 2: Select conformers
    print("  Selecting extended conformers...")
    selection = agent.tools.query_ensemble_tool("extended Rg > 30")
    print(f"  Selected: {selection['n_selected']}/{selection['n_total']} conformers")
    print(f"  Rg: {selection['rg_stats']['mean']:.1f} ± {selection['rg_stats']['std']:.1f} Å")
    
    # Step 3: Screen fragments
    print("  Screening fragment library...")
    screening = agent.tools.dock_fragments_tool(n_conformers=30, top_n=10)
    print(f"  Screened: {screening['n_screened']} fragments")
    
    # Step 4: Discover rules
    print("  Discovering binding rules (PySR)...")
    rules = agent.tools.discover_rules_tool(['rg', 'aromatic_exposure'])
    
    # Display results
    print("\n" + "="*50)
    print("Discovery Results")
    print("="*50)
    
    print("\nTop Fragment Hits:")
    for i, hit in enumerate(screening['top_hits'][:5], 1):
        kd = hit['predicted_Kd_nM']
        dg = hit['dG_ensemble_kcal_mol']
        marker = "🏆" if i == 1 else "  "
        print(f"  {marker}{i}. {hit['ligand_name']:20} Kd={kd:8.1f} nM  ΔG={dg:.2f} kcal/mol")
    
    print(f"\nDiscovered Binding Rule:")
    print(f"  {rules['equation']}")
    print(f"\nInterpretation:")
    print(f"  {rules['interpretation']}")
    
    print("\nRecommendations:")
    print("  1. Focus on aromatic scaffolds (indole, benzimidazole)")
    print("  2. Target extended conformations (Rg > 35Å)")
    print("  3. Optimize for residues G86, F94, K96")
    
    return {
        "validation": validation,
        "selection": selection,
        "screening": screening,
        "rules": rules
    }

def run_validation():
    """Validate ensemble against experimental data"""
    from tools.ensemble import EnsembleProcessor
    
    show_banner()
    print("\nValidating PED00024 ensemble...\n")
    
    processor = EnsembleProcessor("./data")
    ensemble = processor.load_ensemble("PED00024")
    validation = processor.validate_ensemble(ensemble)
    
    print(f"  Conformers: {validation['n_conformers']}")
    print(f"  Rg (mean): {validation['rg_mean']:.1f} Å")
    print(f"  Rg (std): {validation['rg_std']:.1f} Å")
    print(f"  Expected: {validation['expected_rg']:.1f} Å")
    print(f"  χ² vs SAXS: {validation['chi2']:.2f}")
    
    status = "✓ PASSED" if validation['passed'] else "✗ FAILED"
    print(f"\n  Validation: {status}")
    
    return validation

def run_benchmark():
    """Benchmark against known ligands"""
    from tools.ensemble import EnsembleProcessor
    from tools.docking import dock_ligand
    
    show_banner()
    console.print("\n[bold]Benchmarking known α-synuclein ligands...[/]\n")
    
    known_ligands = [
        {"name": "BF-79", "smiles": "c1ccc2c(c1)cc(N)cc2", "kd_exp": 4.77},
        {"name": "EGCG", "smiles": "Oc1cc(O)c2c(c1)OC(c1cc(O)c(O)c(O)c1)C(O)C2", "kd_exp": 1000},
        {"name": "Fasudil", "smiles": "CC(=O)Nc1ccc2c(c1)c1cccnc1n2C", "kd_exp": 500},
    ]
    
    processor = EnsembleProcessor("./data")
    ensemble = processor.load_ensemble()
    
    table = Table(title="Benchmark Results")
    table.add_column("Ligand", style="cyan")
    table.add_column("Kd_exp (nM)", justify="right")
    table.add_column("Kd_pred (nM)", justify="right")
    table.add_column("ΔG (kcal/mol)", justify="right")
    
    results = []
    for lig in known_ligands:
        console.print(f"  Docking {lig['name']}...")
        result = dock_ligand(lig['smiles'], lig['name'], ensemble.pdb_paths, 30)
        results.append({**lig, **result})
        table.add_row(
            lig['name'],
            f"{lig['kd_exp']:.1f}",
            f"{result['predicted_Kd_nM']:.1f}",
            f"{result['dG_ensemble_kcal_mol']:.2f}"
        )
    
    console.print("\n")
    console.print(table)
    
    return results

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run web server with dashboard"""
    import uvicorn
    from api.server import app
    
    show_banner()
    console.print(f"\n[bold green]Starting server at http://{host}:{port}[/]\n")
    uvicorn.run(app, host=host, port=port)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="IDoIDR - IDP Drug Discovery")
    parser.add_argument("--demo", action="store_true", help="Run full discovery demo")
    parser.add_argument("--validate", action="store_true", help="Validate ensemble")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark known ligands")
    parser.add_argument("--server", action="store_true", help="Run web server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--query", type=str, help="Custom discovery query")
    
    args = parser.parse_args()
    
    # Ensure directories
    Path("./data").mkdir(exist_ok=True)
    Path("./outputs").mkdir(exist_ok=True)
    
    if args.validate:
        run_validation()
    elif args.benchmark:
        run_benchmark()
    elif args.server:
        run_server(port=args.port)
    elif args.query:
        from agents.discovery import run_discovery
        result = run_discovery(args.query)
        console.print(result)
    else:
        run_demo()

if __name__ == "__main__":
    main()