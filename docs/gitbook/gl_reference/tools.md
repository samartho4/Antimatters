# Tool Reference

## Research Agent Tools

| Tool | Description |
|------|-------------|
| `fetch_ped_ensemble` | Fetch PDB ensemble from Protein Ensemble Database |
| `search_compounds` | ChEMBL similarity search |
| `bc_get_europepmc_articles` | Literature search via EuropePMC |
| `create_protocol` | Create Protocol artifact |
| `validate_protocol` | Validate protocol with evidence |

## Engineer Agent Tools

| Tool | Description |
|------|-------------|
| `prepare_ligand` | SMILES → mol2/PDBQT conversion |
| `cluster_conformations` | t-SNE clustering of conformations |
| `dock_ensemble` | AutoDock Vina ensemble docking |
| `analyze_interactions` | H-bond, aromatic, hydrophobic analysis |
| `create_experiment` | Create ExperimentMatrix artifact |
| `update_ligand_result` | Update ligand status in matrix |
| `get_experiment_table` | Get LiveReport markdown table |

## Evolution Agent Tools

| Tool | Description |
|------|-------------|
| `add_entity_to_graph` | Create Neo4j node |
| `add_relationship` | Create Neo4j relationship |
| `discover_sar_insights` | Analyze docking for patterns |
| `suggest_molecule_modifications` | Gemini 3 molecule generation |
| `ucb_ranking` | Prioritize with UCB scores |
| `create_discovery_report` | Create DiscoveryReport artifact |

## Utility Functions

| Function | File | Description |
|----------|------|-------------|
| `calculate_box_size` | server.py | Ligand diameter + 10Å padding |
| `apply_cluster_weights` | server.py | Population-weighted averaging |
| `detect_ligand_properties` | server.py | RDKit property detection |
| `jaro_winkler_distance` | evolution/agent.py | Entity resolution |
