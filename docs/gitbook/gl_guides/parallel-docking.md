# Parallel Docking

## How It Works

Engineer agent uses `ParallelAgent` to dock multiple ligands concurrently.

### Factory Function

```python
# engineering/agent.py L398-420
def create_parallel_docking_agent(ligands):
    """Create a ParallelAgent with one sub-agent per ligand."""
    sub_agents = []
    for lig in ligands:
        sub_agents.append(
            create_ligand_agent(lig["name"], lig["smiles"])
        )
    
    return ParallelAgent(
        name="parallel_docking",
        sub_agents=sub_agents
    )
```

### Per-Ligand Agent

Each sub-agent runs:
1. `prepare_ligand(smiles)` — SMILES → PDBQT
2. `dock_ensemble(pdbqt, frames, protein)` — Vina docking
3. `analyze_interactions(protein, ligand)` — contacts
4. `update_ligand_result(experiment_id, ligand_name, result)` — update matrix

### Real-Time Updates

```python
# engineering/agent.py
def update_ligand_result(experiment_artifact_id, ligand_name, result):
    """Update a single ligand's status in the ExperimentMatrix."""
    artifact = read_artifact(experiment_artifact_id)
    content = artifact["content"]
    
    for lig in content["ligand_results"]:
        if lig["ligand_name"] == ligand_name:
            lig.update(result)
            break
    
    update_artifact(experiment_artifact_id, content)
```

Frontend receives SSE events as each ligand completes.

## Example Flow

```
Protocol: 6 ligands
      ↓
ParallelAgent (6 sub-agents)
      ↓
┌───────┬───────┬───────┬───────┬───────┬───────┐
│ Lig 1 │ Lig 2 │ Lig 3 │ Lig 4 │ Lig 5 │ Lig 6 │
└───┬───┴───┬───┴───┬───┴───┬───┴───┬───┴───┬───┘
    │       │       │       │       │       │
    └───────┴───────┼───────┴───────┴───────┘
                    ↓
          ExperimentMatrix updated per completion
```

## UCB Ranking After Completion

```python
# evolution/agent.py L250-287
def calculate_ucb_ranking(ligand_results):
    """UCB = -energy + c * sqrt(log(N)/n)"""
    # Balances exploitation (best energy) vs exploration (uncertainty)
```
