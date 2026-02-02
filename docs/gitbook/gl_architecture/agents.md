# Agent Architecture

## Overview

```
coordinator.py
└── root_agent (LlmAgent, gemini-2.5-pro)
    └── docking_workflow (SequentialAgent)
        ├── research_agent
        ├── engineer_agent
        └── evolution_agent
```

`SequentialAgent` guarantees order. Each agent receives previous agent's output via `tool_context.state`.

## Research Agent

**File**: `agents/antimatters/_subagents/research/agent.py`

**Tools**:
- `fetch_ped_ensemble` — downloads PDB from Protein Ensemble Database
- `search_compounds` — ChEMBL similarity search
- `bc_get_europepmc_articles` — literature via BioContext MCP
- `create_protocol`, `validate_protocol`

**Output**: Protocol artifact with target info, ligand candidates, evidence.

## Engineer Agent

**File**: `agents/antimatters/_subagents/engineering/agent.py`

**Tools**:
- `cluster_conformations` — t-SNE + k-medoids on RMSD
- `prepare_ligand` — SMILES → 3D mol2 → PDBQT
- `dock_ensemble` — AutoDock Vina per cluster
- `analyze_interactions` — H-bonds, hydrophobic, aromatic
- `create_experiment`, `update_ligand_result`

**Parallel execution**: `ParallelAgent` spawns one sub-agent per ligand. Each updates ExperimentMatrix independently.

```python
# engineering/agent.py L398-420
def create_parallel_docking_agent(ligands):
    sub_agents = [create_ligand_agent(lig["name"], lig["smiles"]) for lig in ligands]
    return ParallelAgent(name="parallel_docking", sub_agents=sub_agents)
```

**Output**: ExperimentMatrix artifact with per-ligand status/energy.

## Evolution Agent

**File**: `agents/antimatters/_subagents/evolution/agent.py`

**Tools**:
- `analyze_experiment_results` — UCB rankings, SAR discovery
- `propose_knowledge_items`, `approve_knowledge_items` — propose/approve pattern
- `build_knowledge_graph` — Neo4j entities/relationships
- `create_discovery_report`

**UCB Ranking** (rUCB algorithm):
```python
# evolution/agent.py L250-287
ucb = -energy + c * sqrt(log(N)/n)  # exploitation + exploration
```

**SAR Discovery**:
```python
# evolution/agent.py L294-461
# Compares best vs worst binders to find patterns:
# - Interaction preferences
# - Residue hotspots
# - Functional group correlations
```

**Output**: DiscoveryReport artifact + Neo4j graph.

## State Sharing

```python
# Research stores:
tool_context.state["latest_protocol_id"] = artifact["id"]

# Engineer reads:
protocol_id = tool_context.state.get("latest_protocol_id")
```

## Models Used

| Agent | Model | Reason |
|-------|-------|--------|
| Coordinator | gemini-2.5-pro | 2M context for workflow |
| Research | gemini-2.5-flash | Fast tool calls |
| Engineer | gemini-2.5-flash | Parallel execution |
| Evolution | gemini-2.5-flash | Analysis + graph |
| Frontend | gemini-3-pro-preview | Deep reasoning (16k thinking) |
