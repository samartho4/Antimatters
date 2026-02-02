# Artifacts

Three artifact types, each created by a different agent.

## Protocol

**Created by**: Research Agent

Target and ligand selection with evidence.

```python
{
    "type": "protocol",
    "content": {
        "target": "Alpha-Synuclein",
        "ped_id": "PED00006e001",
        "binding_site": [125, 133, 136],
        "ligands": [
            {"name": "Fasudil", "smiles": "...", "chembl_id": "CHEMBL888"}
        ],
        "validation_status": "validated",
        "evidence": ["PMC123456"]
    }
}
```

## ExperimentMatrix

**Created by**: Engineer Agent

Per-ligand docking status. Updates in real-time as parallel agents complete.

```python
{
    "type": "experiment_matrix",
    "content": {
        "experiment_name": "Alpha-syn screening",
        "protein_pdb_path": "/tmp/PED00006.pdb",
        "n_clusters": 20,
        "ligand_results": [
            {
                "ligand_name": "Fasudil",
                "smiles": "...",
                "status": "completed",  # queued | running | completed | failed
                "best_energy": -7.2,
                "best_cluster": 3,
                "interaction_types": ["H-bond", "aromatic"]
            }
        ]
    }
}
```

## DiscoveryReport

**Created by**: Evolution Agent

Analysis summary with rankings, SAR insights, and graph link.

```python
{
    "type": "discovery_report",
    "content": {
        "executive_summary": "...",
        "energy_stats": {
            "n_completed": 6,
            "best_energy": -7.2,
            "n_drug_like": 3
        },
        "ucb_rankings": [...],
        "sar_insights": [...],
        "knowledge_graph_id": "...",
        "recommendations": [...]
    }
}
```

## Storage

Artifacts stored in SQLite via `ArtifactService`:

```python
# api/services.py
CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    type TEXT NOT NULL,
    content TEXT NOT NULL,  -- JSON
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
```

## Cross-Agent Discovery

```python
# Research sets:
tool_context.state["latest_protocol_id"] = protocol["id"]

# Engineer reads:
protocol_id = tool_context.state.get("latest_protocol_id")
```
