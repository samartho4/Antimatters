# Engineer: The Parallel Computation Engine

The Engineer agent is the "hands" of the system. It scales your intent.

## Capabilities

*   **Ensemble Docking**: Instead of one simulation, it runs dozens. It docks ligands against entire **PED** ensembles to catch transient binding pockets essential for IDPs.
*   **Ligand Prep**: Uses **RDKit** to convert 1D SMILES into 3D energy-minimized structures.
*   **Parallelism**: Spawns concurrent sub-agents to maximize throughput.

## Technical Implementation

*   **Model**: Gemini 2.5 Flash (Instruction following).
*   **Tools**: `dock_ensemble` (Custom Vina MCP), `prepare_ligand` (RDKit).
*   **Code**: `core/agents/antimatters/_subagents/engineering/agent.py`

> [!TIP]
> **Example Use**
> 
> **Input**: A `Protocol` listing "Fasudil" and "Tau (PED00017)".
> 
> **Engineer (Falcon)**:
> 1. Fetches `PED00017` (20 structural models).
> 2. Prepares "Fasudil" 3D conformers.
> 3. **Spawns 20 sub-processes**: Docks Fasudil against *each* protein model simultaneously.
> 4. **Output**: An `Experimental Matrix` showing binding energies ranging from -4.5 to -8.2 kcal/mol across the ensemble.

## Why It Matters
IDPs are moving targets. Docking against one structure is useless. The **Engineer** automates the complexity of *ensemble* simulation, doing in minutes what would take a human days of script-writing.
