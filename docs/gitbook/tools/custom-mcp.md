# Custom MCP: The Physics Engine

Scientific discovery requires more than text; it requires **Physics**. Our Custom MCP server provides the "hard science" capabilities that allow Antimatters to validate molecules in silico.

## The Toolset

### 1. AutoDock Vina (Simulation)
*   **Function**: `dock_ensemble`
*   **Role**: Performs the actual energy minimization and binding scoring.
*   **Why**: It provides the ground truth ($\Delta G$) for whether a molecule physically fits a protein.

### 2. RDKit (Cheminformatics)
*   **Function**: `prepare_ligand`
*   **Role**: Handles the messy reality of Chemistry. Converts 1D SMILES strings into valid, 3D structures with correct protonation states.
*   **Why**: You can't dock a string. You need a 3D object.

### 3. PED (Protein Ensemble Database)
*   **Function**: `fetch_ped_ensemble`
*   **Implementation**: Python-based tool (`core/mcp_servers/ped/server.py`) wrapping the `idpet` library.
*   **Role**: Provides the **Structure** for the simulation. It downloads experimentally validated conformational ensembles of IDPs.
*   **Why**: IDPs are flexible. A single PDB file is a lie. PED gives us the "video" instead of the "snapshot."

## Integration
These tools run on the **Engineer** agent's local host, often containerized or parallelized across an HPC cluster, ensuring low-latency access to heavy compute.
