"""Parallel Agent: Concurrent multi-ligand docking."""
from google.adk.agents import LlmAgent, ParallelAgent
from core.mcp_servers.toolsets.docking import docking_tools


def create_dock_agent(name: str, smiles: str) -> LlmAgent:
    """Create a docking agent for a specific ligand."""
    return LlmAgent(
        name=f"dock_{name}",
        model="gemini-2.0-flash",
        description=f"Docking agent for {name}",
        instruction=f"""Dock {name} (SMILES: {smiles}):
1. Read state['pdb_path'] and state['representative_frames']
2. Call prepare_ligand with SMILES: {smiles}
3. Call dock_ensemble with pdbqt_path and representative_frames
4. Report binding energy""",
        tools=[docking_tools],
        output_key=f"dock_{name}_result",
    )


def create_parallel_docking_agent(ligands: dict) -> ParallelAgent:
    """Create a parallel docking agent for multiple ligands.

    Args:
        ligands: Dict of {name: smiles} for ligands to dock

    Returns:
        ParallelAgent configured to dock all ligands concurrently
    """
    return ParallelAgent(
        name="parallel_docking",
        sub_agents=[
            create_dock_agent(name, smiles)
            for name, smiles in ligands.items()
        ],
    )


# Default empty parallel agent (configure via create_parallel_docking_agent)
parallel_docking_agent = ParallelAgent(
    name="parallel_docking",
    sub_agents=[],
)
