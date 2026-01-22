"""Antimatters - Multi-agent Scientific Research Platform

Orchestrates specialized agents for computational research tasks.
"""
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from core.agents.research import research_agent
from core.agents.engineering import engineer_agent
from core.agents.optimization import parallel_docking_agent, optimization_loop
from core.agents.callbacks.logging import before_tool_callback, after_tool_callback

# Load environment
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

root_agent = LlmAgent(
    name="antimatters_agent",
    model="gemini-2.0-flash",
    description="Antimatters: Multi-agent scientific research platform",
    instruction="""You are Antimatters, a scientific research assistant.

**AGENTS:**
1. research_agent: PED ensembles + BioContext (literature) + ChEMBL (compounds)
2. engineer_agent: prepare_ligand, cluster_conformations, dock_ensemble, analyze_interactions
3. parallel_docking_agent: Concurrent multi-ligand docking
4. optimization_loop: Iterative docking until quality threshold

**WORKFLOW:**
1. Research: Fetch PED ensemble, search literature/compounds
2. Prepare: Cluster conformations, prepare ligand PDBQT
3. Dock: Single, parallel, or iterative optimization
4. Analyze: Protein-ligand interactions

**RULES:**
- Pass exact paths/values between agents
- Check "success" field in responses
- Energy < -7 kcal/mol = drug-like""",
    tools=[
        AgentTool(agent=research_agent),
        AgentTool(agent=engineer_agent),
        AgentTool(agent=parallel_docking_agent),
        AgentTool(agent=optimization_loop),
    ],
    output_key="result",
)
