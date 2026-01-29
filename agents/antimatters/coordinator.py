"""Antimatters - Multi-agent Scientific Research Platform

THREE specialized agents for IDP ensemble docking:
1. Research Agent → Protocol artifact (literature validation)
2. Engineer Agent → ExperimentMatrix artifact (parallel docking, real-time updates)
3. Evolution Agent → DiscoveryReport artifact (SAR, knowledge graph)

MCP Tools are the "backward dots":
- docking: prepare_ligand, cluster_conformations, dock_ensemble, analyze_interactions
- ped: fetch_ped_ensemble
- chembl: search_compounds, similarity_search
- biocontext: literature search (EuropePMC, Google Scholar, bioRxiv)

Inspired by:
- Antigravity: AI-native artifacts, non-blocking agent work
- Microsoft Discovery: Knowledge graph for scientific reasoning
- Schrödinger LiveDesign: Real-time matrix updates
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent 'agents' directory to path for shared imports
# This is standard ADK practice per: https://github.com/google/adk-python/discussions/3117
agents_dir = Path(__file__).parent.parent
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

# Load environment first
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Configure Gemini with retry BEFORE importing agents
# This applies exponential backoff to all Gemini API calls
import gemini_config  # noqa: F401

from google.adk.agents import LlmAgent, SequentialAgent  # SequentialAgent for guaranteed multi-step execution
from config import MODELS  # Import model configuration

from ._subagents.research import research_agent
from ._subagents.engineering import engineer_coordinator  # NEW: Custom BaseAgent
from ._subagents.evolution import evolution_agent

# Docking workflow: Guaranteed sequential execution (research → engineer → evolution)
docking_workflow = SequentialAgent(
    name="docking_workflow",
    description="Complete docking workflow: research → engineer_coordinator → evolution (sequential, guaranteed)",
    sub_agents=[research_agent, engineer_coordinator, evolution_agent],
)

root_agent = LlmAgent(
    name="antimatters_agent",
    model=MODELS.coordinator,  # Use configured Gemini 3 Pro
    description="Antimatters: IDP docking platform with guaranteed sequential workflow execution",
    instruction="""You coordinate a complete IDP ligand docking workflow.

**YOUR JOB:**
Transfer ALL user requests to **docking_workflow** (SequentialAgent).

The workflow automatically executes three agents sequentially:
1. **research_agent**: Gather PED ensembles, ChEMBL compounds, literature
2. **engineer_agent**: Cluster conformations, prepare ligands, run docking
3. **evolution_agent**: Analyze results, build knowledge graph, generate insights

After workflow completes, summarize the key findings to the user.

**ARTIFACTS CREATED:**
- Protocol (research_agent): Target validation + ligand selection
- ExperimentMatrix (engineer_agent): Real-time docking status per ligand
- DiscoveryReport (evolution_agent): Rankings, SAR patterns, graph entities

**IMPORTANT:**
- SequentialAgent guarantees all 3 agents run in order
- You make ONE transfer, workflow handles execution
- If tools fail, agents report errors (no hallucination)""",
    tools=[],
    sub_agents=[docking_workflow],
    output_key="result",
)
