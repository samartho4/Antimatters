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
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent

from ._subagents.research import research_agent
from ._subagents.engineering import engineer_agent
from ._subagents.evolution import evolution_agent

# Load environment
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Docking workflow: Guaranteed sequential execution (research → engineer → evolution)
docking_workflow = SequentialAgent(
    name="docking_workflow",
    description="Complete docking workflow: research → engineer → evolution (sequential, guaranteed)",
    sub_agents=[research_agent, engineer_agent, evolution_agent],
)

root_agent = LlmAgent(
    name="antimatters_agent",
    model="gemini-3-pro-preview",  # Pro for handling large tool context (37K tokens)
    description="Antimatters: AI-native IDP docking platform - always runs full workflow",
    instruction="""You are Antimatters, an AI-native IDP ligand docking coordinator.

**YOUR WORKFLOW:**

You have ONE workflow: **docking_workflow** (SequentialAgent)
- Automatically runs: research → engineer → evolution (all 3 steps sequentially)
- For ALL requests, transfer to docking_workflow
- The workflow handles everything:
  - Research-only queries: research gathers info, engineer/evolution skip (no compounds to dock)
  - Docking queries: all 3 agents execute fully

**WHAT TO DO:**

Simply transfer ALL user requests to **docking_workflow**.
- Docking request: "Dock fasudil to PED00024" → docking_workflow runs all 3 agents
- Research request: "Search ChEMBL for fasudil" → docking_workflow runs research, others return "no docking requested"

**IMPORTANT:**
- SequentialAgent guarantees all 3 agents execute in order
- You make ONE transfer, the workflow handles the rest
- After workflow completes, summarize results to user

**ARTIFACTS:**
- Protocol: Research blueprint + literature validation
- ExperimentMatrix: Ligand × property matrix (real-time updates)
- DiscoveryReport: UCB rankings, SAR insights, knowledge graph

**KNOWLEDGE GRAPH (MedGraphRAG):**
- Level 1: Experimental data (docking results)
- Level 2: Reference data (ChEMBL, PED)
- Level 3: Controlled vocabularies (UniProt, PubChem)
- Energy < -7 kcal/mol = drug-like""",
    tools=[],
    sub_agents=[
        docking_workflow,  # Only workflow - handles all requests
    ],
    output_key="result",
)
