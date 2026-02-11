<p align="center">
  <h1 align="center">Antimatters</h1>
  <p align="center">
    <strong> "Serendipitize" Bits to Atoms </strong>
    <br />
    <em>AI that researches, experiments, and discovers—not just analyzes</em>
  </p>
</p>

<p align="center">
  <a href="https://deepwiki.com/samartho4/Antimatters"><img src="https://img.shields.io/badge/docs-deepwiki-blue" alt="Documentation"></a>
  <a href="https://github.com/samartho4/asclepius.git"><img src="https://img.shields.io/badge/prototype-asclepius-green" alt="Prototype"></a>
  <img src="https://img.shields.io/badge/version-0.0.7-blue" alt="Version">
  <img src="https://img.shields.io/badge/Gemini-3-orange" alt="Gemini 3">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## The Paradigm Shift

Language models have evolved from learning language—a human's expression of thinking—to actually thinking. While significant effort models the natural world, less focus has been on modeling the process of Science itself.

We believe models with infinite intelligence cannot solve complex problems like neurodegenerative diseases alone—**experiments are still needed**. Antimatters is our answer: AI agents that don't just query knowledge, but run experiments and build structured world models from the results.

This is the emerging paradigm of [Agentic Science](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1649155/full)—a closed loop of hypothesis → experiment → analysis → refined hypothesis, with AI orchestrating each step.

---

## The Loop: Information → Computation → Evolution

Three specialized agents operate in sequence, each handing off structured artifacts to the next:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   INFORMATION          COMPUTATION           EVOLUTION                  │
│   ─────────────        ─────────────         ──────────                 │
│   Research Agent       Engineering Agent     Evolution Agent            │
│                                                                         │
│   • Query PED, ChEMBL  • Cluster ensembles   • Extract SAR patterns    │
│   • Validate binding   • Dock via Vina       • Build knowledge graph   │
│     sites from lit     • Analyze contacts    • Generate figures        │
│   • Create protocol    • Score affinities    • Propose molecules       │
│                                                                         │
│   Output: Protocol     Output: Experiment    Output: Discovery         │
│   Artifact             Matrix                Report                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Knowledge Graph      │
                    │    (Neo4j)              │
                    │                         │
                    │    Entities, relations, │
                    │    SAR patterns persist │
                    │    across sessions      │
                    └─────────────────────────┘
```

The Knowledge Graph is the key differentiator. Unlike stateless LLM sessions, Antimatters remembers what worked, what didn't, and why—a **structured world model** that makes each subsequent experiment smarter.

---

## Proof Case: The "Undruggable" Proteins

To validate this paradigm, we tackled one of drug discovery's hardest problems.

Intrinsically disordered proteins (IDPs) lack stable secondary/tertiary structures—existing instead as ensembles of rapidly interconverting conformations. This makes them effectively **"undruggable"** by conventional methods that assume rigid binding pockets.

**α-synuclein**, whose aggregation causes neuronal death in Parkinson's disease, is the canonical example. Our agents:

1. **Information**: Fetched 576 conformations from PED, validated binding sites (Y125, Y133, Y136) from literature
2. **Computation**: Clustered via t-SNE, docked across representatives with AutoDock Vina, analyzed contacts with MDTraj
3. **Evolution**: Discovered which structural features predict binding, proposed novel molecules

The result: correctly predicted relative binding affinities matching [experimental NMR spectroscopy](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00370) (Ligand-47 > Fasudil > Ligand-23), and reproduced atomic-resolution details of ligand binding modes.

Because it's based on physics and first principles, we can treat computational data as if it were a biological assay.

---

## Gemini 3: The Engine

Antimatters runs on **Gemini 3**, exploiting capabilities purpose-built for agentic workflows:

| Capability | Application |
|------------|-------------|
| **Code Execution** | Evolution agent writes matplotlib/py3Dmol code, runs it, returns rendered figures—no templates |
| **Multimodal Reasoning** | Researchers annotate 3D structures; Gemini reasons about spatial relationships at atomic resolution |
| **Tool Orchestration** | Coordinates MCP servers (docking, databases, literature) with automatic error recovery |
| **Long Context** | Full experiment history persists across agent handoffs without summarization loss |

```python
# Model configuration spreads load across Flash and Pro
class ModelConfig:
    coordinator: str = "gemini-3-flash-preview"   # Fast routing
    research:    str = "gemini-3-flash-preview"   # Multimodal literature
    engineering: str = "gemini-3-flash-preview"   # Tool coordination
    evolution:   str = "gemini-3-pro-preview"     # SAR requires Pro reasoning
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                   │
│  Sidebar │ Computation (Chat) │ Evolution (KG) │ Artifacts  │
└────────────────────────────┬────────────────────────────────┘
                             │ AG-UI SSE
┌────────────────────────────▼────────────────────────────────┐
│               Google ADK Multi-Agent System                  │
│                                                              │
│     Research ──────► Engineering ──────► Evolution          │
│                                                              │
└────────────────────────────┬────────────────────────────────┘
                             │ MCP Protocol
┌────────────────────────────▼────────────────────────────────┐
│                     Tool Servers                             │
│   PED (ensembles) │ ChEMBL (compounds) │ Docking (Vina)    │
└─────────────────────────────────────────────────────────────┘
```

The stack: FastAPI for AG-UI streaming, Google ADK for agent orchestration, MCP for tool integration, Neo4j for persistent knowledge.

---

## Quick Start

```bash
# Requirements: Python 3.10+, Node.js 18+, GOOGLE_API_KEY in .env

cd core
python -m api.agui_server &
cd apps/research-ui && npm run dev &
```

Navigate to `http://localhost:5173`. Try:

> *"Dock fasudil to alpha-synuclein and analyze the binding pattern"*

---

## Two Modes

| Mode | Description |
|------|-------------|
| **Serendipitize** | Full pipeline. Research validates, Engineering runs simulations, Evolution extracts patterns. Builds the knowledge graph. |
| **Planning** | Uses Evolution agent only, querying existing Knowledge Graph to propose experiments or generate candidate molecules. Fast iteration once you have data. |

---

## Project Structure

```
core/
├── agents/
│   ├── config.py              # Models, timeouts, protein/ligand definitions
│   └── antimatters/
│       ├── coordinator.py     # Mode routing
│       └── _subagents/
│           ├── research/      # Literature + databases
│           ├── engineering/   # Parallel docking
│           └── evolution/     # SAR + visualization
├── mcp_servers/
│   ├── docking/server.py      # Vina + MDTraj
│   ├── ped/server.py          # Protein Ensemble Database
│   └── toolsets/              # MCP client wrappers
├── api/agui_server.py         # FastAPI backend
└── apps/research-ui/          # React frontend
```

---

## API

```bash
# Run experiment (SSE stream)
POST http://localhost:8002/
{"messages": [{"id": "1", "role": "user", "content": "..."}], "mode": "serendipitize"}

# Get knowledge graph
GET http://localhost:8002/evolution/graph
```

---

## Extending

Add an MCP server:

```python
from fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def my_tool(input: str) -> dict:
    """Description for the agent."""
    return {"result": "..."}
```

Add an agent:

```python
from google.adk.agents import LlmAgent
from agents.config import MODELS

my_agent = LlmAgent(
    name="my_agent",
    model=MODELS.research,
    instruction="...",
    tools=[my_tools],
)
```

---

## References

- [Ensemble Docking for IDPs](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c00370) — Dhar et al., *J. Chem. Inf. Model.* 2025
- [Agentic AI for Scientific Discovery](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1649155/full) — *Frontiers in AI* 2025
- [Google ADK](https://google.github.io/adk-docs/) · [Gemini 3](https://ai.google.dev/gemini-api/docs/gemini-3) · [MCP](https://modelcontextprotocol.org/)

---

## License

MIT

---

<p align="center">
  <a href="https://deepwiki.com/samartho4/Antimatters">Documentation</a> · <a href="https://github.com/samartho4/asclepius.git">Initial Prototype from AI studio</a>
</p>
