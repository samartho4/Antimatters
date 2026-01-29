# Antimatters

Multi-agent scientific discovery platform. Current application: IDP ensemble docking.

## Problem

Scientific workflows lose context. You run an experiment, analyze results, form hypotheses—then start the next experiment from scratch. Insights don't accumulate. Tools don't talk to each other.

## Solution

Agents that remember. Each experiment builds a knowledge graph. SAR patterns from docking results become queryable insights. The next hypothesis starts from accumulated understanding, not zero.

## How It Works

Three agents run sequentially:
1. **Research** — fetches protein ensemble (PED), finds compounds (ChEMBL), gathers literature
2. **Engineer** — clusters conformations, prepares ligands, runs docking in parallel
3. **Evolution** — discovers SAR patterns, builds knowledge graph, stores learnings

Each produces an artifact: Protocol → ExperimentMatrix → DiscoveryReport.

## Stack

- **Agents**: Google ADK (`SequentialAgent`, `ParallelAgent`)
- **LLM**: Gemini 2.5 Flash/Pro (agents), Gemini 3 Pro Preview (frontend reasoning)
- **Graph**: Neo4j AuraDB
- **Docking**: MCP server wrapping AutoDock Vina
- **Backend**: FastAPI + SQLite
- **Frontend**: React + SSE streaming

## Gemini

- **gemini-2.5-pro** — Coordinator (2M context for complex workflows)
- **gemini-2.5-flash** — Agent execution (fast tool calls)
- **gemini-3-pro-preview** — Frontend reasoning (16k thinking budget)
- **text-embedding-004** — Semantic search over knowledge

## Structure

```
core/
├── agents/antimatters/           # Agent definitions
│   ├── coordinator.py            # Root agent + SequentialAgent
│   └── _subagents/
│       ├── research/agent.py     # Protocol creation
│       ├── engineering/agent.py  # Parallel docking
│       └── evolution/agent.py    # KG + SAR
├── api/
│   ├── agui_server.py            # FastAPI + SSE
│   └── services.py               # SQLite services
├── mcp_servers/docking/          # MCP tools
└── apps/research-ui/             # React frontend
```

## Running

```bash
export GOOGLE_API_KEY="..."
export NEO4J_URI="neo4j+s://..."
export NEO4J_PASSWORD="..."

cd core
uvicorn api.agui_server:app --port 8001

cd apps/research-ui
npm run dev
```

## Documentation

- [Knowledge Graph](architecture/knowledge-graph.md)
- [Agent Architecture](architecture/agents.md)
- [MCP Docking Tools](components/mcp-docking.md)
- [Artifacts](architecture/artifacts.md)
- [Backend](components/backend.md)
- [Async Patterns](architecture/async-patterns.md)
