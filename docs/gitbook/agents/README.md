# Agents

Antimatters is a **Multi-Agent System**. We don't have "one AI" that does everything. We have a squad of specialized professionals.

## The Workforce

Our agents map directly to the scientific method:

1.  **[Researcher](researcher.md): The Information Gatherer**
    *   *Role*: Hypothesis generation, literature synthesis (`BioContext`).
    *   *Codebase*: `core/agents/antimatters/_subagents/research/agent.py`
    
2.  **[Engineer](engineer.md): The Parallel Computation Engine**
    *   *Role*: Simulation (`Vina`), molecule prep (`RDKit`), HPC orchestration.
    *   *Codebase*: `core/agents/antimatters/_subagents/engineering/agent.py`

3.  **[Evolution](evolution.md): The Complexity Improver**
    *   *Role*: SAR analysis, GraphRAG construction, Self-Improvement.
    *   *Codebase*: `core/agents/antimatters/_subagents/evolution/agent.py`

## Orchestration

These agents communicate via the **Vibe Framework**, passing structured artifacts (Protocol $\to$ Matrix $\to$ Report) to ensure that the output of one becomes the valid input of the next.
