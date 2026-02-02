# The Vibe Framework

The **Vibe Framework** is the operating system of Antimatters. It is not just a collection of tools; it is an **Agentic Workflow** designed from first principles to handle the complexity of modern science.

## From Static to Agentic

Traditional tools like ELNs (Electronic Lab Notebooks) and LIMS (Lab Information Management Systems) are passive. They wait for you to type data in. They are **Repositories**.

The Vibe Framework is **Active**. It consists of specialized Agents that *reason*, *execute*, and *evolve*.

| Traditional Workflow (Static) | Antimatters Vibe (Agentic) |
| :--- | :--- |
| **Protocol** is a PDF file. | **Protocol** is a structured, validated JSON artifact. |
| **Experiment** is a human pipetting. | **Experiment** is the **Engineer** agent spawning parallel sub-agents. |
| **Insight** is a mental note. | **Discovery Report** is an evolvable Graph node (MedGraphRAG). |

## Core Artifact Flow

Our code defines three core artifacts that drive this flow. This isn't a metaphor; it's how `core/agents/` is architected:

### 1. The Protocol (The Plan)
*   **Agent**: **Researcher**
*   **Input**: Unstructured intent ("Find inhibitors for Tau").
*   **Process**: Queries **BioContext** (Information) to validate targets.
*   **Output**: A rigorous, citation-backed plan.
*   **The Vibe**: "Trust but Verify." Nothing proceeds without a citation.

### 2. The Experimental Matrix (The Reality)
*   **Agent**: **Engineer**
*   **Input**: The Protocol.
*   **Process**: Spawns **ParallelAgents** to execute `dock_ensemble` across the **PED** structure.
*   **Output**: A dynamic matrix of binding energies ($\Delta G$) and interaction counts.
*   **The Vibe**: "Massive Parallelism." The system creates its own compute resources to solve the problem.

### 3. The Discovery Report (The Insight)
*   **Agent**: **Evolution**
*   **Input**: The Matrix + The Protocol.
*   **Process**: Uses **Gemini 3** to deduce Structure-Activity Relationships (SAR).
*   **Output**: A graph update. New nodes are added to **Neo4j**, linking the *Physics* (Binding Energy) back to the *Information* (Literature).
*   **The Vibe**: "Evolve Understanding." The end of one experiment is the smarter beginning of the next.

This framework ensures that **Information** (Protocol) is converted into **Energy** (Matrix) and solidified into **Matter** (Graph/Insight).
