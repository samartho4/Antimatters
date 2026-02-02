# Evolution: The Complexity Improver

The Evolution agent is the "brain." It turns raw data into **Wisdom**.

## Capabilities

*   **SAR Analysis**: Detects "Structure-Activity Relationships." It realizes *why* a molecule worked (e.g., "The hydroxyl group formed a hydrogen bond with Lys12").
*   **Graph Construction**: Updates the **MedGraphRAG** (Neo4j) with new insights.
*   **Self-Improvement**: Uses **rUCB** (Ranked Upper Confidence Bound) to suggest the *next* best molecule to test foundationally.

## Technical Implementation

*   **Model**: **Gemini 3 Pro** (Reasoning). It uses a high thinking budget (16k tokens) to analyze the physics.
*   **Tools**: `neo4j` (Graph DB), Internal SAR logic.
*   **Code**: `core/agents/antimatters/_subagents/evolution/agent.py`

> [!TIP]
> **Example Use**
> 
> **Input**: An `Experimental Matrix` for Fasudil binding to Tau.
> 
> **Evolution (Phoenix)**:
> 1. Analyzes the matrix: "Models 3, 7, and 12 showed strong binding (-8.0 kcal/mol)."
> 2. Identifies the cause: "Common interaction: H-bond with Tyr29."
> 3. **Reasoning**: "To improve this, we need to stabilize the aromatic ring interaction."
> 4. **Output**: A `Discovery Report` recommending a derivative of Fasudil with an added methyl group, and updating the Knowledge Graph.

## Why It Matters
This is the loop closer. It ensures that the system doesn't just "run runs"; it *learns*. Every experiment makes the next one smarter.
