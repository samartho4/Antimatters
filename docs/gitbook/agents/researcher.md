# Researcher: The Information Gatherer

The Researcher agent is your "First Principle" validator. It ensures that no compute is wasted on scientific hallucinations.

## Capabilities

*   **Literature Synthesis**: Uses `BioContext` to pull abstracts and full-text papers.
*   **Target Validation**: Checks if a target (e.g., "Alpha-Synuclein") has valid chemical biology data in **ChEMBL**.
*   **Protocol Generation**: Authors the `Protocol` artifact, the "source code" for the experiment.

## Technical Implementation

*   **Model**: Gemini 2.5 Flash (optimized for high-speed tool calling).
*   **Tools**: `biocontext_kb` (via `uvx`), `chembl` (via Node.js).
*   **Code**: `core/agents/antimatters/_subagents/research/agent.py`

> [!TIP]
> **Example Use**
> 
> **Scientist**: "Find me small molecules that bind to the N-terminus of Tau."
> 
> **Researcher (Crow)**: 
> 1. Queries **BioContext (PubMed)** for "Tau N-terminus inhibitors".
> 2. Retrieves 5 relevant papers.
> 3. extracted candidate ligands: "Methylene Blue", "EGCG".
> 4. Validates structures in **ChEMBL**.
> 5. **Output**: A `Protocol` suggesting these ligands for docking against `PED00017`.

## Why It Matters
In a standard LLM chat, the AI might hallucinate a molecule. The **Researcher** doesn't guess; it cites. It bridges the gap between "I think" and "The literature says."
