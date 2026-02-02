# Databases: The Foundation

Scientific data needs structure. Antimatters uses a tiered storage architecture to balance speed, structure, and semantic depth.

## 1. PED (Protein Ensemble Database)
*   **Function**: Source of truth for IDP structures.
*   **Usage**: We fetch extensively validated structural ensembles (e.g., `PED00006` for Alpha-Synuclein) to ensure our starting point is biologically accurate.

## 2. Neo4j (Graph Database)
*   **Function**: The "Long-Term Memory."
*   **Structure**: **MedGraphRAG** compliance (Level 1: Experiments, Level 2: Reference, Level 3: Vocabulary).
*   **Value**: Enables the **Evolution** agent to perform multi-hop reasoning (e.g., "Find all ligands that bind to residue 125 with hydrophobic interactions").

## 3. SQLite (Operational Store)
*   **Function**: The "Working Memory."
*   **Usage**: Stores high-speed transactional data (Artifact states, user sessions, chat history) for the Agent interaction loop.

## 4. Vector Store (Embeddings)
*   **Function**: Semantic Search.
*   **Usage**: Stores `text-embedding-004` vectors of all specific insights, allowing users to ask natural language questions like "What works best against Alpha-Synuclein?"
