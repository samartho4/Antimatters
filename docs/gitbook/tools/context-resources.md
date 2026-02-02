# Context Resources: The External Brain

A scientist who doesn't read is just a lab technician. Antimatters uses **BioContext** to ensure every experiment is grounded in existing human knowledge.

## BioContext

*   **Implementation**: `uvx biocontext_kb@latest`
*   **Role**: The "Literature Search Engine."

### Capabilities
1.  **Literature Search**: Scans **PubMed** and **EuropePMC** for relevant papers.
2.  **Full Text Retrieval**: Pulls the actual content of open-access papers, not just abstracts.
3.  **Visualization**: Can generate string network images (`bc_get_string_network_image`) to visualize protein-protein interactions.

### Why It Matters
Before the **Engineer** spins up a simulation, the **Researcher** uses BioContext to answer: "Has this been tried before?" This prevents redundancy and ensures we stand on the shoulders of giants.
