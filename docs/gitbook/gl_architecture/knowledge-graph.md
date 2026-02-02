# Knowledge Graph

## Implementation

Neo4j stores a 3-level graph structure (based on MedGraphRAG pattern):

```
Level 1: Experimental    →   Level 2: Reference     →   Level 3: Vocabulary
(our docking results)        (ChEMBL, PED entries)      (UniProt, PubChem IDs)
```

### Entities

```python
# evolution/agent.py L92-106
@dataclass
class ScientificEntity:
    entity_id: str
    entity_type: str    # Protein, Ligand, Residue, Interaction
    level: int          # 1=experimental, 2=reference, 3=vocabulary
    name: str
    properties: Dict
    external_ids: Dict  # uniprot, chembl, pubchem
```

### Relationships

```python
# evolution/agent.py L109-124
@dataclass
class ScientificRelationship:
    relationship_id: str
    subject_id: str
    predicate: str      # DOCKS_TO, INTERACTS_WITH, CORRESPONDS_TO
    object_id: str
    properties: Dict
```

### Entity Resolution

Uses Jaro-Winkler similarity to link experimental entities to reference databases:

```python
# evolution/agent.py L147-212
def jaro_winkler_similarity(s1: str, s2: str) -> float:
    # Matches "α-synuclein" to "alpha-synuclein" to PED00006
```

### Neo4j Persistence

```python
# evolution/agent.py L799-867
def build_knowledge_graph(experiment_artifact_id, ...):
    driver, database = get_neo4j_driver()
    with driver.session(database=database) as session:
        for entity in entities:
            session.run("MERGE (e:$($type) {entity_id: $id}) SET e += $props", ...)
        for rel in relationships:
            session.run("MATCH (a {entity_id: $s}), (b {entity_id: $o}) MERGE (a)-[r:$($p)]->(b)", ...)
```

## SQLite Storage

Backend also stores knowledge items in SQLite for UI display:

```python
# api/services.py KnowledgeService
def create(workspace_id, knowledge_type, title, content, tags):
    # Types: insight, procedure, result, feedback, reference
```

Semantic search uses `text-embedding-004`:

```python
def semantic_search(self, query, limit=5):
    query_embedding = genai.embed_content(model="models/text-embedding-004", ...)
    # Cosine similarity against stored embeddings
```

## Status

- [x] Entity/relationship dataclasses
- [x] Jaro-Winkler resolution
- [x] Neo4j driver integration
- [x] SQLite knowledge items
- [ ] Embedding population (column exists, not always populated)
