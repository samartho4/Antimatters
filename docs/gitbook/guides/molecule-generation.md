# Molecule Generation

## Current State

SAR insights are discovered from docking results. Molecule generation is **partially implemented**.

## Implemented: SAR Discovery

```python
# evolution/agent.py L294-461
def discover_sar_insights(ligand_results: List[Dict]) -> List[SARInsight]:
    """
    Compares best vs worst binders to find patterns.
    """
    # Sort by energy
    best_half = sorted_results[:mid]
    worst_half = sorted_results[mid:]
    
    # Find interaction types unique to best binders
    beneficial = best_interactions - worst_interactions
    
    # Functional group correlations
    # IF aromatic_rings > 0 AND π-stacking detected → SAR insight
```

Outputs:
- `interaction_preference` — which contacts help binding
- `residue_hotspot` — which residues are key
- `functional_group_correlation` — which groups enable interactions
- `conformational_selection` — which protein clusters favor binding

## Planned: Gemini-Based Suggestions

```python
# evolution/agent.py (planned)
def suggest_molecule_modifications(insight_ids, seed_smiles, n_suggestions=3):
    """
    Use Gemini to suggest SMILES modifications based on SAR.
    """
    insights = [get_insight(id) for id in insight_ids]
    
    prompt = f"""SAR insights: {format(insights)}
    Seed: {seed_smiles}
    Suggest modifications."""
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    # Parse JSON suggestions
```

## Data Flow

```
dock_ensemble results
      ↓
discover_sar_insights()
      ↓
SARInsight objects (stored in graph + SQLite)
      ↓
suggest_molecule_modifications() [planned]
      ↓
New SMILES for next docking cycle
```
