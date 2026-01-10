"""Gemini agent with 6 tools for IDP discovery."""

import anthropic


def create_idp_discovery_agent():
    """Initialize Gemini agent with 6 tools."""
    client = anthropic.Anthropic()
    
    tools = [
        {
            "name": "load_pdb",
            "description": "Load and parse PDB protein structure files",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pdb_id": {"type": "string", "description": "PDB identifier"}
                },
                "required": ["pdb_id"]
            }
        },
        {
            "name": "validate_structure",
            "description": "Validate protein structure integrity",
            "input_schema": {
                "type": "object",
                "properties": {
                    "structure": {"type": "string", "description": "Structure data"}
                },
                "required": ["structure"]
            }
        },
        {
            "name": "run_docking",
            "description": "Execute molecular docking simulation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "protein": {"type": "string"},
                    "ligand": {"type": "string"}
                },
                "required": ["protein", "ligand"]
            }
        },
        {
            "name": "calculate_boltzmann",
            "description": "Calculate Boltzmann-averaged binding energies",
            "input_schema": {
                "type": "object",
                "properties": {
                    "energies": {"type": "array", "items": {"type": "number"}}
                },
                "required": ["energies"]
            }
        },
        {
            "name": "extract_rules",
            "description": "Extract symbolic rules from data using PySR",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Input data path"}
                },
                "required": ["data"]
            }
        },
        {
            "name": "predict_idp",
            "description": "Predict intrinsically disordered protein regions",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sequence": {"type": "string", "description": "Protein sequence"}
                },
                "required": ["sequence"]
            }
        }
    ]
    
    return client, tools


def run_agent(query: str):
    """Run the IDP discovery agent with a query."""
    client, tools = create_idp_discovery_agent()
    
    messages = [
        {"role": "user", "content": query}
    ]
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    return response


if __name__ == "__main__":
    result = run_agent("Analyze protein structure for IDP regions")
    print(result)
