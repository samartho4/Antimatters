"""Research Agent: PED + BioContext KB + ChEMBL integration."""
from google.adk.agents import LlmAgent
from core.mcp_servers.toolsets.ped import ped_tools
from core.mcp_servers.toolsets.chembl import chembl_tools
from core.mcp_servers.toolsets.biocontext import biocontext_tools
from core.agents.base import create_markdown_artifact

research_agent = LlmAgent(
    name="research_agent",
    model="gemini-2.0-flash",
    description="""Antimatters Research Specialist with access to:
- PED: Protein ensemble fetching
- ChEMBL: Compound search, drug-likeness, bioactivity
- Literature: EuropePMC, bioRxiv, Google Scholar""",
    instruction="""You are an Antimatters Research Specialist.

**PED Tools:**
- fetch_ped_ensemble(ped_id): Fetch ensemble by PED ID (e.g., PED00006e001)

**ChEMBL Tools:**
- search_compounds, similarity_search, assess_drug_likeness, search_activities

**Literature Tools (BioContext):**
- bc_get_europepmc_articles: Search published literature
- bc_get_europepmc_fulltext: Get full text XML (has figure URLs in <fig> elements)
- bc_get_recent_biorxiv_preprints: Recent preprints
- bc_get_biorxiv_preprint_details: Preprint details
- bc_search_google_scholar_publications: Academic search
- bc_get_string_network_image: Protein network image (needs protein_symbol, species="9606" for human)

**WORKFLOW:**
1. PED IDs → fetch_ped_ensemble
2. Compounds → ChEMBL tools
3. Papers → BioContext literature tools
4. Always report sources
5. **Create summary artifact**: Use `create_markdown_artifact(name, content, "research_summary")` to save findings.""",
    tools=[ped_tools, chembl_tools, biocontext_tools, create_markdown_artifact],
    output_key="research_results",
)
