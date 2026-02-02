"""
Antimatters Agent Configuration
===============================

Centralized configuration for all agents, following best practices:
- No hardcoded values in agent definitions
- Environment-aware settings
- Easy to extend for new proteins/ligands
- Configurable from frontend

Reference: https://ai.google.dev/gemini-api/docs/prompting-strategies
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


# =============================================================================
# Path Configuration
# =============================================================================

CORE_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = CORE_ROOT.parent

# MCP Server paths (relative to project)
MCP_SERVERS = {
    "docking": CORE_ROOT / "mcp_servers" / "docking" / "server.py",
    "ped": CORE_ROOT / "mcp_servers" / "ped" / "server.py",
    "chembl": PROJECT_ROOT / "chembl-mcp-server" / "build" / "index.js",
}

# Python interpreter
PYTHON_CMD = sys.executable


# =============================================================================
# Timeout Configuration (seconds)
# =============================================================================

@dataclass
class TimeoutConfig:
    """Configurable timeouts for different operations."""
    ped: int = 180  # PED downloads can be slow
    docking: int = 7200  # Docking: 2 ligands × 5 clusters × ~10min/run = ~100min (2hr buffer)
    biocontext: int = 120  # Web searches (Google Scholar can be slow)
    chembl: int = 120  # ChEMBL API queries - increased for similarity searches
    default: int = 120  # Default timeout


TIMEOUTS = TimeoutConfig()


# =============================================================================
# Retry Configuration (for handling 503 overload errors)
# =============================================================================

@dataclass
class RetryConfig:
    """Exponential backoff configuration for API 503 errors.

    Gemini models experience 45% error rate during peak hours (2026-01).
    Exponential backoff improves success rate from ~30% to ~70%.
    Enhanced for sustained overload periods with model fallback.
    """
    max_retries: int = 10  # Up to 10 retry attempts for sustained overload
    initial_delay: float = 5.0  # Start with 5-second delay (longer for 503)
    max_delay: float = 120.0  # Cap at 2 minutes (longer for 503 overload)
    exponential_base: float = 1.5  # Slower backoff (1.5x) for more attempts
    jitter: bool = True  # Add randomness to prevent thundering herd

    # Which status codes to retry
    retryable_codes: List[int] = field(default_factory=lambda: [503, 429, 500])

    # Which error messages indicate transient issues
    retryable_messages: List[str] = field(default_factory=lambda: [
        "overloaded",
        "unavailable",
        "quota exceeded",
        "rate limit",
        "resource_exhausted",
        "service unavailable",
        "temporarily unavailable",
        "try again later"
    ])

    # Model fallback settings
    model_rotation_enabled: bool = True  # Enable automatic model fallback
    model_switch_after_retries: int = 3  # Switch to fallback after 3 consecutive 503s


RETRY = RetryConfig()


# =============================================================================
# Model Configuration
# =============================================================================

class GeminiModel(Enum):
    """Available Gemini models with their characteristics."""
    # Gemini 2.x (Production)
    FLASH_2_5 = "gemini-2.5-flash"  # Fast, cost-effective
    FLASH_2_0 = "gemini-2.0-flash"  # Stable
    PRO_2_5 = "gemini-2.5-pro"  # Higher quality, slower

    # Gemini 3.x (Preview - Latest, Best Performance)
    FLASH_3 = "gemini-3-flash-preview"  # Fastest, low-latency
    PRO_3 = "gemini-3-pro-preview"  # Pro-grade reasoning

    @property
    def is_multimodal(self) -> bool:
        return True  # All Gemini 2.x+ models support multimodal


@dataclass
class ModelConfig:
    """Model selection based on task complexity.

    Using Gemini 3 (Flash/Pro):
    - Latest models with best performance
    - Flash for simple tasks, Pro for complex reasoning
    - Supports molecule generation capabilities

    Note: Each model has separate quota buckets. Using different models
    helps avoid rate limiting during intensive workflows.
    """
    research: str = GeminiModel.PRO_3.value  # Gemini 3 Pro for complex research
    engineering: str = GeminiModel.PRO_2_5.value  # Gemini 2.5 Pro for docking (2.0-pro not available)
    evolution: str = GeminiModel.PRO_3.value  # Gemini 3 Pro for SAR discovery
    coordinator: str = GeminiModel.PRO_2_5.value  # Gemini 2.5 Pro for routing


MODELS = ModelConfig()


# =============================================================================
# Model Fallback Chains (for 503 UNAVAILABLE handling)
# =============================================================================

@dataclass
class FallbackChain:
    """Ordered list of fallback models when primary is overloaded.

    When a model returns 503 UNAVAILABLE, the system will try the next
    model in the chain. gemini-2.0-flash is the most stable fallback.
    """
    # Research agent fallbacks (need reasoning capability)
    research: List[str] = field(default_factory=lambda: [
        GeminiModel.PRO_2_5.value,      # Primary
        GeminiModel.PRO_3.value,         # Fallback 1: 3.x Pro
        GeminiModel.FLASH_2_5.value,     # Fallback 2: Faster 2.5
        GeminiModel.FLASH_2_0.value,     # Fallback 3: Most stable
    ])

    # Engineering agent fallbacks (need orchestration)
    engineering: List[str] = field(default_factory=lambda: [
        GeminiModel.PRO_3.value,         # Primary
        GeminiModel.PRO_2_5.value,       # Fallback 1: 2.5 Pro
        GeminiModel.FLASH_2_5.value,     # Fallback 2: Faster
        GeminiModel.FLASH_2_0.value,     # Fallback 3: Most stable
    ])

    # Evolution agent fallbacks (need generation capability)
    evolution: List[str] = field(default_factory=lambda: [
        GeminiModel.PRO_3.value,         # Primary
        GeminiModel.PRO_2_5.value,       # Fallback 1: 2.5 Pro
        GeminiModel.FLASH_2_5.value,     # Fallback 2: Faster
        GeminiModel.FLASH_2_0.value,     # Fallback 3: Most stable
    ])

    # Coordinator fallbacks (fast routing)
    coordinator: List[str] = field(default_factory=lambda: [
        GeminiModel.FLASH_3.value,       # Primary
        GeminiModel.FLASH_2_5.value,     # Fallback 1: 2.5 Flash
        GeminiModel.FLASH_2_0.value,     # Fallback 2: Most stable (always works)
    ])


FALLBACKS = FallbackChain()


# =============================================================================
# Protein Configuration (Extensible)
# =============================================================================

@dataclass
class ProteinConfig:
    """Configuration for a target protein."""
    name: str
    ped_id: str
    binding_site_residues: List[int]
    description: str = ""
    uniprot_id: Optional[str] = None
    default_n_clusters: int = 20

    # Docking parameters
    default_exhaustiveness: int = 8
    box_padding: float = 10.0  # Angstroms


# Pre-configured proteins (easily extendable)
PROTEINS: Dict[str, ProteinConfig] = {
    "alpha_synuclein": ProteinConfig(
        name="Alpha-Synuclein",
        ped_id="PED00006e001",
        binding_site_residues=[125, 133, 136],  # C-terminus tyrosines
        description="Intrinsically disordered protein implicated in Parkinson's disease",
        uniprot_id="P37840",
        default_n_clusters=20,
        default_exhaustiveness=8,
    ),
    "tau": ProteinConfig(
        name="Tau Protein",
        ped_id="",  # To be configured
        binding_site_residues=[],
        description="Microtubule-associated protein in Alzheimer's",
        uniprot_id="P10636",
    ),
    "p53_ntd": ProteinConfig(
        name="p53 N-terminal Domain",
        ped_id="",
        binding_site_residues=[],
        description="Intrinsically disordered region of tumor suppressor",
        uniprot_id="P04637",
    ),
}


# =============================================================================
# Ligand Configuration (Extensible)
# =============================================================================

@dataclass
class LigandConfig:
    """Configuration for a ligand."""
    name: str
    smiles: str
    source: str = "user"
    chembl_id: Optional[str] = None
    description: str = ""


# Pre-configured ligands (easily extendable from frontend)
LIGANDS: Dict[str, LigandConfig] = {
    "fasudil": LigandConfig(
        name="Fasudil",
        smiles="CC(=O)Nc1ccc2c(c1)C(=O)N(C2)C3CCCNC3",
        source="ChEMBL",
        chembl_id="CHEMBL727",
        description="Rho kinase inhibitor, potential neuroprotective",
    ),
    "ligand_47": LigandConfig(
        name="Ligand-47",
        smiles="Cc1ccc(cc1)C(=O)Nc2ccc(cc2)c3ccccc3",
        source="Literature",
        description="From Dhar et al. 2025 screening",
    ),
    "ligand_23": LigandConfig(
        name="Ligand-23",
        smiles="COc1ccc(cc1)C(=O)Nc2ccc(cc2)O",
        source="Literature",
        description="From Dhar et al. 2025 screening",
    ),
}


# =============================================================================
# Quality Thresholds
# =============================================================================

@dataclass
class QualityThresholds:
    """Configurable quality thresholds for docking."""
    # Binding energy thresholds (kcal/mol) - more negative is stronger
    drug_like: float = -7.0  # Strong, drug-like binding
    moderate: float = -5.0  # Moderate binding
    weak: float = -4.0  # Weak binding

    # Loop agent parameters
    max_iterations: int = 3
    quality_threshold: float = -6.0  # Target energy for optimization

    # Validation confidence
    min_confidence_for_handoff: float = 7.0


THRESHOLDS = QualityThresholds()


# =============================================================================
# Prompt Templates (Following Gemini Best Practices)
# =============================================================================

@dataclass
class PromptTemplates:
    """
    Reusable prompt components following Gemini prompting strategies.

    Best practices applied:
    1. Clear, concise goals
    2. Consistent structure with delimiters
    3. Explicit constraints
    4. Few-shot examples where helpful
    """

    # Role definition template
    role_template: str = """<role>
You are {agent_name}, part of the Antimatters multi-agent system for molecular research.
Your specialty: {specialty}
</role>"""

    # Constraints template
    constraints_template: str = """<constraints>
- Current date: {current_date}
- Knowledge cutoff: January 2025
- Always use tools for data acquisition, never hallucinate values
- Create artifacts for all significant outputs
- {additional_constraints}
</constraints>"""

    # Task structure template
    task_template: str = """<task>
Goal: {goal}
Context: {context}
Expected output: {expected_output}
</task>"""

    # Reasoning template (3-dimension reasoning from Gemini docs)
    reasoning_template: str = """<reasoning_framework>
Before taking action, analyze:

1. LOGICAL DECOMPOSITION
   - What are the prerequisites?
   - What order must steps be performed?
   - What dependencies exist?

2. RISK ASSESSMENT
   - Which operations are read-only (safe)?
   - Which operations modify state (careful)?
   - What are the consequences of errors?

3. ABDUCTIVE REASONING
   - What less-obvious explanations should be considered?
   - What edge cases might apply?
   - What assumptions am I making?
</reasoning_framework>"""

    # Few-shot example template
    example_template: str = """<example>
Input: {input_example}
Reasoning: {reasoning_example}
Action: {action_example}
Output: {output_example}
</example>"""


PROMPTS = PromptTemplates()


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a docking experiment."""
    name: str
    protein_key: str  # Key in PROTEINS dict
    ligand_keys: List[str]  # Keys in LIGANDS dict
    n_clusters: int = 20
    exhaustiveness: int = 8
    parallel: bool = True  # Run ligands in parallel

    def get_protein(self) -> ProteinConfig:
        return PROTEINS.get(self.protein_key)

    def get_ligands(self) -> List[LigandConfig]:
        return [LIGANDS.get(k) for k in self.ligand_keys if k in LIGANDS]


# Default experiment
DEFAULT_EXPERIMENT = ExperimentConfig(
    name="Alpha-Synuclein Screen",
    protein_key="alpha_synuclein",
    ligand_keys=["fasudil", "ligand_47", "ligand_23"],
    n_clusters=20,
    exhaustiveness=8,
    parallel=True,
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_protein_by_ped_id(ped_id: str) -> Optional[ProteinConfig]:
    """Find protein config by PED ID."""
    for config in PROTEINS.values():
        if config.ped_id == ped_id:
            return config
    return None


def get_ligand_by_name(name: str) -> Optional[LigandConfig]:
    """Find ligand config by name (case-insensitive)."""
    name_lower = name.lower()
    for key, config in LIGANDS.items():
        if config.name.lower() == name_lower or key == name_lower:
            return config
    return None


def add_protein(key: str, config: ProteinConfig) -> None:
    """Add a new protein configuration (for frontend extensibility)."""
    PROTEINS[key] = config


def add_ligand(key: str, config: LigandConfig) -> None:
    """Add a new ligand configuration (for frontend extensibility)."""
    LIGANDS[key] = config


def create_experiment(
    name: str,
    protein_key: str,
    ligand_keys: List[str],
    **kwargs
) -> ExperimentConfig:
    """Create a new experiment configuration."""
    return ExperimentConfig(
        name=name,
        protein_key=protein_key,
        ligand_keys=ligand_keys,
        **kwargs
    )


# =============================================================================
# Environment Configuration
# =============================================================================

def get_api_key() -> Optional[str]:
    """Get Google API key from environment."""
    return os.getenv("GOOGLE_API_KEY")


def validate_config() -> Dict[str, Any]:
    """Validate configuration and return status."""
    issues = []

    # Check API key
    if not get_api_key():
        issues.append("GOOGLE_API_KEY not set")

    # Check MCP servers exist
    for name, path in MCP_SERVERS.items():
        if not path.exists():
            issues.append(f"MCP server not found: {name} at {path}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "proteins": list(PROTEINS.keys()),
        "ligands": list(LIGANDS.keys()),
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Paths
    "CORE_ROOT",
    "PROJECT_ROOT",
    "MCP_SERVERS",
    "PYTHON_CMD",

    # Configs
    "TIMEOUTS",
    "RETRY",
    "MODELS",
    "FALLBACKS",
    "PROTEINS",
    "LIGANDS",
    "THRESHOLDS",
    "PROMPTS",

    # Classes
    "TimeoutConfig",
    "RetryConfig",
    "ModelConfig",
    "FallbackChain",
    "ProteinConfig",
    "LigandConfig",
    "QualityThresholds",
    "PromptTemplates",
    "ExperimentConfig",
    "GeminiModel",

    # Functions
    "get_protein_by_ped_id",
    "get_ligand_by_name",
    "add_protein",
    "add_ligand",
    "create_experiment",
    "get_api_key",
    "validate_config",

    # Defaults
    "DEFAULT_EXPERIMENT",
]
