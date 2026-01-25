"""Configuration constants for Antimatters."""
from pathlib import Path

# Project paths - relative to core/ directory
CORE_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = CORE_ROOT.parent

# MCP Server paths
DOCKING_SERVER = CORE_ROOT / "mcp_servers" / "docking" / "server.py"
PED_SERVER = CORE_ROOT / "mcp_servers" / "ped" / "server.py"
CHEMBL_SERVER = PROJECT_ROOT / "chembl-mcp-server" / "build" / "index.js"

# Python interpreter
import sys
PYTHON_CMD = sys.executable

# Timeouts (seconds)
PED_TIMEOUT = 180  # PED downloads can be slow
DOCKING_TIMEOUT = 1800  # Docking operations are computationally intensive
BIOCONTEXT_TIMEOUT = 60  # Web searches
CHEMBL_TIMEOUT = 30  # ChEMBL API queries

# Docking parameters
DEFAULT_N_CLUSTERS = 20
DEFAULT_EXHAUSTIVENESS = 8
DEFAULT_RESIDUE_RANGE = [125, 133, 136]  # Alpha-synuclein binding site

# Loop agent parameters
MAX_DOCKING_ITERATIONS = 3
QUALITY_THRESHOLD_KCAL = -6.0  # kcal/mol - drug-like threshold

# Binding energy interpretation thresholds (kcal/mol)
STRONG_BINDING_THRESHOLD = -7.0
MODERATE_BINDING_THRESHOLD = -5.0
WEAK_BINDING_THRESHOLD = -4.0
