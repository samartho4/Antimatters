#!/usr/bin/env python3
"""PED MCP Server - Fetch protein ensembles from Protein Ensemble Database

Based on PED API documentation: https://deposition.proteinensemble.org/api/v1
API Reference: https://proteinensemble.org/api
"""

import os
import sys
import pathlib
import warnings
import requests
import gzip

# Suppress all warnings that could corrupt MCP JSON-RPC stdout
warnings.filterwarnings('ignore')
os.environ['OPENMM_LOG_LEVEL'] = '0'

# Redirect BOTH stdout and stderr at the file descriptor level during imports
_saved_stdout_fd = os.dup(1)
_saved_stderr_fd = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 1)
os.dup2(_devnull, 2)

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing_extensions import Annotated

try:
    from idpet.ensemble import Ensemble
    from idpet.ensemble_analysis import EnsembleAnalysis
    IDPET_AVAILABLE = True
except ImportError:
    IDPET_AVAILABLE = False

# Restore stdout and stderr
os.dup2(_saved_stdout_fd, 1)
os.dup2(_saved_stderr_fd, 2)
os.close(_devnull)
os.close(_saved_stdout_fd)
os.close(_saved_stderr_fd)

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

mcp = FastMCP("ped-ensemble-server")

# PED API v1 base URL (from official API documentation)
PED_API_BASE = "https://deposition.proteinensemble.org/api/v1"

# Set default data directory
DEFAULT_DATA_DIR = pathlib.Path.home() / ".idpet" / "data"
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

@mcp.tool()
async def search_ped_protein(
    protein_name: Annotated[str, Field(description="Protein name to search for (e.g., 'Tau', 'Alpha-Synuclein', 'p53')")] = None,
    uniprot_acc: Annotated[str, Field(description="UniProt accession (e.g., 'P10636' for Tau, 'P37840' for Alpha-Synuclein')")] = None,
    free_text: Annotated[str, Field(description="Free text search query")] = None
) -> dict:
    """Search PED database for protein entries.

    Searches using the official PED API v1 endpoint /entries with multiple search strategies:
    - protein_name: Search by protein name
    - uniprot_acc: Search by UniProt accession (most reliable)
    - free_text: Free text search across all fields

    Returns PED IDs and metadata for matching proteins.
    """
    try:
        # Build query parameters
        params = {"limit": 20}

        if uniprot_acc:
            params["uniprot_acc"] = uniprot_acc
        elif protein_name:
            params["protein_name"] = protein_name
        elif free_text:
            params["free_text"] = free_text
        else:
            return {"success": False, "error": "Must provide protein_name, uniprot_acc, or free_text"}

        # Call PED API
        response = requests.get(f"{PED_API_BASE}/entries", params=params, timeout=20)

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"PED API returned status {response.status_code}",
                "details": response.text[:200]
            }

        data = response.json()
        results = data.get('result', [])

        if not results:
            return {
                "success": True,
                "found": False,
                "search_params": params,
                "message": "No PED entries found. Try using UniProt accession for better results."
            }

        # Extract relevant information from API response
        entries = []
        for entry in results[:10]:
            desc = entry.get('description', {})

            # Extract UniProt IDs from construct chains
            uniprot_ids = []
            for chain in entry.get('construct_chains', []):
                for frag in chain.get('fragments', []):
                    if frag.get('uniprot_acc'):
                        uniprot_ids.append(frag['uniprot_acc'])

            # Extract ensemble information
            ensembles = entry.get('ensembles', [])
            ensemble_info = []
            for ens in ensembles:
                ensemble_info.append({
                    "ensemble_id": ens.get('ensemble_id'),
                    "n_models": ens.get('models')
                })

            entries.append({
                "ped_id": entry.get('entry_id'),
                "title": desc.get('title', 'N/A'),
                "authors": [a.get('name') for a in desc.get('authors', [])[:3]],
                "uniprot_ids": list(set(uniprot_ids)),
                "n_ensembles": len(ensembles),
                "ensembles": ensemble_info,
                "total_models": sum(e.get('models', 0) for e in ensembles)
            })

        return {
            "success": True,
            "found": True,
            "search_params": params,
            "n_results": len(entries),
            "entries": entries
        }

    except Exception as e:
        logger.error(f"Error searching PED: {e}")
        return {"success": False, "error": type(e).__name__, "details": str(e)}


@mcp.tool()
async def get_ped_entry_details(
    ped_id: Annotated[str, Field(description="PED ID (e.g., 'PED00017' for Tau, 'PED00006' for Alpha-Synuclein)")]
) -> dict:
    """Get detailed information about a specific PED entry.

    Uses the official PED API v1 endpoint /entries/{identifier}.
    Returns complete metadata including ensembles, methods, and UniProt information.
    """
    try:
        response = requests.get(f"{PED_API_BASE}/entries/{ped_id}", timeout=20)

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"PED entry {ped_id} not found",
                "status_code": response.status_code
            }

        entry = response.json()
        desc = entry.get('description', {})

        # Extract UniProt IDs
        uniprot_ids = []
        for chain in entry.get('construct_chains', []):
            for frag in chain.get('fragments', []):
                if frag.get('uniprot_acc'):
                    uniprot_ids.append(frag['uniprot_acc'])

        # Extract ensemble details
        ensembles = []
        for ens in entry.get('ensembles', []):
            ensembles.append({
                "ensemble_id": ens.get('ensemble_id'),
                "n_models": ens.get('models'),
                "chains": ens.get('chains', [])
            })

        return {
            "success": True,
            "ped_id": entry.get('entry_id'),
            "title": desc.get('title'),
            "authors": [a.get('name') for a in desc.get('authors', [])],
            "uniprot_ids": list(set(uniprot_ids)),
            "ensembles": ensembles,
            "methods": [term.get('name') for term in desc.get('ontology_terms', [])],
            "cross_references": desc.get('entry_cross_reference', [])
        }

    except Exception as e:
        logger.error(f"Error fetching PED entry: {e}")
        return {"success": False, "error": type(e).__name__, "details": str(e)}


@mcp.tool()
async def fetch_ped_ensemble(
    ped_id: Annotated[str, Field(description="PED ID (e.g., 'PED00017', 'PED00006e001')")],
    ensemble_id: Annotated[str, Field(description="Ensemble ID (e.g., 'e001', 'e002'). Use get_ped_entry_details to find available ensembles.")] = "e001",
    output_dir: Annotated[str, Field(description="Output directory for downloaded files")] = None
) -> dict:
    """Fetch protein ensemble from PED database.

    Downloads the ensemble PDB file using the official PED API v1 endpoint:
    /entries/{identifier}/ensembles/{ensemble_id}/ensemble-pdb

    Also attempts to use IDPET library if available for additional processing.
    """
    if not IDPET_AVAILABLE:
        return {"success": False, "error": "idpet not installed", "details": "pip install idpet"}

    try:
        output_dir = output_dir or str(DEFAULT_DATA_DIR)
        os.makedirs(output_dir, exist_ok=True)

        # Check for cached PDB file first
        pdb_path = os.path.join(output_dir, f'{ped_id}_{ensemble_id}.pdb')
        if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 1000:
            logger.info(f"Using cached PDB: {pdb_path}")
            # Return cached file info
            try:
                saved_stdout = os.dup(1)
                saved_stderr = os.dup(2)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                try:
                    ensemble_obj = Ensemble(code=ped_id, database='ped')
                    analysis = EnsembleAnalysis([ensemble_obj], output_dir=output_dir)
                    analysis.load_trajectories()
                    ensemble = analysis.ensembles[0]
                    n_frames = ensemble.get_size()
                    n_residues = ensemble.get_num_residues()
                finally:
                    os.dup2(saved_stdout, 1)
                    os.dup2(saved_stderr, 2)
                    os.close(devnull)
                    os.close(saved_stdout)
                    os.close(saved_stderr)
                return {
                    "success": True,
                    "ped_id": ped_id,
                    "ensemble_id": ensemble_id,
                    "pdb_path": pdb_path,
                    "n_frames": n_frames,
                    "n_residues": n_residues,
                    "source": "cached"
                }
            except Exception as e:
                return {
                    "success": True,
                    "ped_id": ped_id,
                    "ensemble_id": ensemble_id,
                    "pdb_path": pdb_path,
                    "source": "cached",
                    "note": "IDPET processing skipped"
                }

        # Download ensemble PDB from PED API
        download_url = f"{PED_API_BASE}/entries/{ped_id}/ensembles/{ensemble_id}/ensemble-pdb"
        logger.info(f"Downloading from: {download_url}")

        response = requests.get(download_url, timeout=60)

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to download ensemble from PED API",
                "status_code": response.status_code,
                "ped_id": ped_id,
                "ensemble_id": ensemble_id
            }

        # Save PDB file (decompress if gzipped)
        pdb_path = os.path.join(output_dir, f'{ped_id}_{ensemble_id}.pdb')

        # Check if content is gzipped (starts with 0x1f 0x8b)
        content = response.content
        if content[:2] == b'\x1f\x8b':
            # Decompress gzipped content
            content = gzip.decompress(content)

        with open(pdb_path, 'wb') as f:
            f.write(content)

        # Try to process with IDPET for additional metadata
        try:
            # Redirect stdout/stderr during loading
            saved_stdout = os.dup(1)
            saved_stderr = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)

            try:
                ensemble_obj = Ensemble(code=ped_id, database='ped')
                analysis = EnsembleAnalysis([ensemble_obj], output_dir=output_dir)
                analysis.load_trajectories()
                ensemble = analysis.ensembles[0]
                n_frames = ensemble.get_size()
                n_residues = ensemble.get_num_residues()
            finally:
                os.dup2(saved_stdout, 1)
                os.dup2(saved_stderr, 2)
                os.close(devnull)
                os.close(saved_stdout)
                os.close(saved_stderr)
        except Exception as e:
            logger.warning(f"IDPET processing failed, using PDB file only: {e}")
            # Count frames from PDB
            n_frames = 0
            n_residues = 0
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('MODEL'):
                        n_frames += 1
                    elif line.startswith('ATOM') and n_frames <= 1:
                        resnum = int(line[22:26].strip())
                        n_residues = max(n_residues, resnum)

        return {
            "success": True,
            "ped_id": ped_id,
            "ensemble_id": ensemble_id,
            "pdb_path": pdb_path,
            "n_conformations": n_frames if n_frames > 0 else 1,
            "n_residues": n_residues,
            "output_dir": output_dir,
            "download_url": download_url
        }

    except Exception as e:
        logger.error(f"Error fetching PED ensemble: {e}")
        return {"success": False, "error": type(e).__name__, "details": str(e)}

if __name__ == "__main__":
    logger.info("Starting PED MCP Server with robust API integration...")
    logger.info(f"PED API Base: {PED_API_BASE}")
    logger.info(f"IDPET Available: {IDPET_AVAILABLE}")
    mcp.run()
