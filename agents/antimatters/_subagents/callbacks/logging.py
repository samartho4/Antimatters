"""Logging callbacks for tool execution tracking."""
import logging
import time
from typing import Any, Optional

logger = logging.getLogger("antimatters")


async def before_tool_callback(tool: Any, args: dict, ctx: Any, tool_context: Any = None) -> Optional[Any]:
    """Log tool invocation and store start time."""
    name = getattr(tool, "name", str(tool))
    logger.info(f"Starting: {name}")
    ctx.state[f"temp:start_{name}"] = time.time()
    return None


async def after_tool_callback(tool: Any, args: dict, ctx: Any, response: Any, tool_context: Any = None) -> Optional[dict]:
    """Log completion and cache important results in state."""
    name = getattr(tool, "name", str(tool))
    start = ctx.state.get(f"temp:start_{name}")
    duration = time.time() - start if start else 0
    logger.info(f"Completed: {name} ({duration:.1f}s)")

    if not isinstance(response, dict) or not response.get("success"):
        return None

    # Cache key results
    if name == "fetch_ped_ensemble":
        ctx.state["pdb_path"] = response.get("pdb_path")
        ctx.state["n_conformations"] = response.get("n_conformations")

    elif name == "prepare_ligand":
        ctx.state["pdbqt_path"] = response.get("pdbqt_path")

    elif name == "cluster_conformations":
        ctx.state["representative_frames"] = response.get("representative_frames")

    elif name == "dock_ensemble":
        energy = response.get("ensemble_summary", {}).get("average_best_energy")
        if energy:
            best = ctx.state.get("best_energy")
            if best is None or energy < best:
                ctx.state["best_energy"] = energy
                logger.info(f"New best: {energy:.2f} kcal/mol")

    return None
