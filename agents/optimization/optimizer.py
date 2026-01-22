"""Optimization Loop Agent: Iterative docking refinement."""
from google.adk.agents import LlmAgent, LoopAgent
from core.mcp_servers.toolsets.docking import docking_tools
from core.agents.config import MAX_DOCKING_ITERATIONS, QUALITY_THRESHOLD_KCAL

dock_iteration = LlmAgent(
    name="dock_iteration",
    model="gemini-2.0-flash",
    description="Single docking iteration with increasing exhaustiveness",
    instruction=f"""Run one docking iteration:
1. Read iteration_count from state (default 0)
2. Calculate exhaustiveness = 8 * (iteration + 1)
3. Run dock_ensemble with current exhaustiveness
4. Report energy and iteration number""",
    tools=[docking_tools],
)

quality_check = LlmAgent(
    name="quality_check",
    model="gemini-2.0-flash",
    description="Check if docking quality threshold is met",
    instruction=f"""Evaluate docking quality:

**STOP if any:**
- best_energy <= {QUALITY_THRESHOLD_KCAL} kcal/mol → {{"escalate": true, "reason": "threshold met"}}
- iteration >= {MAX_DOCKING_ITERATIONS} → {{"escalate": true, "reason": "max iterations"}}

**CONTINUE otherwise** - just report status.""",
)

optimization_loop = LoopAgent(
    name="docking_optimizer",
    sub_agents=[dock_iteration, quality_check],
    max_iterations=MAX_DOCKING_ITERATIONS,
)
