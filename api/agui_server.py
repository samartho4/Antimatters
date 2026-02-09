"""
AG-UI
================================
use https://docs.ag-ui.com/llms-full.txt for ADK
- Artifact generation working but needs to be aligend for scientifc workflow like for pre docking there should be molecualar and protein analysis
- State management for protein/ligand/docking data for multimodal analysis, would work on browser capturing afterwards. Not sure for frontend's real-time tool streaming. 
"""
import asyncio
import json
import os
import sys
import uuid
import base64
import logging
from pathlib import Path
from typing import Optional, AsyncGenerator, Any, Dict, List, Union
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

# Configure logging to show verification logs
# Use force=True to override any existing logging config (Python 3.8+)
# Set root logger to INFO so ALL child loggers (including agents) will output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console output
    ],
    force=True  # Override any existing config from imports
)
# Also explicitly set root logger level (belt and suspenders)
logging.getLogger().setLevel(logging.INFO)
# Ensure no loggers are silenced
logging.getLogger().handlers[0].setLevel(logging.INFO)

# Add core to path
CORE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CORE_ROOT))
sys.path.insert(0, str(CORE_ROOT.parent))

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import persistent services
from core.api.services import (
    workspace_service,
    conversation_service,
    artifact_service,
    knowledge_service,
    feedback_service
)

# Import retry utilities for handling Gemini 503 errors
from core.agents.utils import is_transient_error, model_switcher, create_coordinator_with_fallback
from core.agents.config import RETRY
import time
import random

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

load_dotenv(CORE_ROOT / ".env")


# =============================================================================
# Event Types (Complete Protocol Spec)
# =============================================================================

class EventType(str, Enum):
    # Lifecycle Events
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"

    # Text Message Events
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"

    # Tool Call Events
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"

    # State Management Events
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"

    # Activity Events (for artifacts/progress)
    ACTIVITY_SNAPSHOT = "ACTIVITY_SNAPSHOT"
    ACTIVITY_DELTA = "ACTIVITY_DELTA"

    # Custom Events
    CUSTOM = "CUSTOM"
    RAW = "RAW"


# =============================================================================
# Artifact Types
# =============================================================================
# needs to be aligned for scientifc workflow like there should be dynamic starting from the connected tools and ending with evaluation metrics
# like in our case can be pdb path, pdbqt path, representative frames to all the way to @interaction analysis tool, later RL anf PySR would be employed for ADMET, free binding enerygy deriving correlatin/forces field for causation and clinincal trials as a whole
class ArtifactType(str, Enum):
    TASK_LIST = "task_list"
    IMPLEMENTATION_PLAN = "implementation_plan"
    WALKTHROUGH = "walkthrough"
    SCREENSHOT = "screenshot"
    STRUCTURE_3D = "structure_3d"
    DOCKING_RESULT = "docking_result"
    INTERACTION_MAP = "interaction_map"
    TRAJECTORY = "trajectory"
    CLUSTER_VISUALIZATION = "cluster_visualization"
    LITERATURE_RESULT = "literature_result"
    
    # Master Container
    SCIENTIFIC_EXPERIMENT = "scientific_experiment"

# the last 4 types can be utilzed for validating against resaerch paper and presenting one. For better explanation genemedia a built in tool can be leveraged too.

# =============================================================================
# Event Encoder (SSE Format)
# =============================================================================

class EventEncoder:
    """Encodes AG-UI events to SSE format."""

    def __init__(self, accept: Optional[str] = None):
        self.accept = accept or "text/event-stream"

    def encode(self, event: Dict[str, Any]) -> str:
        """Encode event to SSE format."""
        # Convert enums to strings
        if "type" in event and isinstance(event["type"], Enum):
            event["type"] = event["type"].value
        data = json.dumps(event, default=str)
        return f"data: {data}\n\n"

    def get_content_type(self) -> str:
        return "text/event-stream"


# =============================================================================
# Input/Output Models
# =============================================================================

class Message(BaseModel):
    id: str
    role: str  # "user", "assistant", "system", "tool"
    content: Optional[Union[str, List[Dict]]] = None  # String or multimodal content
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Context(BaseModel):
    type: str
    data: Any


class AnnotationRegion(BaseModel):
    """Region annotation from frontend for @analyze_interactions tool, add screen recording too"""
    residue_ids: List[int] = Field(default_factory=list)
    atom_ids: List[int] = Field(default_factory=list)
    description: Optional[str] = None
    screenshot_base64: Optional[str] = None


class RunAgentInput(BaseModel):
    """AG-UI standard input format."""
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = Field(default_factory=list)
    tools: List[Tool] = Field(default_factory=list)
    state: Optional[Dict[str, Any]] = None
    context: Optional[List[Context]] = None
    forwarded_props: Optional[Dict[str, Any]] = None
    # Custom fields for Antimatters
    annotation: Optional[AnnotationRegion] = None
    mode: Optional[str] = None  # "planning" or "serendipitize" - routes to different agents


# =============================================================================
# Artifact Manager, this should be shown in side tab as "Evolution" where we would leverage some graph library, filtering with proximity and ranking
# =============================================================================

class ArtifactManager:
    """Manages artifacts generated during agent runs with disk persistence."""

    def __init__(self, storage_path: str = "artifacts.json"):
        self.storage_path = Path(storage_path)
        self.artifacts: Dict[str, Dict] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    self.artifacts = json.load(f)
                print(f"Loaded {len(self.artifacts)} artifacts from disk.")
            except Exception as e:
                print(f"Error loading artifacts: {e}")

    def _save_to_disk(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.artifacts, f, indent=2)
        except Exception as e:
            print(f"Error saving artifacts: {e}")

    def create_structure_artifact(self, run_id: str, pdb_data: str, metadata: Dict) -> Dict:
        """Create a 3D structure artifact for Mol* visualization."""
        artifact = {
            "id": f"artifact_{uuid.uuid4().hex[:8]}",
            "type": ArtifactType.STRUCTURE_3D.value,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "content": {
                "pdb_data": pdb_data,
                "format": "pdb",
                "metadata": metadata,
                "view_settings": {
                    "representation": "cartoon",
                    "color_scheme": "chain-id",
                    "highlight_residues": metadata.get("binding_site", [])
                }
            }
        }
        self.artifacts[artifact["id"]] = artifact
        self._save_to_disk()
        return artifact

    def create_docking_artifact(self, run_id: str, results: Dict) -> Dict:
        """Create a docking result artifact."""
        artifact = {
            "id": f"artifact_{uuid.uuid4().hex[:8]}",
            "type": ArtifactType.DOCKING_RESULT.value,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "content": {
                "binding_scores": results.get("binding_scores", []),
                "ensemble_average": results.get("ensemble_average"),
                "best_pose": results.get("best_pose"),
                "ligand_pdbqt": results.get("ligand_pdbqt"),
                "visualization": {
                    "chart_type": "bar",
                    "x_label": "Cluster",
                    "y_label": "Binding Energy (kcal/mol)",
                    "threshold_lines": [
                        {"value": -7.0, "label": "Strong binding", "color": "#22c55e"},
                        {"value": -5.0, "label": "Moderate binding", "color": "#eab308"}
                    ]
                }
            }
        }
        self.artifacts[artifact["id"]] = artifact
        self._save_to_disk()
        return artifact

    def create_interaction_artifact(self, run_id: str, interactions: Dict) -> Dict:
        """Create an interaction map artifact."""
        artifact = {
            "id": f"artifact_{uuid.uuid4().hex[:8]}",
            "type": ArtifactType.INTERACTION_MAP.value,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "content": {
                "interactions": interactions,
                "visualization": {
                    "type": "heatmap",
                    "residues": interactions.get("residues", []),
                    "interaction_types": ["h_bond", "hydrophobic", "aromatic", "ionic"]
                }
            }
        }
        self.artifacts[artifact["id"]] = artifact
        self._save_to_disk()
        return artifact

    def create_cluster_artifact(self, run_id: str, cluster_data: Dict) -> Dict:
        """Create a cluster visualization artifact."""
        artifact = {
            "id": f"artifact_{uuid.uuid4().hex[:8]}",
            "type": ArtifactType.CLUSTER_VISUALIZATION.value,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "content": {
                "n_clusters": cluster_data.get("n_clusters"),
                "populations": cluster_data.get("populations", []),
                "representatives": cluster_data.get("representatives", []),
                "tsne_coordinates": cluster_data.get("tsne_coordinates", []),
                "visualization": {
                    "type": "scatter",
                    "x_label": "t-SNE 1",
                    "y_label": "t-SNE 2",
                    "color_by": "cluster"
                }
            }
        }
        self.artifacts[artifact["id"]] = artifact
        self._save_to_disk()
        return artifact


# =============================================================================
# Known Binding Sites Lookup
# =============================================================================

KNOWN_BINDING_SITES = {
    "PED00006e001": [125, 133, 136],  # Alpha-synuclein tyrosine cluster (Y125, Y133, Y136)
    "PED00006e002": [125, 133, 136],  # Alpha-synuclein variant
    "PED00024e001": [39, 46, 53],      # Example: other IDP binding site
}

# =============================================================================
# ADK Agent Integration with Artifact Generation. Three things need to be done: 1. async computation as parallel agent, think of mission control as agent manager in sidebar name as "compuation" 2. browser testing artifacts as validation and async feedback to evolve, this should also go in evolution tabe either we make comments on docs or images. 3. sidebar should show chat name along with artifact names as bullets
# =============================================================================

class ADKPart:
    def __init__(self, text=None):
        self.text = text

class ADKMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        self.parts = [ADKPart(text=content)]

class ADKAgentRunner:
    """Runs ADK agents and translates events to AG-UI format with artifacts."""

    def __init__(self):
        self._runner = None
        self._session_service = None
        self._coordinator = None
        self._initialized = False
        self.artifact_manager = ArtifactManager()
        # Default workspace for new conversations
        self.default_workspace_id = "ws_core"
        # SINGLETON session service - persists across requests for conversation continuity
        # This fixes the issue where session state was lost between requests
        self._global_session_service = None

    def _get_session_service(self):
        """Get or create singleton session service.

        This ensures conversation state persists across API requests,
        following the pattern from agent-chat-ui-main and open-canvas-main.
        """
        if self._global_session_service is None:
            from google.adk.sessions import InMemorySessionService
            self._global_session_service = InMemorySessionService()
            print("📦 [Session] Created singleton InMemorySessionService", flush=True)
        return self._global_session_service

    async def initialize(self):
        """Lazy initialization of ADK components."""
        if self._initialized:
            return

        try:
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService

            # Import the coordinator agent
            from core.agents.antimatters.coordinator import root_agent

            self._coordinator = root_agent
            self._session_service = InMemorySessionService()
            self._runner = Runner(
                agent=root_agent,
                app_name="agui_core",
                session_service=self._session_service
            )
            self._initialized = True
            print("ADK Agent Runner initialized successfully")
        except Exception as e:
            print(f"Failed to initialize ADK Agent Runner: {e}")
            raise

    async def run_with_events(
        self,
        input_data: RunAgentInput,
        encoder: EventEncoder
    ) -> AsyncGenerator[str, None]:
        """Run the agent and yield AG-UI events with artifacts."""
        # Create a fresh runner for every request to ensure clean MCP sessions
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from core.agents.antimatters.coordinator import root_agent, planning_agent, docking_workflow

        # Route based on mode — annotation with screenshot always goes to
        # planning_agent which has Gemini 3 visual-analysis capability.
        mode = input_data.mode or "serendipitize"
        if input_data.annotation and input_data.annotation.screenshot_base64:
            mode = "planning"
            print(f"🎨 [Mode] Annotation with screenshot → planning_agent (visual analysis)", flush=True)
        if mode == "planning":
            selected_agent = planning_agent
            print(f"🎯 [Mode] Planning mode - using planning_agent", flush=True)
        else:
            selected_agent = docking_workflow
            print(f"🔬 [Mode] Serendipitize mode - using docking_workflow", flush=True)

        # Helper to create runner with current model configuration
        # Uses SINGLETON session service for conversation continuity (FIX 1)
        def create_runner_with_models(use_fallback: bool = False, agent=None):
            agent = agent or selected_agent
            if use_fallback:
                try:
                    agent = create_coordinator_with_fallback()
                    print(f"🔄 [ModelFallback] Created agent with fallback models", flush=True)
                except Exception as fe:
                    print(f"⚠️ [ModelFallback] Failed to create fallback agent: {fe}, using default", flush=True)

            # Use singleton session service instead of creating fresh one each request
            # This preserves state (experiment_artifact_id, docking_completed, etc.)
            # across mode switches (serendipitize → planning)
            session_service = self._get_session_service()
            runner = Runner(
                agent=agent,
                app_name="agui_core",
                session_service=session_service
            )
            return runner, session_service

        local_runner, local_session_service = create_runner_with_models(use_fallback=False)

        thread_id = input_data.thread_id
        run_id = input_data.run_id

        # Get workspace from forwarded props or use default
        workspace_id = (input_data.forwarded_props or {}).get("workspace_id", self.default_workspace_id)

        # Get or create conversation using the thread_id
        conversation = conversation_service.get(thread_id)
        if not conversation:
            # Extract user message for title
            user_msg = ""
            for msg in reversed(input_data.messages):
                if msg.role == "user" and msg.content:
                    user_msg = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

            conversation = conversation_service.create(
                workspace_id=workspace_id,
                title=user_msg[:50] + "..." if len(user_msg) > 50 else user_msg or "New Research"
            )
            # Use the new conversation ID as thread_id
            thread_id = conversation["id"]

        # RUN_STARTED
        yield encoder.encode({
            "type": EventType.RUN_STARTED.value,
            "threadId": thread_id,
            "runId": run_id,
            "timestamp": datetime.now().isoformat(),
            "conversationId": conversation["id"],
            "workspaceId": workspace_id,
        })

        try:
            # Extract user message from input
            user_message = ""
            for msg in reversed(input_data.messages):
                if msg.role == "user" and msg.content:
                    user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

            if not user_message:
                raise ValueError("No user message found in input")

            # FIX 3 (Improved): Build conversation context for the agent
            # Two sources: (A) frontend sends history via FIX 2, or (B) load from DB as fallback
            if len(input_data.messages) > 1:
                # Case A: Frontend sent full conversation history (FIX 2 working)
                # Build context from the messages (excluding the last user message which is current)
                prior_msgs = [m for m in input_data.messages[:-1]]  # All but last
                if prior_msgs:
                    history_parts = ["[CONVERSATION HISTORY - use this for context:]"]
                    for msg in prior_msgs[-10:]:  # Last 10 prior messages
                        role = msg.role.upper() if hasattr(msg, 'role') else 'UNKNOWN'
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        content = content[:500]  # Truncate long messages
                        history_parts.append(f"{role}: {content}")
                    history_parts.append("[END HISTORY]\n")
                    user_message = "\n".join(history_parts) + "\n" + user_message
                    print(f"📜 [History] Built context from {len(prior_msgs[-10:])} frontend messages", flush=True)
            elif conversation:
                # Case B: Frontend sent only current message (backwards compatibility)
                # Load prior messages from DB
                try:
                    conv_with_messages = conversation_service.get(
                        conversation["id"],
                        include_messages=True
                    )
                    prior_messages = conv_with_messages.get("messages", []) if conv_with_messages else []

                    if prior_messages:
                        history_parts = ["[CONVERSATION HISTORY - use this for context:]"]
                        # Get last 10 messages for context (truncate long content)
                        for msg in prior_messages[-10:]:
                            role = msg.get('role', 'unknown').upper()
                            content = msg.get('content', '')[:500]  # Truncate long messages
                            history_parts.append(f"{role}: {content}")
                        history_parts.append("[END HISTORY]\n")
                        user_message = "\n".join(history_parts) + "\n" + user_message
                        print(f"📜 [History] Loaded {len(prior_messages[-10:])} messages from DB", flush=True)
                except Exception as hist_err:
                    print(f"⚠️ [History] Failed to load prior messages: {hist_err}", flush=True)

            # Persist user message to conversation
            conversation_service.add_message(
                conversation_id=conversation["id"],
                role="user",
                content=user_message
            )

            # Add annotation context if provided
            if input_data.annotation:
                if input_data.annotation.screenshot_base64:
                    # Multimodal path: image is attached as the next part.
                    # Tell planning_agent to analyze it directly — no tool call needed.
                    annotation_context = (
                        "\n\n[ANNOTATION] A screenshot of the annotated 3D molecular structure "
                        "is attached to this message. Analyze the spatial details you see directly "
                        "from the image: atoms, bonds, hydrogen bonds, hydrophobic contacts, "
                        "aromatic stacking, distances, geometry. Do NOT call "
                        "analyze_artifact_visualization — the image is already here."
                    )
                else:
                    annotation_context = (
                        f"\n[User annotated region: residues {input_data.annotation.residue_ids}, "
                        f"description: {input_data.annotation.description}]"
                    )
                user_message += annotation_context

            # Expand @Type:Name references to KG context (e.g., "@Ligand:Fasudil")
            import re
            at_refs = re.findall(r'@[\w-]+(?::[\w-]+)?', user_message)
            if at_refs:
                try:
                    from core.agents.antimatters._subagents.evolution.agent import get_neo4j_driver
                    driver, database = get_neo4j_driver()
                    kg_context_parts = ["[KG CONTEXT for referenced entities:]"]

                    for ref in at_refs:
                        parts = ref[1:].split(":", 1)
                        entity_name = parts[1] if len(parts) > 1 else parts[0]

                        with driver.session(database=database) as session:
                            # Query entity + INTERACTS_WITH relationships for spatial context
                            result = session.run("""
                                MATCH (n:Entity)
                                WHERE toLower(n.name) = toLower($name)
                                OPTIONAL MATCH (n)-[r:RELATIONSHIP {type: 'INTERACTS_WITH'}]->(residue:Entity)
                                WITH n, collect({
                                    residue_name: residue.name,
                                    residue_id: residue.id,
                                    interaction_props: r.properties
                                }) AS interactions
                                RETURN n.id AS id, n.name AS name, n.type AS type,
                                       n.properties AS properties, interactions
                                LIMIT 1
                            """, {"name": entity_name})
                            record = result.single()

                            if record:
                                props = json.loads(record["properties"]) if record["properties"] else {}
                                kg_context_parts.append(f"• {ref}: id={record['id']}, type={record['type']}")
                                if props.get("smiles"):
                                    kg_context_parts.append(f"  SMILES: {props['smiles']}")
                                if props.get("best_energy"):
                                    kg_context_parts.append(f"  Energy: {props['best_energy']} kcal/mol")
                                # Include spatial relationships from INTERACTS_WITH
                                interactions = record.get("interactions", [])
                                if interactions and interactions[0].get("residue_name"):
                                    int_summary = []
                                    for intr in interactions:
                                        if intr.get("residue_name"):
                                            int_props = json.loads(intr["interaction_props"]) if intr.get("interaction_props") else {}
                                            int_type = int_props.get("interaction_type", "unknown")
                                            int_summary.append(f"{intr['residue_name']}({int_type})")
                                    if int_summary:
                                        kg_context_parts.append(f"  INTERACTS_WITH: {', '.join(int_summary)}")

                    driver.close()

                    if len(kg_context_parts) > 1:
                        user_message = "\n".join(kg_context_parts) + "\n\n" + user_message
                        print(f"[@context] Expanded {len(at_refs)} references", flush=True)
                except Exception as kg_err:
                    print(f"[@context] KG expansion failed: {kg_err}", flush=True)

            # Build the message object — multimodal when annotation includes a screenshot
            # so Gemini can perform spatial reasoning on the annotated 3D view (AI Studio style).
            new_message: Any = ADKMessage("user", user_message)
            if input_data.annotation and input_data.annotation.screenshot_base64:
                try:
                    from google.genai import types as genai_types
                    b64 = input_data.annotation.screenshot_base64
                    # Strip the data-URL prefix (data:image/png;base64,...)
                    if ',' in b64:
                        b64 = b64.split(',', 1)[1]
                    img_bytes = base64.b64decode(b64)
                    new_message = genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(text=user_message),
                            genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        ],
                    )
                    print(f"[Multimodal] Annotation screenshot included ({len(img_bytes)} bytes)", flush=True)
                except Exception as mm_err:
                    print(f"[Multimodal] Falling back to text-only: {mm_err}", flush=True)

            # STEP_STARTED
            yield encoder.encode({
                "type": EventType.STEP_STARTED.value,
                "stepName": "agent_processing",
            })

            message_id = str(uuid.uuid4())

            # TEXT_MESSAGE_START
            yield encoder.encode({
                "type": EventType.TEXT_MESSAGE_START.value,
                "messageId": message_id,
                "role": "assistant",
            })

            # Ensure session exists
            try:
                # InMemorySessionService is async in this version
                session = await local_session_service.get_session(
                    app_name="agui_core", 
                    user_id="agui_user", 
                    session_id=thread_id
                )
                
                if not session:
                    await local_session_service.create_session(
                        app_name="agui_core", 
                        user_id="agui_user", 
                        session_id=thread_id
                    )
            except Exception as e:
                print(f"WARNING: Session check/create failed: {e}", flush=True)
                try:
                    await local_session_service.create_session(
                        app_name="agui_core", 
                        user_id="agui_user", 
                        session_id=thread_id
                    )
                except Exception:
                    pass

            # Initialize state
            current_state = dict(input_data.state) if input_data.state else {}
            current_state["run_id"] = run_id
            current_state["artifacts"] = []

            response_text = ""
            tool_calls_made = []

            # --- helper: extract Google's retryDelay from error payload (seconds) ---
            def _parse_retry_delay(exc: Exception) -> Optional[float]:
                """Pull retryDelay from Google's error — checks exc + __cause__ chain.

                Google embeds the delay in two places:
                  1. details[].retryDelay field:  'retryDelay': '41s'
                  2. message text:                'Please retry in 41.489538868s.'
                """
                import re
                # Collect the full error text: the exception itself + any chained cause
                candidates = [str(exc)]
                cause = getattr(exc, '__cause__', None)
                if cause:
                    candidates.append(str(cause))
                full_text = " ".join(candidates)

                # Pattern 1: structured retryDelay field (Python dict repr)
                m = re.search(r"['\"]retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s['\"]", full_text)
                if m:
                    print(f"[retryDelay] parsed from details field: {m.group(1)}s", flush=True)
                    return float(m.group(1))

                # Pattern 2: human-readable message "Please retry in 41.489s."
                m = re.search(r"[Rr]etry in (\d+(?:\.\d+)?)s", full_text)
                if m:
                    print(f"[retryDelay] parsed from message text: {m.group(1)}s", flush=True)
                    return float(m.group(1))

                # Nothing found — log first 300 chars for debugging next time
                print(f"[retryDelay] could not parse delay. Error snippet: {full_text[:300]}", flush=True)
                return None

            MAX_AGENT_RETRIES = 5
            agent_attempt = 0

            while agent_attempt <= MAX_AGENT_RETRIES:
                try:
                    async for event in local_runner.run_async(
                        user_id="agui_user",
                        session_id=thread_id,
                        new_message=new_message
                    ):
                        # Handle text content
                        if hasattr(event, 'content') and event.content:
                            for part in event.content.parts if hasattr(event.content, 'parts') else [event.content]:
                                if hasattr(part, 'text') and part.text:
                                    yield encoder.encode({
                                        "type": EventType.TEXT_MESSAGE_CONTENT.value,
                                        "messageId": message_id,
                                        "delta": part.text,
                                    })
                                    response_text += part.text

                                # Handle function calls from ADK
                                if hasattr(part, 'function_call') and part.function_call:
                                    fc = part.function_call
                                    tool_call_id = str(uuid.uuid4())
                                    tool_name = fc.name if hasattr(fc, 'name') else str(fc)
                                    tool_args = dict(fc.args) if hasattr(fc, 'args') else {}

                                    # TOOL_CALL_START
                                    yield encoder.encode({
                                        "type": EventType.TOOL_CALL_START.value,
                                        "toolCallId": tool_call_id,
                                        "toolCallName": tool_name,
                                        "parentMessageId": message_id,
                                    })

                                    # TOOL_CALL_ARGS
                                    yield encoder.encode({
                                        "type": EventType.TOOL_CALL_ARGS.value,
                                        "toolCallId": tool_call_id,
                                        "delta": json.dumps(tool_args),
                                    })

                                    # TOOL_CALL_END
                                    yield encoder.encode({
                                        "type": EventType.TOOL_CALL_END.value,
                                        "toolCallId": tool_call_id,
                                    })

                                    tool_calls_made.append({
                                        "id": tool_call_id,
                                        "name": tool_name,
                                        "args": tool_args
                                    })

                                # Handle function responses
                                if hasattr(part, 'function_response') and part.function_response:
                                    fr = part.function_response
                                    tool_name = fr.name if hasattr(fr, 'name') else "unknown"
                                    result = fr.response if hasattr(fr, 'response') else {}

                                    # Create artifacts based on tool results
                                    artifact = await self._create_artifact_from_result(
                                        run_id, tool_name, result,
                                        conversation_id=conversation["id"],
                                        workspace_id=workspace_id
                                    )
                                    if artifact:
                                        current_state["artifacts"].append(artifact["id"])

                                        # Also persist to database for search/filtering
                                        try:
                                            artifact_service.create(
                                                artifact_type=artifact.get("type", "unknown"),
                                                content=artifact.get("content", {}),
                                                conversation_id=conversation["id"],
                                                workspace_id=workspace_id,
                                                title=artifact.get("title") or artifact_service._generate_title(
                                                    artifact.get("type"), artifact.get("content", {})
                                                ),
                                                metadata={"run_id": run_id, "original_id": artifact["id"]}
                                            )
                                        except Exception as db_err:
                                            print(f"Warning: Failed to persist artifact to DB: {db_err}")

                                        # Send artifact as custom event
                                        yield encoder.encode({
                                            "type": EventType.CUSTOM.value,
                                            "name": "artifact_created",
                                            "value": artifact,
                                        })

                                    # Update state based on results
                                    self._update_state_from_result(current_state, tool_name, result)

                                    # Send state delta
                                    yield encoder.encode({
                                        "type": EventType.STATE_DELTA.value,
                                        "delta": [
                                            {"op": "add", "path": f"/{tool_name}_result", "value": True}
                                        ]
                                    })

                        # Handle direct text from event
                        elif hasattr(event, 'text') and event.text:
                            yield encoder.encode({
                                "type": EventType.TEXT_MESSAGE_CONTENT.value,
                                "messageId": message_id,
                                "delta": event.text,
                            })
                            response_text += event.text

                    # Stream completed successfully — break out of retry loop
                    model_switcher.record_success("coordinator")
                    break

                except Exception as e:
                    import traceback as _tb
                    error_msg = str(e).lower()
                    is_429 = "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg
                    is_503 = "503" in error_msg or "overload" in error_msg or "unavailable" in error_msg
                    is_retryable = is_429 or is_503

                    if not is_retryable or agent_attempt >= MAX_AGENT_RETRIES:
                        # Non-retryable or exhausted all attempts — surface the error
                        _tb.print_exc()
                        if agent_attempt >= MAX_AGENT_RETRIES and is_retryable:
                            yield encoder.encode({
                                "type": EventType.TEXT_MESSAGE_CONTENT.value,
                                "messageId": message_id,
                                "delta": f"\n\n❌ **API quota/overload persists after {MAX_AGENT_RETRIES} retries.** "
                                         f"Please wait a few minutes and try again, or upgrade your Google API plan.\n",
                            })
                        raise e

                    agent_attempt += 1

                    # --- determine wait time ---
                    # 429: honour Google's retryDelay; 503: use backoff
                    google_delay = _parse_retry_delay(e)
                    if google_delay is not None:
                        retry_delay = google_delay + random.uniform(1, 5)  # small jitter on top
                    else:
                        retry_delay = min(
                            RETRY.initial_delay * (RETRY.exponential_base ** agent_attempt),
                            RETRY.max_delay
                        ) * (0.5 + random.random())

                    # --- build user-facing status ---
                    if is_429:
                        status_label = "Rate limit / quota hit (429)"
                    else:
                        status_label = "Model overloaded (503)"

                    print(f"⚠️  [{status_label}] attempt {agent_attempt}/{MAX_AGENT_RETRIES} — "
                          f"waiting {retry_delay:.1f}s before retry...", flush=True)

                    yield encoder.encode({
                        "type": EventType.TEXT_MESSAGE_CONTENT.value,
                        "messageId": message_id,
                        "delta": f"\n\n⏳ **{status_label}.** Retry {agent_attempt}/{MAX_AGENT_RETRIES} "
                                 f"in {retry_delay:.0f}s...\n",
                    })

                    await asyncio.sleep(retry_delay)

                    # --- switch to fallback model on BOTH 429 and 503 ---
                    # 429 quota is per-model: switching to a different model
                    # hits a different RPM bucket immediately.
                    model_switcher.record_failure("coordinator")
                    model_switcher.record_failure("research")
                    model_switcher.record_failure("engineering")
                    model_switcher.record_failure("evolution")
                    local_runner, local_session_service = create_runner_with_models(use_fallback=True)
                    try:
                        await local_session_service.create_session(
                            app_name="agui_core",
                            user_id="agui_user",
                            session_id=thread_id
                        )
                    except Exception:
                        pass

            # TEXT_MESSAGE_END
            yield encoder.encode({
                "type": EventType.TEXT_MESSAGE_END.value,
                "messageId": message_id,
            })

            # STEP_FINISHED
            yield encoder.encode({
                "type": EventType.STEP_FINISHED.value,
                "stepName": "agent_processing",
            })

            # Get session state
            try:
                session = await local_session_service.get_session(
                    app_name="agui_core",
                    user_id="agui_user",
                    session_id=thread_id
                )
                if session and hasattr(session, 'state'):
                    current_state.update(dict(session.state))
            except Exception:
                pass

            # Emit agent artifacts to frontend
            adk_artifacts_dir = CORE_ROOT / "agents" / ".adk" / "artifacts"
            if adk_artifacts_dir.exists():
                for artifact_file in adk_artifacts_dir.glob("*.json"):
                    try:
                        with open(artifact_file, "r") as f:
                            artifact = json.load(f)
                        artifact_id = artifact.get("id", artifact_file.stem)
                        if artifact_id not in self.artifact_manager.artifacts:
                            self.artifact_manager.artifacts[artifact_id] = artifact
                            yield encoder.encode({
                                "type": EventType.ACTIVITY_SNAPSHOT.value,
                                "content": artifact,
                            })
                    except Exception as e:
                        continue

            # STATE_SNAPSHOT
            yield encoder.encode({
                "type": EventType.STATE_SNAPSHOT.value,
                "snapshot": current_state,
            })

            # Persist assistant response to conversation
            if response_text:
                conversation_service.add_message(
                    conversation_id=conversation["id"],
                    role="assistant",
                    content=response_text,
                    tool_calls=tool_calls_made if tool_calls_made else None,
                    metadata={"run_id": run_id, "artifacts": current_state.get("artifacts", [])}
                )

            # Extract knowledge from conversation (Antigravity-style self-improvement)
            try:
                extracted_knowledge = knowledge_service.extract_from_conversation(
                    conversation_id=conversation["id"],
                    workspace_id=workspace_id
                )
                if extracted_knowledge:
                    yield encoder.encode({
                        "type": EventType.CUSTOM.value,
                        "name": "knowledge_extracted",
                        "value": {"count": len(extracted_knowledge), "items": extracted_knowledge}
                    })
            except Exception as ke:
                print(f"Knowledge extraction warning: {ke}")

            # RUN_FINISHED
            yield encoder.encode({
                "type": EventType.RUN_FINISHED.value,
                "threadId": thread_id,
                "runId": run_id,
                "conversationId": conversation["id"],
                "result": {
                    "response": response_text,
                    "toolCalls": tool_calls_made,
                    "artifacts": current_state.get("artifacts", []),
                }
            })

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in agent run: {error_trace}", flush=True)

            # Check if this is a Gemini overload error (503)
            error_msg = str(e).lower()
            is_503_error = any(pattern in error_msg for pattern in ["503", "overload", "unavailable"])

            # Prepare user-friendly error message
            if is_503_error:
                user_message = (
                    "⚠️ Gemini API overload detected (503 UNAVAILABLE). "
                    "This is a temporary infrastructure issue at Google. "
                    "The system attempted automatic retries but capacity remains exhausted. "
                    "\n\n**Recommendations:**\n"
                    "- Retry during off-peak hours (00:00-06:00 Beijing time)\n"
                    "- Or wait 5-10 minutes and try again\n"
                    f"\n**Technical details:** {str(e)}"
                )
                error_code = "GEMINI_503_OVERLOAD"
            else:
                user_message = f"Agent execution failed: {str(e)}"
                error_code = "AGENT_ERROR"

            yield encoder.encode({
                "type": EventType.RUN_ERROR.value,
                "message": user_message,
                "code": error_code,
            })
    async def _create_artifact_from_result(self, run_id: str, tool_name: str, result: Any,
                                           conversation_id: str = None, workspace_id: str = None) -> Optional[Dict]:
        """Create appropriate artifact from tool result and persist to database."""
        result_dict = dict(result) if hasattr(result, 'items') else {}
        workspace_id = workspace_id or self.default_workspace_id

        # Check for artifact_id (from wrapper tools) or id (from direct artifact returns)
        if "artifact_id" in result_dict or "id" in result_dict:
            artifact_id = result_dict.get("artifact_id") or result_dict.get("id")
            artifact = None

            # 1. .adk/artifacts/ first — this is where agents write in-flight
            #    updates (e.g. update_task_status).  SQLite is a snapshot taken
            #    at creation time and will be stale for any mutated artifact.
            adk_artifacts_dir = CORE_ROOT / "agents" / ".adk" / "artifacts"
            artifact_file = adk_artifacts_dir / f"{artifact_id}.json"
            if artifact_file.exists():
                try:
                    with open(artifact_file) as f:
                        artifact = json.load(f)
                    print(f"Found artifact in .adk/artifacts/: {artifact_id}")
                except Exception as e:
                    print(f"Error reading .adk artifact {artifact_id}: {e}")

            # 2. data/artifacts/ directory (persisted from previous runs)
            if not artifact:
                data_artifacts_dir = CORE_ROOT / "data" / "artifacts"
                artifact_file = data_artifacts_dir / f"{artifact_id}.json"
                if artifact_file.exists():
                    try:
                        with open(artifact_file) as f:
                            artifact = json.load(f)
                        print(f"Found artifact in data/artifacts/: {artifact_id}")
                    except Exception as e:
                        print(f"Error reading data artifact {artifact_id}: {e}")

            # 3. SQLite as last resort (may be stale for in-flight artifacts)
            if not artifact:
                try:
                    from core.api.services import artifact_service
                    artifact = artifact_service.get(artifact_id)
                except Exception as e:
                    print(f"SQLite lookup failed for {artifact_id}: {e}")

            if artifact:
                # Register with local ArtifactManager for UI persistence/listing
                if "run_id" not in artifact:
                    artifact["run_id"] = run_id

                self.artifact_manager.artifacts[artifact["id"]] = artifact
                self.artifact_manager._save_to_disk()
                return artifact

        if tool_name == "fetch_ped_ensemble":
            pdb_path = result_dict.get("pdb_path")
            ped_id = result_dict.get("ped_id", "")
            if pdb_path and os.path.exists(pdb_path):
                try:
                    with open(pdb_path, 'r') as f:
                        pdb_data = f.read()

                    # Look up binding site from result, or from known sites, or default to empty
                    binding_site = (
                        result_dict.get("binding_site") or
                        result_dict.get("binding_site_residues") or
                        KNOWN_BINDING_SITES.get(ped_id, [])
                    )

                    return self.artifact_manager.create_structure_artifact(
                        run_id, pdb_data, {
                            "ped_id": ped_id,
                            "n_conformations": result_dict.get("n_conformations"),
                            "binding_site": binding_site
                        }
                    )
                except Exception as e:
                    print(f"Failed to read PDB: {e}")

        elif tool_name == "prepare_ligand":
            # Generate SVG artifact for the prepared ligand
            if RDKIT_AVAILABLE and "smiles" in result_dict:
                try:
                    mol = Chem.MolFromSmiles(result_dict["smiles"])
                    if mol:
                        svg = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(300, 300), useSVG=True)
                        # Clean up SVG XML header
                        svg = svg.replace('<?xml version="1.0" encoding="iso-8859-1"?>', '')
                        svg = svg.replace('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">', '')
                        
                        return {
                            "id": f"artifact_{uuid.uuid4().hex[:8]}",
                            "type": "ligand_svg",
                            "run_id": run_id,
                            "created_at": datetime.now().isoformat(),
                            "content": {
                                "svg": svg,
                                "smiles": result_dict["smiles"],
                                "name": result_dict.get("ligand_id", "Ligand")
                            }
                        }
                except Exception as e:
                    print(f"RDKit rendering failed: {e}")

        elif tool_name == "bc_get_europepmc_fulltext":
            # Create Literature Artifact with Figures
            if "figures" in result_dict and result_dict["figures"]:
                return {
                    "id": f"artifact_{uuid.uuid4().hex[:8]}",
                    "type": ArtifactType.LITERATURE_RESULT.value,
                    "run_id": run_id,
                    "created_at": datetime.now().isoformat(),
                    "content": {
                        "title": result_dict.get("title", "Literature Evidence"),
                        "pmcid": result_dict.get("pmcid"),
                        "figures": result_dict["figures"], # List of {url, caption}
                        "summary": result_dict.get("abstract", "")[:200] + "..."
                    }
                }

        elif tool_name == "cluster_conformations":
            return self.artifact_manager.create_cluster_artifact(run_id, result_dict)

        elif tool_name == "dock_ensemble":
            return self.artifact_manager.create_docking_artifact(run_id, result_dict)

        elif tool_name == "analyze_interactions":
            return self.artifact_manager.create_interaction_artifact(run_id, result_dict)

        elif tool_name == "generate_molecules_direct":
            suggestions = result_dict.get("suggestions", [])
            if suggestions:
                # Strip redundant coordinates_3d — pdb_block carries the same data
                clean = [{k: v for k, v in s.items() if k != 'coordinates_3d'} for s in suggestions]
                return {
                    "id": f"artifact_{uuid.uuid4().hex[:8]}",
                    "type": "molecule_suggestions",
                    "run_id": run_id,
                    "created_at": datetime.now().isoformat(),
                    "content": {
                        "title": "Generated Molecules",
                        "modification_type": result_dict.get("modification_type", "optimize"),
                        "generation_method": result_dict.get("generation_method", "smiles_rdkit"),
                        "suggestions": clean,
                    }
                }

        return None

    def _update_state_from_result(self, state: Dict, tool_name: str, result: Any):
        """Update state based on tool results."""
        result_dict = dict(result) if hasattr(result, 'items') else {}

        if tool_name == "fetch_ped_ensemble":
            state["pdb_path"] = result_dict.get("pdb_path")
            state["n_conformations"] = result_dict.get("n_conformations")
            state["has_ensemble"] = True

        elif tool_name == "prepare_ligand":
            state["pdbqt_path"] = result_dict.get("pdbqt_path")
            state["ligand_id"] = result_dict.get("ligand_id")
            state["has_ligand"] = True

        elif tool_name == "cluster_conformations":
            state["representative_frames"] = result_dict.get("representative_frames")
            state["n_clusters"] = result_dict.get("n_clusters")
            state["has_clusters"] = True

        elif tool_name == "dock_ensemble":
            state["binding_scores"] = result_dict.get("binding_scores")
            state["ensemble_average"] = result_dict.get("ensemble_summary", {}).get("weighted_average")
            state["best_energy"] = min(result_dict.get("binding_scores", [0])) if result_dict.get("binding_scores") else None
            state["has_docking"] = True

        elif tool_name == "analyze_interactions":
            state["interactions"] = result_dict
            state["has_interactions"] = True


# Global runner instance
_agent_runner = ADKAgentRunner()


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    print("=" * 60)
    print("Antimatters Server Starting")
    print("=" * 60)
    print(f"API Key configured: {'Yes' if os.getenv('GOOGLE_API_KEY') else 'No'}")
    print(f"Protocol: AG-UI v1.0")
    print(f"Features: Artifacts, Streaming, State Management, Annotations")
    print("=" * 60)
    yield
    print("Antimatters Server shutting down...")


app = FastAPI(
    title="Antimatters AG-UI Server",
    description="""
    AG-UI Protocol compatible server for Antimatters.

    Features:
    - Real-time event streaming and Tool call visualization should be made better. One, add more detailed thinking and dislpay like structure in box, lines and bullets. Second, there can be phrase while it thinks whcih tell about antimatters iteslf like physics that realtes to information, compuation and evolution.
    - Artifact generation (3D structures, docking results, interaction maps), annotation support for multimodal analysis. Add storage forit along with chat like sidebar should show chat name along with artifact names as bullets
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check and capabilities."""
    return {
        "status": "ok",
        "service": "Antimatters AG-UI Server",
        "protocol": "AG-UI v1.0",
        "capabilities": {
            "streaming": True,
            "tools": True,
            "state_management": True,
            "artifacts": True,
            "annotations": True,
            "multimodal": True,
        },
        "artifact_types": [t.value for t in ArtifactType],
        "event_types": [e.value for e in EventType],
        "agents": ["coordinator", "research_agent", "engineer_agent", "parallel_docking", "docking_optimizer"],
        "mcp_servers": ["docking", "ped", "chembl", "biocontext"],
    }


@app.get("/model-status")
async def get_model_status():
    """Get current model configuration and fallback status."""
    from core.agents.config import MODELS, FALLBACKS
    status = model_switcher.get_status()
    return {
        "primary_models": {
            "research": MODELS.research,
            "engineering": MODELS.engineering,
            "evolution": MODELS.evolution,
            "coordinator": MODELS.coordinator,
        },
        "current_models": {
            agent: info["current_model"]
            for agent, info in status.items()
        },
        "fallback_status": status,
        "retry_config": {
            "max_retries": RETRY.max_retries,
            "model_switch_after_retries": RETRY.model_switch_after_retries,
            "model_rotation_enabled": RETRY.model_rotation_enabled,
        }
    }


@app.get("/evolution/graph")
async def get_evolution_graph(protein_name: str = "Alpha-Synuclein"):
    """Get knowledge graph from Neo4j via evolution agent.

    Returns entities and relationships for visualization.
    Leverages existing Neo4j connection and query tools.
    """
    try:
        from core.agents.antimatters._subagents.evolution.agent import query_knowledge_graph

        result = query_knowledge_graph(entity_type=None, property_filter_json=None, limit=200)

        if result.get("success"):
            return {
                "entities": result.get("entities", []),
                "relationships": result.get("relationships", []),
                "success": True
            }
        else:
            return {
                "entities": [],
                "relationships": [],
                "success": False,
                "error": result.get("error", "Unknown error")
            }
    except Exception as e:
        return {
            "entities": [],
            "relationships": [],
            "success": False,
            "error": str(e)
        }


@app.post("/evolution/expand")
async def expand_references(request: Request):
    """Expand @Type:Name references to full KG context.

    Input: {"references": ["@Ligand:Fasudil", "@Residue:Y125"]}
    Output: {"context": {...entity data...}, "markdown": "...formatted context..."}
    """
    try:
        from core.agents.antimatters._subagents.evolution.agent import get_neo4j_driver

        data = await request.json()
        refs = data.get("references", [])

        context = {}
        markdown_parts = []

        driver, database = get_neo4j_driver()

        for ref in refs:
            # Parse @Type:Name format
            if not ref.startswith("@"):
                continue
            parts = ref[1:].split(":", 1)
            entity_type = parts[0] if len(parts) > 1 else None
            entity_name = parts[1] if len(parts) > 1 else parts[0]

            # Query KG by name (case-insensitive) + INTERACTS_WITH relationships
            with driver.session(database=database) as session:
                result = session.run("""
                    MATCH (n:Entity)
                    WHERE toLower(n.name) = toLower($name)
                    OPTIONAL MATCH (n)-[r:RELATIONSHIP {type: 'INTERACTS_WITH'}]->(residue:Entity)
                    WITH n, collect({
                        residue_name: residue.name,
                        residue_id: residue.id,
                        interaction_props: r.properties
                    }) AS interactions
                    RETURN n.id AS id, n.name AS name, n.type AS type,
                           n.properties AS properties, interactions
                    LIMIT 1
                """, {"name": entity_name})

                record = result.single()
                if record:
                    props = json.loads(record["properties"]) if record["properties"] else {}
                    interactions = record.get("interactions", [])

                    # Extract interaction types
                    interaction_list = []
                    for intr in interactions:
                        if intr.get("residue_name"):
                            int_props = json.loads(intr["interaction_props"]) if intr.get("interaction_props") else {}
                            interaction_list.append({
                                "residue": intr["residue_name"],
                                "residue_id": intr["residue_id"],
                                "type": int_props.get("interaction_type", "unknown")
                            })

                    context[ref] = {
                        "id": record["id"],
                        "type": record["type"],
                        "name": record["name"],
                        "properties": props,
                        "interactions": interaction_list  # Spatial relationships
                    }

                    # Format for LLM context
                    md = f"**{ref}** ({record['type']})\n"
                    if props.get("smiles"):
                        md += f"- SMILES: `{props['smiles']}`\n"
                    if props.get("best_energy"):
                        md += f"- Best energy: {props['best_energy']} kcal/mol\n"
                    if interaction_list:
                        int_summary = [f"{i['residue']}({i['type']})" for i in interaction_list]
                        md += f"- INTERACTS_WITH: {', '.join(int_summary)}\n"
                    markdown_parts.append(md)

        driver.close()

        return {
            "success": True,
            "context": context,
            "markdown": "\n".join(markdown_parts) if markdown_parts else None,
            "n_resolved": len(context)
        }

    except Exception as e:
        return {"success": False, "error": str(e), "context": {}}


@app.post("/model-status/reset")
async def reset_model_status():
    """Reset all models to primary configuration."""
    model_switcher.reset()
    return {"status": "ok", "message": "All models reset to primary", "current": model_switcher.get_status()}

# Legacy feedback input (deprecated, use /feedback POST with FeedbackCreate)
class FeedbackInputLegacy(BaseModel):
    run_id: str
    artifact_id: str
    comment: str
    selection: Optional[str] = None


@app.post("/feedback/legacy")
async def handle_feedback_legacy(feedback: FeedbackInputLegacy):
    """Handle async user feedback on an artifact (legacy endpoint)."""
    print(f"Received feedback for {feedback.artifact_id}: {feedback.comment}")

    # Use the new feedback service
    fb = feedback_service.add(
        artifact_id=feedback.artifact_id,
        comment=feedback.comment
    )

    return {"status": "received", "feedback_id": fb["id"]}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str = None, workspace_id: str = None):
    """WebSocket endpoint for real-time artifact updates.

    Clients connect here to receive live updates as artifacts are created/modified.
    Inspired by LiveDesign's real-time collaboration model.
    """
    from core.api.websocket_manager import ws_manager

    await ws_manager.connect(websocket, user_id=user_id, workspace_id=workspace_id)

    try:
        while True:
            # Keep connection alive, wait for client messages (if any)
            data = await websocket.receive_text()
            # Echo back as heartbeat
            await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket)


@app.post("/")
async def run_agent(
    input_data: RunAgentInput,
    request: Request
):
    """AG-UI protocol endpoint - accepts RunAgentInput, returns event stream."""
    accept_header = request.headers.get("accept", "text/event-stream")
    encoder = EventEncoder(accept=accept_header)

    async def event_generator():
        async for event in _agent_runner.run_with_events(input_data, encoder):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# where is bioconext, those 3 litrature tools

@app.get("/tools")
async def get_available_tools():
    """Return available tools for frontend tool definitions."""
    return {
        "tools": [
            {
                "name": "fetch_ped_ensemble",
                "description": "Fetch protein ensemble from PED database for IDP analysis",
                "category": "data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ped_id": {
                            "type": "string",
                            "description": "PED entry ID (e.g., PED00006e001)"
                        }
                    },
                    "required": ["ped_id"]
                }
            },
            {
                "name": "prepare_ligand",
                "description": "Prepare ligand from SMILES string for molecular docking",
                "category": "preparation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "smiles": {
                            "type": "string",
                            "description": "SMILES string of the ligand molecule"
                        },
                        "optimize": {
                            "type": "boolean",
                            "description": "Whether to optimize 3D geometry",
                            "default": True
                        }
                    },
                    "required": ["smiles"]
                }
            },
            {
                "name": "cluster_conformations",
                "description": "Cluster protein conformations using t-SNE and k-means for ensemble docking",
                "category": "analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_path": {
                            "type": "string",
                            "description": "Path to multi-model PDB file"
                        },
                        "n_clusters": {
                            "type": "integer",
                            "description": "Number of representative clusters",
                            "default": 20
                        }
                    },
                    "required": ["pdb_path"]
                }
            },
            {
                "name": "dock_ensemble",
                "description": "Dock ligand to protein ensemble using AutoDock Vina",
                "category": "docking",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "protein_pdb": {"type": "string", "description": "Path to protein PDB"},
                        "ligand_pdbqt": {"type": "string", "description": "Path to ligand PDBQT"},
                        "representative_frames": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Frame indices to dock"
                        },
                        "residue_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Binding site residues",
                            "default": [125, 133, 136]
                        },
                        "exhaustiveness": {
                            "type": "integer",
                            "description": "Search exhaustiveness",
                            "default": 32
                        }
                    },
                    "required": ["protein_pdb", "ligand_pdbqt", "representative_frames"]
                }
            },
            {
                "name": "analyze_interactions",
                "description": "Analyze protein-ligand interactions (H-bonds, hydrophobic, aromatic)",
                "category": "analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "protein_pdb": {"type": "string"},
                        "ligand_pdbqt": {"type": "string"},
                        "frame_indices": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "required": ["protein_pdb", "ligand_pdbqt"]
                }
            },
            {
                "name": "search_compounds",
                "description": "Search ChEMBL database for compounds",
                "category": "research",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "similarity_search",
                "description": "Find similar compounds by SMILES structure",
                "category": "research",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "smiles": {"type": "string", "description": "Reference SMILES"},
                        "threshold": {"type": "number", "default": 0.7}
                    },
                    "required": ["smiles"]
                }
            }
        ]
    }


@app.get("/ligands")
async def get_known_ligands():
    """Return example ligands - users can provide any SMILES."""
    return {
        "ligands": [],
        "note": "Provide ligand SMILES directly in your request"
    }

# make in dynamic as whatver .pdb files is fetced form protein ensemble database: mcp_server/ensemble_docking_server.py
@app.get("/ensembles")
async def get_example_ensembles():
    """Return example PED ensembles for IDP analysis."""
    return {
        "ensembles": [
            {
                "id": "PED00006e001",
                "name": "Alpha-synuclein",
                "conformations": 576,
                "description": "Intrinsically disordered protein linked to Parkinson's disease",
                "binding_site": [125, 133, 136],
                "binding_site_description": "Tyrosine cluster (Y125, Y133, Y136)"
            }
        ]
    }


@app.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str):
    """Get a specific artifact by ID.

    Searches in:
    1. ArtifactManager (runtime artifacts)
    2. .adk/artifacts/ directory (agent-created artifacts)
    """
    # First check ArtifactManager
    artifact = _agent_runner.artifact_manager.artifacts.get(artifact_id)
    if artifact:
        return artifact

    # Then check .adk/artifacts/ directory
    adk_artifacts_dir = CORE_ROOT / "agents" / ".adk" / "artifacts"
    artifact_file = adk_artifacts_dir / f"{artifact_id}.json"
    if artifact_file.exists():
        try:
            with open(artifact_file, "r") as f:
                artifact = json.load(f)

            # Normalize experiment_matrix content for frontend
            if artifact.get("type") == "experiment_matrix":
                content = artifact.get("content", {})
                if "ligand_results" in content and "ligands" not in content:
                    content["ligands"] = [
                        {
                            "name": lr.get("ligand_name", lr.get("name", "unknown")),
                            "smiles": lr.get("smiles", ""),
                            "status": lr.get("status", "queued"),
                            "best_energy": lr.get("best_energy"),
                            "best_cluster": lr.get("best_cluster"),
                            "interaction_types": lr.get("interaction_types", []),
                        }
                        for lr in content["ligand_results"]
                    ]

            return artifact
        except Exception as e:
            print(f"Warning: Failed to load artifact {artifact_file}: {e}")

    raise HTTPException(status_code=404, detail="Artifact not found")

# this should be shown in side tab as "Evolution" where we would leverage some graph library, filtering with proximity and ranking
@app.get("/artifacts")
async def list_artifacts(run_id: Optional[str] = None):
    """List all artifacts, optionally filtered by run_id.

    Agent artifacts (.adk/artifacts/) take priority - they have real data.
    """
    artifacts_dict = {}

    # Load agent artifacts first (source of truth)
    adk_artifacts_dir = CORE_ROOT / "agents" / ".adk" / "artifacts"
    if adk_artifacts_dir.exists():
        for artifact_file in adk_artifacts_dir.glob("*.json"):
            try:
                with open(artifact_file, "r") as f:
                    artifact = json.load(f)
                artifact_id = artifact.get("id", artifact_file.stem)
                if artifact_id not in artifacts_dict:
                    # Normalize format for frontend compatibility
                    # Agent artifacts have metadata.created_at, frontend expects top-level created_at
                    metadata = artifact.get("metadata", {})
                    if "run_id" not in artifact:
                        artifact["run_id"] = metadata.get("parent_id") or "unknown"
                    if "created_at" not in artifact:
                        artifact["created_at"] = metadata.get("created_at", datetime.now().isoformat())
                    if "updated_at" not in artifact:
                        artifact["updated_at"] = metadata.get("updated_at", artifact.get("created_at"))
                    if "version" not in artifact:
                        artifact["version"] = metadata.get("version", 1)

                    # Normalize experiment_matrix content (backend uses ligand_results, frontend expects ligands)
                    if artifact.get("type") == "experiment_matrix":
                        content = artifact.get("content", {})
                        if "ligand_results" in content and "ligands" not in content:
                            # Map ligand_results to ligands format expected by frontend
                            content["ligands"] = [
                                {
                                    "name": lr.get("ligand_name", lr.get("name", "unknown")),
                                    "smiles": lr.get("smiles", ""),
                                    "status": lr.get("status", "queued"),
                                    "best_energy": lr.get("best_energy"),
                                    "best_cluster": lr.get("best_cluster"),
                                    "interaction_types": lr.get("interaction_types", []),
                                }
                                for lr in content["ligand_results"]
                            ]

                    artifacts_dict[artifact_id] = artifact
            except Exception as e:
                print(f"Warning: Failed to load artifact {artifact_file}: {e}")

    # Add tool-generated artifacts from ArtifactManager (ligand SVGs, etc.) as fallback
    for aid, artifact in _agent_runner.artifact_manager.artifacts.items():
        if aid not in artifacts_dict:
            artifacts_dict[aid] = artifact

    artifacts = list(artifacts_dict.values())
    if run_id:
        artifacts = [a for a in artifacts if a.get("run_id") == run_id]

    # Sort by created_at descending (newest first)
    artifacts.sort(key=lambda a: a.get("created_at", ""), reverse=True)

    return {"artifacts": artifacts}

# would browser recording benefical and feasible?
@app.post("/annotate")
async def process_annotation(annotation: AnnotationRegion):
    """Process an annotation from the frontend (AI Studio-style)."""
    return {
        "status": "received",
        "annotation": {
            "residue_count": len(annotation.residue_ids),
            "atom_count": len(annotation.atom_ids),
            "description": annotation.description,
            "has_screenshot": annotation.screenshot_base64 is not None
        },
        "suggestion": f"You can now ask questions about residues {annotation.residue_ids}"
    }


# =============================================================================
# Workspace API Endpoints
# =============================================================================

class WorkspaceCreate(BaseModel):
    name: str
    description: str = ""
    config: Optional[Dict[str, Any]] = None


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@app.get("/workspaces")
async def list_workspaces():
    """List all workspaces."""
    workspaces = workspace_service.list_all()
    # Include stats for each workspace
    for ws in workspaces:
        ws["stats"] = workspace_service.get_stats(ws["id"])
    return {"workspaces": workspaces}


@app.post("/workspaces")
async def create_workspace(data: WorkspaceCreate):
    """Create a new workspace."""
    workspace = workspace_service.create(
        name=data.name,
        description=data.description,
        config=data.config
    )
    return workspace


@app.get("/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str):
    """Get workspace by ID."""
    workspace = workspace_service.get(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    workspace["stats"] = workspace_service.get_stats(workspace_id)
    return workspace


@app.put("/workspaces/{workspace_id}")
async def update_workspace(workspace_id: str, data: WorkspaceUpdate):
    """Update workspace."""
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    workspace = workspace_service.update(workspace_id, **update_data)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@app.delete("/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str):
    """Delete workspace and all associated data."""
    if workspace_id == "ws_core":
        raise HTTPException(status_code=400, detail="Cannot delete default workspace")
    workspace_service.delete(workspace_id)
    return {"status": "deleted", "workspace_id": workspace_id}


# =============================================================================
# Conversation API Endpoints
# =============================================================================

class ConversationCreate(BaseModel):
    workspace_id: str = "ws_core"
    title: Optional[str] = None
    initial_message: Optional[str] = None


class MessageCreate(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Dict]] = None
    metadata: Optional[Dict[str, Any]] = None


@app.get("/conversations")
async def list_conversations(
    workspace_id: str = Query(default="ws_core", description="Filter by workspace")
):
    """List conversations for a workspace."""
    conversations = conversation_service.list_by_workspace(workspace_id)
    return {"conversations": conversations, "workspace_id": workspace_id}


@app.post("/conversations")
async def create_conversation(data: ConversationCreate):
    """Create a new conversation."""
    conversation = conversation_service.create(
        workspace_id=data.workspace_id,
        title=data.title,
        initial_message=data.initial_message
    )
    return conversation


@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    include_messages: bool = Query(default=True, description="Include message history")
):
    """Get conversation by ID with optional message history."""
    conversation = conversation_service.get(conversation_id, include_messages=include_messages)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/conversations/{conversation_id}/messages")
async def add_message(conversation_id: str, data: MessageCreate):
    """Add a message to a conversation."""
    conversation = conversation_service.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    message = conversation_service.add_message(
        conversation_id=conversation_id,
        role=data.role,
        content=data.content,
        tool_calls=data.tool_calls,
        metadata=data.metadata
    )
    return message


@app.put("/conversations/{conversation_id}/title")
async def update_conversation_title(conversation_id: str, title: str = Query(...)):
    """Update conversation title."""
    conversation = conversation_service.update_title(conversation_id, title)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    conversation_service.delete(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}


# =============================================================================
# Knowledge/Evolution API Endpoints
# =============================================================================

class KnowledgeCreate(BaseModel):
    workspace_id: str = "ws_core"
    type: str  # insight, procedure, result, feedback, reference
    title: str
    content: str
    tags: Optional[List[str]] = None
    source_conversation_id: Optional[str] = None
    source_artifact_id: Optional[str] = None


@app.get("/knowledge")
async def list_knowledge(
    workspace_id: str = Query(default="ws_core"),
    knowledge_type: Optional[str] = Query(default=None, description="Filter by type")
):
    """List knowledge items for a workspace (Evolution tab)."""
    knowledge = knowledge_service.list_by_workspace(workspace_id, knowledge_type)
    return {"knowledge": knowledge, "workspace_id": workspace_id}


@app.post("/knowledge")
async def create_knowledge(data: KnowledgeCreate):
    """Manually add knowledge item."""
    knowledge = knowledge_service.create(
        workspace_id=data.workspace_id,
        knowledge_type=data.type,
        title=data.title,
        content=data.content,
        tags=data.tags,
        source_conversation_id=data.source_conversation_id,
        source_artifact_id=data.source_artifact_id
    )
    return knowledge


@app.get("/knowledge/{knowledge_id}")
async def get_knowledge(knowledge_id: str):
    """Get knowledge item by ID."""
    knowledge = knowledge_service.get(knowledge_id)
    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge item not found")
    return knowledge


@app.get("/knowledge/search")
async def search_knowledge(
    q: str = Query(..., description="Search query"),
    workspace_id: Optional[str] = Query(default=None)
):
    """Search knowledge base."""
    results = knowledge_service.search(q, workspace_id)
    return {"results": results, "query": q}


@app.get("/knowledge/context")
async def get_knowledge_context(
    query: str = Query(...),
    workspace_id: str = Query(default="ws_core")
):
    """Get relevant knowledge context for agent prompts (self-improvement)."""
    context = knowledge_service.get_relevant_context(query, workspace_id)
    return {"context": context, "query": query}


@app.post("/conversations/{conversation_id}/extract-knowledge")
async def extract_knowledge_from_conversation(conversation_id: str):
    """Manually trigger knowledge extraction from a conversation."""
    conversation = conversation_service.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    extracted = knowledge_service.extract_from_conversation(
        conversation_id=conversation_id,
        workspace_id=conversation["workspace_id"]
    )
    return {"extracted": extracted, "count": len(extracted)}


# =============================================================================
# Enhanced Artifact API with Database Persistence
# =============================================================================

@app.get("/artifacts/search")
async def search_artifacts(
    q: str = Query(..., description="Search query"),
    workspace_id: Optional[str] = Query(default=None)
):
    """Search artifacts by title or content."""
    results = artifact_service.search(q, workspace_id)
    return {"results": results, "query": q}


@app.get("/artifacts/by-conversation/{conversation_id}")
async def get_artifacts_by_conversation(conversation_id: str):
    """Get all artifacts for a specific conversation."""
    artifacts = artifact_service.list_by_conversation(conversation_id)
    return {"artifacts": artifacts, "conversation_id": conversation_id}


# =============================================================================
# Feedback API Endpoints
# =============================================================================

class FeedbackCreate(BaseModel):
    artifact_id: str
    comment: str
    conversation_id: Optional[str] = None


@app.get("/feedback/{artifact_id}")
async def get_artifact_feedback(artifact_id: str):
    """Get all feedback for an artifact."""
    feedback_list = feedback_service.list_by_artifact(artifact_id)
    return {"feedback": feedback_list, "artifact_id": artifact_id}


@app.post("/feedback")
async def create_feedback(data: FeedbackCreate):
    """Add feedback to an artifact (Google Docs-style comments)."""
    feedback = feedback_service.add(
        artifact_id=data.artifact_id,
        comment=data.comment,
        conversation_id=data.conversation_id
    )

    # Also add to knowledge base as feedback type
    try:
        artifact = artifact_service.get(data.artifact_id)
        if artifact:
            knowledge_service.create(
                workspace_id=artifact.get("workspace_id", "ws_core"),
                knowledge_type="feedback",
                title=f"Feedback on {artifact.get('title', 'artifact')}",
                content=data.comment,
                source_artifact_id=data.artifact_id,
                tags=["user_feedback"]
            )
    except Exception as e:
        print(f"Warning: Could not add feedback to knowledge base: {e}")

    return feedback


@app.put("/feedback/{feedback_id}/resolve")
async def resolve_feedback(feedback_id: str):
    """Mark feedback as resolved."""
    feedback = feedback_service.resolve(feedback_id)
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return feedback


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
