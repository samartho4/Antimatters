"""Antimatters API Server - Connects frontend to ADK agents."""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

# Add core to path BEFORE other imports
CORE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CORE_ROOT))
sys.path.insert(0, str(CORE_ROOT.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment
load_dotenv(CORE_ROOT / ".env")

# Import ADK components
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Session service for state persistence
session_service = InMemorySessionService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    print("Antimatters API Server starting...")
    print(f"API Key configured: {'Yes' if os.getenv('GOOGLE_API_KEY') else 'No'}")
    yield
    print("Antimatters API Server shutting down...")

app = FastAPI(
    title="Antimatters API",
    description="Multi-agent scientific research platform",
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

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    mode: Optional[str] = "agent"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    artifacts: Optional[dict] = None

class DockingRequest(BaseModel):
    ped_id: str
    ligand_smiles: str
    ligand_name: Optional[str] = "ligand"
    n_clusters: int = 10
    session_id: Optional[str] = "default"

# Store active runners per session
_runners: dict = {}

def get_runner(session_id: str) -> Runner:
    """Get or create a runner for a session."""
    if session_id not in _runners:
        _runners[session_id] = Runner(
            agent=root_agent,
            app_name="antimatters_api",
            session_service=session_service
        )
    return _runners[session_id]


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "Antimatters API",
        "agents": ["research_agent", "engineer_agent", "parallel_docking", "docking_optimizer"],
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """Send a message to the agent system."""
    try:
        runner = get_runner(request.session_id)

        # Run the agent
        response = await runner.run_async(
            user_id="api_user",
            session_id=request.session_id,
            new_message=request.message
        )

        # Extract text response
        response_text = ""
        artifacts = {}

        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'content'):
            response_text = str(response.content)
        else:
            response_text = str(response)

        # Check for tool results in session state
        session = session_service.get_session(
            app_name="antimatters_api",
            user_id="api_user",
            session_id=request.session_id
        )
        if session and hasattr(session, 'state'):
            artifacts = {
                "pdb_path": session.state.get("pdb_path"),
                "pdbqt_path": session.state.get("pdbqt_path"),
                "representative_frames": session.state.get("representative_frames"),
                "best_energy": session.state.get("best_energy"),
            }

        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            artifacts=artifacts
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response."""
    async def generate() -> AsyncGenerator[str, None]:
        try:
            runner = get_runner(request.session_id)

            yield f"data: {json.dumps({'type': 'start', 'message': 'Processing...'})}\n\n"

            response = await runner.run_async(
                user_id="api_user",
                session_id=request.session_id,
                new_message=request.message
            )

            response_text = str(response) if response else "No response"

            chunk_size = 50
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                await asyncio.sleep(0.01)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/dock")
async def dock_ligand(request: DockingRequest):
    """Run a complete docking workflow."""
    try:
        runner = get_runner(request.session_id)

        # Construct the docking prompt
        prompt = f"""Run a complete docking workflow:
1. Fetch ensemble {request.ped_id}
2. Prepare ligand {request.ligand_name} (SMILES: {request.ligand_smiles})
3. Cluster to {request.n_clusters} representatives
4. Dock and report binding energies

Execute all steps and report results."""

        response = await runner.run_async(
            user_id="api_user",
            session_id=request.session_id,
            new_message=prompt
        )

        # Get state
        session = session_service.get_session(
            app_name="antimatters_api",
            user_id="api_user",
            session_id=request.session_id
        )

        state = {}
        if session and hasattr(session, 'state'):
            state = dict(session.state)

        return {
            "success": True,
            "response": str(response),
            "results": state
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/state")
async def get_session_state(session_id: str):
    """Get current session state."""
    try:
        session = session_service.get_session(
            app_name="antimatters_api",
            user_id="api_user",
            session_id=session_id
        )

        if not session:
            return {"state": {}}

        return {"state": dict(session.state) if hasattr(session, 'state') else {}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
