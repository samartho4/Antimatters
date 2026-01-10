"""
IDoIDR API Server - FastAPI with WebSocket for real-time agent updates
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IDoIDR API",
    description="Agentic IDP Drug Discovery Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections
connections: List[WebSocket] = []

# State
discovery_state = {
    "status": "idle",
    "current_step": None,
    "results": {},
    "logs": []
}


class DiscoveryRequest(BaseModel):
    query: str = "Find novel binders for α-synuclein"
    n_conformers: int = 50
    top_n: int = 10


class DockingRequest(BaseModel):
    smiles: str
    name: str
    n_conformers: int = 50


# WebSocket broadcast
async def broadcast(message: Dict):
    """Broadcast message to all connected clients"""
    for ws in connections:
        try:
            await ws.send_json(message)
        except:
            pass


def log_event(event: str, data: Any = None):
    """Log event and broadcast"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "data": data
    }
    discovery_state["logs"].append(entry)
    asyncio.create_task(broadcast({"type": "log", **entry}))


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard"""
    dashboard_path = Path(__file__).parent.parent / "frontend" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return HTMLResponse("<h1>IDoIDR API</h1><p>Dashboard not found</p>")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    connections.append(websocket)
    
    # Send current state
    await websocket.send_json({
        "type": "state",
        "data": discovery_state
    })
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg.get("type") == "start_discovery":
                asyncio.create_task(run_discovery_async(msg.get("query", "")))
            
    except WebSocketDisconnect:
        connections.remove(websocket)


@app.get("/api/status")
async def get_status():
    """Get current discovery status"""
    return discovery_state


@app.post("/api/discovery")
async def start_discovery(request: DiscoveryRequest):
    """Start discovery workflow"""
    asyncio.create_task(run_discovery_async(request.query))
    return {"status": "started", "query": request.query}


@app.get("/api/ensemble")
async def get_ensemble():
    """Get ensemble statistics"""
    from tools.ensemble import EnsembleProcessor
    
    processor = EnsembleProcessor("./data")
    ensemble = processor.load_ensemble()
    validation = processor.validate_ensemble(ensemble)
    
    return {
        "metadata": ensemble.metadata,
        "validation": validation,
        "feature_summary": {
            "rg": {
                "mean": float(ensemble.features['rg'].mean()),
                "std": float(ensemble.features['rg'].std())
            }
        }
    }


@app.post("/api/dock")
async def dock_ligand_endpoint(request: DockingRequest):
    """Dock single ligand"""
    from tools.ensemble import EnsembleProcessor
    from tools.docking import dock_ligand
    
    processor = EnsembleProcessor("./data")
    ensemble = processor.load_ensemble()
    
    result = dock_ligand(
        request.smiles,
        request.name,
        ensemble.pdb_paths,
        request.n_conformers
    )
    
    return result


@app.get("/api/fragments")
async def screen_fragments(n_conformers: int = 50, top_n: int = 10):
    """Screen fragment library"""
    from tools.ensemble import EnsembleProcessor
    from tools.docking import dock_fragment_library
    
    processor = EnsembleProcessor("./data")
    ensemble = processor.load_ensemble()
    
    return dock_fragment_library(ensemble.pdb_paths, n_conformers, top_n)


async def run_discovery_async(query: str):
    """Run discovery workflow with real-time updates"""
    from tools.ensemble import EnsembleProcessor
    from tools.docking import dock_fragment_library
    from tools.symbolic import discover_rules
    
    discovery_state["status"] = "running"
    discovery_state["results"] = {}
    
    try:
        processor = EnsembleProcessor("./data")
        
        # Step 1: Load and validate
        log_event("step_start", {"step": "validation"})
        discovery_state["current_step"] = "Validating ensemble..."
        await broadcast({"type": "step", "step": "validation"})
        
        ensemble = processor.load_ensemble()
        validation = processor.validate_ensemble(ensemble)
        discovery_state["results"]["validation"] = validation
        log_event("step_complete", {"step": "validation", "result": validation})
        await broadcast({"type": "validation", "data": validation})
        
        await asyncio.sleep(0.5)
        
        # Step 2: Select conformers
        log_event("step_start", {"step": "selection"})
        discovery_state["current_step"] = "Selecting conformers..."
        await broadcast({"type": "step", "step": "selection"})
        
        selected = processor.select_conformers(ensemble, rg_min=30.0, n_max=100)
        selection_stats = {
            "n_selected": len(selected.features),
            "n_total": len(ensemble.features),
            "rg_mean": float(selected.features['rg'].mean())
        }
        discovery_state["results"]["selection"] = selection_stats
        log_event("step_complete", {"step": "selection", "result": selection_stats})
        await broadcast({"type": "selection", "data": selection_stats})
        
        await asyncio.sleep(0.5)
        
        # Step 3: Screen fragments
        log_event("step_start", {"step": "screening"})
        discovery_state["current_step"] = "Screening fragments..."
        await broadcast({"type": "step", "step": "screening"})
        
        screening = dock_fragment_library(selected.pdb_paths, 30, 10)
        discovery_state["results"]["screening"] = screening
        log_event("step_complete", {"step": "screening", "result": screening})
        await broadcast({"type": "screening", "data": screening})
        
        await asyncio.sleep(0.5)
        
        # Step 4: Discover rules
        log_event("step_start", {"step": "rules"})
        discovery_state["current_step"] = "Discovering rules..."
        await broadcast({"type": "step", "step": "rules"})
        
        rules = discover_rules(
            screening.get('top_hits', []),
            ensemble.features,
            ['rg', 'aromatic_exposure']
        )
        discovery_state["results"]["rules"] = rules
        log_event("step_complete", {"step": "rules", "result": rules})
        await broadcast({"type": "rules", "data": rules})
        
        # Complete
        discovery_state["status"] = "complete"
        discovery_state["current_step"] = None
        await broadcast({
            "type": "complete",
            "results": discovery_state["results"]
        })
        
    except Exception as e:
        discovery_state["status"] = "error"
        discovery_state["current_step"] = None
        log_event("error", {"message": str(e)})
        await broadcast({"type": "error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)