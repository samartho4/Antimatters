# Backend API

## Overview

FastAPI server implementing the **AG-UI protocol** for real-time agent communication.

**Location**: `api/agui_server.py`

## Endpoints

### `POST /v1/run`

Run agent with SSE streaming.

```python
@app.post("/v1/run")
async def run_agent(input: RunAgentInput):
    return StreamingResponse(
        agent_runner.run_stream(input),
        media_type="text/event-stream"
    )
```

**Request**:
```json
{
  "thread_id": "conv_abc123",
  "run_id": "run_xyz789",
  "messages": [
    {"role": "user", "content": "Dock Fasudil to PED00006"}
  ]
}
```

### `GET /workspaces`

List all workspaces.

### `GET /workspaces/{id}`

Get workspace details.

### `POST /workspaces`

Create workspace.

### `GET /conversations`

List conversations for workspace.

### `POST /conversations`

Create conversation.

### `GET /artifacts/{id}`

Get artifact by ID.

## Services

```python
# api/services.py

class WorkspaceService:
    def create(name, description, config) -> Dict
    def get(workspace_id) -> Optional[Dict]
    def list_all() -> List[Dict]

class ConversationService:
    def create(workspace_id, title) -> Dict
    def add_message(conversation_id, role, content) -> Dict
    def list_by_workspace(workspace_id) -> List[Dict]

class ArtifactService:
    def create(artifact_type, content, metadata) -> Dict
    def get(artifact_id) -> Optional[Dict]
    def search(query) -> List[Dict]

class KnowledgeService:
    def create(knowledge_type, title, content) -> Dict
    def semantic_search(query, limit) -> List[Dict]
```

## ADK Integration

```python
from google.adk.runners import Runner

class ADKAgentRunner:
    def __init__(self):
        self.runner = None
        
    async def run_stream(self, input: RunAgentInput):
        async for event in self.runner.run_async(...):
            yield self.encoder.encode(translate_event(event))
```

## Running

```bash
cd core
uvicorn api.agui_server:app --host 0.0.0.0 --port 8001 --reload
```
