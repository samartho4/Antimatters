# Database

## SQLite Schema

**Location**: `data/antimatters.db`

```sql
-- Workspaces (project contexts)
CREATE TABLE workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    config TEXT DEFAULT '{}',     -- JSON: binding_site, target_protein
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Conversations (chat history)
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
);

-- Messages (full history replay)
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,           -- user | assistant | tool
    content TEXT,
    tool_calls TEXT,              -- JSON array
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Artifacts (Protocol, ExperimentMatrix, DiscoveryReport)
CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    workspace_id TEXT,
    type TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,        -- JSON
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Knowledge/Learnings (Evolution)
CREATE TABLE knowledge (
    id TEXT PRIMARY KEY,
    workspace_id TEXT,
    type TEXT NOT NULL,           -- insight | procedure | result | feedback
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_conversation_id TEXT,
    source_artifact_id TEXT,
    tags TEXT DEFAULT '[]',       -- JSON array
    embedding TEXT,               -- text-embedding-004 vector (JSON)
    created_at TEXT NOT NULL
);

-- Feedback/Comments
CREATE TABLE feedback (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    conversation_id TEXT,
    user_id TEXT DEFAULT 'scientist',
    comment TEXT NOT NULL,
    resolved INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(id)
);
```

## Default Workspace

Created on first run:

```python
INSERT INTO workspaces VALUES (
    'ws_core',
    'core',
    'Default Antimatters workspace for IDP ensemble docking research',
    '{"binding_site": [125, 133, 136], "target_protein": "alpha-synuclein"}',
    ...
);
```

## Usage

```python
from api.services import (
    WorkspaceService, 
    ConversationService, 
    ArtifactService,
    KnowledgeService
)

# Create workspace
ws = WorkspaceService()
workspace = ws.create("My Project", config={"target": "tau"})

# Create conversation
conv = ConversationService()
conversation = conv.create(workspace["id"], "Docking experiment")

# Store artifact
art = ArtifactService()
artifact = art.create(
    artifact_type="protocol",
    content={"target": "tau", "ligands": [...]},
    conversation_id=conversation["id"]
)
```
