# API Reference

## Run Agent

```http
POST /v1/run
Content-Type: application/json

{
  "thread_id": "conv_abc123",
  "run_id": "run_xyz789",
  "messages": [
    {"role": "user", "content": "Dock Fasudil to alpha-synuclein"}
  ]
}
```

**Response**: SSE stream

## Workspaces

```http
GET /workspaces
GET /workspaces/{workspace_id}
POST /workspaces
PATCH /workspaces/{workspace_id}
DELETE /workspaces/{workspace_id}
```

## Conversations

```http
GET /conversations?workspace_id={id}
GET /conversations/{conversation_id}
POST /conversations
```

## Artifacts

```http
GET /artifacts/{artifact_id}
GET /artifacts?conversation_id={id}
POST /artifacts
```

## Knowledge

```http
GET /knowledge?workspace_id={id}
POST /knowledge
POST /knowledge/search
```

**Search Request**:
```json
{
  "query": "aromatic interactions",
  "limit": 5,
  "min_similarity": 0.7
}
```
