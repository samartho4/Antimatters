# Async Patterns

## SSE Streaming

Backend streams events to frontend via Server-Sent Events.

```python
# api/agui_server.py
@app.post("/v1/run")
async def run_agent(input: RunAgentInput):
    return StreamingResponse(
        agent_runner.run_stream(input),
        media_type="text/event-stream"
    )
```

Event format:
```
data: {"type": "text_message_content", "content": "..."}

data: {"type": "tool_call_start", "tool_name": "dock_ensemble"}

data: {"type": "state_delta", "delta": {"latest_protocol_id": "..."}}
```

## ADK Runner

Agent execution uses `Runner.run_async`:

```python
from google.adk.runners import Runner

async for event in runner.run_async(user_id, session_id, new_message):
    if event.is_final_response():
        yield event.content.parts[0].text
```

## Parallel Docking

Engineer agent spawns concurrent ligand agents:

```python
# engineering/agent.py
def create_parallel_docking_agent(ligands):
    sub_agents = [create_ligand_agent(lig["name"], lig["smiles"]) for lig in ligands]
    return ParallelAgent(
        name="parallel_docking",
        sub_agents=sub_agents
    )
```

Each sub-agent:
1. Calls `prepare_ligand`
2. Calls `dock_ensemble`
3. Calls `update_ligand_result` to update ExperimentMatrix

Updates happen independently — matrix reflects real-time progress.

## Frontend Handling

```typescript
// aguiService.ts
async *stream(endpoint, body): AsyncGenerator<AGUIEvent> {
    const response = await fetch(endpoint, { method: 'POST', body: JSON.stringify(body) });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        for (const line of text.split('\n')) {
            if (line.startsWith('data: ')) {
                yield JSON.parse(line.slice(6));
            }
        }
    }
}
```

## Error Handling

Gemini 503 errors (API overload) are caught and retried:

```python
# agents/gemini_config.py
# Exponential backoff with jitter
```
