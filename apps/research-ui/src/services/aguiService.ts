/**
 * AG-UI Protocol Service
 * ======================
 * Connects to the AG-UI compatible backend with:
 * - Real-time event streaming
 * - Artifact management
 * - Tool call visualization
 * - State management
 * - Annotation support
 */

import { Subject, Observable } from 'rxjs';

// AG-UI Backend URL (agui_server.py runs on 8002)
const AGUI_BASE = 'http://localhost:8002';

// =============================================================================
// AG-UI Event Types
// =============================================================================

export enum EventType {
  // Lifecycle
  RUN_STARTED = 'RUN_STARTED',
  RUN_FINISHED = 'RUN_FINISHED',
  RUN_ERROR = 'RUN_ERROR',
  STEP_STARTED = 'STEP_STARTED',
  STEP_FINISHED = 'STEP_FINISHED',

  // Text Messages
  TEXT_MESSAGE_START = 'TEXT_MESSAGE_START',
  TEXT_MESSAGE_CONTENT = 'TEXT_MESSAGE_CONTENT',
  TEXT_MESSAGE_END = 'TEXT_MESSAGE_END',

  // Tool Calls
  TOOL_CALL_START = 'TOOL_CALL_START',
  TOOL_CALL_ARGS = 'TOOL_CALL_ARGS',
  TOOL_CALL_END = 'TOOL_CALL_END',
  TOOL_CALL_RESULT = 'TOOL_CALL_RESULT',

  // State
  STATE_SNAPSHOT = 'STATE_SNAPSHOT',
  STATE_DELTA = 'STATE_DELTA',

  // Activity (Artifacts)
  ACTIVITY_SNAPSHOT = 'ACTIVITY_SNAPSHOT',
  ACTIVITY_DELTA = 'ACTIVITY_DELTA',

  // Custom
  CUSTOM = 'CUSTOM',
}

export interface AGUIEvent {
  type: EventType;
  [key: string]: any;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, any>;
}

export interface Tool {
  name: string;
  description: string;
  category?: string;
  parameters: Record<string, any>;
}

export interface AnnotationRegion {
  residue_ids: number[];
  atom_ids: number[];
  description?: string;
  screenshot_base64?: string;
}

export interface RunAgentInput {
  thread_id?: string;
  run_id?: string;
  messages: Message[];
  tools?: Tool[];
  state?: Record<string, any>;
  context?: any[];
  annotation?: AnnotationRegion;
}

// =============================================================================
// Artifact Types
// =============================================================================

export enum ArtifactType {
  TASK_LIST = 'task_list',
  STRUCTURE_3D = 'structure_3d',
  DOCKING_RESULT = 'docking_result',
  INTERACTION_MAP = 'interaction_map',
  CLUSTER_VISUALIZATION = 'cluster_visualization',
  IMPLEMENTATION_PLAN = 'implementation_plan',
  WALKTHROUGH = 'walkthrough',
  LITERATURE_RESULT = 'literature_result',
  SCIENTIFIC_EXPERIMENT = 'scientific_experiment',
  LIGAND_SVG = 'ligand_svg',
}

export interface Artifact {
  id: string;
  type: ArtifactType;
  run_id: string;
  created_at: string;
  content: any;
}

export interface TaskListContent {
  tasks: Array<{
    name: string;
    status: 'pending' | 'in_progress' | 'completed';
    tool: string;
  }>;
  completed: number;
  total: number;
}

export interface Structure3DContent {
  pdb_data: string;
  format: string;
  metadata: {
    ped_id?: string;
    n_conformations?: number;
    binding_site?: number[];
  };
  view_settings: {
    representation: string;
    color_scheme: string;
    highlight_residues: number[];
  };
}

export interface DockingResultContent {
  binding_scores: number[];
  ensemble_average?: number;
  best_pose?: any;
  visualization: {
    chart_type: string;
    threshold_lines: Array<{ value: number; label: string; color: string }>;
  };
}

// =============================================================================
// AG-UI Service Class
// =============================================================================

export class AGUIService {
  private eventSubject = new Subject<AGUIEvent>();
  private abortController: AbortController | null = null;

  // Observable for subscribing to events
  public events$: Observable<AGUIEvent> = this.eventSubject.asObservable();

  // Current state
  public state: Record<string, any> = {};
  public artifacts: Artifact[] = [];
  public toolCalls: ToolCall[] = [];
  public messages: Message[] = [];

  /**
   * Check if AG-UI backend is available
   */
  async checkHealth(): Promise<{ status: string; capabilities: any; agents: string[] }> {
    const response = await fetch(`${AGUI_BASE}/`);
    if (!response.ok) throw new Error('AG-UI backend not available');
    return response.json();
  }

  /**
   * Get available tools
   */
  async getTools(): Promise<Tool[]> {
    const response = await fetch(`${AGUI_BASE}/tools`);
    if (!response.ok) throw new Error('Failed to fetch tools');
    const data = await response.json();
    return data.tools;
  }

  /**
   * Get known ligands
   */
  async getLigands(): Promise<Array<{ name: string; smiles: string; description: string }>> {
    const response = await fetch(`${AGUI_BASE}/ligands`);
    if (!response.ok) throw new Error('Failed to fetch ligands');
    const data = await response.json();
    return data.ligands;
  }

  /**
   * Get example ensembles
   */
  async getEnsembles(): Promise<Array<{ id: string; name: string; conformations: number; description: string }>> {
    const response = await fetch(`${AGUI_BASE}/ensembles`);
    if (!response.ok) throw new Error('Failed to fetch ensembles');
    const data = await response.json();
    return data.ensembles;
  }

  /**
   * Get artifact by ID
   */
  async getArtifact(artifactId: string): Promise<Artifact> {
    const response = await fetch(`${AGUI_BASE}/artifacts/${artifactId}`);
    if (!response.ok) throw new Error('Artifact not found');
    return response.json();
  }

  /**
   * List all artifacts
   */
  async listArtifacts(runId?: string): Promise<Artifact[]> {
    const url = runId ? `${AGUI_BASE}/artifacts?run_id=${runId}` : `${AGUI_BASE}/artifacts`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to list artifacts');
    const data = await response.json();
    return data.artifacts;
  }

  /**
   * Send annotation to backend
   */
  async sendAnnotation(annotation: AnnotationRegion): Promise<any> {
    const response = await fetch(`${AGUI_BASE}/annotate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(annotation)
    });
    if (!response.ok) throw new Error('Failed to send annotation');
    return response.json();
  }

  /**
   * Run agent with AG-UI protocol (streaming)
   */
  async runAgent(input: RunAgentInput): Promise<void> {
    // Cancel any existing request
    this.abort();

    this.abortController = new AbortController();

    // Reset state for new run
    this.toolCalls = [];
    this.artifacts = [];

    const response = await fetch(`${AGUI_BASE}/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(input),
      signal: this.abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`AG-UI request failed: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No reader available');

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6)) as AGUIEvent;
              this.processEvent(event);
              this.eventSubject.next(event);
            } catch (e) {
              console.warn('Failed to parse AG-UI event:', line);
            }
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        throw e;
      }
    }
  }

  /**
   * Process incoming event and update local state
   */
  private processEvent(event: AGUIEvent): void {
    switch (event.type) {
      case EventType.STATE_SNAPSHOT:
        this.state = event.snapshot || {};
        break;

      case EventType.STATE_DELTA:
        // Apply JSON patch
        if (event.delta) {
          for (const op of event.delta) {
            if (op.op === 'add' || op.op === 'replace') {
              const path = op.path.replace(/^\//, '').split('/');
              let obj: any = this.state;
              for (let i = 0; i < path.length - 1; i++) {
                if (!obj[path[i]]) obj[path[i]] = {};
                obj = obj[path[i]];
              }
              obj[path[path.length - 1]] = op.value;
            }
          }
        }
        break;

      case EventType.TOOL_CALL_START:
        this.toolCalls.push({
          id: event.toolCallId,
          name: event.toolCallName,
          args: {}
        });
        break;

      case EventType.TOOL_CALL_ARGS:
        const call = this.toolCalls.find(tc => tc.id === event.toolCallId);
        if (call && event.delta) {
          try {
            call.args = JSON.parse(event.delta);
          } catch (e) {
            // Args may be streamed in chunks
          }
        }
        break;

      case EventType.CUSTOM:
        if (event.name === 'artifact_created' && event.value) {
          this.artifacts.push(event.value);
        }
        break;

      case EventType.ACTIVITY_SNAPSHOT:
        if (event.content) {
          const existingIdx = this.artifacts.findIndex(a => a.id === event.content.id);
          if (existingIdx >= 0) {
            this.artifacts[existingIdx] = event.content;
          } else {
            this.artifacts.push(event.content);
          }
        }
        break;

      case EventType.ACTIVITY_DELTA:
        // Apply JSON patch to artifact content (for task progress updates)
        if (event.messageId && event.patch) {
          // The messageId is "activity_{artifactId}"
          const targetId = event.messageId.replace('activity_', '');
          const artifact = this.artifacts.find(a => a.id === targetId);

          if (artifact) {
            for (const op of event.patch) {
              if (op.op === 'replace' && op.path && op.value !== undefined) {
                const pathParts = op.path.replace(/^\//, '').split('/');
                let target: any = artifact;
                for (let i = 0; i < pathParts.length - 1; i++) {
                  if (target[pathParts[i]]) {
                    target = target[pathParts[i]];
                  }
                }
                target[pathParts[pathParts.length - 1]] = op.value;
              }
            }
          }
        }
        break;
    }
  }

  /**
   * Abort current request
   */
  abort(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  /**
   * Helper: Create user message
   */
  createUserMessage(content: string): Message {
    return {
      id: crypto.randomUUID(),
      role: 'user',
      content
    };
  }

  /**
   * Restore session artifacts
   */
  async restoreSession(): Promise<void> {
    try {
      const artifacts = await this.listArtifacts();
      this.artifacts = artifacts;
      
      // Emit events for each artifact to update subscribers
      artifacts.forEach(artifact => {
        this.eventSubject.next({
          type: EventType.CUSTOM,
          name: 'artifact_created',
          value: artifact
        });
      });
    } catch (e) {
      console.warn('Failed to restore session:', e);
    }
  }
}

// Singleton instance
export const aguiService = new AGUIService();

// =============================================================================
// React Hooks
// =============================================================================

import { useState, useEffect, useCallback } from 'react';

/**
 * Hook for AG-UI connection status
 */
export function useAGUIStatus() {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [capabilities, setCapabilities] = useState<any>(null);

  useEffect(() => {
    aguiService.checkHealth()
      .then(health => {
        setStatus('online');
        setCapabilities(health.capabilities);
      })
      .catch(() => setStatus('offline'));
  }, []);

  return { status, capabilities };
}

/**
 * Hook for AG-UI event stream
 */
export function useAGUIEvents() {
  const [events, setEvents] = useState<AGUIEvent[]>([]);
  const [currentEvent, setCurrentEvent] = useState<AGUIEvent | null>(null);

  useEffect(() => {
    const subscription = aguiService.events$.subscribe(event => {
      setCurrentEvent(event);
      setEvents(prev => [...prev, event]);
    });

    return () => subscription.unsubscribe();
  }, []);

  const clearEvents = useCallback(() => setEvents([]), []);

  return { events, currentEvent, clearEvents };
}

/**
 * Hook for running agent
 */
export function useAGUIAgent() {
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [state, setState] = useState<Record<string, any>>(aguiService.state);
  const [artifacts, setArtifacts] = useState<Artifact[]>(aguiService.artifacts);
  const [toolCalls, setToolCalls] = useState<ToolCall[]>(aguiService.toolCalls);
  const [textContent, setTextContent] = useState('');

  useEffect(() => {
    const subscription = aguiService.events$.subscribe(event => {
      setState({ ...aguiService.state });
      setArtifacts([...aguiService.artifacts]);
      setToolCalls([...aguiService.toolCalls]);

      if (event.type === EventType.TEXT_MESSAGE_CONTENT) {
        setTextContent(prev => prev + (event.delta || ''));
      }

      if (event.type === EventType.RUN_FINISHED || event.type === EventType.RUN_ERROR) {
        setIsRunning(false);
      }

      if (event.type === EventType.RUN_ERROR) {
        setError(new Error(event.message));
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const runAgent = useCallback(async (message: string, annotation?: AnnotationRegion) => {
    setIsRunning(true);
    setError(null);
    setTextContent('');

    try {
      await aguiService.runAgent({
        messages: [aguiService.createUserMessage(message)],
        annotation
      });
    } catch (e) {
      setError(e as Error);
      setIsRunning(false);
    }
  }, []);

  const abort = useCallback(() => {
    aguiService.abort();
    setIsRunning(false);
  }, []);

  return {
    isRunning,
    error,
    state,
    artifacts,
    toolCalls,
    textContent,
    runAgent,
    abort
  };
}
