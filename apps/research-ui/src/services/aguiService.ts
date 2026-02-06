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
  startTime?: number;  // timestamp when tool started
  endTime?: number;    // timestamp when tool finished
  duration?: number;   // calculated duration in ms
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

export type ResearchMode = 'planning' | 'serendipitize';

export interface RunAgentInput {
  thread_id?: string;
  run_id?: string;
  messages: Message[];
  tools?: Tool[];
  state?: Record<string, any>;
  context?: any[];
  annotation?: AnnotationRegion;
  mode?: ResearchMode;  // Routes to planning_agent or docking_workflow
}

// =============================================================================
// Artifact Types
// =============================================================================

export enum ArtifactType {
  // Research Phase
  TASK_LIST = 'task_list',
  PROTOCOL = 'protocol',
  LITERATURE_RESULT = 'literature_result',

  // Engineering Phase
  STRUCTURE_3D = 'structure_3d',
  EXPERIMENT_MATRIX = 'experiment_matrix',
  DOCKING_RESULT = 'docking_result',
  INTERACTION_MAP = 'interaction_map',
  CLUSTER_VISUALIZATION = 'cluster_visualization',
  LIGAND_SVG = 'ligand_svg',

  // Evolution Phase
  DISCOVERY_REPORT = 'discovery_report',
  EVOLUTION_TRACE = 'evolution_trace',
  MOLECULE_SUGGESTIONS = 'molecule_suggestions',
  PUBLICATION_FIGURE = 'publication_figure',

  // System
  IMPLEMENTATION_PLAN = 'implementation_plan',
  WALKTHROUGH = 'walkthrough',
  SCIENTIFIC_EXPERIMENT = 'scientific_experiment',
}

export interface Artifact {
  id: string;
  type: ArtifactType;
  name?: string;
  run_id: string;
  created_at: string;
  content: any;
}

export interface TaskListContent {
  title?: string;  // Dynamic title for the task list
  tasks: Array<{
    name?: string;
    step?: string;  // Alternative to name (from research agent)
    status: 'pending' | 'in_progress' | 'completed';
    tool?: string;
    details?: string;
    result?: string;
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

export interface ProtocolContent {
  ped_id: string;
  protein_name: string;
  protein_pdb_path?: string;
  ensemble_id: string;
  n_conformations: number;
  n_residues?: number;
  binding_site_residues: number[];
  ligands: Array<{
    name: string;
    smiles: string;
    chembl_id?: string;
    image_base64?: string;
    image_url?: string;
  }>;
  visuals: {
    ped_viewer_url: string;
    ped_api_image?: string;
  };
  figure_urls?: string[];
  literature_results?: Array<{ title: string; pmcid?: string }>;
  validation_status: 'pending' | 'validated' | 'failed';
  confidence_score: number;
}

export interface ExperimentMatrixContent {
  protocol_id: string;
  ligands: Array<{
    name: string;
    smiles: string;
    status: 'queued' | 'running' | 'completed' | 'failed';
    best_energy?: number;
    best_cluster?: number;
    interaction_types?: string[];
  }>;
  n_clusters: number;
  representative_frames: number[];
}

export interface DiscoveryReportContent {
  title: string;
  executive_summary: string;
  energy_stats: {
    n_completed: number;
    best_energy: number;
    n_drug_like: number;
  };
  ucb_rankings: Array<{
    ligand_name: string;
    best_energy: number;
    ucb_score: number;
    drug_like: boolean;
  }>;
  sar_insights: Array<{
    pattern_type: string;
    title: string;
    description: string;
    confidence: number;
    recommendation: string;
  }>;
  recommendations: Array<{
    type: string;
    priority: 'high' | 'medium' | 'low';
    recommendation: string;
  }>;
}

export interface PublicationFigureContent {
  title: string;
  figure_type: 'interaction_map' | 'binding_energy' | 'cluster_analysis' | 'sar_plot';
  image_base64: string;
  caption: string;
  annotations?: Array<{
    x: number;
    y: number;
    text: string;
    arrow?: boolean;
  }>;
  data?: any;
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

  // Streaming state tracking for real-time UI feedback
  public isStreaming: boolean = false;
  public firstTokenReceived: boolean = false;
  public streamStartTime: number | null = null;
  public currentToolName: string | null = null;

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
    const data = await response.json();
    return data.artifact || data;
  }

  /**
   * Refresh current artifacts from backend to pick up updates
   * (e.g., task statuses that were updated on disk during the run)
   */
  async refreshCurrentArtifacts(): Promise<void> {
    if (this.artifacts.length === 0) return;
    try {
      const refreshed = await Promise.all(
        this.artifacts.map(async (a) => {
          try {
            return await this.getArtifact(a.id);
          } catch {
            return a;
          }
        })
      );
      this.artifacts = refreshed;
    } catch (e) {
      console.warn('Artifact refresh failed:', e);
    }
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
   * List artifacts by conversation ID
   */
  async listArtifactsByConversation(conversationId: string): Promise<Artifact[]> {
    const response = await fetch(`${AGUI_BASE}/artifacts/by-conversation/${conversationId}`);
    if (!response.ok) throw new Error('Failed to list artifacts for conversation');
    const data = await response.json();
    return data.artifacts || [];
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

    // Reset state for new run (following RxJS best practice: emit event for state changes)
    this.toolCalls = [];
    // DON'T clear artifacts - they should accumulate across runs (Q5 bonus: allArtifacts pattern)
    // this.artifacts = [];

    // Emit RUN_STARTED immediately so React subscribers sync before backend events arrive
    // This prevents Simulation Campaign box from showing stale data (Q5: chat robustness)
    this.eventSubject.next({
      type: EventType.RUN_STARTED,
      timestamp: Date.now()
    });

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
    // Track streaming state based on event type
    switch (event.type) {
      case EventType.RUN_STARTED:
        this.isStreaming = true;
        this.firstTokenReceived = false;
        this.streamStartTime = Date.now();
        this.currentToolName = null;
        break;

      case EventType.TEXT_MESSAGE_START:
        this.isStreaming = true;
        break;

      case EventType.TEXT_MESSAGE_CONTENT:
        if (!this.firstTokenReceived) {
          this.firstTokenReceived = true;
        }
        break;

      case EventType.TEXT_MESSAGE_END:
      case EventType.RUN_FINISHED:
      case EventType.RUN_ERROR:
        this.isStreaming = false;
        this.currentToolName = null;
        break;

      case EventType.TOOL_CALL_START:
        this.currentToolName = event.toolCallName || null;
        break;

      case EventType.TOOL_CALL_END:
        this.currentToolName = null;
        break;
    }

    // Process event data
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
          args: {},
          startTime: Date.now()
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

      case EventType.TOOL_CALL_END:
        const endCall = this.toolCalls.find(tc => tc.id === event.toolCallId);
        if (endCall && endCall.startTime) {
          endCall.endTime = Date.now();
          endCall.duration = endCall.endTime - endCall.startTime;
        }
        break;

      case EventType.CUSTOM:
        if (event.name === 'artifact_created' && event.value) {
          // Deduplicate by ID
          const existingIdx = this.artifacts.findIndex(a => a.id === event.value.id);
          if (existingIdx >= 0) {
            this.artifacts[existingIdx] = event.value;
          } else {
            this.artifacts.push(event.value);
          }
        }
        break;

      case EventType.ACTIVITY_SNAPSHOT:
        if (event.content && event.content.id) {
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
            // CRITICAL: Create new array reference so React detects the change
            this.artifacts = [...this.artifacts];
            // Emit event to notify React subscribers
            this.eventSubject.next(event);
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
   * Restore session artifacts (optionally filtered by conversation)
   */
  async restoreSession(conversationId?: string): Promise<void> {
    try {
      // Don't load all artifacts by default - only load for specific conversation
      if (conversationId) {
        const artifacts = await this.listArtifactsByConversation(conversationId);
        this.artifacts = artifacts;
      } else {
        // Start with empty artifacts - they'll be loaded when conversation is selected
        this.artifacts = [];
      }
    } catch (e) {
      console.warn('Failed to restore session:', e);
      this.artifacts = [];
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

  // Streaming state for real-time UI feedback
  const [isStreaming, setIsStreaming] = useState(false);
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const [currentToolName, setCurrentToolName] = useState<string | null>(null);

  useEffect(() => {
    const subscription = aguiService.events$.subscribe(event => {
      setState({ ...aguiService.state });
      setArtifacts([...aguiService.artifacts]);
      setToolCalls([...aguiService.toolCalls]);

      // Update streaming state from service
      setIsStreaming(aguiService.isStreaming);
      setFirstTokenReceived(aguiService.firstTokenReceived);
      setCurrentToolName(aguiService.currentToolName);

      if (event.type === EventType.TEXT_MESSAGE_CONTENT) {
        setTextContent(prev => prev + (event.delta || ''));
      }

      if (event.type === EventType.RUN_FINISHED) {
        setIsRunning(false);
        setIsStreaming(false);
        // Refresh artifacts from backend — picks up task status updates
        // that happened on disk during the run but weren't streamed as events
        aguiService.refreshCurrentArtifacts().then(() => {
          setArtifacts([...aguiService.artifacts]);
        });
      }

      if (event.type === EventType.RUN_ERROR) {
        setIsRunning(false);
        setIsStreaming(false);
        setError(new Error(event.message));
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const runAgent = useCallback(async (message: string, mode?: ResearchMode, annotation?: AnnotationRegion) => {
    setIsRunning(true);
    setError(null);
    setTextContent('');

    try {
      await aguiService.runAgent({
        messages: [aguiService.createUserMessage(message)],
        mode,
        annotation
      });
    } catch (e) {
      console.error('Agent run error:', e);
      setError(e as Error);
    } finally {
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
    abort,
    // Streaming state for real-time UI feedback
    isStreaming,
    firstTokenReceived,
    currentToolName
  };
}
