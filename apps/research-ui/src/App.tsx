/**
 * Antimatters - Molecular Research, Simplified
 * Information -> Computation -> Evolution
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  aguiService,
  useAGUIStatus,
  useAGUIEvents,
  useAGUIAgent,
  EventType,
  Artifact,
  ArtifactType,
  Tool,
  ResearchMode,
} from './services/aguiService';
import { Sidebar } from './components/Sidebar';
import { Computation } from './components/Computation';
import { ArtifactPanel } from './components/ArtifactPanel';
import { EvolutionView } from './components/EvolutionView';
import { CopilotKit, useCopilotReadable } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

// Backend base URL
const API_BASE = 'http://localhost:8002';

// =============================================================================
// Types for backend data
// =============================================================================

interface Workspace {
  id: string;
  name: string;
  description: string;
  config: Record<string, any>;
  stats?: { conversations: number; artifacts: number; knowledge_items: number };
  created_at: string;
  updated_at: string;
}

interface Conversation {
  id: string;
  workspace_id: string;
  title: string;
  status: string;
  message_count?: number;
  messages?: Array<{
    id: string;
    role: string;
    content: string;
    tool_calls?: any[];
    metadata?: Record<string, any>;
    created_at: string;
  }>;
  created_at: string;
  updated_at: string;
}

interface KnowledgeItem {
  id: string;
  workspace_id: string;
  type: string;
  title: string;
  content: string;
  tags: string[];
  created_at: string;
}

// =============================================================================
// Error Boundary
// =============================================================================

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('React Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen w-full bg-slate-900 flex items-center justify-center">
          <div className="max-w-md text-center p-6">
            <div className="text-red-400 text-lg font-semibold mb-2">Something went wrong</div>
            <div className="text-slate-400 text-sm mb-4">
              {this.state.error?.message || 'An unexpected error occurred'}
            </div>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// =============================================================================
// App
// =============================================================================

export default function App() {
  // Get API key from environment variable (fallback for dev)
  const copilotApiKey = import.meta.env.VITE_COPILOT_API_KEY || '';

  return (
    <ErrorBoundary>
      <CopilotKit publicApiKey="ck_pub_7aa7b776ea278992ea22100dd49f8b7b">
        <ResearchEngine />
      </CopilotKit>
    </ErrorBoundary>
  );
}

function ResearchEngine() {
  const { status } = useAGUIStatus();
  const { events, currentEvent, clearEvents } = useAGUIEvents();
  const { isRunning, state, artifacts, toolCalls, textContent, runAgent, abort } = useAGUIAgent();

  // UI State — messages now include tool_calls per-message (not global)
  // Also supports screenshot_base64 for annotated messages
  const [messages, setMessages] = useState<Array<{
    id: string;
    role: string;
    content: string;
    screenshot_base64?: string;  // For annotated messages
    tool_calls?: Array<{ id: string; name: string; args: Record<string, any>; startTime?: number; endTime?: number; duration?: number }>;
  }>>([]);
  const [pdbData, setPdbData] = useState<string | null>(null);
  const [currentTool, setCurrentTool] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [tools, setTools] = useState<Tool[]>([]);
  const [dismissedArtifacts, setDismissedArtifacts] = useState<Set<string>>(new Set());

  // Filter out dismissed artifacts
  const visibleArtifacts = artifacts.filter(a => !dismissedArtifacts.has(a.id));

  // Make artifacts readable by the Copilot
  useCopilotReadable({
    description: "The complete list of scientific artifacts generated in previous experiments.",
    value: visibleArtifacts
  });

  // Track current assistant message ID for associating tool calls
  const [currentAssistantMsgId, setCurrentAssistantMsgId] = useState<string | null>(null);

  // Associate incoming tool calls with the current assistant message
  useEffect(() => {
    if (toolCalls.length === 0 || !currentAssistantMsgId) return;

    // ===== VERIFICATION LOGGING: Tool merge useEffect =====
    console.log('[App.tsx] [VERIFY] Tool merge useEffect: toolCalls.length=', toolCalls.length, ', currentAssistantMsgId=', currentAssistantMsgId);
    console.log('[App.tsx] [VERIFY] Tool merge useEffect: toolCall IDs=', toolCalls.map(tc => tc.id).join(', '));

    setMessages(prev => {
      const targetMsg = prev.find(m => m.id === currentAssistantMsgId);
      console.log('[App.tsx] [VERIFY] Tool merge: Target message found=', !!targetMsg, ', existing tool_calls=', (targetMsg?.tool_calls || []).length);

      return prev.map(msg => {
        if (msg.id === currentAssistantMsgId) {
          // Merge new tool calls, avoiding duplicates
          const existingIds = new Set((msg.tool_calls || []).map(tc => tc.id));
          const newCalls = toolCalls.filter(tc => !existingIds.has(tc.id));
          if (newCalls.length === 0) {
            console.log('[App.tsx] [VERIFY] Tool merge: No new calls to add (all already exist)');
            return msg;
          }
          console.log('[App.tsx] [VERIFY] Tool merge: Adding', newCalls.length, 'new tools to message', msg.id);
          return {
            ...msg,
            tool_calls: [...(msg.tool_calls || []), ...newCalls]
          };
        }
        return msg;
      });
    });
  }, [toolCalls, currentAssistantMsgId]);

  // Real data from backend
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string>('ws_core');
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | undefined>();
  const [knowledge, setKnowledge] = useState<KnowledgeItem[]>([]);
  const [showExperiment, setShowExperiment] = useState(false);
  const [showEvolution, setShowEvolution] = useState(false);

  // Annotation state
  const [isAnnotating, setIsAnnotating] = useState(false);

  // Resizable right panel
  const [panelWidth, setPanelWidth] = useState(400);
  const isDragging = useRef(false);

  // Track when a new run starts to prevent race condition where tools from run N+1
  // get merged into run N's assistant message (due to async state updates)
  const newRunStartedRef = useRef(false);

  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    const startX = e.clientX;
    const startWidth = panelWidth;

    const onMouseMove = (ev: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = startX - ev.clientX; // drag left → wider panel
      setPanelWidth(Math.max(280, Math.min(600, startWidth + delta)));
    };
    const onMouseUp = () => {
      isDragging.current = false;
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  }, [panelWidth]);

  // Fetch workspaces on mount
  useEffect(() => {
    const abortController = new AbortController();

    const fetchWorkspaces = async () => {
      try {
        const response = await fetch(`${API_BASE}/workspaces`, {
          signal: abortController.signal
        });
        if (response.ok) {
          const data = await response.json();
          setWorkspaces(data.workspaces || []);
          // Set active workspace to first one (usually ws_core)
          if (data.workspaces?.length > 0 && !activeWorkspaceId) {
            setActiveWorkspaceId(data.workspaces[0].id);
          }
        }
      } catch (e) {
        // Backend not available yet or request aborted
        if ((e as Error).name !== 'AbortError') {
          console.warn('Failed to fetch workspaces:', e);
        }
      }
    };
    fetchWorkspaces();

    return () => abortController.abort();
  }, []);

  // Fetch conversations when workspace changes
  useEffect(() => {
    if (!activeWorkspaceId) return;

    const abortController = new AbortController();

    const fetchConversations = async () => {
      try {
        const response = await fetch(`${API_BASE}/conversations?workspace_id=${activeWorkspaceId}`, {
          signal: abortController.signal
        });
        if (response.ok) {
          const data = await response.json();
          setConversations(data.conversations || []);
        }
      } catch (e) {
        // Backend not available yet or request aborted
        if ((e as Error).name !== 'AbortError') {
          console.warn('Failed to fetch conversations:', e);
        }
      }
    };
    fetchConversations();

    return () => abortController.abort();
  }, [activeWorkspaceId]);

  // Fetch knowledge (Evolution) when workspace changes
  useEffect(() => {
    if (!activeWorkspaceId) return;

    const abortController = new AbortController();

    const fetchKnowledge = async () => {
      try {
        const response = await fetch(`${API_BASE}/knowledge?workspace_id=${activeWorkspaceId}`, {
          signal: abortController.signal
        });
        if (response.ok) {
          const data = await response.json();
          setKnowledge(data.knowledge || []);
        }
      } catch (e) {
        if ((e as Error).name !== 'AbortError') {
          console.log('Knowledge endpoint not available');
        }
      }
    };
    fetchKnowledge();

    return () => abortController.abort();
  }, [activeWorkspaceId]);

  // Load artifacts when conversation changes
  useEffect(() => {
    const loadArtifacts = async () => {
      if (activeConversationId) {
        try {
          await aguiService.restoreSession(activeConversationId);
        } catch (e) {
          console.warn('Failed to load artifacts for conversation:', e);
        }
      } else {
        // No active conversation - clear artifacts
        aguiService.artifacts = [];
      }
    };
    loadArtifacts();
  }, [activeConversationId]);

  // Fetch tools on mount
  useEffect(() => {
    const abortController = new AbortController();

    const fetchTools = async () => {
      try {
        const response = await fetch(`${API_BASE}/tools`, {
          signal: abortController.signal
        });
        if (response.ok) {
          const data = await response.json();
          setTools(data.tools || []);
        }
      } catch (e) {
        if ((e as Error).name !== 'AbortError') {
          console.log('Tools endpoint not available');
        }
      }
    };
    fetchTools();

    return () => abortController.abort();
  }, []);

  // Handle AG-UI events
  useEffect(() => {
    if (!currentEvent) return;

    switch (currentEvent.type) {
      case EventType.RUN_STARTED:
        clearEvents();
        setShowExperiment(true);
        // Mark that a new run has started (ref is synchronous, unlike state)
        // This prevents tools from the new run being merged to old messages
        newRunStartedRef.current = true;
        // ===== VERIFICATION LOGGING: RUN_STARTED =====
        console.log('[App.tsx] [VERIFY] RUN_STARTED: newRunStartedRef.current = TRUE, currentAssistantMsgId will be set to NULL');
        console.log('[App.tsx] [VERIFY] RUN_STARTED: This prevents tools from new run being merged to old messages');
        // Don't create message yet - wait for TOOL_CALL_START or TEXT_MESSAGE_START
        setCurrentAssistantMsgId(null);
        break;
      case EventType.TEXT_MESSAGE_START:
        // New message started, clear the new-run flag
        console.log('[App.tsx] [VERIFY] TEXT_MESSAGE_START: Clearing newRunStartedRef (was', newRunStartedRef.current, '), messageId=', currentEvent.messageId);
        newRunStartedRef.current = false;
        // If we already have an assistant message for this run (created on RUN_STARTED),
        // update its ID to match the backend's message ID. Otherwise create a new one.
        setMessages(prev => {
          // Find the last empty assistant message (our placeholder from RUN_STARTED)
          let lastAssistantIdx = -1;
          for (let i = prev.length - 1; i >= 0; i--) {
            if (prev[i].role === 'assistant' && !prev[i].content) {
              lastAssistantIdx = i;
              break;
            }
          }
          if (lastAssistantIdx >= 0) {
            // Update the placeholder message with the real ID
            return prev.map((m, i) =>
              i === lastAssistantIdx
                ? { ...m, id: currentEvent.messageId }
                : m
            );
          }
          // No placeholder exists, create new message
          return [...prev, {
            id: currentEvent.messageId,
            role: 'assistant',
            content: '',
            tool_calls: [],
          }];
        });
        setCurrentAssistantMsgId(currentEvent.messageId);
        break;
      case EventType.TEXT_MESSAGE_CONTENT:
        setMessages(prev => {
          // First, try to find the message with exact ID match
          const idxById = prev.findIndex(m => m.id === currentEvent.messageId);
          if (idxById >= 0) {
            return prev.map((m, i) =>
              i === idxById
                ? { ...m, content: (m.content || '') + (currentEvent.delta || '') }
                : m
            );
          }
          // Fallback: update the last assistant message (defensive for race conditions)
          for (let i = prev.length - 1; i >= 0; i--) {
            if (prev[i].role === 'assistant') {
              return prev.map((m, j) =>
                j === i
                  ? { ...m, content: (m.content || '') + (currentEvent.delta || '') }
                  : m
              );
            }
          }
          // No assistant message found - shouldn't happen, but log for debugging
          console.warn('[TEXT_MESSAGE_CONTENT] No matching message found for ID:', currentEvent.messageId);
          return prev;
        });
        break;
      case EventType.TOOL_CALL_START:
        setCurrentTool(currentEvent.toolCallName || 'Processing');
        // If a new run started OR no assistant message exists, create a new message
        // The ref check handles the race condition where currentAssistantMsgId
        // hasn't been reset to null yet (due to async state batching)

        // ===== VERIFICATION LOGGING: TOOL_CALL_START =====
        console.log('[App.tsx] [VERIFY] TOOL_CALL_START: toolCallName=', currentEvent.toolCallName);
        console.log('[App.tsx] [VERIFY] TOOL_CALL_START: newRunStartedRef.current=', newRunStartedRef.current, ', currentAssistantMsgId=', currentAssistantMsgId);

        if (newRunStartedRef.current || !currentAssistantMsgId) {
          console.log('[App.tsx] [VERIFY] TOOL_CALL_START: ✓ Creating NEW assistant message (ref=', newRunStartedRef.current, ', msgId=', currentAssistantMsgId, ')');
          newRunStartedRef.current = false; // Reset after creating message for this run
          const toolMsgId = `assistant-${Date.now()}`;
          console.log('[App.tsx] [VERIFY] TOOL_CALL_START: New message ID =', toolMsgId);
          setCurrentAssistantMsgId(toolMsgId);
          setMessages(prev => [...prev, {
            id: toolMsgId,
            role: 'assistant',
            content: '',
            tool_calls: [],
          }]);
        } else {
          console.log('[App.tsx] [VERIFY] TOOL_CALL_START: Using existing message ID =', currentAssistantMsgId);
        }
        break;
      case EventType.TOOL_CALL_END:
        setCurrentTool(null);
        break;
      case EventType.CUSTOM:
        if (currentEvent.name === 'artifact_created') {
          const artifact = currentEvent.value as Artifact;
          if (artifact.type === ArtifactType.STRUCTURE_3D && artifact.content?.pdb_data) {
            setPdbData(artifact.content.pdb_data);
          } else if (artifact.type === 'ligand_svg') {
            setMessages(prev => [...prev, {
              id: `mol-${Date.now()}`,
              role: 'assistant',
              content: `__MOLECULE_CARD__:${JSON.stringify(artifact.content)}`
            }]);
          }
        }
        break;
      case EventType.RUN_FINISHED:
        console.log('[App.tsx] [VERIFY] RUN_FINISHED: Clearing currentAssistantMsgId (was', currentAssistantMsgId, ')');
        setCurrentTool(null);
        setCurrentAssistantMsgId(null);  // Run complete, clear tracking
        break;
    }
  }, [currentEvent, clearEvents]);

  // Send message with mode routing and optional annotation
  // FIX 2: Pass conversation history to backend (following agent-chat-ui pattern)
  const handleSend = useCallback(async (message: string, mode?: ResearchMode) => {
    if (!message.trim() || status !== 'online') return;

    const userMsg = { id: `u-${Date.now()}`, role: 'user', content: message };

    // Capture current messages BEFORE adding new one (for history context)
    const currentHistory = [...messages];

    setMessages(prev => [...prev, userMsg]);

    // Create conversation on backend if new
    if (!activeConversationId) {
      try {
        const response = await fetch(`${API_BASE}/conversations`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            workspace_id: activeWorkspaceId,
            title: message.slice(0, 50) + (message.length > 50 ? '...' : ''),
            initial_message: message
          })
        });
        if (response.ok) {
          const newConv = await response.json();
          setConversations(prev => [newConv, ...prev]);
          setActiveConversationId(newConv.id);
        }
      } catch (e) {
        console.error('Failed to create conversation:', e);
      }
    }

    try {
      // Pass conversation history for context continuity
      // Agent can now understand "the above result", "its spatial interaction", etc.
      await runAgent(message, mode, undefined, currentHistory);
    } catch (e) {
      console.error(e);
    }
  }, [status, runAgent, activeConversationId, activeWorkspaceId, messages]);

  // Annotation handlers
  const handleStartAnnotation = useCallback(() => {
    setIsAnnotating(true);
  }, []);

  // Unified: annotation drawing + question → multimodal send to Gemini
  // Always use planning mode for annotations (Gemini 3 visual analysis)
  // Now includes screenshot in the message so user can see what was captured
  // FIX 2: Also pass conversation history for context continuity
  const handleAnnotationSend = useCallback((annotation: any, question: string) => {
    // Capture current messages BEFORE adding new one (for history context)
    const currentHistory = [...messages];

    setMessages(prev => [...prev, {
      id: `u-${Date.now()}`,
      role: 'user',
      content: question,
      screenshot_base64: annotation?.screenshot_base64 || undefined,  // Show the captured screenshot
    }]);
    runAgent(question, 'planning', annotation, currentHistory);  // Pass history for context
    setIsAnnotating(false);
  }, [runAgent, messages]);

  const handleCancelAnnotation = useCallback(() => {
    setIsAnnotating(false);
  }, []);

  // Dismiss artifact (hide from view)
  const handleDismissArtifact = useCallback((artifactId: string) => {
    setDismissedArtifacts(prev => new Set(prev).add(artifactId));
  }, []);

  // New chat - clear all state for fresh session
  const handleNewChat = () => {
    setMessages([]);
    setCurrentAssistantMsgId(null);
    setDismissedArtifacts(new Set());
    setActiveConversationId(undefined);
    aguiService.clearArtifacts(); // Properly clear artifacts and notify React
    setShowExperiment(false);
    setPdbData(null);
  };

  // Select conversation
  const handleSelectConversation = (id: string) => {
    setActiveConversationId(id);
  };

  // Select tool (show info)
  const handleSelectTool = (tool: Tool) => {
    console.log('Selected tool:', tool);
  };

  // Select artifact
  const handleArtifactClick = (artifact: Artifact) => {
    if (artifact.type === 'structure_3d') {
      setPdbData(artifact.content.pdb_data);
    }
    setShowExperiment(true);
  };

  return (
    <div className="h-screen w-full bg-am-primary flex overflow-hidden">
      {/* Skip Navigation Link */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:top-4 focus:left-4 focus:px-4 focus:py-2 focus:bg-am-accent focus:text-white focus:rounded-lg focus:outline-none focus:ring-2 focus:ring-am-accent"
      >
        Skip to main content
      </a>

      {/* Evolution View (Full Screen) */}
      {showEvolution ? (
        <EvolutionView
          artifacts={artifacts}
          conversations={conversations}
          onClose={() => setShowEvolution(false)}
        />
      ) : (
        <>
          {/* Sidebar */}
          <Sidebar
            tools={tools}
            artifacts={visibleArtifacts}
            conversations={conversations}
            knowledge={knowledge}
            onNewChat={handleNewChat}
            onSelectConversation={handleSelectConversation}
            onSelectTool={handleSelectTool}
            onSelectArtifact={handleArtifactClick}
            onSelectKnowledge={() => setShowEvolution(true)}
            activeConversationId={activeConversationId}
            isCollapsed={sidebarCollapsed}
            onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          />

          {/* Main Computation Area */}
          <div id="main-content" className="flex-1 flex flex-col lg:flex-row overflow-hidden">
            <Computation
              messages={messages}
              artifacts={visibleArtifacts}
              isRunning={isRunning}
              currentTool={currentTool}
              textContent={textContent}
              onSend={handleSend}
              onAbort={abort}
              onArtifactClick={handleArtifactClick}
              onStructureClick={(pdb) => setPdbData(pdb)}
              status={status}
              workspaces={workspaces}
              activeWorkspaceId={activeWorkspaceId}
              onWorkspaceChange={setActiveWorkspaceId}
              onStartAnnotation={handleStartAnnotation}
            />

            {/* Drag handle + Artifact Panel (Right Side) - only show when there are visible artifacts */}
            {visibleArtifacts.length > 0 && (
              <>
                {/* Resize grip — desktop only */}
                <div
                  className="hidden lg:block flex-shrink-0 w-1.5 cursor-col-resize hover:bg-am-accent/30 bg-am-border/40 transition-colors"
                  onMouseDown={handleDragStart}
                  aria-label="Resize artifact panel"
                />
                <aside
                  className="flex-shrink-0 bg-am-primary border-t lg:border-t-0 lg:border-l border-am-border flex flex-col"
                  aria-label="Artifact viewer"
                  style={{ width: panelWidth }}
                >
                  <ArtifactPanel
                    artifacts={visibleArtifacts}
                    onStructureClick={(pdb) => setPdbData(pdb)}
                    onClose={handleDismissArtifact}
                    isAnnotating={isAnnotating}
                    onStartAnnotation={handleStartAnnotation}
                    onAnnotationSend={handleAnnotationSend}
                    onCancelAnnotation={handleCancelAnnotation}
                  />
                </aside>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
