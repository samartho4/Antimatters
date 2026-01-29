/**
 * Antimatters - Molecular Research, Simplified
 * Information -> Computation -> Evolution
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  aguiService,
  useAGUIStatus,
  useAGUIEvents,
  useAGUIAgent,
  EventType,
  Artifact,
  ArtifactType,
  Tool,
} from './services/aguiService';
import { Sidebar } from './components/Sidebar';
import { Computation } from './components/Computation';
import { ExperimentRenderer } from './components/ExperimentRenderer';
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
// App
// =============================================================================

export default function App() {
  return (
    <CopilotKit publicApiKey="ck_pub_7aa7b776ea278992ea22100dd49f8b7b">
      <ResearchEngine />
    </CopilotKit>
  );
}

function ResearchEngine() {
  const { status } = useAGUIStatus();
  const { events, currentEvent, clearEvents } = useAGUIEvents();
  const { isRunning, state, artifacts, toolCalls, textContent, runAgent, abort } = useAGUIAgent();

  // Make artifacts readable by the Copilot
  useCopilotReadable({
    description: "The complete list of scientific artifacts generated in previous experiments.",
    value: artifacts
  });

  // UI State
  const [messages, setMessages] = useState<Array<{id: string; role: string; content: string}>>([]);
  const [pdbData, setPdbData] = useState<string | null>(null);
  const [currentTool, setCurrentTool] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [tools, setTools] = useState<Tool[]>([]);

  // Real data from backend
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string>('ws_core');
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | undefined>();
  const [knowledge, setKnowledge] = useState<KnowledgeItem[]>([]);
  const [showExperiment, setShowExperiment] = useState(false);

  // Fetch workspaces on mount
  useEffect(() => {
    const fetchWorkspaces = async () => {
      try {
        const response = await fetch(`${API_BASE}/workspaces`);
        if (response.ok) {
          const data = await response.json();
          setWorkspaces(data.workspaces || []);
          // Set active workspace to first one (usually ws_core)
          if (data.workspaces?.length > 0 && !activeWorkspaceId) {
            setActiveWorkspaceId(data.workspaces[0].id);
          }
        }
      } catch (e) {
        // Backend not available yet
      }
    };
    fetchWorkspaces();
  }, []);

  // Fetch conversations when workspace changes
  useEffect(() => {
    const fetchConversations = async () => {
      if (!activeWorkspaceId) return;
      try {
        const response = await fetch(`${API_BASE}/conversations?workspace_id=${activeWorkspaceId}`);
        if (response.ok) {
          const data = await response.json();
          setConversations(data.conversations || []);
        }
      } catch (e) {
        // Backend not available yet
      }
    };
    fetchConversations();
  }, [activeWorkspaceId]);

  // Fetch knowledge (Evolution) when workspace changes
  useEffect(() => {
    const fetchKnowledge = async () => {
      if (!activeWorkspaceId) return;
      try {
        const response = await fetch(`${API_BASE}/knowledge?workspace_id=${activeWorkspaceId}`);
        if (response.ok) {
          const data = await response.json();
          setKnowledge(data.knowledge || []);
        }
      } catch (e) {
        console.log('Knowledge endpoint not available');
      }
    };
    fetchKnowledge();
  }, [activeWorkspaceId]);

  // Restore artifacts on load
  useEffect(() => {
    aguiService.restoreSession();
  }, []);

  // Fetch tools on mount
  useEffect(() => {
    const fetchTools = async () => {
      try {
        const response = await fetch(`${API_BASE}/tools`);
        if (response.ok) {
          const data = await response.json();
          setTools(data.tools || []);
        }
      } catch (e) {
        console.log('Tools endpoint not available');
      }
    };
    fetchTools();
  }, []);

  // Handle AG-UI events
  useEffect(() => {
    if (!currentEvent) return;

    switch (currentEvent.type) {
      case EventType.RUN_STARTED:
        clearEvents();
        setShowExperiment(true);
        break;
      case EventType.TEXT_MESSAGE_START:
        setMessages(prev => [...prev, {
          id: currentEvent.messageId,
          role: 'assistant',
          content: '',
        }]);
        break;
      case EventType.TEXT_MESSAGE_CONTENT:
        setMessages(prev => prev.map(m =>
          m.id === currentEvent.messageId
            ? { ...m, content: m.content + (currentEvent.delta || '') }
            : m
        ));
        break;
      case EventType.TOOL_CALL_START:
        setCurrentTool(currentEvent.toolCallName || 'Processing');
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
        setCurrentTool(null);
        break;
    }
  }, [currentEvent, clearEvents]);

  // Send message
  const handleSend = useCallback(async (message: string) => {
    if (!message.trim() || status !== 'online') return;

    const userMsg = { id: `u-${Date.now()}`, role: 'user', content: message };
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
      await runAgent(message);
    } catch (e) {
      console.error(e);
    }
  }, [status, runAgent, activeConversationId, activeWorkspaceId]);

  // Handle feedback
  const handleFeedback = (artifactId: string, comment: string) => {
    const steeringPrompt = `[USER FEEDBACK on Artifact ${artifactId.slice(0,8)}]: "${comment}". Please adjust your plan or analysis based on this.`;
    setMessages(prev => [...prev, {
      id: `feedback-${Date.now()}`,
      role: 'user',
      content: `Feedback: ${comment}`
    }]);
    runAgent(steeringPrompt);
  };

  // New chat
  const handleNewChat = () => {
    setMessages([]);
    setActiveConversationId(undefined);
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

  const hasExperiment = artifacts.some(a => a.type === 'scientific_experiment');

  return (
    <div className="h-screen w-full bg-am-primary flex overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        tools={tools}
        artifacts={artifacts}
        conversations={conversations}
        knowledge={knowledge}
        onNewChat={handleNewChat}
        onSelectConversation={handleSelectConversation}
        onSelectTool={handleSelectTool}
        onSelectArtifact={handleArtifactClick}
        onSelectKnowledge={(item) => {
          console.log('Selected knowledge:', item);
          // TODO: Show knowledge detail modal
        }}
        activeConversationId={activeConversationId}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Computation Area */}
      <div className="flex-1 flex overflow-hidden">
        <Computation
          messages={messages}
          artifacts={artifacts}
          toolCalls={toolCalls}
          isRunning={isRunning}
          currentTool={currentTool}
          textContent={textContent}
          onSend={handleSend}
          onAbort={abort}
          onArtifactClick={handleArtifactClick}
          status={status}
          workspaces={workspaces}
          activeWorkspaceId={activeWorkspaceId}
          onWorkspaceChange={setActiveWorkspaceId}
        />

        {/* Experiment Panel (Right Side) */}
        {showExperiment && hasExperiment && (
          <div className="w-[45%] bg-am-secondary border-l border-am-border flex flex-col">
            <ExperimentRenderer
              experiment={artifacts.find(a => a.type === 'scientific_experiment')!}
              artifacts={artifacts}
              onFeedback={handleFeedback}
              onStructureClick={(pdb) => setPdbData(pdb)}
            />
          </div>
        )}
      </div>
    </div>
  );
}
