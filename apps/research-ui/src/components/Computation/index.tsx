/**
 * Antimatters Computation Area
 * ============================
 * Main workspace where simulations run
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send,
  Square,
  Mic,
  Plus,
  ChevronDown,
  Sparkles,
  Atom,
  Inbox,
} from 'lucide-react';
import { Artifact, ToolCall } from '../../services/aguiService';
import { TraceStream } from '../TraceStream';
import { ArtifactRenderer } from '../ArtifactRenderer';

interface Workspace {
  id: string;
  name: string;
  description: string;
  stats?: { conversations: number; artifacts: number; knowledge_items: number };
}

interface ComputationProps {
  messages: Array<{ id: string; role: string; content: string }>;
  artifacts: Artifact[];
  toolCalls: ToolCall[];
  isRunning: boolean;
  currentTool: string | null;
  textContent: string;
  onSend: (message: string) => void;
  onAbort: () => void;
  onArtifactClick: (artifact: Artifact) => void;
  onStructureClick?: (pdbData: string) => void;
  status: 'checking' | 'online' | 'offline';
  workspaces?: Workspace[];
  activeWorkspaceId?: string;
  onWorkspaceChange?: (workspaceId: string) => void;
}

type Mode = 'planning' | 'execution' | 'analysis';

export function Computation({
  messages,
  artifacts,
  toolCalls,
  isRunning,
  currentTool,
  textContent,
  onSend,
  onAbort,
  onArtifactClick,
  onStructureClick,
  status,
  workspaces = [],
  activeWorkspaceId = 'ws_core',
  onWorkspaceChange,
}: ComputationProps) {
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<Mode>('execution');
  const [showWorkspaceMenu, setShowWorkspaceMenu] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const activeWorkspace = workspaces.find(w => w.id === activeWorkspaceId);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, textContent, toolCalls.length]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  const handleSend = useCallback(() => {
    if (!input.trim() || status !== 'online') return;
    onSend(input);
    setInput('');
  }, [input, status, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const hasContent = messages.length > 0 || isRunning;

  return (
    <div className="flex-1 flex flex-col h-full bg-am-primary">
      {/* Header */}
      <div className="flex-shrink-0 h-12 flex items-center justify-between px-6 border-b border-am-border">
        <div className="flex items-center gap-2 text-sm text-am-text-secondary relative">
          <span>Workspace:</span>
          <button
            onClick={() => setShowWorkspaceMenu(!showWorkspaceMenu)}
            className="flex items-center gap-1 px-2 py-1 text-am-text-primary hover:bg-am-tertiary rounded transition-colors"
          >
            <span className="font-medium">{activeWorkspace?.name || 'core'}</span>
            <ChevronDown className="w-3 h-3" />
          </button>

          {/* Workspace Dropdown */}
          {showWorkspaceMenu && workspaces.length > 0 && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setShowWorkspaceMenu(false)} />
              <div className="absolute top-full left-0 mt-1 bg-am-secondary border border-am-border rounded-lg shadow-xl z-20 min-w-[280px]">
                <div className="py-1">
                  {workspaces.map(ws => (
                    <button
                      key={ws.id}
                      onClick={() => {
                        onWorkspaceChange?.(ws.id);
                        setShowWorkspaceMenu(false);
                      }}
                      className={`w-full px-3 py-2 text-left transition-colors ${
                        ws.id === activeWorkspaceId
                          ? 'bg-am-accent/20 text-am-accent'
                          : 'text-am-text-secondary hover:bg-am-tertiary hover:text-am-text-primary'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium truncate">{ws.name}</div>
                          {ws.description && (
                            <div className="text-xs text-am-text-muted mt-0.5 line-clamp-1">
                              {ws.description}
                            </div>
                          )}
                        </div>
                        {ws.stats && (
                          <div className="flex items-center gap-1 text-xs text-am-text-muted flex-shrink-0">
                            <span>{ws.stats.conversations}</span>
                            <span className="opacity-50">|</span>
                            <span>{ws.stats.artifacts}</span>
                          </div>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-1.5 px-2 py-1 text-xs text-am-text-muted hover:text-am-text-secondary transition-colors">
            <Inbox className="w-3.5 h-3.5" />
            <span>View History</span>
          </button>
          <div className={`flex items-center gap-2 text-xs px-2 py-1 rounded-full ${
            status === 'online' ? 'bg-green-500/10 text-green-400' : 'bg-am-tertiary text-am-text-muted'
          }`}>
            <div className={`w-1.5 h-1.5 rounded-full ${status === 'online' ? 'bg-green-400' : 'bg-am-text-muted'}`} />
            {status === 'online' ? 'Online' : 'Connecting...'}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {/* Empty State - Clean welcome without static buttons */}
        {!hasContent && (
          <div className="flex-1 flex flex-col items-center justify-center px-6">
            <div className="max-w-md text-center space-y-4">
              <div className="w-12 h-12 mx-auto bg-am-tertiary rounded-xl flex items-center justify-center">
                <Atom className="w-6 h-6 text-am-accent" />
              </div>
              <div className="space-y-2">
                <h2 className="text-lg font-medium text-am-text-primary">Start a new research session</h2>
                <p className="text-sm text-am-text-muted">
                  Describe your research goal below. I can help with protein ensemble analysis, molecular docking, literature search, and more.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Messages */}
        {hasContent && (
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
            {messages.map((msg) => {
              // Check for special content types
              if (msg.content?.startsWith('__MOLECULE_CARD__:')) {
                try {
                  const data = JSON.parse(msg.content.substring(18));
                  return <MoleculeCard key={msg.id} data={data} />;
                } catch (e) {
                  console.error('Failed to parse molecule card:', e);
                  return null;
                }
              }

              return (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] ${
                    msg.role === 'user'
                      ? 'bg-am-accent text-white rounded-2xl rounded-tr-sm px-4 py-2.5'
                      : 'text-am-text-primary'
                  }`}>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  </div>
                </div>
              );
            })}

            {/* Inline Artifacts */}
            {artifacts
              .filter(a => a.type !== 'scientific_experiment')
              .slice(-3)
              .map(artifact => (
                <div key={artifact.id} className="flex justify-start">
                  <div
                    className="max-w-[90%] cursor-pointer hover:opacity-90 transition-opacity"
                    onClick={() => onArtifactClick(artifact)}
                  >
                    <ArtifactRenderer artifact={artifact} onStructureClick={onStructureClick} />
                  </div>
                </div>
              ))
            }

            {/* Tool Execution Trace */}
            {toolCalls.length > 0 && (
              <TraceStream
                toolCalls={toolCalls}
                isRunning={isRunning}
                currentTool={currentTool}
              />
            )}

            {/* Streaming Text */}
            {isRunning && textContent && !messages.find(m => m.content === textContent) && (
              <div className="flex justify-start">
                <div className="text-am-text-primary max-w-[80%]">
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{textContent}</p>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Input Area */}
        <div className="flex-shrink-0 px-6 pb-6">
          <div className="bg-am-secondary border border-am-border rounded-xl overflow-hidden focus-within:border-am-accent/50 transition-colors">
            {/* Input */}
            <div className="relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything, @ for context"
                className="w-full px-4 py-3 bg-transparent text-am-text-primary placeholder-am-text-muted text-sm resize-none focus:outline-none"
                rows={1}
                style={{ minHeight: '44px', maxHeight: '200px' }}
              />
            </div>

            {/* Toolbar */}
            <div className="flex items-center justify-between px-3 py-2 border-t border-am-border/50">
              <div className="flex items-center gap-2">
                {/* Attach */}
                <button className="p-1.5 text-am-text-muted hover:text-am-text-secondary hover:bg-am-tertiary rounded transition-colors">
                  <Plus className="w-4 h-4" />
                </button>

                {/* Mode Selector */}
                <ModeSelector mode={mode} onChange={setMode} />

                {/* Model Indicator */}
                <div className="flex items-center gap-1 px-2 py-1 text-xs text-am-text-muted">
                  <Sparkles className="w-3 h-3" />
                  <span>Gemini 2.0</span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {/* Voice */}
                <button className="p-1.5 text-am-text-muted hover:text-am-text-secondary hover:bg-am-tertiary rounded transition-colors">
                  <Mic className="w-4 h-4" />
                </button>

                {/* Send/Stop */}
                <button
                  onClick={isRunning ? onAbort : handleSend}
                  disabled={!isRunning && !input.trim()}
                  className={`p-2 rounded-lg transition-colors ${
                    isRunning
                      ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                      : input.trim()
                        ? 'bg-am-accent text-white hover:bg-am-accent-hover'
                        : 'bg-am-tertiary text-am-text-muted'
                  }`}
                >
                  {isRunning ? <Square className="w-4 h-4" /> : <Send className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Sub-components
// =============================================================================

function QuickAction({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-4 py-2 text-sm text-am-text-secondary bg-am-secondary border border-am-border rounded-lg hover:bg-am-tertiary hover:text-am-text-primary transition-colors"
    >
      {label}
    </button>
  );
}

function ModeSelector({ mode, onChange }: { mode: Mode; onChange: (m: Mode) => void }) {
  const [isOpen, setIsOpen] = useState(false);
  const modes: Mode[] = ['planning', 'execution', 'analysis'];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1 px-2 py-1 text-xs text-am-text-secondary hover:text-am-text-primary hover:bg-am-tertiary rounded transition-colors"
      >
        <span className="capitalize">{mode}</span>
        <ChevronDown className="w-3 h-3" />
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setIsOpen(false)} />
          <div className="absolute bottom-full left-0 mb-1 bg-am-secondary border border-am-border rounded-lg shadow-xl z-20 py-1 min-w-[120px]">
            {modes.map(m => (
              <button
                key={m}
                onClick={() => { onChange(m); setIsOpen(false); }}
                className={`w-full px-3 py-1.5 text-xs text-left transition-colors ${
                  mode === m
                    ? 'bg-am-accent/20 text-am-accent'
                    : 'text-am-text-secondary hover:bg-am-tertiary'
                }`}
              >
                <span className="capitalize">{m}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function MoleculeCard({ data }: { data: { name: string; smiles: string; svg: string } }) {
  return (
    <div className="flex justify-start">
      <div className="bg-am-secondary border border-am-border rounded-xl p-4 max-w-[280px]">
        <div className="flex items-center gap-2 mb-3">
          <div className="p-1.5 bg-purple-500/20 rounded-lg">
            <Sparkles className="w-4 h-4 text-purple-400" />
          </div>
          <div>
            <div className="text-[10px] font-medium text-purple-400 uppercase tracking-wider">Ligand</div>
            <div className="text-sm font-medium text-am-text-primary">{data.name}</div>
          </div>
        </div>
        <div
          className="bg-am-tertiary rounded-lg p-2 flex justify-center"
          dangerouslySetInnerHTML={{ __html: data.svg }}
        />
        <div className="mt-2 text-[10px] text-am-text-muted font-mono truncate px-1">
          {data.smiles}
        </div>
      </div>
    </div>
  );
}
