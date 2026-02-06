/**
 * Antimatters Computation Area
 * ============================
 * Main workspace where simulations run
 *
 * Steve Jobs: "Design is not just what it looks like. Design is how it works."
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send,
  Square,
  Plus,
  ChevronDown,
  Atom,
  Pencil,
  X,
} from 'lucide-react';
import { Artifact, ToolCall, AnnotationRegion } from '../../services/aguiService';
import { TraceStream } from '../TraceStream';

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
  textContent?: string;   // kept for backward-compat; no longer rendered here
  onSend: (message: string, mode?: ResearchMode, annotation?: AnnotationRegion) => void;
  onAbort: () => void;
  onArtifactClick?: (artifact: Artifact) => void;  // artifacts are shown in the right panel only
  onStructureClick?: (pdbData: string) => void;
  status: 'checking' | 'online' | 'offline';
  workspaces?: Workspace[];
  activeWorkspaceId?: string;
  onWorkspaceChange?: (workspaceId: string) => void;
  isStreaming?: boolean;
  firstTokenReceived?: boolean;
  onStartAnnotation?: () => void;  // Callback to start annotation mode on artifact panel
}

/**
 * Research Modes - Maps to agent routing:
 * - planning: Fast molecule generation using KG + SAR (planning_agent)
 * - serendipitize: Full workflow - protocol → experiment_matrix → discovery_report (docking_workflow)
 */
type ResearchMode = 'planning' | 'serendipitize';

// =============================================================================
// Lightweight markdown → HTML  (assistant messages only)
// =============================================================================
// Safety: raw text is HTML-escaped first; only whitelisted tags are injected.
// Links restricted to https:// origins.  Used exclusively on assistant content
// which originates from the backend, never from user input rendered this way.

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Inline transforms: **bold**, *italic*, `code`, [text](https://…) */
function inl(t: string): string {
  return t
    // links first so inner text isn't consumed by bold/italic
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, (_, text, url) =>
      `<a href="${url}" target="_blank" rel="noopener noreferrer" class="text-am-accent hover:underline">${text}</a>`
    )
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-am-text-primary">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em class="italic text-am-text-secondary">$1</em>')
    .replace(/`(.+?)`/g, '<code class="text-xs bg-am-tertiary px-1 rounded font-mono text-am-accent">$1</code>');
}

/** Line-level block transforms then inline pass.  Returns HTML string. */
function renderMd(raw: string): string {
  if (!raw) return '';
  const escaped = escHtml(raw);

  return escaped.split('\n').map(line => {
    // h1 / h2 / h3
    const h3 = line.match(/^###\s+(.*)/);
    if (h3) return `<h3 class="text-sm font-semibold text-am-text-primary mt-3 mb-1">${inl(h3[1])}</h3>`;
    const h2 = line.match(/^##\s+(.*)/);
    if (h2) return `<h2 class="text-sm font-semibold text-am-text-primary mt-3 mb-1">${inl(h2[1])}</h2>`;
    const h1 = line.match(/^#\s+(.*)/);
    if (h1) return `<h1 class="text-base font-semibold text-am-text-primary mt-3 mb-1">${inl(h1[1])}</h1>`;

    // unordered list item
    const li = line.match(/^[-*]\s+(.*)/);
    if (li) return `<div class="flex items-start gap-2 text-sm leading-relaxed ml-3"><span class="text-am-accent mt-0.5 flex-shrink-0">•</span><span>${inl(li[1])}</span></div>`;

    // ordered list item
    const oli = line.match(/^(\d+)\.\s+(.*)/);
    if (oli) return `<div class="text-sm leading-relaxed ml-3"><span class="text-am-accent font-semibold">${oli[1]}.</span> ${inl(oli[2])}</div>`;

    // blank line → small spacer
    if (line.trim() === '') return '<div class="h-2"></div>';

    // default paragraph
    return `<p class="text-sm leading-relaxed">${inl(line)}</p>`;
  }).join('');
}

export function Computation({
  messages,
  artifacts,
  toolCalls,
  isRunning,
  currentTool,
  onSend,
  onAbort,
  status,
  workspaces = [],
  activeWorkspaceId = 'ws_core',
  onWorkspaceChange,
  onStartAnnotation,
}: ComputationProps) {
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<ResearchMode>('serendipitize');
  const [showWorkspaceMenu, setShowWorkspaceMenu] = useState(false);
  const [showModeMenu, setShowModeMenu] = useState(false);
  const [pendingAnnotation, setPendingAnnotation] = useState<AnnotationRegion | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const activeWorkspace = workspaces.find(w => w.id === activeWorkspaceId);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, toolCalls.length]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  const handleSend = useCallback(() => {
    if (!input.trim() || status !== 'online') return;
    onSend(input, mode, pendingAnnotation || undefined);
    setInput('');
    setPendingAnnotation(null);
  }, [input, status, onSend, mode, pendingAnnotation]);

  // Start annotation mode - opens drawing canvas on artifact panel
  const startAnnotation = () => {
    onStartAnnotation?.();
  };

  // Receive annotation from artifact panel
  const handleAnnotationComplete = useCallback((annotation: AnnotationRegion) => {
    setPendingAnnotation(annotation);
    inputRef.current?.focus();
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const hasContent = messages.length > 0 || isRunning;

  return (
    <main className="flex-1 flex flex-col h-full bg-am-primary" role="main" aria-label="Research computation workspace">
      {/* Header */}
      <header className="flex-shrink-0 h-12 sm:h-14 flex items-center justify-between px-4 sm:px-6 border-b border-am-border">
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
              <div 
                className="fixed inset-0 z-10" 
                onClick={() => setShowWorkspaceMenu(false)}
                aria-hidden="true"
              />
              <div 
                className="absolute top-full left-0 mt-1 bg-am-secondary border border-am-border rounded-lg shadow-xl z-20 min-w-[280px]"
                role="menu"
                aria-label="Workspace selection"
              >
                <div className="py-1">
                  {workspaces.map(ws => (
                    <button
                      key={ws.id}
                      onClick={() => {
                        onWorkspaceChange?.(ws.id);
                        setShowWorkspaceMenu(false);
                      }}
                      className={`w-full px-3 py-3 text-left transition-colors min-h-[44px] ${
                        ws.id === activeWorkspaceId
                          ? 'bg-am-accent/20 text-am-accent'
                          : 'text-am-text-secondary hover:bg-am-tertiary hover:text-am-text-primary'
                      }`}
                      role="menuitem"
                      tabIndex={0}
                      aria-label={`Switch to ${ws.name} workspace`}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          onWorkspaceChange?.(ws.id);
                          setShowWorkspaceMenu(false);
                        } else if (e.key === 'Escape') {
                          setShowWorkspaceMenu(false);
                        }
                      }}
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
          {/* Streaming indicator — only while a named tool is actively running */}
          {isRunning && currentTool && (
            <div
              className="flex items-center gap-2 text-xs px-2 py-1 rounded-full bg-am-accent/10 text-am-accent"
              role="status"
              aria-live="polite"
              aria-label={`Running ${currentTool.replace(/_/g, ' ')}`}
            >
              <div className="w-1.5 h-1.5 rounded-full bg-am-accent animate-pulse" aria-hidden="true" />
              <span>{currentTool.replace(/_/g, ' ')}</span>
            </div>
          )}
        </div>
      </header>

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
            {/*
              Rendering contract (progressive-disclosure, no duplication):
                1. Messages in chronological order.  Molecule-card specials
                   are rendered inline; everything else is a plain bubble.
                2. A streaming cursor blinks on the last assistant message
                   while the run is live — no separate "streaming text" block.
                3. The TraceStream (collapsible tool-execution tree) sits
                   below the conversation.  It auto-collapses when the run
                   finishes so it does not eat scroll space.
                4. Full artifact renderers live exclusively in the right-hand
                   ArtifactPanel; nothing is duplicated here.
            */}
            {messages.map((msg, index) => {
              // ── special inline card ──────────────────────────────────
              if (msg.content?.startsWith('__MOLECULE_CARD__:')) {
                try {
                  const data = JSON.parse(msg.content.substring(18));
                  return <MoleculeCard key={msg.id} data={data} />;
                } catch (e) {
                  console.error('Failed to parse molecule card:', e);
                  return null;
                }
              }

              // ── Check if this is the last assistant message ──
              const isLastAssistant =
                msg.role === 'assistant' && index === messages.length - 1;

              // ── Q5c FIX: Only hide empty CURRENT assistant message while tools run ──
              // The backend emits TEXT_MESSAGE_START (empty content) before text arrives.
              // Hide ONLY the current (last) empty message, NOT old messages from previous responses
              if (msg.role === 'assistant' && !msg.content && toolCalls.length > 0 && isLastAssistant && isRunning) {
                return null;
              }

              // ── normal message ───────────────────────────────────────

              const showCursor =
                isLastAssistant && isRunning && (msg.content || toolCalls.length === 0);

              // While streaming the last assistant message, split at the
              // last newline: completed lines get full markdown treatment,
              // the tail (still being typed) renders as plain text so
              // characters stream in smoothly without block-level jumps.
              let assistantBody: React.ReactNode;
              if (msg.role === 'assistant') {
                if (isLastAssistant && isRunning && msg.content) {
                  const lastNL = msg.content.lastIndexOf('\n');
                  const done  = lastNL >= 0 ? msg.content.slice(0, lastNL + 1) : '';
                  const tail  = lastNL >= 0 ? msg.content.slice(lastNL + 1) : msg.content;
                  assistantBody = (
                    <>
                      {done && <div className="text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: renderMd(done) }} />}
                      {tail && <p className="text-sm leading-relaxed text-am-text-secondary">{tail}</p>}
                    </>
                  );
                } else {
                  assistantBody = <div className="text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: renderMd(msg.content) }} />;
                }
              }

              return (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] ${
                    msg.role === 'user'
                      ? 'bg-am-accent text-white rounded-2xl rounded-tr-sm px-4 py-2.5'
                      : 'text-am-text-primary'
                  }`}>
                    {msg.role === 'assistant'
                      ? assistantBody
                      : <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                    }
                    {showCursor && (
                      <span className="inline-block w-2 h-4 bg-am-accent ml-0.5 animate-pulse" />
                    )}
                  </div>
                </div>
              );
            })}

            {/* ── Tool-execution trace (collapsible, auto-collapses on finish) ── */}
            {toolCalls.length > 0 && (
              <TraceStream
                toolCalls={toolCalls}
                isRunning={isRunning}
                currentTool={currentTool}
              />
            )}

            {/* ── Typing indicator ─── shown only before ANY output exists ── */}
            {isRunning && messages.every(m => m.role === 'user') && toolCalls.length === 0 && (
              <div className="flex justify-start">
                <div className="bg-am-secondary border border-am-border rounded-2xl rounded-tl-sm px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="animate-typing flex gap-1">
                      <span className="w-2 h-2 bg-am-accent rounded-full"></span>
                      <span className="w-2 h-2 bg-am-accent rounded-full"></span>
                      <span className="w-2 h-2 bg-am-accent rounded-full"></span>
                    </div>
                    <span className="text-sm text-am-text-muted">Thinking...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Input Area */}
        <div className="flex-shrink-0 px-4 sm:px-6 pb-4 sm:pb-6">
          <div className="bg-am-secondary border-2 border-am-border rounded-xl overflow-hidden focus-within:border-am-accent focus-within:ring-2 focus-within:ring-am-accent/20 transition-all">
            {/* Input */}
            <div className="relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything, @ for context"
                className="w-full px-4 py-3 bg-transparent text-am-text-primary placeholder-am-text-muted text-sm resize-none focus:outline-none focus:ring-0"
                rows={1}
                style={{ minHeight: '44px', maxHeight: '200px' }}
                aria-label="Research input message"
                disabled={status !== 'online'}
              />
            </div>

            {/* Toolbar */}
            <div className="flex items-center justify-between px-3 py-2 border-t border-am-border/50">
              <div className="flex items-center gap-2">
                {/* Attach */}
                <button
                  className="p-2 text-am-text-muted hover:text-am-text-secondary hover:bg-am-tertiary rounded transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                  aria-label="Attach files or context"
                >
                  <Plus className="w-4 h-4" />
                </button>

                {/* Mode Selector Dropdown */}
                <ModeDropdown mode={mode} onChange={setMode} isOpen={showModeMenu} setIsOpen={setShowModeMenu} />

                {/* Annotation indicator */}
                {pendingAnnotation && (
                  <div className="flex items-center gap-1 px-2 py-1 bg-cyan-500/20 text-cyan-400 rounded text-xs">
                    <Pencil className="w-3 h-3" />
                    <span>Annotated</span>
                    <button onClick={() => setPendingAnnotation(null)} className="hover:text-cyan-200">
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2">
                {/* Annotate Button - Draw on artifacts */}
                <button
                  onClick={startAnnotation}
                  disabled={artifacts.length === 0}
                  className={`p-2 rounded transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center ${
                    artifacts.length > 0
                      ? 'text-am-text-muted hover:text-cyan-400 hover:bg-cyan-500/10'
                      : 'text-am-text-muted/30 cursor-not-allowed'
                  }`}
                  title={artifacts.length > 0 ? 'Draw on artifact' : 'Run analysis first'}
                >
                  <Pencil className="w-4 h-4" />
                </button>

                {/* Send/Stop */}
                <button
                  onClick={isRunning ? onAbort : handleSend}
                  disabled={!isRunning && (!input.trim() || status !== 'online')}
                  className={`p-3 rounded-lg transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center ${
                    isRunning
                      ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                      : input.trim() && status === 'online'
                        ? 'bg-am-accent text-white hover:bg-am-accent-hover'
                        : 'bg-am-tertiary text-am-text-muted cursor-not-allowed'
                  }`}
                  aria-label={isRunning ? 'Stop generation' : 'Send message'}
                >
                  {isRunning ? <Square className="w-4 h-4" /> : <Send className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}

// =============================================================================
// Sub-components
// =============================================================================

/** Mode dropdown - shows one at a time */
function ModeDropdown({
  mode,
  onChange,
  isOpen,
  setIsOpen,
}: {
  mode: ResearchMode;
  onChange: (m: ResearchMode) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}) {
  const modes: { id: ResearchMode; label: string }[] = [
    { id: 'serendipitize', label: 'Serendipitize' },
    { id: 'planning', label: 'Planning' },
  ];
  const current = modes.find(m => m.id === mode) || modes[0];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1 px-2 py-1.5 text-xs text-am-text-secondary hover:text-am-text-primary hover:bg-am-tertiary rounded transition-colors"
      >
        <span>{current.label}</span>
        <ChevronDown className="w-3 h-3" />
      </button>
      {isOpen && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setIsOpen(false)} />
          <div className="absolute bottom-full left-0 mb-1 bg-am-secondary border border-am-border rounded-lg shadow-xl z-20 py-1 min-w-[140px]">
            {modes.map(m => (
              <button
                key={m.id}
                onClick={() => { onChange(m.id); setIsOpen(false); }}
                className={`w-full px-3 py-2 text-xs text-left transition-colors ${
                  mode === m.id ? 'bg-am-accent/20 text-am-accent' : 'text-am-text-secondary hover:bg-am-tertiary'
                }`}
              >
                {m.label}
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
            <Atom className="w-4 h-4 text-purple-400" />
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
