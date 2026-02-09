/**
 * ArtifactPanel — Document-first artifact viewer
 *
 * Filters out noise (structure_3d rendered in viewer, ligand_svg rendered inline,
 * scientific_experiment shells with no real content).
 * Auto-selects the most relevant artifact. Navigates with prev/next + dropdown.
 */

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  FileText,
  Beaker,
  BarChart3,
  Atom,
  FlaskConical,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Download,
  Copy,
  X,
  Maximize2,
  Minimize2,
  Check,
  Trash2,
  Send,
} from 'lucide-react';
import { Artifact, AnnotationRegion } from '../../services/aguiService';
import { ArtifactRenderer } from '../ArtifactRenderer';

// =============================================================================
// Filtering — what actually belongs in this panel
// =============================================================================

/** Types rendered elsewhere; showing them here is redundant */
const INLINE_TYPES = new Set(['ligand_svg', 'cluster_visualization']);

function filterVisibleArtifacts(artifacts: Artifact[]): Artifact[] {
  const filtered = artifacts.filter(a => {
    if (INLINE_TYPES.has(a.type)) return false;
    // structure_3d without viewable PDB data is noise
    if (a.type === 'structure_3d' && !a.content?.pdb_data && !a.content?.html_viewer) return false;
    // scientific_experiment shells only have {title, status} — useless card
    if (a.type === 'scientific_experiment') {
      const keys = Object.keys(a.content || {}).filter(
        k => !['title', 'status', 'name', 'metadata'].includes(k)
      );
      return keys.length > 0;
    }
    return true;
  });

  // Keep only the latest evolution_trace — the agent may emit multiple
  // versions with different IDs as the knowledge graph grows during a run.
  const lastEvIdx = filtered.reduce(
    (last, a, i) => (a.type === 'evolution_trace' ? i : last), -1
  );
  if (lastEvIdx < 0) return filtered;
  return filtered.filter((a, i) => a.type !== 'evolution_trace' || i === lastEvIdx);
}

/**
 * Pick the best artifact to show on initial load.
 * discovery_report = run finished, most informative.
 * experiment_matrix = docking in progress.
 * task_list (largest) = research planning.
 * protocol = fallback structured artifact.
 */
function findBestArtifactIndex(artifacts: Artifact[]): number {
  // If there's a discovery_report, the run is done — lead with that
  const reportIdx = artifacts.findIndex(a => a.type === 'discovery_report');
  if (reportIdx >= 0) return reportIdx;

  // Otherwise lead with experiment_matrix (live docking data)
  const matrixIdx = artifacts.findIndex(a => a.type === 'experiment_matrix');
  if (matrixIdx >= 0) return matrixIdx;

  // Generated molecules (user explicitly requested these)
  const molSugIdx = artifacts.findIndex(a => a.type === 'molecule_suggestions');
  if (molSugIdx >= 0) return molSugIdx;

  // 3D docking visualization
  const structIdx = artifacts.findIndex(a => a.type === 'structure_3d');
  if (structIdx >= 0) return structIdx;

  // Otherwise the largest task_list (overall workflow)
  let bestTaskIdx = -1, bestTaskCount = 0;
  artifacts.forEach((a, i) => {
    if (a.type === 'task_list') {
      const count = (a.content?.tasks || []).length;
      if (count > bestTaskCount) { bestTaskCount = count; bestTaskIdx = i; }
    }
  });
  if (bestTaskIdx >= 0) return bestTaskIdx;

  // Protocol or first available
  const protoIdx = artifacts.findIndex(a => a.type === 'protocol');
  return protoIdx >= 0 ? protoIdx : 0;
}

// =============================================================================
// Helpers
// =============================================================================

function getArtifactTitle(artifact?: Artifact): string {
  if (!artifact) return 'Untitled';

  // Top-level name from the backend envelope
  if (typeof artifact.name === 'string' && artifact.name) return artifact.name;

  const content = artifact.content as any;

  if (typeof content?.title === 'string' && content.title) return content.title;
  if (typeof content?.name === 'string' && content.name) return content.name;
  if (typeof content?.protein_name === 'string') return content.protein_name;

  switch (artifact.type) {
    case 'task_list':
      return content?.tasks ? `Research Tasks (${content.tasks.length} steps)` : 'Research Tasks';
    case 'protocol':
      return content?.protein_name || 'Protocol';
    case 'experiment_matrix': {
      const n = content?.ligands?.length || content?.ligand_results?.length || 0;
      return n > 0 ? `Experiment Matrix (${n} ligands)` : 'Experiment Matrix';
    }
    case 'docking_result': {
      const p = content?.binding_scores?.length || 0;
      return p > 0 ? `Docking Results (${p} poses)` : 'Docking Results';
    }
    case 'discovery_report': return 'Discovery Report';
    case 'evolution_trace':  return content?.protein_name || 'Knowledge Graph';
    case 'structure_3d':     return content?.ligand_name ? `3D: ${content.ligand_name}` : '3D Visualization';
    case 'molecule_suggestions': {
      const n = content?.suggestions?.length || 0;
      return n > 0 ? `Generated Molecules (${n})` : 'Generated Molecules';
    }
    default:
      return artifact.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }
}

// =============================================================================
// Props
// =============================================================================

interface ArtifactPanelProps {
  artifacts: Artifact[];
  onStructureClick?: (pdb: string) => void;
  onClose?: (artifactId: string) => void;
  isAnnotating?: boolean;
  onStartAnnotation?: () => void;
  onAnnotationSend?: (annotation: AnnotationRegion, question: string) => void;
  onCancelAnnotation?: () => void;
}

// =============================================================================
// Main component
// =============================================================================

export function ArtifactPanel({
  artifacts,
  onStructureClick,
  onClose,
  isAnnotating = false,
  onStartAnnotation,
  onAnnotationSend,
  onCancelAnnotation,
}: ArtifactPanelProps) {
  const visibleArtifacts = useMemo(() => filterVisibleArtifacts(artifacts), [artifacts]);

  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isExpanded, setIsExpanded] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Annotation: captured screenshot lives here between "Done" and "Send"
  const [pendingAnnotation, setPendingAnnotation] = useState<AnnotationRegion | null>(null);
  const [annotationQuestion, setAnnotationQuestion] = useState('');

  // Track previous visible count so we can detect new arrivals
  const prevCountRef = useRef(0);
  const initializedRef = useRef(false);
  // Once the user manually picks an artifact (prev/next/dropdown) we stop
  // auto-jumping so they can actually browse.  Resets when artifacts clear
  // (new run starts).
  const userNavigatedRef = useRef(false);

  /** Wrap every user-driven navigation so we can stop auto-jump. */
  const selectIndex = useCallback((idx: number) => {
    userNavigatedRef.current = true;
    setSelectedIndex(idx);
  }, []);

  // Auto-select: once on first load, then jump to newest — but only while
  // the user hasn't manually navigated.
  useEffect(() => {
    if (visibleArtifacts.length === 0) {
      initializedRef.current = false;
      prevCountRef.current = 0;
      userNavigatedRef.current = false; // reset for next run
      return;
    }

    if (!initializedRef.current) {
      // First meaningful render — pick the most informative artifact
      initializedRef.current = true;
      setSelectedIndex(findBestArtifactIndex(visibleArtifacts));
    } else if (visibleArtifacts.length > prevCountRef.current && !userNavigatedRef.current) {
      // A new artifact arrived and user hasn't manually browsed — follow it
      setSelectedIndex(visibleArtifacts.length - 1);
    }
    prevCountRef.current = visibleArtifacts.length;
  }, [visibleArtifacts]);

  // Clamp selectedIndex if artifacts shrink
  useEffect(() => {
    if (visibleArtifacts.length > 0 && selectedIndex >= visibleArtifacts.length) {
      setSelectedIndex(visibleArtifacts.length - 1);
    }
  }, [visibleArtifacts.length, selectedIndex]);

  // ── Annotation canvas ──────────────────────────────────────────────────────
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [paths, setPaths] = useState<Array<{x: number; y: number}[]>>([]);
  const [currentPath, setCurrentPath] = useState<{x: number; y: number}[]>([]);

  useEffect(() => {
    if (isAnnotating && canvasRef.current && containerRef.current) {
      const canvas = canvasRef.current;
      canvas.width  = containerRef.current.offsetWidth;
      canvas.height = containerRef.current.offsetHeight;
      setPaths([]);
      setCurrentPath([]);

      // Stop 3D viewer spin for stable annotation
      const findViewer = (el: Element | null): any => {
        if (!el) return null;
        if ((el as any).__viewer3d) return (el as any).__viewer3d;
        for (const child of Array.from(el.children)) {
          const found = findViewer(child);
          if (found) return found;
        }
        return null;
      };
      const viewer = findViewer(containerRef.current);
      if (viewer?.spin) {
        viewer.spin(false);
      }
    }
  }, [isAnnotating]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    [...paths, currentPath].forEach(path => {
      if (path.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(path[0].x, path[0].y);
      path.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.stroke();
    });
  }, [paths, currentPath]);

  const getCanvasPos = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDrawing(true);
    setCurrentPath([getCanvasPos(e)]);
  };
  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    setCurrentPath(prev => [...prev, getCanvasPos(e)]);
  };
  const stopDrawing = () => {
    if (currentPath.length > 1) setPaths(prev => [...prev, currentPath]);
    setCurrentPath([]);
    setIsDrawing(false);
  };
  const clearCanvas = () => { setPaths([]); setCurrentPath([]); };

  // Capture the visible artifact (3D viewer or iframe or plain content) +
  // user annotation strokes → single PNG base64 for multimodal Gemini input.
  const submitAnnotation = useCallback(async () => {
    if (!canvasRef.current) return;

    const w = canvasRef.current.width;
    const h = canvasRef.current.height;
    const combined = document.createElement('canvas');
    combined.width  = w;
    combined.height = h;
    const ctx = combined.getContext('2d');
    if (!ctx) return;

    // ── 1. Try to get the 3D viewer background via exportImage() ──
    let got3D = false;
    const findViewer = (el: Element | null): any => {
      if (!el) return null;
      if ((el as any).__viewer3d) return (el as any).__viewer3d;
      for (const child of Array.from(el.children)) {
        const found = findViewer(child);
        if (found) return found;
      }
      return null;
    };
    const viewer = findViewer(containerRef.current);
    if (viewer?.exportImage) {
      try {
        const imgUrl: string = await viewer.exportImage();
        const img = new Image();
        img.src = imgUrl;
        await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = () => rej(); });
        ctx.drawImage(img, 0, 0, w, h);
        got3D = true;
      } catch { /* fall through */ }
    }

    // ── 2. Fallback: raw canvas toDataURL (works for non-WebGL canvases) ──
    if (!got3D) {
      const canvases = containerRef.current?.querySelectorAll('canvas') || [];
      for (const c of Array.from(canvases)) {
        if (c === canvasRef.current) continue;
        try {
          const img = new Image();
          img.src = c.toDataURL('image/png');
          await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = () => rej(); });
          ctx.drawImage(img, 0, 0, w, h);
          got3D = true;
          break;
        } catch { /* tainted canvas — skip */ }
      }
    }

    // ── 3. If still nothing, fill with the panel background so strokes show ──
    if (!got3D) {
      ctx.fillStyle = '#1a1d21';
      ctx.fillRect(0, 0, w, h);
    }

    // ── 4. Overlay user annotation strokes ──
    ctx.drawImage(canvasRef.current, 0, 0);

    setPendingAnnotation({
      residue_ids: [],
      atom_ids: [],
      description: 'User drawing annotation on 3D structure',
      screenshot_base64: combined.toDataURL('image/png'),
    });
  }, []);

  // User typed a question (or left it empty for default) → send multimodal
  const sendAnnotation = useCallback(() => {
    if (!pendingAnnotation) return;
    const q = annotationQuestion.trim() ||
      'Analyze the annotated region. What interactions are present at the binding site? Consider H-bonds, hydrophobic contacts, and aromatic stacking. Reference any SAR insights from prior experiments.';
    onAnnotationSend?.(pendingAnnotation, q);
    setPendingAnnotation(null);
    setAnnotationQuestion('');
    onCancelAnnotation?.();
  }, [pendingAnnotation, annotationQuestion, onAnnotationSend, onCancelAnnotation]);

  // Right-click on artifact → enter annotation mode (replaces FeedbackOverlay)
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    if (isAnnotating || pendingAnnotation) return;
    e.preventDefault();
    onStartAnnotation?.();
  }, [isAnnotating, pendingAnnotation, onStartAnnotation]);

  // Ctrl+K / Cmd+K → annotation, Escape → dismiss
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        if (!isAnnotating && !pendingAnnotation) onStartAnnotation?.();
      }
      if (e.key === 'Escape' && (isAnnotating || pendingAnnotation)) {
        setPendingAnnotation(null);
        setAnnotationQuestion('');
        onCancelAnnotation?.();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isAnnotating, pendingAnnotation, onStartAnnotation, onCancelAnnotation]);

  // ── Actions ────────────────────────────────────────────────────────────────
  const currentArtifact = visibleArtifacts[selectedIndex] || visibleArtifacts[0];

  const handleCopy = async () => {
    if (!currentArtifact) return;
    await navigator.clipboard.writeText(JSON.stringify(currentArtifact.content, null, 2));
    setCopiedId(currentArtifact.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleDownload = () => {
    if (!currentArtifact) return;
    const blob = new Blob([JSON.stringify(currentArtifact.content, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentArtifact.type}_${currentArtifact.id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Empty / waiting state ──────────────────────────────────────────────────
  if (visibleArtifacts.length === 0) {
    return (
      <div className={`bg-am-primary flex flex-col ${isExpanded ? 'fixed inset-0 z-50' : 'h-full'}`}>
        {/* Minimal empty header */}
        <div className="flex-shrink-0 flex items-center justify-between px-3 py-2.5 border-b border-am-border/50">
          <h2 className="text-sm font-medium text-am-text-primary">Artifacts</h2>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 text-am-text-muted hover:text-am-text-secondary transition-colors"
          >
            {isExpanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center px-8">
            <div className="w-14 h-14 mx-auto mb-3 rounded-xl bg-am-tertiary flex items-center justify-center">
              <Beaker className="w-7 h-7 text-am-text-muted" />
            </div>
            <h3 className="text-sm font-medium text-am-text-secondary mb-1">
              {artifacts.length > 0 ? 'Processing…' : 'No artifacts yet'}
            </h3>
            <p className="text-xs text-am-text-muted max-w-[180px] mx-auto">
              {artifacts.length > 0
                ? 'Results will appear here once ready'
                : 'Artifacts appear as your research progresses'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  const hasMultiple   = visibleArtifacts.length > 1;
  const canGoBack     = selectedIndex > 0;
  const canGoForward  = selectedIndex < visibleArtifacts.length - 1;

  // Get version from artifact (top-level or metadata)
  const currentVersion = currentArtifact?.version ?? currentArtifact?.metadata?.version;

  return (
    <div className={`bg-am-primary flex flex-col ${isExpanded ? 'fixed inset-0 z-50' : 'h-full'}`}>
      {/* Minimal Header — Steve Jobs style: one clean row */}
      <div className="flex-shrink-0 flex items-center gap-2 px-3 py-2.5 border-b border-am-border/50">
        {/* Navigation arrows (only if multiple artifacts) */}
        {hasMultiple && (
          <div className="flex items-center">
            <button
              onClick={() => canGoBack && selectIndex(selectedIndex - 1)}
              disabled={!canGoBack}
              className="p-1 text-am-text-muted hover:text-am-text-primary disabled:opacity-20 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => canGoForward && selectIndex(selectedIndex + 1)}
              disabled={!canGoForward}
              className="p-1 text-am-text-muted hover:text-am-text-primary disabled:opacity-20 transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Title + Version */}
        <div className="flex-1 min-w-0 flex items-center gap-2">
          <h2 className="text-sm font-medium text-am-text-primary truncate">
            {getArtifactTitle(currentArtifact)}
          </h2>
          {currentVersion !== undefined && (
            <span className="flex-shrink-0 text-[10px] text-am-accent font-mono">
              v{currentVersion}
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-0.5">
          {hasMultiple && (
            <ArtifactSelector artifacts={visibleArtifacts} selectedIndex={selectedIndex} onSelect={selectIndex} />
          )}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 text-am-text-muted hover:text-am-text-secondary transition-colors"
          >
            {isExpanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
          {onClose && (
            <button
              onClick={() => onClose(currentArtifact.id)}
              className="p-1.5 text-am-text-muted hover:text-am-text-secondary transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Document body */}
      <div className="flex-1 overflow-y-auto relative" ref={containerRef} onContextMenu={handleContextMenu}>
        <div className="p-5">
          <ArtifactRenderer artifact={currentArtifact} onStructureClick={onStructureClick} />
        </div>

        {/* Annotation canvas (draw mode) — hidden once screenshot is captured */}
        {isAnnotating && !pendingAnnotation && (
          <>
            <canvas
              ref={canvasRef}
              className="absolute inset-0 cursor-crosshair z-10"
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              onTouchStart={startDrawing}
              onTouchMove={draw}
              onTouchEnd={stopDrawing}
            />
            <div className="absolute top-2 right-2 z-20 flex items-center gap-1 bg-am-secondary/90 backdrop-blur rounded-lg p-1 border border-am-border">
              <button onClick={clearCanvas} className="p-2 text-am-text-muted hover:text-am-text-secondary hover:bg-am-tertiary rounded" title="Clear">
                <Trash2 className="w-4 h-4" />
              </button>
              <button onClick={onCancelAnnotation} className="p-2 text-am-text-muted hover:text-red-400 hover:bg-red-500/10 rounded" title="Cancel">
                <X className="w-4 h-4" />
              </button>
              <button
                onClick={submitAnnotation}
                className="px-3 py-1.5 text-xs rounded bg-cyan-500 text-white hover:bg-cyan-600"
              >
                Done
              </button>
            </div>
            {paths.length === 0 && !isDrawing && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-5">
                <div className="bg-am-secondary/80 backdrop-blur px-4 py-2 rounded-lg text-sm text-am-text-muted">
                  Draw to highlight, or click Done to send this view
                </div>
              </div>
            )}
          </>
        )}

        {/* Question bar — appears after Done, sends annotation + question to Gemini */}
        {pendingAnnotation && (
          <div className="absolute bottom-0 left-0 right-0 z-20 bg-am-secondary/95 backdrop-blur-sm border-t border-am-border p-3">
            <div className="text-xs text-am-text-muted mb-2 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
              View captured — ask Gemini about it (or press Enter for default analysis)
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => { setPendingAnnotation(null); setAnnotationQuestion(''); }}
                className="p-1.5 text-am-text-muted hover:text-am-text-secondary transition-colors"
                title="Dismiss"
              >
                <X className="w-3.5 h-3.5" />
              </button>
              <input
                type="text"
                value={annotationQuestion}
                onChange={e => setAnnotationQuestion(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && sendAnnotation()}
                placeholder="What interactions do you see at the binding site?"
                className="flex-1 text-sm bg-am-primary border border-am-border rounded-lg px-3 py-2 text-am-text-primary placeholder-am-text-muted focus:border-am-accent focus:outline-none focus:ring-1 focus:ring-am-accent"
                autoFocus
              />
              <button
                onClick={sendAnnotation}
                className="p-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg transition-colors"
                title="Send to Gemini"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 border-t border-am-border px-4 py-2 bg-am-secondary/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <button onClick={handleCopy} className="flex items-center gap-1.5 px-2 py-1.5 text-xs text-am-text-muted hover:text-am-text-secondary hover:bg-am-tertiary rounded transition-colors">
              {copiedId === currentArtifact.id ? (
                <><Check className="w-3.5 h-3.5 text-green-400" /><span className="text-green-400">Copied</span></>
              ) : (
                <><Copy className="w-3.5 h-3.5" /><span>Copy</span></>
              )}
            </button>
            <button onClick={handleDownload} className="flex items-center gap-1.5 px-2 py-1.5 text-xs text-am-text-muted hover:text-am-text-secondary hover:bg-am-tertiary rounded transition-colors">
              <Download className="w-3.5 h-3.5" /><span>Export</span>
            </button>
          </div>
          <div className="text-[10px] text-am-text-muted font-mono">{currentArtifact.id.slice(0, 12)}</div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Artifact selector dropdown
// =============================================================================

function ArtifactSelector({
  artifacts,
  selectedIndex,
  onSelect,
}: {
  artifacts: Artifact[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const selected = artifacts[selectedIndex];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 bg-am-tertiary border border-am-border rounded-lg text-xs text-am-text-secondary hover:border-am-accent/50 transition-colors"
      >
        <ArtifactTypeIcon type={selected?.type} />
        <span className="max-w-[140px] truncate">{getArtifactTitle(selected)}</span>
        <ChevronDown className={`w-3.5 h-3.5 text-am-text-muted transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setIsOpen(false)} />
          <div className="absolute top-full right-0 mt-1 w-72 bg-am-secondary border border-am-border rounded-lg shadow-xl z-20 max-h-[280px] overflow-y-auto">
            {artifacts.map((artifact, idx) => (
              <button
                key={artifact.id}
                onClick={() => { onSelect(idx); setIsOpen(false); }}
                className={`w-full flex items-center gap-2 px-3 py-2.5 text-xs transition-colors border-b border-am-border/30 last:border-b-0 ${
                  idx === selectedIndex
                    ? 'bg-am-accent/10 text-am-accent'
                    : 'text-am-text-secondary hover:bg-am-tertiary hover:text-am-text-primary'
                }`}
              >
                <ArtifactTypeIcon type={artifact.type} />
                <div className="flex-1 min-w-0 text-left">
                  <div className="truncate font-medium">{getArtifactTitle(artifact)}</div>
                  <div className="text-[10px] text-am-text-muted flex items-center gap-1.5">
                    {(artifact.version !== undefined || artifact.metadata?.version !== undefined) && (
                      <span className="text-am-accent font-mono">v{artifact.version ?? artifact.metadata?.version}</span>
                    )}
                    <span className="opacity-60">{artifact.type.replace(/_/g, ' ')}</span>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// =============================================================================
// Type icon
// =============================================================================

function ArtifactTypeIcon({ type }: { type?: string }) {
  const cls = "w-4 h-4";
  switch (type) {
    case 'task_list':             return <FileText className={`${cls} text-blue-400`} />;
    case 'protocol':             return <FileText className={`${cls} text-indigo-400`} />;
    case 'experiment_matrix':    return <Beaker className={`${cls} text-purple-400`} />;
    case 'docking_result':       return <BarChart3 className={`${cls} text-emerald-400`} />;
    case 'discovery_report':     return <BarChart3 className={`${cls} text-green-400`} />;
    case 'evolution_trace':      return <Atom className={`${cls} text-cyan-400`} />;
    case 'structure_3d':         return <Atom className={`${cls} text-teal-400`} />;
    case 'molecule_suggestions': return <FlaskConical className={`${cls} text-amber-400`} />;
    case 'ligand_svg':           return <FlaskConical className={`${cls} text-orange-400`} />;
    default:                     return <FileText className={`${cls} text-am-text-muted`} />;
  }
}
