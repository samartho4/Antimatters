/**
 * Tool Call Visualizer Component
 * ===============================
 * Displays real-time tool calls from AG-UI events:
 * - Shows tool name, arguments, and status
 * - Animated execution indicator
 * - Collapsible argument details
 * - Links to related artifacts
 */

import React, { useState } from 'react';
import { ToolCall, EventType, AGUIEvent } from '../services/aguiService';

// Tool icons mapping
const TOOL_ICONS: Record<string, React.ReactNode> = {
  fetch_ped_ensemble: (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  ),
  prepare_ligand: (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v4m0 12v4M2 12h4m12 0h4" />
    </svg>
  ),
  cluster_conformations: (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="2" />
      <circle cx="6" cy="6" r="2" />
      <circle cx="18" cy="6" r="2" />
      <circle cx="6" cy="18" r="2" />
      <circle cx="18" cy="18" r="2" />
    </svg>
  ),
  dock_ensemble: (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2L2 7l10 5 10-5-10-5z" />
      <path d="M2 17l10 5 10-5" />
      <path d="M2 12l10 5 10-5" />
    </svg>
  ),
  analyze_interactions: (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  ),
  search_compounds: (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  ),
};

// Tool colors
const TOOL_COLORS: Record<string, string> = {
  fetch_ped_ensemble: 'from-blue-500 to-cyan-500',
  prepare_ligand: 'from-purple-500 to-pink-500',
  cluster_conformations: 'from-orange-500 to-amber-500',
  dock_ensemble: 'from-green-500 to-emerald-500',
  analyze_interactions: 'from-cyan-500 to-teal-500',
  search_compounds: 'from-indigo-500 to-blue-500',
};

// Tool descriptions
const TOOL_DESCRIPTIONS: Record<string, string> = {
  fetch_ped_ensemble: 'Fetching protein ensemble from PED database...',
  prepare_ligand: 'Converting SMILES to 3D structure...',
  cluster_conformations: 'Clustering conformations with t-SNE + k-means...',
  dock_ensemble: 'Running AutoDock Vina on ensemble...',
  analyze_interactions: 'Analyzing protein-ligand interactions...',
  search_compounds: 'Searching ChEMBL database...',
};

interface ToolCallItemProps {
  toolCall: ToolCall;
  status: 'pending' | 'running' | 'completed' | 'error';
  result?: any;
}

const ToolCallItem: React.FC<ToolCallItemProps> = ({ toolCall, status, result }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const icon = TOOL_ICONS[toolCall.name] || (
    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
    </svg>
  );

  const gradient = TOOL_COLORS[toolCall.name] || 'from-slate-500 to-slate-600';
  const description = TOOL_DESCRIPTIONS[toolCall.name] || 'Executing tool...';

  return (
    <div className="bg-slate-800/50 rounded-lg overflow-hidden">
      {/* Header */}
      <div
        className="flex items-center gap-3 p-3 cursor-pointer hover:bg-slate-800/80 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {/* Icon with gradient background */}
        <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${gradient} flex items-center justify-center text-white`}>
          {icon}
        </div>

        {/* Tool Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-200 truncate">
              {toolCall.name.replace(/_/g, ' ')}
            </span>
            {status === 'running' && (
              <span className="flex items-center gap-1 text-[10px] text-blue-400">
                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" />
                Running
              </span>
            )}
            {status === 'completed' && (
              <svg className="w-4 h-4 text-green-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            )}
            {status === 'error' && (
              <svg className="w-4 h-4 text-red-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
            )}
          </div>
          <div className="text-xs text-slate-500 truncate">
            {status === 'running' ? description : `ID: ${toolCall.id.slice(0, 8)}...`}
          </div>
        </div>

        {/* Expand Button */}
        <svg
          className={`w-4 h-4 text-slate-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t border-slate-700 p-3 space-y-3">
          {/* Arguments */}
          {Object.keys(toolCall.args).length > 0 && (
            <div>
              <div className="text-[10px] text-slate-500 uppercase mb-1">Arguments</div>
              <div className="bg-slate-900 rounded p-2 font-mono text-xs text-slate-300 overflow-x-auto">
                <pre>{JSON.stringify(toolCall.args, null, 2)}</pre>
              </div>
            </div>
          )}

          {/* Result */}
          {result && (
            <div>
              <div className="text-[10px] text-slate-500 uppercase mb-1">Result</div>
              <div className="bg-slate-900 rounded p-2 font-mono text-xs text-green-400 overflow-x-auto max-h-32">
                <pre>{typeof result === 'string' ? result : JSON.stringify(result, null, 2)}</pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// =============================================================================
// Main Tool Call Visualizer
// =============================================================================

interface ToolCallVisualizerProps {
  toolCalls: ToolCall[];
  events?: AGUIEvent[];
  isRunning?: boolean;
}

export const ToolCallVisualizer: React.FC<ToolCallVisualizerProps> = ({
  toolCalls,
  events = [],
  isRunning = false,
}) => {
  // Build status map from events
  const getToolStatus = (toolCallId: string): 'pending' | 'running' | 'completed' | 'error' => {
    const hasStart = events.some(e => e.type === EventType.TOOL_CALL_START && e.toolCallId === toolCallId);
    const hasEnd = events.some(e => e.type === EventType.TOOL_CALL_END && e.toolCallId === toolCallId);
    const hasResult = events.some(e => e.type === EventType.TOOL_CALL_RESULT && e.toolCallId === toolCallId);

    if (hasResult || hasEnd) return 'completed';
    if (hasStart) return 'running';
    return 'pending';
  };

  const getToolResult = (toolCallId: string): any => {
    const resultEvent = events.find(
      e => e.type === EventType.TOOL_CALL_RESULT && e.toolCallId === toolCallId
    );
    return resultEvent?.content;
  };

  if (toolCalls.length === 0 && !isRunning) {
    return null;
  }

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
          <svg className="w-4 h-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
          </svg>
          Tool Execution
        </h3>
        {isRunning && (
          <span className="flex items-center gap-1.5 text-xs text-blue-400">
            <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
            Processing
          </span>
        )}
      </div>

      {/* Tool Calls List */}
      <div className="space-y-2">
        {toolCalls.map(tc => (
          <ToolCallItem
            key={tc.id}
            toolCall={tc}
            status={getToolStatus(tc.id)}
            result={getToolResult(tc.id)}
          />
        ))}
      </div>

      {/* Empty Running State */}
      {toolCalls.length === 0 && isRunning && (
        <div className="bg-slate-800/50 rounded-lg p-4 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-slate-700 flex items-center justify-center">
            <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          </div>
          <div>
            <div className="text-sm text-slate-300">Analyzing request...</div>
            <div className="text-xs text-slate-500">Agent is determining which tools to use</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ToolCallVisualizer;
