import React, { useState, useEffect } from 'react';
import {
  Bot,
  Terminal,
  Database,
  ChevronRight,
  ChevronDown,
  CheckCircle2,
  AlertCircle,
  Clock,
  ArrowRight,
  Cpu,
  Beaker,
  FileCode,
  Activity,
  Loader2,
  Network
} from 'lucide-react';
import { ToolCall } from '../services/aguiService';

interface TraceStreamProps {
  toolCalls: ToolCall[];
  isRunning: boolean;
  currentTool: string | null;
}

export function TraceStream({ toolCalls, isRunning, currentTool }: TraceStreamProps) {
  return (
    <div className="flex flex-col space-y-4 my-6 font-sans">
      {/* Level 1: Goal / Root Node */}
      <div className="flex items-center gap-3 px-4 py-3 bg-am-secondary border border-am-border rounded-xl">
        <div className="p-2 bg-am-accent/20 rounded-lg">
          <Activity className={`w-5 h-5 text-am-accent ${isRunning ? 'animate-pulse' : ''}`} />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-am-text-primary">Simulation Campaign</h3>
          <div className="text-xs text-am-text-muted font-mono flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-400 animate-pulse' : 'bg-am-text-muted'}`} />
            {isRunning ? 'EXECUTING' : 'COMPLETED'}
          </div>
        </div>
      </div>

      {/* Level 2 & 3: The Stream */}
      <div className="relative pl-6 border-l-2 border-am-border ml-6 space-y-6">
        {toolCalls.map((call, idx) => (
          <TraceNode key={call.id} call={call} isLast={idx === toolCalls.length - 1} isRunning={isRunning} />
        ))}

        {/* Active Tool Indicator */}
        {isRunning && currentTool && (
          <div className="relative pl-8 animate-fade-in">
            <div className="absolute -left-[29px] top-0 bg-am-primary p-1">
              <div className="w-3 h-3 bg-am-accent rounded-full ring-4 ring-am-primary animate-pulse" />
            </div>
            <div className="flex items-center gap-2 px-3 py-2 bg-am-accent/10 border border-am-accent/30 rounded-lg">
              <Loader2 className="w-3 h-3 text-am-accent animate-spin" />
              <span className="text-xs font-mono text-am-accent">
                EXEC :: {currentTool}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function TraceNode({ call, isLast, isRunning }: { call: ToolCall, isLast: boolean, isRunning: boolean }) {
  const [isOpen, setIsOpen] = useState(false);

  // Auto-expand the last active node if running
  useEffect(() => {
    if (isLast && isRunning) setIsOpen(true);
  }, [isLast, isRunning]);

  return (
    <div className="relative pl-8 group">
      {/* Timeline Connector */}
      <div className="absolute -left-[29px] top-0.5 bg-am-primary p-1">
        <div className="w-3 h-3 bg-am-tertiary rounded-full ring-4 ring-am-primary group-hover:bg-am-accent transition-colors" />
      </div>

      <div
        className="bg-am-secondary border border-am-border rounded-lg overflow-hidden transition-all hover:border-am-accent/50 cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
      >
        {/* Header Row */}
        <div className="flex items-center justify-between px-3 py-2.5 bg-am-tertiary/50">
          <div className="flex items-center gap-3">
            <div className={`p-1.5 rounded-md ${getToolColor(call.name)}`}>
              <ToolIcon name={call.name} className="w-3.5 h-3.5" />
            </div>
            <span className="text-sm font-medium text-am-text-primary">
              {formatToolName(call.name)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <ResultChip name={call.name} args={call.args} />
            <ChevronDown className={`w-4 h-4 text-am-text-muted transition-transform ${isOpen ? 'rotate-180' : ''}`} />
          </div>
        </div>

        {/* Level 4: Deep Dive (Terminal) */}
        {isOpen && (
          <div className="border-t border-am-border bg-am-primary p-3 animate-fade-in">
            <div className="font-mono text-[10px] space-y-3">

              {/* Command Line */}
              <div className="flex items-center gap-2 text-am-text-muted border-b border-am-border pb-2">
                <Terminal className="w-3 h-3" />
                <span>PROTOCOL TRACE {call.id.slice(0, 8)}</span>
              </div>

              {/* Arguments */}
              <div className="space-y-1 pl-1">
                {Object.entries(call.args).map(([k, v]) => (
                  <div key={k} className="flex gap-2 items-start">
                    <span className="text-am-accent shrink-0">{k}:</span>
                    <span className="text-am-text-secondary break-all">
                      {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                    </span>
                  </div>
                ))}
              </div>

              {/* Status Footer */}
              <div className="flex items-center justify-between pt-2 text-am-text-muted border-t border-am-border mt-2">
                <div className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  <span>1.24s</span>
                </div>
                <div className="flex items-center gap-1 text-green-400">
                  <CheckCircle2 className="w-3 h-3" />
                  <span>VERIFIED</span>
                </div>
              </div>

            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Scientific Widgets (Chips)
// =============================================================================

function ResultChip({ name, args }: { name: string, args: any }) {
  if (name.includes('dock')) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-purple-500/20 text-purple-400 border border-purple-500/30">
        <Beaker className="w-3 h-3" /> VINA
      </span>
    );
  }
  if (name.includes('ped') || name.includes('fetch')) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">
        <Database className="w-3 h-3" /> {args.ped_id || 'DB'}
      </span>
    );
  }
  if (name.includes('cluster')) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-orange-500/20 text-orange-400 border border-orange-500/30">
        <Network className="w-3 h-3" /> T-SNE
      </span>
    );
  }
  return null;
}

// =============================================================================
// Icons & Formatting
// =============================================================================

function ToolIcon({ name, className }: { name: string, className: string }) {
  if (name.includes('ped') || name.includes('fetch')) return <Database className={className} />;
  if (name.includes('dock') || name.includes('ligand')) return <Beaker className={className} />;
  if (name.includes('cluster')) return <Network className={className} />;
  return <Terminal className={className} />;
}

function getToolColor(name: string) {
  if (name.includes('ped')) return 'bg-blue-500/20 text-blue-400';
  if (name.includes('dock')) return 'bg-purple-500/20 text-purple-400';
  if (name.includes('cluster')) return 'bg-orange-500/20 text-orange-400';
  return 'bg-am-tertiary text-am-text-muted';
}

const formatToolName = (name: string) => {
  if (name.includes('fetch_ped')) return "Query Protein Database";
  if (name.includes('prepare_ligand')) return "Prepare Ligand Topology";
  if (name.includes('cluster')) return "Cluster Conformations";
  if (name.includes('dock')) return "Ensemble Docking";
  return name.replace(/_/g, ' ');
};
