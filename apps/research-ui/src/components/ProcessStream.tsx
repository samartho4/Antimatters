import React, { useState, useEffect } from 'react';
import { 
  Loader2, 
  CheckCircle2, 
  Database, 
  Search, 
  Beaker, 
  BrainCircuit, 
  ChevronDown, 
  ChevronRight,
  Server,
  FileCode
} from 'lucide-react';
import { ToolCall } from '../services/aguiService';

interface ProcessStreamProps {
  toolCalls: ToolCall[];
  isRunning: boolean;
  currentTool: string | null;
}

export function ProcessStream({ toolCalls, isRunning, currentTool }: ProcessStreamProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  
  // Auto-expand when new tools are added
  useEffect(() => {
    if (isRunning) setIsExpanded(true);
  }, [toolCalls.length, isRunning]);

  // Antimatters Context-Aware Engine
  const [thought, setThought] = useState("Initializing computational reality...");
  
  useEffect(() => {
    if (!isRunning) return;
    
    // Map tools to Antimatters philosophy
    const getDeepThought = (tool: string | null) => {
      if (!tool) return "Evolving information complexity...";
      
      const t = tool.toLowerCase();
      if (t.includes('fetch') || t.includes('ped')) return "Sourcing ground truth from the chaos...";
      if (t.includes('search') || t.includes('chembl')) return "Navigating the landscape of possibility...";
      if (t.includes('ligand') || t.includes('prepare')) return "Structuring information into matter...";
      if (t.includes('cluster') || t.includes('tsne')) return "Unfolding the geometry of disorder...";
      if (t.includes('dock') || t.includes('simulate')) return "Simulating matter via computational energy...";
      if (t.includes('analyz') || t.includes('interaction')) return "Deciphering the fundamental reality...";
      
      return "Computing the wave function of logic...";
    };

    setThought(getDeepThought(currentTool));
  }, [currentTool, isRunning]);

  if (toolCalls.length === 0 && !isRunning) return null;

  return (
    <div className="my-4 border border-gray-200 bg-white rounded-xl shadow-sm overflow-hidden transition-all duration-300">
      {/* Header */}
      <button 
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 border-b border-gray-100 hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-3">
          {isRunning ? (
            <div className="relative flex items-center justify-center">
              <div className="absolute inset-0 bg-blue-400 rounded-full animate-ping opacity-20"></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
            </div>
          ) : (
            <CheckCircle2 className="w-4 h-4 text-green-600" />
          )}
          <span className="text-sm font-medium text-gray-700 italic">
            {isRunning ? thought : 'Simulation convergence achieved.'}
          </span>
        </div>
        {isExpanded ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronRight className="w-4 h-4 text-gray-400" />}
      </button>

      {/* Stream Content */}
      {isExpanded && (
        <div className="p-2 bg-white space-y-1">
          {toolCalls.map((call, idx) => {
            const isLast = idx === toolCalls.length - 1;
            const isComplete = !isLast || !isRunning;
            
            return (
              <ProcessStep 
                key={call.id} 
                call={call} 
                isComplete={isComplete} 
                isActive={isLast && isRunning} 
              />
            );
          })}
          
          {/* Active "Thinking" State */}
          {isRunning && !currentTool && (
            <div className="flex items-center gap-3 px-3 py-2 text-gray-500 animate-pulse">
              <BrainCircuit className="w-4 h-4" />
              <span className="text-xs">Reasoning...</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ProcessStep({ call, isComplete, isActive }: { call: ToolCall, isComplete: boolean, isActive: boolean }) {
  // Determine Icon based on tool name
  const getIcon = (name: string) => {
    if (name.includes('verify') || name.includes('valid') || name.includes('assess')) return CheckCircle2;
    if (name.includes('ped') || name.includes('fetch')) return Database;
    if (name.includes('search') || name.includes('google')) return Search;
    if (name.includes('dock') || name.includes('ligand')) return Beaker;
    if (name.includes('cluster')) return Server;
    return FileCode;
  };

  const Icon = getIcon(call.name);
  const isVerification = call.name.includes('verify') || call.name.includes('assess');

  return (
    <div className={`group flex items-start gap-3 px-3 py-2 rounded-lg transition-colors ${isActive ? 'bg-blue-50/50' : 'hover:bg-gray-50'}`}>
      <div className={`mt-0.5 p-1 rounded-md ${
        isActive ? 'bg-blue-100 text-blue-600' : 
        isVerification ? 'bg-green-100 text-green-600' : 
        'bg-gray-100 text-gray-500'
      }`}>
        <Icon className="w-3.5 h-3.5" />
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span className={`text-sm font-medium ${isActive ? 'text-blue-700' : 'text-gray-700'}`}>
            {formatToolName(call.name)}
          </span>
          {isComplete && <CheckCircle2 className="w-3.5 h-3.5 text-green-500 opacity-0 group-hover:opacity-100 transition-opacity" />}
        </div>
        
        {/* Arguments Preview */}
        {call.args && Object.keys(call.args).length > 0 && (
          <div className="mt-1 text-xs text-gray-500 font-mono bg-gray-50 px-2 py-1 rounded border border-gray-100 truncate">
             {Object.entries(call.args).map(([k, v]) => `${k}: ${v}`).join(', ')}
          </div>
        )}
      </div>
    </div>
  );
}

// Helper
const formatToolName = (name: string) => {
  return name
    .replace(/_/g, ' ')
    .replace(/bc /g, '')
    .replace(/\b\w/g, l => l.toUpperCase());
};
