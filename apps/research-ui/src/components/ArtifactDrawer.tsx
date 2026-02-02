import React from 'react';
import {
  FileText,
  Database,
  BarChart,
  Box,
  Clock,
  ChevronRight,
  Search
} from 'lucide-react';
import { Artifact } from '../services/aguiService';

interface ArtifactDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  artifacts: Artifact[];
  onSelect: (artifact: Artifact) => void;
}

export function ArtifactDrawer({ isOpen, onClose, artifacts, onSelect }: ArtifactDrawerProps) {
  return (
    <div
      className={`fixed inset-y-0 right-0 w-80 bg-am-secondary border-l border-am-border shadow-2xl transform transition-transform duration-300 ease-in-out z-50 flex flex-col ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-am-border bg-am-tertiary">
        <h3 className="text-sm font-semibold text-am-text-primary flex items-center gap-2">
          <Database className="w-4 h-4 text-am-text-muted" />
          Artifact Library
        </h3>
        <button onClick={onClose} className="p-1 hover:bg-am-primary rounded transition-colors">
          <ChevronRight className="w-4 h-4 text-am-text-muted" />
        </button>
      </div>

      {/* Search */}
      <div className="p-3 border-b border-am-border">
        <div className="relative">
          <Search className="absolute left-2.5 top-2 w-3.5 h-3.5 text-am-text-muted" />
          <input
            type="text"
            placeholder="Filter artifacts..."
            className="w-full pl-8 pr-3 py-1.5 text-xs bg-am-primary border border-am-border rounded-md focus:border-am-accent focus:outline-none text-am-text-primary placeholder-am-text-muted"
          />
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {artifacts.length === 0 ? (
          <div className="text-center text-xs text-am-text-muted py-10">No artifacts yet</div>
        ) : (
          artifacts.map((artifact) => (
            <ArtifactItem key={artifact.id} artifact={artifact} onSelect={onSelect} />
          ))
        )}
      </div>

      {/* Footer Status */}
      <div className="p-3 border-t border-am-border bg-am-tertiary text-[10px] text-am-text-muted flex justify-between">
        <span>{artifacts.length} Items</span>
        <span>Synced to Disk</span>
      </div>
    </div>
  );
}

function ArtifactItem({ artifact, onSelect }: { artifact: Artifact, onSelect: (a: Artifact) => void }) {
  const Icon = getIcon(artifact.type);
  const time = new Date(artifact.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <button
      onClick={() => onSelect(artifact)}
      className="w-full flex items-start gap-3 p-3 hover:bg-am-tertiary rounded-lg text-left group transition-colors border border-transparent hover:border-am-border"
    >
      <div className="p-2 bg-am-primary border border-am-border rounded-md group-hover:border-am-accent/50 group-hover:text-am-accent transition-colors">
        <Icon className="w-4 h-4 text-am-text-muted group-hover:text-am-accent" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-xs font-medium text-am-text-primary truncate">
          {getArtifactTitle(artifact)}
        </div>
        <div className="flex items-center gap-2 mt-1 text-[10px] text-am-text-muted">
          <span className="font-mono bg-am-primary px-1 rounded">{artifact.type.replace(/_/g, '-')}</span>
          <span className="flex items-center gap-0.5">
            <Clock className="w-2.5 h-2.5" /> {time}
          </span>
        </div>
      </div>
    </button>
  );
}

function getIcon(type: string) {
  if (type.includes('task')) return FileText;
  if (type.includes('dock') || type.includes('result')) return BarChart;
  if (type.includes('structure') || type.includes('ligand')) return Box;
  return Database;
}

function getArtifactTitle(artifact: Artifact) {
  if (artifact.type === 'scientific_experiment') return artifact.content.title;
  if (artifact.type === 'task_list') return "Protocol Checklist";
  if (artifact.type === 'docking_result') return "Docking Analysis";
  if (artifact.type === 'structure_3d') return `Structure: ${artifact.content.metadata?.ped_id || 'Unknown'}`;
  if (artifact.type === 'ligand_svg') return `Ligand: ${artifact.content.name}`;
  return artifact.id;
}
