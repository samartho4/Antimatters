import React, { useState } from 'react';
import {
  Beaker,
  FileText,
  Database,
  BarChart,
  Layers,
  Clock,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { Artifact } from '../services/aguiService';
import { ArtifactRenderer } from './ArtifactRenderer';
import { FeedbackOverlay } from './FeedbackOverlay';

interface ExperimentRendererProps {
  experiment: Artifact;
  artifacts: Artifact[];
  onFeedback: (id: string, comment: string) => void;
  onStructureClick: (pdb: string) => void;
}

type TabKey = 'protocol' | 'data' | 'results' | 'validation';

const TABS: { key: TabKey; label: string; icon: React.ElementType }[] = [
  { key: 'protocol', label: 'Protocol', icon: FileText },
  { key: 'data', label: 'Data', icon: Database },
  { key: 'results', label: 'Results', icon: BarChart },
  { key: 'validation', label: 'Validation', icon: Layers },
];

export const ExperimentRenderer: React.FC<ExperimentRendererProps> = ({
  experiment,
  artifacts,
  onFeedback,
  onStructureClick
}) => {
  const [activeTab, setActiveTab] = useState<TabKey>('protocol');
  const [selectedArtifactIndex, setSelectedArtifactIndex] = useState(0);

  const content = experiment?.content || {};
  const title = content.title || "Untitled Experiment";
  const sections = content.sections || {};
  const createdAt = experiment?.created_at ? new Date(experiment.created_at) : new Date();

  const getArtifacts = (ids: string[] = []) =>
    (ids || []).map(id => artifacts.find(a => a.id === id)).filter(Boolean) as Artifact[];

  const tabArtifacts: Record<TabKey, Artifact[]> = {
    protocol: getArtifacts(sections.plan),
    data: getArtifacts(sections.materials),
    results: getArtifacts(sections.results),
    validation: getArtifacts(sections.validation),
  };

  const currentArtifacts = tabArtifacts[activeTab];

  // Count non-empty tabs
  const tabCounts = TABS.map(t => ({ ...t, count: tabArtifacts[t.key].length }));

  // Reset selected index when tab changes
  React.useEffect(() => {
    setSelectedArtifactIndex(0);
  }, [activeTab]);

  // Get currently selected artifact
  const selectedArtifact = currentArtifacts[selectedArtifactIndex];

  return (
    <div className="h-full flex flex-col bg-am-primary overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 bg-am-secondary border-b border-am-border px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="p-1.5 bg-purple-500/20 rounded-lg">
            <Beaker className="w-4 h-4 text-purple-400" />
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-sm font-semibold text-am-text-primary truncate">{title}</h2>
            <div className="flex items-center gap-2 text-[10px] text-am-text-muted">
              <Clock className="w-3 h-3" />
              {createdAt.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>

      {/* Tabs - Like Claude/ChatGPT artifact selector */}
      <nav 
        className="flex-shrink-0 border-b border-am-border bg-am-secondary/50" 
        role="tablist"
        aria-label="Experiment sections"
      >
        <div className="flex px-2">
          {tabCounts.map(({ key, label, icon: Icon, count }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`flex items-center gap-1.5 px-3 py-3 text-xs font-medium border-b-2 transition-colors min-h-[44px] ${
                activeTab === key
                  ? 'border-am-accent text-am-accent'
                  : 'border-transparent text-am-text-muted hover:text-am-text-secondary'
              }`}
              role="tab"
              aria-selected={activeTab === key}
              aria-controls={`${key}-panel`}
              tabIndex={activeTab === key ? 0 : -1}
            >
              <Icon className="w-3.5 h-3.5" aria-hidden="true" />
              <span>{label}</span>
              {count > 0 && (
                <span className={`ml-1 px-1.5 py-0.5 text-[10px] rounded-full ${
                  activeTab === key ? 'bg-am-accent/20' : 'bg-am-tertiary'
                }`} aria-label={`${count} items`}>
                  {count}
                </span>
              )}
            </button>
          ))}
        </div>
      </nav>

      {/* Single Artifact View - ONE artifact at a time */}
      <div 
        className="flex-1 overflow-y-auto p-4"
        role="tabpanel"
        id={`${activeTab}-panel`}
        aria-labelledby={`${activeTab}-tab`}
      >
        {currentArtifacts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-12 h-12 rounded-xl bg-am-tertiary flex items-center justify-center mb-3">
              {TABS.find(t => t.key === activeTab)?.icon &&
                React.createElement(TABS.find(t => t.key === activeTab)!.icon, { className: "w-6 h-6 text-am-text-muted" })}
            </div>
            <p className="text-sm text-am-text-muted">No {activeTab} artifacts yet</p>
            <p className="text-xs text-am-text-muted mt-1">They will appear as the experiment progresses</p>
          </div>
        ) : (
          <FeedbackOverlay key={selectedArtifact.id} artifactId={selectedArtifact.id} onFeedbackSubmit={onFeedback}>
            <ArtifactRenderer artifact={selectedArtifact} onStructureClick={onStructureClick} />
          </FeedbackOverlay>
        )}
      </div>

      {/* Artifact Selector - bottom navigation like Claude artifacts */}
      {currentArtifacts.length > 1 && (
        <div className="flex-shrink-0 border-t border-am-border bg-am-secondary px-4 py-2">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setSelectedArtifactIndex(Math.max(0, selectedArtifactIndex - 1))}
              disabled={selectedArtifactIndex === 0}
              className="p-2 rounded hover:bg-am-tertiary disabled:opacity-30 disabled:cursor-not-allowed transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
              aria-label={`Previous artifact (${selectedArtifactIndex} of ${currentArtifacts.length})`}
            >
              <ChevronLeft className="w-4 h-4 text-am-text-muted" />
            </button>

            <div className="flex items-center gap-2" role="status" aria-live="polite">
              <span className="text-xs text-am-text-muted">
                {selectedArtifactIndex + 1} / {currentArtifacts.length}
              </span>
              <span className="text-[10px] text-am-text-muted px-2 py-0.5 bg-am-tertiary rounded">
                {selectedArtifact.type.replace(/_/g, ' ')}
              </span>
            </div>

            <button
              onClick={() => setSelectedArtifactIndex(Math.min(currentArtifacts.length - 1, selectedArtifactIndex + 1))}
              disabled={selectedArtifactIndex === currentArtifacts.length - 1}
              className="p-2 rounded hover:bg-am-tertiary disabled:opacity-30 disabled:cursor-not-allowed transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
              aria-label={`Next artifact (${selectedArtifactIndex + 2} of ${currentArtifacts.length})`}
            >
              <ChevronRight className="w-4 h-4 text-am-text-muted" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
