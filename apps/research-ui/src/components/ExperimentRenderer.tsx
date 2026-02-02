import React, { useState } from 'react';
import {
  Beaker,
  FileText,
  Database,
  BarChart,
  Layers,
  Clock,
  CheckCircle2,
  Loader2
} from 'lucide-react';
import { Artifact, ArtifactType } from '../services/aguiService';
import { ArtifactRenderer } from './ArtifactRenderer';
import { FeedbackOverlay } from './FeedbackOverlay';

interface ExperimentRendererProps {
  experiment: Artifact;
  artifacts: Artifact[];
  onFeedback: (id: string, comment: string) => void;
  onStructureClick: (pdb: string) => void;
}

export const ExperimentRenderer: React.FC<ExperimentRendererProps> = ({
  experiment,
  artifacts,
  onFeedback,
  onStructureClick
}) => {
  // Defensive destructuring
  const content = experiment?.content || {};
  const title = content.title || "Untitled Experiment";
  const sections = content.sections || {};
  const createdAt = experiment?.created_at ? new Date(experiment.created_at) : new Date();

  // Helper to find artifacts by ID list
  const getArtifacts = (ids: string[] = []) =>
    (ids || []).map(id => artifacts.find(a => a.id === id)).filter(Boolean) as Artifact[];

  const planArtifacts = getArtifacts(sections.plan);
  const materialArtifacts = getArtifacts(sections.materials);
  const resultArtifacts = getArtifacts(sections.results);
  const validationArtifacts = getArtifacts(sections.validation);

  return (
    <div className="h-full flex flex-col bg-am-primary overflow-hidden">
      {/* Experiment Header */}
      <div className="flex-shrink-0 bg-am-secondary border-b border-am-border px-6 py-4">
        <div className="flex items-center gap-3 mb-1">
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <Beaker className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-am-text-primary">{title}</h2>
            <div className="flex items-center gap-3 text-xs text-am-text-muted">
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {createdAt.toLocaleTimeString()}
              </span>
              <span className="flex items-center gap-1 px-2 py-0.5 bg-green-500/10 text-green-400 rounded-full border border-green-500/20">
                <Loader2 className="w-3 h-3 animate-spin" />
                Running
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Notebook Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-8">

        {/* SECTION 1: PROTOCOL */}
        <section>
          <SectionHeader icon={FileText} title="Experimental Protocol" count={planArtifacts.length} />
          <div className="space-y-4">
            {planArtifacts.map(artifact => (
              <FeedbackOverlay key={artifact.id} artifactId={artifact.id} onFeedbackSubmit={onFeedback}>
                <ArtifactRenderer artifact={artifact} />
              </FeedbackOverlay>
            ))}
          </div>
        </section>

        {/* SECTION 2: MATERIALS & DATA */}
        {materialArtifacts.length > 0 && (
          <section>
            <SectionHeader icon={Database} title="Data & Materials" count={materialArtifacts.length} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {materialArtifacts.map(artifact => (
                <FeedbackOverlay key={artifact.id} artifactId={artifact.id} onFeedbackSubmit={onFeedback}>
                  <ArtifactRenderer
                    artifact={artifact}
                    onStructureClick={onStructureClick}
                  />
                </FeedbackOverlay>
              ))}
            </div>
          </section>
        )}

        {/* SECTION 3: RESULTS & ANALYSIS */}
        {resultArtifacts.length > 0 && (
          <section>
            <SectionHeader icon={BarChart} title="Analysis Results" count={resultArtifacts.length} />
            <div className="space-y-4">
              {resultArtifacts.map(artifact => (
                <FeedbackOverlay key={artifact.id} artifactId={artifact.id} onFeedbackSubmit={onFeedback}>
                  <ArtifactRenderer artifact={artifact} />
                </FeedbackOverlay>
              ))}
            </div>
          </section>
        )}

        {/* SECTION 4: VALIDATION */}
        {validationArtifacts.length > 0 && (
          <section>
            <SectionHeader icon={Layers} title="Validation & Literature" count={validationArtifacts.length} />
            <div className="grid grid-cols-1 gap-4">
              {validationArtifacts.map(artifact => (
                <FeedbackOverlay key={artifact.id} artifactId={artifact.id} onFeedbackSubmit={onFeedback}>
                  <ArtifactRenderer artifact={artifact} />
                </FeedbackOverlay>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
};

const SectionHeader = ({ icon: Icon, title, count }: any) => (
  <div className="flex items-center gap-2 mb-4 pb-2 border-b border-am-border">
    <Icon className="w-4 h-4 text-am-text-muted" />
    <h3 className="text-sm font-semibold text-am-text-secondary uppercase tracking-wider">{title}</h3>
    <span className="ml-auto text-xs bg-am-tertiary px-2 py-0.5 rounded-full text-am-text-muted font-mono">
      {count} Items
    </span>
  </div>
);
