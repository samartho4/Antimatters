/**
 * Artifact Renderer — Document-first views
 *
 * Each artifact type renders as a standalone document that fills the panel.
 * No card wrappers, no "show details" toggles. Content is always visible.
 */

import React, { useState } from 'react';
import {
  CheckCircle2, Circle, Loader2, XCircle,
  ExternalLink, Link2
} from 'lucide-react';
import { Artifact } from '../services/aguiService';
import { MoleculeViewer } from '../MoleculeViewer';

// =============================================================================
// Shared primitives
// =============================================================================

const StatusIcon: React.FC<{ status: string; size?: string }> = ({ status, size = 'w-4 h-4' }) => {
  switch (status) {
    case 'completed': return <CheckCircle2 className={`${size} text-green-400`} />;
    case 'running':
    case 'in_progress': return <Loader2 className={`${size} text-am-accent animate-spin`} />;
    case 'failed': return <XCircle className={`${size} text-red-400`} />;
    default: return <Circle className={`${size} text-am-text-muted`} />;
  }
};

const ProgressBar: React.FC<{ done: number; total: number }> = ({ done, total }) => {
  const pct = total > 0 ? (done / total) * 100 : 0;
  return (
    <div>
      <div className="flex justify-between text-xs text-am-text-muted mb-1.5">
        <span>{done} of {total} complete</span>
        <span>{Math.round(pct)}%</span>
      </div>
      <div className="h-1.5 bg-am-tertiary rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${pct === 100 ? 'bg-green-400' : 'bg-am-accent'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
};

const SectionHeader: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="text-xs font-semibold text-am-text-muted uppercase tracking-wider mb-2.5">{children}</div>
);

const Divider: React.FC = () => <div className="border-t border-am-border/50 my-5" />;

const ExternalLinkBadge: React.FC<{ href: string; label: string }> = ({ href, label }) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className="inline-flex items-center gap-1 px-2 py-0.5 bg-am-tertiary hover:bg-am-accent/20 rounded text-xs text-am-accent transition-colors"
  >
    {label}
    <ExternalLink className="w-3 h-3" />
  </a>
);

// =============================================================================
// Task List
// =============================================================================

const TaskListView: React.FC<{ content: any }> = ({ content }) => {
  const tasks = content.tasks || [];
  const done = tasks.filter((t: any) => t.status === 'completed').length;
  const total = tasks.length;
  const currentTask = tasks.find((t: any) => t.status === 'in_progress');

  return (
    <div>
      {/* Live progress header */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-am-text-primary">
            {done === total && total > 0
              ? 'All tasks complete'
              : currentTask
                ? `Running: ${currentTask.step || currentTask.name}`
                : total > 0 ? `Step ${done + 1} of ${total}` : 'No tasks'}
          </span>
          {content.target_protein && (
            <span className="text-xs text-am-text-muted">{content.target_protein}</span>
          )}
        </div>
        <ProgressBar done={done} total={total} />
      </div>

      {/* Task rows — status drives left-border color */}
      <div className="space-y-1">
        {tasks.map((task: any, i: number) => {
          const isDone = task.status === 'completed';
          const isRunning = task.status === 'in_progress';
          return (
            <div
              key={i}
              className={`flex items-start gap-3 py-2.5 px-3 rounded-r-md border-l-2 ${
                isDone ? 'border-green-500 bg-green-500/5' :
                isRunning ? 'border-am-accent bg-am-accent/5' :
                'border-am-border/40'
              }`}
            >
              <div className="mt-0.5 flex-shrink-0">
                <StatusIcon status={task.status} />
              </div>
              <div className="flex-1 min-w-0">
                <div className={`text-sm ${isDone ? 'text-am-text-muted line-through' : 'text-am-text-primary font-medium'}`}>
                  {task.step || task.name}
                </div>
                {task.details && !isDone && (
                  <div className="text-xs text-am-text-muted mt-0.5">{task.details}</div>
                )}
                {task.result && (
                  <div className="text-xs text-green-400 mt-1 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> {task.result}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Ligand context — only if present */}
      {content.ligand_names?.length > 0 && (
        <>
          <Divider />
          <SectionHeader>Targets</SectionHeader>
          <div className="flex gap-2 flex-wrap">
            {content.ligand_names.map((name: string, i: number) => (
              <span key={i} className="px-2.5 py-1 bg-purple-500/15 border border-purple-500/30 text-purple-400 rounded text-xs font-medium">
                {name}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

// =============================================================================
// Protocol
// =============================================================================

const ProtocolView: React.FC<{ content: any }> = ({ content }) => {
  const c = content;
  return (
    <div>
      {/* Protein header row */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-base font-semibold text-am-text-primary">{c.protein_name || 'Unknown Protein'}</h3>
          <div className="text-xs text-am-text-muted mt-0.5 flex items-center gap-2">
            {c.ped_id && <span className="font-mono">{c.ped_id}</span>}
            {c.n_conformations && <><span>•</span><span>{c.n_conformations} conformations</span></>}
          </div>
        </div>
        {/* Validation status badge */}
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
          c.validation_status === 'validated' ? 'bg-green-500/15 text-green-400' :
          c.validation_status === 'failed' ? 'bg-red-500/15 text-red-400' :
          'bg-am-tertiary text-am-text-muted'
        }`}>
          <StatusIcon status={c.validation_status === 'validated' ? 'completed' : c.validation_status === 'failed' ? 'failed' : 'pending'} size="w-3.5 h-3.5" />
          <span>{c.validation_status || 'pending'}</span>
          {c.confidence_score != null && <span className="text-am-text-muted ml-1">({c.confidence_score}/10)</span>}
        </div>
      </div>

      {/* Binding site */}
      {c.binding_site_residues?.length > 0 && (
        <>
          <SectionHeader>Binding Site</SectionHeader>
          <div className="flex gap-2 flex-wrap mb-4">
            {c.binding_site_residues.map((r: number) => (
              <span key={r} className="px-2.5 py-1 bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 rounded-md text-sm font-mono">
                Y{r}
              </span>
            ))}
          </div>
        </>
      )}

      <Divider />

      {/* Ligands */}
      {c.ligands?.length > 0 && (
        <>
          <SectionHeader>Ligands ({c.ligands.length})</SectionHeader>
          <div className="space-y-3 mb-1">
            {c.ligands.map((lig: any, i: number) => (
              <div key={i} className="flex items-start gap-3 p-3 bg-am-secondary rounded-lg border border-am-border">
                {lig.image_base64 && (
                  <div className="bg-white rounded-md p-1.5 flex-shrink-0">
                    <img src={`data:image/png;base64,${lig.image_base64}`} alt={lig.name} className="h-16 w-auto" />
                  </div>
                )}
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-am-text-primary">{lig.name}</div>
                  <div className="text-xs text-am-text-muted font-mono mt-0.5 truncate">{lig.smiles}</div>
                  <div className="flex items-center gap-2 mt-2">
                    {lig.chembl_id && (
                      <ExternalLinkBadge href={`https://www.ebi.ac.uk/chembl/compound_report_card/${lig.chembl_id}`} label={lig.chembl_id} />
                    )}
                    {lig.source && <span className="text-xs text-am-text-muted">via {lig.source}</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Literature */}
      {(c.primary_citation || c.literature_evidence?.length > 0 || c.pmcid) && (
        <>
          <Divider />
          <SectionHeader>Literature</SectionHeader>
          {c.primary_citation && (
            <p className="text-sm text-am-text-secondary italic mb-2">{c.primary_citation}</p>
          )}
          {c.pmcid && (
            <div className="mb-2">
              <ExternalLinkBadge href={`https://www.ncbi.nlm.nih.gov/pmc/articles/${c.pmcid}`} label={c.pmcid} />
            </div>
          )}
          {c.literature_evidence?.map((ev: any, i: number) => (
            <div key={i} className="text-xs text-am-text-muted mb-1 italic">
              {typeof ev === 'string' ? ev : (ev?.title || ev?.description || '')}
            </div>
          ))}
        </>
      )}

      {/* Validation notes */}
      {c.validation_notes && (
        <>
          <Divider />
          <SectionHeader>Validation Notes</SectionHeader>
          <p className="text-sm text-am-text-secondary">{c.validation_notes}</p>
        </>
      )}

      {/* PED viewer link */}
      {c.visuals?.ped_viewer_url && (
        <>
          <Divider />
          <ExternalLinkBadge href={c.visuals.ped_viewer_url} label="View on PED" />
        </>
      )}
    </div>
  );
};

// =============================================================================
// Experiment Matrix
// =============================================================================

const ExperimentMatrixView: React.FC<{ content: any }> = ({ content }) => {
  const c = content;
  const results = c.ligand_results || c.ligands || [];
  const completed = results.filter((l: any) => l.status === 'completed').length;
  const total = results.length;

  return (
    <div>
      {/* Progress — only while docking is active */}
      {completed < total && (
        <div className="mb-4">
          <ProgressBar done={completed} total={total} />
        </div>
      )}

      {/* Results table — best row is highlighted */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-am-text-muted border-b border-am-border">
              <th className="text-left py-2 pr-3 font-medium">Ligand</th>
              <th className="text-center py-2 px-2 font-medium">Status</th>
              <th className="text-right py-2 px-2 font-medium">Energy</th>
              <th className="text-center py-2 px-2 font-medium">Cluster</th>
              <th className="text-center py-2 pl-2 font-medium">Residue</th>
            </tr>
          </thead>
          <tbody>
            {results.map((lig: any, i: number) => (
              <tr key={i} className={`border-b border-am-border/30 last:border-0 ${(lig.ligand_name || lig.name) === c.best_ligand ? 'bg-green-500/8' : ''}`}>
                <td className="py-2.5 pr-3 text-am-text-primary font-medium">{lig.ligand_name || lig.name}{(lig.ligand_name || lig.name) === c.best_ligand && <span className="ml-1.5 text-[10px] text-green-400 font-semibold">BEST</span>}</td>
                <td className="py-2.5 px-2">
                  <div className="flex justify-center">
                    <StatusIcon status={lig.status || 'pending'} />
                  </div>
                </td>
                <td className="py-2.5 px-2 text-right font-mono text-green-400 font-medium">
                  {lig.best_energy != null ? `${lig.best_energy.toFixed(1)}` : '—'}
                </td>
                <td className="py-2.5 px-2 text-center text-am-text-muted font-mono text-xs">
                  {lig.best_cluster != null ? `C${lig.best_cluster + 1}` : '—'}
                </td>
                <td className="py-2.5 pl-2 text-center text-cyan-400 font-mono text-xs">
                  {lig.best_residue ? `Y${lig.best_residue}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Interaction badges */}
      {results.some((r: any) => r.interaction_types?.length > 0) && (
        <>
          <Divider />
          <SectionHeader>Interactions</SectionHeader>
          <div className="space-y-2">
            {results.filter((r: any) => r.interaction_types?.length > 0).map((r: any, i: number) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-sm text-am-text-primary font-medium">{r.ligand_name || r.name}</span>
                <div className="flex gap-1.5 flex-wrap">
                  {r.interaction_types.map((t: string, j: number) => (
                    <span key={j} className="px-2 py-0.5 bg-purple-500/15 border border-purple-500/30 text-purple-400 rounded text-xs">
                      {t.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Ensemble info */}
      {c.n_clusters != null && (
        <>
          <Divider />
          <SectionHeader>Ensemble</SectionHeader>
          <div className="text-sm text-am-text-secondary">
            {c.n_clusters} representative clusters
            {c.representative_frames?.length > 0 && (
              <span className="text-am-text-muted"> • frames: {c.representative_frames.slice(0, 8).join(', ')}{c.representative_frames.length > 8 ? ` +${c.representative_frames.length - 8}` : ''}</span>
            )}
          </div>
        </>
      )}
    </div>
  );
};

// =============================================================================
// Discovery Report
// =============================================================================

const DiscoveryReportView: React.FC<{ content: any }> = ({ content }) => {
  const c = content;
  const rankings = c.ucb_rankings || c.ligand_rankings || [];
  const maxAbsEnergy = rankings.length > 0
    ? Math.max(...rankings.map((r: any) => Math.abs(r.best_energy || r.energy || 0)))
    : 1;

  return (
    <div>
      {/* Title with date */}
      {c.title && (
        <div className="mb-4 pb-3 border-b border-am-border/30">
          <h3 className="text-base font-semibold text-am-text-primary">{c.title}</h3>
          {c.created_at && <div className="text-xs text-am-text-muted mt-1">{new Date(c.created_at).toLocaleDateString()}</div>}
        </div>
      )}

      {/* Executive summary */}
      {c.executive_summary && (
        <p className="text-sm text-am-text-secondary leading-relaxed mb-5">{c.executive_summary}</p>
      )}

      {/* Rankings - clean list */}
      {rankings.length > 0 && (
        <>
          <div className="text-xs font-medium text-am-text-muted uppercase tracking-wide mb-2">Ranked by Binding Affinity</div>
          <div className="space-y-0.5">
            {rankings.map((r: any, i: number) => {
              const energy = r.best_energy || r.energy_mean || r.energy || 0;
              return (
                <div key={i} className={`flex items-center gap-3 py-2 px-2.5 rounded ${i === 0 ? 'bg-am-accent/8' : ''}`}>
                  <span className="text-xs text-am-text-muted w-4">{i + 1}</span>
                  <span className="text-sm text-am-text-primary flex-1">{r.ligand_name || r.name}</span>
                  <span className="text-sm font-mono text-green-400 tabular-nums">{energy.toFixed(1)}</span>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Key interactions */}
      {c.key_interactions?.length > 0 && (
        <>
          <Divider />
          <SectionHeader>Key Interactions</SectionHeader>
          <div className="space-y-1.5">
            {c.key_interactions.map((int: any, i: number) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                <span className="px-2 py-0.5 bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 rounded font-mono text-xs">{int.residue}</span>
                <span className="text-am-text-muted">—</span>
                <span className="text-am-text-secondary">{int.type?.replace(/_/g, ' ')}</span>
                <span className="text-am-text-muted">with</span>
                <span className="text-am-text-primary font-medium">{int.ligand}</span>
                {int.occupancy != null && <span className="text-am-text-muted text-xs">({int.occupancy}%)</span>}
              </div>
            ))}
          </div>
        </>
      )}

      {/* SAR insights — fall back to derived insights when agent left them empty */}
      {(() => {
        const sarInsights = c.sar_insights?.length > 0 ? c.sar_insights :
          rankings.length >= 2 ? [{
            title: 'Binding-energy spread',
            description: `Top ligand "${rankings[0]?.ligand_name || rankings[0]?.name}" binds ${
              (Math.abs(rankings[0]?.best_energy || rankings[0]?.energy || 0) -
               Math.abs(rankings[rankings.length - 1]?.best_energy || rankings[rankings.length - 1]?.energy || 0))
              .toFixed(1)
            } kcal/mol tighter than the weakest candidate — a ${
              (Math.abs(rankings[0]?.best_energy || rankings[0]?.energy || 0) -
               Math.abs(rankings[rankings.length - 1]?.best_energy || rankings[rankings.length - 1]?.energy || 0)) > 1.5 ? 'significant' : 'moderate'
            } structure-activity difference worth optimising.`
          }] : [];

        return sarInsights.length > 0 ? (
          <>
            <Divider />
            <SectionHeader>SAR Insights</SectionHeader>
            <div className="space-y-2">
              {sarInsights.map((ins: any, i: number) => (
                <div key={i} className="p-2.5 bg-am-secondary rounded-lg border border-am-border">
                  <div className="text-sm font-medium text-am-accent">{ins.title || ins.pattern_type}</div>
                  {ins.description && <p className="text-xs text-am-text-muted mt-0.5">{ins.description}</p>}
                </div>
              ))}
            </div>
          </>
        ) : null;
      })()}

      {/* Recommendations — fall back to derived suggestions when agent left them empty */}
      {(() => {
        const recs = c.recommendations?.length > 0 ? c.recommendations :
          rankings.length > 0 ? [
            { recommendation: `Prioritise "${rankings[0]?.ligand_name || rankings[0]?.name}" for lead optimisation — strongest binder at ${(rankings[0]?.best_energy || rankings[0]?.energy || 0).toFixed(1)} kcal/mol.`, priority: 'high' },
            { recommendation: 'Run ADMET predictions on the top-ranked ligand before advancing to wet-lab validation.', priority: 'high' },
            { recommendation: `Perform similarity search around the top ligand SMILES to expand the candidate space.`, priority: 'medium' },
          ] : [];

        return recs.length > 0 ? (
          <>
            <Divider />
            <SectionHeader>Recommendations</SectionHeader>
            <div className="space-y-2">
              {recs.map((rec: any, i: number) => (
                <div key={i} className="flex items-start gap-2 text-sm">
                  <span className="text-am-accent font-bold mt-0.5 flex-shrink-0">→</span>
                  <span className="text-am-text-secondary flex-1">
                    {typeof rec === 'string' ? rec : rec.recommendation}
                  </span>
                  {rec.priority && (
                    <span className={`text-xs px-1.5 py-0.5 rounded flex-shrink-0 ${
                      rec.priority === 'high' ? 'bg-red-500/15 text-red-400' :
                      rec.priority === 'medium' ? 'bg-yellow-500/15 text-yellow-400' :
                      'bg-am-tertiary text-am-text-muted'
                    }`}>
                      {rec.priority}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </>
        ) : null;
      })()}
    </div>
  );
};

// =============================================================================
// Evolution Trace (Knowledge Graph)
// =============================================================================

const EvolutionTraceView: React.FC<{ content: any }> = ({ content }) => {
  const c = content;
  const rawEntities: any[] = c.entities || [];
  const relationships = c.relationships || [];

  // Deduplicate entities by (entity_type, name) — the agent emits one entry
  // from ChEMBL lookup and a second from docking; merge their properties so
  // each ligand/protein appears exactly once with the union of metadata.
  const entityMap = new Map<string, any>();
  for (const e of rawEntities) {
    const key = `${e.entity_type || 'Other'}::${e.name || ''}`;
    const existing = entityMap.get(key);
    if (existing) {
      // Merge properties and external_ids — later values win on conflict
      existing.properties = { ...existing.properties, ...e.properties };
      existing.external_ids = { ...existing.external_ids, ...e.external_ids };
    } else {
      entityMap.set(key, { ...e });
    }
  }
  const entities = Array.from(entityMap.values());

  // Group entities by type
  const grouped = entities.reduce((acc: Record<string, any[]>, e: any) => {
    const type = e.entity_type || 'Other';
    if (!acc[type]) acc[type] = [];
    acc[type].push(e);
    return acc;
  }, {});

  return (
    <div>
      {/* Graph summary */}
      <div className="flex items-center gap-3 text-sm text-am-text-muted mb-4">
        <span className="font-mono">{entities.length} entities</span>
        <span>•</span>
        <span className="font-mono">{relationships.length} connections</span>
      </div>

      {/* Entities grouped by type */}
      {Object.entries(grouped).map(([type, items]: [string, any[]]) => (
        <div key={type} className="mb-4">
          <SectionHeader>{type} ({items.length})</SectionHeader>
          <div className="space-y-2">
            {items.map((e: any, i: number) => (
              <div key={i} className="p-2.5 bg-am-secondary rounded-lg border border-am-border">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-am-text-primary text-sm">{e.name}</span>
                  {e.properties?.best_energy != null && (
                    <span className="text-green-400 font-mono text-xs">{Number(e.properties.best_energy).toFixed(1)} kcal/mol</span>
                  )}
                </div>
                <div className="flex gap-2 mt-1.5 flex-wrap">
                  {e.external_ids?.uniprot && (
                    <ExternalLinkBadge href={`https://www.uniprot.org/uniprotkb/${e.external_ids.uniprot}`} label={`UniProt: ${e.external_ids.uniprot}`} />
                  )}
                  {e.external_ids?.chembl && (
                    <ExternalLinkBadge href={`https://www.ebi.ac.uk/chembl/compound_report_card/${e.external_ids.chembl}`} label={`ChEMBL: ${e.external_ids.chembl}`} />
                  )}
                  {e.external_ids?.ped && (
                    <ExternalLinkBadge href={`https://proteinensemble.org/entries/${e.external_ids.ped.slice(0, 8)}`} label={`PED: ${e.external_ids.ped}`} />
                  )}
                </div>
                {e.properties?.smiles && (
                  <div className="text-xs text-am-text-muted font-mono mt-1 truncate">{e.properties.smiles}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Relationship edges */}
      {relationships.length > 0 && (
        <>
          <Divider />
          <SectionHeader>Relationships</SectionHeader>
          <div className="space-y-1">
            {relationships.slice(0, 12).map((rel: any, i: number) => (
              <div key={i} className="flex items-center gap-2 text-xs py-1 text-am-text-secondary">
                <span className="font-mono text-am-text-muted">{rel.subject_id?.slice(-8)}</span>
                <span className="text-am-accent flex items-center gap-1">
                  <Link2 className="w-3 h-3" />
                  {rel.predicate?.replace(/_/g, ' ')}
                </span>
                <span className="font-mono text-am-text-muted">{rel.object_id?.slice(-8)}</span>
                {rel.properties?.best_energy != null && (
                  <span className="text-green-400 ml-auto">{Number(rel.properties.best_energy).toFixed(1)}</span>
                )}
              </div>
            ))}
            {relationships.length > 12 && (
              <div className="text-xs text-am-text-muted">+{relationships.length - 12} more</div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

// =============================================================================
// 3D Structure (protein + ligand docking visualization)
// =============================================================================

const StructureView: React.FC<{ content: any }> = ({ content }) => {
  const interactions = content.interactions ? {
    hbond_residues: content.interactions.hbond_residues || [],
    hydrophobic_residues: content.interactions.hydrophobic_residues || [],
    aromatic_residues: content.interactions.aromatic_residues || [],
  } : undefined;

  const bindingSite = content.highlight_residues || content.metadata?.binding_site || [125, 133, 136];

  // If we have raw PDB data → use our interactive viewer.
  // Otherwise fall back to the self-contained HTML viewer the agent generated
  // (rendered in an isolated iframe so its own scripts execute cleanly).
  const hasPdbData = !!(content.pdb_data && content.pdb_data.trim().length > 10);

  return (
    <div>
      <div className="rounded-lg overflow-hidden border border-am-border" style={{ height: '360px' }}>
        {hasPdbData ? (
          <MoleculeViewer
            pdbData={content.pdb_data}
            interactions={interactions}
            highlightResidues={bindingSite}
          />
        ) : content.html_viewer ? (
          <iframe
            title="3D Structure Viewer"
            srcDoc={content.html_viewer}
            className="w-full h-full border-0"
            sandbox="allow-scripts"
          />
        ) : (
          <div className="w-full h-full bg-am-secondary flex items-center justify-center">
            <span className="text-xs text-am-text-muted">No 3D data available</span>
          </div>
        )}
      </div>

      {/* Context chips */}
      <div className="mt-3 flex gap-2 flex-wrap">
        {content.ligand_name && (
          <span className="px-2.5 py-1 bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 rounded text-xs font-medium">
            {content.ligand_name}
          </span>
        )}
        {bindingSite.map((r: number) => (
          <span key={r} className="px-2 py-0.5 bg-am-tertiary text-am-text-muted rounded text-xs font-mono">Y{r}</span>
        ))}
      </div>

      {/* Interaction legend — only when data is present */}
      {interactions && (interactions.hbond_residues!.length > 0 || interactions.hydrophobic_residues!.length > 0 || interactions.aromatic_residues!.length > 0) && (
        <>
          <Divider />
          <SectionHeader>Interactions</SectionHeader>
          <div className="space-y-1.5">
            {interactions.hbond_residues!.length > 0 && (
              <div className="flex items-center gap-2 text-sm">
                <span className="w-3 h-3 rounded-full bg-blue-500 flex-shrink-0" />
                <span className="text-am-text-secondary">H-bonds</span>
                <span className="text-am-text-muted font-mono text-xs">Y{interactions.hbond_residues!.join(', Y')}</span>
              </div>
            )}
            {interactions.hydrophobic_residues!.length > 0 && (
              <div className="flex items-center gap-2 text-sm">
                <span className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0" />
                <span className="text-am-text-secondary">Hydrophobic</span>
                <span className="text-am-text-muted font-mono text-xs">Y{interactions.hydrophobic_residues!.join(', Y')}</span>
              </div>
            )}
            {interactions.aromatic_residues!.length > 0 && (
              <div className="flex items-center gap-2 text-sm">
                <span className="w-3 h-3 rounded-full bg-purple-500 flex-shrink-0" />
                <span className="text-am-text-secondary">Aromatic</span>
                <span className="text-am-text-muted font-mono text-xs">Y{interactions.aromatic_residues!.join(', Y')}</span>
              </div>
            )}
          </div>
        </>
      )}

      <div className="mt-3 text-xs text-am-text-muted italic">
        Drag to rotate • Scroll to zoom • Use pencil to annotate
      </div>
    </div>
  );
};

// =============================================================================
// Molecule Suggestions (generated drug candidates)
// =============================================================================

const MoleculeSuggestionsView: React.FC<{ content: any }> = ({ content }) => {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const suggestions: any[] = content.suggestions || [];
  const selected = suggestions[selectedIdx];

  return (
    <div>
      {content.title && (
        <h3 className="text-base font-semibold text-am-text-primary mb-1">{content.title}</h3>
      )}
      {(content.generation_method || content.modification_type) && (
        <div className="text-xs text-am-text-muted mb-3">
          {content.generation_method && <span>Method: {content.generation_method.replace(/_/g, ' ')}</span>}
          {content.generation_method && content.modification_type && <span> • </span>}
          {content.modification_type && <span>Type: {content.modification_type}</span>}
        </div>
      )}

      {/* 3D viewer for selected molecule — or fallback when pdb_block is absent */}
      <div className="rounded-lg overflow-hidden border border-am-border mb-3" style={{ height: '260px' }}>
        {selected?.pdb_block ? (
          <MoleculeViewer pdbData={selected.pdb_block} highlightResidues={[]} />
        ) : selected ? (
          <div className="w-full h-full bg-am-secondary flex flex-col items-center justify-center gap-2 px-4">
            <span className="text-xs text-am-text-muted">3D coordinates not generated</span>
            {selected.smiles && (
              <span className="text-[10px] text-am-text-muted font-mono truncate max-w-full">{selected.smiles}</span>
            )}
            {selected.error_3d && (
              <span className="text-[10px] text-red-400">{selected.error_3d}</span>
            )}
          </div>
        ) : null}
      </div>

      {/* Molecule list — click to switch 3D view */}
      <div className="space-y-2">
        {suggestions.map((s: any, i: number) => (
          <button
            key={i}
            onClick={() => setSelectedIdx(i)}
            className={`w-full text-left p-2.5 rounded-lg border transition-colors ${
              i === selectedIdx
                ? 'bg-am-accent/10 border-am-accent/50'
                : 'bg-am-secondary border-am-border hover:border-am-accent/30'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-am-text-primary">{s.name || `Molecule ${i + 1}`}</span>
              <div className="flex gap-1.5">
                {s.drug_like && (
                  <span className="px-1.5 py-0.5 bg-green-500/15 text-green-400 rounded text-[10px] font-semibold">DRUG-LIKE</span>
                )}
                {s.valid === false && (
                  <span className="px-1.5 py-0.5 bg-red-500/15 text-red-400 rounded text-[10px]">INVALID</span>
                )}
              </div>
            </div>
            <div className="text-xs text-am-text-muted font-mono mt-0.5 truncate">{s.smiles}</div>
            <div className="flex gap-3 mt-1.5 text-xs text-am-text-muted">
              {s.molecular_weight != null && <span>MW: {typeof s.molecular_weight === 'number' ? s.molecular_weight.toFixed(1) : s.molecular_weight}</span>}
              {s.logp != null && <span>LogP: {typeof s.logp === 'number' ? s.logp.toFixed(2) : s.logp}</span>}
              {s.hbd != null && <span>HBD: {s.hbd}</span>}
              {s.hba != null && <span>HBA: {s.hba}</span>}
            </div>
            {s.rationale && (
              <div className="text-xs text-am-accent mt-1 italic">{s.rationale}</div>
            )}
          </button>
        ))}
      </div>

      <div className="mt-3 text-xs text-am-text-muted italic">
        Click a molecule to view in 3D • Use pencil to annotate
      </div>
    </div>
  );
};

// =============================================================================
// Generic fallback
// =============================================================================

const GenericView: React.FC<{ content: any }> = ({ content }) => {
  if (!content) return <div className="text-sm text-am-text-muted italic">No content available</div>;

  if (content.executive_summary) {
    return <p className="text-sm text-am-text-secondary whitespace-pre-line">{content.executive_summary}</p>;
  }

  // Filter out metadata and nested objects, show meaningful key-value pairs
  const entries = Object.entries(content)
    .filter(([k, v]) => k !== 'metadata' && typeof v !== 'object' && v != null)
    .slice(0, 12);

  if (entries.length === 0) {
    return <div className="text-sm text-am-text-muted italic">No displayable content</div>;
  }

  return (
    <div className="space-y-2 text-sm">
      {entries.map(([k, v]) => (
        <div key={k} className="flex gap-3">
          <span className="text-am-text-muted font-medium min-w-[90px] flex-shrink-0">{k.replace(/_/g, ' ')}</span>
          <span className="text-am-text-secondary">{String(v).slice(0, 80)}</span>
        </div>
      ))}
    </div>
  );
};

// =============================================================================
// Main entry point
// =============================================================================

interface ArtifactRendererProps {
  artifact: Artifact;
  onStructureClick?: (pdbData: string) => void;
}

export const ArtifactRenderer: React.FC<ArtifactRendererProps> = ({ artifact, onStructureClick }) => {
  if (!artifact?.type) return null;

  const c = artifact.content as any;

  switch (artifact.type) {
    case 'task_list':
      return <TaskListView content={c} />;
    case 'protocol':
      return <ProtocolView content={c} />;
    case 'experiment_matrix':
      return <ExperimentMatrixView content={c} />;
    case 'discovery_report':
      return <DiscoveryReportView content={c} />;
    case 'evolution_trace':
      return <EvolutionTraceView content={c} />;
    case 'structure_3d':
      return <StructureView content={c} />;
    case 'molecule_suggestions':
      return <MoleculeSuggestionsView content={c} />;
    default:
      return <GenericView content={c} />;
  }
};

export default ArtifactRenderer;
