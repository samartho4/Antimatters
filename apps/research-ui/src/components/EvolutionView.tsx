/**
 * Knowledge Graph View - Neo4j-powered Scientific Knowledge Browser
 * ==================================================================
 * TWO MODES:
 * 1. Graph View (default): Neo4j NVL visualization of knowledge graph from Neo4j backend
 * 2. Tree View: Optional artifact browser for administrative access
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { InteractiveNvlWrapper, NVL } from '@neo4j-nvl/base';
import {
  Sparkles,
  X,
  ChevronDown,
  ChevronRight,
  FileText,
  Beaker,
  BarChart,
  Layers,
  Database,
  Network,
  Atom,
  BookOpen,
  ListTodo,
  FlaskConical,
  Search,
  Filter,
  List,
  GitBranch,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Info,
} from 'lucide-react';
import { Artifact } from '../services/aguiService';
import { ArtifactRenderer } from './ArtifactRenderer';

interface Conversation {
  id: string;
  title: string;
  message_count?: number;
  created_at: string;
}

interface EvolutionViewProps {
  artifacts: Artifact[];
  conversations: Conversation[];
  onClose: () => void;
}

// Graph Types
interface Entity {
  entity_id: string;
  entity_type: string;
  level: number;
  name: string;
  properties: Record<string, any>;
  external_ids: Record<string, string>;
}

interface Relationship {
  relationship_id: string;
  subject_id: string;
  predicate: string;
  object_id: string;
  properties: Record<string, any>;
}

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  entity: Entity;
  degree: number;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  relationship: Relationship;
}

const GRAPH_COLORS = {
  experimental: '#3b82f6',
  reference: '#10b981',
  vocabulary: '#f59e0b',
  edge: {
    DOCKS_TO: '#a855f7',
    INTERACTS_WITH: '#ec4899',
    CORRESPONDS_TO: '#6b7280',
    HAS_PROPERTY: '#14b8a6',
  },
};

const NODE_SIZE = {
  Protein: 12,
  Ligand: 10,
  Residue: 6,
  default: 8,
};

// Artifact type to icon and color mapping
const getArtifactMeta = (type: string): { icon: React.ElementType; color: string; label: string } => {
  const typeMap: Record<string, { icon: React.ElementType; color: string; label: string }> = {
    task_list: { icon: ListTodo, color: 'text-blue-400', label: 'Task List' },
    protocol: { icon: FileText, color: 'text-indigo-400', label: 'Protocol' },
    structure_3d: { icon: Atom, color: 'text-green-400', label: '3D Structure' },
    docking_result: { icon: Beaker, color: 'text-purple-400', label: 'Docking Results' },
    interaction_map: { icon: Network, color: 'text-cyan-400', label: 'Interactions' },
    cluster_visualization: { icon: Layers, color: 'text-orange-400', label: 'Clusters' },
    experiment_matrix: { icon: BarChart, color: 'text-pink-400', label: 'Experiment Matrix' },
    discovery_report: { icon: Sparkles, color: 'text-yellow-400', label: 'Discovery Report' },
    evolution_trace: { icon: Network, color: 'text-emerald-400', label: 'Evolution Trace' },
    publication_figure: { icon: BarChart, color: 'text-rose-400', label: 'Figure' },
    literature_result: { icon: BookOpen, color: 'text-blue-400', label: 'Literature' },
    ligand_svg: { icon: FlaskConical, color: 'text-amber-400', label: 'Ligand' },
    scientific_experiment: { icon: Beaker, color: 'text-purple-400', label: 'Experiment' },
  };

  return typeMap[type] || { icon: FileText, color: 'text-am-text-muted', label: 'Artifact' };
};

export const EvolutionView: React.FC<EvolutionViewProps> = ({ artifacts, conversations, onClose }) => {
  const [viewMode, setViewMode] = useState<'tree' | 'graph'>('graph');
  const [selected, setSelected] = useState<Artifact | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Graph-specific state
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphLink | null>(null);
  const [graphFilters, setGraphFilters] = useState({
    experimental: true,
    reference: true,
    vocabulary: true,
    DOCKS_TO: true,
    INTERACTS_WITH: true,
    CORRESPONDS_TO: true,
    HAS_PROPERTY: true,
  });
  const [showFilters, setShowFilters] = useState(false);
  const nvlRef = useRef<NVL | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Graph data from Neo4j backend
  const [graphData, setGraphData] = useState<{ entities: Entity[]; relationships: Relationship[] }>({
    entities: [],
    relationships: []
  });

  // Group artifacts by run_id (conversation), hide evolution_trace (they're just graph data)
  const grouped = artifacts
    .filter(a => a.type !== 'evolution_trace')
    .reduce((acc, art) => {
      const id = art.run_id || 'unknown';
      (acc[id] = acc[id] || []).push(art);
      return acc;
    }, {} as Record<string, Artifact[]>);

  // Fetch graph from Neo4j backend when in graph mode
  useEffect(() => {
    if (viewMode === 'graph') {
      fetch('http://localhost:8002/evolution/graph')
        .then(res => res.json())
        .then(data => {
          if (data.success) {
            setGraphData({
              entities: data.entities || [],
              relationships: data.relationships || []
            });
          }
        })
        .catch(err => console.error('Failed to fetch graph:', err));
    }
  }, [viewMode]);

  // Calculate node degrees for graph
  const nodeDegrees = useMemo(() => {
    const degrees = new Map<string, number>();
    graphData.entities.forEach(e => degrees.set(e.entity_id, 0));
    graphData.relationships.forEach(r => {
      degrees.set(r.subject_id, (degrees.get(r.subject_id) || 0) + 1);
      degrees.set(r.object_id, (degrees.get(r.object_id) || 0) + 1);
    });
    return degrees;
  }, [graphData]);

  // Apply graph filters
  const filteredGraphData = useMemo(() => {
    const levelNames = ['experimental', 'reference', 'vocabulary'] as const;
    const activeEntities = graphData.entities.filter(e => {
      const levelName = levelNames[e.level - 1];
      if (!graphFilters[levelName]) return false;
      if (searchQuery && !e.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    });

    const activeEntityIds = new Set(activeEntities.map(e => e.entity_id));
    const activeRelationships = graphData.relationships.filter(r => {
      if (!graphFilters[r.predicate as keyof typeof graphFilters]) return false;
      return activeEntityIds.has(r.subject_id) && activeEntityIds.has(r.object_id);
    });

    return { entities: activeEntities, relationships: activeRelationships };
  }, [graphData, graphFilters, searchQuery]);

  // Convert to D3 graph format
  const { nodes, links } = useMemo(() => {
    const nodes: GraphNode[] = filteredGraphData.entities.map(entity => ({
      id: entity.entity_id,
      entity,
      degree: nodeDegrees.get(entity.entity_id) || 0,
    }));

    const links: GraphLink[] = filteredGraphData.relationships.map(rel => ({
      source: rel.subject_id,
      target: rel.object_id,
      relationship: rel,
    }));

    return { nodes, links };
  }, [filteredGraphData, nodeDegrees]);

  // Smart defaults: expand first, select first artifact
  useEffect(() => {
    const entries = Object.entries(grouped);
    if (entries.length > 0 && !selected) {
      const [firstId, firstArts] = entries[0];
      setExpanded({ [firstId]: true });
      if (firstArts.length > 0) {
        setSelected(firstArts[0]);
      }
    }
  }, [artifacts.length]);

  // Keyboard navigation: Escape to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // Initialize Neo4j NVL when graph data changes
  useEffect(() => {
    if (viewMode !== 'graph' || !containerRef.current || nodes.length === 0) return;

    // Transform data to Neo4j NVL format
    const nvlNodes = nodes.map(n => ({
      id: n.id,
      caption: n.entity.name,
      size: NODE_SIZE[n.entity.entity_type as keyof typeof NODE_SIZE] || NODE_SIZE.default,
      color: n.entity.level === 1 ? GRAPH_COLORS.experimental :
             n.entity.level === 2 ? GRAPH_COLORS.reference : GRAPH_COLORS.vocabulary,
    }));

    const nvlRelationships = links.map((l, i) => ({
      id: `rel_${i}`,
      from: typeof l.source === 'object' ? l.source.id : l.source,
      to: typeof l.target === 'object' ? l.target.id : l.target,
      caption: l.relationship.predicate,
    }));

    // Initialize NVL instance
    const nvl = new NVL(containerRef.current, nvlNodes, nvlRelationships, {
      layout: 'force',
      nodeSize: 20,
      edgeDirectional: true,
      backgroundColor: '#0a0a0a',
      nodeOutlineColor: '#ffffff',
      nodeOutlineWidth: 2,
    });

    nvl.onNodeClick((node: any) => {
      const graphNode = nodes.find(n => n.id === node.id);
      if (graphNode) {
        setSelectedNode(graphNode);
        setSelectedEdge(null);
      }
    });

    nvl.onRelationshipClick((rel: any) => {
      const graphLink = links.find((l, i) => `rel_${i}` === rel.id);
      if (graphLink) {
        setSelectedEdge(graphLink);
        setSelectedNode(null);
      }
    });

    nvlRef.current = nvl;

    return () => {
      if (nvlRef.current) {
        nvlRef.current.destroy();
      }
    };
  }, [viewMode, nodes, links]);

  // NVL zoom controls
  const handleZoomIn = () => {
    if (nvlRef.current) {
      nvlRef.current.zoomIn();
    }
  };

  const handleZoomOut = () => {
    if (nvlRef.current) {
      nvlRef.current.zoomOut();
    }
  };

  const handleZoomReset = () => {
    if (nvlRef.current) {
      nvlRef.current.fit();
    }
  };

  const getTitle = (id: string) => {
    const conv = conversations.find(c => c.id === id);
    if (conv) return conv.title;
    return `Experiment ${id.slice(0, 8)}`;
  };

  const getName = (art: Artifact) => {
    return (typeof art.content?.title === 'string' ? art.content.title : null) ||
           (typeof art.content?.name === 'string' ? art.content.name : null) ||
           getArtifactMeta(art.type).label;
  };

  const getTimestamp = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit' 
      });
    } catch {
      return '';
    }
  };

  // Filter artifacts by search and type
  const filteredGrouped = Object.entries(grouped).reduce((acc, [id, arts]) => {
    const filteredArts = arts.filter(art => {
      const matchesSearch = searchQuery === '' || 
        getName(art).toLowerCase().includes(searchQuery.toLowerCase()) ||
        art.type.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesType = filterType === 'all' || art.type === filterType;
      return matchesSearch && matchesType;
    });
    if (filteredArts.length > 0) {
      acc[id] = filteredArts;
    }
    return acc;
  }, {} as Record<string, Artifact[]>);

  // Get unique artifact types for filter
  const artifactTypes = Array.from(new Set(artifacts.map(a => a.type)));

  const toggleExpanded = (id: string) => {
    setExpanded(prev => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="h-full flex bg-am-primary" role="main" aria-label="Evolution Knowledge Browser">
      {/* Left Panel: Navigation / Controls */}
      <div className="w-80 bg-am-secondary border-r border-am-border flex flex-col">
        {/* Header with Mode Toggle */}
        <div className="flex-shrink-0 border-b border-am-border">
          <div className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-am-accent" aria-hidden="true" />
              <h2 className="text-sm font-semibold text-am-text-primary">Knowledge Graph</h2>
              <span className="px-2 py-0.5 text-xs bg-am-tertiary text-am-text-muted rounded-full">
                {artifacts.length}
              </span>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 hover:bg-am-tertiary rounded transition-colors"
              aria-label="Close Evolution view"
            >
              <X className="w-4 h-4 text-am-text-muted" />
            </button>
          </div>
          {/* Mode Toggle */}
          <div className="px-4 pb-3 flex gap-1">
            <button
              onClick={() => setViewMode('tree')}
              className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 text-xs font-medium rounded transition-all ${
                viewMode === 'tree'
                  ? 'bg-am-accent text-white'
                  : 'bg-am-tertiary text-am-text-secondary hover:text-am-text-primary'
              }`}
            >
              <List className="w-3.5 h-3.5" />
              <span>Tree</span>
            </button>
            <button
              onClick={() => setViewMode('graph')}
              className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 text-xs font-medium rounded transition-all ${
                viewMode === 'graph'
                  ? 'bg-am-accent text-white'
                  : 'bg-am-tertiary text-am-text-secondary hover:text-am-text-primary'
              }`}
            >
              <GitBranch className="w-3.5 h-3.5" />
              <span>Graph</span>
            </button>
          </div>
        </div>

        {/* Search & Filter */}
        <div className="flex-shrink-0 px-3 py-3 space-y-2 border-b border-am-border">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-am-text-muted" aria-hidden="true" />
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={viewMode === 'graph' ? 'Search entities...' : 'Search artifacts...'}
              className="w-full pl-9 pr-3 py-2 bg-am-primary border border-am-border rounded-lg text-xs text-am-text-primary placeholder-am-text-muted focus:outline-none focus:ring-2 focus:ring-am-accent focus:border-transparent"
              aria-label="Search"
            />
          </div>
          {viewMode === 'tree' && (
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-am-text-muted" aria-hidden="true" />
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="w-full pl-9 pr-3 py-2 bg-am-primary border border-am-border rounded-lg text-xs text-am-text-primary focus:outline-none focus:ring-2 focus:ring-am-accent focus:border-transparent appearance-none cursor-pointer"
                aria-label="Filter by artifact type"
              >
                <option value="all">All Types</option>
                {artifactTypes.map(type => (
                  <option key={type} value={type}>{getArtifactMeta(type).label}</option>
                ))}
              </select>
            </div>
          )}
          {viewMode === 'graph' && (
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium bg-am-primary border border-am-border rounded-lg text-am-text-secondary hover:text-am-text-primary transition-colors"
            >
              <div className="flex items-center gap-2">
                <Filter className="w-3.5 h-3.5" />
                <span>Graph Filters</span>
              </div>
              {showFilters ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            </button>
          )}
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {viewMode === 'tree' ? (
            /* Tree View */
            <div role="tree" aria-label="Artifact tree">
              {Object.entries(filteredGrouped).length === 0 ? (
                <div className="text-center py-8">
                  <div className="text-am-text-muted text-xs">No artifacts found</div>
                </div>
              ) : (
                Object.entries(filteredGrouped).map(([id, arts]) => (
                  <div key={id} role="treeitem" aria-expanded={expanded[id]}>
                    <button
                      onClick={() => toggleExpanded(id)}
                      className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-am-text-primary hover:bg-am-tertiary rounded-lg transition-colors group"
                      aria-label={`${expanded[id] ? 'Collapse' : 'Expand'} ${getTitle(id)}`}
                    >
                      {expanded[id] ? (
                        <ChevronDown className="w-4 h-4 text-am-text-muted flex-shrink-0" aria-hidden="true" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-am-text-muted flex-shrink-0" aria-hidden="true" />
                      )}
                      <span className="truncate flex-1 text-left">{getTitle(id)}</span>
                      <span className="text-xs text-am-text-muted bg-am-tertiary px-2 py-0.5 rounded-full">
                        {arts.length}
                      </span>
                    </button>
                    {expanded[id] && (
                      <div className="ml-6 mt-1 space-y-0.5" role="group">
                        {arts.map(art => {
                          const meta = getArtifactMeta(art.type);
                          const Icon = meta.icon;
                          const isSelected = selected?.id === art.id;
                          return (
                            <button
                              key={art.id}
                              onClick={() => setSelected(art)}
                              className={`w-full flex items-start gap-2 px-3 py-2 text-xs rounded-lg transition-all ${
                                isSelected
                                  ? 'bg-am-accent/20 text-am-accent border border-am-accent/30'
                                  : 'text-am-text-secondary hover:bg-am-tertiary hover:text-am-text-primary border border-transparent'
                              }`}
                              role="treeitem"
                              aria-label={`${getName(art)} - ${meta.label}`}
                              aria-selected={isSelected}
                            >
                              <Icon className={`w-4 h-4 flex-shrink-0 mt-0.5 ${meta.color}`} aria-hidden="true" />
                              <div className="flex-1 text-left min-w-0">
                                <div className="truncate font-medium">{getName(art)}</div>
                                <div className="flex items-center gap-2 mt-0.5">
                                  <span className="text-[10px] text-am-text-muted px-1.5 py-0.5 bg-am-primary rounded">
                                    {meta.label}
                                  </span>
                                  {art.created_at && (
                                    <span className="text-[10px] text-am-text-muted">
                                      {getTimestamp(art.created_at)}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          ) : (
            /* Graph View Controls */
            <>
              {showFilters && (
                <div className="space-y-3">
                  <div className="text-[10px] font-medium text-am-text-muted uppercase tracking-wider">Levels</div>
                  {(['experimental', 'reference', 'vocabulary'] as const).map(level => (
                    <label key={level} className="flex items-center gap-2 text-xs text-am-text-secondary cursor-pointer">
                      <input
                        type="checkbox"
                        checked={graphFilters[level]}
                        onChange={(e) => setGraphFilters({ ...graphFilters, [level]: e.target.checked })}
                        className="rounded border-am-border"
                      />
                      <span className="capitalize">{level}</span>
                      <span className="text-[10px] text-am-text-muted">
                        ({graphData.entities.filter(e =>
                          e.level === (level === 'experimental' ? 1 : level === 'reference' ? 2 : 3)
                        ).length})
                      </span>
                    </label>
                  ))}
                  <div className="text-[10px] font-medium text-am-text-muted uppercase tracking-wider mt-3">Relationships</div>
                  {(['DOCKS_TO', 'INTERACTS_WITH', 'CORRESPONDS_TO', 'HAS_PROPERTY'] as const).map(type => (
                    <label key={type} className="flex items-center gap-2 text-xs text-am-text-secondary cursor-pointer">
                      <input
                        type="checkbox"
                        checked={graphFilters[type]}
                        onChange={(e) => setGraphFilters({ ...graphFilters, [type]: e.target.checked })}
                        className="rounded border-am-border"
                      />
                      <span className="text-[10px]">{type.replace(/_/g, ' ')}</span>
                      <span className="text-[10px] text-am-text-muted">
                        ({graphData.relationships.filter(r => r.predicate === type).length})
                      </span>
                    </label>
                  ))}
                </div>
              )}
              <div className="mt-4 space-y-2">
                <div className="text-[10px] font-medium text-am-text-muted uppercase tracking-wider">Legend</div>
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-xs text-am-text-secondary">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: GRAPH_COLORS.experimental }} />
                    <span>Experimental</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-am-text-secondary">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: GRAPH_COLORS.reference }} />
                    <span>Reference</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-am-text-secondary">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: GRAPH_COLORS.vocabulary }} />
                    <span>Vocabulary</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer Stats */}
        <div className="flex-shrink-0 border-t border-am-border px-4 py-3">
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-lg font-bold text-am-text-primary">{conversations.length}</div>
              <div className="text-[10px] text-am-text-muted">Experiments</div>
            </div>
            <div>
              <div className="text-lg font-bold text-am-accent">{artifacts.length}</div>
              <div className="text-[10px] text-am-text-muted">Artifacts</div>
            </div>
            <div>
              <div className="text-lg font-bold text-am-text-primary">{artifactTypes.length}</div>
              <div className="text-[10px] text-am-text-muted">Types</div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel */}
      {viewMode === 'tree' ? (
        /* Tree Mode: Artifact Display */
        <div className="flex-1 overflow-y-auto bg-am-primary" role="region" aria-label="Artifact detail view">
          {selected ? (
            <div className="max-w-5xl mx-auto p-8">
              <div className="mb-6 pb-4 border-b border-am-border">
                <div className="flex items-start gap-3 mb-2">
                  {(() => {
                    const meta = getArtifactMeta(selected.type);
                    const Icon = meta.icon;
                    return (
                      <div className={`p-2 rounded-lg bg-am-secondary border border-am-border`}>
                        <Icon className={`w-5 h-5 ${meta.color}`} aria-hidden="true" />
                      </div>
                    );
                  })()}
                  <div className="flex-1 min-w-0">
                    <h3 className="text-xl font-semibold text-am-text-primary mb-1">{getName(selected)}</h3>
                    <div className="flex flex-wrap items-center gap-3 text-xs text-am-text-muted">
                      <span className="px-2 py-1 bg-am-secondary border border-am-border rounded">
                        {getArtifactMeta(selected.type).label}
                      </span>
                      <span>•</span>
                      <span>{selected.created_at ? new Date(selected.created_at).toLocaleString() : 'Unknown date'}</span>
                      <span>•</span>
                      <span className="font-mono">ID: {selected.id.slice(0, 8)}</span>
                    </div>
                  </div>
                </div>
              </div>
              <ArtifactRenderer artifact={selected} onStructureClick={() => {}} />
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-center px-8">
              <div>
                <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-am-secondary border border-am-border flex items-center justify-center">
                  <Sparkles className="w-8 h-8 text-am-accent" aria-hidden="true" />
                </div>
                <h3 className="text-lg font-medium text-am-text-primary mb-2">Select an artifact to view</h3>
                <p className="text-sm text-am-text-muted max-w-md">
                  Browse through your research history using the tree navigation on the left.
                </p>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Graph Mode: Canvas + Properties */
        <div className="flex-1 flex">
          {graphData.entities.length === 0 ? (
            <div className="flex-1 flex items-center justify-center bg-am-primary">
              <div className="text-center max-w-md px-4">
                <GitBranch className="w-16 h-16 text-am-text-muted mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-am-text-primary mb-2">No Knowledge Graph Yet</h3>
                <p className="text-sm text-am-text-secondary">
                  Run docking experiments to build your scientific knowledge graph.
                  Each experiment adds entities and relationships.
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Graph Canvas */}
              <div className="flex-1 flex flex-col">
                <div className="flex items-center justify-between px-4 py-2 border-b border-am-border bg-am-secondary">
                  <div className="text-xs text-am-text-muted">
                    {filteredGraphData.entities.length} nodes, {filteredGraphData.relationships.length} edges
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={handleZoomOut} className="p-1.5 text-am-text-muted hover:text-am-text-primary hover:bg-am-tertiary rounded transition-colors" title="Zoom out">
                      <ZoomOut className="w-4 h-4" />
                    </button>
                    <button onClick={handleZoomIn} className="p-1.5 text-am-text-muted hover:text-am-text-primary hover:bg-am-tertiary rounded transition-colors" title="Zoom in">
                      <ZoomIn className="w-4 h-4" />
                    </button>
                    <button onClick={handleZoomReset} className="p-1.5 text-am-text-muted hover:text-am-text-primary hover:bg-am-tertiary rounded transition-colors" title="Fit to screen">
                      <Maximize2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                <div className="flex-1 relative" ref={containerRef} style={{ background: '#0a0a0a' }}>
                  {!selectedNode && !selectedEdge && (
                    <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-am-secondary/90 backdrop-blur-sm border border-am-border rounded-lg px-4 py-2">
                      <p className="text-xs text-am-text-muted text-center">
                        Click nodes/edges to inspect • Drag to reposition • Scroll to zoom
                      </p>
                    </div>
                  )}
                </div>
              </div>
              {/* Properties Panel */}
              {(selectedNode || selectedEdge) && (
                <div className="w-80 border-l border-am-border bg-am-secondary flex flex-col">
                  <div className="flex items-center justify-between px-4 py-3 border-b border-am-border">
                    <div className="flex items-center gap-2">
                      <Info className="w-4 h-4 text-am-accent" />
                      <span className="text-sm font-medium text-am-text-primary">
                        {selectedNode ? 'Entity Details' : 'Relationship Details'}
                      </span>
                    </div>
                    <button onClick={() => { setSelectedNode(null); setSelectedEdge(null); }} className="p-1 text-am-text-muted hover:text-am-text-primary rounded transition-colors">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {selectedNode && (
                      <>
                        <div>
                          <div className="text-xs font-medium text-am-text-muted mb-1">Name</div>
                          <div className="text-sm text-am-text-primary font-medium">{selectedNode.entity.name}</div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-am-text-muted mb-1">Type</div>
                          <div className="text-sm text-am-text-primary">{selectedNode.entity.entity_type}</div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-am-text-muted mb-1">Level</div>
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full" style={{
                              backgroundColor: selectedNode.entity.level === 1 ? GRAPH_COLORS.experimental :
                                selectedNode.entity.level === 2 ? GRAPH_COLORS.reference : GRAPH_COLORS.vocabulary
                            }} />
                            <span className="text-sm text-am-text-primary capitalize">
                              {selectedNode.entity.level === 1 ? 'Experimental' :
                                selectedNode.entity.level === 2 ? 'Reference' : 'Vocabulary'}
                            </span>
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-am-text-muted mb-1">Connections</div>
                          <div className="text-sm text-am-text-primary">{selectedNode.degree}</div>
                        </div>
                        {Object.keys(selectedNode.entity.properties).length > 0 && (
                          <div>
                            <div className="text-xs font-medium text-am-text-muted mb-2">Properties</div>
                            <div className="space-y-1">
                              {Object.entries(selectedNode.entity.properties).map(([key, value]) => (
                                <div key={key} className="flex justify-between text-xs gap-2">
                                  <span className="text-am-text-muted">{key}:</span>
                                  <span className="text-am-text-primary font-mono text-right break-all">
                                    {typeof value === 'number' ? value.toFixed(3) : String(value).substring(0, 50)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        {Object.keys(selectedNode.entity.external_ids).length > 0 && (
                          <div>
                            <div className="text-xs font-medium text-am-text-muted mb-2">External IDs</div>
                            <div className="space-y-1">
                              {Object.entries(selectedNode.entity.external_ids).map(([key, value]) => (
                                <div key={key} className="flex justify-between text-xs">
                                  <span className="text-am-text-muted uppercase">{key}:</span>
                                  <span className="text-am-accent font-mono">{value}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </>
                    )}
                    {selectedEdge && (
                      <>
                        <div>
                          <div className="text-xs font-medium text-am-text-muted mb-1">Relationship</div>
                          <div className="text-sm font-medium" style={{ color: GRAPH_COLORS.edge[selectedEdge.relationship.predicate as keyof typeof GRAPH_COLORS.edge] }}>
                            {selectedEdge.relationship.predicate.replace(/_/g, ' ')}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-am-text-muted mb-1">From</div>
                          <div className="text-sm text-am-text-primary">
                            {nodes.find(n => n.id === selectedEdge.relationship.subject_id)?.entity.name || 'Unknown'}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-am-text-muted mb-1">To</div>
                          <div className="text-sm text-am-text-primary">
                            {nodes.find(n => n.id === selectedEdge.relationship.object_id)?.entity.name || 'Unknown'}
                          </div>
                        </div>
                        {Object.keys(selectedEdge.relationship.properties).length > 0 && (
                          <div>
                            <div className="text-xs font-medium text-am-text-muted mb-2">Properties</div>
                            <div className="space-y-1">
                              {Object.entries(selectedEdge.relationship.properties).map(([key, value]) => (
                                <div key={key} className="flex justify-between text-xs">
                                  <span className="text-am-text-muted">{key}:</span>
                                  <span className="text-am-text-primary font-mono">
                                    {typeof value === 'number' ? value.toFixed(3) : String(value)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};
