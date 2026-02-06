/**
 * Antimatters Sidebar
 * ===================
 * Information → Computation → Evolution
 */

import React, { useState, useEffect } from 'react';
import {
  Plus,
  ChevronDown,
  ChevronRight,
  Wrench,
  FileText,
  BookOpen,
  BarChart3,
  MessageSquare,
  Settings,
  HelpCircle,
  Inbox,
  FolderOpen,
  Atom,
  Database,
  FlaskConical,
  Sparkles,
  PanelLeftClose,
  PanelLeft,
} from 'lucide-react';
import { Artifact, Tool } from '../../services/aguiService';

interface Conversation {
  id: string;
  title: string;
  message_count?: number;
  created_at: string;
  updated_at: string;
}

interface KnowledgeItem {
  id: string;
  type: string;
  title: string;
  content: string;
  tags: string[];
  created_at: string;
}

interface SidebarProps {
  tools: Tool[];
  artifacts: Artifact[];
  conversations: Conversation[];
  knowledge?: KnowledgeItem[];
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onSelectTool: (tool: Tool) => void;
  onSelectArtifact: (artifact: Artifact) => void;
  onSelectKnowledge?: (item: KnowledgeItem) => void;
  activeConversationId?: string;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export function Sidebar({
  tools,
  artifacts,
  conversations,
  knowledge = [],
  onNewChat,
  onSelectConversation,
  onSelectTool,
  onSelectArtifact,
  onSelectKnowledge,
  activeConversationId,
  isCollapsed = false,
  onToggleCollapse,
}: SidebarProps) {
  const [expandedSections, setExpandedSections] = useState({
    experiments: true,
    information: true,
    evolution: true,
  });

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  // Group tools by category
  const toolsByCategory = tools.reduce((acc, tool) => {
    const category = tool.category || 'general';
    if (!acc[category]) acc[category] = [];
    acc[category].push(tool);
    return acc;
  }, {} as Record<string, Tool[]>);


  if (isCollapsed) {
    return (
      <aside 
        className="w-16 md:w-20 bg-am-secondary border-r border-am-border flex flex-col items-center py-4 gap-4"
        aria-label="Collapsed navigation sidebar"
      >
        <button
          onClick={onToggleCollapse}
          className="p-3 text-am-text-secondary hover:text-am-text-primary hover:bg-am-tertiary rounded-lg transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
          aria-label="Expand sidebar"
          aria-expanded="false"
        >
          <PanelLeft className="w-5 h-5" />
        </button>
        <button
          onClick={onNewChat}
          className="p-3 bg-am-accent text-white rounded-lg hover:bg-am-accent-hover transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
          aria-label="Start new research session"
        >
          <Plus className="w-5 h-5" />
        </button>
        <div className="flex-1" />
        <button 
          className="p-3 text-am-text-muted hover:text-am-text-secondary rounded-lg transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
          aria-label="Open settings"
        >
          <Settings className="w-5 h-5" />
        </button>
      </aside>
    );
  }

  return (
    <aside 
      className="w-60 sm:w-64 md:w-72 lg:w-80 bg-am-secondary border-r border-am-border flex flex-col h-full"
      aria-label="Main navigation sidebar"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-am-border">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-am-accent rounded-lg flex items-center justify-center" aria-hidden="true">
            <Atom className="w-4 h-4 text-white" />
          </div>
          <h1 className="text-sm font-semibold text-am-text-primary">Antimatters</h1>
        </div>
        <button
          onClick={onToggleCollapse}
          className="p-2 text-am-text-muted hover:text-am-text-secondary hover:bg-am-tertiary rounded transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
          aria-label="Collapse sidebar"
          aria-expanded="true"
        >
          <PanelLeftClose className="w-4 h-4" />
        </button>
      </div>

      {/* New Research Button */}
      <div className="px-3 py-3">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-3 py-3 text-sm text-am-text-secondary hover:text-am-text-primary hover:bg-am-tertiary rounded-lg transition-colors min-h-[44px]"
          aria-label="Start new research session"
        >
          <Plus className="w-4 h-4" aria-hidden="true" />
          <span>New Research</span>
        </button>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto px-2 space-y-1">

        {/* EXPERIMENTS Section (conversations = experiments) */}
        <SidebarSection
          title="Experiments"
          icon={FlaskConical}
          isExpanded={expandedSections.experiments}
          onToggle={() => toggleSection('experiments')}
          badge={conversations.length > 0 ? conversations.length : undefined}
        >
          {conversations.length === 0 ? (
            <div className="px-3 py-2 text-xs text-am-text-muted">No experiments yet</div>
          ) : (
            conversations.map(conv => (
              <button
                key={conv.id}
                onClick={() => onSelectConversation(conv.id)}
                className={`w-full flex flex-col items-start gap-1 px-3 py-1.5 text-xs rounded-md transition-colors ${
                  activeConversationId === conv.id
                    ? 'bg-am-accent/20 text-am-accent'
                    : 'text-am-text-secondary hover:text-am-text-primary hover:bg-am-tertiary'
                }`}
              >
                <div className="flex items-center gap-2 w-full">
                  <MessageSquare className="w-3 h-3 flex-shrink-0" />
                  <span className="truncate flex-1">{conv.title}</span>
                  {conv.message_count && (
                    <span className="text-[10px] text-am-text-muted">{conv.message_count}</span>
                  )}
                </div>
                <div className="text-[10px] text-am-text-muted pl-5">
                  {new Date(conv.updated_at).toLocaleDateString()}
                </div>
              </button>
            ))
          )}
        </SidebarSection>

        {/* INFORMATION Section */}
        <SidebarSection
          title="Information"
          icon={Database}
          isExpanded={expandedSections.information}
          onToggle={() => toggleSection('information')}
        >
          {/* Tools */}
          <div className="space-y-0.5">
            <div className="px-3 py-1 text-[10px] font-medium text-am-text-muted uppercase tracking-wider">
              Tools ({tools.length})
            </div>
            {Object.entries(toolsByCategory).slice(0, 3).map(([category, categoryTools]) => (
              <div key={category} className="space-y-0.5">
                {categoryTools.slice(0, 3).map(tool => (
                  <button
                    key={tool.name}
                    onClick={() => onSelectTool(tool)}
                    className="w-full flex items-center gap-2 px-3 py-1 text-xs text-am-text-secondary hover:text-am-text-primary hover:bg-am-tertiary rounded-md transition-colors"
                  >
                    <Wrench className="w-3 h-3 flex-shrink-0 text-am-text-muted" />
                    <span className="truncate">{formatToolName(tool.name)}</span>
                  </button>
                ))}
              </div>
            ))}
            {tools.length > 9 && (
              <button className="w-full px-3 py-1 text-[10px] text-am-accent hover:text-am-accent-hover">
                View all {tools.length} tools...
              </button>
            )}
          </div>

          {/* Context/Data Sources */}
          <div className="mt-2 space-y-0.5">
            <div className="px-3 py-1 text-[10px] font-medium text-am-text-muted uppercase tracking-wider">
              Data Sources
            </div>
            <DataSourceItem icon={Database} label="PED Ensembles" count={5} />
            <DataSourceItem icon={FlaskConical} label="ChEMBL" count={22} />
            <DataSourceItem icon={BookOpen} label="Literature" count={6} />
          </div>
        </SidebarSection>

        {/* EVOLUTION Section - Opens full Evolution view */}
        <div className="py-1">
          <button
            onClick={() => onSelectKnowledge?.(null as any)}
            className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-am-text-secondary hover:text-am-text-primary hover:bg-am-tertiary rounded-md transition-colors min-h-[44px]"
            aria-label={`Open Evolution knowledge browser. ${artifacts.length} artifacts available`}
          >
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4" aria-hidden="true" />
              <span>Evolution</span>
              {artifacts.length > 0 && (
                <span className="px-1.5 py-0.5 text-[9px] bg-am-tertiary text-am-text-muted rounded-full" aria-hidden="true">
                  {artifacts.length}
                </span>
              )}
            </div>
            <ChevronRight className="w-3 h-3 text-am-text-muted" aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Footer */}
      <nav className="border-t border-am-border px-2 py-2 space-y-0.5" aria-label="Settings and help">
        <button 
          className="w-full flex items-center gap-2 px-3 py-2 text-xs text-am-text-muted hover:text-am-text-secondary hover:bg-am-tertiary rounded-md transition-colors min-h-[44px]"
          aria-label="Open settings"
        >
          <Settings className="w-4 h-4" aria-hidden="true" />
          <span>Settings</span>
        </button>
        <button 
          className="w-full flex items-center gap-2 px-3 py-2 text-xs text-am-text-muted hover:text-am-text-secondary hover:bg-am-tertiary rounded-md transition-colors min-h-[44px]"
          aria-label="Get help and provide feedback"
        >
          <HelpCircle className="w-4 h-4" aria-hidden="true" />
          <span>Help & Feedback</span>
        </button>
      </nav>
    </aside>
  );
}

// =============================================================================
// Sub-components
// =============================================================================

interface SidebarSectionProps {
  title: string;
  icon: React.ElementType;
  isExpanded: boolean;
  onToggle: () => void;
  badge?: number;
  children: React.ReactNode;
}

function SidebarSection({ title, icon: Icon, isExpanded, onToggle, badge, children }: SidebarSectionProps) {
  return (
    <div className="py-1">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-1.5 text-xs font-medium text-am-text-secondary hover:text-am-text-primary transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon className="w-3.5 h-3.5" />
          <span>{title}</span>
          {badge && (
            <span className="px-1.5 py-0.5 text-[9px] bg-am-tertiary text-am-text-muted rounded-full">
              {badge}
            </span>
          )}
        </div>
        {isExpanded ? (
          <ChevronDown className="w-3 h-3 text-am-text-muted" />
        ) : (
          <ChevronRight className="w-3 h-3 text-am-text-muted" />
        )}
      </button>
      {isExpanded && <div className="mt-1">{children}</div>}
    </div>
  );
}

function DataSourceItem({ icon: Icon, label, count }: { icon: React.ElementType; label: string; count: number }) {
  return (
    <div className="flex items-center justify-between px-3 py-1 text-xs text-am-text-secondary">
      <div className="flex items-center gap-2">
        <Icon className="w-3 h-3 text-am-text-muted" />
        <span>{label}</span>
      </div>
      <span className="text-[10px] text-am-text-muted">{count} tools</span>
    </div>
  );
}

function ArtifactIcon({ type }: { type: string }) {
  if (type.includes('task')) return <FileText className="w-3 h-3 flex-shrink-0 text-blue-400" />;
  if (type.includes('structure')) return <Atom className="w-3 h-3 flex-shrink-0 text-green-400" />;
  if (type.includes('dock') || type.includes('result')) return <BarChart3 className="w-3 h-3 flex-shrink-0 text-purple-400" />;
  if (type.includes('ligand')) return <FlaskConical className="w-3 h-3 flex-shrink-0 text-orange-400" />;
  if (type.includes('experiment')) return <Sparkles className="w-3 h-3 flex-shrink-0 text-yellow-400" />;
  return <FileText className="w-3 h-3 flex-shrink-0 text-am-text-muted" />;
}

// =============================================================================
// Utilities
// =============================================================================

function formatToolName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/^bc /, '')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function getArtifactTitle(artifact: Artifact): string {
  if (artifact.type === 'scientific_experiment') return (typeof artifact.content?.title === 'string' ? artifact.content.title : null) || 'Experiment';
  if (artifact.type === 'task_list') return 'Task List';
  if (artifact.type === 'structure_3d') return artifact.content?.metadata?.ped_id || 'Structure';
  if (artifact.type === 'docking_result') return 'Docking Results';
  if (artifact.type === 'ligand_svg') return (typeof artifact.content?.name === 'string' ? artifact.content.name : null) || 'Ligand';
  return artifact.id.slice(0, 8);
}
