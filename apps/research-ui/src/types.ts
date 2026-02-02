/**
 * CoRE Research UI Types
 * ======================
 * Type definitions for the AG-UI integrated frontend
 */

// Re-export AG-UI types
export {
  EventType,
  ArtifactType,
  type AGUIEvent,
  type Message as AGUIMessage,
  type ToolCall,
  type Tool,
  type Artifact,
  type AnnotationRegion,
  type RunAgentInput,
} from './services/aguiService';

// Legacy types for backward compatibility
export enum MessageRole {
  User = 'user',
  Assistant = 'assistant',
  System = 'system',
}

export enum AppMode {
  Ask = 'ask',
  Agent = 'agent',
  Live = 'live',
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  isThinking?: boolean;
  contextData?: any;
  relatedArtifactId?: string;
  contextImage?: string;
}

export interface LegacyArtifact {
  id: string;
  type: 'visualization' | 'code' | 'data' | 'reasoning';
  title: string;
  content: string;
  status: 'generating' | 'complete' | 'error';
}

// Session state from backend
export interface SessionState {
  pdb_path?: string;
  pdbqt_path?: string;
  representative_frames?: number[];
  best_energy?: number;
  n_conformations?: number;
  n_clusters?: number;
  binding_scores?: number[];
  has_ensemble?: boolean;
  has_ligand?: boolean;
  has_clusters?: boolean;
  has_docking?: boolean;
  has_interactions?: boolean;
}

// Ligand type
export interface Ligand {
  name: string;
  smiles: string;
  description?: string;
}

// Ensemble type
export interface Ensemble {
  id: string;
  name: string;
  conformations: number;
  description: string;
  binding_site?: number[];
}

// Protein visualization data
export interface ProteinNode {
  id: number;
  x: number;
  y: number;
  type: 'backbone' | 'ligand' | 'pocket';
  residue: string;
  charge?: number;
  hydrophobicity?: number;
  vx?: number;
  vy?: number;
  radius?: number;
}

export interface SimulationState {
  step: number;
  nodes: ProteinNode[];
  energy: number;
  logs: string[];
}
