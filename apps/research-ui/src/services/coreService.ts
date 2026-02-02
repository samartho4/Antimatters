/**
 * CoRE API Service - Connects to the real ADK agent backend
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ChatParams {
  message: string;
  sessionId?: string;
  mode?: 'ask' | 'agent' | 'live';
}

export interface DockingParams {
  pedId: string;
  ligandName: string;
  nClusters: number;
  sessionId?: string;
}

export interface Ligand {
  name: string;
  smiles: string;
}

export interface SessionState {
  pdb_path?: string;
  pdbqt_path?: string;
  representative_frames?: number[];
  best_energy?: number;
  n_conformations?: number;
}

/**
 * Check API health
 */
export const checkHealth = async (): Promise<{ status: string; agents: string[]; known_ligands: string[] }> => {
  const response = await fetch(`${API_BASE}/`);
  if (!response.ok) throw new Error('API not available');
  return response.json();
};

/**
 * Get known ligands
 */
export const getLigands = async (): Promise<Ligand[]> => {
  const response = await fetch(`${API_BASE}/ligands`);
  if (!response.ok) throw new Error('Failed to fetch ligands');
  const data = await response.json();
  return data.ligands;
};

/**
 * Send chat message to agent system
 */
export const sendMessage = async (params: ChatParams): Promise<{ response: string; artifacts?: SessionState }> => {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: params.message,
      session_id: params.sessionId || 'default',
      mode: params.mode || 'agent'
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Chat request failed');
  }

  return response.json();
};

/**
 * Stream chat response
 */
export async function* streamMessage(params: ChatParams): AsyncGenerator<{ type: string; text?: string; message?: string }> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: params.message,
      session_id: params.sessionId || 'default',
      mode: params.mode || 'agent'
    })
  });

  if (!response.ok) {
    throw new Error('Stream request failed');
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No reader available');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          yield data;
        } catch (e) {
          // Skip invalid JSON
        }
      }
    }
  }
}

/**
 * Run complete docking workflow
 */
export const runDocking = async (params: DockingParams): Promise<{ success: boolean; response: string; results: SessionState }> => {
  const response = await fetch(`${API_BASE}/dock`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ped_id: params.pedId,
      ligand_name: params.ligandName,
      n_clusters: params.nClusters,
      session_id: params.sessionId || 'default'
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Docking request failed');
  }

  return response.json();
};

/**
 * Get session state
 */
export const getSessionState = async (sessionId: string = 'default'): Promise<SessionState> => {
  const response = await fetch(`${API_BASE}/session/${sessionId}/state`);
  if (!response.ok) throw new Error('Failed to get session state');
  const data = await response.json();
  return data.state;
};
