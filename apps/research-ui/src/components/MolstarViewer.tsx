/**
 * Mol* 3D Protein Viewer Component
 * =================================
 * Real-time 3D visualization using Mol* (Molstar)
 * - Displays protein structures from PDB data
 * - Supports trajectory animation for IDP ensembles
 * - Annotation mode for region selection (AI Studio-style)
 * - Highlights binding sites and ligands
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';

// We'll use 3Dmol.js as a lighter alternative that works better in React
// Mol* requires more complex setup - 3Dmol provides similar features

declare global {
  interface Window {
    $3Dmol: any;
  }
}

interface MolstarViewerProps {
  pdbData?: string;
  pdbUrl?: string;
  ligandData?: string;
  highlightResidues?: number[];
  bindingSite?: number[];
  isAnnotationMode?: boolean;
  onAnnotationComplete?: (selectedResidues: number[], screenshot: string) => void;
  representationStyle?: 'cartoon' | 'stick' | 'sphere' | 'surface';
  colorScheme?: 'chainId' | 'residueIndex' | 'secondaryStructure' | 'bFactor';
  showControls?: boolean;
  trajectoryFrames?: number;
  currentFrame?: number;
  onFrameChange?: (frame: number) => void;
}

export const MolstarViewer: React.FC<MolstarViewerProps> = ({
  pdbData,
  pdbUrl,
  ligandData,
  highlightResidues = [],
  bindingSite = [],
  isAnnotationMode = false,
  onAnnotationComplete,
  representationStyle = 'cartoon',
  colorScheme = 'chainId',
  showControls = true,
  trajectoryFrames,
  currentFrame = 0,
  onFrameChange,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [selectedResidues, setSelectedResidues] = useState<number[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef<number | null>(null);

  // Load 3Dmol.js script
  useEffect(() => {
    if (window.$3Dmol) {
      setIsLoaded(true);
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://3dmol.org/build/3Dmol-min.js';
    script.async = true;
    script.onload = () => setIsLoaded(true);
    document.head.appendChild(script);

    return () => {
      if (script.parentNode) {
        script.parentNode.removeChild(script);
      }
    };
  }, []);

  // Initialize viewer
  useEffect(() => {
    if (!isLoaded || !containerRef.current || !window.$3Dmol) return;

    const config = {
      backgroundColor: 'black',
      antialias: true,
    };

    viewerRef.current = window.$3Dmol.createViewer(containerRef.current, config);

    return () => {
      if (viewerRef.current) {
        viewerRef.current.clear();
      }
    };
  }, [isLoaded]);

  // Load structure
  useEffect(() => {
    if (!viewerRef.current) return;

    const viewer = viewerRef.current;
    viewer.clear();

    const loadStructure = async () => {
      let data = pdbData;

      if (!data && pdbUrl) {
        try {
          const response = await fetch(pdbUrl);
          data = await response.text();
        } catch (e) {
          console.error('Failed to load PDB:', e);
          return;
        }
      }

      if (!data) return;

      // Add protein model
      viewer.addModel(data, 'pdb');

      // Apply representation
      const styleMap: Record<string, any> = {
        cartoon: { cartoon: { color: 'spectrum' } },
        stick: { stick: { colorscheme: 'chainHetatm' } },
        sphere: { sphere: { colorscheme: 'chainHetatm', radius: 0.5 } },
        surface: { surface: { opacity: 0.8, colorscheme: 'chainHetatm' } },
      };

      viewer.setStyle({}, styleMap[representationStyle] || styleMap.cartoon);

      // Highlight binding site residues
      if (bindingSite.length > 0) {
        viewer.setStyle(
          { resi: bindingSite },
          { stick: { color: 'red' }, cartoon: { color: 'red' } }
        );
      }

      // Highlight selected residues
      if (highlightResidues.length > 0) {
        viewer.setStyle(
          { resi: highlightResidues },
          { stick: { color: 'yellow' }, cartoon: { color: 'yellow' } }
        );
      }

      // Add ligand if provided
      if (ligandData) {
        viewer.addModel(ligandData, 'pdb');
        viewer.setStyle({ model: 1 }, { stick: { colorscheme: 'greenCarbon' } });
      }

      viewer.zoomTo();
      viewer.render();
    };

    loadStructure();
  }, [pdbData, pdbUrl, ligandData, representationStyle, bindingSite, highlightResidues]);

  // Handle annotation mode clicks
  useEffect(() => {
    if (!viewerRef.current || !isAnnotationMode) return;

    const viewer = viewerRef.current;

    const handleClick = (atom: any) => {
      if (!atom) return;

      const residue = atom.resi;
      setSelectedResidues(prev => {
        if (prev.includes(residue)) {
          return prev.filter(r => r !== residue);
        }
        return [...prev, residue];
      });

      // Highlight selected residue
      viewer.setStyle(
        { resi: residue },
        { stick: { color: 'orange' }, cartoon: { color: 'orange' } }
      );
      viewer.render();
    };

    viewer.setClickable({}, true, handleClick);

    return () => {
      viewer.setClickable({}, false, () => {});
    };
  }, [isAnnotationMode]);

  // Trajectory animation
  useEffect(() => {
    if (!isPlaying || !trajectoryFrames || !onFrameChange) return;

    let frame = currentFrame;
    animationRef.current = window.setInterval(() => {
      frame = (frame + 1) % trajectoryFrames;
      onFrameChange(frame);
    }, 100);

    return () => {
      if (animationRef.current) {
        clearInterval(animationRef.current);
      }
    };
  }, [isPlaying, trajectoryFrames, currentFrame, onFrameChange]);

  // Complete annotation
  const handleAnnotationComplete = useCallback(() => {
    if (!viewerRef.current || !onAnnotationComplete) return;

    // Get screenshot
    const canvas = containerRef.current?.querySelector('canvas');
    const screenshot = canvas?.toDataURL('image/png') || '';

    onAnnotationComplete(selectedResidues, screenshot);
    setSelectedResidues([]);
  }, [selectedResidues, onAnnotationComplete]);

  // Zoom controls
  const handleZoomIn = () => viewerRef.current?.zoom(1.2);
  const handleZoomOut = () => viewerRef.current?.zoom(0.8);
  const handleReset = () => {
    viewerRef.current?.zoomTo();
    viewerRef.current?.render();
  };

  return (
    <div className="relative w-full h-full bg-black">
      {/* 3D Viewer Container */}
      <div
        ref={containerRef}
        className="w-full h-full"
        style={{ position: 'relative' }}
      />

      {/* Loading State */}
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-slate-400 text-sm">Loading 3D Viewer...</span>
          </div>
        </div>
      )}

      {/* No Data State */}
      {isLoaded && !pdbData && !pdbUrl && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-slate-500">
            <svg className="w-16 h-16 mx-auto mb-4 opacity-50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 6v6l4 2" />
            </svg>
            <p className="text-sm">No structure loaded</p>
            <p className="text-xs mt-1">Fetch an ensemble to visualize</p>
          </div>
        </div>
      )}

      {/* Controls Overlay */}
      {showControls && isLoaded && (pdbData || pdbUrl) && (
        <div className="absolute bottom-4 left-4 flex items-center gap-2">
          {/* Zoom Controls */}
          <div className="flex items-center gap-1 bg-slate-800/90 rounded-lg p-1">
            <button
              onClick={handleZoomIn}
              className="p-2 hover:bg-slate-700 rounded text-slate-300 hover:text-white transition-colors"
              title="Zoom In"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
                <line x1="11" y1="8" x2="11" y2="14" />
                <line x1="8" y1="11" x2="14" y2="11" />
              </svg>
            </button>
            <button
              onClick={handleZoomOut}
              className="p-2 hover:bg-slate-700 rounded text-slate-300 hover:text-white transition-colors"
              title="Zoom Out"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
                <line x1="8" y1="11" x2="14" y2="11" />
              </svg>
            </button>
            <button
              onClick={handleReset}
              className="p-2 hover:bg-slate-700 rounded text-slate-300 hover:text-white transition-colors"
              title="Reset View"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
            </button>
          </div>

          {/* Trajectory Controls */}
          {trajectoryFrames && trajectoryFrames > 1 && (
            <div className="flex items-center gap-2 bg-slate-800/90 rounded-lg p-1 px-3">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-2 hover:bg-slate-700 rounded text-slate-300 hover:text-white transition-colors"
              >
                {isPlaying ? (
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="4" width="4" height="16" />
                    <rect x="14" y="4" width="4" height="16" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                )}
              </button>
              <span className="text-xs text-slate-400 font-mono">
                {currentFrame + 1}/{trajectoryFrames}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Annotation Mode Overlay */}
      {isAnnotationMode && (
        <div className="absolute top-4 left-4 right-4 flex items-center justify-between">
          <div className="bg-amber-500/90 text-black px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2">
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 20h9" />
              <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
            </svg>
            Click residues to select • {selectedResidues.length} selected
          </div>
          {selectedResidues.length > 0 && (
            <button
              onClick={handleAnnotationComplete}
              className="bg-green-500 hover:bg-green-600 text-white px-4 py-1.5 rounded-lg text-sm font-medium transition-colors"
            >
              Done
            </button>
          )}
        </div>
      )}

      {/* Binding Site Legend */}
      {bindingSite.length > 0 && (
        <div className="absolute top-4 right-4 bg-slate-800/90 rounded-lg p-3 text-xs">
          <div className="text-slate-400 mb-2 font-medium">Binding Site</div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded bg-red-500" />
            <span className="text-slate-300">Y{bindingSite.join(', Y')}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MolstarViewer;
