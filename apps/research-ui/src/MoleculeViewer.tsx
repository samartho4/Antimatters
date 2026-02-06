import { useRef, useEffect, useState, useCallback } from 'react';
import { Loader2 } from 'lucide-react';

// =============================================================================
// Molecule Viewer — interactive 3Dmol.js viewer as a React component.
//
// 3Dmol.js is loaded as a UMD global via <script> in index.html (CDN).
// Access: window.$3Dmol   — avoids bundler CJS/ESM interop entirely.
//
// Annotation export: after the viewer renders, we store a reference on the
// container DOM element (containerRef.current.__viewer3d) so that
// ArtifactPanel can call exportImage() without needing React refs/context.
//
// Props:
//   pdbData          – PDB string for the primary model (protein or ligand)
//   ligandPdb        – optional second PDB model rendered in cyan
//   interactions     – residue indices per interaction type (for color coding)
//   highlightResidues – binding-site residues rendered in cyan (default: α-syn Y125/133/136)
// =============================================================================

declare global {
  interface HTMLDivElement { __viewer3d?: any; }
}

interface InteractionData {
  hbond_residues?: number[];
  hydrophobic_residues?: number[];
  aromatic_residues?: number[];
}

interface MoleculeViewerProps {
  pdbData: string;
  ligandPdb?: string;
  interactions?: InteractionData;
  highlightResidues?: number[];
}

export function MoleculeViewer({
  pdbData,
  ligandPdb,
  interactions,
  highlightResidues = [125, 133, 136],
}: MoleculeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef   = useRef<any>(null);
  const [error, setError]       = useState<string | null>(null);
  const [loading, setLoading]   = useState(true);   // true until first successful render

  // ── wait for the CDN global + a container with real layout dimensions
  //     before we try createViewer.  This avoids the classic "0×0 div" bug.
  const waitReady = useCallback((): Promise<typeof window.$3Dmol> => {
    return new Promise((resolve, reject) => {
      let attempts = 0;
      const check = () => {
        attempts++;
        const el = containerRef.current;
        if (window.$3Dmol && el && el.offsetWidth > 0 && el.offsetHeight > 0) {
          resolve(window.$3Dmol);
          return;
        }
        if (attempts > 40) {   // 40 × 100 ms = 4 s timeout
          reject(new Error(
            window.$3Dmol
              ? 'Viewer container has zero dimensions — layout issue'
              : '3Dmol.js failed to load from CDN'
          ));
          return;
        }
        requestAnimationFrame(() => setTimeout(check, 100));
      };
      check();
    });
  }, []);

  useEffect(() => {
    if (!pdbData) {
      setLoading(false);   // nothing to show — not an error, just empty
      return;
    }
    setError(null);
    setLoading(true);

    let cancelled = false;

    (async () => {
      try {
        const $3Dmol = await waitReady();
        if (cancelled) return;

        // Create viewer once; reuse on subsequent prop changes
        if (!viewerRef.current) {
          viewerRef.current = $3Dmol.createViewer(containerRef.current!, {
            backgroundColor: '#09090b',
          });
          // Stop auto-spin on user interaction
          containerRef.current!.addEventListener('mousedown', () => {
            viewerRef.current?.spin(false);
          }, { passive: true });
        }

        const viewer = viewerRef.current;
        viewer.removeAllModels();

        viewer.addModel(pdbData, 'pdb');

        // Detect data type: cartoon is for protein backbone (ATOM records).
        // Small molecules are HETATM-only — cartoon renders nothing for them.
        const isSmallMol = !pdbData.includes('\nATOM  ');

        if (isSmallMol) {
          // Small molecule: ball + stick
          viewer.setStyle({}, { stick: { colorscheme: 'Jmol', radius: 0.15 }, sphere: { scale: 0.3 } });
        } else {
          // Protein (± ligand): cartoon backbone, stick for any HETATM
          viewer.setStyle({}, { cartoon: { color: 'spectrum', opacity: 0.9 } });
          viewer.setStyle({ hetAtom: true }, { stick: { color: '#00bcd4', radius: 0.15 } });

          // Binding-site highlight (cyan cartoon + stick)
          highlightResidues.forEach(resn => {
            viewer.setStyle({ resi: resn }, {
              cartoon: { color: '#22d3ee' },
              stick:   { color: '#22d3ee', radius: 0.15 },
            });
          });

          // Interaction coloring
          if (interactions) {
            interactions.hbond_residues?.forEach(resn => {
              viewer.addStyle({ resi: resn }, { stick: { color: '#2196f3', radius: 0.2 } });
            });
            interactions.hydrophobic_residues?.forEach(resn => {
              viewer.addStyle({ resi: resn }, { stick: { color: '#4caf50', radius: 0.2 } });
            });
            interactions.aromatic_residues?.forEach(resn => {
              viewer.addStyle({ resi: resn }, { stick: { color: '#9c27b0', radius: 0.2 } });
            });
          }
        }

        // Second model (ligand overlay on protein)
        if (ligandPdb) {
          viewer.addModel(ligandPdb, 'pdb');
          viewer.setStyle({ model: 1 }, { stick: { color: '#00bcd4', radius: 0.15 } });
        }

        viewer.zoomTo();
        viewer.render();
        viewer.spin('y', 0.5);

        // ── expose viewer on the DOM element for annotation export ──
        if (containerRef.current) {
          containerRef.current.__viewer3d = viewer;
        }

        if (!cancelled) setLoading(false);
      } catch (e: any) {
        if (cancelled) return;
        console.error('3Dmol error:', e);
        setError(e?.message || String(e));
        setLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [pdbData, ligandPdb, interactions, highlightResidues, waitReady]);

  // ── cleanup: detach stored viewer on unmount ──
  useEffect(() => {
    return () => {
      if (containerRef.current) {
        containerRef.current.__viewer3d = undefined;
      }
    };
  }, []);

  return (
    <div className="w-full h-full relative">
      {/* the div 3Dmol.js owns */}
      <div ref={containerRef} className="w-full h-full" />

      {/* Spinner — visible while CDN/layout settle */}
      {loading && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/40 z-10">
          <Loader2 className="w-8 h-8 text-am-accent animate-spin" />
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-10">
          <div className="bg-slate-800 border border-red-500/40 rounded-lg px-4 py-3 max-w-xs text-center">
            <div className="text-red-400 text-xs font-semibold mb-1">3D Viewer Error</div>
            <div className="text-slate-400 text-xs">{error}</div>
          </div>
        </div>
      )}
    </div>
  );
}
