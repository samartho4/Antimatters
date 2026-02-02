
// =============================================================================
// Molecule Viewer
// =============================================================================

interface MoleculeViewerProps {
  pdbData: string;
}

export function MoleculeViewer({ pdbData }: MoleculeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current || !pdbData) return;

    const init = async () => {
      try {
        // @ts-ignore
        const $3Dmol = await import('3dmol');

        if (!viewerRef.current) {
          viewerRef.current = $3Dmol.createViewer(containerRef.current, {
            backgroundColor: '#09090b',
          });
        }

        const viewer = viewerRef.current;
        viewer.removeAllModels();
        viewer.addModel(pdbData, 'pdb');

        viewer.setStyle({}, { cartoon: { color: 'spectrum', opacity: 0.9 } });

        // Highlight binding site
        [125, 133, 136].forEach(resn => {
          viewer.setStyle({ resi: resn }, {
            cartoon: { color: '#22d3ee' },
            stick: { color: '#22d3ee', radius: 0.15 }
          });
        });

        viewer.zoomTo();
        viewer.render();
        viewer.spin('y', 0.5);
      } catch (e) {
        console.error('3Dmol error:', e);
      }
    };

    init();
  }, [pdbData]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full"
      style={{ minHeight: '400px' }}
    />
  );
}
