/**
 * Evolution View - Scientific Binding Graph
 *
 * Steve Jobs philosophy: The graph should tell a story.
 * - Protein is the anchor (center, large)
 * - Best ligand GLOWS (the hero)
 * - Connections are curved and elegant
 * - Hover illuminates the binding path
 * - Minimal chrome, maximum clarity
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { X, Copy, Check } from 'lucide-react';

interface Entity {
  id: string;
  type: string;
  name: string;
  properties: Record<string, any>;
}

interface Relationship {
  subject_id: string;
  object_id: string;
  predicate: string;
  properties: Record<string, any>;
}

interface Node {
  id: string;
  entity: Entity;
  x: number;
  y: number;
  size: number;
  color: string;
  glow: boolean;
}

interface Edge {
  id: string;
  from: Node;
  to: Node;
  strength: number;
}

interface EvolutionViewProps {
  onClose: () => void;
}

export const EvolutionView: React.FC<EvolutionViewProps> = ({ onClose }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [entities, setEntities] = useState<Entity[]>([]);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [loading, setLoading] = useState(true);
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [copied, setCopied] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const animationRef = useRef<number>(0);
  const timeRef = useRef<number>(0);

  // Fetch data
  useEffect(() => {
    fetch('http://localhost:8002/evolution/graph')
      .then(r => r.json())
      .then(data => {
        // API returns {entities: [...], relationships: [...]} directly
        const rawEntities = data.entities || [];
        if (rawEntities.length > 0) {
          // Deduplicate
          const seen = new Map<string, Entity>();
          const normalize = (s: string) => s.toLowerCase().replace(/[\s\-_()]/g, '');
          for (const e of rawEntities) {
            const key = `${e.type}:${normalize(e.name)}`;
            if (!seen.has(key) || e.name.length < seen.get(key)!.name.length) {
              seen.set(key, e);
            }
          }
          setEntities(Array.from(seen.values()));
          setRelationships(data.relationships || []);
          console.log(`[Evolution] Loaded ${seen.size} entities, ${(data.relationships || []).length} relationships`);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // Handle resize
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // ESC to close
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (selectedNode) setSelectedNode(null);
        else onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, selectedNode]);

  // Build graph layout
  const { nodes, edges, idToNode } = useMemo(() => {
    if (entities.length === 0) return { nodes: [], edges: [], idToNode: new Map() };

    const { width, height } = dimensions;
    const cx = width / 2;
    const cy = height / 2;

    const proteins = entities.filter(e => e.type === 'Protein');
    const ligands = entities.filter(e => e.type === 'Ligand');
    const residues = entities.filter(e => e.type === 'Residue');

    // Sort ligands by energy (best first)
    const sortedLigands = [...ligands].sort((a, b) =>
      (a.properties?.best_energy ?? 0) - (b.properties?.best_energy ?? 0)
    );
    const bestLigand = sortedLigands[0];

    const nodes: Node[] = [];
    const idToNode = new Map<string, Node>();

    // Protein at center - the anchor
    proteins.forEach((p, i) => {
      const node: Node = {
        id: p.id,
        entity: p,
        x: cx + i * 40,
        y: cy,
        size: 50,
        color: '#818cf8', // Violet for protein
        glow: false,
      };
      nodes.push(node);
      idToNode.set(p.id, node);
    });

    // Ligands in a ring around protein
    const ligandRadius = Math.min(width, height) * 0.3;
    sortedLigands.forEach((l, i) => {
      const angle = (2 * Math.PI * i) / Math.max(sortedLigands.length, 1) - Math.PI / 2;
      const isBest = l.id === bestLigand?.id;
      const node: Node = {
        id: l.id,
        entity: l,
        x: cx + ligandRadius * Math.cos(angle),
        y: cy + ligandRadius * Math.sin(angle),
        size: isBest ? 35 : 24,
        color: isBest ? '#10b981' : '#34d399', // Emerald, brighter for best
        glow: isBest,
      };
      nodes.push(node);
      idToNode.set(l.id, node);
    });

    // Residues in outer ring
    const residueRadius = Math.min(width, height) * 0.42;
    residues.forEach((r, i) => {
      const angle = (2 * Math.PI * i) / Math.max(residues.length, 1) + Math.PI / 6;
      const node: Node = {
        id: r.id,
        entity: r,
        x: cx + residueRadius * Math.cos(angle),
        y: cy + residueRadius * Math.sin(angle),
        size: 14,
        color: '#fbbf24', // Amber for residues
        glow: false,
      };
      nodes.push(node);
      idToNode.set(r.id, node);
    });

    // Build edges from relationships
    const edges: Edge[] = [];
    const seenEdges = new Set<string>();

    relationships.forEach((rel, i) => {
      const fromNode = idToNode.get(rel.subject_id);
      const toNode = idToNode.get(rel.object_id);
      if (fromNode && toNode) {
        const key = [fromNode.id, toNode.id].sort().join('-');
        if (!seenEdges.has(key)) {
          seenEdges.add(key);
          edges.push({
            id: `e${i}`,
            from: fromNode,
            to: toNode,
            strength: rel.properties?.energy ? Math.abs(rel.properties.energy) / 10 : 0.5,
          });
        }
      }
    });

    // Add implicit edges: protein to all ligands
    proteins.forEach(protein => {
      const proteinNode = idToNode.get(protein.id);
      if (proteinNode) {
        sortedLigands.forEach((ligand, i) => {
          const ligandNode = idToNode.get(ligand.id);
          if (ligandNode) {
            const key = [proteinNode.id, ligandNode.id].sort().join('-');
            if (!seenEdges.has(key)) {
              seenEdges.add(key);
              edges.push({
                id: `ep${i}`,
                from: proteinNode,
                to: ligandNode,
                strength: ligand.id === bestLigand?.id ? 1 : 0.5,
              });
            }
          }
        });
      }
    });

    return { nodes, edges, idToNode };
  }, [entities, relationships, dimensions]);

  // Check if edge is connected to hovered/selected node
  const isEdgeHighlighted = (edge: Edge): boolean => {
    const target = hoveredNode || selectedNode;
    if (!target) return false;
    return edge.from.id === target.id || edge.to.id === target.id;
  };

  // Canvas rendering with animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = dimensions.width * dpr;
    canvas.height = dimensions.height * dpr;
    ctx.scale(dpr, dpr);

    const draw = (time: number) => {
      timeRef.current = time;
      ctx.clearRect(0, 0, dimensions.width, dimensions.height);

      // Draw edges first (behind nodes)
      edges.forEach(edge => {
        const highlighted = isEdgeHighlighted(edge);
        const alpha = highlighted ? 0.8 : 0.15;

        ctx.beginPath();

        // Curved bezier edge
        const dx = edge.to.x - edge.from.x;
        const dy = edge.to.y - edge.from.y;
        const cx1 = edge.from.x + dx * 0.3;
        const cy1 = edge.from.y + dy * 0.1;
        const cx2 = edge.from.x + dx * 0.7;
        const cy2 = edge.to.y - dy * 0.1;

        ctx.moveTo(edge.from.x, edge.from.y);
        ctx.bezierCurveTo(cx1, cy1, cx2, cy2, edge.to.x, edge.to.y);

        // Gradient for highlighted edges
        if (highlighted) {
          const gradient = ctx.createLinearGradient(
            edge.from.x, edge.from.y, edge.to.x, edge.to.y
          );
          gradient.addColorStop(0, edge.from.color);
          gradient.addColorStop(1, edge.to.color);
          ctx.strokeStyle = gradient;
          ctx.lineWidth = 2;
        } else {
          ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
          ctx.lineWidth = 1;
        }

        ctx.stroke();
      });

      // Draw nodes
      nodes.forEach(node => {
        const isHovered = hoveredNode?.id === node.id;
        const isSelected = selectedNode?.id === node.id;
        const isConnected = hoveredNode && edges.some(e =>
          (e.from.id === hoveredNode.id && e.to.id === node.id) ||
          (e.to.id === hoveredNode.id && e.from.id === node.id)
        );

        // Breathing animation for glow nodes
        const breathe = node.glow ? Math.sin(time / 1000) * 0.1 + 1 : 1;
        const size = node.size * breathe;

        // Glow effect
        if (node.glow || isHovered || isSelected) {
          const glowSize = size * 2;
          const gradient = ctx.createRadialGradient(
            node.x, node.y, size * 0.5,
            node.x, node.y, glowSize
          );
          gradient.addColorStop(0, node.color + '60');
          gradient.addColorStop(1, node.color + '00');
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(node.x, node.y, glowSize, 0, Math.PI * 2);
          ctx.fill();
        }

        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, size / 2, 0, Math.PI * 2);

        // Fill with gradient
        const nodeGradient = ctx.createRadialGradient(
          node.x - size * 0.2, node.y - size * 0.2, 0,
          node.x, node.y, size / 2
        );
        nodeGradient.addColorStop(0, node.color);
        nodeGradient.addColorStop(1, adjustColor(node.color, -30));
        ctx.fillStyle = nodeGradient;
        ctx.fill();

        // Border for hovered/selected
        if (isHovered || isSelected) {
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Dim unconnected nodes when hovering
        if (hoveredNode && !isHovered && !isConnected && hoveredNode.id !== node.id) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, size / 2, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(9, 9, 11, 0.6)';
          ctx.fill();
        }

        // Label
        const showLabel = isHovered || isSelected || node.entity.type === 'Protein' || node.glow;
        if (showLabel) {
          ctx.font = node.entity.type === 'Protein' ? 'bold 14px Inter, sans-serif' : '12px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';

          const label = node.entity.name;
          const labelY = node.y + size / 2 + 16;

          // Text shadow
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.fillText(label, node.x + 1, labelY + 1);

          // Text
          ctx.fillStyle = isHovered || isSelected ? '#fff' : 'rgba(255, 255, 255, 0.7)';
          ctx.fillText(label, node.x, labelY);

          // Energy label for ligands
          if (node.entity.type === 'Ligand' && node.entity.properties?.best_energy) {
            ctx.font = '10px Inter, sans-serif';
            const energyLabel = `${node.entity.properties.best_energy.toFixed(1)} kcal/mol`;
            ctx.fillStyle = node.glow ? '#10b981' : 'rgba(255, 255, 255, 0.5)';
            ctx.fillText(energyLabel, node.x, labelY + 14);
          }
        }
      });

      animationRef.current = requestAnimationFrame(draw);
    };

    animationRef.current = requestAnimationFrame(draw);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes, edges, hoveredNode, selectedNode, dimensions]);

  // Mouse interaction
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Find node under cursor
    const found = nodes.find(node => {
      const dx = node.x - x;
      const dy = node.y - y;
      return Math.sqrt(dx * dx + dy * dy) < node.size / 2 + 5;
    });

    setHoveredNode(found || null);
    canvas.style.cursor = found ? 'pointer' : 'default';
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (hoveredNode) {
      setSelectedNode(hoveredNode);
    } else {
      setSelectedNode(null);
    }
  };

  // Copy @reference
  const copyRef = () => {
    if (!selectedNode) return;
    const ref = `@${selectedNode.entity.type}:${selectedNode.entity.name}`;
    navigator.clipboard.writeText(ref);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  // Helper to darken/lighten color
  function adjustColor(hex: string, amount: number): string {
    const num = parseInt(hex.replace('#', ''), 16);
    const r = Math.min(255, Math.max(0, (num >> 16) + amount));
    const g = Math.min(255, Math.max(0, ((num >> 8) & 0x00FF) + amount));
    const b = Math.min(255, Math.max(0, (num & 0x0000FF) + amount));
    return `#${(1 << 24 | r << 16 | g << 8 | b).toString(16).slice(1)}`;
  }

  if (loading) {
    return (
      <div className="fixed inset-0 z-50 bg-[#09090b] flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-violet-500/30 border-t-violet-500 rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-[#09090b] flex flex-col">
      {/* Minimal header */}
      <header className="absolute top-0 left-0 right-0 h-12 px-6 flex items-center justify-between z-10">
        <div className="text-sm font-medium text-white/60">
          Binding Graph
        </div>
        <button
          onClick={onClose}
          className="p-2 text-white/40 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </header>

      {/* Canvas */}
      <div ref={containerRef} className="flex-1 relative">
        {entities.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-lg text-white/40 mb-2">No discoveries yet</div>
              <div className="text-sm text-white/20">Run an experiment to build the graph</div>
            </div>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            width={dimensions.width}
            height={dimensions.height}
            onMouseMove={handleMouseMove}
            onClick={handleClick}
            onMouseLeave={() => setHoveredNode(null)}
            className="absolute inset-0"
            style={{ width: dimensions.width, height: dimensions.height }}
          />
        )}
      </div>

      {/* Selected node panel */}
      {selectedNode && (
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-[#18181b]/90 backdrop-blur-xl border border-white/10 rounded-2xl p-4 min-w-[280px] shadow-2xl">
          <div className="flex items-center gap-3 mb-3">
            <div
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: selectedNode.color }}
            />
            <div>
              <div className="font-medium text-white">{selectedNode.entity.name}</div>
              <div className="text-xs text-white/40">{selectedNode.entity.type}</div>
            </div>
          </div>

          {selectedNode.entity.properties?.best_energy && (
            <div className="text-sm text-emerald-400 font-mono mb-3">
              {selectedNode.entity.properties.best_energy.toFixed(2)} kcal/mol
            </div>
          )}

          <button
            onClick={copyRef}
            className={`w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm transition-all ${
              copied
                ? 'bg-emerald-500/20 text-emerald-400'
                : 'bg-white/5 text-white/70 hover:bg-white/10 hover:text-white'
            }`}
          >
            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
            {copied ? 'Copied!' : `Copy @${selectedNode.entity.type}:${selectedNode.entity.name}`}
          </button>
        </div>
      )}

      {/* Hint */}
      {!selectedNode && entities.length > 0 && (
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 text-xs text-white/20">
          Click a node to copy @reference · ESC to close
        </div>
      )}
    </div>
  );
};
